#!/usr/bin/env python3
"""
Test MedSeRL with real production prompts from med_serl repo.
Uses Qwen3-4B on M3 Max with actual injector/assessor prompts.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sft'))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from inference_utils import apply_vcf

# Load prompts
PROMPT_DIR = "/Users/josmaiga/Documents/GitHub/med_serl/configs/prompts"

with open(f"{PROMPT_DIR}/error_injection_prompts_v2.json") as f:
    injector_prompts = json.load(f)

with open(f"{PROMPT_DIR}/error_detection_prompts.json") as f:
    assessor_prompts = json.load(f)

print("=" * 60)
print("MedSeRL Test with Production Prompts")
print("=" * 60)

# Load model
model_name = "Qwen/Qwen3-4B"
print(f"\n[1/6] Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

print("[2/6] Model loaded!")
device = next(model.parameters()).device
print(f"Device: {device}")

if torch.backends.mps.is_available():
    mem_used = torch.mps.current_allocated_memory() / 1e9
    print(f"Memory: {mem_used:.2f} GB / 48 GB")

# Test note
original_note = """Patient presents with 3-week history of progressive dyspnea on exertion.
Physical exam reveals bilateral lower extremity edema and elevated jugular venous pressure.
Chest X-ray shows cardiomegaly and pulmonary congestion.
BNP level is markedly elevated at 850 pg/mL.
Started on furosemide 40mg daily and lisinopril 10mg daily for acute decompensated heart failure."""

print("\n" + "=" * 60)
print("TEST 1: INJECTOR - Introduce Medical Error")
print("=" * 60)

# Build injector prompt (incorrect)
error_type = "medication error (wrong dosage or drug)"
injector_user_prompt = injector_prompts["injector_incorrect_template"].format(
    note=original_note,
    prompt_intent=error_type
)

messages_injector = [
    {"role": "system", "content": injector_prompts["system_prompt_incorrect"]},
    {"role": "user", "content": injector_user_prompt}
]

print("\n[3/6] Generating with INJECTOR (production prompt)...")
print(f"Error type to inject: {error_type}")

text = tokenizer.apply_chat_template(
    messages_injector,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer([text], return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=512,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

generated_full = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

# Extract generated_note from output
if "generated_note:" in generated_full:
    generated_note = generated_full.split("generated_note:")[1].split("final_answer:")[0].strip()
else:
    generated_note = generated_full.split('\n\n')[0].strip()

print(f"\nOriginal (first 100 chars): {original_note[:100]}...")
print(f"\nGenerated (first 100 chars): {generated_note[:100]}...")

# Show the change if extracted
if "changes_made:" in generated_full:
    changes = generated_full.split("changes_made:")[1].split("}")[0] + "}"
    print(f"\nChanges made: {changes[:150]}...")

# Apply VCF
print("\n[4/6] Applying VCF filters...")
result = apply_vcf(
    original_note,
    generated_note,
    min_jaccard=0.85,
    max_jaccard=0.99,
    max_word_edits=6,
)

print(f"VCF Result: {'✅ PASS' if result.passed else '❌ REJECT'}")
print(f"  Jaccard: {result.score_jaccard if result.score_jaccard else 'N/A'}")
print(f"  Word edits: {result.word_edits}")
print(f"  Reason: {result.reason or 'Passed all filters'}")

print("\n" + "=" * 60)
print("TEST 2: ASSESSOR - Detect Error")
print("=" * 60)

# Build assessor prompt
assessor_user_prompt = assessor_prompts["user_template"].format(note=generated_note)

messages_assessor = [
    {"role": "system", "content": assessor_prompts["system_prompt"]},
    {"role": "user", "content": assessor_user_prompt}
]

print("\n[5/6] Running ASSESSOR (production prompt)...")

text_assessor = tokenizer.apply_chat_template(
    messages_assessor,
    tokenize=False,
    add_generation_prompt=True
)
inputs_assessor = tokenizer([text_assessor], return_tensors="pt").to(device)

with torch.no_grad():
    outputs_assessor = model.generate(
        **inputs_assessor,
        max_new_tokens=128,
        temperature=0.3,
    )

assessment_full = tokenizer.decode(outputs_assessor[0][inputs_assessor['input_ids'].shape[1]:], skip_special_tokens=True)

# Extract prediction
if "INCORRECT" in assessment_full.upper():
    prediction = "INCORRECT"
elif "CORRECT" in assessment_full.upper():
    prediction = "CORRECT"
else:
    prediction = "UNKNOWN"

print(f"Assessor prediction: {prediction}")

# Extract explanation if present
if "Explanation:" in assessment_full:
    explanation = assessment_full.split("Explanation:")[1].strip().split('\n')[0]
    print(f"Explanation: {explanation[:150]}...")

# Compute reward (zero-sum)
ground_truth = "INCORRECT"  # We injected an error
assessor_correct = (prediction == ground_truth)
reward = 1.0 if assessor_correct else -1.0

print(f"\nGround truth: {ground_truth}")
print(f"Assessor correct: {assessor_correct}")
print(f"Reward: {reward:.1f} ({'Assessor wins' if reward > 0 else 'Injector wins'})")

print("\n[6/6] Complete!")

if torch.backends.mps.is_available():
    mem_used = torch.mps.current_allocated_memory() / 1e9
    print(f"\nMemory: {mem_used:.2f} GB / 48 GB")

print("\n" + "=" * 60)
print("✅ Production Prompt Test Complete!")
print("=" * 60)

print("\nSummary:")
print(f"  ✅ Loaded production prompts from med_serl repo")
print(f"  ✅ Injector: {'Generated modified note' if generated_note != original_note else 'Needs tuning'}")
print(f"  ✅ VCF: {result.score_jaccard if result.score_jaccard else 'N/A'}")
print(f"  ✅ Assessor: {prediction}")
print(f"  ✅ Zero-sum reward: {reward:.1f}")

print("\nNext steps:")
print("  1. If VCF passed, ready for mini training!")
print("  2. If VCF failed, may need to adjust thresholds")
print("  3. Run: python scripts/train_mini_with_prompts.py")
