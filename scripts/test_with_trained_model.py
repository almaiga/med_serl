#!/usr/bin/env python3
"""
Test MedSeRL with your trained LoRA adapter.
Uses production prompts + fine-tuned Qwen3-4B.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sft'))

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
from inference_utils import apply_vcf

# Paths
PROMPT_DIR = "/Users/josmaiga/Documents/GitHub/med_serl/configs/prompts"
ADAPTER_PATH = "/Users/josmaiga/Documents/GitHub/med_serl/outputs/local_training/sft/sft_checkpoint"

# Load prompts
with open(f"{PROMPT_DIR}/error_injection_prompts_v2.json") as f:
    injector_prompts = json.load(f)

with open(f"{PROMPT_DIR}/error_detection_prompts.json") as f:
    assessor_prompts = json.load(f)

print("=" * 60)
print("MedSeRL Test with YOUR Trained Model!")
print("=" * 60)

# Load base model
model_name = "Qwen/Qwen3-4B"
print(f"\n[1/7] Loading base model: {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

# Load LoRA adapter
print(f"[2/7] Loading your LoRA adapter...")
print(f"Adapter path: {ADAPTER_PATH}")

model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model = model.merge_and_unload()  # Merge for faster inference

print("[3/7] Model loaded with your fine-tuning!")
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
print("TEST: Full MedSeRL Pipeline with Fine-Tuned Model")
print("=" * 60)

# Build injector prompt
error_type = "medication error"
injector_user_prompt = injector_prompts["injector_incorrect_template"].format(
    note=original_note,
    prompt_intent=error_type
)

messages_injector = [
    {"role": "system", "content": injector_prompts["system_prompt_incorrect"]},
    {"role": "user", "content": injector_user_prompt}
]

print("\n[4/7] INJECTOR generating...")
print(f"Error to inject: {error_type}")

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

# Extract generated_note
if "generated_note:" in generated_full:
    generated_note = generated_full.split("generated_note:")[1].split("final_answer:")[0].strip()
else:
    # Fallback: take first paragraph
    generated_note = generated_full.split('\n\n')[0].strip()

print(f"\nInjector output length: {len(generated_full)} chars")
print(f"Generated note length: {len(generated_note)} chars")

# Show changes
if "changes_made:" in generated_full:
    try:
        changes_str = generated_full.split("changes_made:")[1].split('\n')[0]
        print(f"Changes: {changes_str[:100]}...")
    except:
        pass

# Apply VCF
print("\n[5/7] Applying VCF filters...")
result = apply_vcf(
    original_note,
    generated_note,
    min_jaccard=0.85,
    max_jaccard=0.99,
    max_word_edits=6,
)

print(f"VCF Result: {'✅ PASS' if result.passed else '❌ REJECT'}")
if result.score_jaccard:
    print(f"  Jaccard: {result.score_jaccard:.3f}")
print(f"  Word edits: {result.word_edits}")
print(f"  Sentences changed: {result.sentences_changed}")
print(f"  Reason: {result.reason or 'Passed all filters'}")

# Assessor
print("\n[6/7] ASSESSOR evaluating...")

assessor_user_prompt = assessor_prompts["user_template"].format(note=generated_note)
messages_assessor = [
    {"role": "system", "content": assessor_prompts["system_prompt"]},
    {"role": "user", "content": assessor_user_prompt}
]

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

assessment = tokenizer.decode(outputs_assessor[0][inputs_assessor['input_ids'].shape[1]:], skip_special_tokens=True)

# Parse prediction
if "INCORRECT" in assessment.upper():
    prediction = "INCORRECT"
elif "CORRECT" in assessment.upper():
    prediction = "CORRECT"
else:
    prediction = "UNKNOWN"

print(f"Assessor prediction: {prediction}")

# Extract explanation
if "Explanation:" in assessment:
    explanation = assessment.split("Explanation:")[1].strip().split('\n')[0]
    print(f"Explanation: {explanation[:100]}...")

# Compute reward
ground_truth = "INCORRECT"
assessor_correct = (prediction == ground_truth)
reward = 1.0 if assessor_correct else -1.0

print(f"\nGround truth: {ground_truth}")
print(f"Assessor correct: {assessor_correct}")
print(f"Reward: {reward:.1f} ({'✅ Assessor wins' if reward > 0 else '❌ Injector wins'})")

print("\n[7/7] Complete!")

if torch.backends.mps.is_available():
    mem_used = torch.mps.current_allocated_memory() / 1e9
    print(f"\nMemory: {mem_used:.2f} GB / 48 GB")

print("\n" + "=" * 60)
print("✅ Test with YOUR Fine-Tuned Model Complete!")
print("=" * 60)

print("\nResults Summary:")
print(f"  Model: Qwen3-4B + YOUR LoRA adapter ✅")
print(f"  Prompts: Production prompts from med_serl ✅")
print(f"  Injector: {generated_note != original_note}")
print(f"  VCF: {'PASS' if result.passed else f'REJECT ({result.reason})'}")
print(f"  Assessor: {prediction} (expected: INCORRECT)")
print(f"  Reward: {reward:.1f}")

print("\nReady for mini training!")
print("Next: Create mini training script with your model + prompts")
