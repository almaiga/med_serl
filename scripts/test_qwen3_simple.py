#!/usr/bin/env python3
"""
Test WITHOUT thinking mode to see if that works better for injector.
Thinking mode is great for assessor (short outputs) but may not be suitable
for injector (needs to output full clinical notes).
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sft'))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from inference_utils import (
    build_messages, load_assessor_prompts, load_injector_prompts,
    apply_vcf, extract_generated_note, extract_final_answer
)

# Paths
PROMPT_DIR = "/Users/josmaiga/Documents/GitHub/med_serl/configs/prompts"

# Load prompts
assessor_prompts = load_assessor_prompts(f"{PROMPT_DIR}/error_detection_prompts.json")
injector_prompts = load_injector_prompts(f"{PROMPT_DIR}/error_injection_prompts_v2.json")

print("=" * 60)
print("Qwen3-4B WITHOUT Thinking Mode (Standard Generation)")
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
print("TEST: Injector + Assessor WITHOUT Thinking Mode")
print("=" * 60)

# Build injector messages (NO thinking budget)
error_type = "medication error"
messages_injector = build_messages(
    mode="injector",
    note=original_note,
    prompt_intent=error_type,
    assessor_prompts=None,
    injector_prompts=injector_prompts,
    thinking_budget=0,  # NO thinking mode
    injector_is_correct=False,
)

print("\n[3/6] INJECTOR generating (standard mode)...")
print(f"Error to inject: {error_type}")

# Apply chat template WITHOUT thinking
prompt_injector = tokenizer.apply_chat_template(
    messages_injector,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,  # Disable thinking mode
)

# Standard generation
inputs = tokenizer([prompt_injector], return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=1024,
        temperature=0.7,
        top_p=0.9,
        top_k=20,
        min_p=0.05,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.1,
    )

generated_full = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)

print(f"\nGenerated output length: {len(generated_full)} chars")

# Extract generated note
generated_note = extract_generated_note(generated_full)

if generated_note:
    print(f"\n✅ Successfully extracted generated note!")
    print(f"Original (first 80 chars): {original_note[:80]}...")
    print(f"Generated (first 80 chars): {generated_note[:80]}...")
else:
    print(f"\n❌ Failed to extract generated note!")
    print(f"Full output:\n{generated_full[:500]}...")

# Apply VCF
if generated_note:
    print("\n[4/6] Applying VCF filters...")
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

    # Build assessor messages
    print("\n[5/6] ASSESSOR evaluating...")

    messages_assessor = build_messages(
        mode="assessor",
        note=generated_note,
        prompt_intent="",
        assessor_prompts=assessor_prompts,
        injector_prompts=None,
        thinking_budget=0,  # Also no thinking for assessor in this test
    )

    # Apply chat template
    prompt_assessor = tokenizer.apply_chat_template(
        messages_assessor,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    # Generate
    inputs_assessor = tokenizer([prompt_assessor], return_tensors="pt").to(device)

    with torch.no_grad():
        outputs_assessor = model.generate(
            **inputs_assessor,
            max_new_tokens=256,
            temperature=0.3,
            top_p=0.95,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    assessment_full = tokenizer.decode(outputs_assessor[0][inputs_assessor['input_ids'].shape[1]:], skip_special_tokens=True)

    # Extract prediction
    prediction = extract_final_answer(assessment_full)

    print(f"Assessor prediction: {prediction}")

    # Compute reward
    ground_truth = "INCORRECT"  # We injected an error
    assessor_correct = (prediction == ground_truth)
    reward = 1.0 if assessor_correct else -1.0

    print(f"\nGround truth: {ground_truth}")
    print(f"Assessor correct: {assessor_correct}")
    print(f"Reward: {reward:.1f} ({'✅ Assessor wins' if reward > 0 else '❌ Injector wins'})")

print("\n[6/6] Complete!")

if torch.backends.mps.is_available():
    mem_used = torch.mps.current_allocated_memory() / 1e9
    print(f"\nMemory: {mem_used:.2f} GB / 48 GB")

print("\n" + "=" * 60)
print("✅ Test WITHOUT Thinking Mode Complete!")
print("=" * 60)
