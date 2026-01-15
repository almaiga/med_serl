#!/usr/bin/env python3
"""
Test Qwen3-4B with proper thinking mode (COT) as used in quick_infer.
Uses your production prompts and generation functions.
"""

import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sft'))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from inference_utils import (
    build_messages, load_assessor_prompts, load_injector_prompts,
    apply_vcf, extract_generated_note, extract_final_answer
)

# Import generation function from quick_infer
from quick_infer_qwen3_4b_lora import generate_qwen_with_thinking

# Paths
PROMPT_DIR = "/Users/josmaiga/Documents/GitHub/med_serl/configs/prompts"

# Load prompts
assessor_prompts = load_assessor_prompts(f"{PROMPT_DIR}/error_detection_prompts.json")
injector_prompts = load_injector_prompts(f"{PROMPT_DIR}/error_injection_prompts_v2.json")

print("=" * 60)
print("Qwen3-4B with Proper Thinking Mode (COT)")
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
print("TEST: Injector + Assessor with Thinking Mode")
print("=" * 60)

# Build injector messages
error_type = "medication error"
messages_injector = build_messages(
    mode="injector",
    note=original_note,
    prompt_intent=error_type,
    assessor_prompts=None,
    injector_prompts=injector_prompts,
    thinking_budget=512,
    injector_is_correct=False,  # We want to inject an error
)

print("\n[3/6] INJECTOR generating with thinking mode...")
print(f"Error to inject: {error_type}")

# Apply chat template with thinking enabled
prompt_injector = tokenizer.apply_chat_template(
    messages_injector,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # Enable COT
)

# Generate with thinking
# IMPORTANT: For injector, we need MORE answer tokens since it outputs a full note
# Reduce thinking budget to leave room for the answer
generated_full = generate_qwen_with_thinking(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt_injector,
    thinking_budget=256,  # Reduced - leave more room for answer
    max_new_tokens=1536,  # Increased total budget
    temperature=0.7,
    top_p=0.9,
    min_p=0.05,
)

print(f"\nGenerated output length: {len(generated_full)} chars")

# Check if thinking is present
has_thinking = "<think>" in generated_full
print(f"Has thinking: {'✅ Yes' if has_thinking else '❌ No'}")

# Extract generated note
generated_note = extract_generated_note(generated_full)

print(f"\nOriginal (first 80 chars): {original_note[:80]}...")
print(f"Generated (first 80 chars): {generated_note[:80]}...")

# Show thinking if present
if has_thinking:
    thinking_section = generated_full.split("<think>")[1].split("</think>")[0]
    print(f"\nThinking (first 100 chars): {thinking_section[:100]}...")

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
if result.score_jaccard:
    print(f"  Jaccard: {result.score_jaccard:.3f}")
print(f"  Word edits: {result.word_edits}")
print(f"  Sentences changed: {result.sentences_changed}")
print(f"  Reason: {result.reason or 'Passed all filters'}")

# Build assessor messages
print("\n[5/6] ASSESSOR evaluating with thinking mode...")

messages_assessor = build_messages(
    mode="assessor",
    note=generated_note,
    prompt_intent="",
    assessor_prompts=assessor_prompts,
    injector_prompts=None,
    thinking_budget=256,
)

# Apply chat template with thinking
prompt_assessor = tokenizer.apply_chat_template(
    messages_assessor,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,  # Enable COT
)

# Generate with thinking
assessment_full = generate_qwen_with_thinking(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt_assessor,
    thinking_budget=256,  # Shorter budget for assessor
    max_new_tokens=512,
    temperature=0.3,
    top_p=0.95,
    min_p=0.05,
)

# Extract prediction
prediction = extract_final_answer(assessment_full)

print(f"Assessor prediction: {prediction}")

# Show thinking
if "<think>" in assessment_full:
    assessor_thinking = assessment_full.split("<think>")[1].split("</think>")[0]
    print(f"Assessor thinking (first 100 chars): {assessor_thinking[:100]}...")

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
print("✅ Test with Thinking Mode Complete!")
print("=" * 60)

print("\nResults Summary:")
print(f"  ✅ Production prompts loaded")
print(f"  ✅ Thinking mode (COT): {has_thinking}")
print(f"  ✅ Injector: {'Modified note' if generated_note != original_note else 'Needs review'}")
print(f"  ✅ VCF: {'PASS' if result.passed else f'REJECT ({result.reason})'}")
print(f"  ✅ Assessor: {prediction} (expected: {ground_truth})")
print(f"  ✅ Reward: {reward:.1f}")

print("\n" + "=" * 60)
print("✅ Ready for MedSeRL training with:")
print("  - Qwen3-4B on M3 Max (8GB)")
print("  - Production prompts from med_serl")
print("  - Thinking mode enabled")
print("  - VCF filtering")
print("=" * 60)
