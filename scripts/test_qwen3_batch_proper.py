#!/usr/bin/env python3
"""
Test Qwen3-4B with proper batch thinking mode (with correct answer_tokens).
Uses the batch generation function with answer_tokens=1024 for injector.
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

# Import BATCH generation function from quick_infer
from quick_infer_qwen3_4b_lora import generate_qwen_with_thinking_batch

# Paths
PROMPT_DIR = "/Users/josmaiga/Documents/GitHub/med_serl/configs/prompts"

# Load prompts
assessor_prompts = load_assessor_prompts(f"{PROMPT_DIR}/error_detection_prompts.json")
injector_prompts = load_injector_prompts(f"{PROMPT_DIR}/error_injection_prompts_v2.json")

print("=" * 60)
print("Qwen3-4B with BATCH Thinking Mode (Correct Answer Tokens)")
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
print("TEST: Injector + Assessor with BATCH Thinking Mode")
print("=" * 60)

# Build injector messages
error_type = "medication error"
messages_injector = build_messages(
    mode="injector",
    note=original_note,
    prompt_intent=error_type,
    assessor_prompts=None,
    injector_prompts=injector_prompts,
    thinking_budget=512,  # Thinking budget
    injector_is_correct=False,
)

print("\n[3/6] INJECTOR generating with BATCH thinking mode...")
print(f"Error to inject: {error_type}")

# Apply chat template with thinking enabled
prompt_injector = tokenizer.apply_chat_template(
    messages_injector,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)

# Generate with BATCH function (even for single example)
# This allows us to set answer_tokens=1024
generated_outputs = generate_qwen_with_thinking_batch(
    model=model,
    tokenizer=tokenizer,
    prompts=[prompt_injector],  # Pass as list
    thinking_budget=512,  # Budget for <think> section
    max_new_tokens=1536,  # Total budget
    temperature=0.7,
    top_p=0.9,
    min_p=0.05,
    answer_tokens=1024,  # KEY: Allow 1024 tokens for answer (not 128!)
)

generated_full = generated_outputs[0]  # Extract from batch

print(f"\nGenerated output length: {len(generated_full)} chars")

# Check if thinking is present
has_thinking = "<think>" in generated_full
print(f"Has thinking: {'✅ Yes' if has_thinking else '❌ No'}")

# Extract generated note
generated_note = extract_generated_note(generated_full)

if generated_note:
    print(f"\n✅ Successfully extracted generated note!")
    print(f"Original (first 80 chars): {original_note[:80]}...")
    print(f"Generated (first 80 chars): {generated_note[:80]}...")
else:
    print(f"\n❌ Failed to extract generated note!")
    print(f"Full output (first 500 chars):\n{generated_full[:500]}...")

# Show thinking if present
if has_thinking and "</think>" in generated_full:
    thinking_section = generated_full.split("<think>")[1].split("</think>")[0]
    print(f"\nThinking (first 100 chars): {thinking_section[:100]}...")

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
    print("\n[5/6] ASSESSOR evaluating with BATCH thinking mode...")

    messages_assessor = build_messages(
        mode="assessor",
        note=generated_note,
        prompt_intent="",
        assessor_prompts=assessor_prompts,
        injector_prompts=None,
        thinking_budget=512,
    )

    # Apply chat template with thinking
    prompt_assessor = tokenizer.apply_chat_template(
        messages_assessor,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )

    # Generate with BATCH thinking (answer_tokens=512 for assessor)
    assessment_outputs = generate_qwen_with_thinking_batch(
        model=model,
        tokenizer=tokenizer,
        prompts=[prompt_assessor],
        thinking_budget=512,
        max_new_tokens=1024,
        temperature=0.3,
        top_p=0.95,
        min_p=0.05,
        answer_tokens=512,  # Assessor needs less (just CORRECT/INCORRECT + explanation)
    )

    assessment_full = assessment_outputs[0]

    # Extract prediction
    prediction = extract_final_answer(assessment_full)

    print(f"Assessor prediction: {prediction}")

    # Show thinking
    if "<think>" in assessment_full and "</think>" in assessment_full:
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
print("✅ Test with BATCH Thinking Mode Complete!")
print("=" * 60)

if generated_note and result.passed:
    print("\nResults Summary:")
    print(f"  ✅ Production prompts loaded")
    print(f"  ✅ Thinking mode (COT): {has_thinking}")
    print(f"  ✅ Injector: {'Modified note' if generated_note != original_note else 'Needs review'}")
    print(f"  ✅ VCF: {'PASS' if result.passed else f'REJECT ({result.reason})'}")
    if prediction:
        print(f"  ✅ Assessor: {prediction} (expected: {ground_truth})")
        print(f"  ✅ Reward: {reward:.1f}")

    print("\n" + "=" * 60)
    print("✅ Ready for MedSeRL training!")
    print("=" * 60)
