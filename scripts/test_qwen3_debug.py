#!/usr/bin/env python3
"""
Debug version - print full outputs to see what Qwen3 produces.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sft'))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from inference_utils import (
    build_messages, load_assessor_prompts, load_injector_prompts,
)

# Import generation function from quick_infer
from quick_infer_qwen3_4b_lora import generate_qwen_with_thinking

# Paths
PROMPT_DIR = "/Users/josmaiga/Documents/GitHub/med_serl/configs/prompts"

# Load prompts
assessor_prompts = load_assessor_prompts(f"{PROMPT_DIR}/error_detection_prompts.json")
injector_prompts = load_injector_prompts(f"{PROMPT_DIR}/error_injection_prompts_v2.json")

print("=" * 60)
print("DEBUG: Qwen3-4B Output Inspection")
print("=" * 60)

# Load model
model_name = "Qwen/Qwen3-4B"
print(f"\n[1/3] Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

print("[2/3] Model loaded!")
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

# Build injector messages
error_type = "medication error"
messages_injector = build_messages(
    mode="injector",
    note=original_note,
    prompt_intent=error_type,
    assessor_prompts=None,
    injector_prompts=injector_prompts,
    thinking_budget=512,
    injector_is_correct=False,
)

print("\n[3/3] INJECTOR generating...")
print(f"Error to inject: {error_type}")

# Apply chat template with thinking enabled
prompt_injector = tokenizer.apply_chat_template(
    messages_injector,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)

print("\n" + "=" * 60)
print("PROMPT SENT TO MODEL:")
print("=" * 60)
print(prompt_injector[-500:])  # Last 500 chars of prompt

# Generate with thinking
generated_full = generate_qwen_with_thinking(
    model=model,
    tokenizer=tokenizer,
    prompt=prompt_injector,
    thinking_budget=256,  # Reduced to leave room for answer
    max_new_tokens=1536,  # Increased total
    temperature=0.7,
    top_p=0.9,
    min_p=0.05,
)

print("\n" + "=" * 60)
print("FULL GENERATED OUTPUT:")
print("=" * 60)
print(generated_full)
print("\n" + "=" * 60)

print(f"\nOutput length: {len(generated_full)} chars")
print(f"Has <think> tags: {'Yes' if '<think>' in generated_full else 'No'}")
print(f"Has generated_note: {'Yes' if 'generated_note:' in generated_full.lower() else 'No'}")
print(f"Has final_answer: {'Yes' if 'final_answer:' in generated_full.lower() else 'No'}")

if torch.backends.mps.is_available():
    mem_used = torch.mps.current_allocated_memory() / 1e9
    print(f"\nMemory: {mem_used:.2f} GB / 48 GB")

print("\n" + "=" * 60)
print("DEBUG Complete!")
print("=" * 60)
