#!/usr/bin/env python3
"""
Debug version to see full output from batch thinking mode.
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sft'))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from inference_utils import build_messages, load_injector_prompts
from quick_infer_qwen3_4b_lora import generate_qwen_with_thinking_batch

PROMPT_DIR = "/Users/josmaiga/Documents/GitHub/med_serl/configs/prompts"
injector_prompts = load_injector_prompts(f"{PROMPT_DIR}/error_injection_prompts_v2.json")

print("=" * 60)
print("DEBUG: Full Output from Batch Thinking Mode")
print("=" * 60)

model_name = "Qwen/Qwen3-4B"
print(f"\nLoading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

print("Model loaded!")

original_note = """Patient presents with 3-week history of progressive dyspnea on exertion.
Physical exam reveals bilateral lower extremity edema and elevated jugular venous pressure.
Chest X-ray shows cardiomegaly and pulmonary congestion.
BNP level is markedly elevated at 850 pg/mL.
Started on furosemide 40mg daily and lisinopril 10mg daily for acute decompensated heart failure."""

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

prompt_injector = tokenizer.apply_chat_template(
    messages_injector,
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=True,
)

print("\nGenerating...")

generated_outputs = generate_qwen_with_thinking_batch(
    model=model,
    tokenizer=tokenizer,
    prompts=[prompt_injector],
    thinking_budget=512,
    max_new_tokens=1536,
    temperature=0.7,
    top_p=0.9,
    min_p=0.05,
    answer_tokens=1024,
)

generated_full = generated_outputs[0]

print("\n" + "=" * 60)
print("FULL GENERATED OUTPUT:")
print("=" * 60)
print(generated_full)
print("=" * 60)

print(f"\nLength: {len(generated_full)} chars")
print(f"Has <think>: {'Yes' if '<think>' in generated_full else 'No'}")
print(f"Has </think>: {'Yes' if '</think>' in generated_full else 'No'}")
print(f"Has 'generated_note:': {'Yes' if 'generated_note:' in generated_full.lower() else 'No'}")
print(f"Has 'final_answer:': {'Yes' if 'final_answer:' in generated_full.lower() else 'No'}")
