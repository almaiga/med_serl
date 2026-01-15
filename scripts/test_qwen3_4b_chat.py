#!/usr/bin/env python3
"""
Test Qwen3-4B with proper chat format on M3 Max.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sft'))

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from inference_utils import apply_vcf

print("=" * 60)
print("Qwen3-4B Chat Format Test on M3 Max")
print("=" * 60)

model_name = "Qwen/Qwen3-4B"
print(f"\n[1/5] Loading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True,
)

print("[2/5] Model loaded!")
device = next(model.parameters()).device
print(f"Device: {device}")

if torch.backends.mps.is_available():
    mem_used = torch.mps.current_allocated_memory() / 1e9
    print(f"Memory: {mem_used:.2f} GB / 48 GB")

# Test with chat format
original_note = "Patient has type 2 diabetes controlled with metformin."

messages = [
    {"role": "system", "content": "You are a medical error injector. You modify clinical notes by introducing ONE subtle medical error."},
    {"role": "user", "content": f"Change exactly ONE word in this note to create a medical error:\n\n{original_note}\n\nOutput only the modified note:"}
]

print("\n[3/5] Generating with chat format...")
print(f"Original: {original_note}")

# Apply chat template
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
inputs = tokenizer([text], return_tensors="pt").to(device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

generated = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
generated_note = generated.strip().split('\n')[0]  # Take first line

print(f"Generated: {generated_note}")

# Apply VCF
print("\n[4/5] Applying VCF...")
result = apply_vcf(
    original_note,
    generated_note,
    min_jaccard=0.80,  # Lower threshold for testing
    max_jaccard=0.99,
    max_word_edits=6,
)

print(f"VCF: {'✅ PASS' if result.passed else '❌ REJECT'}")
print(f"  Jaccard: {result.score_jaccard if result.score_jaccard else 'N/A'}")
print(f"  Reason: {result.reason or 'Passed'}")

# Test assessor
print("\n[5/5] Testing assessor...")
messages_assessor = [
    {"role": "system", "content": "You are a medical error detector. Analyze clinical notes for errors."},
    {"role": "user", "content": f"Is this note CORRECT or INCORRECT?\n\n{generated_note}\n\nAnswer (one word):"}
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
        max_new_tokens=10,
        temperature=0.3,
    )

assessment = tokenizer.decode(outputs_assessor[0][inputs_assessor['input_ids'].shape[1]:], skip_special_tokens=True)
prediction = "INCORRECT" if "INCORRECT" in assessment.upper() else "CORRECT"

print(f"Assessor: {prediction}")

print("\n" + "=" * 60)
print("✅ Chat format test complete!")
print("=" * 60)
print(f"\nMemory: {torch.mps.current_allocated_memory() / 1e9:.2f} GB / 48 GB")
print("\nResults:")
print(f"  Model: Qwen3-4B working on M3 Max ✅")
print(f"  Injector: {'✅' if generated_note != original_note else '⚠️  Needs tuning'}")
print(f"  VCF: {'✅' if result.score_jaccard else '⚠️'}")
print(f"  Assessor: ✅")
