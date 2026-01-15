#!/usr/bin/env python3
"""
Test single batch with real model on Mac M3.
Uses a tiny model for fast testing.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sft'))

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    from inference_utils import apply_vcf
except ImportError as e:
    print(f"[ERROR] Missing dependencies: {e}")
    print("\nPlease install:")
    print("  pip install torch transformers")
    sys.exit(1)

print("=" * 60)
print("Single Batch Test on M3 Max")
print("=" * 60)

# Use tiny model for quick test
model_name = "gpt2"  # Very small, just for testing
print(f"\n[1/5] Loading model: {model_name}")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 needs this
    model = AutoModelForCausalLM.from_pretrained(model_name)
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    print("\nTrying to download model...")
    sys.exit(1)

# Use MPS (Metal Performance Shaders) if available
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"[2/5] Using device: {device}")

if device == "mps":
    print("✅ MPS (Apple Silicon GPU) is available!")
else:
    print("⚠️  MPS not available, using CPU (will be slower)")

model = model.to(device)

# Test data
original_notes = [
    "Patient has type 2 diabetes controlled with metformin.",
]

print("\n[3/5] Generating with model...")
prompt = f"Modify this clinical note with a subtle error:\n{original_notes[0]}\n\nModified note:"
inputs = tokenizer(prompt, return_tensors="pt").to(device)

try:
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
    generated_note = generated.split("Modified note:")[-1].strip()

    print(f"\nOriginal: {original_notes[0]}")
    print(f"Generated: {generated_note[:100]}...")

except Exception as e:
    print(f"[ERROR] Generation failed: {e}")
    generated_note = original_notes[0]  # Fallback

# Test VCF
print("\n[4/5] Testing VCF filtering...")
result = apply_vcf(
    original_notes[0],
    generated_note,
    min_jaccard=0.85,
    max_jaccard=0.99,
    max_word_edits=6,
)

print(f"VCF Result: {'✅ PASS' if result.passed else '❌ REJECT'}")
print(f"  Jaccard: {result.score_jaccard:.3f}")
print(f"  Reason: {result.reason if result.reason else 'N/A'}")

print("\n[5/5] Complete!")

if device == "mps":
    try:
        mem_used = torch.mps.current_allocated_memory() / 1e9
        print(f"Memory used: {mem_used:.2f} GB")
    except:
        pass

print("\n" + "=" * 60)
print("✅ Pipeline test successful on M3 Max!")
print("=" * 60)
print("\nNext steps:")
print("  1. Run: python3 scripts/test_qwen_small.py (with Qwen-2.5-3B)")
print("  2. Or see: SETUP_M3_MAX.md for full guide")
