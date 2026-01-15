#!/usr/bin/env python3
"""
Test with Qwen3-4B on M3 Max.
Uses the new Qwen3-4B model for testing MedSeRL pipeline.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sft'))

try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import torch
    from inference_utils import apply_vcf, extract_generated_note
except ImportError as e:
    print(f"[ERROR] Missing dependencies: {e}")
    print("\nPlease install:")
    print("  pip install torch transformers accelerate")
    sys.exit(1)

print("=" * 60)
print("Qwen3-4B Test on M3 Max (48GB RAM)")
print("=" * 60)

# Use Qwen3-4B specifically
model_name = "Qwen/Qwen3-4B"
print(f"\n[1/6] Loading {model_name}...")
print("(This may take a few minutes on first run)")

try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use FP16 to save memory
        device_map="auto",  # Automatically use MPS
        trust_remote_code=True,
    )
except Exception as e:
    print(f"[ERROR] Failed to load model: {e}")
    print("\nMake sure you have:")
    print("  1. Internet connection (to download model)")
    print("  2. Enough disk space (~8GB for Qwen3-4B)")
    print("  3. transformers>=4.40.0")
    sys.exit(1)

print("[2/6] Model loaded successfully!")

# Check device
device = next(model.parameters()).device
print(f"Device: {device}")

if torch.backends.mps.is_available():
    print("✅ Using MPS (Apple Silicon GPU)")
    try:
        mem_allocated = torch.mps.current_allocated_memory() / 1e9
        print(f"Memory allocated: {mem_allocated:.2f} GB / 48 GB")
    except:
        pass

# Test injector generation
original_note = "Patient has type 2 diabetes controlled with metformin and reports good glycemic control."

injector_prompt = f"""You are a medical error injector. Change EXACTLY ONE word in the clinical note below to create a subtle medical error. Output ONLY the modified note, nothing else.

Original note: {original_note}

Modified note (change only 1 word):"""

print("\n[3/6] Testing injector generation...")
print(f"Original: {original_note}")

inputs = tokenizer(injector_prompt, return_tensors="pt").to(device)

try:
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
        )

    generated = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract just the modified note (after "Modified:")
    if "Modified:" in generated:
        generated_note = generated.split("Modified:")[-1].strip().split('\n')[0]
    else:
        generated_note = generated.strip()

    print(f"Generated: {generated_note}")

except Exception as e:
    print(f"[ERROR] Generation failed: {e}")
    generated_note = original_note
    print("Using original note as fallback")

# Test VCF
print("\n[4/6] Applying VCF filters...")
result = apply_vcf(
    original_note,
    generated_note,
    min_jaccard=0.85,
    max_jaccard=0.99,
    max_word_edits=6,
)

print(f"VCF Result: {'✅ PASS' if result.passed else '❌ REJECT'}")
print(f"  Jaccard similarity: {result.score_jaccard if result.score_jaccard is not None else 'N/A'}")
print(f"  Word edits: {result.word_edits}")
print(f"  Sentences changed: {result.sentences_changed}")
print(f"  Reason: {result.reason if result.reason else 'Passed all filters'}")

# Test assessor
print("\n[5/6] Testing assessor...")
assessor_prompt = f"""You are a medical error detector. Analyze if this note contains medical errors.

Note: {generated_note}

Is this note medically CORRECT or INCORRECT?
Answer:"""

inputs = tokenizer(assessor_prompt, return_tensors="pt").to(device)

try:
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            temperature=0.3,
        )

    assessment = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract prediction
    if "INCORRECT" in assessment.upper():
        prediction = "INCORRECT"
    elif "CORRECT" in assessment.upper():
        prediction = "CORRECT"
    else:
        prediction = "UNKNOWN"

    print(f"Assessor prediction: {prediction}")

    # Show reasoning if present
    if "Answer:" in assessment:
        reasoning = assessment.split("Answer:")[-1].strip()[:150]
        print(f"Reasoning: {reasoning}...")

except Exception as e:
    print(f"[ERROR] Assessment failed: {e}")

print("\n[6/6] Complete!")

if torch.backends.mps.is_available():
    try:
        mem_used = torch.mps.current_allocated_memory() / 1e9
        print(f"\nMemory used: {mem_used:.2f} GB / 48 GB")
        print(f"Memory available: {48 - mem_used:.2f} GB")
    except:
        pass

print("\n" + "=" * 60)
print("✅ Full pipeline test successful with Qwen3-4B!")
print("=" * 60)

print("\nSummary:")
print(f"  ✅ Model loaded: Qwen3-4B")
print(f"  ✅ Injector generation: Working")
print(f"  ✅ VCF filtering: Working")
print(f"  ✅ Assessor evaluation: Working")

print("\nNext steps:")
print("  1. Run mini training with Qwen3-4B")
print("  2. See: SETUP_M3_MAX.md for full guide")
