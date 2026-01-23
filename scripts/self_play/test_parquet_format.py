#!/usr/bin/env python3
"""Test the self-play data preprocessing and verify parquet format.

Run this before training to verify:
1. Parquet file loads correctly
2. 'prompt' field is a native list (NOT JSON string)
3. ground_truth is "CORRECT" or "INCORRECT"
4. Prompts contain actual clinical content
"""

import json
from pathlib import Path

from datasets import Dataset


def test_parquet_format(parquet_path: str = "data_processed/self_play/train.parquet"):
    """Verify parquet file format matches verl expectations."""
    
    print(f"Testing: {parquet_path}")
    print("=" * 60)
    
    # Load with HuggingFace datasets (same as verl)
    ds = Dataset.from_parquet(parquet_path)
    print(f"Total examples: {len(ds)}")
    
    # Check first example
    example = ds[0]
    
    print("\n--- Field Types ---")
    for key, val in example.items():
        val_type = type(val).__name__
        if isinstance(val, list):
            val_type = f"list[{type(val[0]).__name__}]" if val else "list[empty]"
        elif isinstance(val, dict):
            val_type = f"dict with keys: {list(val.keys())}"
        print(f"  {key}: {val_type}")
    
    # Critical check: prompt should be a list, not a string
    print("\n--- Critical Checks ---")
    prompt = example.get("prompt")
    if isinstance(prompt, str):
        print("❌ FAIL: 'prompt' is a STRING - verl expects a LIST of message dicts!")
        print(f"   Got: {prompt[:100]}...")
        return False
    elif isinstance(prompt, list):
        print("✅ PASS: 'prompt' is a LIST")
        print(f"   Roles: {[m.get('role') for m in prompt]}")
    else:
        print(f"❌ FAIL: 'prompt' is unexpected type: {type(prompt)}")
        return False
    
    # Check ground_truth
    reward_model = example.get("reward_model", {})
    ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
    
    if ground_truth in ["CORRECT", "INCORRECT"]:
        print(f"✅ PASS: ground_truth is '{ground_truth}'")
    else:
        print(f"❌ FAIL: ground_truth is '{ground_truth}' - expected 'CORRECT' or 'INCORRECT'")
        return False
    
    # Check mode
    extra_info = example.get("extra_info", {})
    mode = extra_info.get("mode") if isinstance(extra_info, dict) else None
    if mode in ["benign", "error_injection"]:
        print(f"✅ PASS: mode is '{mode}'")
    else:
        print(f"❌ FAIL: mode is '{mode}' - expected 'benign' or 'error_injection'")
        return False
    
    # Check prompt content is not placeholder
    user_content = ""
    for msg in prompt:
        if msg.get("role") == "user":
            user_content = msg.get("content", "")
            break
    
    if "Awaiting" in user_content or "incomplete" in user_content.lower():
        print(f"❌ FAIL: Prompt contains placeholder text!")
        print(f"   Content: {user_content[:200]}")
        return False
    elif len(user_content) > 100:  # Should have substantial clinical content
        print(f"✅ PASS: Prompt has substantial content ({len(user_content)} chars)")
    else:
        print(f"⚠️  WARNING: Prompt might be too short ({len(user_content)} chars)")
    
    # Count benign vs error examples
    print("\n--- Distribution ---")
    benign_count = 0
    error_count = 0
    for ex in ds:
        ei = ex.get("extra_info", {})
        m = ei.get("mode") if isinstance(ei, dict) else None
        if m == "benign":
            benign_count += 1
        elif m == "error_injection":
            error_count += 1
    
    print(f"  Benign (CORRECT):  {benign_count}")
    print(f"  Error (INCORRECT): {error_count}")
    
    if benign_count > 0 and error_count > 0:
        print("✅ PASS: Both modes present")
    else:
        print("❌ FAIL: Missing one mode")
        return False
    
    # Show sample prompts
    print("\n--- Sample BENIGN Prompt (first 500 chars) ---")
    for ex in ds:
        ei = ex.get("extra_info", {})
        if isinstance(ei, dict) and ei.get("mode") == "benign":
            for msg in ex["prompt"]:
                if msg.get("role") == "user":
                    print(msg.get("content", "")[:500])
                    break
            break
    
    print("\n--- Sample ERROR Prompt (first 500 chars) ---")
    for ex in ds:
        ei = ex.get("extra_info", {})
        if isinstance(ei, dict) and ei.get("mode") == "error_injection":
            for msg in ex["prompt"]:
                if msg.get("role") == "user":
                    print(msg.get("content", "")[:500])
                    break
            break
    
    print("\n" + "=" * 60)
    print("All checks passed! Ready for verl training.")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import sys
    parquet_path = sys.argv[1] if len(sys.argv) > 1 else "data_processed/self_play/train.parquet"
    success = test_parquet_format(parquet_path)
    sys.exit(0 if success else 1)
