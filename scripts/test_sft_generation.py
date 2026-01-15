#!/usr/bin/env python3
"""
Test SFT data generation pipeline with a small sample.

Usage:
    export OPENAI_API_KEY="your-key"
    python scripts/test_sft_generation.py
"""

import os
import sys
import json

# Test with mock data
test_notes = [
    {
        "text": "Patient presents with 3-week history of progressive dyspnea on exertion. Physical exam reveals bilateral lower extremity edema and elevated jugular venous pressure. Chest X-ray shows cardiomegaly and pulmonary congestion. BNP level is markedly elevated at 850 pg/mL. Started on furosemide 40mg daily and lisinopril 10mg daily for acute decompensated heart failure."
    },
    {
        "text": "Patient has type 2 diabetes mellitus controlled with metformin 1000mg twice daily. Hemoglobin A1c is 6.8%. Reports good adherence to medication and diet. No hypoglycemic episodes in past 3 months."
    },
]

# Write test input
os.makedirs("data/test_sft", exist_ok=True)
with open("data/test_sft/test_notes.jsonl", "w") as f:
    for note in test_notes:
        f.write(json.dumps(note) + "\n")

print("=" * 60)
print("Testing SFT Data Generation Pipeline")
print("=" * 60)

# Check for API key
if not os.getenv("OPENAI_API_KEY"):
    print("\n❌ ERROR: OPENAI_API_KEY not set")
    print("Set it with: export OPENAI_API_KEY='your-key'")
    sys.exit(1)

print("\n[INFO] Test input created: data/test_sft/test_notes.jsonl")
print("[INFO] OpenAI API key found")

# Run generation
print("\n[INFO] Running generation pipeline...")
print("Command:")
cmd = """python scripts/generate_sft_data.py \\
    --input-jsonl data/test_sft/test_notes.jsonl \\
    --output-dir data/test_sft/output \\
    --prompt-file configs/prompts/error_injection_prompts_v3_enhanced.json \\
    --num-correct 2 \\
    --num-incorrect 2 \\
    --openai-api-key $OPENAI_API_KEY"""

print(cmd)

import subprocess
result = subprocess.run(cmd, shell=True, capture_output=False, text=True)

if result.returncode == 0:
    print("\n✅ Generation completed successfully!")
    print("\nCheck outputs:")
    print("  data/test_sft/output/sft_correct_verified.jsonl")
    print("  data/test_sft/output/sft_incorrect_verified.jsonl")
    print("  data/test_sft/output/generation_stats.json")

    # Show stats
    if os.path.exists("data/test_sft/output/generation_stats.json"):
        with open("data/test_sft/output/generation_stats.json") as f:
            stats = json.load(f)
        print("\n" + "=" * 60)
        print("Generation Statistics:")
        print("=" * 60)
        print(json.dumps(stats, indent=2))
else:
    print("\n❌ Generation failed!")
    sys.exit(1)
