#!/usr/bin/env python3
"""
Test safe SFT data generation with a small batch.

Quick verification before scaling to full dataset.
"""

import json
import os
from pathlib import Path

# Test configuration
TEST_CONFIG = {
    "input_jsonl": "data_processed/medec_paired/train_val_split/sft_train.jsonl",
    "output_dir": "data/sft_test_safe",
    "prompt_file": "configs/prompts/gpt4o_medec_safe_augmentation.json",
    "num_pairs": 10,  # Small test batch
    "add_reasoning": True,
}


def verify_output(output_dir: str):
    """Verify generated data quality."""
    combined_file = os.path.join(output_dir, "sft_combined_safe.jsonl")
    stats_file = os.path.join(output_dir, "generation_stats_safe.json")

    print("\n" + "="*60)
    print("Verification Report")
    print("="*60)

    # Load statistics
    with open(stats_file, 'r') as f:
        stats = json.load(f)

    print("\nGeneration Statistics:")
    print(f"  Total pairs: {stats['total_pairs']}")
    print(f"  CORRECT generated: {stats['correct_generated']}")
    print(f"  INCORRECT used: {stats['incorrect_used']}")
    print(f"  Reasoning added: {stats['reasoning_added']}")
    print(f"  Total tokens: {stats['total_tokens_used']:,}")

    # Load and analyze combined output
    examples = []
    with open(combined_file, 'r') as f:
        for line in f:
            examples.append(json.loads(line.strip()))

    print(f"\nTotal examples: {len(examples)}")

    # Count by label
    correct_count = sum(1 for ex in examples if ex["label"] == "CORRECT")
    incorrect_count = sum(1 for ex in examples if ex["label"] == "INCORRECT")

    print(f"  CORRECT: {correct_count} ({correct_count/len(examples)*100:.1f}%)")
    print(f"  INCORRECT: {incorrect_count} ({incorrect_count/len(examples)*100:.1f}%)")

    # Sample examples
    print("\n" + "="*60)
    print("Sample CORRECT Paraphrase:")
    print("="*60)
    correct_ex = next(ex for ex in examples if ex["label"] == "CORRECT")
    print(f"Note ID: {correct_ex['original_note_id']}")
    print(f"Source: {correct_ex['source']}")
    print(f"Note (first 500 chars):")
    print(correct_ex['note'][:500] + "...")

    print("\n" + "="*60)
    print("Sample INCORRECT (MEDEC verified):")
    print("="*60)
    incorrect_ex = next(ex for ex in examples if ex["label"] == "INCORRECT")
    print(f"Note ID: {incorrect_ex['original_note_id']}")
    print(f"Source: {incorrect_ex['source']}")
    print(f"Error type: {incorrect_ex['error_type']}")
    print(f"Error location: {incorrect_ex['error_location']}")
    print(f"Correction: {incorrect_ex['correction']}")
    if incorrect_ex.get("reasoning"):
        print(f"\nReasoning provided:")
        reasoning = incorrect_ex["reasoning"]
        print(f"  Error identified: {reasoning.get('error_identified', 'N/A')}")
        print(f"  Why wrong: {reasoning.get('why_wrong', 'N/A')[:200]}...")

    print("\n" + "="*60)
    print("✅ Verification complete - review output before scaling")
    print("="*60)


if __name__ == "__main__":
    import subprocess
    import sys

    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Set it with: export OPENAI_API_KEY=your-key-here")
        sys.exit(1)

    # Run generation script
    cmd = [
        "python3",
        "scripts/generate_sft_data_safe.py",
        "--input-jsonl", TEST_CONFIG["input_jsonl"],
        "--output-dir", TEST_CONFIG["output_dir"],
        "--prompt-file", TEST_CONFIG["prompt_file"],
        "--num-pairs", str(TEST_CONFIG["num_pairs"]),
        "--openai-api-key", os.environ["OPENAI_API_KEY"],
    ]

    if TEST_CONFIG["add_reasoning"]:
        cmd.append("--add-reasoning")

    print("Running safe SFT data generation test...")
    print(f"Command: {' '.join(cmd)}\n")

    result = subprocess.run(cmd, cwd=Path(__file__).parent.parent)

    if result.returncode == 0:
        # Verify output
        verify_output(TEST_CONFIG["output_dir"])
    else:
        print("Generation failed - check errors above")
        sys.exit(1)
