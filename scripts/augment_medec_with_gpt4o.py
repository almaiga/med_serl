#!/usr/bin/env python3
"""
Augment MEDEC error pairs with GPT-4o reasoning and variations.

Strategy:
1. Start from verified MEDEC error pairs (don't generate errors from scratch)
2. Add rich clinical reasoning (teach small model WHY errors are wrong)
3. Create variations (same error pattern, different clinical context)
4. Use copy-then-modify format (teach precision)
5. Apply filters to verify correctness

Usage:
    python scripts/augment_medec_with_gpt4o.py \
        --input-jsonl data_processed/medec_paired/train_val_split/sft_train.jsonl \
        --output-dir data/sft_medec_augmented \
        --num-pairs 100 \
        --openai-api-key $OPENAI_API_KEY
"""

import argparse
import json
import os
from typing import Dict, List, Optional
from tqdm import tqdm

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("[ERROR] OpenAI package required: pip install openai")
    exit(1)


def load_augmentation_prompts(prompt_file: str) -> Dict:
    """Load augmentation prompts."""
    with open(prompt_file, 'r') as f:
        return json.load(f)


def call_gpt4o_json(
    user_prompt: str,
    system_prompt: str,
    client: OpenAI,
    temperature: float = 0.3,
) -> Dict:
    """Call GPT-4o with JSON mode."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=4000,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


def add_clinical_reasoning(
    medec_pair: Dict,
    prompts: Dict,
    client: OpenAI,
) -> Dict:
    """Task 1: Add rich clinical reasoning explaining the error."""
    system_prompt = prompts["augmentation_task_1_add_reasoning"]["system_prompt"]
    user_prompt = prompts["augmentation_task_1_add_reasoning"]["user_template"].format(
        incorrect_note=medec_pair["incorrect_note"],
        correct_note=medec_pair["correct_note"],
        error_type=medec_pair["error_type"],
        error_sentence=medec_pair["error_sentence"],
        corrected_sentence=medec_pair["corrected_sentence"],
    )

    reasoning = call_gpt4o_json(user_prompt, system_prompt, client, temperature=0.3)
    return reasoning


def create_variations(
    medec_pair: Dict,
    prompts: Dict,
    client: OpenAI,
) -> List[Dict]:
    """Task 2: Create variations with same error pattern."""
    system_prompt = prompts["augmentation_task_2_create_variations"]["system_prompt"]
    user_prompt = prompts["augmentation_task_2_create_variations"]["user_template"].format(
        incorrect_note=medec_pair["incorrect_note"],
        correct_note=medec_pair["correct_note"],
        error_type=medec_pair["error_type"],
        error_sentence=medec_pair["error_sentence"],
        corrected_sentence=medec_pair["corrected_sentence"],
    )

    variations = call_gpt4o_json(user_prompt, system_prompt, client, temperature=0.7)
    return variations.get("variations", [])


def create_copy_modify_example(
    medec_pair: Dict,
    prompts: Dict,
    client: OpenAI,
) -> Dict:
    """Task 3: Create copy-then-modify training format."""
    system_prompt = prompts["augmentation_task_3_copy_then_modify"]["system_prompt"]
    user_prompt = prompts["augmentation_task_3_copy_then_modify"]["user_template"].format(
        correct_note=medec_pair["correct_note"],
        error_sentence=medec_pair["corrected_sentence"],  # Note: we want to inject the error
        corrected_sentence=medec_pair["error_sentence"],  # So we reverse these
        error_type=medec_pair["error_type"],
    )

    copy_modify = call_gpt4o_json(user_prompt, system_prompt, client, temperature=0.2)
    return copy_modify


def verify_single_change(original: str, modified: str) -> bool:
    """Verify only one location was changed."""
    # Simple check: split into sentences and count differences
    orig_sentences = original.split('. ')
    mod_sentences = modified.split('. ')

    if len(orig_sentences) != len(mod_sentences):
        return False  # Structure changed

    differences = sum(1 for o, m in zip(orig_sentences, mod_sentences) if o != m)
    return differences == 1  # Exactly one sentence changed


def verify_numbers_preserved(original: str, modified: str, exclude_location: str) -> bool:
    """Verify all numbers preserved except in error location."""
    import re

    # Extract all numbers (labs, vitals, ages, doses)
    orig_numbers = set(re.findall(r'\d+\.?\d*', original.replace(exclude_location, '')))
    mod_numbers = set(re.findall(r'\d+\.?\d*', modified.replace(exclude_location, '')))

    return orig_numbers == mod_numbers


def apply_filters(
    original: str,
    modified: str,
    error_location: str,
) -> Dict:
    """Apply verification filters."""
    filters = {
        "single_change": verify_single_change(original, modified),
        "numbers_preserved": verify_numbers_preserved(original, modified, error_location),
        "not_identical": original != modified,
        "error_location_changed": error_location not in modified or modified != original,
    }

    passed = all(filters.values())

    return {
        "passed": passed,
        "filters": filters,
        "reason": None if passed else f"Failed: {[k for k, v in filters.items() if not v]}"
    }


def augment_medec_pair(
    medec_pair: Dict,
    prompts: Dict,
    client: OpenAI,
) -> Dict:
    """Augment a single MEDEC pair with all tasks."""
    result = {
        "original_pair": medec_pair,
        "augmentations": {}
    }

    # Task 1: Add reasoning
    try:
        reasoning = add_clinical_reasoning(medec_pair, prompts, client)
        result["augmentations"]["reasoning"] = reasoning
    except Exception as e:
        print(f"[WARNING] Failed to add reasoning: {e}")
        result["augmentations"]["reasoning"] = None

    # Task 2: Create variations
    try:
        variations = create_variations(medec_pair, prompts, client)
        result["augmentations"]["variations"] = variations
    except Exception as e:
        print(f"[WARNING] Failed to create variations: {e}")
        result["augmentations"]["variations"] = []

    # Task 3: Copy-then-modify
    try:
        copy_modify = create_copy_modify_example(medec_pair, prompts, client)
        result["augmentations"]["copy_then_modify"] = copy_modify
    except Exception as e:
        print(f"[WARNING] Failed to create copy-modify: {e}")
        result["augmentations"]["copy_then_modify"] = None

    # Apply filters to original pair
    filter_result = apply_filters(
        medec_pair["correct_note"],
        medec_pair["incorrect_note"],
        medec_pair["error_sentence"],
    )
    result["filter_verification"] = filter_result

    return result


def main():
    parser = argparse.ArgumentParser(description="Augment MEDEC with GPT-4o reasoning and variations")
    parser.add_argument("--input-jsonl", required=True, help="Input MEDEC JSONL file")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    parser.add_argument("--prompt-file", default="configs/prompts/gpt4o_medec_augmentation.json")
    parser.add_argument("--num-pairs", type=int, default=100, help="Number of MEDEC pairs to augment")
    parser.add_argument("--openai-api-key", required=True)

    args = parser.parse_args()

    # Initialize
    client = OpenAI(api_key=args.openai_api_key)
    prompts = load_augmentation_prompts(args.prompt_file)

    # Load MEDEC pairs
    with open(args.input_jsonl, 'r') as f:
        medec_pairs = [json.loads(line) for line in f]

    print(f"[INFO] Loaded {len(medec_pairs)} MEDEC pairs")
    print(f"[INFO] Augmenting {args.num_pairs} pairs")

    # Augment
    os.makedirs(args.output_dir, exist_ok=True)
    augmented_pairs = []

    stats = {
        "attempted": 0,
        "reasoning_added": 0,
        "variations_created": 0,
        "copy_modify_created": 0,
        "filter_passed": 0,
    }

    for i, medec_pair in enumerate(tqdm(medec_pairs[:args.num_pairs])):
        stats["attempted"] += 1

        augmented = augment_medec_pair(medec_pair, prompts, client)
        augmented_pairs.append(augmented)

        # Update stats
        if augmented["augmentations"].get("reasoning"):
            stats["reasoning_added"] += 1
        if augmented["augmentations"].get("variations"):
            stats["variations_created"] += len(augmented["augmentations"]["variations"])
        if augmented["augmentations"].get("copy_then_modify"):
            stats["copy_modify_created"] += 1
        if augmented["filter_verification"]["passed"]:
            stats["filter_passed"] += 1

    # Write outputs
    augmented_output = os.path.join(args.output_dir, "medec_augmented.jsonl")
    with open(augmented_output, 'w') as f:
        for aug in augmented_pairs:
            f.write(json.dumps(aug) + "\n")

    # Create flattened training format
    training_output = os.path.join(args.output_dir, "sft_training_data.jsonl")
    with open(training_output, 'w') as f:
        for aug in augmented_pairs:
            # Original pair
            f.write(json.dumps({
                "note_id": aug["original_pair"]["note_id"],
                "correct_note": aug["original_pair"]["correct_note"],
                "incorrect_note": aug["original_pair"]["incorrect_note"],
                "error_type": aug["original_pair"]["error_type"],
                "reasoning": aug["augmentations"].get("reasoning"),
                "source": "medec_original"
            }) + "\n")

            # Variations
            for var in aug["augmentations"].get("variations", []):
                f.write(json.dumps({
                    "note_id": aug["original_pair"]["note_id"] + f"_var{var['variation_id']}",
                    "correct_note": var["correct_note"],
                    "incorrect_note": var["incorrect_note"],
                    "error_type": aug["original_pair"]["error_type"],
                    "reasoning": var.get("error_explanation"),
                    "source": "gpt4o_variation"
                }) + "\n")

    # Write stats
    stats_output = os.path.join(args.output_dir, "augmentation_stats.json")
    with open(stats_output, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print("AUGMENTATION COMPLETE")
    print(f"{'='*60}")
    print(f"Pairs augmented: {stats['attempted']}")
    print(f"Reasoning added: {stats['reasoning_added']}")
    print(f"Variations created: {stats['variations_created']}")
    print(f"Copy-modify examples: {stats['copy_modify_created']}")
    print(f"Filter passed: {stats['filter_passed']}")
    print(f"\nOutputs:")
    print(f"  {augmented_output}")
    print(f"  {training_output}")
    print(f"  {stats_output}")


if __name__ == "__main__":
    main()
