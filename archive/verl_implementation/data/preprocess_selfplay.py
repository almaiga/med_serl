#!/usr/bin/env python3
"""
Self-Play Data Preprocessor for verl

Converts medical note pairs into verl-compatible parquet format for the
two-phase self-play game (Injector → Assessor).

For each pair (correct_note, incorrect_note), generates 2 examples:
1. Benign mode: Injector makes surface edits, Assessor classifies → CORRECT
2. Error mode: Injector injects error, Assessor classifies → sentence number

Notes are pre-numbered (1-indexed) so the model can reference sentence numbers.
Ground truth is "CORRECT" or str(sentence_id).

Usage:
    python preprocess_selfplay.py --input /path/to/rl_train.jsonl --output_dir ~/data/selfplay
"""

import json
import argparse
import sys
import random
from pathlib import Path
from typing import List, Dict
import pandas as pd

# Add project root to path for shared utils
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.self_play.utils import number_sentences, find_error_sentence_id


def load_prompts(prompt_file: str) -> dict:
    """Load prompts from JSON config file."""
    with open(prompt_file, 'r') as f:
        return json.load(f)


def create_benign_example(
    pair: dict,
    injection_prompts: dict,
    idx: int
) -> dict:
    """Create a benign modification example (ground_truth = CORRECT).
    
    Injector receives pre-numbered sentences and makes surface edits.
    A random benign change type is selected from the prompt config.
    """
    system_prompt = injection_prompts["system_prompt_correct"]
    user_template = injection_prompts["injector_correct_template"]

    # Pre-number the sentences
    sentences = number_sentences(pair["correct_note"])

    # Pick a random benign change type
    change_types = injection_prompts.get("benign_change_types", {})
    change_type = random.choice(list(change_types.keys())) if change_types else "pseudo_factual"
    change_desc = change_types.get(change_type, "Replace a medical term with an equivalent synonym.")

    user_prompt = user_template.format(
        sentences=sentences,
        change_type=change_type,
        change_type_description=change_desc,
    )
    
    note_id = pair.get("note_id", f"selfplay-{idx}")
    
    return {
        "data_source": "medec_selfplay",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "ability": "medical_error_detection",
        "reward_model": {
            "style": "rule",
            "ground_truth": "CORRECT"
        },
        "interaction_kwargs": {
            "name": "medical_game",
            "ground_truth": "CORRECT",
            "mode": "benign",
            "sentences": sentences,
            "note_data": {
                "correct_note": pair["correct_note"],
                "note_id": f"{note_id}-benign"
            }
        },
        "extra_info": {
            "note_id": f"{note_id}-benign",
            "split": pair.get("split", "train"),
            "mode": "benign",
            "sentences": sentences,
            "change_type": change_type,
        }
    }


def create_error_example(
    pair: dict,
    injection_prompts: dict,
    idx: int
) -> dict:
    """Create an error injection example (ground_truth = sentence number).
    
    Injector receives pre-numbered sentences and error_type to guide injection.
    Ground truth is the 1-indexed sentence number containing the error.
    """
    system_prompt = injection_prompts["system_prompt_incorrect"]
    user_template = injection_prompts["injector_incorrect_template"]
    error_type = pair.get("error_type", "clinical error")

    # Pre-number the sentences (use incorrect_note so error_sentence can be located)
    sentences = number_sentences(pair["incorrect_note"])

    # Find error sentence ID by matching error_sentence text
    error_sentence_text = pair.get("error_sentence", "")
    error_sentence_id = find_error_sentence_id(pair["incorrect_note"], error_sentence_text)
    
    # Ground truth is the sentence number (as string), not "INCORRECT"
    ground_truth = str(error_sentence_id) if error_sentence_id else "INCORRECT"

    user_prompt = user_template.format(
        sentences=sentences,
        prompt_intent=error_type,
    )
    
    note_id = pair.get("note_id", f"selfplay-{idx}")
    
    return {
        "data_source": "medec_selfplay",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "ability": "medical_error_detection",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "interaction_kwargs": {
            "name": "medical_game",
            "ground_truth": ground_truth,
            "mode": "error_injection",
            "sentences": sentences,
            "error_type": error_type,
            "error_sentence_id": error_sentence_id,
            "note_data": {
                "correct_note": pair["correct_note"],
                "incorrect_note": pair["incorrect_note"],
                "error_type": pair.get("error_type", ""),
                "error_sentence": pair.get("error_sentence", ""),
                "corrected_sentence": pair.get("corrected_sentence", ""),
                "note_id": f"{note_id}-error"
            }
        },
        "extra_info": {
            "note_id": f"{note_id}-error",
            "split": pair.get("split", "train"),
            "mode": "error_injection",
            "sentences": sentences,
            "error_type": pair.get("error_type", ""),
            "error_sentence_id": error_sentence_id,
        }
    }


def process_selfplay_data(
    input_file: Path,
    injection_prompts: dict
) -> List[dict]:
    """Process medical note pairs into self-play examples.
    
    Each pair generates 2 examples:
    - 1 benign modification (CORRECT)
    - 1 error injection (INCORRECT)
    """
    examples = []
    
    print(f"Loading pairs from {input_file}")
    
    with open(input_file, 'r') as f:
        pairs = [json.loads(line) for line in f]
    
    print(f"Loaded {len(pairs)} pairs")
    
    for idx, pair in enumerate(pairs):
        # Validate required fields
        if not pair.get("correct_note") or not pair.get("incorrect_note"):
            print(f"Warning: Skipping pair {idx} - missing correct_note or incorrect_note")
            continue
        
        # Create benign example
        benign_example = create_benign_example(pair, injection_prompts, idx)
        examples.append(benign_example)
        
        # Create error example
        error_example = create_error_example(pair, injection_prompts, idx)
        examples.append(error_example)
    
    print(f"Generated {len(examples)} examples ({len(pairs)} benign + {len(pairs)} error)")
    
    return examples


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess medical note pairs for self-play RL training"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data_processed/medec_paired/train_val_split/rl_train.jsonl",
        help="Path to input JSONL file with note pairs"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_processed/selfplay",
        help="Directory to save output parquet files"
    )
    parser.add_argument(
        "--injection_prompts",
        type=str,
        default="configs/prompts/error_injection_prompts_v4.json",
        help="Path to injection prompts JSON"
    )
    parser.add_argument(
        "--split_ratio",
        type=float,
        default=0.9,
        help="Train/test split ratio (default: 0.9 for train)"
    )
    
    args = parser.parse_args()
    
    # Load prompts
    injection_prompts = load_prompts(args.injection_prompts)
    
    # Process data
    input_path = Path(args.input)
    examples = process_selfplay_data(input_path, injection_prompts)
    
    # Split into train/test
    split_idx = int(len(examples) * args.split_ratio)
    train_examples = examples[:split_idx]
    test_examples = examples[split_idx:]
    
    print(f"\nSplit: {len(train_examples)} train, {len(test_examples)} test")
    
    # Convert to DataFrame and save
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    train_df = pd.DataFrame(train_examples)
    test_df = pd.DataFrame(test_examples)
    
    train_path = output_dir / "train.parquet"
    test_path = output_dir / "test.parquet"
    
    train_df.to_parquet(train_path, index=False)
    test_df.to_parquet(test_path, index=False)
    
    print(f"\nSaved train data to: {train_path}")
    print(f"Saved test data to: {test_path}")
    
    # Print sample
    print("\n" + "="*80)
    print("SAMPLE BENIGN EXAMPLE:")
    print("="*80)
    sample_benign = train_examples[0]
    print(f"Data source: {sample_benign['data_source']}")
    print(f"Ground truth: {sample_benign['reward_model']['ground_truth']}")
    print(f"\nSystem prompt:\n{sample_benign['prompt'][0]['content'][:200]}...")
    print(f"\nUser prompt:\n{sample_benign['prompt'][1]['content'][:300]}...")
    print(f"\nInteraction kwargs: {sample_benign['interaction_kwargs']}")
    
    print("\n" + "="*80)
    print("SAMPLE ERROR EXAMPLE:")
    print("="*80)
    sample_error = train_examples[1]
    print(f"Data source: {sample_error['data_source']}")
    print(f"Ground truth: {sample_error['reward_model']['ground_truth']}")
    print(f"\nSystem prompt:\n{sample_error['prompt'][0]['content'][:200]}...")
    print(f"\nUser prompt:\n{sample_error['prompt'][1]['content'][:300]}...")
    print(f"\nInteraction kwargs: {sample_error['interaction_kwargs']}")


if __name__ == "__main__":
    main()
