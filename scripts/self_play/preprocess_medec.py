"""Preprocess MEDEC data for self-play training with verl.

Converts JSONL data to verl-compatible format.
For each note pair, generates 2 examples:
1. Benign mode (ground_truth="CORRECT") - only uses correct_note
2. Error mode (ground_truth="INCORRECT") - uses full pair for error injection

Prompts are loaded from JSON config files at preprocessing time.

CRITICAL: verl's RLHFDataset expects 'prompt' as a native list of message dicts,
NOT a JSON-encoded string. We use HuggingFace datasets for proper serialization.
"""

import json
import argparse
from pathlib import Path
from typing import Any


def load_jsonl(filepath: Path) -> list[dict]:
    """Load JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_prompts(prompt_file: str) -> dict:
    """Load prompts from JSON config file."""
    with open(prompt_file, 'r') as f:
        return json.load(f)


def create_benign_example(
    pair: dict,
    injection_prompts: dict,
    idx: int,
    data_source: str = "medec_selfplay",
) -> dict:
    """Create a benign modification example (ground_truth = CORRECT).
    
    Injector receives only the correct_note and makes surface edits.
    """
    system_prompt = injection_prompts["system_prompt_correct"]
    user_template = injection_prompts["injector_correct_template"]
    user_prompt = user_template.format(note=pair["correct_note"])
    
    note_id = pair.get("note_id", f"selfplay-{idx}")
    
    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "ability": "medical_error_detection",
        "reward_model": {
            "style": "rule",
            "ground_truth": "CORRECT"  # FIXED: was using note_id
        },
        "extra_info": {
            "note_id": f"{note_id}-benign",
            "correct_note": pair["correct_note"],
            "incorrect_note": "",  # Not needed for benign
            "error_type": "",
            "error_sentence": "",
            "corrected_sentence": "",
            "mode": "benign",
        }
    }


def create_error_example(
    pair: dict,
    injection_prompts: dict,
    idx: int,
    data_source: str = "medec_selfplay",
) -> dict:
    """Create an error injection example (ground_truth = INCORRECT).
    
    Injector receives the correct_note and error_type to guide injection.
    """
    system_prompt = injection_prompts["system_prompt_incorrect"]
    user_template = injection_prompts["injector_incorrect_template"]
    error_type = pair.get("error_type", "clinical error")
    user_prompt = user_template.format(
        note=pair["correct_note"],
        prompt_intent=error_type
    )
    
    note_id = pair.get("note_id", f"selfplay-{idx}")
    
    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "ability": "medical_error_detection",
        "reward_model": {
            "style": "rule",
            "ground_truth": "INCORRECT"  # FIXED: was using note_id
        },
        "extra_info": {
            "note_id": f"{note_id}-error",
            "correct_note": pair["correct_note"],
            "incorrect_note": pair["incorrect_note"],
            "error_type": pair.get("error_type", ""),
            "error_sentence": pair.get("error_sentence", ""),
            "corrected_sentence": pair.get("corrected_sentence", ""),
            "mode": "error_injection",
        }
    }


def convert_to_parquet(
    input_path: Path,
    output_path: Path,
    injection_prompts_path: str = "configs/prompts/error_injection_prompts_v2.json",
    data_source: str = "medec_selfplay",
) -> None:
    """Convert JSONL to Parquet format for verl.
    
    Each pair generates 2 examples (1 benign + 1 error).
    
    IMPORTANT: verl expects 'prompt' as a native list of dicts, NOT a JSON string.
    See: https://github.com/volcengine/verl/blob/main/verl/utils/dataset/rl_dataset.py
    """
    
    pairs = load_jsonl(input_path)
    print(f"Loaded {len(pairs)} pairs from {input_path}")
    
    # Load prompts from config
    injection_prompts = load_prompts(injection_prompts_path)
    print(f"Loaded prompts from {injection_prompts_path}")
    
    # Generate 2 examples per pair
    verl_examples = []
    for idx, pair in enumerate(pairs):
        if not pair.get("correct_note") or not pair.get("incorrect_note"):
            print(f"Warning: Skipping pair {idx} - missing required fields")
            continue
            
        # Benign example (CORRECT)
        benign_ex = create_benign_example(pair, injection_prompts, idx, data_source)
        verl_examples.append(benign_ex)
        
        # Error example (INCORRECT)
        error_ex = create_error_example(pair, injection_prompts, idx, data_source)
        verl_examples.append(error_ex)
    
    print(f"Generated {len(verl_examples)} examples ({len(pairs)} benign + {len(pairs)} error)")
    
    # Use HuggingFace datasets for proper serialization
    # verl uses datasets.load_dataset("parquet", ...) which handles list columns properly
    from datasets import Dataset
    
    # Flatten the data structure for HuggingFace datasets
    rows = []
    for ex in verl_examples:
        rows.append({
            "data_source": ex["data_source"],
            "prompt": ex["prompt"],  # Keep as native list, NOT json.dumps!
            "ability": ex["ability"],
            "reward_model": ex["reward_model"],  # Keep as native dict
            "extra_info": ex["extra_info"],  # Keep as native dict
        })
    
    # Create HuggingFace dataset and save as parquet
    hf_dataset = Dataset.from_list(rows)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hf_dataset.to_parquet(str(output_path))
    print(f"Saved {len(rows)} examples to {output_path}")
    
    # Print sample for verification
    if verl_examples:
        print("\n--- Sample BENIGN example ---")
        sample = verl_examples[0]
        print(f"ground_truth: {sample['reward_model']['ground_truth']}")
        print(f"mode: {sample['extra_info']['mode']}")
        user_content = sample['prompt'][1]['content'] if len(sample['prompt']) > 1 else sample['prompt'][0]['content']
        print(f"prompt preview (user): {user_content[:300]}...")
        
        print("\n--- Sample ERROR example ---")
        sample = verl_examples[1]
        print(f"ground_truth: {sample['reward_model']['ground_truth']}")
        print(f"mode: {sample['extra_info']['mode']}")
        user_content = sample['prompt'][1]['content'] if len(sample['prompt']) > 1 else sample['prompt'][0]['content']
        print(f"prompt preview (user): {user_content[:300]}...")
    
    # Verify the saved file
    print("\n--- Verifying saved parquet ---")
    loaded = Dataset.from_parquet(str(output_path))
    print(f"Loaded {len(loaded)} examples")
    first_row = loaded[0]
    print(f"prompt type: {type(first_row['prompt'])}")
    print(f"prompt[0]: {first_row['prompt'][0] if isinstance(first_row['prompt'], list) else 'NOT A LIST!'}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MEDEC data for self-play training (generates 2 examples per pair)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data_processed/medec_paired/train_val_split/rl_train.jsonl"),
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_processed/self_play/train.parquet"),
        help="Output Parquet file path",
    )
    parser.add_argument(
        "--injection-prompts",
        type=str,
        default="configs/prompts/error_injection_prompts_v2.json",
        help="Path to injection prompts JSON",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="medec_selfplay",
        help="Data source identifier",
    )
    
    args = parser.parse_args()
    convert_to_parquet(
        args.input, 
        args.output, 
        args.injection_prompts,
        args.data_source
    )


if __name__ == "__main__":
    main()
