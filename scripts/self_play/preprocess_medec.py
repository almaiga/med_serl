"""Preprocess MEDEC data for self-play training with verl.

Converts JSONL data to verl-compatible format with all fields needed
for both Injector modes (benign and error injection).
"""

import json
import argparse
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq


def load_jsonl(filepath: Path) -> list[dict]:
    """Load JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def create_verl_example(
    example: dict,
    data_source: str = "medec_selfplay",
) -> dict:
    """Convert a MEDEC example to verl format.
    
    verl expects:
    - data_source: str
    - prompt: list[dict] (chat format)
    - ability: str (optional)
    - reward_model: dict with ground_truth
    - extra_info: dict with additional data
    
    For self-play, we store all note data in extra_info and
    construct prompts dynamically during rollout based on mode.
    """
    # Initial prompt is minimal - the game tool will construct
    # the actual prompt based on randomly selected mode
    initial_prompt = [
        {
            "role": "system",
            "content": "You are participating in a medical note game."
        },
        {
            "role": "user", 
            "content": "Awaiting game initialization..."
        }
    ]
    
    return {
        "data_source": data_source,
        "prompt": initial_prompt,
        "ability": "medical_error_detection",
        "reward_model": {
            "ground_truth": {
                "note_id": example.get("note_id", ""),
                "error_type": example.get("error_type", ""),
            }
        },
        "extra_info": {
            "note_id": example.get("note_id", ""),
            "correct_note": example["correct_note"],
            "incorrect_note": example["incorrect_note"],
            "error_type": example.get("error_type", ""),
            "error_sentence": example.get("error_sentence", ""),
            "corrected_sentence": example.get("corrected_sentence", ""),
        }
    }


def convert_to_parquet(
    input_path: Path,
    output_path: Path,
    data_source: str = "medec_selfplay",
) -> None:
    """Convert JSONL to Parquet format for verl."""
    
    examples = load_jsonl(input_path)
    print(f"Loaded {len(examples)} examples from {input_path}")
    
    verl_examples = [
        create_verl_example(ex, data_source) 
        for ex in examples
    ]
    
    # Convert to PyArrow table
    # verl expects specific schema
    schema = pa.schema([
        ("data_source", pa.string()),
        ("prompt", pa.string()),  # JSON-encoded chat messages
        ("ability", pa.string()),
        ("reward_model", pa.string()),  # JSON-encoded
        ("extra_info", pa.string()),  # JSON-encoded
    ])
    
    rows = []
    for ex in verl_examples:
        rows.append({
            "data_source": ex["data_source"],
            "prompt": json.dumps(ex["prompt"]),
            "ability": ex["ability"],
            "reward_model": json.dumps(ex["reward_model"]),
            "extra_info": json.dumps(ex["extra_info"]),
        })
    
    table = pa.Table.from_pylist(rows, schema=schema)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, output_path)
    print(f"Saved {len(rows)} examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MEDEC data for self-play training"
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
        "--data-source",
        type=str,
        default="medec_selfplay",
        help="Data source identifier",
    )
    
    args = parser.parse_args()
    convert_to_parquet(args.input, args.output, args.data_source)


if __name__ == "__main__":
    main()
