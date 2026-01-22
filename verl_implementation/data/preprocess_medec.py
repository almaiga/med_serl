#!/usr/bin/env python3
"""
MEDEC Data Preprocessor for verl

Converts MEDEC clinical note error detection data into verl-compatible parquet format.
Supports both MS (Microsoft) and UW (University of Washington) datasets.

Usage:
    python preprocess_medec.py --input_dir /path/to/MEDEC --output_dir ~/data/medec

Output Format (verl parquet):
    - data_source: str ("medec_ms" or "medec_uw")
    - prompt: list[dict] (chat format messages)
    - ability: str ("medical_error_detection")
    - reward_model: dict (ground_truth for reward computation)
    - extra_info: dict (metadata)
"""

import json
import argparse
import os
from pathlib import Path
from typing import Optional
import pandas as pd

# System prompt for medical error detection
SYSTEM_PROMPT = """You are a medical expert specializing in clinical documentation quality assurance.

Your task is to analyze clinical notes for medical errors. When presented with a clinical note:
1. Determine if there is a medical error (ERROR or CORRECT)
2. If ERROR, identify the problematic sentence
3. If ERROR, provide the corrected version
4. If ERROR, classify the error type

Output your analysis in the following JSON format:
{
    "assessment": "ERROR" or "CORRECT",
    "reasoning": "<brief explanation of your analysis>",
    "error_sentence": "<the problematic sentence if ERROR, null otherwise>",
    "corrected_sentence": "<the corrected version if ERROR, null otherwise>",
    "error_type": "<one of: diagnosis, management, treatment, pharmacotherapy, causalorganism, or null if CORRECT>"
}"""

USER_PROMPT_TEMPLATE = """Analyze the following clinical note for medical errors:

---
{clinical_note}
---

Provide your analysis in the specified JSON format."""


def load_medec_csv(filepath: Path) -> pd.DataFrame:
    """Load MEDEC CSV file with proper encoding handling."""
    try:
        df = pd.read_csv(filepath, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(filepath, encoding='latin-1')
    return df


def process_medec_ms(data_dir: Path) -> list[dict]:
    """Process MEDEC-MS dataset (Microsoft format)."""
    examples = []
    ms_dir = data_dir / "MEDEC-MS"
    
    if not ms_dir.exists():
        print(f"Warning: MEDEC-MS directory not found at {ms_dir}")
        return examples
    
    # Find all CSV files
    csv_files = list(ms_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in MEDEC-MS")
    
    for csv_file in csv_files:
        split_name = csv_file.stem.lower()  # e.g., "Training_Set_MS" -> "training_set_ms"
        
        # Determine split
        if "train" in split_name:
            split = "train"
        elif "val" in split_name or "dev" in split_name:
            split = "val"
        elif "test" in split_name:
            split = "test"
        else:
            split = "unknown"
        
        df = load_medec_csv(csv_file)
        print(f"Processing {csv_file.name}: {len(df)} rows")
        
        # Expected columns: Text ID, Text, Sentences, Error Flag, Error Type, 
        # Error Sentence ID, Error Sentence, Corrected Sentence, Corrected Text
        required_cols = ['Text ID', 'Text', 'Error Flag']
        if not all(col in df.columns for col in required_cols):
            print(f"  Skipping {csv_file.name}: missing required columns")
            print(f"  Found columns: {df.columns.tolist()}")
            continue
        
        for _, row in df.iterrows():
            note_id = str(row.get('Text ID', ''))
            text = str(row.get('Text', ''))
            error_flag = str(row.get('Error Flag', '')).strip().upper()
            
            # Skip empty texts
            if not text or pd.isna(text):
                continue
            
            has_error = error_flag == '1' or 'ERROR' in error_flag or error_flag == 'TRUE'
            
            # Build ground truth
            ground_truth = {
                "has_error": has_error,
                "error_sentence": str(row.get('Error Sentence', '')) if has_error and pd.notna(row.get('Error Sentence')) else None,
                "corrected_sentence": str(row.get('Corrected Sentence', '')) if has_error and pd.notna(row.get('Corrected Sentence')) else None,
                "error_type": str(row.get('Error Type', '')).lower() if has_error and pd.notna(row.get('Error Type')) else None,
                "corrected_text": str(row.get('Corrected Text', '')) if has_error and pd.notna(row.get('Corrected Text')) else None,
            }
            
            # Build verl example
            example = {
                "data_source": "medec_ms",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(clinical_note=text)}
                ],
                "ability": "medical_error_detection",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth
                },
                "extra_info": {
                    "note_id": f"ms-{split}-{note_id}",
                    "split": split,
                    "original_file": csv_file.name
                }
            }
            examples.append(example)
    
    return examples


def process_medec_uw(data_dir: Path) -> list[dict]:
    """Process MEDEC-UW dataset (University of Washington format)."""
    examples = []
    uw_dir = data_dir / "MEDEC-UW"
    
    if not uw_dir.exists():
        print(f"Warning: MEDEC-UW directory not found at {uw_dir}")
        return examples
    
    csv_files = list(uw_dir.glob("*.csv"))
    print(f"Found {len(csv_files)} CSV files in MEDEC-UW")
    
    for csv_file in csv_files:
        split_name = csv_file.stem.lower()
        
        if "train" in split_name:
            split = "train"
        elif "val" in split_name or "dev" in split_name:
            split = "val"
        elif "test" in split_name:
            split = "test"
        else:
            split = "unknown"
        
        df = load_medec_csv(csv_file)
        print(f"Processing {csv_file.name}: {len(df)} rows")
        
        # UW format may have different column names
        # Try to find the text column
        text_col = None
        for col in ['Text', 'text', 'Clinical Note', 'clinical_note', 'note']:
            if col in df.columns:
                text_col = col
                break
        
        if text_col is None:
            print(f"  Skipping {csv_file.name}: no text column found")
            print(f"  Found columns: {df.columns.tolist()}")
            continue
        
        id_col = None
        for col in ['Text ID', 'text_id', 'ID', 'id', 'note_id']:
            if col in df.columns:
                id_col = col
                break
        
        for idx, row in df.iterrows():
            note_id = str(row.get(id_col, idx)) if id_col else str(idx)
            text = str(row.get(text_col, ''))
            
            if not text or pd.isna(text):
                continue
            
            # Determine error status
            error_flag = None
            for col in ['Error Flag', 'error_flag', 'has_error', 'Error']:
                if col in df.columns:
                    error_flag = row.get(col)
                    break
            
            if error_flag is not None:
                has_error = str(error_flag).strip().upper() in ['1', 'TRUE', 'YES', 'ERROR']
            else:
                has_error = False  # Default if no flag column
            
            # Extract error details
            error_sentence = None
            corrected_sentence = None
            error_type = None
            corrected_text = None
            
            if has_error:
                for col in ['Error Sentence', 'error_sentence']:
                    if col in df.columns and pd.notna(row.get(col)):
                        error_sentence = str(row.get(col))
                        break
                
                for col in ['Corrected Sentence', 'corrected_sentence']:
                    if col in df.columns and pd.notna(row.get(col)):
                        corrected_sentence = str(row.get(col))
                        break
                
                for col in ['Error Type', 'error_type']:
                    if col in df.columns and pd.notna(row.get(col)):
                        error_type = str(row.get(col)).lower()
                        break
                
                for col in ['Corrected Text', 'corrected_text']:
                    if col in df.columns and pd.notna(row.get(col)):
                        corrected_text = str(row.get(col))
                        break
            
            ground_truth = {
                "has_error": has_error,
                "error_sentence": error_sentence,
                "corrected_sentence": corrected_sentence,
                "error_type": error_type,
                "corrected_text": corrected_text,
            }
            
            example = {
                "data_source": "medec_uw",
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_PROMPT_TEMPLATE.format(clinical_note=text)}
                ],
                "ability": "medical_error_detection",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth
                },
                "extra_info": {
                    "note_id": f"uw-{split}-{note_id}",
                    "split": split,
                    "original_file": csv_file.name
                }
            }
            examples.append(example)
    
    return examples


def process_existing_jsonl(data_dir: Path) -> list[dict]:
    """Process existing JSONL files in data_processed directory."""
    examples = []
    
    # Check for RL training data
    rl_train_path = data_dir.parent.parent / "data_processed" / "medec_paired" / "train_val_split" / "rl_train.jsonl"
    rl_val_path = data_dir.parent.parent / "data_processed" / "medec_paired" / "train_val_split" / "rl_val.jsonl"
    
    for jsonl_path, split in [(rl_train_path, "train"), (rl_val_path, "val")]:
        if jsonl_path.exists():
            print(f"Processing existing JSONL: {jsonl_path}")
            with open(jsonl_path, 'r') as f:
                for line in f:
                    try:
                        data = json.loads(line.strip())
                    except json.JSONDecodeError:
                        continue
                    
                    # Use incorrect_note as the input (has error)
                    incorrect_note = data.get('incorrect_note', '')
                    correct_note = data.get('correct_note', '')
                    
                    if not incorrect_note:
                        continue
                    
                    ground_truth = {
                        "has_error": True,  # All entries in RL data have errors
                        "error_sentence": data.get('error_sentence'),
                        "corrected_sentence": data.get('corrected_sentence'),
                        "error_type": data.get('error_type'),
                        "corrected_text": correct_note,
                    }
                    
                    example = {
                        "data_source": "medec_jsonl",
                        "prompt": [
                            {"role": "system", "content": SYSTEM_PROMPT},
                            {"role": "user", "content": USER_PROMPT_TEMPLATE.format(clinical_note=incorrect_note)}
                        ],
                        "ability": "medical_error_detection",
                        "reward_model": {
                            "style": "rule",
                            "ground_truth": ground_truth
                        },
                        "extra_info": {
                            "note_id": data.get('note_id', ''),
                            "split": split,
                            "original_file": jsonl_path.name
                        }
                    }
                    examples.append(example)
    
    return examples


def save_parquet(examples: list[dict], output_dir: Path, split: str):
    """Save examples to parquet file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Filter by split
    split_examples = [e for e in examples if e['extra_info']['split'] == split]
    
    if not split_examples:
        print(f"No examples for split '{split}'")
        return
    
    # Convert to DataFrame
    df = pd.DataFrame(split_examples)
    
    # Save to parquet
    output_path = output_dir / f"medec_{split}.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(split_examples)} examples to {output_path}")
    
    # Also save JSONL for debugging
    jsonl_path = output_dir / f"medec_{split}.jsonl"
    with open(jsonl_path, 'w') as f:
        for ex in split_examples:
            f.write(json.dumps(ex) + '\n')
    print(f"Saved JSONL copy to {jsonl_path}")


def main():
    parser = argparse.ArgumentParser(description="Preprocess MEDEC data for verl")
    parser.add_argument("--input_dir", type=str, default="data_raw/MEDEC",
                        help="Input directory containing MEDEC data")
    parser.add_argument("--output_dir", type=str, default="~/data/medec",
                        help="Output directory for parquet files")
    parser.add_argument("--include_jsonl", action="store_true",
                        help="Also process existing JSONL files from data_processed")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Collect all examples
    all_examples = []
    
    # Process MEDEC-MS
    ms_examples = process_medec_ms(input_dir)
    print(f"Loaded {len(ms_examples)} examples from MEDEC-MS")
    all_examples.extend(ms_examples)
    
    # Process MEDEC-UW
    uw_examples = process_medec_uw(input_dir)
    print(f"Loaded {len(uw_examples)} examples from MEDEC-UW")
    all_examples.extend(uw_examples)
    
    # Optionally process existing JSONL
    if args.include_jsonl:
        jsonl_examples = process_existing_jsonl(input_dir)
        print(f"Loaded {len(jsonl_examples)} examples from existing JSONL")
        all_examples.extend(jsonl_examples)
    
    print(f"\nTotal examples: {len(all_examples)}")
    
    # Print split distribution
    split_counts = {}
    for ex in all_examples:
        split = ex['extra_info']['split']
        split_counts[split] = split_counts.get(split, 0) + 1
    print(f"Split distribution: {split_counts}")
    
    # Print error distribution
    error_counts = {"error": 0, "correct": 0}
    for ex in all_examples:
        if ex['reward_model']['ground_truth']['has_error']:
            error_counts["error"] += 1
        else:
            error_counts["correct"] += 1
    print(f"Error distribution: {error_counts}")
    
    # Save to parquet by split
    for split in ['train', 'val', 'test', 'unknown']:
        save_parquet(all_examples, output_dir, split)
    
    print("\nDone! Files saved to:", output_dir)
    print("\nTo use with verl, set:")
    print(f'  data.train_files="{output_dir}/medec_train.parquet"')
    print(f'  data.val_files="{output_dir}/medec_val.parquet"')


if __name__ == "__main__":
    main()
