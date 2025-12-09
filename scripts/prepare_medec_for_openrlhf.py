#!/usr/bin/env python3
"""
Prepare MEDEC dataset for OpenRLHF training.

Converts MEDEC data into the JSONL format expected by OpenRLHF:
{
    "idx": 0,
    "problem": "clinical note text...",  # input_key
    "answer": "CORRECT" or "INCORRECT",   # label_key
    "datasource": "medec",
    "level": 1-5 (based on error type difficulty)
}

Usage:
    python scripts/prepare_medec_for_openrlhf.py --output_dir SeRL/openrlhf/dataset/medec
    
    # With specific sample count
    python scripts/prepare_medec_for_openrlhf.py --output_dir SeRL/openrlhf/dataset/medec --max_samples 500
"""

import argparse
import json
import random
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_processor import MedicalDataProcessor


# Difficulty levels for error types (1=easiest, 5=hardest)
ERROR_TYPE_DIFFICULTY = {
    "Diagnosis": 4,
    "Management": 3,
    "Treatment": 3,
    "Pharmacotherapy": 4,
    "Causal Organism": 5,
    None: 2,  # Clean notes
}

# System prompt for medical error detection
SYSTEM_PROMPT = """You are a healthcare professional specializing in analyzing medical notes.

Important: Medical notes should be presumed CORRECT unless there is an obvious, significant error.

Your task is to identify only clear substitution errors in:
- Diagnostic terms that significantly change the clinical meaning
- Medication terms that would result in wrong treatment
- Treatment protocols that are clearly contraindicated
- Management plans that would harm the patient
- Causal organism identification that is clearly wrong

Classification criteria:
- INCORRECT: Contains a clinically significant error that would change patient care
- CORRECT: Default classification - use this unless there is a clear, significant error

Note: Accept all reasonable medical terminology variations. When in doubt, classify as CORRECT."""

USER_TEMPLATE = """Analyze this clinical note:

{note}

Provide brief reasoning in <think> tags, then your classification in <answer> tags.
Remember: Default to CORRECT unless you find a clear, significant clinical error."""


def format_prompt(note: str, use_chat_format: bool = False) -> str:
    """Format clinical note as prompt."""
    if use_chat_format:
        # Return as conversation for apply_chat_template
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(note=note)}
        ]
    else:
        # Return as plain text
        return USER_TEMPLATE.format(note=note)


def prepare_training_data(
    processor: MedicalDataProcessor,
    max_samples: int = -1,
    balance: bool = True,
    use_chat_format: bool = False,
) -> list[dict]:
    """
    Prepare training data from MEDEC processor.
    
    Args:
        processor: MedicalDataProcessor with loaded data
        max_samples: Maximum samples (-1 for all)
        balance: Whether to balance error/clean samples
        use_chat_format: Whether to format as chat messages
    
    Returns:
        List of formatted samples for OpenRLHF
    """
    samples = []
    idx = 0
    
    # Process error samples
    error_samples = processor.error_pool.copy()
    random.shuffle(error_samples)
    
    for entry in error_samples:
        note = entry.get('text', '')
        if not note.strip():
            continue
        
        error_type = entry.get('error_type')
        difficulty = ERROR_TYPE_DIFFICULTY.get(error_type, 3)
        
        sample = {
            "idx": idx,
            "problem": format_prompt(note, use_chat_format),
            "answer": "INCORRECT",
            "datasource": "medec",
            "level": difficulty,
            "error_type": error_type,
            "subset": entry.get('subset', 'unknown'),
        }
        samples.append(sample)
        idx += 1
    
    # Process clean samples
    clean_samples = processor.clean_pool.copy()
    random.shuffle(clean_samples)
    
    for entry in clean_samples:
        note = entry.get('text', '')
        if not note.strip():
            continue
        
        sample = {
            "idx": idx,
            "problem": format_prompt(note, use_chat_format),
            "answer": "CORRECT",
            "datasource": "medec",
            "level": 2,  # Clean notes are generally easier
            "error_type": None,
            "subset": entry.get('subset', 'unknown'),
        }
        samples.append(sample)
        idx += 1
    
    # Balance if requested
    if balance:
        error_count = sum(1 for s in samples if s['answer'] == 'INCORRECT')
        clean_count = len(samples) - error_count
        
        if error_count > clean_count:
            # Downsample errors
            errors = [s for s in samples if s['answer'] == 'INCORRECT']
            cleans = [s for s in samples if s['answer'] == 'CORRECT']
            random.shuffle(errors)
            samples = errors[:clean_count] + cleans
        elif clean_count > error_count:
            # Downsample cleans
            errors = [s for s in samples if s['answer'] == 'INCORRECT']
            cleans = [s for s in samples if s['answer'] == 'CORRECT']
            random.shuffle(cleans)
            samples = errors + cleans[:error_count]
    
    # Shuffle final dataset
    random.shuffle(samples)
    
    # Limit samples if requested
    if max_samples > 0 and len(samples) > max_samples:
        samples = samples[:max_samples]
    
    # Re-index
    for i, sample in enumerate(samples):
        sample['idx'] = i
    
    return samples


def prepare_seed_data(
    processor: MedicalDataProcessor,
    num_samples: int = 500,
) -> list[dict]:
    """
    Prepare seed data for self-instruction (smaller, balanced subset).
    
    Args:
        processor: MedicalDataProcessor with loaded data
        num_samples: Number of seed samples
    
    Returns:
        List of seed samples
    """
    return prepare_training_data(
        processor,
        max_samples=num_samples,
        balance=True,
        use_chat_format=False,
    )


def save_jsonl(samples: list[dict], output_path: Path) -> None:
    """Save samples to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        for sample in samples:
            # Convert chat format to string if needed
            if isinstance(sample.get('problem'), list):
                # Keep as-is for chat template processing
                pass
            f.write(json.dumps(sample) + '\n')
    
    print(f"Saved {len(samples)} samples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare MEDEC for OpenRLHF")
    parser.add_argument(
        "--medec_path",
        type=str,
        default="data_raw/MEDEC",
        help="Path to MEDEC dataset"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data_processed/medec",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        help="Maximum samples for training (-1 for all)"
    )
    parser.add_argument(
        "--seed_samples",
        type=int,
        default=500,
        help="Number of seed samples for self-instruction"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--no_balance",
        action="store_true",
        help="Don't balance error/clean samples"
    )
    
    args = parser.parse_args()
    random.seed(args.seed)
    
    output_dir = Path(args.output_dir)
    
    print("=" * 60)
    print("Preparing MEDEC for OpenRLHF")
    print("=" * 60)
    
    # Load training data
    print(f"\nLoading MEDEC from {args.medec_path}...")
    processor = MedicalDataProcessor.load_training_data(data_path=args.medec_path)
    
    print(f"Error pool: {len(processor.error_pool)} samples")
    print(f"Clean pool: {len(processor.clean_pool)} samples")
    
    # Prepare full training data
    print("\nPreparing training data...")
    train_samples = prepare_training_data(
        processor,
        max_samples=args.max_samples,
        balance=not args.no_balance,
    )
    
    error_count = sum(1 for s in train_samples if s['answer'] == 'INCORRECT')
    clean_count = len(train_samples) - error_count
    print(f"Training samples: {len(train_samples)} (errors={error_count}, clean={clean_count})")
    
    # Save training data
    train_path = output_dir / "train.jsonl"
    save_jsonl(train_samples, train_path)
    
    # Prepare seed data for self-instruction
    print("\nPreparing seed data for self-instruction...")
    seed_samples = prepare_seed_data(processor, num_samples=args.seed_samples)
    
    seed_error_count = sum(1 for s in seed_samples if s['answer'] == 'INCORRECT')
    seed_clean_count = len(seed_samples) - seed_error_count
    print(f"Seed samples: {len(seed_samples)} (errors={seed_error_count}, clean={seed_clean_count})")
    
    # Save seed data
    seed_path = output_dir / f"seed_{args.seed_samples}.jsonl"
    save_jsonl(seed_samples, seed_path)
    
    # Load and prepare test data
    print("\nPreparing test data...")
    test_processor = MedicalDataProcessor.load_test_data(data_path=args.medec_path)
    
    test_samples = prepare_training_data(
        test_processor,
        max_samples=-1,
        balance=False,  # Don't balance test data
    )
    
    test_error_count = sum(1 for s in test_samples if s['answer'] == 'INCORRECT')
    test_clean_count = len(test_samples) - test_error_count
    print(f"Test samples: {len(test_samples)} (errors={test_error_count}, clean={test_clean_count})")
    
    # Save test data
    test_path = output_dir / "test.jsonl"
    save_jsonl(test_samples, test_path)
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print("=" * 60)
    print(f"\nOutput files:")
    print(f"  Training: {train_path}")
    print(f"  Seed:     {seed_path}")
    print(f"  Test:     {test_path}")
    print(f"\nUse these with OpenRLHF:")
    print(f"  --prompt_data {train_path}")
    print(f"  --input_key problem")
    print(f"  --label_key answer")


if __name__ == "__main__":
    main()
