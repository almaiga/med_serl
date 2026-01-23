#!/usr/bin/env python3
"""
Verify the generated parquet files are in the correct format for verl.
"""

import pandas as pd
import json
from pathlib import Path


def verify_parquet(file_path: str):
    """Verify parquet file structure matches verl requirements."""
    print(f"\n{'='*80}")
    print(f"Verifying: {file_path}")
    print('='*80)
    
    df = pd.read_parquet(file_path)
    
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    
    # Check required fields
    required_fields = ['data_source', 'prompt', 'ability', 'reward_model', 'interaction_kwargs']
    missing_fields = [f for f in required_fields if f not in df.columns]
    
    if missing_fields:
        print(f"\n❌ Missing required fields: {missing_fields}")
        return False
    else:
        print(f"\n✓ All required fields present")
    
    # Check first example
    first_example = df.iloc[0]
    
    print(f"\n--- Sample Example ---")
    print(f"Data source: {first_example['data_source']}")
    print(f"Ability: {first_example['ability']}")
    print(f"Ground truth: {first_example['reward_model']['ground_truth']}")
    print(f"Interaction name: {first_example['interaction_kwargs']['name']}")
    print(f"Mode: {first_example['interaction_kwargs']['mode']}")
    
    # Verify prompt structure
    prompt = first_example['prompt']
    print(f"\nPrompt structure:")
    print(f"  - Type: {type(prompt)}")
    print(f"  - Length: {len(prompt)} messages")
    if prompt:
        print(f"  - First message role: {prompt[0]['role']}")
        print(f"  - First message preview: {prompt[0]['content'][:100]}...")
    
    # Check balance of CORRECT vs INCORRECT
    ground_truths = [row['reward_model']['ground_truth'] for _, row in df.iterrows()]
    correct_count = ground_truths.count('CORRECT')
    incorrect_count = ground_truths.count('INCORRECT')
    
    print(f"\n--- Label Distribution ---")
    print(f"CORRECT: {correct_count} ({correct_count/len(ground_truths)*100:.1f}%)")
    print(f"INCORRECT: {incorrect_count} ({incorrect_count/len(ground_truths)*100:.1f}%)")
    
    if abs(correct_count - incorrect_count) > len(ground_truths) * 0.1:
        print(f"⚠️  Warning: Labels are imbalanced")
    else:
        print(f"✓ Labels are balanced")
    
    # Check interaction_kwargs structure
    sample_kwargs = first_example['interaction_kwargs']
    print(f"\n--- Interaction kwargs ---")
    print(f"Required fields present:")
    print(f"  - name: {'✓' if 'name' in sample_kwargs else '❌'}")
    print(f"  - ground_truth: {'✓' if 'ground_truth' in sample_kwargs else '❌'}")
    print(f"  - mode: {'✓' if 'mode' in sample_kwargs else '❌'}")
    print(f"  - note_data: {'✓' if 'note_data' in sample_kwargs else '❌'}")
    
    return True


def main():
    base_dir = Path("data_processed/selfplay")
    
    train_file = base_dir / "train.parquet"
    test_file = base_dir / "test.parquet"
    
    if not train_file.exists():
        print(f"❌ Train file not found: {train_file}")
        return
    
    if not test_file.exists():
        print(f"❌ Test file not found: {test_file}")
        return
    
    print("\n" + "="*80)
    print("VERIFYING SELF-PLAY PARQUET FILES")
    print("="*80)
    
    train_ok = verify_parquet(str(train_file))
    test_ok = verify_parquet(str(test_file))
    
    if train_ok and test_ok:
        print("\n" + "="*80)
        print("✓ ALL VERIFICATIONS PASSED")
        print("="*80)
        print("\nData is ready for verl training!")
        print(f"\nTo start training, run:")
        print(f"  bash verl_implementation/scripts/run_training.sh")
    else:
        print("\n❌ VERIFICATION FAILED")


if __name__ == "__main__":
    main()
