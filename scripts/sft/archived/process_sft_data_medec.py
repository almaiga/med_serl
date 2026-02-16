import pandas as pd
import os
from pathlib import Path
from sklearn.model_selection import train_test_split

def load_and_format_medec_pairs(data_path, split_name):
    """
    Load MEDEC dataset and create paired examples (incorrect + correct) for CoT augmentation.
    Only keeps rows where Error Flag == 1 (rows with errors that have corrections).
    
    Args:
        data_path: Path to the MEDEC CSV file
        split_name: Name of the split (e.g., 'train', 'validation')
    
    Returns:
        pd.DataFrame: Formatted dataset with note pairs
    """
    # Load the data
    df = pd.read_csv(data_path)
    
    print("="*80)
    print(f"MEDEC DATA LOADING - {split_name.upper()}")
    print("="*80)
    print(f"\nOriginal columns: {df.columns.tolist()}")
    print(f"Shape: {df.shape}")
    print(f"\nError Flag distribution:")
    print(df['Error Flag'].value_counts())
    
    # Filter to only rows with errors (Error Flag == 1)
    error_df = df[df['Error Flag'] == 1].copy()
    
    print(f"\n✓ Filtered to {len(error_df)} rows with errors (Error Flag == 1)")
    print(f"\nError Type distribution in filtered data:")
    print(error_df['Error Type'].value_counts())
    
    # Create formatted dataset with pairs
    formatted_df = pd.DataFrame()
    
    # Add metadata
    formatted_df['split'] = split_name
    formatted_df['note_id'] = error_df['Text ID'].values
    
    # Add the INCORRECT note (contains the error)
    formatted_df['incorrect_note'] = error_df['Text'].values
    
    # Add the CORRECT note (error has been fixed)
    formatted_df['correct_note'] = error_df['Corrected Text'].values
    
    # Add error metadata - normalize error type casing
    formatted_df['error_type'] = error_df['Error Type'].str.lower().values
    formatted_df['error_sentence'] = error_df['Error Sentence'].values
    formatted_df['corrected_sentence'] = error_df['Corrected Sentence'].values
    
    print("\n" + "="*80)
    print(f"FORMATTED PAIRED DATA - {split_name.upper()}")
    print("="*80)
    print(f"\nNew columns: {formatted_df.columns.tolist()}")
    print(f"Total pairs: {len(formatted_df)}")
    print(f"\nError type distribution:")
    print(formatted_df['error_type'].value_counts())
    
    # Show sample pair
    print("\n" + "="*80)
    print(f"SAMPLE NOTE PAIR - {split_name.upper()}")
    print("="*80)
    sample = formatted_df.iloc[0]
    print(f"\nNote ID: {sample['note_id']}")
    print(f"Error Type: {sample['error_type']}")
    
    print(f"\n--- INCORRECT NOTE (contains error) ---")
    print(f"{sample['incorrect_note'][:300]}...")
    
    print(f"\n--- ERROR SENTENCE ---")
    print(f"{sample['error_sentence']}")
    
    print(f"\n--- CORRECTED SENTENCE ---")
    print(f"{sample['corrected_sentence']}")
    
    print(f"\n--- CORRECT NOTE (error fixed) ---")
    print(f"{sample['correct_note'][:300]}...")
    
    return formatted_df


def save_formatted_pairs(df, output_dir, split_name):
    """Save formatted paired data to output directory."""
    split_dir = os.path.join(output_dir, split_name)
    os.makedirs(split_dir, exist_ok=True)
    
    # Save a JSONL version for LLM processing
    json_path = os.path.join(split_dir, f'medec_pairs_{split_name}.jsonl')
    df.to_json(json_path, orient='records', lines=True)
    print(f"✓ Saved {split_name} paired dataset ({len(df)} pairs) to: {json_path}")


def create_sft_rl_split(df, sft_ratio=0.75, random_state=42):
    """
    Split data into SFT (75%) and RL (25%) sets, stratified by error type.
    
    Args:
        df: DataFrame with note pairs
        sft_ratio: Proportion for SFT (default 0.75)
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (sft_df, rl_df)
    """
    print("\n" + "="*80)
    print("CREATING SFT/RL SPLIT (STRATIFIED BY ERROR TYPE)")
    print("="*80)
    
    # Stratified split by error type
    sft_df, rl_df = train_test_split(
        df,
        test_size=(1 - sft_ratio),
        stratify=df['error_type'],
        random_state=random_state
    )
    
    print(f"\n✓ Split completed:")
    print(f"  - SFT set: {len(sft_df)} pairs ({len(sft_df)/len(df)*100:.1f}%)")
    print(f"  - RL set:  {len(rl_df)} pairs ({len(rl_df)/len(df)*100:.1f}%)")
    
    # Verify stratification
    print(f"\n{'='*80}")
    print("ERROR TYPE DISTRIBUTION VERIFICATION")
    print("="*80)
    print(f"\n{'Error Type':<30} {'Original %':<15} {'SFT %':<15} {'RL %':<15}")
    print("-" * 80)
    
    for error_type in sorted(df['error_type'].unique()):
        orig_pct = (df['error_type'] == error_type).sum() / len(df) * 100
        sft_pct = (sft_df['error_type'] == error_type).sum() / len(sft_df) * 100
        rl_pct = (rl_df['error_type'] == error_type).sum() / len(rl_df) * 100
        print(f"{error_type:<30} {orig_pct:<15.2f} {sft_pct:<15.2f} {rl_pct:<15.2f}")
    
    return sft_df, rl_df


if __name__ == "__main__":
    # Paths
    base_data_dir = '/Users/josmaiga/Documents/GitHub/med_serl/data_raw/MEDEC'
    output_dir = '/Users/josmaiga/Documents/GitHub/med_serl/data_processed/medec_paired'
    
    # Define datasets to process - ONLY TRAIN AND VALIDATION (exclude test)
    # MEDEC-MS has train + validation
    # MEDEC-UW has validation only (no train)
    datasets = {
        'medec_ms': {
            'base_dir': os.path.join(base_data_dir, 'MEDEC-MS'),
            'splits': {
                'train': 'MEDEC-Full-TrainingSet-with-ErrorType.csv',
                'validation': 'MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv',
                # 'test': 'MEDEC-MS-TestSet-with-GroundTruth-and-ErrorType.csv'  # EXCLUDED
            }
        },
        'medec_uw': {
            'base_dir': os.path.join(base_data_dir, 'MEDEC-UW'),
            'splits': {
                'validation': 'MEDEC-UW-ValidationSet-with-GroundTruth-and-ErrorType.csv',
                # 'test': 'MEDEC-UW-TestSet-with-GroundTruth-and-ErrorType.csv'  # EXCLUDED
            }
        }
    }
    
    all_pairs = []
    
    # Process each dataset
    for dataset_name, dataset_info in datasets.items():
        print("\n" + "="*80)
        print(f"PROCESSING {dataset_name.upper()} DATASET")
        print("="*80)
        
        for split_name, filename in dataset_info['splits'].items():
            data_path = os.path.join(dataset_info['base_dir'], filename)
            
            if not os.path.exists(data_path):
                print(f"\n⚠️  WARNING: {split_name} file not found at {data_path}")
                continue
            
            print("\n" + "="*80)
            print(f"PROCESSING {dataset_name.upper()} - {split_name.upper()} SPLIT")
            print("="*80)
            
            # Load and format pairs with dataset prefix
            split_label = f"{dataset_name}_{split_name}"
            formatted_df = load_and_format_medec_pairs(data_path, split_label)
            
            # Save individual split
            dataset_output_dir = os.path.join(output_dir, dataset_name)
            save_formatted_pairs(formatted_df, dataset_output_dir, split_name)
            
            all_pairs.append(formatted_df)
    
    # Combine all pairs and create SFT/RL split
    if all_pairs:
        combined_df = pd.concat(all_pairs, ignore_index=True)
        
        print("\n" + "="*80)
        print("COMBINED DATASET STATISTICS")
        print("="*80)
        print(f"\nTotal pairs (TRAIN + VALIDATION): {len(combined_df)}")
        
        # Show breakdown by dataset and split
        print("\nBreakdown by source:")
        split_counts = combined_df['split'].value_counts().sort_index()
        for split, count in split_counts.items():
            print(f"  - {split}: {count}")
        
        print("\nError type distribution:")
        error_counts = combined_df['error_type'].value_counts()
        for error_type, count in error_counts.items():
            print(f"  - {error_type}: {count}")
        
        # Create SFT/RL split (75/25, stratified by error type)
        sft_df, rl_df = create_sft_rl_split(combined_df, sft_ratio=0.75, random_state=42)
        
        # Save SFT and RL splits to a clean directory
        final_output_dir = os.path.join(output_dir, 'train_val_split')
        os.makedirs(final_output_dir, exist_ok=True)
        
        sft_path = os.path.join(final_output_dir, 'sft_train.jsonl')
        rl_path = os.path.join(final_output_dir, 'rl_train.jsonl')
        
        sft_df.to_json(sft_path, orient='records', lines=True)
        rl_df.to_json(rl_path, orient='records', lines=True)
        
        print("\n" + "="*80)
        print("✓ ALL DATA PROCESSING COMPLETE!")
        print("="*80)
        print(f"\nFinal datasets saved to: {final_output_dir}")
        print(f"\n✓ SFT training set: {sft_path}")
        print(f"  - {len(sft_df)} pairs (75%)")
        print(f"\n✓ RL training set: {rl_path}")
        print(f"  - {len(rl_df)} pairs (25%)")
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print("\nData sources:")
        print("  ✓ MEDEC-MS train")
        print("  ✓ MEDEC-MS validation")
        print("  ✓ MEDEC-UW validation")
        print("\nExcluded:")
        print("  ✗ MEDEC-MS test (held out)")
        print("  ✗ MEDEC-UW test (held out)")
        
        print("\nSplit strategy:")
        print("  - 75% for SFT (supervised fine-tuning)")
        print("  - 25% for RL (reinforcement learning)")
        print("  - Stratified by error type (maintains error distribution)")
        
        print("\nEach example contains:")
        print("  - split: source dataset + original split")
        print("  - note_id: unique identifier")
        print("  - incorrect_note: note with medical error")
        print("  - correct_note: note with error fixed")
        print("  - error_type: category of medical error")
        print("  - error_sentence: problematic sentence")
        print("  - corrected_sentence: fixed sentence")
        print("="*80 + "\n")