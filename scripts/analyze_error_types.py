#!/usr/bin/env python3
"""
Analyze error types in UW and MS datasets to compare patterns.
"""

import pandas as pd
from collections import Counter, defaultdict
import sys
from pathlib import Path

def analyze_dataset(csv_path, dataset_name):
    """Analyze error types in a dataset."""
    print(f"\n=== {dataset_name} Dataset Analysis ===")
    
    try:
        df = pd.read_csv(csv_path)
        print(f"Total samples: {len(df)}")
        
        # Count error flags
        error_counts = df['Error Flag'].value_counts()
        print(f"Clean samples (Error Flag = 0): {error_counts.get(0, 0)}")
        print(f"Error samples (Error Flag = 1): {error_counts.get(1, 0)}")
        
        # Analyze error types
        error_samples = df[df['Error Flag'] == 1]
        if len(error_samples) > 0:
            error_types = error_samples['Error Type'].value_counts()
            print(f"\nError Type Distribution:")
            for error_type, count in error_types.items():
                print(f"  {error_type}: {count}")
            
            # Show some examples of each error type
            print(f"\nError Examples:")
            for error_type in error_types.index[:5]:  # Top 5 error types
                examples = error_samples[error_samples['Error Type'] == error_type]
                print(f"\n{error_type} examples:")
                for i, (_, row) in enumerate(examples.head(2).iterrows()):
                    print(f"  {i+1}. Original: {row['Error Sentence'][:100]}...")
                    print(f"     Corrected: {row['Corrected Sentence'][:100]}...")
        
        return {
            'total': len(df),
            'clean': error_counts.get(0, 0),
            'errors': error_counts.get(1, 0),
            'error_types': dict(error_samples['Error Type'].value_counts()) if len(error_samples) > 0 else {}
        }
        
    except Exception as e:
        print(f"Error reading {csv_path}: {e}")
        return None

def main():
    # Paths to datasets
    datasets = {
        'UW Test': 'data_raw/MEDEC/MEDEC-UW/MEDEC-UW-TestSet-with-GroundTruth-and-ErrorType.csv',
        'UW Validation': 'data_raw/MEDEC/MEDEC-UW/MEDEC-UW-ValidationSet-with-GroundTruth-and-ErrorType.csv',
        'MS Test': 'data_raw/MEDEC/MEDEC-MS/MEDEC-MS-TestSet-with-GroundTruth-and-ErrorType.csv',
        'MS Validation': 'data_raw/MEDEC/MEDEC-MS/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv'
    }
    
    results = {}
    
    # Analyze each dataset
    for name, path in datasets.items():
        if Path(path).exists():
            results[name] = analyze_dataset(path, name)
        else:
            print(f"\nDataset not found: {path}")
    
    # Compare UW vs MS
    print(f"\n" + "="*60)
    print("COMPARISON: UW vs MS Datasets")
    print("="*60)
    
    # Combine UW datasets
    uw_stats = {'total': 0, 'clean': 0, 'errors': 0, 'error_types': Counter()}
    for name in ['UW Test', 'UW Validation']:
        if name in results and results[name]:
            uw_stats['total'] += results[name]['total']
            uw_stats['clean'] += results[name]['clean']
            uw_stats['errors'] += results[name]['errors']
            uw_stats['error_types'].update(results[name]['error_types'])
    
    # Combine MS datasets
    ms_stats = {'total': 0, 'clean': 0, 'errors': 0, 'error_types': Counter()}
    for name in ['MS Test', 'MS Validation']:
        if name in results and results[name]:
            ms_stats['total'] += results[name]['total']
            ms_stats['clean'] += results[name]['clean']
            ms_stats['errors'] += results[name]['errors']
            ms_stats['error_types'].update(results[name]['error_types'])
    
    print(f"\nUW Combined Stats:")
    print(f"  Total: {uw_stats['total']}")
    print(f"  Clean: {uw_stats['clean']} ({uw_stats['clean']/uw_stats['total']*100:.1f}%)")
    print(f"  Errors: {uw_stats['errors']} ({uw_stats['errors']/uw_stats['total']*100:.1f}%)")
    
    print(f"\nMS Combined Stats:")
    print(f"  Total: {ms_stats['total']}")
    print(f"  Clean: {ms_stats['clean']} ({ms_stats['clean']/ms_stats['total']*100:.1f}%)")
    print(f"  Errors: {ms_stats['errors']} ({ms_stats['errors']/ms_stats['total']*100:.1f}%)")
    
    print(f"\nError Type Comparison:")
    all_error_types = set(uw_stats['error_types'].keys()) | set(ms_stats['error_types'].keys())
    
    print(f"{'Error Type':<20} {'UW Count':<10} {'MS Count':<10} {'UW %':<8} {'MS %':<8}")
    print("-" * 60)
    
    for error_type in sorted(all_error_types):
        uw_count = uw_stats['error_types'].get(error_type, 0)
        ms_count = ms_stats['error_types'].get(error_type, 0)
        uw_pct = uw_count / uw_stats['errors'] * 100 if uw_stats['errors'] > 0 else 0
        ms_pct = ms_count / ms_stats['errors'] * 100 if ms_stats['errors'] > 0 else 0
        
        print(f"{error_type:<20} {uw_count:<10} {ms_count:<10} {uw_pct:<8.1f} {ms_pct:<8.1f}")

if __name__ == "__main__":
    main()