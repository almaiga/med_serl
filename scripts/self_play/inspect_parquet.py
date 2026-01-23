#!/usr/bin/env python3
"""Inspect parquet data for self-play training.

Usage:
    python scripts/self_play/inspect_parquet.py [parquet_file]
"""

import json
import sys
from pathlib import Path

import pyarrow.parquet as pq


def main():
    # Find parquet file
    if len(sys.argv) > 1:
        parquet_path = Path(sys.argv[1])
    else:
        parquet_path = Path("data_processed/self_play/train.parquet")
    
    print(f"Inspecting: {parquet_path}")
    print()
    
    # Read parquet
    table = pq.read_table(parquet_path)
    df = table.to_pandas()
    
    print("=" * 60)
    print("SCHEMA")
    print("=" * 60)
    print(table.schema)
    print()
    
    print("=" * 60)
    print(f"TOTAL ROWS: {len(df)}")
    print("=" * 60)
    print()
    
    # Check column types
    print("=" * 60)
    print("COLUMNS")
    print("=" * 60)
    for col in df.columns:
        sample = df[col].iloc[0]
        print(f"  {col}: {type(sample).__name__}")
    print()
    
    # Show first benign and first error example
    for mode_filter in ["benign", "error"]:
        print("=" * 60)
        print(f"SAMPLE: {mode_filter.upper()} MODE")
        print("=" * 60)
        
        # Find matching row
        row = None
        for i, r in df.iterrows():
            extra = r.get("extra_info", {})
            if isinstance(extra, dict) and mode_filter in extra.get("mode", ""):
                row = r
                break
            elif isinstance(extra, str):
                try:
                    extra_dict = json.loads(extra)
                    if mode_filter in extra_dict.get("mode", ""):
                        row = r
                        break
                except:
                    pass
        
        if row is None:
            # Just use first row
            row = df.iloc[0 if mode_filter == "benign" else min(1, len(df)-1)]
        
        # Print each field
        for col in df.columns:
            val = row[col]
            print(f"\n--- {col} ---")
            
            if col == "prompt":
                # Parse and pretty print prompt
                if isinstance(val, str):
                    try:
                        prompt_data = json.loads(val)
                        for msg in prompt_data:
                            role = msg.get("role", "?")
                            content = msg.get("content", "")
                            print(f"[{role}]:")
                            # Show first 500 chars of content
                            print(content[:500] + ("..." if len(content) > 500 else ""))
                    except:
                        print(val[:500])
                else:
                    print(val)
            
            elif col == "reward_model":
                if isinstance(val, dict):
                    print(json.dumps(val, indent=2))
                else:
                    print(val)
            
            elif col == "extra_info":
                if isinstance(val, dict):
                    for k, v in val.items():
                        if isinstance(v, str) and len(v) > 200:
                            print(f"  {k}: {v[:200]}...")
                        else:
                            print(f"  {k}: {v}")
                else:
                    print(val)
            
            else:
                print(val)
        
        print()
    
    # Check ground_truth distribution
    print("=" * 60)
    print("GROUND TRUTH DISTRIBUTION")
    print("=" * 60)
    gt_counts = {}
    for i, row in df.iterrows():
        rm = row.get("reward_model", {})
        if isinstance(rm, dict):
            gt = rm.get("ground_truth", "N/A")
        else:
            gt = "N/A"
        gt_counts[gt] = gt_counts.get(gt, 0) + 1
    
    for gt, count in sorted(gt_counts.items()):
        print(f"  {gt}: {count}")
    
    # Check mode distribution
    print()
    print("=" * 60)
    print("MODE DISTRIBUTION")
    print("=" * 60)
    mode_counts = {}
    for i, row in df.iterrows():
        extra = row.get("extra_info", {})
        if isinstance(extra, dict):
            mode = extra.get("mode", "N/A")
        elif isinstance(extra, str):
            try:
                mode = json.loads(extra).get("mode", "N/A")
            except:
                mode = "N/A"
        else:
            mode = "N/A"
        mode_counts[mode] = mode_counts.get(mode, 0) + 1
    
    for mode, count in sorted(mode_counts.items()):
        print(f"  {mode}: {count}")


if __name__ == "__main__":
    main()
