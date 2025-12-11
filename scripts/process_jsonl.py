#!/usr/bin/env python3
"""
Process JSONL file with unquoted NaN values.
Converts unquoted NaN to null (JSON-compliant) and validates output.
"""

import re
import json
import sys
from pathlib import Path

def fix_nan_values(line: str) -> str:
    """Replace unquoted NaN with null for JSON compliance."""
    # Match NaN that's a value (after : and before , or })
    # This handles: "key": NaN, or "key": NaN}
    return re.sub(r':\s*NaN\s*([,}])', r': null\1', line)

def process_jsonl(input_path: str, output_path: str = None):
    """Process JSONL file, fixing NaN values and validating JSON."""
    input_file = Path(input_path)
    
    if output_path is None:
        output_path = input_file.parent / f"{input_file.stem}_cleaned.jsonl"
    
    output_file = Path(output_path)
    
    total_lines = 0
    fixed_lines = 0
    valid_lines = 0
    errors = []
    
    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:
        
        for line_num, line in enumerate(infile, 1):
            total_lines += 1
            original_line = line
            
            # Fix NaN values
            fixed_line = fix_nan_values(line)
            if fixed_line != original_line:
                fixed_lines += 1
            
            # Validate JSON
            try:
                data = json.loads(fixed_line)
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')
                valid_lines += 1
            except json.JSONDecodeError as e:
                errors.append((line_num, str(e), line[:100]))
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Processing Complete!")
    print(f"{'='*50}")
    print(f"Input file:  {input_file}")
    print(f"Output file: {output_file}")
    print(f"{'='*50}")
    print(f"Total lines:     {total_lines}")
    print(f"Lines with NaN:  {fixed_lines}")
    print(f"Valid JSON:      {valid_lines}")
    print(f"Errors:          {len(errors)}")
    
    if errors:
        print(f"\nFirst 5 errors:")
        for line_num, err, preview in errors[:5]:
            print(f"  Line {line_num}: {err}")
            print(f"    Preview: {preview}...")
    
    return output_file, valid_lines, errors

def verify_output(output_path: str, sample_size: int = 5):
    """Verify the output file by parsing a few records."""
    print(f"\n{'='*50}")
    print(f"Verification - First {sample_size} records:")
    print(f"{'='*50}")
    
    with open(output_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            data = json.loads(line)
            print(f"\nRecord {i+1}:")
            print(f"  text_id: {data.get('text_id')}")
            print(f"  error_type: {data.get('error_type')} (type: {type(data.get('error_type')).__name__})")
            print(f"  predicted_label: {data.get('predicted_label')}")
            print(f"  correct: {data.get('correct')}")

if __name__ == "__main__":
    input_file = "results/inference/Final MS Results Dec 10 2025 (1).jsonl"
    
    # Allow override from command line
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    
    output_file, valid_count, errors = process_jsonl(input_file)
    
    if valid_count > 0:
        verify_output(output_file)
