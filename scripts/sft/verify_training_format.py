#!/usr/bin/env python3
"""Verify that training data has consistent final_answer formatting."""

import json
import re
from pathlib import Path
from collections import Counter


def check_final_answer_format(jsonl_path: str):
    """Check consistency of final_answer formatting in training data."""
    
    issues = []
    formats = Counter()
    
    with open(jsonl_path, 'r') as f:
        for line_num, line in enumerate(f, 1):
            record = json.loads(line)
            
            # Check if this is an assessor task
            if record.get('role') != 'critic':
                continue
            
            reasoning = record.get('reasoning', '')
            label = record.get('label', '')
            
            # Extract final_answer from reasoning
            match = re.search(
                r'final_answer:\s*"?(CORRECT|INCORRECT)"?',
                reasoning,
                re.IGNORECASE
            )
            
            if match:
                extracted = match.group(1).upper()
                formats[f'final_answer: "{extracted}"'] += 1
                
                # Check consistency with label
                if extracted != label:
                    issues.append({
                        'line': line_num,
                        'note_id': record.get('note_id'),
                        'label': label,
                        'extracted': extracted,
                        'issue': 'Label mismatch'
                    })
            else:
                issues.append({
                    'line': line_num,
                    'note_id': record.get('note_id'),
                    'label': label,
                    'issue': 'Missing final_answer'
                })
    
    print(f"\n{'='*60}")
    print(f"FORMAT ANALYSIS: {jsonl_path}")
    print(f"{'='*60}\n")
    
    print("Final Answer Formats Found:")
    for fmt, count in formats.most_common():
        print(f"  {fmt}: {count}")
    
    if issues:
        print(f"\n⚠️  Found {len(issues)} issues:")
        for issue in issues[:10]:  # Show first 10
            print(f"  Line {issue['line']}: {issue['note_id']} - {issue['issue']}")
            if 'extracted' in issue:
                print(f"    Label: {issue['label']}, Extracted: {issue['extracted']}")
    else:
        print("\n✅ No formatting issues found!")
    
    print(f"\n{'='*60}\n")


if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage: python verify_training_format.py <path_to_jsonl>")
        sys.exit(1)
    
    check_final_answer_format(sys.argv[1])
