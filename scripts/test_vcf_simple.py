#!/usr/bin/env python3
"""
Simple VCF test without torch dependency.
Tests just the Jaccard similarity and filtering logic.
"""

import re
from typing import List

def tokenize_for_jaccard(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())

def jaccard_similarity(text1: str, text2: str) -> float:
    set1 = set(tokenize_for_jaccard(text1))
    set2 = set(tokenize_for_jaccard(text2))
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)

def word_counts(text: str):
    counts = {}
    for token in tokenize_for_jaccard(text):
        counts[token] = counts.get(token, 0) + 1
    return counts

def has_word_change(original: str, generated: str) -> bool:
    return word_counts(original) != word_counts(generated)

# Test cases
print("=" * 60)
print("VCF Core Logic Tests (No torch required)")
print("=" * 60)

# Test 1: Single word change
original = "Patient has type 2 diabetes controlled with metformin."
modified = "Patient has type 1 diabetes controlled with metformin."
jac = jaccard_similarity(original, modified)
has_change = has_word_change(original, modified)

print(f"\nTest 1: Single word change")
print(f"  Original: {original}")
print(f"  Modified: {modified}")
print(f"  Jaccard: {jac:.3f} (expected: ~0.91)")
print(f"  Has change: {has_change} (expected: True)")
print(f"  Would pass VCF: {0.85 <= jac <= 0.99 and has_change}")

# Test 2: Too many changes
modified2 = "Patient diagnosed with severe hypertension requiring multiple medications."
jac2 = jaccard_similarity(original, modified2)
has_change2 = has_word_change(original, modified2)

print(f"\nTest 2: Too many changes")
print(f"  Jaccard: {jac2:.3f} (expected: <0.85)")
print(f"  Has change: {has_change2}")
print(f"  Would pass VCF: {0.85 <= jac2 <= 0.99 and has_change2}")

# Test 3: No change
jac3 = jaccard_similarity(original, original)
has_change3 = has_word_change(original, original)

print(f"\nTest 3: No change")
print(f"  Jaccard: {jac3:.3f} (expected: 1.00)")
print(f"  Has change: {has_change3} (expected: False)")
print(f"  Would pass VCF: {0.85 <= jac3 <= 0.99 and has_change3}")

print("\n" + "=" * 60)
print("✅ Core VCF logic tests complete!")
print("=" * 60)
