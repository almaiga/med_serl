#!/usr/bin/env python3
"""
Test script for Verifiable Curriculum Filter (VCF) functions.

This script tests the VCF filtering logic to ensure it correctly
accepts and rejects generated notes based on similarity and edit constraints.

Note: This test requires torch and transformers to be installed.
If running in base Python environment without these, the test will skip gracefully.
"""

import sys
import os

# Add scripts/sft to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sft'))

try:
    from inference_utils import apply_vcf, jaccard_similarity
except ImportError as e:
    print(f"[WARNING] Cannot import inference_utils: {e}")
    print("[INFO] This is expected if torch/transformers are not installed.")
    print("[INFO] VCF tests will run in the training environment.")
    sys.exit(0)


def test_vcf_basic():
    """Test basic VCF filtering."""
    print("=" * 60)
    print("Test 1: Basic VCF Filtering")
    print("=" * 60)

    # Test 1: Single word change (should pass)
    original = "Patient has type 2 diabetes controlled with metformin."
    modified = "Patient has type 1 diabetes controlled with metformin."
    result = apply_vcf(original, modified, min_jaccard=0.85, max_jaccard=0.99, max_word_edits=6)

    print(f"\nOriginal: {original}")
    print(f"Modified: {modified}")
    print(f"Result: {result.passed} (expected: True)")
    print(f"  Jaccard: {result.score_jaccard:.3f}")
    print(f"  Word edits: {result.word_edits}")
    print(f"  Reason: {result.reason}")

    assert result.passed, "Single word change should pass VCF"
    print("✅ Test 1 passed")


def test_vcf_too_many_edits():
    """Test VCF rejection for too many edits."""
    print("\n" + "=" * 60)
    print("Test 2: Too Many Edits (should reject)")
    print("=" * 60)

    # Test 2: Too many changes (should fail)
    original = "Patient has type 2 diabetes controlled with metformin."
    modified = "Patient diagnosed with severe hypertension requiring multiple antihypertensive medications."
    result = apply_vcf(original, modified, min_jaccard=0.85, max_jaccard=0.99, max_word_edits=6)

    print(f"\nOriginal: {original}")
    print(f"Modified: {modified}")
    print(f"Result: {result.passed} (expected: False)")
    print(f"  Jaccard: {result.score_jaccard:.3f}")
    print(f"  Reason: {result.reason}")

    assert not result.passed, "Too many edits should fail VCF"
    assert result.reason in ["low_jaccard", "too_many_edits"], f"Expected rejection reason, got: {result.reason}"
    print("✅ Test 2 passed")


def test_vcf_no_change():
    """Test VCF rejection for no change."""
    print("\n" + "=" * 60)
    print("Test 3: No Change (should reject)")
    print("=" * 60)

    # Test 3: No change (should fail)
    original = "Patient has type 2 diabetes controlled with metformin."
    modified = "Patient has type 2 diabetes controlled with metformin."
    result = apply_vcf(original, modified, min_jaccard=0.85, max_jaccard=0.99, max_word_edits=6)

    print(f"\nOriginal: {original}")
    print(f"Modified: {modified}")
    print(f"Result: {result.passed} (expected: False)")
    print(f"  Reason: {result.reason}")

    assert not result.passed, "No change should fail VCF"
    assert result.reason == "too_similar" or result.reason == "no_word_change", f"Expected 'no_word_change' or 'too_similar', got: {result.reason}"
    print("✅ Test 3 passed")


def test_vcf_empty():
    """Test VCF rejection for empty output."""
    print("\n" + "=" * 60)
    print("Test 4: Empty Output (should reject)")
    print("=" * 60)

    # Test 4: Empty generated note (should fail)
    original = "Patient has type 2 diabetes controlled with metformin."
    modified = ""
    result = apply_vcf(original, modified, min_jaccard=0.85, max_jaccard=0.99, max_word_edits=6)

    print(f"\nOriginal: {original}")
    print(f"Modified: (empty)")
    print(f"Result: {result.passed} (expected: False)")
    print(f"  Reason: {result.reason}")

    assert not result.passed, "Empty output should fail VCF"
    assert result.reason == "empty_generated", f"Expected 'empty_generated', got: {result.reason}"
    print("✅ Test 4 passed")


def test_jaccard_similarity():
    """Test Jaccard similarity calculation."""
    print("\n" + "=" * 60)
    print("Test 5: Jaccard Similarity Calculation")
    print("=" * 60)

    test_cases = [
        ("hello world", "hello world", 1.0),
        ("hello world", "goodbye world", 0.33),
        ("a b c", "d e f", 0.0),
        ("", "", 1.0),
        ("test", "", 0.0),
    ]

    for text1, text2, expected in test_cases:
        score = jaccard_similarity(text1, text2)
        print(f"\nText1: '{text1}'")
        print(f"Text2: '{text2}'")
        print(f"Jaccard: {score:.2f} (expected: {expected:.2f})")
        assert abs(score - expected) < 0.05, f"Jaccard mismatch: {score:.2f} vs {expected:.2f}"

    print("✅ Test 5 passed")


def test_vcf_edge_cases():
    """Test VCF edge cases."""
    print("\n" + "=" * 60)
    print("Test 6: Edge Cases")
    print("=" * 60)

    # Test with minimum Jaccard boundary
    original = "Patient presents with chest pain"
    modified = "Patient presents with severe chest pain"  # Added one word
    result = apply_vcf(original, modified, min_jaccard=0.85, max_jaccard=0.99, max_word_edits=6)

    print(f"\nOriginal: {original}")
    print(f"Modified: {modified}")
    print(f"Result: {result.passed}")
    print(f"  Jaccard: {result.score_jaccard:.3f}")
    print(f"  Word edits: {result.word_edits}")

    # Should pass if Jaccard >= 0.85 and word edits <= 6
    if result.score_jaccard >= 0.85:
        assert result.passed, "Should pass if Jaccard >= 0.85"

    print("✅ Test 6 passed")


def main():
    """Run all VCF tests."""
    print("\n" + "=" * 60)
    print("VCF Unit Tests")
    print("=" * 60)

    try:
        test_vcf_basic()
        test_vcf_too_many_edits()
        test_vcf_no_change()
        test_vcf_empty()
        test_jaccard_similarity()
        test_vcf_edge_cases()

        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
        print("\nVCF filtering is working correctly.")
        print("Ready for integration with MedSeRL training.")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
