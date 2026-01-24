"""Test note extraction for Hidden CoT compliance.

Verifies that the Assessor only sees the clinical note,
with no leaked information about CORRECT/INCORRECT answer.

Run: python scripts/self_play/test_note_extraction.py
"""

import sys
sys.path.insert(0, ".")

from scripts.self_play.cot_parser import (
    parse_injector_output, 
    extract_note_for_assessor,
    extract_thinking,
)
from scripts.self_play.interactions.medical_game_interaction import MedicalGameInteraction


def test_basic_extraction():
    """Test basic note extraction strips answer and metadata."""
    
    injector_output = """<think>
I need to make a small change to keep the note correct.
I'll change "started" to "initiated" - a synonym that preserves meaning.
</think>

generated_note:
A 45-year-old male presented with chest pain. Treatment was initiated with aspirin.
Blood pressure was elevated at 150/95 mmHg. Patient was admitted for observation.

final_answer: "CORRECT"

changes_made:
{"original_sentence": "Treatment was started with aspirin.", "modified_sentence": "Treatment was initiated with aspirin.", "words_changed": "started → initiated"}"""
    
    # Test extraction
    note = extract_note_for_assessor(injector_output)
    
    # Verify no leaked info
    assert "<think>" not in note, "CoT thinking leaked to assessor!"
    assert "final_answer" not in note.lower(), "final_answer leaked to assessor!"
    assert "CORRECT" not in note, "Answer CORRECT leaked to assessor!"
    assert "changes_made" not in note.lower(), "changes_made leaked to assessor!"
    assert "words_changed" not in note.lower(), "words_changed leaked to assessor!"
    assert "original_sentence" not in note.lower(), "original_sentence leaked to assessor!"
    
    # Verify note content is preserved
    assert "45-year-old male" in note, "Note content missing!"
    assert "chest pain" in note, "Note content missing!"
    assert "aspirin" in note, "Note content missing!"
    
    print("✓ Basic extraction test passed")
    print(f"  Extracted note ({len(note)} chars):")
    print(f"  {note[:200]}...")
    return True


def test_error_injection_extraction():
    """Test extraction from error injection mode (INCORRECT answer)."""
    
    injector_output = """<think>
I need to inject a subtle error. The note says "right" sided pneumonia.
I'll change it to "left" - a laterality error that's clinically significant.
</think>

generated_note:
A 62-year-old woman presents with cough and fever. Chest X-ray shows left lower lobe consolidation
consistent with pneumonia. Started on amoxicillin-clavulanate. Follow up in 1 week.

final_answer: "INCORRECT"

changes_made:
{"original_sentence": "Chest X-ray shows right lower lobe consolidation", "modified_sentence": "Chest X-ray shows left lower lobe consolidation", "error_type": "laterality", "words_changed": "right → left"}"""
    
    note = extract_note_for_assessor(injector_output)
    
    # Verify no leaked info
    assert "INCORRECT" not in note, "Answer INCORRECT leaked to assessor!"
    assert "final_answer" not in note.lower(), "final_answer leaked to assessor!"
    assert "error_type" not in note.lower(), "error_type leaked to assessor!"
    assert "laterality" not in note.lower(), "Error type 'laterality' leaked!"
    assert "right → left" not in note, "Change info leaked!"
    
    # Verify note content
    assert "62-year-old woman" in note
    assert "pneumonia" in note
    
    print("✓ Error injection extraction test passed")
    return True


def test_malformed_output():
    """Test extraction handles malformed outputs gracefully."""
    
    # Missing closing think tag (truncated)
    truncated = """<think>
I need to make a change but the output got cut off

generated_note:
Patient presents with headache and nausea."""
    
    note = extract_note_for_assessor(truncated)
    
    # Should still extract what's there without leaking
    assert "<think>" not in note
    assert "headache" in note or note == ""  # Either extracts note or returns empty
    
    print("✓ Malformed output test passed")
    return True


def test_no_answer_hints():
    """Verify no hints about correct/incorrect status leak through."""
    
    # Edge case: note content that mentions "correct" in medical context
    injector_output = """<think>Keeping it correct with a synonym change.</think>

generated_note:
The patient's posture was corrected using physical therapy. Range of motion improved.

final_answer: "CORRECT"

changes_made:
{"words_changed": "improved → enhanced"}"""
    
    note = extract_note_for_assessor(injector_output)
    
    # "corrected" in medical context is fine, but "CORRECT" as answer should be stripped
    assert "final_answer" not in note.lower()
    # The word "corrected" in note content is OK
    assert "corrected" in note.lower()  
    
    print("✓ No answer hints test passed")
    return True


def test_interaction_class_extraction():
    """Test the MedicalGameInteraction class extraction method."""
    
    config = {"detection_prompts_path": "configs/prompts/error_detection_prompts.json"}
    interaction = MedicalGameInteraction(config)
    
    injector_output = """<think>Making a benign edit</think>

generated_note:
A 30-year-old patient with diabetes presents for routine checkup.
HbA1c level is 7.2%, indicating fair glycemic control.

final_answer: "CORRECT"

changes_made:
{"original_sentence": "HbA1c is 7.2%", "modified_sentence": "HbA1c level is 7.2%", "words_changed": "added 'level'"}"""
    
    note = interaction._extract_generated_note(injector_output)
    
    # Should be sanitized
    assert "CORRECT" not in note, f"Answer leaked! Got: {note}"
    assert "final_answer" not in note.lower()
    assert "changes_made" not in note.lower()
    assert "diabetes" in note  # Content preserved
    
    print("✓ MedicalGameInteraction extraction test passed")
    return True


def test_forbidden_keyword_removal():
    """Test that forbidden keywords are stripped even if regex fails."""
    
    # Weird formatting that might bypass initial regex
    weird_output = """generated_note:
Patient has chest pain. Prescribed nitroglycerin.

final_answer: "INCORRECT"
changes_made: {"error_type": "wrong medication"}
INCORRECT"""  # Stray keyword at end
    
    note = extract_note_for_assessor(weird_output)
    
    assert "INCORRECT" not in note, f"INCORRECT keyword leaked! Got: {note}"
    assert "final_answer" not in note.lower()
    assert "error_type" not in note.lower()
    
    print("✓ Forbidden keyword removal test passed")
    return True


def run_all_tests():
    """Run all extraction tests."""
    print("=" * 60)
    print("Testing Note Extraction for Hidden CoT Compliance")
    print("=" * 60)
    print()
    
    tests = [
        test_basic_extraction,
        test_error_injection_extraction,
        test_malformed_output,
        test_no_answer_hints,
        test_interaction_class_extraction,
        test_forbidden_keyword_removal,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except AssertionError as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test.__name__} ERROR: {e}")
            failed += 1
        print()
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✓ All tests passed! Assessor will not see leaked answers.")
    else:
        print("\n✗ Some tests failed. Fix extraction logic before training.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
