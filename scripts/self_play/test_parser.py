"""Test script for CoT parser (sentence-level localization)."""

import sys
sys.path.insert(0, '/Users/josmaiga/Documents/GitHub/med_serl')

from scripts.self_play.cot_parser import parse_injector_output, parse_assessor_output, extract_note_for_assessor
from scripts.self_play.utils import parse_assessor_answer, number_sentences, reconstruct_note

print("=== Injector Parse Test (compact format) ===")

# Test compact injector output: "N. <modified sentence>"
injector_response = '''<think>I will change "presents" to "comes" in sentence 1.</think>

1. A 26-year-old immigrant from Mexico comes to your clinic for a physical.'''

result = parse_injector_output(injector_response)
print(f'Thinking: {result.thinking[:50] if result.thinking else None}...')
print(f'Changed sentence ID: {result.changed_sentence_id}')
print(f'Modified text: {result.modified_text}')
print(f'Parse success: {result.parse_success}')
assert result.changed_sentence_id == 1
assert "comes to your clinic" in result.modified_text
print("  PASS")

# Test injector with different sentence number
injector_response2 = '''<think>Changing the treatment</think>

7. Azithromycin and ceftriaxone are prescribed.'''

result2 = parse_injector_output(injector_response2)
print(f'\nChanged sentence ID: {result2.changed_sentence_id}')
print(f'Modified text: {result2.modified_text}')
assert result2.changed_sentence_id == 7
assert "Azithromycin" in result2.modified_text
print("  PASS")

print('\n=== Assessor Parse Test (CORRECT or sentence number) ===')

# Test CORRECT
assessor_correct = '''<think>The note appears clinically accurate</think>

CORRECT'''

label, sid = parse_assessor_answer(assessor_correct)
print(f'Label: {label}, Sentence ID: {sid}')
assert label == "CORRECT" and sid is None
print("  PASS")

# Test sentence number
assessor_error = '''<think>Sentence 3 has wrong dosage</think>

3'''

label, sid = parse_assessor_answer(assessor_error)
print(f'Label: {label}, Sentence ID: {sid}')
assert label == "ERROR" and sid == 3
print("  PASS")

# Test bare integer without think block
label, sid = parse_assessor_answer("7")
print(f'Bare "7": Label: {label}, Sentence ID: {sid}')
assert label == "ERROR" and sid == 7
print("  PASS")

# Test invalid output
label, sid = parse_assessor_answer("I'm not sure about this note.")
print(f'Invalid: Label: {label}, Sentence ID: {sid}')
assert label == "UNKNOWN" and sid is None
print("  PASS")

print('\n=== Reconstruct Note Test ===')

original = "1. First sentence.\n2. Second sentence.\n3. Third sentence."
modified = reconstruct_note(original, 2, "Modified second sentence.")
print(f'Original:\n{original}')
print(f'Modified (sentence 2):\n{modified}')
assert "1. First sentence." in modified
assert "2. Modified second sentence." in modified
assert "3. Third sentence." in modified
print("  PASS")

print('\n=== All tests passed! ===')
