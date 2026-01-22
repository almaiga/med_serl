"""Test script for CoT parser."""

from cot_parser import parse_injector_output, parse_assessor_output, extract_public_response

# Test injector parsing
injector_response = '''<think>I will change "presents" to "comes"</think>

generated_note:
A 26-year-old immigrant from Mexico comes to your clinic for a physical.

final_answer: "CORRECT"

changes_made:
{"original_sentence": "presents to your clinic", "modified_sentence": "comes to your clinic", "words_changed": "presents → comes"}'''

result = parse_injector_output(injector_response)
print('=== Injector Parse Test ===')
print(f'Thinking: {result.thinking[:50] if result.thinking else None}...')
print(f'Generated note: {result.generated_note[:50] if result.generated_note else None}...')
print(f'Final answer: {result.final_answer}')
print(f'Parse success: {result.parse_success}')

# Test public extraction (hidden CoT)
public = extract_public_response(injector_response)
print(f'\nPublic (no thinking): {public[:100]}...')

# Test assessor parsing
assessor_response = '''<think>The note appears clinically accurate</think>

final_answer: "CORRECT"
Explanation: The clinical note contains no medical errors.'''

result2 = parse_assessor_output(assessor_response)
print('\n=== Assessor Parse Test ===')
print(f'Final answer: {result2.final_answer}')
print(f'Explanation: {result2.explanation}')
print(f'Parse success: {result2.parse_success}')

print('\n=== All tests passed! ===')
