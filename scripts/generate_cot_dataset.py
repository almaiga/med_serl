"""
Generate Chain-of-Thought reasoning traces from existing correct/incorrect note pairs.
Uses the unified prompt to create balanced training data for both correct and error cases.
"""

import json
import os
from pathlib import Path
from typing import Dict, List
# ...existing imports...

def load_note_pairs(jsonl_path: str) -> List[Dict]:
    """Load the paired correct/incorrect notes from JSONL file."""
    pairs = []
    with open(jsonl_path, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))
    return pairs

def create_cot_prompt(note_pair: Dict, note_type: str) -> str:
    """
    Create the prompt for generating CoT reasoning.
    
    Args:
        note_pair: Dict containing correct_note, incorrect_note, error metadata
        note_type: Either 'correct' or 'incorrect'
    
    Returns:
        Formatted prompt string for the teacher model
    """
    prompt_template = Path('prompts/generate_cot_from_pairs.md').read_text()
    
    if note_type == 'correct':
        return f"""{prompt_template}

**Input**:
- Clinical Note: {note_pair['correct_note']}
- Ground Truth: CORRECT

**Generate the reasoning trace following the CORRECT note format:**
"""
    else:  # incorrect
        return f"""{prompt_template}

**Input**:
- Clinical Note: {note_pair['incorrect_note']}
- Ground Truth: ERROR
- Error Type: {note_pair['error_type']}
- Error Sentence: {note_pair['error_sentence']}
- Corrected Sentence: {note_pair['corrected_sentence']}

**Generate the reasoning trace following the ERROR note format:**
"""

def generate_cot_trace(prompt: str, teacher_model_api) -> str:
    """
    Call teacher model (GPT-4o/Claude) to generate reasoning trace.
    
    Args:
        prompt: The formatted prompt with note and instructions
        teacher_model_api: API client for the teacher model
    
    Returns:
        Generated CoT reasoning text
    """
    # ...existing code for API call...
    pass

def process_note_pair(pair: Dict, teacher_model_api) -> List[Dict]:
    """
    Generate CoT traces for both correct and incorrect versions of a note pair.
    
    Returns:
        List of two training examples (one correct, one error)
    """
    results = []
    
    # Generate CoT for CORRECT note
    correct_prompt = create_cot_prompt(pair, 'correct')
    correct_cot = generate_cot_trace(correct_prompt, teacher_model_api)
    
    results.append({
        'note_id': pair['note_id'] + '_correct',
        'note': pair['correct_note'],
        'reasoning': correct_cot,
        'label': 'CORRECT',
        'error_type': None
    })
    
    # Generate CoT for INCORRECT note
    error_prompt = create_cot_prompt(pair, 'incorrect')
    error_cot = generate_cot_trace(error_prompt, teacher_model_api)
    
    results.append({
        'note_id': pair['note_id'] + '_error',
        'note': pair['incorrect_note'],
        'reasoning': error_cot,
        'label': 'ERROR',
        'error_type': pair['error_type'],
        'error_details': {
            'error_sentence': pair['error_sentence'],
            'corrected_sentence': pair['corrected_sentence']
        }
    })
    
    return results

def main():
    """Generate complete CoT dataset from paired notes."""
    # Load existing pairs
    pairs = load_note_pairs('data_processed/medec_paired/medec_pairs_combined.jsonl')
    
    # Initialize teacher model API
    # teacher_model = initialize_teacher_model()
    
    # Process all pairs
    training_data = []
    for pair in pairs:
        print(f"Processing {pair['note_id']}...")
        cot_examples = process_note_pair(pair, teacher_model_api=None)  # Replace with actual API
        training_data.extend(cot_examples)
    
    # Save training dataset
    output_path = 'data_processed/medec_cot/training_data_with_cot.jsonl'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for example in training_data:
            f.write(json.dumps(example) + '\n')
    
    print(f"Generated {len(training_data)} training examples")
    print(f"Saved to {output_path}")

if __name__ == '__main__':
    main()
