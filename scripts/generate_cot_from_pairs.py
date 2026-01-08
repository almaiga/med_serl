"""
Generate CoT reasoning traces from existing correct/incorrect note pairs.
Uses a teacher model (GPT-4o/Claude) to create structured reasoning.
"""

def generate_cot_for_pair(note_pair, teacher_model, cot_prompt_template):
    """
    Given a (correct_note, incorrect_note, error_details) tuple,
    generate two CoT traces: one for correct, one for error detection.
    
    Args:
        note_pair: Dict with keys {correct_note, incorrect_note, error_type, error_location, correction}
        teacher_model: API client for GPT-4o or Claude
        cot_prompt_template: The unified prompt from above
    
    Returns:
        [
            {note: correct_note, reasoning: cot_correct, label: "CORRECT"},
            {note: incorrect_note, reasoning: cot_error, label: "ERROR", error_details: {...}}
        ]
    """
    results = []
    
    # Generate CoT for CORRECT note
    correct_prompt = f"""{cot_prompt_template}
    
Input:
- Clinical Note: {note_pair['correct_note']}
- Ground Truth: CORRECT

Generate the reasoning trace:
"""
    # ...existing code...
    
    # Generate CoT for ERROR note
    error_prompt = f"""{cot_prompt_template}

Input:
- Clinical Note: {note_pair['incorrect_note']}
- Ground Truth: ERROR
- Error Details: Type={note_pair['error_type']}, Location={note_pair['error_location']}, Correction={note_pair['correction']}

Generate the reasoning trace:
"""
    # ...existing code...
    
    return results


def process_all_pairs(pairs_dataset, teacher_model, output_path):
    """
    Process entire dataset of note pairs.
    Creates balanced training set with CoT for both correct and error cases.
    """
    training_data = []
    
    for pair in pairs_dataset:
        # ...existing code...
        cot_traces = generate_cot_for_pair(pair, teacher_model, COT_PROMPT)
        training_data.extend(cot_traces)
    
    # ...existing code...
    # Save to jsonl for SFT training