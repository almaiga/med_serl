"""
Self-play data generation pipeline:
1. Generate correct notes (Prompt 1)
2. Inject errors into subset (Prompt 2)  
3. Generate critic reasoning for both (Prompt 3)
Result: Balanced dataset of correct/incorrect with full CoT traces
"""

from tqdm import tqdm
import random

def generate_self_play_data(base_scenarios, teacher_model, ratio_errors=0.5):
    """
    Args:
        base_scenarios: List of patient scenarios
        teacher_model: GPT-4o or Claude for generating reasoning
        ratio_errors: Proportion of notes to inject errors into
    
    Returns:
        training_data: List of (note, reasoning_trace, label) tuples
    """
    training_data = []
    
    # Step 1: Generate correct notes with reasoning
    for scenario in tqdm(base_scenarios, desc="Generating correct notes"):
        prompt = f"Generate a patient note for the following scenario: {scenario}"
        note = teacher_model.generate(prompt)
        
        prompt_reasoning = f"Explain the reasoning behind this note: {note}"
        reasoning_trace = teacher_model.generate(prompt_reasoning)
        
        training_data.append((note, reasoning_trace, "correct"))
    
    # Step 2: Select subset for error injection
    num_errors = int(len(training_data) * ratio_errors)
    error_indices = random.sample(range(len(training_data)), num_errors)
    
    # Step 3: Generate critic reasoning (both types)
    for idx in tqdm(error_indices, desc="Injecting errors and generating critic reasoning"):
        note, _, _ = training_data[idx]
        
        # Injecting simple errors for self-play
        error_note = inject_error(note)
        training_data[idx] = (error_note, _, "incorrect")
        
        prompt_critic = f"Critique the following note: {error_note}"
        critic_reasoning = teacher_model.generate(prompt_critic)
        
        training_data.append((error_note, critic_reasoning, "incorrect"))
    
    return training_data

def inject_error(note):
    # A placeholder function to inject errors into the notes
    # This should be replaced with the actual error injection logic
    return note.replace("a", "@").replace("e", "3")  # Example: simple char replacement