"""
Generate complete CoT dataset for both Critic AND Generator training.

Phase 1: Critic CoT (assessment reasoning)
Phase 2: Generator CoT (creation reasoning)

This creates a unified dataset where the model learns BOTH:
- How to CREATE correct/incorrect notes (generator role)
- How to ASSESS correct/incorrect notes (critic role)
"""

import json
import os
from pathlib import Path
from typing import Dict, List

# ...existing imports...

def load_prompts():
    """Load all prompt templates."""
    prompts_dir = Path('prompts')
    return {
        'critic': (prompts_dir / 'generate_cot_from_pairs.md').read_text(),
        'generator_correct': (prompts_dir / '01_correct_note_generator.md').read_text(),
        'generator_error': (prompts_dir / '02_error_injection_with_reasoning.md').read_text(),
    }

def create_critic_prompt(pair: Dict, note_type: str, prompt_template: str) -> str:
    """Create prompt for critic CoT generation (how to assess notes)."""
    if note_type == 'correct':
        return f"""{prompt_template}

**Input**:
- Clinical Note: {pair['correct_note']}
- Ground Truth: CORRECT

Generate the reasoning trace following the CORRECT note format.
"""
    else:
        return f"""{prompt_template}

**Input**:
- Clinical Note: {pair['incorrect_note']}
- Ground Truth: ERROR
- Error Type: {pair['error_type']}
- Error Sentence: {pair['error_sentence']}
- Corrected Sentence: {pair['corrected_sentence']}

Generate the reasoning trace following the ERROR note format.
"""

def create_generator_correct_prompt(pair: Dict, prompt_template: str) -> str:
    """Create prompt for generator CoT (how to create correct notes)."""
    return f"""{prompt_template}

**Input**:
- Clinical Note: {pair['correct_note']}

Reverse-engineer the reasoning that led to creating this correct note.
Generate the <generation_reasoning> trace.
"""

def create_generator_error_prompt(pair: Dict, prompt_template: str) -> str:
    """Create prompt for generator CoT (how to inject errors)."""
    return f"""{prompt_template}

**Input**:
- Correct Note: {pair['correct_note']}
- Incorrect Note: {pair['incorrect_note']}
- Error Metadata: {{
    "error_type": "{pair['error_type']}",
    "error_sentence": "{pair['error_sentence']}",
    "corrected_sentence": "{pair['corrected_sentence']}"
  }}

Reverse-engineer the adversarial reasoning that led to this error injection.
Generate the <error_injection_reasoning> trace.
"""

def generate_cot_trace(prompt: str, teacher_model_api) -> str:
    """Call teacher model to generate CoT trace."""
    # ...existing code for API call...
    pass

def process_pair_full_cot(pair: Dict, prompts: Dict, teacher_model) -> List[Dict]:
    """
    Generate ALL CoT traces for a single note pair:
    1. Critic CoT for correct note (how to verify it's correct)
    2. Critic CoT for incorrect note (how to detect error)
    3. Generator CoT for correct note (how it was created)
    4. Generator CoT for error note (how error was injected)
    
    Returns 4 training examples per pair.
    """
    results = []
    
    # === CRITIC TRAINING DATA ===
    
    # 1. Critic assessing CORRECT note
    critic_correct_prompt = create_critic_prompt(pair, 'correct', prompts['critic'])
    critic_correct_cot = generate_cot_trace(critic_correct_prompt, teacher_model)
    
    results.append({
        'note_id': f"{pair['note_id']}_critic_correct",
        'role': 'critic',
        'note': pair['correct_note'],
        'reasoning': critic_correct_cot,
        'label': 'CORRECT',
        'task': 'assessment'
    })
    
    # 2. Critic assessing INCORRECT note  
    critic_error_prompt = create_critic_prompt(pair, 'incorrect', prompts['critic'])
    critic_error_cot = generate_cot_trace(critic_error_prompt, teacher_model)
    
    results.append({
        'note_id': f"{pair['note_id']}_critic_error",
        'role': 'critic',
        'note': pair['incorrect_note'],
        'reasoning': critic_error_cot,
        'label': 'ERROR',
        'error_type': pair['error_type'],
        'error_details': {
            'error_sentence': pair['error_sentence'],
            'corrected_sentence': pair['corrected_sentence']
        },
        'task': 'assessment'
    })
    
    # === GENERATOR TRAINING DATA ===
    
    # 3. Generator creating CORRECT note
    gen_correct_prompt = create_generator_correct_prompt(pair, prompts['generator_correct'])
    gen_correct_cot = generate_cot_trace(gen_correct_prompt, teacher_model)
    
    results.append({
        'note_id': f"{pair['note_id']}_generator_correct",
        'role': 'generator',
        'note': pair['correct_note'],
        'reasoning': gen_correct_cot,
        'label': 'CORRECT',
        'task': 'generation'
    })
    
    # 4. Generator injecting ERROR
    gen_error_prompt = create_generator_error_prompt(pair, prompts['generator_error'])
    gen_error_cot = generate_cot_trace(gen_error_prompt, teacher_model)
    
    results.append({
        'note_id': f"{pair['note_id']}_generator_error",
        'role': 'generator',
        'correct_note': pair['correct_note'],
        'incorrect_note': pair['incorrect_note'],
        'reasoning': gen_error_cot,
        'label': 'ERROR',
        'error_type': pair['error_type'],
        'error_details': {
            'error_sentence': pair['error_sentence'],
            'corrected_sentence': pair['corrected_sentence']
        },
        'task': 'generation'
    })
    
    return results

def main():
    """Generate complete training dataset with both critic and generator CoT."""
    # Load existing pairs
    pairs_path = 'data_processed/medec_paired/medec_pairs_combined.jsonl'
    pairs = []
    with open(pairs_path, 'r') as f:
        for line in f:
            pairs.append(json.loads(line))
    
    # Load prompts
    prompts = load_prompts()
    
    # Initialize teacher model
    # teacher_model = initialize_teacher_model()
    
    # Process all pairs
    all_training_data = []
    for pair in pairs[:5]:  # Start with first 5 for testing
        print(f"Processing {pair['note_id']}...")
        cot_examples = process_pair_full_cot(pair, prompts, teacher_model=None)
        all_training_data.extend(cot_examples)
        print(f"  Generated {len(cot_examples)} training examples")
    
    # Save complete dataset
    output_path = 'data_processed/medec_cot/complete_training_data.jsonl'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with open(output_path, 'w') as f:
        for example in all_training_data:
            f.write(json.dumps(example) + '\n')
    
    # Print statistics
    critic_examples = [ex for ex in all_training_data if ex['task'] == 'assessment']
    generator_examples = [ex for ex in all_training_data if ex['task'] == 'generation']
    
    print(f"\n=== Dataset Statistics ===")
    print(f"Total examples: {len(all_training_data)}")
    print(f"Critic examples: {len(critic_examples)}")
    print(f"Generator examples: {len(generator_examples)}")
    print(f"Correct notes: {len([ex for ex in all_training_data if ex['label'] == 'CORRECT'])}")
    print(f"Error notes: {len([ex for ex in all_training_data if ex['label'] == 'ERROR'])}")
    print(f"\nSaved to {output_path}")

if __name__ == '__main__':
    main()
