#!/usr/bin/env python3
"""
Generate Post-Fill Reasoning for MEDEC Training Data

This script generates reasoning by showing the model the correct answer first,
then asking it to explain why that classification is accurate (self-distillation).

Process:
1. Load MEDEC training data from raw CSVs
2. Transform Error Flag (0/1) to labels (CORRECT/INCORRECT)
3. For each example, generate post-fill reasoning:
   - Show model the note + correct label
   - Model generates reasoning explaining the classification
4. Save data with: note, label, reasoning

Output format:
{
  "text_id": "ms-train-0",
  "dataset": "MS",
  "note": "[medical note]",
  "label": "CORRECT" or "INCORRECT",
  "error_flag": 0 or 1,
  "reasoning": "[model-generated reasoning]"
}
"""

import os
import sys
import json
import argparse
import pandas as pd
import torch
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def load_prompts(prompt_file: str = None) -> Dict[str, str]:
    """Load post-fill prompts from JSON configuration file."""
    if prompt_file is None:
        script_dir = Path(__file__).parent.parent
        prompt_file = script_dir / "configs" / "prompts" / "postfill_prompts.json"
    
    prompt_path = Path(prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r') as f:
        prompts = json.load(f)
    
    return prompts


def load_training_data(data_path: str = "data_raw/MEDEC") -> List[Dict]:
    """
    Load MEDEC training data using existing MedicalDataProcessor.
    
    Args:
        data_path: Path to MEDEC dataset
    
    Returns:
        List of training examples with text, error_flag, dataset info
    """
    from src.data_processor import MedicalDataProcessor
    
    processor = MedicalDataProcessor.load_training_data(data_path=data_path)
    print(f"✅ Loaded MEDEC data")
    print(f"  Error pool: {len(processor.error_pool)} samples")
    print(f"  Clean pool: {len(processor.clean_pool)} samples")
    
    # Convert to list format for generation
    training_examples = []
    
    # Add error samples
    for sample in processor.error_pool:
        training_examples.append({
            'text_id': sample.get('text_id', 'unknown'),
            'dataset': sample.get('dataset', 'unknown'),
            'text': sample.get('original_text', sample.get('text', '')),
            'error_flag': 1,
            'label': 'INCORRECT',
            'error_type': sample.get('meta', {}).get('error_type', 'unknown')
        })
    
    # Add clean samples
    for sample in processor.clean_pool:
        training_examples.append({
            'text_id': sample.get('text_id', 'unknown'),
            'dataset': sample.get('dataset', 'unknown'),
            'text': sample.get('original_text', sample.get('text', '')),
            'error_flag': 0,
            'label': 'CORRECT',
            'error_type': None
        })
    
    print(f"📊 Total training examples: {len(training_examples)}")
    print(f"  INCORRECT: {sum(1 for ex in training_examples if ex['label'] == 'INCORRECT')}")
    print(f"  CORRECT: {sum(1 for ex in training_examples if ex['label'] == 'CORRECT')}")
    
    return training_examples


def build_postfill_prompt(note: str, label: str, prompts: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Build post-fill prompt where model sees the answer and generates reasoning.
    
    Args:
        note: Medical note text
        label: "CORRECT" or "INCORRECT"
        prompts: Dictionary containing prompt templates
    
    Returns: messages list for chat template
    """
    system_prompt = prompts['system_prompt']
    
    # Choose template based on label
    if label == "CORRECT":
        user_prompt = prompts['postfill_template_correct'].format(note=note)
    else:
        user_prompt = prompts['postfill_template_incorrect'].format(note=note)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    return messages


def generate_reasoning_batch(
    model,
    tokenizer,
    notes: List[str],
    labels: List[str],
    prompts: Dict[str, str],
    temperature: float = 0.7,
    max_new_tokens: int = 256
) -> List[str]:
    """
    Generate reasoning for a batch of examples.
    
    Returns: List of generated reasoning strings
    """
    reasonings = []
    
    for note, label in zip(notes, labels):
        # Build prompt
        messages = build_postfill_prompt(note, label, prompts)
        
        # Apply chat template
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
        input_length = inputs.input_ids.size(-1)
        
        # Generate
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.95 if temperature > 0 else None,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        # Decode
        output_ids = generated_ids[0][input_length:]
        reasoning = tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        reasonings.append(reasoning)
    
    return reasonings


def generate_postfill_data(
    model,
    tokenizer,
    training_examples: List[Dict],
    prompts: Dict[str, str],
    batch_size: int = 1,
    temperature: float = 0.7,
    max_new_tokens: int = 256,
    max_samples: int = None
) -> List[Dict]:
    """
    Generate post-fill reasoning for all training examples.
    """
    results = []
    
    # Limit samples if specified
    if max_samples:
        training_examples = training_examples[:max_samples]
    
    model.eval()
    
    # Process in batches
    total_batches = (len(training_examples) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(0, len(training_examples), batch_size), 
                          total=total_batches, 
                          desc="Generating post-fill reasoning"):
        batch_examples = training_examples[batch_idx:batch_idx + batch_size]
        
        # Extract batch data
        notes = [ex['text'] for ex in batch_examples]
        labels = [ex['label'] for ex in batch_examples]
        
        # Generate reasoning
        reasonings = generate_reasoning_batch(
            model=model,
            tokenizer=tokenizer,
            notes=notes,
            labels=labels,
            prompts=prompts,
            temperature=temperature,
            max_new_tokens=max_new_tokens
        )
        
        # Store results
        for i, ex in enumerate(batch_examples):
            results.append({
                'text_id': ex['text_id'],
                'dataset': ex['dataset'],
                'note': ex['text'],
                'label': ex['label'],
                'error_flag': ex['error_flag'],
                'reasoning': reasonings[i]
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Generate Post-Fill Reasoning for MEDEC Training Data")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to base model (e.g., google/medgemma-2b)")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Model name for output file")
    
    # Data arguments
    parser.add_argument("--medec_path", type=str, default="data_raw/MEDEC",
                       help="Path to MEDEC dataset")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum samples to process (for testing)")
    
    # Prompt arguments
    parser.add_argument("--prompt_file", type=str, default=None,
                       help="Path to post-fill prompts JSON file")
    
    # Generation arguments
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for generation (default: 1)")
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (default: 0.7)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                       help="Maximum tokens to generate (default: 256)")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="data_processed/medec",
                       help="Output directory for generated data")
    parser.add_argument("--output_name", type=str, default=None,
                       help="Output filename (default: auto-generated)")
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set model name
    model_name = args.model_name or args.model_path.split('/')[-1]
    
    print(f"\n{'='*60}")
    print(f"🔬 Generating Post-Fill Reasoning Data")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"MEDEC Path: {args.medec_path}")
    print(f"Temperature: {args.temperature}")
    print(f"Max Tokens: {args.max_new_tokens}")
    print(f"Batch Size: {args.batch_size}")
    if args.max_samples:
        print(f"Max Samples: {args.max_samples}")
    print(f"{'='*60}\n")
    
    # Load prompts
    prompts = load_prompts(args.prompt_file)
    print(f"✅ Loaded post-fill prompts")
    
    # Load model
    print(f"\n📦 Loading model: {args.model_path}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print(f"✅ Model loaded successfully")
    
    # Load training data
    print(f"\n📊 Loading training data...")
    training_examples = load_training_data(args.medec_path)
    
    # Generate post-fill reasoning
    print(f"\n🔄 Generating reasoning...")
    results = generate_postfill_data(
        model=model,
        tokenizer=tokenizer,
        training_examples=training_examples,
        prompts=prompts,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_samples=args.max_samples
    )
    
    # Calculate statistics
    correct_count = sum(1 for r in results if r['label'] == 'CORRECT')
    incorrect_count = len(results) - correct_count
    
    print(f"\n{'='*60}")
    print(f"📊 Generation Summary")
    print(f"{'='*60}")
    print(f"Total examples: {len(results)}")
    print(f"CORRECT: {correct_count} ({correct_count/len(results)*100:.1f}%)")
    print(f"INCORRECT: {incorrect_count} ({incorrect_count/len(results)*100:.1f}%)")
    print(f"{'='*60}\n")
    
    # Sample outputs for review
    print(f"📝 Sample Generated Reasonings:\n")
    for i, result in enumerate(results[:3]):
        print(f"Example {i+1}:")
        print(f"Label: {result['label']}")
        print(f"Reasoning: {result['reasoning'][:200]}...")
        print()
    
    # Save results
    output_name = args.output_name or f"train_postfill_{model_name}.jsonl"
    output_path = os.path.join(args.output_dir, output_name)
    
    with open(output_path, 'w') as f:
        for result in results:
            f.write(json.dumps(result) + '\n')
    
    print(f"✅ Post-fill reasoning data saved to: {output_path}")
    print(f"\n💡 Next step: Run SFT training on this data")


if __name__ == "__main__":
    main()
