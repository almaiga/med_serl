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
import json
import argparse
import pandas as pd
import torch
from pathlib import Path
from typing import List, Dict
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


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


def load_training_data(data_source: str = "raw") -> pd.DataFrame:
    """
    Load MEDEC training data (excluding test sets).
    
    Args:
        data_source: "raw" for CSV files, "processed" for existing JSONL
    """
    dfs = []
    
    if data_source == "raw":
        # Try multiple possible data paths
        data_dirs = ["data_raw/MEDEC", "data_copy/MEDEC", "data/MEDEC"]
        
        # Load MS training data
        ms_loaded = False
        for data_dir in data_dirs:
            # Look for training file (not test file)
            ms_files = [
                f"{data_dir}/MEDEC-MS/MEDEC-MS.csv",
                f"{data_dir}/MEDEC-MS/train.csv",
                f"{data_dir}/MEDEC-MS/MEDEC-MS-TrainingSet.csv"
            ]
            
            for ms_path in ms_files:
                if os.path.exists(ms_path):
                    df_ms = pd.read_csv(ms_path)
                    # Filter out rows with null Text ID
                    df_ms = df_ms[df_ms['Text ID'].notna()].copy()
                    df_ms['dataset'] = 'MS'
                    dfs.append(df_ms)
                    print(f"✅ Loaded MS training set: {len(df_ms)} examples from {ms_path}")
                    ms_loaded = True
                    break
            if ms_loaded:
                break
        
        if not ms_loaded:
            print(f"⚠️ MS training set not found. Will look for all MS data...")
            for data_dir in data_dirs:
                ms_path = f"{data_dir}/MEDEC-MS/MEDEC-MS.csv"
                if os.path.exists(ms_path):
                    df_ms = pd.read_csv(ms_path)
                    # Filter out test examples if 'split' column exists
                    if 'split' in df_ms.columns:
                        df_ms = df_ms[df_ms['split'] == 'train'].copy()
                    df_ms = df_ms[df_ms['Text ID'].notna()].copy()
                    df_ms['dataset'] = 'MS'
                    dfs.append(df_ms)
                    print(f"✅ Loaded MS data: {len(df_ms)} examples from {ms_path}")
                    break
        
        # Load UW training data
        uw_loaded = False
        for data_dir in data_dirs:
            uw_files = [
                f"{data_dir}/MEDEC-UW/MEDEC-UW.csv",
                f"{data_dir}/MEDEC-UW/train.csv",
                f"{data_dir}/MEDEC-UW/MEDEC-UW-TrainingSet.csv"
            ]
            
            for uw_path in uw_files:
                if os.path.exists(uw_path):
                    df_uw = pd.read_csv(uw_path)
                    df_uw = df_uw[df_uw['Text ID'].notna()].copy()
                    df_uw['dataset'] = 'UW'
                    dfs.append(df_uw)
                    print(f"✅ Loaded UW training set: {len(df_uw)} examples from {uw_path}")
                    uw_loaded = True
                    break
            if uw_loaded:
                break
        
        if not uw_loaded:
            print(f"⚠️ UW training set not found. Will look for all UW data...")
            for data_dir in data_dirs:
                uw_path = f"{data_dir}/MEDEC-UW/MEDEC-UW.csv"
                if os.path.exists(uw_path):
                    df_uw = pd.read_csv(uw_path)
                    if 'split' in df_uw.columns:
                        df_uw = df_uw[df_uw['split'] == 'train'].copy()
                    df_uw = df_uw[df_uw['Text ID'].notna()].copy()
                    df_uw['dataset'] = 'UW'
                    dfs.append(df_uw)
                    print(f"✅ Loaded UW data: {len(df_uw)} examples from {uw_path}")
                    break
    
    if not dfs:
        raise FileNotFoundError("No training data found. Please check data paths.")
    
    df = pd.concat(dfs, ignore_index=True)
    
    # Exclude test examples if they're in the same file
    if 'Text ID' in df.columns:
        # Filter out test IDs (they usually have 'test' in the ID)
        df = df[~df['Text ID'].str.contains('test', case=False, na=False)].copy()
    
    print(f"📊 Total training examples: {len(df)}")
    
    return df


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
    train_df: pd.DataFrame,
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
        train_df = train_df.head(max_samples)
    
    model.eval()
    
    # Process in batches
    total_batches = (len(train_df) + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(0, len(train_df), batch_size), 
                          total=total_batches, 
                          desc="Generating post-fill reasoning"):
        batch_df = train_df.iloc[batch_idx:batch_idx + batch_size]
        
        # Extract batch data
        notes = batch_df['Text'].tolist()
        error_flags = batch_df['Error Flag'].tolist()
        labels = ["INCORRECT" if flag == 1 else "CORRECT" for flag in error_flags]
        
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
        for idx, row in batch_df.iterrows():
            result_idx = batch_idx + (idx - batch_df.index[0])
            results.append({
                'text_id': row.get('Text ID', f"train_{result_idx}"),
                'dataset': row.get('dataset', 'unknown'),
                'note': row['Text'],
                'label': labels[result_idx - batch_idx],
                'error_flag': int(row['Error Flag']),
                'reasoning': reasonings[result_idx - batch_idx]
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
    parser.add_argument("--data_source", type=str, default="raw",
                       choices=["raw", "processed"],
                       help="Data source: raw CSV or processed JSONL")
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
    print(f"Data Source: {args.data_source}")
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
    train_df = load_training_data(args.data_source)
    
    # Generate post-fill reasoning
    print(f"\n🔄 Generating reasoning...")
    results = generate_postfill_data(
        model=model,
        tokenizer=tokenizer,
        train_df=train_df,
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
