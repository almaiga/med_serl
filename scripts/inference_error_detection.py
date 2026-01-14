#!/usr/bin/env python3
"""
Medical Error Detection Inference Script

Tests different model versions on MEDEC test data:
- Qwen3 models (Qwen/Qwen3-4B, etc.)
- MedGemma models (google/medgemma-4b-it, google/medgemma-4b-pt)
- Fine-tuned models (from SFT/GRPO)

Uses CoT prompting with few-shot examples for error detection.
Supports both Qwen3 thinking format and standard generation.
"""

import os
import json
import argparse
import pandas as pd
import torch
from datetime import datetime
from typing import List, Dict, Tuple
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Try to import PEFT for LoRA support
try:
    from peft import PeftModel, PeftConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

# Qwen3 special token IDs (from official documentation)
THINK_END_TOKEN_ID = 151668  # </think>
IM_END_TOKEN_ID = 151645  # <|im_end|>

# Model type detection
MODEL_TYPE_QWEN = "qwen"
MODEL_TYPE_GENERIC = "generic"  # Gemma, MedGemma, and other CausalLM models


def load_prompts(prompt_file: str = None) -> Dict[str, str]:
    """Load prompts from JSON configuration file."""
    if prompt_file is None:
        # Default path relative to script location
        script_dir = Path(__file__).parent.parent
        prompt_file = script_dir / "configs" / "prompts" / "error_detection_prompts.json"
    
    prompt_path = Path(prompt_file)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_path}")
    
    with open(prompt_path, 'r') as f:
        prompts = json.load(f)
    
    return prompts


def is_lora_adapter(model_path: str) -> bool:
    """Check if the path contains a LoRA adapter."""
    adapter_config = os.path.join(model_path, "adapter_config.json")
    return os.path.exists(adapter_config)


def get_base_model_from_adapter(model_path: str) -> str:
    """Get the base model name from adapter_config.json."""
    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, 'r') as f:
            config = json.load(f)
        return config.get("base_model_name_or_path", "")
    return ""


def detect_model_type(model_path: str) -> str:
    """Detect model type - only need to identify Qwen for special handling."""
    model_path_lower = model_path.lower()
    
    # Check if it's a LoRA adapter and get base model
    if is_lora_adapter(model_path):
        base_model = get_base_model_from_adapter(model_path)
        if base_model:
            model_path_lower = base_model.lower()
    
    # Check for Qwen in path
    if "qwen" in model_path_lower:
        return MODEL_TYPE_QWEN
    
    # Check config.json for Qwen if it's a local path
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            model_type_str = config.get("model_type", "").lower()
            architectures = config.get("architectures", [])
            arch_str = " ".join(architectures).lower()
            
            if "qwen" in model_type_str or "qwen" in arch_str:
                return MODEL_TYPE_QWEN
        except Exception:
            pass
    
    return MODEL_TYPE_GENERIC


def load_model_and_tokenizer(model_path: str, model_type: str):
    """Load model and tokenizer from local or HF path based on model type."""
    print(f"Loading model from: {model_path}")
    print(f"Model type: {model_type}")
    
    # Check if this is a LoRA adapter
    is_adapter = is_lora_adapter(model_path)
    if is_adapter:
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT is required to load LoRA adapters. Install with: pip install peft")
        base_model_path = get_base_model_from_adapter(model_path)
        print(f"🔧 Detected LoRA adapter, base model: {base_model_path}")
    
    # Load base model first if it's a LoRA adapter
    if is_adapter:
        base_model_path = get_base_model_from_adapter(model_path)
        print(f"📦 Loading base model: {base_model_path}")
        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"🔗 Loading LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        # Merge adapter for faster inference
        print(f"⚡ Merging adapter for faster inference...")
        model = model.merge_and_unload()
        
        # Load tokenizer from adapter path (has chat template) or base model
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
    else:
        # Standard model loading
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    adapter_info = " (with LoRA)" if is_adapter else ""
    print(f"✅ Model loaded successfully{adapter_info}")
    return model, tokenizer


def load_test_data(dataset_name: str = "all") -> pd.DataFrame:
    """
    Load MEDEC test data.
    
    Args:
        dataset_name: "ms", "uw", or "all"
    """
    dfs = []
    
    # Try multiple possible data paths
    data_dirs = ["data_raw/MEDEC", "data_copy/MEDEC", "data/MEDEC"]
    
    if dataset_name in ["ms", "all"]:
        ms_loaded = False
        for data_dir in data_dirs:
            ms_path = f"{data_dir}/MEDEC-MS/MEDEC-MS-TestSet-with-GroundTruth-and-ErrorType.csv"
            if os.path.exists(ms_path):
                df_ms = pd.read_csv(ms_path)
                # Filter out rows with null Text ID
                df_ms = df_ms[df_ms['Text ID'].notna()].copy()
                df_ms['dataset'] = 'MS'
                dfs.append(df_ms)
                print(f"✅ Loaded MS test set: {len(df_ms)} examples from {ms_path}")
                ms_loaded = True
                break
        if not ms_loaded and dataset_name == "ms":
            print(f"⚠️ MS test set not found in any of: {data_dirs}")
    
    if dataset_name in ["uw", "all"]:
        uw_loaded = False
        for data_dir in data_dirs:
            uw_path = f"{data_dir}/MEDEC-UW/MEDEC-UW-TestSet-with-GroundTruth-and-ErrorType.csv"
            if os.path.exists(uw_path):
                df_uw = pd.read_csv(uw_path)
                # Filter out rows with null Text ID
                df_uw = df_uw[df_uw['Text ID'].notna()].copy()
                df_uw['dataset'] = 'UW'
                dfs.append(df_uw)
                print(f"✅ Loaded UW test set: {len(df_uw)} examples from {uw_path}")
                uw_loaded = True
                break
        if not uw_loaded and dataset_name == "uw":
            print(f"⚠️ UW test set not found in any of: {data_dirs}")
    
    if not dfs:
        raise FileNotFoundError(f"No test data found for dataset: {dataset_name}")
    
    df = pd.concat(dfs, ignore_index=True)
    print(f"📊 Total test examples: {len(df)}")
    
    return df





def build_error_detection_prompt(
    note: str,
    prompts: Dict[str, str],
    use_cot: bool = True,
    model_type: str = MODEL_TYPE_QWEN
) -> Tuple[List[Dict[str, str]], bool]:
    """
    Build prompt for error detection with CoT.
    
    Args:
        note: Medical note to analyze
        prompts: Dictionary containing 'system_prompt' and 'user_template'
        use_cot: Whether to enable chain-of-thought reasoning
        model_type: Type of model being used
    
    Returns: (messages list for chat template, enable_thinking flag)
    """
    system_prompt = prompts['system_prompt']
    messages = [{"role": "system", "content": system_prompt}]
    
    # Format the user query using the template
    query = prompts['user_template'].format(note=note)
    messages.append({"role": "user", "content": query})
    
    # Return enable_thinking flag for Qwen3's native CoT (only for Qwen models)
    enable_thinking = use_cot and model_type == MODEL_TYPE_QWEN
    return messages, enable_thinking


def parse_qwen3_output(tokenizer, input_ids, generated_ids) -> Tuple[str, str]:
    """
    Parse Qwen3 output using official method (token-based parsing).
    
    Returns: (thinking_content, content)
    """
    input_length = input_ids.shape[1]
    output_ids = generated_ids[0, input_length:].tolist()
    
    # Parse thinking content using token ID (official Qwen3 method)
    try:
        # Find </think> token (151668)
        index = len(output_ids) - output_ids[::-1].index(THINK_END_TOKEN_ID)
    except ValueError:
        # No thinking content found
        index = 0
    
    thinking_content = tokenizer.decode(
        output_ids[:index], 
        skip_special_tokens=True
    ).strip("\n")
    
    content = tokenizer.decode(
        output_ids[index:], 
        skip_special_tokens=True
    ).strip("\n")
    
    return thinking_content, content


def parse_response(thinking: str, content: str) -> Tuple[str, str, str]:
    """
    Parse model response to extract thinking, label, and explanation.
    
    Args:
        thinking: The thinking content (from <think> block)
        content: The final response content
    
    Returns: (thinking, label, explanation)
    """
    label = "Unknown"
    explanation = ""
    
    # Extract label and explanation from the content
    lines = content.split('\n')
    for i, line in enumerate(lines):
        line_lower = line.lower().strip()
        
        # Look for label
        if 'label:' in line_lower or 'answer:' in line_lower:
            label_text = line.split(':', 1)[1].strip()
            # Normalize label
            if 'correct' in label_text.lower() and 'incorrect' not in label_text.lower():
                label = 'CORRECT'
            elif 'incorrect' in label_text.lower():
                label = 'INCORRECT'
        
        # Look for explanation
        if 'explanation:' in line_lower:
            explanation = line.split(':', 1)[1].strip()
            # Get rest of explanation if multi-line
            if i + 1 < len(lines):
                remaining = '\n'.join(lines[i+1:]).strip()
                if remaining:
                    explanation = explanation + " " + remaining
            break
    
    # If no structured format, try to infer from text
    if label == "Unknown":
        content_lower = content.lower()
        if 'no error' in content_lower or 'is correct' in content_lower or 'correct' in content_lower:
            label = 'CORRECT'
        elif 'error' in content_lower or 'incorrect' in content_lower or 'mistake' in content_lower:
            label = 'INCORRECT'
    
    # If still no explanation, use the whole content
    if not explanation:
        explanation = content
    
    return thinking, label, explanation


def run_inference(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    model_type: str,
    prompts: Dict[str, str],
    use_cot: bool = True,
    max_samples: int = None,
    temperature: float = 0.3,
    max_new_tokens: int = 512,
    thinking_budget: int = 1024,
    batch_size: int = 1
) -> List[Dict]:
    """
    Unified inference function for all model types with batch processing support.
    Handles Qwen3's special two-stage generation and standard generation for others.
    
    Args:
        batch_size: Number of samples to process in parallel (default: 1)
    """
    results = []
    
    # Limit samples if specified
    if max_samples:
        test_df = test_df.head(max_samples)
    
    model.eval()
    is_qwen = (model_type == MODEL_TYPE_QWEN)
    
    # Set padding side to left for batch generation (required for causal LM)
    original_padding_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    
    # Process in batches
    num_samples = len(test_df)
    num_batches = (num_samples + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), desc="Processing batches"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, num_samples)
        batch_df = test_df.iloc[batch_start:batch_end]
        
        # Prepare batch data
        batch_notes = []
        batch_metadata = []
        batch_prompts = []
        enable_thinking_flags = []
        
        for idx, row in batch_df.iterrows():
            note = row['Text']
            # Handle NaN values properly - convert to empty string or None
            error_type = row.get('Error Type', '')
            if pd.isna(error_type) or error_type is None:
                error_type = ''
            else:
                error_type = str(error_type)
            
            # Handle text_id NaN
            text_id = row.get('Text ID', f'sample_{idx}')
            if pd.isna(text_id):
                text_id = f'sample_{idx}'
            else:
                text_id = str(text_id)
            
            # Handle dataset NaN
            dataset = row.get('dataset', 'unknown')
            if pd.isna(dataset):
                dataset = 'unknown'
            else:
                dataset = str(dataset)
            
            batch_notes.append(note)
            batch_metadata.append({
                'index': idx,
                'text_id': text_id,
                'dataset': dataset,
                'ground_truth': int(row['Error Flag']),
                'error_type': error_type,
                'note': str(note)
            })
            
            # Build prompt
            messages, enable_thinking = build_error_detection_prompt(note, prompts, use_cot, model_type)
            enable_thinking_flags.append(enable_thinking)
            
            # Apply chat template
            if is_qwen:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=enable_thinking
                )
            else:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            batch_prompts.append(prompt)
        
        # Tokenize batch with padding
        model_inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048
        ).to(model.device)
        
        input_lengths = (model_inputs.attention_mask.sum(dim=1)).tolist()
        
        # Generate - different logic for Qwen vs others
        if is_qwen and any(enable_thinking_flags):
            # Qwen3 two-stage generation with thinking
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=thinking_budget,
                    temperature=temperature,
                    top_p=0.95,
                    do_sample=temperature > 0,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Process each sample in batch individually
            batch_thinking_contents = []
            batch_final_inputs = []
            needs_second_stage = []
            
            for i in range(len(batch_prompts)):
                input_length = input_lengths[i]
                output_ids = generated_ids[i, input_length:].tolist()
                
                # Check if generation finished or thinking budget reached
                if IM_END_TOKEN_ID not in output_ids:
                    # Check if thinking process finished
                    if THINK_END_TOKEN_ID not in output_ids:
                        # Thinking budget reached - inject early stopping prompt
                        early_stopping_text = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
                        early_stopping_ids = tokenizer(
                            early_stopping_text,
                            return_tensors="pt",
                            add_special_tokens=False
                        ).input_ids.to(model.device)
                        
                        new_input = torch.cat([generated_ids[i:i+1], early_stopping_ids], dim=-1)
                    else:
                        new_input = generated_ids[i:i+1]
                    
                    batch_final_inputs.append(new_input)
                    needs_second_stage.append(True)
                else:
                    # Generation complete in first stage
                    batch_final_inputs.append(generated_ids[i:i+1])
                    needs_second_stage.append(False)
            
            # Second generation stage if needed
            if any(needs_second_stage):
                # Pad inputs for batch processing
                max_len = max(inp.size(-1) for inp in batch_final_inputs)
                padded_inputs = []
                attention_masks = []
                
                for inp in batch_final_inputs:
                    pad_len = max_len - inp.size(-1)
                    if pad_len > 0:
                        padding = torch.full(
                            (1, pad_len),
                            tokenizer.pad_token_id,
                            dtype=inp.dtype,
                            device=inp.device
                        )
                        padded_inp = torch.cat([padding, inp], dim=-1)
                        attn_mask = torch.cat([
                            torch.zeros((1, pad_len), dtype=torch.long, device=inp.device),
                            torch.ones((1, inp.size(-1)), dtype=torch.long, device=inp.device)
                        ], dim=-1)
                    else:
                        padded_inp = inp
                        attn_mask = torch.ones_like(inp, dtype=torch.long)
                    
                    padded_inputs.append(padded_inp)
                    attention_masks.append(attn_mask)
                
                input_ids_batch = torch.cat(padded_inputs, dim=0)
                attention_mask_batch = torch.cat(attention_masks, dim=0)
                
                # Calculate remaining tokens
                remaining_tokens = max_new_tokens - (input_ids_batch.size(-1) - max(input_lengths))
                if remaining_tokens > 0:
                    with torch.no_grad():
                        generated_ids = model.generate(
                            input_ids=input_ids_batch,
                            attention_mask=attention_mask_batch,
                            max_new_tokens=remaining_tokens,
                            temperature=temperature,
                            top_p=0.95,
                            do_sample=temperature > 0,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=tokenizer.eos_token_id
                        )
            
            # Parse each sample individually
            for i, metadata in enumerate(batch_metadata):
                input_length = input_lengths[i]
                
                # Decode only the valid tokens (excluding padding)
                valid_tokens = generated_ids[i][generated_ids[i] != tokenizer.pad_token_id]
                output_ids = valid_tokens[input_length:].tolist()
                
                # Parse using official Qwen3 method (token-based)
                try:
                    index = len(output_ids) - output_ids[::-1].index(THINK_END_TOKEN_ID)
                except ValueError:
                    index = 0
                
                thinking_content = tokenizer.decode(
                    output_ids[:index],
                    skip_special_tokens=True
                ).strip("\n")
                
                content = tokenizer.decode(
                    output_ids[index:],
                    skip_special_tokens=True
                ).strip("\n")
                
                # Parse response
                thinking, predicted_label, explanation = parse_response(thinking_content, content)
                
                # Store result - ensure all values are JSON serializable with proper NaN handling
                gt_label = "INCORRECT" if metadata['ground_truth'] == 1 else "CORRECT"
                correct = (predicted_label == gt_label)
                
                results.append({
                    'text_id': str(metadata['text_id']),
                    'dataset': str(metadata['dataset']),
                    'note': str(metadata['note']),
                    'ground_truth_flag': int(metadata['ground_truth']),
                    'ground_truth_label': gt_label,
                    'error_type': str(metadata['error_type']) if metadata['error_type'] else '',
                    'predicted_label': predicted_label,
                    'explanation': explanation if explanation else '',
                    'thinking': thinking if thinking else '',
                    'correct': bool(correct),
                    'thinking_content': thinking_content if thinking_content else '',
                    'final_content': content if content else ''
                })
        
        else:
            # Standard generation for non-Qwen models or Qwen without thinking
            with torch.no_grad():
                generated_ids = model.generate(
                    **model_inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature if temperature > 0 else None,
                    do_sample=temperature > 0,
                    top_p=0.95 if temperature > 0 else None,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )
            
            # Decode each sample individually
            for i, metadata in enumerate(batch_metadata):
                input_length = input_lengths[i]
                
                # Get only valid tokens (excluding padding)
                valid_tokens = generated_ids[i][generated_ids[i] != tokenizer.pad_token_id]
                output_ids = valid_tokens[input_length:]
                
                # Decode
                content = tokenizer.decode(output_ids, skip_special_tokens=True)
                thinking_content = ""
                
                # Parse response
                thinking, predicted_label, explanation = parse_response(thinking_content, content)
                
                # Store result - ensure all values are JSON serializable with proper NaN handling
                gt_label = "INCORRECT" if metadata['ground_truth'] == 1 else "CORRECT"
                correct = (predicted_label == gt_label)
                
                results.append({
                    'text_id': str(metadata['text_id']),
                    'dataset': str(metadata['dataset']),
                    'note': str(metadata['note']),
                    'ground_truth_flag': int(metadata['ground_truth']),
                    'ground_truth_label': gt_label,
                    'error_type': str(metadata['error_type']) if metadata['error_type'] else '',
                    'predicted_label': predicted_label,
                    'explanation': explanation if explanation else '',
                    'thinking': thinking if thinking else '',
                    'correct': bool(correct),
                    'thinking_content': thinking_content if thinking_content else '',
                    'final_content': content if content else ''
                })
    
    # Restore original padding side
    tokenizer.padding_side = original_padding_side
    
    return results





def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate accuracy, precision, recall, F1."""
    total = len(results)
    correct = sum(1 for r in results if r['correct'])
    
    # Binary classification metrics (CORRECT vs INCORRECT)
    tp = sum(1 for r in results if r['predicted_label'] == 'INCORRECT' and r['ground_truth_label'] == 'INCORRECT')
    fp = sum(1 for r in results if r['predicted_label'] == 'INCORRECT' and r['ground_truth_label'] == 'CORRECT')
    tn = sum(1 for r in results if r['predicted_label'] == 'CORRECT' and r['ground_truth_label'] == 'CORRECT')
    fn = sum(1 for r in results if r['predicted_label'] == 'CORRECT' and r['ground_truth_label'] == 'INCORRECT')
    
    accuracy = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return {
        'total_samples': total,
        'correct': correct,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': {
            'true_positive': tp,
            'false_positive': fp,
            'true_negative': tn,
            'false_negative': fn
        }
    }


def main():
    parser = argparse.ArgumentParser(description="Medical Error Detection Inference")
    
    # Model arguments
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to model (local or HuggingFace)")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Name for this model in results (default: use model_path)")
    parser.add_argument("--model_type", type=str, default=None,
                       choices=["qwen", "generic"],
                       help="Model type (auto-detected if not specified)")
    
    # Data arguments
    parser.add_argument("--dataset", type=str, default="all",
                       choices=["ms", "uw", "all"],
                       help="Which test dataset to use")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to test (default: all)")
    
    # Prompting arguments
    parser.add_argument("--prompt_file", type=str, default=None,
                       help="Path to prompt configuration JSON file")
    parser.add_argument("--no_cot", action="store_true",
                       help="Disable chain-of-thought reasoning")
    
    # Generation arguments
    parser.add_argument("--temperature", type=float, default=0.7,
                       help="Sampling temperature (default: 0.7)")
    parser.add_argument("--max_new_tokens", type=int, default=512,
                       help="Maximum tokens to generate (default: 512)")
    parser.add_argument("--thinking_budget", type=int, default=1024,
                       help="Thinking budget for Qwen3 (default: 1024)")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for parallel inference (default: 1)")
    
    # Output arguments
    parser.add_argument("--output_dir", type=str, default="results/inference",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Detect or use specified model type
    model_type = args.model_type or detect_model_type(args.model_path)
    
    # Set model name
    model_name = args.model_name or args.model_path.replace('/', '_')
    
    # Create output directory with CoT info
    cot_suffix = "cot" if not args.no_cot else "no_cot"
    output_subdir = os.path.join(args.output_dir, model_name, cot_suffix)
    os.makedirs(output_subdir, exist_ok=True)
    
    # Load prompts
    prompts = load_prompts(args.prompt_file)
    
    print(f"\n{'='*60}")
    print(f"🔬 Medical Error Detection Inference")
    print(f"{'='*60}")
    print(f"Model: {args.model_path}")
    print(f"Model Type: {model_type}")
    print(f"Dataset: {args.dataset}")
    print(f"Batch Size: {args.batch_size}")
    print(f"CoT: {not args.no_cot}")
    print(f"Temperature: {args.temperature}")
    print(f"Output: {output_subdir}")
    print(f"{'='*60}\n")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_path, model_type)
    
    # Load test data
    test_df = load_test_data(args.dataset)
    
    # Run inference
    results = run_inference(
        model=model,
        tokenizer=tokenizer,
        test_df=test_df,
        model_type=model_type,
        prompts=prompts,
        use_cot=not args.no_cot,
        max_samples=args.max_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        thinking_budget=args.thinking_budget,
        batch_size=args.batch_size
    )
    
    # Calculate metrics
    metrics = calculate_metrics(results)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"📊 Results Summary")
    print(f"{'='*60}")
    print(f"Total samples: {metrics['total_samples']}")
    print(f"Correct: {metrics['correct']}")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    print(f"\nConfusion Matrix:")
    print(f"  TP: {metrics['confusion_matrix']['true_positive']}")
    print(f"  FP: {metrics['confusion_matrix']['false_positive']}")
    print(f"  TN: {metrics['confusion_matrix']['true_negative']}")
    print(f"  FN: {metrics['confusion_matrix']['false_negative']}")
    print(f"{'='*60}\n")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results with proper JSON serialization
    results_file = os.path.join(
        output_subdir,
        f"{args.dataset}_{timestamp}_results.jsonl"
    )
    with open(results_file, 'w', encoding='utf-8') as f:
        for result in results:
            # Ensure proper JSON serialization
            json_str = json.dumps(result, ensure_ascii=False)
            f.write(json_str + '\n')
    print(f"✅ Detailed results saved to: {results_file}")
    
    # Save summary
    summary = {
        'model_path': args.model_path,
        'model_name': model_name,
        'dataset': args.dataset,
        'cot': not args.no_cot,
        'batch_size': args.batch_size,
        'temperature': args.temperature,
        'max_new_tokens': args.max_new_tokens,
        'thinking_budget': args.thinking_budget if not args.no_cot else None,
        'timestamp': timestamp,
        'metrics': metrics
    }
    
    summary_file = os.path.join(
        output_subdir,
        f"{args.dataset}_{timestamp}_summary.json"
    )
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ Summary saved to: {summary_file}")


if __name__ == "__main__":
    main()
