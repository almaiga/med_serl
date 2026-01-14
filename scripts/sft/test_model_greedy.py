#!/usr/bin/env python3
"""
Quick test script for model inference with greedy decoding.

Usage:
  python scripts/sft/test_model_greedy.py \
    --model-path outputs/qwen3-4b-lora-no-reasoning \
    --test-file data/test.jsonl \
    --num-samples 10
"""

import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_model(model_path: str, base_model: str = None):
    """Load model (LoRA or full)."""
    model_path = Path(model_path)
    
    # Check if it's a LoRA adapter
    adapter_config = model_path / "adapter_config.json"
    if adapter_config.exists():
        with open(adapter_config) as f:
            config = json.load(f)
        base_model = base_model or config.get("base_model_name_or_path", "Qwen/Qwen3-4B")
        print(f"Loading base model: {base_model}")
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16, device_map="auto")
        print(f"Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
    else:
        print(f"Loading full model: {model_path}")
        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, device_map="auto")
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    return model, tokenizer


def load_prompts(prompts_dir: Path = None):
    """Load prompt configs."""
    if prompts_dir is None:
        prompts_dir = PROJECT_ROOT / "configs" / "prompts"
    
    prompt_file = prompts_dir / "error_detection_prompts.json"
    if prompt_file.exists():
        with open(prompt_file) as f:
            return json.load(f)
    return {"system_prompt": "", "user_template": "{note}"}


def run_inference(model, tokenizer, note: str, prompts: dict):
    """Run single inference with greedy decoding."""
    system_prompt = prompts.get("system_prompt", "")
    user_template = prompts.get("user_template", "{note}")
    user_content = user_template.replace("{note}", note)
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,  # Greedy decoding
            temperature=None,
            top_p=None,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return response


def parse_label(response: str) -> str:
    """Extract label from response."""
    response_lower = response.lower()
    if 'final_answer: "incorrect"' in response_lower or 'final_answer: "error"' in response_lower:
        return "INCORRECT"
    elif 'final_answer: "correct"' in response_lower:
        return "CORRECT"
    elif "incorrect" in response_lower or "error" in response_lower:
        return "INCORRECT"
    elif "correct" in response_lower:
        return "CORRECT"
    return "UNKNOWN"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--base-model", default=None)
    parser.add_argument("--test-file", required=True)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--prompts-dir", default=None)
    args = parser.parse_args()

    print("Loading model...")
    model, tokenizer = load_model(args.model_path, args.base_model)
    model.eval()
    
    prompts_dir = Path(args.prompts_dir) if args.prompts_dir else None
    prompts = load_prompts(prompts_dir)
    
    print(f"\nLoading test data from {args.test_file}...")
    with open(args.test_file) as f:
        test_data = [json.loads(line) for line in f]
    
    test_data = test_data[:args.num_samples]
    
    correct = 0
    total = 0
    
    print(f"\nRunning inference on {len(test_data)} samples (greedy decoding)...\n")
    print("="*80)
    
    for i, example in enumerate(test_data):
        note = example.get("note", "")
        ground_truth = example.get("label", example.get("ground_truth_label", "CORRECT")).strip().upper()
        if ground_truth == "ERROR":
            ground_truth = "INCORRECT"
        
        response = run_inference(model, tokenizer, note, prompts)
        predicted = parse_label(response)
        
        is_correct = predicted == ground_truth
        correct += int(is_correct)
        total += 1
        
        status = "✓" if is_correct else "✗"
        print(f"\n[{i+1}] {status} GT: {ground_truth} | Pred: {predicted}")
        print(f"Note: {note[:200]}...")
        print(f"Response: {response[:300]}...")
        print("-"*80)
    
    print("\n" + "="*80)
    print(f"RESULTS: {correct}/{total} correct ({100*correct/total:.1f}%)")
    print("="*80)


if __name__ == "__main__":
    main()
