#!/usr/bin/env python3
"""
LoRA SFT training for Qwen3-4B using existing CoT JSONL data.

Prompts loaded from configs/prompts/ directory.
Training data should be paired (correct/incorrect) for contrastive learning.

Example:
  python scripts/sft/train_qwen3_4b_sft_lora.py \
    --train-file data_processed/medec_cot/sft_cot_training_data.jsonl \
    --output-dir outputs/qwen3-4b-lora
"""

import argparse
import inspect
import json
import os
import re
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


DEFAULT_MODEL = "Qwen/Qwen3-4B"
PROJECT_ROOT = Path(__file__).resolve().parents[2]


# =============================================================================
# PROMPT LOADING
# =============================================================================

def load_prompt_config(prompt_file: Path) -> Dict:
    """Load prompts from a JSON config file."""
    if not prompt_file.exists():
        raise FileNotFoundError(f"Prompt file not found: {prompt_file}")
    with open(prompt_file, "r") as f:
        return json.load(f)


def get_prompt_configs(prompts_dir: Path = None) -> Dict[str, Dict]:
    """Load all prompt configurations from config files."""
    if prompts_dir is None:
        prompts_dir = PROJECT_ROOT / "configs" / "prompts"
    
    configs = {}
    
    # Load assessor/detector prompts
    assessor_file = prompts_dir / "error_detection_prompts.json"
    if assessor_file.exists():
        configs["assessor"] = load_prompt_config(assessor_file)
        print(f"✓ Loaded assessor prompts from {assessor_file}")
    else:
        raise FileNotFoundError(f"Required file not found: {assessor_file}")
    
    # Load injector prompts
    injector_file = prompts_dir / "error_injection_prompts_v2.json"
    if injector_file.exists():
        configs["injector"] = load_prompt_config(injector_file)
        print(f"✓ Loaded injector prompts from {injector_file}")
    else:
        print(f"Warning: {injector_file} not found, injector training disabled")
    
    return configs


# =============================================================================
# REASONING EXTRACTION
# =============================================================================

def extract_reasoning(text: str, skip_reasoning: bool = False) -> str:
    """Extract and clean reasoning content."""
    if not text or skip_reasoning:
        return ""
    
    # Try to extract from tags
    for tag in ["reasoning", "generation_reasoning", "error_injection_reasoning"]:
        pattern = rf'<{tag}>(.*?)</{tag}>'
        match = re.search(pattern, text, re.DOTALL)
        if match:
            return match.group(1).strip()
    
    # Remove any final answer blocks if present
    text = re.split(r'\n\s*\*\*Final Answer\*\*:', text, flags=re.IGNORECASE)[0]
    text = re.split(r'\n\s*Error Detected:', text, flags=re.IGNORECASE)[0]
    
    return text.strip()


# =============================================================================
# MESSAGE BUILDING
# =============================================================================

def build_messages(example: Dict, prompt_configs: Dict, skip_reasoning: bool = False) -> List[Dict[str, str]]:
    """Build chat messages using loaded prompt configs."""
    role = example.get("role", "critic")
    task = example.get("task", "assessment")
    note = example.get("note", "").strip()
    label = example.get("label", "CORRECT").strip().upper()
    reasoning = example.get("reasoning", "")
    
    is_error = label in ("ERROR", "INCORRECT")
    
    if role == "critic" and task == "assessment":
        # Use assessor config
        config = prompt_configs.get("assessor", {})
        system_prompt = config.get("system_prompt", "")
        user_template = config.get("user_template", "{note}")
        
        # Format user message
        user_content = user_template.replace("{note}", note)
        
        # Build assistant response with optional reasoning
        cleaned_reasoning = extract_reasoning(reasoning, skip_reasoning)
        if cleaned_reasoning:
            think_block = f"<think>\n{cleaned_reasoning}\n</think>"
            assistant_content = f'{think_block}\n\nfinal_answer: "{("INCORRECT" if is_error else "CORRECT")}"'
        else:
            # Simple response without reasoning
            assistant_content = f'final_answer: "{("INCORRECT" if is_error else "CORRECT")}"'
        
    else:
        # Use injector config
        config = prompt_configs.get("injector", {})
        system_prompt = config.get("system_prompt", "")
        
        correct_note = example.get("correct_note", "")
        source_note = correct_note.strip() if correct_note.strip() else note
        error_type = example.get("error_type", "")
        
        # Get appropriate user template
        if is_error:
            user_template = config.get("error_user_template", config.get("user_template", "{note}"))
            user_content = user_template.replace("{note}", source_note).replace("{error_type}", error_type)
        else:
            user_template = config.get("correct_user_template", config.get("user_template", "{note}"))
            user_content = user_template.replace("{note}", source_note)
        
        # Build assistant response
        cleaned_reasoning = extract_reasoning(reasoning, skip_reasoning)
        final_label = "INCORRECT" if is_error else "CORRECT"
        if cleaned_reasoning:
            think_block = f"<think>\n{cleaned_reasoning}\n</think>"
            assistant_content = f'{think_block}\n\ngenerated_note:\n{note}\n\nfinal_answer: "{final_label}"'
        else:
            assistant_content = f'generated_note:\n{note}\n\nfinal_answer: "{final_label}"'
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def format_for_sft(example: Dict, tokenizer, prompt_configs: Dict, skip_reasoning: bool = False) -> Dict[str, str]:
    """Format example for SFT training."""
    messages = build_messages(example, prompt_configs, skip_reasoning)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


# =============================================================================
# DATA ANALYSIS
# =============================================================================

def analyze_dataset(dataset, role_filter: str = None):
    """Print dataset statistics."""
    if role_filter:
        dataset = dataset.filter(lambda ex: ex.get("role") == role_filter)
    
    total = len(dataset)
    labels = [ex.get("label", "CORRECT").strip().upper() for ex in dataset]
    
    error_count = sum(1 for l in labels if l in ("ERROR", "INCORRECT"))
    correct_count = sum(1 for l in labels if l == "CORRECT")
    
    print(f"\nDataset Statistics:")
    print(f"  Total examples: {total}")
    print(f"  CORRECT: {correct_count} ({100*correct_count/total:.1f}%)")
    print(f"  ERROR/INCORRECT: {error_count} ({100*error_count/total:.1f}%)")
    
    return dataset


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT LoRA training for Qwen3-4B.")
    parser.add_argument("--train-file", required=True, help="Path to training JSONL.")
    parser.add_argument("--eval-file", default=None, help="Optional eval JSONL.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--prompts-dir", default=None, help="Prompts config directory.")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--num-train-epochs", type=int, default=2)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=50)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj")
    parser.add_argument("--filter-role", default=None, choices=["critic", "generator"])
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    # New: skip reasoning to test if it's the source of the problem
    parser.add_argument("--skip-reasoning", action="store_true",
                        help="Skip GPT-4o reasoning in training (train on labels only)")
    parser.add_argument("--debug-samples", type=int, default=0,
                        help="Print N training samples for debugging")
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    print("\n" + "="*70)
    print("SFT LORA TRAINING")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Training: epochs={args.num_train_epochs}, lr={args.learning_rate}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}")
    if args.skip_reasoning:
        print("⚠️  SKIP REASONING MODE: Training without GPT-4o reasoning")
    print("="*70)

    # Load prompts from config files
    prompts_dir = Path(args.prompts_dir) if args.prompts_dir else None
    prompt_configs = get_prompt_configs(prompts_dir)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load and analyze dataset
    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file

    dataset = load_dataset("json", data_files=data_files)
    
    train_data = analyze_dataset(dataset["train"], args.filter_role)
    if args.filter_role:
        train_data = train_data.filter(lambda ex: ex.get("role") == args.filter_role)
    
    # Format for training
    train_ds = train_data.map(
        lambda ex: format_for_sft(ex, tokenizer, prompt_configs, args.skip_reasoning), 
        remove_columns=train_data.column_names
    )
    
    eval_ds = None
    if "validation" in dataset:
        eval_data = dataset["validation"]
        if args.filter_role:
            eval_data = eval_data.filter(lambda ex: ex.get("role") == args.filter_role)
        eval_ds = eval_data.map(
            lambda ex: format_for_sft(ex, tokenizer, prompt_configs, args.skip_reasoning),
            remove_columns=eval_data.column_names
        )

    # Debug: show samples
    n_debug = args.debug_samples if args.debug_samples > 0 else 1
    print(f"\nSample training examples (showing {n_debug}):")
    print("-"*70)
    for i in range(min(n_debug, len(train_ds))):
        print(f"\n--- Sample {i+1} ---")
        print(train_ds[i]["text"][-800:])
    print("-"*70)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto")

    # LoRA config
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
    )

    # Training config
    sft_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps if eval_ds else None,
        "evaluation_strategy": "steps" if eval_ds else "no",
        "bf16": args.bf16,
        "fp16": args.fp16,
        "save_total_limit": 3,
        "load_best_model_at_end": bool(eval_ds),
        "metric_for_best_model": "eval_loss" if eval_ds else None,
        "report_to": "none",
        "lr_scheduler_type": "cosine",
        "dataloader_num_workers": args.dataloader_num_workers,
    }
    
    sig = inspect.signature(SFTConfig.__init__)
    sft_kwargs["max_seq_length" if "max_seq_length" in sig.parameters else "max_length"] = args.max_seq_length
    training_args = SFTConfig(**{k: v for k, v in sft_kwargs.items() if k in sig.parameters})

    # Create trainer
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "processing_class": tokenizer,
        "args": training_args,
        "peft_config": peft_config,
    }
    trainer_sig = inspect.signature(SFTTrainer.__init__)
    trainer = SFTTrainer(**{k: v for k, v in trainer_kwargs.items() if k in trainer_sig.parameters})

    # Train
    print("\nStarting training...")
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✓ Model saved to {args.output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
