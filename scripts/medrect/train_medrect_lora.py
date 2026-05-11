#!/usr/bin/env python3
"""
LoRA SFT training for Qwen3-4B on MEDRECT detection-only task.

Stage 0 of the MedSeRL pipeline: ground the model in medical error
localization before MedPRM chain training (Stage 1) and self-play RL (Stage 2).

Task: Given numbered clinical sentences, output CORRECT or the sentence number
containing the error. Reasoning goes inside <think>...</think> tags.

Prompts are loaded from configs/prompts/sft/medrect_detection_prompts.json
at runtime — nothing is hardcoded.

Example:
    python scripts/medrect/train_medrect_lora.py \\
        --train-file data_processed/medrect/medrect_sft_train.jsonl \\
        --output-dir outputs/local_training/qwen3-4b-medrect-sft

    # Quick test run
    python scripts/medrect/train_medrect_lora.py \\
        --train-file data_processed/medrect/medrect_sft_train.jsonl \\
        --output-dir /tmp/medrect_test \\
        --num-train-epochs 1 --limit 50 --debug-samples 3
"""

import argparse
import inspect
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_CONFIG = PROJECT_ROOT / "configs" / "prompts" / "sft" / "medrect_detection_prompts.json"
DEFAULT_DATA_FILE = PROJECT_ROOT / "data_processed" / "medrect" / "medrect_sft_train.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "local_training" / "qwen3-4b-medrect-sft"


# =============================================================================
# DATA LOADING
# =============================================================================

def load_sft_data(data_file: str, limit: Optional[int] = None) -> Dataset:
    """Load prepared MEDRECT SFT JSONL data.
    
    Supports comma-separated file lists for combining datasets:
        --train-file "medrect_en.jsonl,medrect_ja.jsonl"
    """
    raw_paths = [p.strip() for p in data_file.split(",") if p.strip()]
    
    records = []
    for path in raw_paths:
        print(f"  Loading: {path}")
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    
    if limit:
        records = records[:limit]
    
    dataset = Dataset.from_list(records)
    print(f"  Loaded {len(dataset)} examples from {len(raw_paths)} file(s)")
    return dataset


# =============================================================================
# FORMATTING FOR SFT
# =============================================================================

def format_for_sft(example: Dict, tokenizer) -> Dict[str, str]:
    """Format a prepared MEDRECT record for SFT training.
    
    The system_prompt and user_prompt are already set in the JSONL
    (populated from the prompt config at prepare time).
    
    Response format follows Qwen3 conventions:
        <think>
        {reasoning if available}
        </think>
        
        {label}
    
    Supported label formats:
        - Assessor: "CORRECT" or "{sentence_number}" (detection + localization)
        - Injector: "N. <modified sentence>" (error injection / benign change)
    Both are used as-is after the </think> tag.
    """
    system_prompt = example["system_prompt"]
    user_prompt = example["user_prompt"]
    label = example["label"]
    reasoning = example.get("reasoning")
    
    # Build assistant response with Qwen3 think tags
    if reasoning:
        assistant_content = f"<think>\n{reasoning}\n</think>\n\n{label}"
    else:
        # Empty think block per Qwen3 convention (signals "no reasoning mode")
        assistant_content = f"<think>\n\n</think>\n\n{label}"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_content},
    ]
    
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    return {"text": text}


# =============================================================================
# DATASET ANALYSIS
# =============================================================================

def analyze_dataset(dataset: Dataset) -> None:
    """Print dataset statistics."""
    total = len(dataset)
    
    # Label distribution
    labels = Counter(ex.get("label", "UNKNOWN") for ex in dataset)
    correct = labels.get("CORRECT", 0)
    error = total - correct
    
    # Language distribution
    langs = Counter(ex.get("language", "unknown") for ex in dataset)
    
    # Error type distribution
    error_types = Counter(
        ex.get("error_type") for ex in dataset if ex.get("error_type")
    )
    
    # Reasoning availability
    has_reasoning = sum(1 for ex in dataset if ex.get("reasoning"))
    
    print(f"\n{'='*60}")
    print(f"  Dataset Statistics")
    print(f"{'='*60}")
    print(f"  Total examples:      {total}")
    print(f"  CORRECT (no error):  {correct} ({100*correct/total:.1f}%)")
    print(f"  Error detected:      {error} ({100*error/total:.1f}%)")
    print(f"  Has reasoning:       {has_reasoning} ({100*has_reasoning/total:.1f}%)")
    
    print(f"\n  By language:")
    for lang, count in sorted(langs.items()):
        print(f"    {lang}: {count} ({100*count/total:.1f}%)")
    
    if error_types:
        print(f"\n  By error type (top 10):")
        for etype, count in error_types.most_common(10):
            print(f"    {etype}: {count}")
    
    # Unique sentence IDs in labels (for error examples)
    sentence_ids = [
        int(ex["label"]) for ex in dataset 
        if ex.get("label", "CORRECT") != "CORRECT" and ex["label"].isdigit()
    ]
    if sentence_ids:
        print(f"\n  Error sentence IDs: min={min(sentence_ids)}, max={max(sentence_ids)}, "
              f"mean={sum(sentence_ids)/len(sentence_ids):.1f}")
    
    print(f"{'='*60}\n")


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MEDRECT detection-only LoRA SFT training for Qwen3."
    )
    
    # Data arguments
    parser.add_argument(
        "--train-file", default=str(DEFAULT_DATA_FILE),
        help="Path to prepared SFT JSONL (comma-separated for multiple files)"
    )
    parser.add_argument(
        "--eval-split", type=float, default=0.05,
        help="Fraction of data for evaluation (default: 0.05)"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit total records (for quick testing)"
    )
    
    # Model arguments
    parser.add_argument(
        "--model-name", default="Qwen/Qwen3-4B",
        help="HuggingFace model name or local path (default: Qwen/Qwen3-4B)"
    )
    parser.add_argument(
        "--output-dir", default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for adapter and config"
    )
    
    # Training hyperparameters (paper defaults: lr=1e-4, r=64, alpha=128)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument(
        "--per-device-train-batch-size", type=int, default=2,
        help="Batch size per GPU (default: 2 for 4B model)"
    )
    parser.add_argument(
        "--gradient-accumulation-steps", type=int, default=8,
        help="Gradient accumulation (default: 8, effective batch=16)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4,
        help="Learning rate (default: 1e-4, per MEDRECT paper)"
    )
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", action="store_false", dest="bf16")
    
    # LoRA arguments (paper: r=64, alpha=128 for Qwen3-32B; same for 8B)
    parser.add_argument(
        "--lora-r", type=int, default=64,
        help="LoRA rank (default: 64, per MEDRECT paper)"
    )
    parser.add_argument(
        "--lora-alpha", type=int, default=128,
        help="LoRA alpha (default: 128, per MEDRECT paper)"
    )
    parser.add_argument(
        "--lora-dropout", type=float, default=0.05,
        help="LoRA dropout (default: 0.05)"
    )
    parser.add_argument(
        "--lora-target-modules", default="all-linear",
        help="LoRA target modules (default: all-linear, Qwen3 recommendation)"
    )
    
    # Early stopping
    parser.add_argument("--early-stopping-patience", type=int, default=3)
    parser.add_argument("--no-early-stopping", action="store_true")
    
    # Wandb
    parser.add_argument("--wandb", action="store_true", help="Enable wandb logging")
    parser.add_argument(
        "--wandb-project", default="medrect-sft",
        help="Wandb project name (default: medrect-sft)"
    )
    
    # Misc
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--debug-samples", type=int, default=1,
        help="Print N formatted training samples before training (default: 1)"
    )
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()
    
    print("\n" + "=" * 70)
    print("  MEDRECT DETECTION-ONLY LORA SFT TRAINING")
    print("=" * 70)
    print(f"Model:       {args.model_name}")
    print(f"Output:      {args.output_dir}")
    print(f"Data:        {args.train_file}")
    print(f"Training:    epochs={args.num_train_epochs}, lr={args.learning_rate}")
    print(f"Batch:       size={args.per_device_train_batch_size}, "
          f"grad_accum={args.gradient_accumulation_steps}, "
          f"effective={args.per_device_train_batch_size * args.gradient_accumulation_steps}")
    print(f"LoRA:        r={args.lora_r}, alpha={args.lora_alpha}, "
          f"dropout={args.lora_dropout}, target={args.lora_target_modules}")
    print(f"Early stop:  {'disabled' if args.no_early_stopping else f'patience={args.early_stopping_patience}'}")
    print(f"Eval split:  {args.eval_split * 100:.1f}%")
    print(f"Wandb:       {'enabled' if args.wandb else 'disabled'}")
    print("=" * 70)
    
    # Load tokenizer
    print(f"\nLoading tokenizer from {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, use_fast=True, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    print("\nLoading dataset...")
    dataset = load_sft_data(args.train_file, limit=args.limit)
    analyze_dataset(dataset)
    
    # Train/eval split
    if args.eval_split > 0:
        split = dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"Split: {len(train_dataset)} train, {len(eval_dataset)} eval")
    else:
        train_dataset = dataset
        eval_dataset = None
    
    # Format for SFT
    print("\nFormatting for SFT...")
    train_ds = train_dataset.map(
        lambda ex: format_for_sft(ex, tokenizer),
        remove_columns=train_dataset.column_names,
    )
    
    eval_ds = None
    if eval_dataset is not None:
        eval_ds = eval_dataset.map(
            lambda ex: format_for_sft(ex, tokenizer),
            remove_columns=eval_dataset.column_names,
        )
    
    # Debug: show formatted samples
    if args.debug_samples > 0:
        n = min(args.debug_samples, len(train_ds))
        print(f"\nFormatted training samples (showing {n}):")
        print("-" * 70)
        for i in range(n):
            sample = train_ds[i]["text"]
            print(f"\n--- Sample {i+1} (last 800 chars) ---")
            print(sample[-800:])
        print("-" * 70)
    
    # Load model
    print(f"\nLoading model {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        trust_remote_code=True,
    )
    # Trainer-side activation checkpointing reduces memory substantially for 4k sequences.
    model.config.use_cache = False
    
    # LoRA config
    if args.lora_target_modules.lower() == "all-linear":
        target_modules = "all-linear"
    else:
        target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    
    # Training config
    has_eval = eval_ds is not None
    
    report_to = "none"
    if args.wandb:
        report_to = "wandb"
        os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    
    sft_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_train_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps if has_eval else None,
        "eval_strategy": "steps" if has_eval else "no",
        "bf16": args.bf16,
        "save_total_limit": None,
        "load_best_model_at_end": has_eval and not args.no_early_stopping,
        "metric_for_best_model": "eval_loss" if has_eval else None,
        "greater_is_better": False,
        "report_to": report_to,
        "lr_scheduler_type": "cosine",
        "dataloader_num_workers": args.dataloader_num_workers,
        "seed": args.seed,
        "gradient_checkpointing": True,
        "ddp_find_unused_parameters": False,
    }
    
    # Handle different TRL versions (max_seq_length vs max_length)
    sig = inspect.signature(SFTConfig.__init__)
    if "max_seq_length" in sig.parameters:
        sft_kwargs["max_seq_length"] = args.max_seq_length
    elif "max_length" in sig.parameters:
        sft_kwargs["max_length"] = args.max_seq_length
    if "gradient_checkpointing_kwargs" in sig.parameters:
        sft_kwargs["gradient_checkpointing_kwargs"] = {"use_reentrant": False}
    
    training_args = SFTConfig(**{k: v for k, v in sft_kwargs.items() if k in sig.parameters})
    
    # Callbacks
    callbacks = []
    if has_eval and not args.no_early_stopping:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience)
        )
        print(f"Early stopping enabled (patience={args.early_stopping_patience})")
    
    # Create trainer
    trainer_kwargs = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "processing_class": tokenizer,
        "args": training_args,
        "peft_config": peft_config,
        "callbacks": callbacks if callbacks else None,
    }
    trainer_sig = inspect.signature(SFTTrainer.__init__)
    trainer = SFTTrainer(
        **{k: v for k, v in trainer_kwargs.items() if k in trainer_sig.parameters}
    )
    
    # Train
    print("\nStarting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nModel saved to {args.output_dir}")
    
    # Save training config for reproducibility
    config_path = Path(args.output_dir) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2, default=str)
    print(f"Training config saved to {config_path}")
    
    # Save prompt config reference (so we know which prompts were used)
    prompt_ref_path = Path(args.output_dir) / "prompt_config_used.json"
    prompt_config = load_prompt_config_for_reference(args.train_file)
    if prompt_config:
        with open(prompt_ref_path, "w") as f:
            json.dump(prompt_config, f, indent=2, ensure_ascii=False)
        print(f"Prompt config reference saved to {prompt_ref_path}")


def load_prompt_config_for_reference(train_file: str) -> Optional[Dict]:
    """Extract system_prompt from first record to save as reference."""
    first_path = train_file.split(",")[0].strip()
    try:
        with open(first_path, "r") as f:
            first_line = f.readline().strip()
            if first_line:
                rec = json.loads(first_line)
                return {
                    "system_prompt": rec.get("system_prompt", ""),
                    "note": "Extracted from first training record for reference",
                }
    except Exception:
        pass
    return None


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
