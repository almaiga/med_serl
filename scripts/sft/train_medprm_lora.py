#!/usr/bin/env python3
"""
LoRA SFT training for Qwen3 models using Med-PRM chains data.

Supports:
- Qwen3-4B and Qwen3-8B
- Med-PRM chain format (input_note, reasoning_chain, label)
- Early stopping with eval split
- Auto-detection of latest training data file

Example:
  python scripts/sft/train_medprm_lora.py \
    --output-dir outputs/local_training/qwen3-4b-medprm-lora \
    --model-size 4b
"""

import argparse
import glob
import inspect
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer


# Model configurations
MODEL_CONFIGS = {
    "4b": {
        "name": "Qwen/Qwen3-4B",
        "batch_size": 2,
        "grad_accum": 8,
    },
    "8b": {
        "name": "Qwen/Qwen3-8B", 
        "batch_size": 1,
        "grad_accum": 16,
    },
}

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data_processed" / "medprm_chains"


# =============================================================================
# DATA LOADING
# =============================================================================

def find_latest_medprm_file(data_dir: Path = DEFAULT_DATA_DIR) -> Optional[Path]:
    """Find the latest medprm_chains JSONL file by timestamp in filename."""
    pattern = str(data_dir / "medprm_chains_*.jsonl")
    files = glob.glob(pattern)
    
    if not files:
        return None
    
    # Sort by filename (timestamp is in format YYYYMMDD_HHMMSS)
    files.sort(reverse=True)
    return Path(files[0])


def load_medprm_data(data_file: Optional[str] = None) -> Dataset:
    """Load Med-PRM chains data, auto-detecting latest file if not specified.
    
    Supports multiple files via comma-separated paths or glob patterns:
      --train-file "assessor_chains.jsonl,injector_chains.jsonl"
      --train-file "data_processed/medprm_chains/medprm_chains_*.jsonl"
    """
    if data_file:
        # Support comma-separated file lists
        raw_paths = [p.strip() for p in data_file.split(",") if p.strip()]
        # Expand globs
        data_paths = []
        for p in raw_paths:
            expanded = glob.glob(p)
            if expanded:
                data_paths.extend(expanded)
            else:
                data_paths.append(p)  # Keep as-is if no glob match (will error if missing)
    else:
        latest = find_latest_medprm_file()
        if latest is None:
            raise FileNotFoundError(
                f"No medprm_chains_*.jsonl files found in {DEFAULT_DATA_DIR}. "
                "Please specify --train-file explicitly."
            )
        data_paths = [str(latest)]
    
    for dp in data_paths:
        print(f"📂 Loading data from: {dp}")
    
    # Manually load JSONL to avoid schema inference conflicts across files
    # (e.g. change_type is null in assessor files but a string in benign files,
    # which causes HuggingFace datasets to crash with "Couldn't cast string to null")
    records = []
    for dp in data_paths:
        with open(dp, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    dataset = Dataset.from_list(records)
    
    # Filter to valid chains only
    original_len = len(dataset)
    dataset = dataset.filter(lambda x: x.get("is_valid", True))
    print(f"✓ Loaded {len(dataset)} valid chains from {len(data_paths)} file(s) "
          f"(filtered {original_len - len(dataset)} invalid)")
    
    return dataset


# =============================================================================
# PROMPT CONFIGURATION
# =============================================================================

def load_inference_prompts() -> Dict[str, str]:
    """Load prompts from the same config file used for inference.
    
    This ensures training and inference use identical prompts.
    """
    prompt_file = PROJECT_ROOT / "configs" / "prompts" / "error_detection_prompts.json"
    
    if prompt_file.exists():
        with open(prompt_file, 'r') as f:
            prompts = json.load(f)
        return prompts
    else:
        # Fallback prompts matching inference defaults
        return {
            "system_prompt": "You are a medical note classifier.\n\nClassify each note as CORRECT or INCORRECT.\n\nCORRECT = No medical errors\nINCORRECT = Contains medical error (wrong diagnosis, treatment, or management)\n\nYou must respond with EXACTLY this format:\nfinal_answer: \"CORRECT\"\nExplanation: [one sentence]\n\nOR\n\nfinal_answer: \"INCORRECT\"\nExplanation: [one sentence why it's wrong]",
            "user_template": "Classify this note:\n\n{note}\n\nRespond in the required format:"
        }


def get_system_prompt() -> str:
    """Get the system prompt - same as inference."""
    prompts = load_inference_prompts()
    return prompts.get("system_prompt", "You are a medical note classifier.")


# Module-level cache to avoid re-reading the file per training example
_injector_prompts_cache: Optional[Dict] = None


def load_injector_prompts() -> Dict:
    """Load injector SFT prompts from config (cached after first call)."""
    global _injector_prompts_cache
    if _injector_prompts_cache is not None:
        return _injector_prompts_cache
    
    prompt_file = PROJECT_ROOT / "configs" / "prompts" / "sft" / "injector_sft_prompts.json"
    if prompt_file.exists():
        with open(prompt_file, 'r') as f:
            _injector_prompts_cache = json.load(f)
        return _injector_prompts_cache
    else:
        raise FileNotFoundError(
            f"Injector SFT prompts not found: {prompt_file}. "
            "Create configs/prompts/sft/injector_sft_prompts.json"
        )


def build_user_prompt(input_note: str, scenario: str, error_type: Optional[str] = None,
                      error_sentence: Optional[str] = None, corrected_sentence: Optional[str] = None) -> str:
    """Build user prompt - same format as inference.
    
    NOTE: We do NOT include ground truth hints during training.
    The model must learn to reason without being given the answer.
    """
    prompts = load_inference_prompts()
    user_template = prompts.get("user_template", "Classify this note:\n\n{note}\n\nRespond in the required format:")
    
    # Use the same template for both correct and incorrect notes
    # No hints about whether the note is correct or incorrect!
    return user_template.format(note=input_note)


# =============================================================================
# FORMATTING FOR SFT
# =============================================================================

def format_for_sft(example: Dict, tokenizer, include_thinking: bool = True) -> Dict[str, str]:
    """Format Med-PRM chain example for SFT training.
    
    Role-aware formatting:
    - Assessor: "Classify this note" → reasoning → CORRECT/INCORRECT
    - Injector error: "Introduce a subtle error" → cognitive trap analysis → change
    - Injector benign: "Make a safe change" → semantic equivalence reasoning → change
    """
    input_note = example.get("input_note", "")
    reasoning_chain = example.get("reasoning_chain", "")
    role = example.get("role", "assessor")
    scenario = example.get("scenario", "correct")
    error_type = example.get("error_type")
    error_sentence = example.get("error_sentence")
    corrected_sentence = example.get("corrected_sentence")
    
    # Build role-specific messages
    if role == "injector" and scenario == "error":
        injector_prompts = load_injector_prompts()
        system_prompt = injector_prompts["injector_error"]["system_prompt"]
        user_template = injector_prompts["injector_error"]["user_template"]
        user_prompt = user_template.format(
            note=input_note,
            error_type=error_type or "any"
        )
    elif role == "injector" and scenario == "benign":
        injector_prompts = load_injector_prompts()
        benign_cfg = injector_prompts["injector_benign"]
        system_prompt = benign_cfg["system_prompt"]
        # Get change_type from the chain data (stored during generation)
        change_type = example.get("change_type", "pseudo_factual")
        # Look up the description for this change type
        type_descriptions = benign_cfg.get("change_type_descriptions", {})
        change_type_desc = type_descriptions.get(
            change_type,
            f"Make a {change_type} change that preserves clinical meaning."
        )
        user_prompt = benign_cfg["user_template"].format(
            note=input_note,
            change_type=change_type,
            change_type_description=change_type_desc
        )
    else:
        # Assessor scenarios (correct + incorrect) — use detection prompt
        system_prompt = get_system_prompt()
        user_prompt = build_user_prompt(
            input_note, scenario, error_type, error_sentence, corrected_sentence
        )
    
    # Split reasoning_chain: reasoning goes inside <think>, final_answer outside
    # Per Qwen3 best practices: <think>\nreasoning\n</think>\n\nfinal_answer
    
    # Extract the final_answer portion (final_answer: "X"\nExplanation: Y)
    final_answer_match = re.search(
        r'(final_answer:.*)$',
        reasoning_chain,
        re.DOTALL | re.IGNORECASE
    )
    
    if final_answer_match:
        # Split into reasoning (before final_answer) and answer (final_answer onwards)
        reasoning_only = reasoning_chain[:final_answer_match.start()].strip()
        final_answer_part = final_answer_match.group(1).strip()
    else:
        # Fallback: no final_answer found, use entire chain as reasoning
        reasoning_only = reasoning_chain.strip()
        final_answer_part = ""
    
    # Format per Qwen3 best practices: <think>\nreasoning\n</think>\n\nfinal_answer
    if include_thinking:
        if final_answer_part:
            # Proper Qwen3 format with answer OUTSIDE think tags
            assistant_content = f"<think>\n{reasoning_only}\n</think>\n\n{final_answer_part}"
        else:
            # Fallback to old behavior if no final_answer found
            assistant_content = f"<think>\n{reasoning_chain}\n</think>"
    else:
        # For non-thinking mode, output empty think tags per Qwen3 recommendation
        if final_answer_part:
            assistant_content = f"<think>\n\n</think>\n\n{final_answer_part}"
        else:
            assistant_content = reasoning_chain
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": assistant_content},
    ]
    
    # Apply chat template
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    return {"text": text}


# =============================================================================
# DATA ANALYSIS
# =============================================================================

def analyze_dataset(dataset: Dataset) -> None:
    """Print dataset statistics."""
    total = len(dataset)
    
    # Count by label
    labels = [ex.get("label", "UNKNOWN") for ex in dataset]
    correct = sum(1 for l in labels if l == "CORRECT")
    incorrect = sum(1 for l in labels if l == "INCORRECT")
    
    # Count by scenario
    scenarios = [ex.get("scenario", "unknown") for ex in dataset]
    scenario_counts = {}
    for s in scenarios:
        scenario_counts[s] = scenario_counts.get(s, 0) + 1
    
    # Count by role
    roles = [ex.get("role", "unknown") for ex in dataset]
    role_counts = {}
    for r in roles:
        role_counts[r] = role_counts.get(r, 0) + 1
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total examples: {total}")
    print(f"   CORRECT: {correct} ({100*correct/total:.1f}%)")
    print(f"   INCORRECT: {incorrect} ({100*incorrect/total:.1f}%)")
    print(f"   By scenario: {scenario_counts}")
    print(f"   By role: {role_counts}")
    
    # Token estimates
    token_estimates = [ex.get("token_estimate", 0) for ex in dataset if ex.get("token_estimate")]
    if token_estimates:
        avg_tokens = sum(token_estimates) / len(token_estimates)
        print(f"   Avg reasoning tokens: {avg_tokens:.0f}")


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Med-PRM LoRA SFT training for Qwen3.")
    
    # Data arguments
    parser.add_argument("--train-file", default=None, 
                        help="Path to training JSONL (auto-detects latest if not specified)")
    parser.add_argument("--eval-split", type=float, default=0.05,
                        help="Fraction of data for evaluation (default: 0.05 = 5%%)")
    
    # Model arguments
    parser.add_argument("--model-size", default="4b", choices=["4b", "8b"],
                        help="Qwen3 model size: 4b or 8b")
    parser.add_argument("--model-name", default=None,
                        help="Override model name (default: based on --model-size)")
    parser.add_argument("--output-dir", required=True, help="Output directory")
    
    # Training arguments
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--per-device-train-batch-size", type=int, default=None,
                        help="Batch size (default: auto based on model size)")
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None,
                        help="Gradient accumulation (default: auto based on model size)")
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4, lower for small dataset)")
    parser.add_argument("--num-train-epochs", type=int, default=3,
                        help="Number of epochs (default: 3)")
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=50)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--no-bf16", action="store_false", dest="bf16")
    
    # LoRA arguments
    parser.add_argument("--lora-r", type=int, default=16,
                        help="LoRA rank (default: 16)")
    parser.add_argument("--lora-alpha", type=int, default=32,
                        help="LoRA alpha (default: 32)")
    parser.add_argument("--lora-dropout", type=float, default=0.05,
                        help="LoRA dropout (default: 0.05, lower for small dataset)")
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj")
    
    # Early stopping
    parser.add_argument("--early-stopping-patience", type=int, default=3,
                        help="Early stopping patience (default: 3)")
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Disable early stopping")
    
    # Misc
    parser.add_argument("--no-thinking-tags", action="store_true",
                        help="Don't wrap reasoning in <think> tags")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--debug-samples", type=int, default=1,
                        help="Print N training samples for debugging")
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()
    
    # Get model config
    model_config = MODEL_CONFIGS[args.model_size]
    model_name = args.model_name or model_config["name"]
    batch_size = args.per_device_train_batch_size or model_config["batch_size"]
    grad_accum = args.gradient_accumulation_steps or model_config["grad_accum"]
    
    print("\n" + "="*70)
    print("🏥 MED-PRM LORA SFT TRAINING")
    print("="*70)
    print(f"Model: {model_name}")
    print(f"Output: {args.output_dir}")
    print(f"Training: epochs={args.num_train_epochs}, lr={args.learning_rate}")
    print(f"Batch: size={batch_size}, grad_accum={grad_accum}, effective={batch_size * grad_accum}")
    print(f"LoRA: r={args.lora_r}, alpha={args.lora_alpha}, dropout={args.lora_dropout}")
    print(f"Early stopping: {'disabled' if args.no_early_stopping else f'patience={args.early_stopping_patience}'}")
    print(f"Eval split: {args.eval_split * 100:.1f}%")
    print("="*70)
    
    # Load tokenizer
    print(f"\n📦 Loading tokenizer from {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load and analyze dataset
    dataset = load_medprm_data(args.train_file)
    analyze_dataset(dataset)
    
    # Split into train/eval
    if args.eval_split > 0:
        split = dataset.train_test_split(test_size=args.eval_split, seed=args.seed)
        train_dataset = split["train"]
        eval_dataset = split["test"]
        print(f"\n✂️  Split: {len(train_dataset)} train, {len(eval_dataset)} eval")
    else:
        train_dataset = dataset
        eval_dataset = None
    
    # Format for training
    include_thinking = not args.no_thinking_tags
    train_ds = train_dataset.map(
        lambda ex: format_for_sft(ex, tokenizer, include_thinking),
        remove_columns=train_dataset.column_names
    )
    
    eval_ds = None
    if eval_dataset is not None:
        eval_ds = eval_dataset.map(
            lambda ex: format_for_sft(ex, tokenizer, include_thinking),
            remove_columns=eval_dataset.column_names
        )
    
    # Debug: show samples
    n_debug = args.debug_samples
    if n_debug > 0:
        print(f"\n📝 Sample training examples (showing {n_debug}):")
        print("-" * 70)
        for i in range(min(n_debug, len(train_ds))):
            sample_text = train_ds[i]["text"]
            # Show last 1000 chars to see the response
            print(f"\n--- Sample {i+1} (last 1000 chars) ---")
            print(sample_text[-1000:])
        print("-" * 70)
    
    # Load model
    print(f"\n🔧 Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        torch_dtype="auto",
        trust_remote_code=True
    )
    
    # LoRA config
    # Handle target_modules: can be "all-linear" or comma-separated list
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
    sft_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": batch_size,
        "gradient_accumulation_steps": grad_accum,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "eval_steps": args.eval_steps if has_eval else None,
        "eval_strategy": "steps" if has_eval else "no",
        "bf16": args.bf16,
        "save_total_limit": 3,
        "load_best_model_at_end": has_eval and not args.no_early_stopping,
        "metric_for_best_model": "eval_loss" if has_eval else None,
        "greater_is_better": False,
        "report_to": "none",
        "lr_scheduler_type": "cosine",
        "dataloader_num_workers": args.dataloader_num_workers,
        "seed": args.seed,
    }
    
    # Handle different TRL versions
    sig = inspect.signature(SFTConfig.__init__)
    if "max_seq_length" in sig.parameters:
        sft_kwargs["max_seq_length"] = args.max_seq_length
    elif "max_length" in sig.parameters:
        sft_kwargs["max_length"] = args.max_seq_length
    
    training_args = SFTConfig(**{k: v for k, v in sft_kwargs.items() if k in sig.parameters})
    
    # Callbacks
    callbacks = []
    if has_eval and not args.no_early_stopping:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))
        print(f"✓ Early stopping enabled (patience={args.early_stopping_patience})")
    
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
    trainer = SFTTrainer(**{k: v for k, v in trainer_kwargs.items() if k in trainer_sig.parameters})
    
    # Train
    print("\n🚀 Starting training...")
    trainer.train()
    
    # Save final model
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\n✅ Model saved to {args.output_dir}")
    
    # Save training config for reference
    config_path = Path(args.output_dir) / "training_config.json"
    with open(config_path, "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"✓ Training config saved to {config_path}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
