#!/usr/bin/env python3
"""
LoRA SFT training for Qwen3-4B using existing CoT JSONL data.

Trains a model for medical self-play loop:
- Assessor role: Detect errors in clinical notes
- Injector role: Inject subtle errors or preserve correctness

Example:
  python scripts/sft/train_qwen3_4b_sft_lora.py \
    --train-file data_processed/medec_cot/sft_cot_training_data.jsonl \
    --output-dir outputs/qwen3-4b-lora
"""

import argparse
import inspect
import os
import re
from typing import Dict, List

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


DEFAULT_MODEL = "Qwen/Qwen3-4B"


# =============================================================================
# SYSTEM PROMPTS - Aligned with inference expectations
# =============================================================================

SYSTEM_PROMPTS = {
    ("critic", "assessment"): (
        "You are a medical note assessor. Your task is to analyze clinical notes "
        "for errors. Think through your analysis step by step, then provide your "
        "final verdict.\n\n"
        "Output format:\n"
        "<think>\n[Your step-by-step reasoning]\n</think>\n\n"
        "Label: CORRECT or INCORRECT\n"
        "Explanation: [Brief explanation]"
    ),
    ("generator", "generation"): (
        "You are a medical note editor in a self-play training loop. Your task is "
        "to either preserve a note's correctness with minimal surface edits, or "
        "inject a subtle clinical error. Think through your approach step by step.\n\n"
        "Output format:\n"
        "<think>\n[Your reasoning for the edit]\n</think>\n\n"
        "Generated note:\n[The edited note]\n\n"
        "Label: CORRECT or INCORRECT"
    ),
}


# =============================================================================
# REASONING EXTRACTION AND CLEANING
# =============================================================================

def extract_reasoning_content(text: str) -> str:
    """
    Extract only the reasoning content from GPT-4o output.
    
    GPT-4o outputs look like:
    ```
    <reasoning>
    1. Clinical Picture:
       ...detailed analysis...
    4. Final Verdict:
       - Status: ...
    </reasoning>
    
    **Final Answer**:
    Error Detected: Yes
    ...
    ```
    
    We want ONLY the content inside the tags, stopping before "Final Verdict"
    or any answer-like content to keep reasoning pure.
    """
    if not text:
        return ""
    
    # Define patterns for different tag types
    tag_patterns = [
        (r'<reasoning>(.*?)</reasoning>', 'reasoning'),
        (r'<generation_reasoning>(.*?)</generation_reasoning>', 'generation'),
        (r'<error_injection_reasoning>(.*?)</error_injection_reasoning>', 'injection'),
    ]
    
    extracted = ""
    
    for pattern, _ in tag_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            break
    
    if not extracted:
        # No tags found - try to use raw text but clean it
        extracted = text
    
    # Remove any "Final Verdict" or answer sections that leaked into reasoning
    # These patterns mark where GPT-4o starts concluding
    cutoff_patterns = [
        r'\n\s*\d+\.\s*Final Verdict:.*',  # "4. Final Verdict: ..."
        r'\n\s*Final Verdict:.*',
        r'\n\s*\*\*Final Answer\*\*:.*',
        r'\n\s*Error Detected:.*',
        r'\n\s*Assessment:.*$',
    ]
    
    for pattern in cutoff_patterns:
        extracted = re.split(pattern, extracted, flags=re.DOTALL | re.IGNORECASE)[0]
    
    return extracted.strip()


def format_think_block(reasoning: str) -> str:
    """Wrap cleaned reasoning in Qwen3-style think tags."""
    cleaned = extract_reasoning_content(reasoning)
    if cleaned:
        return f"<think>\n{cleaned}\n</think>"
    return "<think>\nAnalyzing the clinical note...\n</think>"


# =============================================================================
# MESSAGE BUILDING - Creates training examples
# =============================================================================

def build_messages(example: Dict) -> List[Dict[str, str]]:
    """
    Build chat messages for SFT training.
    
    Output format aligns with what inference_error_detection.py expects:
    - Label: CORRECT or Label: INCORRECT
    - Keywords: "error", "incorrect", "correct", "no error"
    """
    role = example.get("role", "critic")
    task = example.get("task", "assessment")
    system_prompt = SYSTEM_PROMPTS.get((role, task), SYSTEM_PROMPTS[("critic", "assessment")])

    note = example.get("note", "").strip()
    label = example.get("label", "CORRECT").strip().upper()
    
    # Normalize label: training data uses "ERROR", inference uses "INCORRECT"
    if label == "ERROR":
        final_label = "INCORRECT"
    else:
        final_label = "CORRECT"

    # Build user content based on role
    if role == "critic" and task == "assessment":
        user_content = (
            "Analyze the following clinical note for medical errors.\n\n"
            f"Clinical note:\n{note}"
        )
    else:
        # Generator/Injector role
        correct_note = example.get("correct_note") or ""
        source_note = correct_note.strip() if correct_note.strip() else note
        error_type = (example.get("error_type") or "").strip()
        
        if final_label == "INCORRECT" and error_type:
            intent = f"Inject a subtle {error_type} error while keeping the note realistic."
        elif final_label == "INCORRECT":
            intent = "Inject a subtle clinical error while keeping the note realistic."
        else:
            intent = "Make minimal surface edits while preserving all clinical correctness."
        
        user_content = (
            f"Task: {intent}\n\n"
            f"Input note:\n{source_note}"
        )

    # Build assistant response with CLEANED reasoning
    think_block = format_think_block(example.get("reasoning", ""))
    
    if role == "critic" and task == "assessment":
        # For assessor: output label and brief explanation
        # Use keywords that inference parser looks for
        if final_label == "INCORRECT":
            explanation = "The note contains a clinical error that needs correction."
        else:
            explanation = "The note is clinically correct with no errors detected."
        
        assistant_content = (
            f"{think_block}\n\n"
            f"Label: {final_label}\n"
            f"Explanation: {explanation}"
        )
    else:
        # For generator: output the note and label
        assistant_content = (
            f"{think_block}\n\n"
            f"Generated note:\n{note}\n\n"
            f"Label: {final_label}"
        )
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def format_for_sft(example: Dict, tokenizer) -> Dict[str, str]:
    """Format example for SFT training using chat template."""
    messages = build_messages(example)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


# =============================================================================
# ARGUMENT PARSING
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SFT LoRA training for Qwen3-4B.")
    parser.add_argument("--train-file", required=True, help="Path to training JSONL.")
    parser.add_argument("--eval-file", default=None, help="Optional eval JSONL.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--model-name", default=DEFAULT_MODEL, help="Base model name.")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--num-train-epochs", type=int, default=1)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=200)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16.")
    parser.add_argument("--fp16", action="store_true", help="Use float16.")
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora-target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated list of target modules.",
    )
    parser.add_argument(
        "--filter-role",
        default=None,
        choices=["critic", "generator"],
        help="Filter training data to specific role (optional).",
    )
    return parser.parse_args()


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

def main() -> None:
    args = parse_args()

    print("\n" + "="*60)
    print("SFT LORA TRAINING - Medical Self-Play Model")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Train file: {args.train_file}")
    print(f"Output dir: {args.output_dir}")
    if args.filter_role:
        print(f"Filtering to role: {args.filter_role}")
    print("="*60 + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file

    dataset = load_dataset("json", data_files=data_files)
    
    # Optional: filter by role
    if args.filter_role:
        dataset["train"] = dataset["train"].filter(
            lambda ex: ex.get("role") == args.filter_role
        )
        print(f"Filtered to {len(dataset['train'])} {args.filter_role} examples")
        if "validation" in dataset:
            dataset["validation"] = dataset["validation"].filter(
                lambda ex: ex.get("role") == args.filter_role
            )

    train_ds = dataset["train"].map(
        lambda ex: format_for_sft(ex, tokenizer),
        remove_columns=dataset["train"].column_names,
    )
    eval_ds = None
    if "validation" in dataset:
        eval_ds = dataset["validation"].map(
            lambda ex: format_for_sft(ex, tokenizer),
            remove_columns=dataset["validation"].column_names,
        )

    # Print sample to verify format
    print("\n" + "="*60)
    print("SAMPLE TRAINING EXAMPLE")
    print("="*60)
    sample_text = train_ds[0]["text"]
    # Show last 800 chars to see the assistant response format
    print("...[truncated]...")
    print(sample_text[-800:])
    print("="*60 + "\n")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
    )

    target_modules = [m.strip() for m in args.lora_target_modules.split(",") if m.strip()]
    peft_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    sft_kwargs = {
        "output_dir": args.output_dir,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "num_train_epochs": args.num_train_epochs,
        "warmup_ratio": args.warmup_ratio,
        "logging_steps": args.logging_steps,
        "save_steps": args.save_steps,
        "evaluation_strategy": "steps" if eval_ds is not None else "no",
        "eval_steps": args.eval_steps if eval_ds is not None else None,
        "bf16": args.bf16,
        "fp16": args.fp16,
        "save_total_limit": 2,
        "report_to": "none",
    }
    sig = inspect.signature(SFTConfig.__init__)
    if "max_seq_length" in sig.parameters:
        sft_kwargs["max_seq_length"] = args.max_seq_length
    else:
        sft_kwargs["max_length"] = args.max_seq_length
    filtered_kwargs = {k: v for k, v in sft_kwargs.items() if k in sig.parameters}
    training_args = SFTConfig(**filtered_kwargs)

    trainer_kwargs = {
        "model": model,
        "train_dataset": train_ds,
        "eval_dataset": eval_ds,
        "processing_class": tokenizer,
        "args": training_args,
        "peft_config": peft_config,
    }
    trainer_sig = inspect.signature(SFTTrainer.__init__)
    trainer_kwargs = {k: v for k, v in trainer_kwargs.items() if k in trainer_sig.parameters}
    trainer = SFTTrainer(**trainer_kwargs)

    print("Starting training...")
    trainer.train()
    trainer.save_model(args.output_dir)

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)
        print(f"\n✓ Model saved to {args.output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
