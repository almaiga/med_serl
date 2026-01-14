#!/usr/bin/env python3
"""
LoRA SFT training for Qwen3-4B using existing CoT JSONL data.

Trains a model for medical self-play loop:
- Assessor role: Detect errors in clinical notes
- Injector role: Inject subtle errors or preserve correctness

Output formats aligned with:
- error_detection_prompts.json (assessor)
- error_injection_prompts_v2.json (injector)

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
from typing import Dict, List, Optional

from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


DEFAULT_MODEL = "Qwen/Qwen3-4B"


# =============================================================================
# SYSTEM PROMPTS - Aligned with self-play inference prompts
# =============================================================================

ASSESSOR_SYSTEM_PROMPT = """You are a medical note classifier.

Classify each note as CORRECT or INCORRECT.

CORRECT = No medical errors
INCORRECT = Contains medical error (wrong diagnosis, treatment, or management)

Think through your analysis step by step in <think> tags, then respond with EXACTLY this format:

final_answer: "CORRECT"
Explanation: [one sentence]

OR

final_answer: "INCORRECT"
Explanation: [one sentence why it's wrong]"""

INJECTOR_CORRECT_SYSTEM_PROMPT = """You are a minimal clinical note editor. Your ONLY task is to make a tiny surface change to ONE sentence while preserving all clinical meaning.

STRICT RULES:
1. Select EXACTLY ONE sentence from the note
2. Change 1-3 words ONLY in that sentence
3. Use a synonym or minor rephrasing that preserves EXACT clinical meaning
4. Leave ALL other sentences completely unchanged - copy them exactly

Think through your choice briefly in <think> tags, then output in this EXACT format:

generated_note:
[Copy the entire note with ONLY 1-3 words changed in ONE sentence]

final_answer: "CORRECT"

changes_made:
{"original_sentence": "<the original sentence>", "modified_sentence": "<the modified sentence>", "words_changed": "<word1> -> <word2>"}"""

INJECTOR_INCORRECT_SYSTEM_PROMPT = """You are a minimal clinical note editor. Your ONLY task is to introduce ONE subtle clinical error by changing 1-3 words in ONE sentence.

STRICT RULES:
1. Select EXACTLY ONE sentence from the note
2. Change 1-3 words ONLY in that sentence to introduce an error
3. The error must be clinically significant but subtle
4. Leave ALL other sentences completely unchanged - copy them exactly

Think through your choice briefly in <think> tags, then output in this EXACT format:

generated_note:
[Copy the entire note with ONLY 1-3 words changed in ONE sentence]

final_answer: "INCORRECT"

changes_made:
{"original_sentence": "<the original sentence>", "modified_sentence": "<the modified sentence>", "error_type": "<type of error>", "words_changed": "<word1> -> <word2>"}"""


def get_system_prompt(role: str, task: str, is_error: bool) -> str:
    """Get the appropriate system prompt based on role and label."""
    if role == "critic" and task == "assessment":
        return ASSESSOR_SYSTEM_PROMPT
    elif role == "generator" and task == "generation":
        return INJECTOR_INCORRECT_SYSTEM_PROMPT if is_error else INJECTOR_CORRECT_SYSTEM_PROMPT
    return ASSESSOR_SYSTEM_PROMPT


# =============================================================================
# REASONING EXTRACTION AND CLEANING
# =============================================================================

def extract_reasoning_content(text: str) -> str:
    """
    Extract only the reasoning content from GPT-4o output, removing
    Final Verdict sections and answer blocks.
    """
    if not text:
        return ""
    
    tag_patterns = [
        r'<reasoning>(.*?)</reasoning>',
        r'<generation_reasoning>(.*?)</generation_reasoning>',
        r'<error_injection_reasoning>(.*?)</error_injection_reasoning>',
    ]
    
    extracted = ""
    for pattern in tag_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            extracted = match.group(1).strip()
            break
    
    if not extracted:
        extracted = text
    
    # Remove answer sections that leaked into reasoning
    cutoff_patterns = [
        r'\n\s*\d+\.\s*Final Verdict:.*',
        r'\n\s*Final Verdict:.*',
        r'\n\s*\*\*Final Answer\*\*:.*',
        r'\n\s*Error Detected:.*',
        r'\n\s*Error Type:.*',
        r'\n\s*Correction:.*',
        r'\n\s*Assessment:.*$',
        r'\n\s*\d+\.\s*Ground Truth Documentation:.*',
        r'\n\s*\d+\.\s*Modified Note with Error:.*',
        r'\n\s*\d+\.\s*Final Note Composition:.*',
        r'\n\s*\d+\.\s*Post-Writing Verification:.*',
    ]
    
    for pattern in cutoff_patterns:
        extracted = re.split(pattern, extracted, flags=re.DOTALL | re.IGNORECASE)[0]
    
    # Keep reasoning focused (max ~30 lines)
    lines = extracted.strip().split('\n')
    if len(lines) > 30:
        extracted = '\n'.join(lines[:30])
    
    return extracted.strip()


def format_think_block(reasoning: str) -> str:
    """Wrap cleaned reasoning in Qwen3-style think tags."""
    cleaned = extract_reasoning_content(reasoning)
    if cleaned:
        return f"<think>\n{cleaned}\n</think>"
    return "<think>\nAnalyzing the clinical note...\n</think>"


# =============================================================================
# RESPONSE BUILDING
# =============================================================================

def build_assessor_response(think_block: str, is_error: bool, error_details: Optional[Dict] = None) -> str:
    """Build assessor response matching error_detection_prompts.json format."""
    if is_error:
        if error_details:
            error_sentence = error_details.get("error_sentence", "")
            if error_sentence:
                explanation = f"The note incorrectly states '{error_sentence[:50]}...' which should be corrected."
            else:
                explanation = "The note contains a clinical error in diagnosis or management."
        else:
            explanation = "The note contains a clinical error that affects patient care."
        
        return f'{think_block}\n\nfinal_answer: "INCORRECT"\nExplanation: {explanation}'
    else:
        return f'{think_block}\n\nfinal_answer: "CORRECT"\nExplanation: The clinical note is accurate with no medical errors detected.'


def build_injector_response(think_block: str, note: str, is_error: bool, error_details: Optional[Dict] = None) -> str:
    """Build injector response matching error_injection_prompts_v2.json format."""
    if is_error:
        if error_details:
            changes = {
                "original_sentence": error_details.get("corrected_sentence", "original sentence"),
                "modified_sentence": error_details.get("error_sentence", "modified sentence"),
                "error_type": error_details.get("error_type", "clinical error"),
                "words_changed": "original -> modified"
            }
        else:
            changes = {
                "original_sentence": "original sentence",
                "modified_sentence": "modified sentence with error",
                "error_type": "clinical error",
                "words_changed": "original -> modified"
            }
        return f'{think_block}\n\ngenerated_note:\n{note}\n\nfinal_answer: "INCORRECT"\n\nchanges_made:\n{json.dumps(changes)}'
    else:
        changes = {
            "original_sentence": "a sentence from the note",
            "modified_sentence": "the same sentence with minor rewording",
            "words_changed": "word -> synonym"
        }
        return f'{think_block}\n\ngenerated_note:\n{note}\n\nfinal_answer: "CORRECT"\n\nchanges_made:\n{json.dumps(changes)}'


# =============================================================================
# MESSAGE BUILDING
# =============================================================================

def build_messages(example: Dict) -> List[Dict[str, str]]:
    """Build chat messages for SFT training aligned with self-play prompts."""
    role = example.get("role", "critic")
    task = example.get("task", "assessment")
    note = example.get("note", "").strip()
    label = example.get("label", "CORRECT").strip().upper()
    
    is_error = label in ("ERROR", "INCORRECT")
    system_prompt = get_system_prompt(role, task, is_error)
    error_details = example.get("error_details")
    error_type = example.get("error_type", "")

    if role == "critic" and task == "assessment":
        user_content = f"Classify this note:\n\n{note}\n\nRespond in the required format:"
    else:
        correct_note = example.get("correct_note") or ""
        source_note = correct_note.strip() if correct_note.strip() else note
        
        if is_error:
            prompt_intent = error_type if error_type else "subtle clinical error"
            user_content = (
                f"TASK: Introduce ONE subtle clinical error in the note below.\n\n"
                f"CONSTRAINTS:\n- Change EXACTLY 1-3 words in ONE sentence\n"
                f"- The change must create a clinical error\n- Copy all other sentences unchanged\n\n"
                f"ERROR TYPE TO INJECT: {prompt_intent}\n\n"
                f"INPUT NOTE:\n{source_note}\n\nOUTPUT (follow format exactly):"
            )
        else:
            user_content = (
                f"TASK: Make a MINIMAL surface edit to the note below.\n\n"
                f"CONSTRAINTS:\n- Change EXACTLY 1-3 words in ONE sentence\n"
                f"- Use a synonym or minor rephrasing\n- Preserve exact clinical meaning\n"
                f"- Copy all other sentences unchanged\n\n"
                f"INPUT NOTE:\n{source_note}\n\nOUTPUT (follow format exactly):"
            )

    think_block = format_think_block(example.get("reasoning", ""))
    
    if role == "critic" and task == "assessment":
        assistant_content = build_assessor_response(think_block, is_error, error_details)
    else:
        assistant_content = build_injector_response(think_block, note, is_error, error_details)
    
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def format_for_sft(example: Dict, tokenizer) -> Dict[str, str]:
    """Format example for SFT training using chat template."""
    messages = build_messages(example)
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
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
    # Batch size settings optimized for RTX 6000 Pro (48GB VRAM)
    parser.add_argument("--per-device-train-batch-size", type=int, default=8,
                        help="Batch size per GPU (default: 8 for 48GB VRAM)")
    parser.add_argument("--per-device-eval-batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=2,
                        help="Accumulate gradients (effective batch = 8*2=16)")
    # Learning rate
    parser.add_argument("--learning-rate", type=float, default=1e-4,
                        help="Learning rate (default: 1e-4)")
    # Epochs
    parser.add_argument("--num-train-epochs", type=int, default=3,
                        help="Number of training epochs (default: 3)")
    parser.add_argument("--warmup-ratio", type=float, default=0.05,
                        help="Warmup ratio (default: 0.05)")
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=100)
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16.")
    parser.add_argument("--fp16", action="store_true", help="Use float16.")
    # LoRA settings - can use higher rank with 48GB VRAM
    parser.add_argument("--lora-r", type=int, default=64,
                        help="LoRA rank (default: 64, you have VRAM for it)")
    parser.add_argument("--lora-alpha", type=int, default=128,
                        help="LoRA alpha (default: 128, typically 2x rank)")
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj")
    parser.add_argument("--filter-role", default=None, choices=["critic", "generator"])
    parser.add_argument("--weight-decay", type=float, default=0.01)
    # Dataloader workers - leverage your 262GB RAM
    parser.add_argument("--dataloader-num-workers", type=int, default=4,
                        help="Number of dataloader workers")
    return parser.parse_args()


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = parse_args()

    effective_batch_size = args.per_device_train_batch_size * args.gradient_accumulation_steps
    
    print("\n" + "="*70)
    print("SFT LORA TRAINING - Medical Self-Play Model")
    print("="*70)
    print(f"Model: {args.model_name}")
    print(f"Train file: {args.train_file}")
    print(f"Output dir: {args.output_dir}")
    print(f"\nHardware: RTX 6000 Pro (48GB VRAM), 262GB RAM")
    print(f"\nTraining Parameters:")
    print(f"  - Epochs: {args.num_train_epochs}")
    print(f"  - Per-device batch size: {args.per_device_train_batch_size}")
    print(f"  - Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"  - Effective batch size: {effective_batch_size}")
    print(f"  - Learning rate: {args.learning_rate}")
    print(f"  - Warmup ratio: {args.warmup_ratio}")
    print(f"  - Weight decay: {args.weight_decay}")
    print(f"\nLoRA Parameters:")
    print(f"  - Rank (r): {args.lora_r}")
    print(f"  - Alpha: {args.lora_alpha}")
    print(f"  - Dropout: {args.lora_dropout}")
    if args.filter_role:
        print(f"\nFiltering to role: {args.filter_role}")
    print("="*70 + "\n")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file

    dataset = load_dataset("json", data_files=data_files)
    
    if args.filter_role:
        dataset["train"] = dataset["train"].filter(lambda ex: ex.get("role") == args.filter_role)
        print(f"Filtered to {len(dataset['train'])} {args.filter_role} examples")
        if "validation" in dataset:
            dataset["validation"] = dataset["validation"].filter(lambda ex: ex.get("role") == args.filter_role)

    train_ds = dataset["train"].map(lambda ex: format_for_sft(ex, tokenizer), remove_columns=dataset["train"].column_names)
    eval_ds = dataset["validation"].map(lambda ex: format_for_sft(ex, tokenizer), remove_columns=dataset["validation"].column_names) if "validation" in dataset else None

    print("\n" + "="*70 + "\nSAMPLE TRAINING EXAMPLE\n" + "="*70)
    for i in range(min(2, len(train_ds))):
        print(f"\n--- Example {i+1} ---\n...[truncated]...\n{train_ds[i]['text'][-600:]}")
    print("="*70 + "\n")

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype="auto")

    peft_config = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout,
        bias="none", task_type="CAUSAL_LM",
        target_modules=[m.strip() for m in args.lora_target_modules.split(",") if m.strip()],
    )

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
        "load_best_model_at_end": True if eval_ds else False,
        "metric_for_best_model": "eval_loss" if eval_ds else None,
        "report_to": "none",
        "lr_scheduler_type": "cosine",
        "dataloader_num_workers": args.dataloader_num_workers,
        "dataloader_pin_memory": True,  # Speeds up data transfer to GPU
    }
    sig = inspect.signature(SFTConfig.__init__)
    sft_kwargs["max_seq_length" if "max_seq_length" in sig.parameters else "max_length"] = args.max_seq_length
    training_args = SFTConfig(**{k: v for k, v in sft_kwargs.items() if k in sig.parameters})

    trainer_kwargs = {"model": model, "train_dataset": train_ds, "eval_dataset": eval_ds,
                      "processing_class": tokenizer, "args": training_args, "peft_config": peft_config}
    trainer_sig = inspect.signature(SFTTrainer.__init__)
    trainer = SFTTrainer(**{k: v for k, v in trainer_kwargs.items() if k in trainer_sig.parameters})

    print("Starting training...")
    trainer.train()
    trainer.save_model(args.output_dir)

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)
        print(f"\n✓ Model saved to {args.output_dir}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
