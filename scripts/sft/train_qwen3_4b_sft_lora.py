#!/usr/bin/env python3
"""
LoRA SFT training for Qwen3-4B using existing CoT JSONL data.

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


SYSTEM_PROMPTS = {
    ("critic", "assessment"): (
        "You are a meticulous clinical note assessor in a self-play loop. Your job"
        " is to analyze the note for clinical correctness, detect errors when they"
        " exist, and provide a clear final answer with a Yes/No error decision."
    ),
    ("generator", "generation"): (
        "You are an error injector in a self-play loop. Given a finished note, explain"
        " the reasoning that led to creating a realistic note, either correct or with"
        " a subtle error, and provide a clear final answer."
    ),
}


REASONING_TAGS = [
    "reasoning",
    "generation_reasoning",
    "error_injection_reasoning",
]


def to_think_tags(text: str) -> str:
    """Convert various reasoning tags into Qwen-style <think> tags."""
    if not text:
        return text
    for tag in REASONING_TAGS:
        text = re.sub(fr"<{tag}>", "<think>", text)
        text = re.sub(fr"</{tag}>", "</think>", text)
    return text


def build_messages(example: Dict) -> List[Dict[str, str]]:
    role = example.get("role", "critic")
    task = example.get("task", "assessment")
    system_prompt = SYSTEM_PROMPTS.get((role, task), SYSTEM_PROMPTS[("critic", "assessment")])

    note = example.get("note", "").strip()
    label = example.get("label", "CORRECT").strip().upper()
    final_answer = "INCORRECT" if label == "ERROR" else "CORRECT"

    if role == "critic" and task == "assessment":
        user_content = (
            "Role: assessor\n"
            "Task: analyze the clinical note for errors and classify it as CORRECT or INCORRECT.\n"
            "Output should include reasoning and a final answer formatted as:\n"
            'final_answer: "CORRECT" or "INCORRECT"\n\n'
            f"Clinical note:\n{note}\n"
        )
    else:
        correct_note = example.get("correct_note") or ""
        source_note = correct_note.strip() or note
        error_type = (example.get("error_type") or "").strip()
        if final_answer == "INCORRECT" and error_type:
            prompt_intent = (
                f"Introduce a {error_type} error while keeping the note realistic."
            )
        elif final_answer == "INCORRECT":
            prompt_intent = "Introduce a subtle clinical error while keeping the note realistic."
        else:
            prompt_intent = "Create a realistic note with no clinical errors."
        user_content = (
            "Role: error injector\n"
            "Task: follow the prompt intent and transform the input note into a new note.\n"
            f'prompt_intent: "{prompt_intent}"\n\n'
            f"input_note:\n{source_note}\n\n"
            "Output should include reasoning and a final answer formatted as:\n"
            'generated_note:\n... \n'
            'final_answer: "CORRECT" or "INCORRECT"\n\n'
        )

    reasoning = to_think_tags(example.get("reasoning", "").strip())
    if role == "critic" and task == "assessment":
        assistant_content = f"{reasoning}\n\nfinal_answer: \"{final_answer}\""
    else:
        assistant_content = (
            f"{reasoning}\n\n"
            f"generated_note:\n{note}\n\n"
            f"final_answer: \"{final_answer}\""
        )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def format_for_sft(example: Dict, tokenizer) -> Dict[str, str]:
    messages = build_messages(example)
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return {"text": text}


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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    data_files = {"train": args.train_file}
    if args.eval_file:
        data_files["validation"] = args.eval_file

    dataset = load_dataset("json", data_files=data_files)
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
    training_args = SFTConfig(**sft_kwargs)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        args=training_args,
        peft_config=peft_config,
    )

    trainer.train()
    trainer.save_model(args.output_dir)

    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(args.output_dir)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
