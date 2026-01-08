#!/usr/bin/env python3
"""
Quick inference utility for a Qwen3-4B LoRA adapter.

Example:
  python scripts/sft/quick_infer_qwen3_4b_lora.py \
    --model-name Qwen/Qwen3-4B \
    --adapter-dir outputs/qwen3-4b-lora \
    --mode assessor \
    --input-note "A 55-year-old man ... Diagnosis: stable angina."
"""

import argparse
import json
from typing import Dict, List

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPTS = {
    "assessor": (
        "You are a meticulous clinical note assessor in a self-play loop. Your job"
        " is to analyze the note for clinical correctness, detect errors when they"
        " exist, and provide a clear final answer with a Yes/No error decision."
    ),
    "injector": (
        "You are an error injector in a self-play loop. Follow the prompt intent to"
        " transform the input note into a new note, either correct or with a subtle"
        " error, and provide a clear final answer."
    ),
}


def build_messages(mode: str, note: str, prompt_intent: str) -> List[Dict[str, str]]:
    if mode == "assessor":
        user_content = (
            "Role: assessor\n"
            "Task: analyze the clinical note for errors and classify it as CORRECT or INCORRECT.\n"
            "Output should include reasoning and a final answer formatted as:\n"
            'final_answer: "CORRECT" or "INCORRECT"\n\n'
            f"Clinical note:\n{note}\n"
        )
        system_prompt = SYSTEM_PROMPTS["assessor"]
    else:
        user_content = (
            "Role: error injector\n"
            "Task: follow the prompt intent and transform the input note into a new note.\n"
            f'prompt_intent: "{prompt_intent}"\n\n'
            f"input_note:\n{note}\n\n"
            "Output should include reasoning and a final answer formatted as:\n"
            "generated_note:\n... \n"
            'final_answer: "CORRECT" or "INCORRECT"\n\n'
        )
        system_prompt = SYSTEM_PROMPTS["injector"]

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick inference for Qwen3-4B LoRA.")
    parser.add_argument("--model-name", required=True, help="Base model name.")
    parser.add_argument("--adapter-dir", required=True, help="LoRA adapter directory.")
    parser.add_argument("--mode", choices=["assessor", "injector"], default="assessor")
    parser.add_argument("--input-note", default=None, help="Input clinical note.")
    parser.add_argument(
        "--jsonl-file",
        default=None,
        help="Optional JSONL file (rl_train.jsonl) for sampling examples.",
    )
    parser.add_argument("--max-examples", type=int, default=3)
    parser.add_argument(
        "--prompt-intent",
        default="Create a realistic note with no clinical errors.",
        help="Injector intent.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    notes = []
    if args.jsonl_file:
        with open(args.jsonl_file, "r", encoding="utf-8") as handle:
            for line in handle:
                if len(notes) >= args.max_examples:
                    break
                record = json.loads(line)
                if args.mode == "assessor":
                    note = record.get("incorrect_note") or record.get("correct_note")
                else:
                    note = record.get("correct_note") or record.get("incorrect_note")
                if note:
                    notes.append(note)
    elif args.input_note:
        notes = [args.input_note]
    else:
        notes = [
            "A 55-year-old man presents with exertional chest pain relieved by rest. "
            "ECG shows ST depressions. Diagnosis: stable angina."
        ]

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    for idx, note in enumerate(notes, start=1):
        messages = build_messages(args.mode, note, args.prompt_intent)
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n=== Example {idx} ===\n{generated}")


if __name__ == "__main__":
    main()
