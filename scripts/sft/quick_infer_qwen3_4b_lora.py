#!/usr/bin/env python3
"""
Quick inference utility for a Qwen3-4B LoRA adapter.

Example:
  python scripts/sft/quick_infer_qwen3_4b_lora.py \
    --model-name Qwen/Qwen3-4B \
    --adapter-dir outputs/qwen3-4b-lora \
    --jsonl-file data_processed/medec_paired/train_val_split/rl_train.jsonl \
    --num-samples 3
"""

import argparse
import json
import os
import random
import re
from datetime import datetime
from typing import Dict, List, Optional

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

# Import change parsing utilities
try:
    from parse_changes import (
        parse_raw_output, get_change_diff, format_change_log, compute_word_level_diff
    )
    HAS_PARSE_CHANGES = True
except ImportError:
    HAS_PARSE_CHANGES = False
    import sys
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    try:
        from parse_changes import (
            parse_raw_output, get_change_diff, format_change_log, compute_word_level_diff
        )
        HAS_PARSE_CHANGES = True
    except ImportError:
        print("[WARNING] parse_changes.py not found in script directory.")
        print(f"[WARNING] Looked in: {script_dir}")
        print("[WARNING] Verbose diff logging and change tracking disabled.")
        print("[WARNING] Copy parse_changes.py to the same directory as this script.")

THINK_END_TOKEN_ID = 151668  # </think>
IM_END_TOKEN_ID = 151645  # <|im_end|>
MODEL_TYPE_QWEN = "qwen"
MODEL_TYPE_GENERIC = "generic"
# Different early stopping for assessor vs injector
EARLY_STOPPING_TEXT_ASSESSOR = '\n</think>\n\nfinal_answer: '
EARLY_STOPPING_TEXT_INJECTOR = '\n</think>\n\ngenerated_note:\n'  # Add newline for clarity

DEFAULT_NOTE_FIELDS = [
    "correct_note",
    "note",
    "text",
    "original_note",
    "clinical_note",
]


def build_messages(
    mode: str,
    note: str,
    prompt_intent: str,
    assessor_prompts: Optional[Dict[str, str]] = None,
    injector_prompts: Optional[Dict[str, str]] = None,
    thinking_budget: int = 0,
    injector_is_correct: Optional[bool] = None,
) -> List[Dict[str, str]]:
    """Build chat messages from JSON prompt configs. No hardcoded fallbacks."""
    if mode == "assessor":
        if not assessor_prompts:
            raise ValueError(
                "assessor_prompts is required. Provide --assessor-prompt-file pointing to a JSON "
                "file with 'system_prompt' and 'user_template' keys."
            )
        system_prompt = assessor_prompts["system_prompt"]
        user_content = assessor_prompts["user_template"].format(note=note)

    else:  # injector
        if not injector_prompts:
            raise ValueError(
                "injector_prompts is required. Provide --injector-prompt-file pointing to a JSON "
                "file with 'system_prompt_correct', 'system_prompt_incorrect', "
                "'injector_correct_template', 'injector_incorrect_template' keys."
            )
        if injector_is_correct is None:
            is_correct = "no clinical errors" in prompt_intent.lower()
        else:
            is_correct = injector_is_correct
        
        system_prompt = (
            injector_prompts["system_prompt_correct"]
            if is_correct
            else injector_prompts["system_prompt_incorrect"]
        )
        template = (
            injector_prompts["injector_correct_template"]
            if is_correct
            else injector_prompts["injector_incorrect_template"]
        )
        user_content = template.format(note=note, prompt_intent=prompt_intent)

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def load_assessor_prompts(prompt_file: Optional[str]) -> Optional[Dict[str, str]]:
    if not prompt_file:
        return None
    with open(prompt_file, "r", encoding="utf-8") as handle:
        prompts = json.load(handle)
    if "system_prompt" not in prompts or "user_template" not in prompts:
        raise ValueError("Prompt file must include system_prompt and user_template.")
    return prompts


def load_injector_prompts(prompt_file: Optional[str]) -> Optional[Dict[str, str]]:
    if not prompt_file:
        return None
    with open(prompt_file, "r", encoding="utf-8") as handle:
        prompts = json.load(handle)
    required = {
        "system_prompt_correct",
        "system_prompt_incorrect",
        "injector_correct_template",
        "injector_incorrect_template",
    }
    if not required.issubset(set(prompts.keys())):
        raise ValueError("Injector prompt file missing required keys.")
    return prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick inference for Qwen3-4B LoRA.")
    parser.add_argument("--model-name", required=True, help="Base model name or local path.")
    parser.add_argument(
        "--adapter-dir",
        default=None,
        help="Optional LoRA adapter directory. If omitted, uses base model only.",
    )
    parser.add_argument(
        "--jsonl-file",
        default=None,
        help="Optional JSONL file (rl_train.jsonl) for sampling examples.",
    )
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scenarios",
        default="assessor_correct,assessor_incorrect,injector_correct,injector_incorrect",
        help="Comma-separated scenarios to run.",
    )
    parser.add_argument(
        "--assessor-prompt-file",
        default="configs/prompts/error_detection_prompts.json",
        help="JSON prompt file for assessor (system_prompt + user_template).",
    )
    parser.add_argument(
        "--injector-prompt-file",
        default="configs/prompts/error_injection_prompts.json",
        help="JSON prompt file for injector prompts.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/inference/quick_test",
        help="Directory to write JSONL results.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name (defaults to timestamp).",
    )
    parser.add_argument("--input-note", default=None, help="Fallback single input note.")
    parser.add_argument(
        "--prompt-intent",
        default="Create a clinically identical variant of the input note.",
        help="Injector intent.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for generation.")
    parser.add_argument(
        "--force-answer",
        action="store_true",
        help="If Answer/CORRECT/INCORRECT is missing, run a short follow-up decode.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--thinking-budget", type=int, default=0, help="Thinking budget tokens.")
    parser.add_argument(
        "--assessor-thinking-budget",
        type=int,
        default=256,
        help="Thinking budget for assessor (less than injector - it's simpler task).",
    )
    parser.add_argument(
        "--min-p",
        type=float,
        default=0.05,
        help="Min-p sampling parameter (helps prevent Chinese output in Qwen).",
    )
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HF embedding model for cosine similarity.",
    )
    parser.add_argument(
        "--disable-embedding",
        action="store_true",
        help="Disable embedding cosine similarity scoring.",
    )
    parser.add_argument(
        "--selfplay",
        action="store_true",
        help="Run injector -> filter -> assessor loop instead of scenario eval.",
    )
    parser.add_argument(
        "--selfplay-mode",
        choices=["correct", "incorrect", "mixed"],
        default="correct",
        help="Injector mode for self-play loop.",
    )
    parser.add_argument(
        "--selfplay-min-jaccard",
        type=float,
        default=0.85,
        help="Minimum Jaccard similarity to accept generated notes.",
    )
    parser.add_argument(
        "--selfplay-max-jaccard",
        type=float,
        default=0.99,
        help="Maximum Jaccard similarity to accept generated notes.",
    )
    parser.add_argument(
        "--selfplay-max-attempts",
        type=int,
        default=3,
        help="Max injector attempts per note before skipping.",
    )
    parser.add_argument(
        "--selfplay-assessor-batch-size",
        type=int,
        default=8,
        help="Batch size for assessor in self-play.",
    )
    parser.add_argument(
        "--selfplay-injector-batch-size",
        type=int,
        default=8,
        help="Batch size for injector in self-play (one attempt per note per round).",
    )
    parser.add_argument(
        "--selfplay-num-notes",
        type=int,
        default=None,
        help="Override number of notes to process in self-play.",
    )
    parser.add_argument(
        "--note-field",
        default=None,
        help="Optional field name for the source note in JSONL.",
    )
    parser.add_argument(
        "--no-thinking",
        action="store_true",
        help="Disable Qwen3 thinking mode for simpler tasks (uses enable_thinking=False).",
    )
    parser.add_argument(
        "--verbose-diff",
        action="store_true",
        help="Print detailed change diffs during selfplay.",
    )
    parser.add_argument(
        "--max-word-edits",
        type=int,
        default=6,
        help="Maximum word edits allowed for a 'minimal' change (default: 6).",
    )
    return parser.parse_args()


def extract_final_answer(text: str) -> Optional[str]:
    """Extract final answer with multiple fallback patterns."""
    # Use findall to get ALL matches, then take the last one
    matches = re.findall(r'final_answer:\s*"(CORRECT|INCORRECT)"', text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()  # Return the LAST match
    
    # Fallback 1: without quotes (also get last match)
    matches = re.findall(r'final_answer:\s*(CORRECT|INCORRECT)', text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    
    # Fallback 2: look for Error Detected field (from model's old format)
    if re.search(r'Error Detected:\s*Yes', text, re.IGNORECASE):
        return "INCORRECT"
    elif re.search(r'Error Detected:\s*No', text, re.IGNORECASE):
        return "CORRECT"

    # Fallback 3: baseline prompt format "Answer: CORRECT/INCORRECT"
    matches = re.findall(r'Answer:\s*(CORRECT|INCORRECT)', text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    
    # Fallback 4: look at assessment line (get last match)
    matches = re.findall(r'Assessment:\s*(CORRECT|INCORRECT)', text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    
    return None


def is_lora_adapter(path: Optional[str]) -> bool:
    if not path:
        return False
    return os.path.exists(os.path.join(path, "adapter_config.json"))


def get_base_model_from_adapter(path: str) -> str:
    adapter_config_path = os.path.join(path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        with open(adapter_config_path, "r", encoding="utf-8") as handle:
            config = json.load(handle)
        return config.get("base_model_name_or_path", "")
    return ""


def detect_model_type(model_name: str, adapter_dir: Optional[str]) -> str:
    candidates = [model_name]
    if adapter_dir:
        candidates.append(adapter_dir)
        if is_lora_adapter(adapter_dir):
            base_model = get_base_model_from_adapter(adapter_dir)
            if base_model:
                candidates.append(base_model)

    for candidate in candidates:
        if "qwen" in candidate.lower():
            return MODEL_TYPE_QWEN
        config_path = os.path.join(candidate, "config.json")
        if os.path.exists(config_path):
            try:
                with open(config_path, "r", encoding="utf-8") as handle:
                    config = json.load(handle)
                model_type_str = config.get("model_type", "").lower()
                architectures = " ".join(config.get("architectures", [])).lower()
                if "qwen" in model_type_str or "qwen" in architectures:
                    return MODEL_TYPE_QWEN
            except Exception:
                pass

    return MODEL_TYPE_GENERIC


def normalize_qwen_thinking(thinking_content: str) -> str:
    if not thinking_content:
        return ""
    cleaned = re.sub(r"</?think>", "", thinking_content, flags=re.IGNORECASE).strip()
    cleaned = re.sub(
        r"Considering the limited time by the user, I have to give the solution based on the thinking directly now\.",
        "",
        cleaned,
        flags=re.IGNORECASE,
    ).strip()
    return cleaned


def strip_qwen_think_from_content(content: str) -> str:
    if not content:
        return content
    content = re.sub(r"<think>.*?</think>", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"</?think>", "", content, flags=re.IGNORECASE)
    content = re.sub(r"</?answer>", "", content, flags=re.IGNORECASE)
    content = re.sub(r"^answer:\s*", "", content, flags=re.IGNORECASE)
    return content.strip()


def parse_qwen3_output(tokenizer, input_ids, generated_ids) -> str:
    input_length = input_ids.shape[1]
    output_ids = generated_ids[0, input_length:].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(THINK_END_TOKEN_ID)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(
        output_ids[:index],
        skip_special_tokens=True,
    ).strip("\n")
    content = tokenizer.decode(
        output_ids[index:],
        skip_special_tokens=True,
    ).strip("\n")

    thinking_content = normalize_qwen_thinking(thinking_content)
    content = strip_qwen_think_from_content(content)

    if thinking_content:
        return f"<think>{thinking_content}</think>\n{content}"
    return content


def parse_qwen3_output_with_length(tokenizer, input_length: int, generated_ids, original_seq_len: int = None) -> str:
    # If original_seq_len provided, use it to slice correctly with left-padding
    if original_seq_len is not None:
        output_ids = generated_ids[original_seq_len:].tolist()
    else:
        output_ids = generated_ids[input_length:].tolist()

    try:
        index = len(output_ids) - output_ids[::-1].index(THINK_END_TOKEN_ID)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(
        output_ids[:index],
        skip_special_tokens=True,
    ).strip("\n")
    content = tokenizer.decode(
        output_ids[index:],
        skip_special_tokens=True,
    ).strip("\n")

    thinking_content = normalize_qwen_thinking(thinking_content)
    content = strip_qwen_think_from_content(content)

    if thinking_content:
        return f"<think>{thinking_content}</think>\n{content}"
    return content


def generate_qwen_with_thinking(
    model,
    tokenizer,
    prompt: str,
    thinking_budget: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    min_p: float = 0.05,
) -> str:
    model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
    input_length = model_inputs.input_ids.size(-1)

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=thinking_budget,
            temperature=temperature,
            top_p=top_p,
            top_k=20,  # Official Qwen3 recommendation
            min_p=min_p,  # Prevents Chinese output
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,  # Light penalty during thinking
        )

    output_ids = generated_ids[0, input_length:].tolist()

    if IM_END_TOKEN_ID not in output_ids:
        if THINK_END_TOKEN_ID not in output_ids:
            thinking_content = tokenizer.decode(
                output_ids,
                skip_special_tokens=True,
            ).strip("\n")
            thinking_content = normalize_qwen_thinking(thinking_content)
            # Close think tag and provide clear instruction for what comes next
            early_stopping_ids = tokenizer(
                [EARLY_STOPPING_TEXT],
                return_tensors="pt",
                add_special_tokens=False,
            ).input_ids.to(model.device)
            input_ids = torch.cat([generated_ids, early_stopping_ids], dim=-1)
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            remaining_tokens = max_new_tokens - (input_ids.size(-1) - input_length)
            if remaining_tokens > 0:
                # After thinking, generate the final answer (just the final_answer line)
                answer_tokens = min(remaining_tokens, 128)  # Increase from 64 to 128
                with torch.no_grad():
                    generated_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=answer_tokens,
                        temperature=0.6,  # Qwen3 official recommendation for thinking mode
                        top_p=0.95,  # Official recommendation
                        top_k=20,
                        min_p=min_p,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.05,  # Very light penalty to avoid forcing Chinese
                    )
                answer = parse_qwen3_output_with_length(tokenizer, input_ids.size(-1), generated_ids)
                answer = strip_qwen_think_from_content(answer).strip()
            else:
                answer = ""
            if thinking_content:
                if answer:
                    return f"<think>{thinking_content}</think>\n{answer}"
                return f"<think>{thinking_content}</think>"
            return answer
        input_ids = generated_ids
        attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
        remaining_tokens = max_new_tokens - (input_ids.size(-1) - input_length)
        if remaining_tokens > 0:
            # After thinking naturally ends, generate final answer
            answer_tokens = min(remaining_tokens, 128)  # Increase from 64
            with torch.no_grad():
                generated_ids = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=answer_tokens,
                    temperature=0.6,  # Qwen3 official recommendation
                    top_p=0.95,  # Official recommendation
                    top_k=20,
                    min_p=min_p,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.05,  # Very light - avoid forcing alt language
                )

    return parse_qwen3_output(tokenizer, model_inputs.input_ids, generated_ids)


def generate_qwen_with_thinking_batch(
    model,
    tokenizer,
    prompts: List[str],
    thinking_budget: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    min_p: float = 0.05,
    answer_tokens: int = 128,
    stop_strings: Optional[List[str]] = None,
    early_stop_text: str = EARLY_STOPPING_TEXT_INJECTOR,  # Default to injector format
) -> List[str]:
    if not prompts:
        return []

    model_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
    input_lengths = model_inputs["attention_mask"].sum(dim=1).tolist()

    with torch.no_grad():
        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=thinking_budget,
            temperature=temperature,
            top_p=top_p,
            top_k=20,
            min_p=min_p,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    outputs: List[str] = []
    for idx, input_length in enumerate(input_lengths):
        original_input_len = model_inputs['input_ids'][idx].shape[0]
        output_ids = generated_ids[idx, original_input_len:].tolist()
        final_ids = generated_ids[idx]

        if IM_END_TOKEN_ID not in output_ids:
            if THINK_END_TOKEN_ID not in output_ids:
                thinking_content = tokenizer.decode(
                    output_ids,
                    skip_special_tokens=True,
                ).strip("\n")
                thinking_content = normalize_qwen_thinking(thinking_content)
                
                # FIX: Extract only the non-padded tokens from this sequence
                # Left-padding means real tokens are at the end
                num_pad_tokens = original_input_len - input_length
                real_tokens = generated_ids[idx, num_pad_tokens:].unsqueeze(0)  # Remove padding
                
                # Now concatenate the early stopping text
                early_stopping_ids = tokenizer(
                    [early_stop_text],  # Use the provided early_stop_text
                    return_tensors="pt",
                    add_special_tokens=False,
                ).input_ids.to(model.device)
                
                # Concatenate to the REAL (non-padded) tokens
                input_ids = torch.cat([real_tokens, early_stopping_ids], dim=-1)
                attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
                
                remaining_tokens = max_new_tokens - (input_ids.size(-1) - input_length)
                if remaining_tokens > 0:
                    actual_answer_tokens = min(remaining_tokens, answer_tokens)
                    stop_token_ids = []
                    if stop_strings:
                        for stop_str in stop_strings:
                            stop_ids = tokenizer.encode(stop_str, add_special_tokens=False)
                            if stop_ids:
                                stop_token_ids.append(stop_ids[-1])
                    eos_ids = [tokenizer.eos_token_id] + stop_token_ids if stop_token_ids else [tokenizer.eos_token_id]
                    
                    with torch.no_grad():
                        followup_ids = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            max_new_tokens=actual_answer_tokens,
                            temperature=0.3,
                            top_p=0.95,
                            top_k=20,
                            min_p=min_p,
                            do_sample=True,
                            pad_token_id=tokenizer.pad_token_id,
                            eos_token_id=eos_ids[0] if len(eos_ids) == 1 else eos_ids,
                            repetition_penalty=1.05,
                        )
                    final_ids = followup_ids[0]
                    answer = parse_qwen3_output_with_length(
                        tokenizer,
                        input_ids.size(-1),
                        final_ids,
                    )
                    answer = strip_qwen_think_from_content(answer).strip()
                else:
                    final_ids = input_ids[0]
                    answer = ""
                if thinking_content:
                    if answer:
                        outputs.append(f"<think>{thinking_content}</think>\n{answer}")
                    else:
                        outputs.append(f"<think>{thinking_content}</think>")
                else:
                    outputs.append(answer)
                continue
                
            # Same fix for the case where thinking naturally ends
            num_pad_tokens = original_input_len - input_length
            input_ids = generated_ids[idx, num_pad_tokens:].unsqueeze(0)  # Remove padding
            attention_mask = torch.ones_like(input_ids, dtype=torch.int64)
            remaining_tokens = max_new_tokens - (input_ids.size(-1) - input_length)
            if remaining_tokens > 0:
                actual_answer_tokens = min(remaining_tokens, answer_tokens)
                stop_token_ids = []
                if stop_strings:
                    for stop_str in stop_strings:
                        stop_ids = tokenizer.encode(stop_str, add_special_tokens=False)
                        if stop_ids:
                            stop_token_ids.append(stop_ids[-1])
                eos_ids = [tokenizer.eos_token_id] + stop_token_ids if stop_token_ids else [tokenizer.eos_token_id]
                with torch.no_grad():
                    followup_ids = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=actual_answer_tokens,
                        temperature=0.3,
                        top_p=0.95,
                        top_k=20,
                        min_p=min_p,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=eos_ids[0] if len(eos_ids) == 1 else eos_ids,
                        repetition_penalty=1.05,
                    )
                final_ids = followup_ids[0]
            else:
                final_ids = input_ids[0]

        outputs.append(parse_qwen3_output_with_length(tokenizer, input_length, final_ids, original_input_len))

    return outputs

def force_answer_from_prompt(
    prompt: str,
    tokenizer,
    model,
    device: str,
    max_new_tokens: int = 12,
) -> Optional[str]:
    """Run a short follow-up decode to elicit the Answer line."""
    followup = prompt.rstrip() + "\nAnswer:"
    inputs = tokenizer(followup, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    generated = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return extract_final_answer(generated)




def strip_think_blocks(text: str) -> str:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL | re.IGNORECASE)
    cleaned = re.sub(r"</?think>", "", cleaned, flags=re.IGNORECASE)
    return cleaned


def clean_generated_note(note: str) -> str:
    if not note:
        return note
    note = strip_think_blocks(note)
    parts = re.split(r"\*\*Injected error\*\*:\s*", note, flags=re.IGNORECASE)
    if len(parts) > 1:
        return parts[0].strip()
    parts = re.split(r"Injected error:\s*", note, flags=re.IGNORECASE)
    if len(parts) > 1:
        return parts[0].strip()
    note = re.sub(r'final_answer:.*', '', note, flags=re.IGNORECASE | re.DOTALL).strip()
    return note.strip().strip('"').strip("'")


def extract_generated_note(text: str) -> Optional[str]:
    """Extract the generated_note section from injector output."""
    # Split by "generated_note:" first to isolate the section (allow same-line)
    parts = re.split(r'generated_note:\s*', text, flags=re.IGNORECASE)

    if len(parts) >= 2:
        # Take everything after "generated_note:"
        after_label = parts[1]

        # Now extract until "final_answer:" (stop marker)
        match = re.search(r'^(.*?)\s*final_answer:', after_label, re.DOTALL | re.IGNORECASE)

        if match:
            generated = match.group(1).strip()
        else:
            # If no final_answer found, take everything
            generated = after_label.strip()

        if generated:
            cleaned = clean_generated_note(generated)
            return cleaned if cleaned else None

    # Fallback 1: "Modified Note with Error" block (quoted)
    match = re.search(r'Modified Note with Error:\s*"(.*?)"', text, re.DOTALL | re.IGNORECASE)
    if match:
        candidate = match.group(1).strip()
        cleaned = clean_generated_note(candidate)
        return cleaned if cleaned else None

    # Fallback 2: "Modified Note with Error" block (unquoted, stop at next section)
    match = re.search(
        r'Modified Note with Error:\s*(.*?)(?:\n\s*\d+\.\s+|Ground Truth|Error location|final_answer:|$)',
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        candidate = match.group(1).strip()
        cleaned = clean_generated_note(candidate)
        return cleaned if cleaned else None

    # Fallback 3: "Modified Note" (non-error wording)
    match = re.search(
        r'Modified Note:\s*(.*?)(?:\n\s*\d+\.\s+|Ground Truth|Error location|final_answer:|$)',
        text,
        re.DOTALL | re.IGNORECASE,
    )
    if match:
        candidate = match.group(1).strip()
        cleaned = clean_generated_note(candidate)
        return cleaned if cleaned else None

    # Fallback 4: strip think + final_answer and treat remainder as note
    cleaned = strip_think_blocks(text)
    cleaned = re.sub(r'final_answer:.*', '', cleaned, flags=re.IGNORECASE | re.DOTALL)
    cleaned = cleaned.strip()
    cleaned = clean_generated_note(cleaned)
    return cleaned if cleaned else None


def tokenize_for_jaccard(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def jaccard_similarity(text1: str, text2: str) -> float:
    set1 = set(tokenize_for_jaccard(text1))
    set2 = set(tokenize_for_jaccard(text2))
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def word_counts(text: str) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for token in tokenize_for_jaccard(text):
        counts[token] = counts.get(token, 0) + 1
    return counts


def has_word_change(original: str, generated: str) -> bool:
    return word_counts(original) != word_counts(generated)


def select_note(record: Dict, note_field: Optional[str]) -> Optional[str]:
    if note_field:
        return record.get(note_field)
    for field in DEFAULT_NOTE_FIELDS:
        value = record.get(field)
        if value:
            return value
    return None


def passes_similarity_filter(
    original_note: str,
    generated_note: str,
    min_jaccard: float,
    max_jaccard: Optional[float] = None,
) -> Dict[str, Optional[float]]:
    if not generated_note:
        return {"passed": False, "score_jaccard": None, "reason": "empty_generated"}
    if not has_word_change(original_note, generated_note):
        return {"passed": False, "score_jaccard": None, "reason": "no_word_change"}
    score_jaccard = jaccard_similarity(original_note, generated_note)
    if score_jaccard < min_jaccard:
        return {"passed": False, "score_jaccard": score_jaccard, "reason": "low_jaccard"}
    if max_jaccard is not None and score_jaccard > max_jaccard:
        return {"passed": False, "score_jaccard": score_jaccard, "reason": "too_similar"}
    return {"passed": True, "score_jaccard": score_jaccard, "reason": None}


def load_embedding_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def embed_texts(tokenizer, model, device: str, texts: List[str]) -> torch.Tensor:
    with torch.no_grad():
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.detach().cpu()


def cosine_similarity(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
    return float((emb_a * emb_b).sum().item())


def load_records(jsonl_file: str) -> List[Dict]:
    records = []
    with open(jsonl_file, "r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def scenario_samples(records: List[Dict], scenario: str, num_samples: int) -> List[Dict]:
    if not records:
        return []
    if scenario == "assessor_correct":
        pool = [r for r in records if r.get("correct_note")]
    elif scenario == "assessor_incorrect":
        pool = [r for r in records if r.get("incorrect_note")]
    elif scenario == "injector_correct":
        pool = [r for r in records if r.get("correct_note")]
    else:
        pool = [r for r in records if r.get("correct_note") and r.get("incorrect_note")]
    if not pool:
        return []
    sample_count = min(num_samples, len(pool))
    return random.sample(pool, sample_count)


def write_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: str, summary: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def summarize_results(rows: List[Dict]) -> Dict:
    summary: Dict[str, Dict] = {}
    for row in rows:
        scenario = row["scenario"]
        stats = summary.setdefault(
            scenario,
            {
                "total": 0,
                "correct": 0,
                "similarity_jaccard": [],
                "similarity_embedding": [],
            },
        )
        stats["total"] += 1
        stats["correct"] += int(bool(row.get("match")))
        if row.get("score_jaccard") is not None:
            stats["similarity_jaccard"].append(row["score_jaccard"])
        if row.get("score_embedding_cosine") is not None:
            stats["similarity_embedding"].append(row["score_embedding_cosine"])

    def agg_sim(values: List[float]) -> Dict:
        if not values:
            return {"avg": None, "min": None, "max": None}
        return {
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    assessor_correct = summary.get("assessor_correct", {"total": 0, "correct": 0})
    assessor_incorrect = summary.get("assessor_incorrect", {"total": 0, "correct": 0})
    injector_correct = summary.get("injector_correct", {"total": 0, "correct": 0})
    injector_incorrect = summary.get("injector_incorrect", {"total": 0, "correct": 0})

    total_assessor = assessor_correct["total"] + assessor_incorrect["total"]
    correct_assessor = assessor_correct["correct"] + assessor_incorrect["correct"]
    total_injector = injector_correct["total"] + injector_incorrect["total"]
    correct_injector = injector_correct["correct"] + injector_incorrect["correct"]

    return {
        "assessor": {
            "correct_notes": {
                "samples": assessor_correct["total"],
                "correct": assessor_correct["correct"],
                "accuracy": (assessor_correct["correct"] / assessor_correct["total"]) if assessor_correct["total"] else 0.0,
            },
            "incorrect_notes": {
                "samples": assessor_incorrect["total"],
                "correct": assessor_incorrect["correct"],
                "accuracy": (assessor_incorrect["correct"] / assessor_incorrect["total"]) if assessor_incorrect["total"] else 0.0,
            },
            "overall": {
                "samples": total_assessor,
                "correct": correct_assessor,
                "accuracy": (correct_assessor / total_assessor) if total_assessor else 0.0,
            },
        },
        "injector": {
            "generate_correct": {
                "samples": injector_correct["total"],
                "correct": injector_correct["correct"],
                "accuracy": (injector_correct["correct"] / injector_correct["total"]) if injector_correct["total"] else 0.0,
                "similarity_jaccard": agg_sim(summary.get("injector_correct", {}).get("similarity_jaccard", [])),
                "similarity_embedding": agg_sim(summary.get("injector_correct", {}).get("similarity_embedding", [])),
            },
            "inject_errors": {
                "samples": injector_incorrect["total"],
                "correct": injector_incorrect["correct"],
                "accuracy": (injector_incorrect["correct"] / injector_incorrect["total"]) if injector_incorrect["total"] else 0.0,
                "similarity_jaccard": agg_sim(summary.get("injector_incorrect", {}).get("similarity_jaccard", [])),
                "similarity_embedding": agg_sim(summary.get("injector_incorrect", {}).get("similarity_embedding", [])),
            },
            "overall": {
                "samples": total_injector,
                "correct": correct_injector,
                "accuracy": (correct_injector / total_injector) if total_injector else 0.0,
                "similarity_jaccard": agg_sim(
                    (summary.get("injector_correct", {}).get("similarity_jaccard", []) +
                     summary.get("injector_incorrect", {}).get("similarity_jaccard", []))
                ),
                "similarity_embedding": agg_sim(
                    (summary.get("injector_correct", {}).get("similarity_embedding", []) +
                     summary.get("injector_incorrect", {}).get("similarity_embedding", []))
                ),
            },
        },
    }


def run_selfplay_loop(
    args: argparse.Namespace,
    model,
    tokenizer,
    model_type: str,
    records: List[Dict],
    assessor_prompts: Dict[str, str],
    injector_prompts: Dict[str, str],
    run_name: str,
    output_path: str,
    summary_path: str,
) -> None:
    # Determine thinking mode: use thinking only if budget > 0 AND not explicitly disabled
    use_qwen_thinking = (
        model_type == MODEL_TYPE_QWEN 
        and args.thinking_budget > 0 
        and not getattr(args, 'no_thinking', False)
    )
    
    # For non-thinking mode with Qwen3, use different temperature settings per best practices
    if model_type == MODEL_TYPE_QWEN and getattr(args, 'no_thinking', False):
        print("[INFO] Using Qwen3 non-thinking mode (enable_thinking=False)")
        # Best practices: temp=0.7, top_p=0.8 for non-thinking mode
        effective_temp = 0.7 if args.temperature == 0.2 else args.temperature  # Only override default
        effective_top_p = 0.8 if args.top_p == 0.9 else args.top_p
    elif use_qwen_thinking:
        assessor_budget = getattr(args, 'assessor_thinking_budget', 256) or 256
        print(f"[INFO] Using Qwen3 thinking mode (injector budget={args.thinking_budget}, assessor budget={assessor_budget})")
        print(f"[INFO] Using min_p={args.min_p} (prevents Chinese output)")
        # Best practices: temp=0.6, top_p=0.95 for thinking mode
        effective_temp = 0.6 if args.temperature == 0.2 else args.temperature
        effective_top_p = 0.95 if args.top_p == 0.9 else args.top_p
    else:
        effective_temp = args.temperature
        effective_top_p = args.top_p
    
    max_notes = args.selfplay_num_notes or args.num_samples
    output_root, output_ext = os.path.splitext(output_path)
    attempts_path = f"{output_root}_attempts{output_ext or '.jsonl'}"

    samples = records
    if max_notes is not None and samples:
        samples = samples[:max_notes]
    if not samples and args.input_note:
        samples = [{"note_id": "input_note", "note": args.input_note}]

    stats = {
        "total_input_notes": 0,
        "attempted_generations": 0,
        "accepted_generations": 0,
        "attempts_logged": 0,
        "attempts_passed_filter": 0,
        "attempts_failed_filter": 0,
        "rejected_empty": 0,
        "rejected_no_word_change": 0,
        "rejected_low_jaccard": 0,
        "rejected_too_similar": 0,
        "rejected_too_many_edits": 0,
        "assessor_total": 0,
        "assessor_match_expected": 0,
    }
    mixed_counter = 0

    def build_prompt(messages: List[Dict[str, str]]) -> str:
        if model_type == MODEL_TYPE_QWEN:
            # For Qwen3: explicitly control thinking mode
            enable_thinking = use_qwen_thinking and not getattr(args, 'no_thinking', False)
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking,
            )
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    def assess_batch(batch: List[Dict]) -> List[Dict]:
        prompts: List[str] = []
        prompt_metas: List[Dict] = []
        # Use assessor-specific thinking budget (shorter - it's a simpler task)
        assessor_budget = getattr(args, 'assessor_thinking_budget', 256) or 256
        for item in batch:
            messages = build_messages(
                "assessor",
                item["generated_note"],
                "Assess note correctness.",
                assessor_prompts,
                injector_prompts,
                assessor_budget,  # Use shorter budget for assessor
                None,
            )
            prompt = build_prompt(messages)
            prompts.append(prompt)
            prompt_metas.append({"prompt": prompt})

        outputs: List[str] = []
        
        # DEBUG: Print assessor parameters and first prompt
        if args.verbose_diff and prompts:
            print("\n" + "="*60)
            print("ASSESSOR GENERATION PARAMETERS:")
            print(f"  thinking_budget: {assessor_budget}")
            print(f"  max_new_tokens: {args.max_new_tokens}")
            print(f"  temperature: {effective_temp}")
            print(f"  top_p: {effective_top_p}")
            print(f"  min_p: {args.min_p}")
            print(f"  use_qwen_thinking: {use_qwen_thinking}")
            print(f"\nFIRST ASSESSOR PROMPT (truncated):")
            print(prompts[0][:2000] + "..." if len(prompts[0]) > 2000 else prompts[0])
            print("="*60 + "\n")
        
        if use_qwen_thinking:
            outputs = generate_qwen_with_thinking_batch(
                model,
                tokenizer,
                prompts,
                assessor_budget,
                args.max_new_tokens,
                effective_temp,
                effective_top_p,
                min_p=args.min_p,
                answer_tokens=512,  # Increased from 256 to capture full explanation
                stop_strings=None,  # Remove stop strings - let explanation complete naturally
                early_stop_text=EARLY_STOPPING_TEXT_ASSESSOR,
            )
        else:
            for start in range(0, len(prompts), args.selfplay_assessor_batch_size):
                batch_prompts = prompts[start:start + args.selfplay_assessor_batch_size]
                inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
                original_lengths = [inputs['input_ids'][i].shape[0] for i in range(len(batch_prompts))]
                with torch.no_grad():
                    batch_outputs = model.generate(
                        **inputs,
                        max_new_tokens=256,  # Limit assessor output - only need final_answer
                        do_sample=effective_temp > 0,
                        temperature=effective_temp,
                        top_p=effective_top_p,
                        top_k=20,  # Official Qwen3 recommendation
                        min_p=args.min_p,  # Prevents Chinese output
                        pad_token_id=tokenizer.eos_token_id,
                        repetition_penalty=1.1,  # Lighter penalty
                    )
                for idx, original_len in enumerate(original_lengths):
                    generated_tokens = batch_outputs[idx][original_len:]
                    generated = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                    outputs.append(generated)

        # DEBUG: Print first raw output
        if args.verbose_diff and outputs:
            print("\n" + "-"*60)
            print("FIRST ASSESSOR RAW OUTPUT:")
            print(outputs[0][:1500] + "..." if len(outputs[0]) > 1500 else outputs[0])
            print("-"*60 + "\n")
        
        assessed_rows: List[Dict] = []
        for item, raw_output, meta in zip(batch, outputs, prompt_metas):
            predicted = extract_final_answer(raw_output)
            if predicted is None and args.force_answer:
                predicted = force_answer_from_prompt(
                    meta["prompt"],
                    tokenizer,
                    model,
                    model.device,
                )
            expected_assessor = item.get("assessor_expected")
            match = predicted == expected_assessor
            stats["assessor_total"] += 1
            stats["assessor_match_expected"] += int(match)
            assessed_rows.append(
                {
                    **item,
                    "scenario": f"selfplay_{item.get('selfplay_mode')}",
                    "assessor_expected": expected_assessor,
                    "assessor_predicted": predicted,
                    "assessor_match_expected": match,
                    "assessor_raw_output": raw_output,
                }
            )
        return assessed_rows

    pending: List[Dict] = []
    injector_batch_size = max(1, args.selfplay_injector_batch_size)

    def generate_injector_outputs(prompts: List[str]) -> List[str]:
        if use_qwen_thinking:
            return generate_qwen_with_thinking_batch(
                model,
                tokenizer,
                prompts,
                args.thinking_budget,
                args.max_new_tokens,
                effective_temp,
                effective_top_p,
                min_p=args.min_p,
                answer_tokens=1024,  # Increased from 768 to capture full note + metadata
                stop_strings=None,
                early_stop_text=EARLY_STOPPING_TEXT_INJECTOR,
            )
        inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
        original_lengths = [inputs['input_ids'][i].shape[0] for i in range(len(prompts))]
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=effective_temp > 0,
                temperature=effective_temp,
                top_p=effective_top_p,
                top_k=20,  # Official Qwen3 recommendation
                min_p=args.min_p,  # Prevents Chinese output
                pad_token_id=tokenizer.eos_token_id,
            )
        decoded: List[str] = []
        for idx, original_len in enumerate(original_lengths):
            generated_tokens = outputs[idx][original_len:]
            decoded.append(tokenizer.decode(generated_tokens, skip_special_tokens=True))
        return decoded

    # Constants for per-note retry logic
    MAX_RETRIES_PER_NOTE = 4
    
    with open(output_path, "a", encoding="utf-8") as out_handle, open(
        attempts_path, "a", encoding="utf-8"
    ) as attempts_handle:
        with tqdm(total=len(samples), desc="Self-play", unit="note") as pbar:
            # Process one note at a time with retry logic
            for record in samples:
                original_note = select_note(record, args.note_field)
                if not original_note:
                    pbar.update(1)
                    continue
                
                stats["total_input_notes"] += 1
                note_id = record.get("note_id")
                error_type = (record.get("error_type") or "").strip()
                
                # Determine selfplay mode
                selfplay_mode = args.selfplay_mode
                if selfplay_mode == "mixed":
                    selfplay_mode = "correct" if mixed_counter % 2 == 0 else "incorrect"
                    mixed_counter += 1
                
                expected_assessor = "CORRECT" if selfplay_mode == "correct" else "INCORRECT"
                prompt_intent = args.prompt_intent
                if selfplay_mode == "incorrect":
                    if error_type:
                        prompt_intent = f"Introduce a {error_type} error while keeping the note realistic."
                    else:
                        prompt_intent = "Introduce a subtle clinical error while keeping the note realistic."
                
                # Retry loop for this note (max 4 attempts)
                success = False
                for attempt_num in range(1, MAX_RETRIES_PER_NOTE + 1):
                    # Build prompt for this attempt
                    messages = build_messages(
                        "injector",
                        original_note,
                        prompt_intent,
                        assessor_prompts,
                        injector_prompts,
                        args.thinking_budget,
                        selfplay_mode == "correct",
                    )
                    prompt = build_prompt(messages)
                    
                    # Generate (single note)
                    outputs = generate_injector_outputs([prompt])
                    stats["attempted_generations"] += 1
                    generated = outputs[0]
                    
                    # Parse output
                    parsed_output = None
                    if HAS_PARSE_CHANGES:
                        parsed_output = parse_raw_output(generated)
                        candidate_note = parsed_output.get("generated_note")
                    else:
                        candidate_note = extract_generated_note(generated)
                    
                    # Apply filters
                    filter_meta = passes_similarity_filter(
                        original_note,
                        candidate_note or "",
                        args.selfplay_min_jaccard,
                        args.selfplay_max_jaccard,
                    )
                    
                    # Word-level edit check
                    word_edit_info = {}
                    if filter_meta["passed"] and candidate_note and HAS_PARSE_CHANGES:
                        word_diff = compute_word_level_diff(original_note, candidate_note)
                        word_edit_info = {
                            "total_word_edits": word_diff["total_word_edits"],
                            "changes": word_diff.get("changes", [])[:3],
                        }
                        max_edits = getattr(args, 'max_word_edits', 6)
                        if word_diff["total_word_edits"] > max_edits:
                            filter_meta["passed"] = False
                            filter_meta["reason"] = "too_many_edits"
                            stats["rejected_too_many_edits"] += 1
                    
                    # Update stats
                    stats["attempts_logged"] += 1
                    if filter_meta["passed"]:
                        stats["attempts_passed_filter"] += 1
                    else:
                        stats["attempts_failed_filter"] += 1
                    
                    # Verbose diff logging
                    if getattr(args, 'verbose_diff', False) and candidate_note and HAS_PARSE_CHANGES:
                        changes_made = parsed_output.get("changes_made") if parsed_output else None
                        diff_output = get_change_diff(
                            original_note,
                            candidate_note,
                            changes_made,
                            colorize=True
                        )
                        print(f"\n--- Note {note_id} Attempt {attempt_num}/{MAX_RETRIES_PER_NOTE} ---")
                        print(diff_output)
                        print(f"Filter: {'PASS' if filter_meta['passed'] else 'FAIL'} ({filter_meta.get('reason', 'ok')})")
                        print("-" * 60)
                    
                    # Build log entry
                    log_entry = {
                        "run_name": run_name,
                        "note_id": note_id,
                        "error_type": error_type or None,
                        "selfplay_mode": selfplay_mode,
                        "prompt_intent": prompt_intent,
                        "attempt_index": attempt_num,
                        "original_note": original_note,
                        "generated_note": candidate_note,
                        "score_jaccard": filter_meta["score_jaccard"],
                        "filter_passed": filter_meta["passed"],
                        "filter_reason": filter_meta["reason"],
                        "injector_raw_output": generated,
                    }
                    
                    if parsed_output and HAS_PARSE_CHANGES:
                        log_entry["model_changes"] = parsed_output.get("changes_made")
                        log_entry["word_edit_info"] = word_edit_info
                    
                    attempts_handle.write(json.dumps(log_entry, ensure_ascii=False) + "\n")
                    attempts_handle.flush()
                    
                    # Check if passed
                    if filter_meta["passed"]:
                        stats["accepted_generations"] += 1
                        success = True
                        
                        # Add to pending for assessor batch
                        pending.append({
                            "run_name": run_name,
                            "note_id": note_id,
                            "error_type": error_type or None,
                            "selfplay_mode": selfplay_mode,
                            "assessor_expected": expected_assessor,
                            "prompt_intent": prompt_intent,
                            "original_note": original_note,
                            "generated_note": candidate_note,
                            "score_jaccard": filter_meta["score_jaccard"],
                            "injector_raw_output": generated,
                        })
                        break  # Success - move to next note
                    else:
                        # Track rejection reasons
                        reason = filter_meta["reason"]
                        if reason == "empty_generated":
                            stats["rejected_empty"] += 1
                        elif reason == "no_word_change":
                            stats["rejected_no_word_change"] += 1
                        elif reason == "low_jaccard":
                            stats["rejected_low_jaccard"] += 1
                        elif reason == "too_similar":
                            stats["rejected_too_similar"] += 1
                        # Continue to next attempt (unless we've hit max)
                
                # If failed all attempts, log warning
                if not success:
                    print(f"[WARNING] Note {note_id} failed all {MAX_RETRIES_PER_NOTE} attempts")
                
                # Batch assess when we have enough
                if len(pending) >= args.selfplay_assessor_batch_size:
                    assessed = assess_batch(pending)
                    for row in assessed:
                        out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    out_handle.flush()
                    pending = []
                
                pbar.update(1)

        if pending:
            assessed = assess_batch(pending)
            for row in assessed:
                out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_handle.flush()

    write_summary(summary_path, stats)
    print(f"Wrote attempt log to {attempts_path}")


def main() -> None:
    args = parse_args()
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    random.seed(args.seed)
    run_name = args.run_name or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"quick_test_{run_name}.jsonl")
    summary_path = os.path.join(args.output_dir, f"quick_test_{run_name}_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)

    records = []
    if args.jsonl_file:
        records = load_records(args.jsonl_file)
        random.shuffle(records)

    if not records and not args.input_note:
        raise SystemExit("Provide --jsonl-file or --input-note.")

    assessor_prompts = load_assessor_prompts(args.assessor_prompt_file)
    injector_prompts = load_injector_prompts(args.injector_prompt_file)
    
    # Debug: verify prompt loading
    if assessor_prompts:
        print(f"[INFO] Loaded assessor prompts from: {args.assessor_prompt_file}")
        print(f"[INFO] Assessor system prompt preview: {assessor_prompts['system_prompt'][:150]}...")
    else:
        print("[WARNING] Using hardcoded fallback assessor prompt (file not loaded)")
    
    if injector_prompts:
        print(f"[INFO] Loaded injector prompts from: {args.injector_prompt_file}")
    else:
        print("[WARNING] Using hardcoded fallback injector prompts")

    tokenizer_source = args.adapter_dir or args.model_name
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Set left padding for decoder-only models (critical for batched generation)
    tokenizer.padding_side = 'left'

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype="auto",
        device_map="auto",
    )
    model = base_model
    if args.adapter_dir:
        model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()
    model_type = detect_model_type(args.model_name, args.adapter_dir)
    use_qwen_thinking = model_type == MODEL_TYPE_QWEN and args.thinking_budget > 0

    embedder = None
    embedder_tokenizer = None
    embed_device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.selfplay and not args.disable_embedding:
        embedder_tokenizer, embedder = load_embedding_model(args.embedding_model, embed_device)

    if args.selfplay:
        run_selfplay_loop(
            args=args,
            model=model,
            tokenizer=tokenizer,
            model_type=model_type,
            records=records,
            assessor_prompts=assessor_prompts,
            injector_prompts=injector_prompts,
            run_name=run_name,
            output_path=output_path,
            summary_path=summary_path,
        )
        print(f"Wrote self-play results to {output_path}")
        print(f"Wrote summary to {summary_path}")
        return

    results_rows: List[Dict] = []
    scenario_stats: Dict[str, Dict[str, int]] = {}
    scenario_samples_map: Dict[str, List[Dict]] = {}

    total_examples = 0
    for scenario in scenarios:
        samples = scenario_samples(records, scenario, args.num_samples)
        if not samples and args.input_note:
            samples = [{"correct_note": args.input_note, "incorrect_note": args_input_note}]
        scenario_samples_map[scenario] = samples
        total_examples += len(samples)

    with open(output_path, "a", encoding="utf-8") as out_handle:
        with tqdm(total=total_examples, desc="Inference", unit="ex") as pbar:
            for scenario in scenarios:
                samples = scenario_samples_map[scenario]
                correct = 0
                total = 0

                prompts: List[str] = []
                metas: List[Dict] = []
                for record in samples:
                    if scenario == "assessor_correct":
                        note = record.get("correct_note")
                        expected = "CORRECT"
                        prompt_intent = "Assess note correctness."
                        messages = build_messages(
                            "assessor",
                            note,
                            prompt_intent,
                            assessor_prompts,
                            injector_prompts,
                            args.thinking_budget,
                            None,
                        )
                    elif scenario == "assessor_incorrect":
                        note = record.get("incorrect_note")
                        expected = "INCORRECT"
                        prompt_intent = "Assess note correctness."
                        messages = build_messages(
                            "assessor",
                            note,
                            prompt_intent,
                            assessor_prompts,
                            injector_prompts,
                            args.thinking_budget,
                            None,
                        )
                    elif scenario == "injector_correct":
                        note = record.get("correct_note")
                        expected = "CORRECT"
                        prompt_intent = args.prompt_intent
                        messages = build_messages(
                            "injector",
                            note,
                            prompt_intent,
                            assessor_prompts,
                            injector_prompts,
                            args.thinking_budget,
                            True,
                        )
                    else:
                        note = record.get("correct_note")
                        expected = "INCORRECT"
                        error_type = (record.get("error_type") or "").strip()
                        if error_type:
                            prompt_intent = f"Introduce a {error_type} error while keeping the note realistic."
                        else:
                            prompt_intent = "Introduce a subtle clinical error while keeping the note realistic."
                        messages = build_messages(
                            "injector",
                            note,
                            prompt_intent,
                            assessor_prompts,
                            injector_prompts,
                            args.thinking_budget,
                            False,
                        )

                    if use_qwen_thinking:
                        prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=True,
                        )
                    else:
                        prompt = tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                        )
                    prompts.append(prompt)
                    metas.append(
                        {
                            "record": record,
                            "note": note,
                            "expected": expected,
                            "prompt_intent": prompt_intent,
                            "is_injector": scenario.startswith("injector"),
                            "prompt": prompt,
                        }
                    )

                if use_qwen_thinking:
                    for meta in metas:
                        generated = generate_qwen_with_thinking(
                            model,
                            tokenizer,
                            meta["prompt"],
                            args.thinking_budget,
                            args.max_new_tokens,
                            args.temperature,
                            args.top_p,
                        )
                        predicted = extract_final_answer(generated)
                        if predicted is None and args.force_answer:
                            predicted = force_answer_from_prompt(
                                meta["prompt"],
                                tokenizer,
                                model,
                                model.device,
                            )
                        is_correct = predicted == meta["expected"]
                        total += 1
                        correct += int(is_correct)

                        generated_note = None
                        score_jaccard = None
                        score_embedding = None

                        if meta["is_injector"]:
                            generated_note = extract_generated_note(generated)

                        if generated_note:
                            score_jaccard = jaccard_similarity(meta["note"], generated_note)
                            if embedder is not None and embedder_tokenizer is not None:
                                embeddings = embed_texts(
                                    embedder_tokenizer,
                                    embedder,
                                    embed_device,
                                    [meta["note"], generated_note],
                                )
                                score_embedding = cosine_similarity(embeddings[0], embeddings[1])

                        row = {
                            "run_name": run_name,
                            "scenario": scenario,
                            "task_group": "injector" if meta["is_injector"] else "assessor",
                            "note_id": meta["record"].get("note_id"),
                            "error_type": meta["record"].get("error_type"),
                            "prompt_intent": meta["prompt_intent"],
                            "original_note": meta["note"],
                            "generated_note": generated_note,
                            "expected": meta["expected"],
                            "predicted": predicted,
                            "match": is_correct,
                            "score_jaccard": score_jaccard,
                            "score_embedding_cosine": score_embedding,
                            "raw_output": generated,
                        }
                        results_rows.append(row)
                        out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out_handle.flush()
                        pbar.update(1)
                else:
                    for start in range(0, len(prompts), args.batch_size):
                        batch_prompts = prompts[start:start + args.batch_size]
                        batch_metas = metas[start:start + args.batch_size]
                        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
                        input_length = inputs["input_ids"].shape[1]

                        with torch.no_grad():
                            outputs = model.generate(
                                **inputs,
                                max_new_tokens=args.max_new_tokens,
                                do_sample=True,
                                temperature=args.temperature,
                                top_p=args.top_p,
                                pad_token_id=tokenizer.eos_token_id,
                            )

                        for idx, meta in enumerate(batch_metas):
                            generated_tokens = outputs[idx][input_length:]
                            generated = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                            predicted = extract_final_answer(generated)
                            if predicted is None:
                                full_text = tokenizer.decode(outputs[idx], skip_special_tokens=True)
                                predicted = extract_final_answer(full_text)
                            if predicted is None and args.force_answer:
                                predicted = force_answer_from_prompt(
                                    meta["prompt"],
                                    tokenizer,
                                    model,
                                    model.device,
                                )
                            is_correct = predicted == meta["expected"]
                            total += 1
                            correct += int(is_correct)

                            generated_note = None
                            score_jaccard = None
                            score_embedding = None

                            if meta["is_injector"]:
                                generated_note = extract_generated_note(generated)

                            if generated_note:
                                score_jaccard = jaccard_similarity(meta["note"], generated_note)
                                if embedder is not None and embedder_tokenizer is not None:
                                    embeddings = embed_texts(
                                        embedder_tokenizer,
                                        embedder,
                                        embed_device,
                                        [meta["note"], generated_note],
                                    )
                                    score_embedding = cosine_similarity(embeddings[0], embeddings[1])

                            row = {
                                "run_name": run_name,
                                "scenario": scenario,
                                "task_group": "injector" if meta["is_injector"] else "assessor",
                                "note_id": meta["record"].get("note_id"),
                                "error_type": meta["record"].get("error_type"),
                                "prompt_intent": meta["prompt_intent"],
                                "original_note": meta["note"],
                                "generated_note": generated_note,
                                "expected": meta["expected"],
                                "predicted": predicted,
                                "match": is_correct,
                                "score_jaccard": score_jaccard,
                                "score_embedding_cosine": score_embedding,
                                "raw_output": generated,
                            }
                            results_rows.append(row)
                            out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                            out_handle.flush()
                            pbar.update(1)

                scenario_stats[scenario] = {"total": total, "correct": correct}

    write_summary(summary_path, summarize_results(results_rows))
    print(f"Wrote {len(results_rows)} rows to {output_path}")
    print(f"Wrote summary to {summary_path}")
    for scenario, stats in scenario_stats.items():
        total = stats["total"]
        correct = stats["correct"]
        acc = (correct / total * 100) if total else 0.0
        print(f"{scenario}: {correct}/{total} ({acc:.1f}%)")


if __name__ == "__main__":
    main()
