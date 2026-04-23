#!/usr/bin/env python3
"""Run a paired injector vs assessor prompt on the same base sample.

Purpose:
  Diagnose whether a mixed-role model stays in assessor mode when prompted
  as an assessor, or whether injector/correction behavior leaks into the
  reasoning/output.

Default data sources are the mixed SFT files used by run_medrect_mixed_sft.sh.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def load_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def find_subsequence(haystack: list[int], needle: list[int]) -> int:
    if not needle or len(needle) > len(haystack):
        return -1
    last_start = len(haystack) - len(needle)
    for start in range(last_start + 1):
        if haystack[start : start + len(needle)] == needle:
            return start
    return -1


def parse_injector_sid(label: str) -> Optional[int]:
    m = re.match(r"^\s*(\d+)\.", label or "")
    return int(m.group(1)) if m else None


def extract_note_block(text: str) -> str:
    lines = (text or "").splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*1\.\s+", line):
            start = i
            break
    if start is None:
        return (text or "").strip()
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].strip().startswith("Respond with EXACTLY"):
            end = i
            break
    return "\n".join(lines[start:end]).strip()


def split_numbered_sentences(note: str) -> list[str]:
    return [line.strip() for line in note.splitlines() if re.match(r"^\d+\.\s+", line.strip())]


def differing_sentence_ids(note_a: str, note_b: str) -> list[int]:
    a = split_numbered_sentences(note_a)
    b = split_numbered_sentences(note_b)
    diff = []
    for i, (la, lb) in enumerate(zip(a, b), start=1):
        if la != lb:
            diff.append(i)
    if len(a) != len(b):
        diff.extend(range(min(len(a), len(b)) + 1, max(len(a), len(b)) + 1))
    return diff


def load_pair(assessor_path: Path, injector_path: Path, base_id: Optional[str]):
    assessor = {}
    for row in load_jsonl(assessor_path):
        sample_id = row.get("sample_id", "")
        normalized_id = re.sub(r"_[01]$", "", sample_id)
        existing = assessor.get(normalized_id)
        if existing is None or (not str(existing.get("label", "")).isdigit() and str(row.get("label", "")).isdigit()):
            assessor[normalized_id] = row

    if base_id:
        inj_key = f"{base_id}_injector_error"
        inj_row = None
        for row in load_jsonl(injector_path):
            if row.get("sample_id") == inj_key:
                inj_row = row
                break
        if not inj_row:
            raise SystemExit(f"Injector row not found for base_id={base_id}")
        ass_row = assessor.get(base_id)
        if not ass_row:
            raise SystemExit(f"Assessor row not found for base_id={base_id}")
        return ass_row, inj_row

    for inj_row in load_jsonl(injector_path):
        bid = inj_row["sample_id"].replace("_injector_error", "")
        ass_row = assessor.get(bid)
        if not ass_row:
            continue
        ass_sid = ass_row.get("label")
        inj_sid = parse_injector_sid(inj_row.get("label", ""))
        ass_note = extract_note_block(ass_row.get("user_prompt", ""))
        inj_note = extract_note_block(inj_row.get("user_prompt", ""))
        diff = differing_sentence_ids(ass_note, inj_note)
        if str(ass_sid).isdigit() and inj_sid is not None and int(ass_sid) == inj_sid and diff == [int(ass_sid)]:
            return ass_row, inj_row

    raise SystemExit("No clean matched pair found")


class ChatModel:
    def __init__(self, model_path: str, device: str = "auto"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if device == "auto":
            if torch.cuda.is_available():
                device_map = "auto"
                # Keep this debug script numerically stable during sampling.
                dtype = torch.float32
            elif torch.backends.mps.is_available():
                device_map = "mps"
                dtype = torch.float32
            else:
                device_map = "cpu"
                dtype = torch.float32
        else:
            device_map = device
            dtype = torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, messages, *, temperature: float, max_new_tokens: int, enable_thinking: bool):
        template_kwargs = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if enable_thinking:
            template_kwargs["enable_thinking"] = True
        prompt = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]

        generation_config = copy.deepcopy(self.model.generation_config)
        generation_config.pad_token_id = self.tokenizer.pad_token_id
        generation_config.eos_token_id = self.tokenizer.eos_token_id
        generation_config.max_new_tokens = max_new_tokens
        generation_config.do_sample = temperature > 0

        # Avoid NaN/Inf issues during sampling
        if temperature > 0:
            generation_config.temperature = temperature
        else:
            generation_config.temperature = None
            generation_config.top_p = None
            generation_config.top_k = None

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                generation_config=generation_config,
            )
        new_ids = out[0, prompt_len:].tolist()
        raw = self.tokenizer.decode(new_ids, skip_special_tokens=False)
        answer = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        thinking = ""

        if enable_thinking:
            import re
            # Match <think>...</think> taking into account newlines
            m = re.search(r"<think>(.*?)</think>\s*", raw, flags=re.DOTALL)
            if m:
                thinking = m.group(1).strip()
                # Remove the think block from the decoded answer
                # Notice we need to be careful with answer, as it has special tokens stripped
                raw_ans = raw[m.end():]
                # Re-decode just the remaining ids to be safe or use raw_ans
                answer = re.sub(r"<\|.*?\|>", "", raw_ans).strip()

        return {
            "thinking": thinking,
            "answer": answer.strip(),
            "raw": raw.strip(),
        }


def print_block(title: str, text: str):
    print("=" * 80)
    print(title)
    print("=" * 80)
    print(text.strip())
    print()


def main():
    parser = argparse.ArgumentParser(description="Test role separation on one paired SFT sample.")
    parser.add_argument("--model_path", default="Abdine/qwen3-4b-medrect-mixed")
    parser.add_argument("--assessor-data", default="data_processed/medrect/generated_assessor_all_sft.jsonl")
    parser.add_argument("--injector-data", default="data_processed/medrect/injector_error_chains_20260310_135156.jsonl")
    parser.add_argument("--base-id", default=None, help="Base sample id like ms-train-1349")
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--no-think", action="store_true", help="Disable Qwen thinking mode.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    ass_row, inj_row = load_pair(Path(args.assessor_data), Path(args.injector_data), args.base_id)
    base_id = ass_row["sample_id"]
    ass_note = extract_note_block(ass_row.get("user_prompt", ""))
    inj_note = extract_note_block(inj_row.get("user_prompt", ""))
    diff_ids = differing_sentence_ids(ass_note, inj_note)

    print(f"Model            : {args.model_path}")
    print(f"Base sample      : {base_id}")
    enable_thinking = not args.no_think

    print(f"Thinking         : {enable_thinking}")
    print(f"Temperature      : {args.temperature}")
    print(f"Assessor label   : {ass_row.get('label')}")
    print(f"Injector label   : {inj_row.get('label')}")
    print(f"Differing sids   : {diff_ids}")
    print()

    model = ChatModel(args.model_path, device=args.device)

    ass_messages = [
        {"role": "system", "content": ass_row["system_prompt"]},
        {"role": "user", "content": ass_row["user_prompt"]},
    ]
    inj_messages = [
        {"role": "system", "content": inj_row["system_prompt"]},
        {"role": "user", "content": inj_row["user_prompt"]},
    ]

    ass_output = model.generate(
        ass_messages,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=enable_thinking,
    )
    inj_output = model.generate(
        inj_messages,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=enable_thinking,
    )

    print_block("ASSESSOR PROMPT", ass_row["user_prompt"])
    print_block("ASSESSOR TRAIN TARGET", ass_row["label"])
    print_block("ASSESSOR MODEL THINKING", ass_output["thinking"] or "<empty>")
    print_block("ASSESSOR MODEL ANSWER", ass_output["answer"] or ass_output["raw"] or "<empty>")

    print_block("INJECTOR PROMPT", inj_row["user_prompt"])
    print_block("INJECTOR TRAIN TARGET", inj_row["label"])
    print_block("INJECTOR MODEL THINKING", inj_output["thinking"] or "<empty>")
    print_block("INJECTOR MODEL ANSWER", inj_output["answer"] or inj_output["raw"] or "<empty>")


if __name__ == "__main__":
    main()
