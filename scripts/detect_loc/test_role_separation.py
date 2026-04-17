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

from scripts.self_play.utils import strip_thinking


def load_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


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
        if device == "auto":
            if torch.cuda.is_available():
                device_map = "auto"
                dtype = "auto"
            elif torch.backends.mps.is_available():
                device_map = "mps"
                dtype = torch.float32
            else:
                device_map = "cpu"
                dtype = torch.float32
        else:
            device_map = device
            dtype = "auto" if device != "cpu" else torch.float32

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            dtype=dtype,
            trust_remote_code=True,
        )
        self.model.eval()

    def generate(self, messages, *, temperature: float, max_new_tokens: int, enable_thinking: bool):
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[1]
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if temperature > 0:
            kwargs["temperature"] = temperature
            kwargs["top_p"] = 0.95
        with torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)
        new_ids = out[0, prompt_len:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)


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
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--think", action="store_true", default=False)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    ass_row, inj_row = load_pair(Path(args.assessor_data), Path(args.injector_data), args.base_id)
    base_id = ass_row["sample_id"]
    ass_note = extract_note_block(ass_row.get("user_prompt", ""))
    inj_note = extract_note_block(inj_row.get("user_prompt", ""))
    diff_ids = differing_sentence_ids(ass_note, inj_note)

    print(f"Model            : {args.model_path}")
    print(f"Base sample      : {base_id}")
    print(f"Thinking         : {args.think}")
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
        enable_thinking=args.think,
    )
    inj_output = model.generate(
        inj_messages,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        enable_thinking=args.think,
    )

    ass_thinking, ass_answer = strip_thinking(ass_output)
    inj_thinking, inj_answer = strip_thinking(inj_output)

    print_block("ASSESSOR PROMPT", ass_row["user_prompt"])
    print_block("ASSESSOR TRAIN TARGET", ass_row["label"])
    print_block("ASSESSOR MODEL THINKING", ass_thinking or "<empty>")
    print_block("ASSESSOR MODEL ANSWER", ass_answer or ass_output)

    print_block("INJECTOR PROMPT", inj_row["user_prompt"])
    print_block("INJECTOR TRAIN TARGET", inj_row["label"])
    print_block("INJECTOR MODEL THINKING", inj_thinking or "<empty>")
    print_block("INJECTOR MODEL ANSWER", inj_answer or inj_output)


if __name__ == "__main__":
    main()
