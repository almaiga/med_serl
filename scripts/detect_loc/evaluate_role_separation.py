#!/usr/bin/env python3
"""Batch-evaluate role separation on paired mixed-SFT examples.

This script probes a mixed-role model on paired assessor/injector examples
from the same base sample and reports:

- assessor output parse rate
- assessor exact-match rate vs training target
- injector output parse rate
- injector target sentence match rate
- assessor thinking leakage into rewrite/correction language

It is intended as a lightweight diagnostic, not a benchmark.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Optional

import torch
from transformers import LogitsProcessor, LogitsProcessorList


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.self_play.utils import parse_assessor_answer, parse_injector_compact, strip_thinking
from scripts.inference_error_detection import detect_model_type, load_model_and_tokenizer


REWRITE_PATTERNS = [
    r"\bcorrection\b",
    r"\bcorrected sentence\b",
    r"\bshould be changed\b",
    r"\bshould read\b",
    r"\breplace\b",
    r"\bchanging sentence\b",
    r"\bchange sentence\b",
    r"\bmodify sentence\b",
    r"\bmodified sentence\b",
    r"\brewrite\b",
]


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


def load_clean_pairs(assessor_path: Path, injector_path: Path) -> list[tuple[str, dict, dict]]:
    assessor = {}
    for row in load_jsonl(assessor_path):
        sample_id = row.get("sample_id", "")
        base_id = re.sub(r"_[01]$", "", sample_id)
        existing = assessor.get(base_id)
        if existing is None or (not str(existing.get("label", "")).isdigit() and str(row.get("label", "")).isdigit()):
            assessor[base_id] = row

    pairs = []
    for inj_row in load_jsonl(injector_path):
        base_id = inj_row["sample_id"].replace("_injector_error", "")
        ass_row = assessor.get(base_id)
        if not ass_row:
            continue
        ass_sid = ass_row.get("label")
        inj_sid = parse_injector_sid(inj_row.get("label", ""))
        if not (str(ass_sid).isdigit() and inj_sid is not None):
            continue
        ass_note = extract_note_block(ass_row.get("user_prompt", ""))
        inj_note = extract_note_block(inj_row.get("user_prompt", ""))
        diff = differing_sentence_ids(ass_note, inj_note)
        if int(ass_sid) == inj_sid and diff == [int(ass_sid)]:
            pairs.append((base_id, ass_row, inj_row))
    return pairs


def contains_rewrite_language(thinking: str) -> tuple[bool, Optional[str]]:
    text = (thinking or "").lower()
    for pat in REWRITE_PATTERNS:
        m = re.search(pat, text, flags=re.IGNORECASE)
        if m:
            return True, m.group(0)
    return False, None


class ChatModel:
    def __init__(self, model_path: str, device: str = "auto"):
        del device
        model_type = detect_model_type(model_path)
        self.model, self.tokenizer = load_model_and_tokenizer(model_path, model_type)
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
            kwargs["top_p"] = 0.8
            kwargs["top_k"] = 20
            kwargs["logits_processor"] = LogitsProcessorList([_SanitizeLogits()])
        with torch.no_grad():
            out = self.model.generate(**inputs, **kwargs)
        new_ids = out[0, prompt_len:]
        return self.tokenizer.decode(new_ids, skip_special_tokens=True)


class _SanitizeLogits(LogitsProcessor):
    """Upcast logits and scrub NaN/inf before sampling."""

    def __call__(self, _input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        scores = scores.to(torch.float32)
        scores = torch.nan_to_num(scores, nan=-1e4, posinf=1e4, neginf=-1e4)
        return scores


def short(text: str, limit: int = 180) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def main():
    parser = argparse.ArgumentParser(description="Evaluate role separation on paired mixed-role samples.")
    parser.add_argument("--model_path", default="Abdine/qwen3-4b-medrect-mixed")
    parser.add_argument("--assessor-data", default="data_processed/medrect/generated_assessor_all_sft.jsonl")
    parser.add_argument("--injector-data", default="data_processed/medrect/injector_error_chains_20260310_135156.jsonl")
    parser.add_argument("--n", type=int, default=8, help="Number of clean paired samples to test")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--think", action="store_true", default=False)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--show", type=int, default=5, help="Number of flagged examples to print")
    parser.add_argument(
        "--full-output",
        action="store_true",
        default=False,
        help="Print full thinking/answer text for flagged examples instead of truncating",
    )
    parser.add_argument(
        "--output-jsonl",
        default=None,
        help="Optional path to save all probe results as JSONL",
    )
    args = parser.parse_args()

    pairs = load_clean_pairs(Path(args.assessor_data), Path(args.injector_data))
    if not pairs:
        raise SystemExit("No clean paired samples found.")

    rng = random.Random(args.seed)
    if args.n < len(pairs):
        pairs = rng.sample(pairs, args.n)
    else:
        pairs = pairs[: args.n]

    print(f"Model               : {args.model_path}")
    print(f"Thinking            : {args.think}")
    print(f"Temperature         : {args.temperature}")
    print(f"Paired samples pool : {len(load_clean_pairs(Path(args.assessor_data), Path(args.injector_data)))}")
    print(f"Samples tested      : {len(pairs)}")
    print()

    model = ChatModel(args.model_path, device=args.device)

    results = []
    for base_id, ass_row, inj_row in pairs:
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
        ass_label, ass_sid = parse_assessor_answer(ass_output)
        inj_sid, inj_text = parse_injector_compact(inj_output)
        leak, marker = contains_rewrite_language(ass_thinking)

        gt_ass_sid = int(ass_row["label"])
        gt_inj_sid = parse_injector_sid(inj_row["label"])

        results.append(
            {
                "base_id": base_id,
                "assessor_gt": gt_ass_sid,
                "injector_gt": gt_inj_sid,
                "assessor_label": ass_label,
                "assessor_pred_sid": ass_sid,
                "injector_pred_sid": inj_sid,
                "injector_pred_text": inj_text,
                "assessor_exact": ass_label == "ERROR" and ass_sid == gt_ass_sid,
                "injector_sid_match": inj_sid == gt_inj_sid,
                "assessor_parse_ok": ass_label != "UNKNOWN",
                "injector_parse_ok": inj_sid is not None and bool(inj_text),
                "assessor_leak": leak,
                "assessor_leak_marker": marker,
                "assessor_thinking": ass_thinking,
                "assessor_answer": ass_answer or ass_output,
                "injector_answer": inj_answer or inj_output,
            }
        )

    total = len(results)
    assessor_parse_ok = sum(r["assessor_parse_ok"] for r in results)
    assessor_exact = sum(r["assessor_exact"] for r in results)
    injector_parse_ok = sum(r["injector_parse_ok"] for r in results)
    injector_sid_match = sum(r["injector_sid_match"] for r in results)
    leak_count = sum(r["assessor_leak"] for r in results)
    markers = Counter(r["assessor_leak_marker"] for r in results if r["assessor_leak_marker"])

    print("Summary")
    print(f"  Assessor parse ok      : {assessor_parse_ok}/{total} ({100*assessor_parse_ok/total:.1f}%)")
    print(f"  Assessor exact target  : {assessor_exact}/{total} ({100*assessor_exact/total:.1f}%)")
    print(f"  Injector parse ok      : {injector_parse_ok}/{total} ({100*injector_parse_ok/total:.1f}%)")
    print(f"  Injector target match  : {injector_sid_match}/{total} ({100*injector_sid_match/total:.1f}%)")
    print(f"  Assessor rewrite leak  : {leak_count}/{total} ({100*leak_count/total:.1f}%)")
    if markers:
        print("  Leak markers           :")
        for marker, count in markers.most_common():
            print(f"    {marker!r}: {count}")

    flagged = [r for r in results if r["assessor_leak"] or not r["assessor_exact"]]
    if flagged:
        print("\nFlagged examples")
        for row in flagged[: args.show]:
            print("=" * 80)
            print(f"Base ID             : {row['base_id']}")
            print(f"Assessor GT         : {row['assessor_gt']}")
            print(f"Assessor pred       : {row['assessor_label']} {row['assessor_pred_sid']}")
            print(f"Injector GT / pred  : {row['injector_gt']} / {row['injector_pred_sid']}")
            print(f"Leak marker         : {row['assessor_leak_marker']}")
            if args.full_output:
                print("Assessor answer")
                print(row["assessor_answer"])
                print("Injector answer")
                print(row["injector_answer"])
                print("Assessor thinking")
                print(row["assessor_thinking"])
            else:
                print(f"Assessor answer     : {short(row['assessor_answer'])}")
                print(f"Injector answer     : {short(row['injector_answer'])}")
                print(f"Assessor thinking   : {short(row['assessor_thinking'], 260)}")

    if args.output_jsonl:
        out_path = Path(args.output_jsonl)
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = PROJECT_ROOT / "results" / "detection" / f"role_separation_{ts}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for row in results:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"\nSaved raw results    : {out_path}")


if __name__ == "__main__":
    main()
