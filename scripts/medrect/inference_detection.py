#!/usr/bin/env python3
"""
Medical Error Detection + Localization Inference

Task  : given numbered clinical sentences, output CORRECT or the error sentence number.
Reuses: load_model_and_tokenizer, detect_model_type, load_test_data, Qwen3 token
        constants from scripts/inference_error_detection.py.

Metrics
-------
- Detection  : precision / recall / F1  (error vs no-error)
- Localization: exact sentence-number match among detected errors

Usage
-----
    python scripts/medrect/inference_detection.py \\
        --model_path FreedomIntelligence/HuatuoGPT-o1-7B \\
        --dataset all --batch_size 8 --mode no-thinking
"""

import json
import os
import re
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm

# ── resolve project root and import shared helpers ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from inference_error_detection import (
    detect_model_type,
    load_model_and_tokenizer,
    load_test_data,
)
from self_play.utils import parse_assessor_answer

DEFAULT_PROMPT_CONFIG = PROJECT_ROOT / "configs" / "prompts" / "detection_localization_prompts.json"

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_TYPE_QWEN = "qwen"


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_prompt_config(path: Path = DEFAULT_PROMPT_CONFIG) -> Dict:
    with open(path) as f:
        return json.load(f)


def sentences_to_1indexed(raw: str) -> str:
    """Convert MEDEC 0-indexed sentences to 1-indexed.

    "0 First sentence.\\n1 Second." → "1. First sentence.\\n2. Second."
    """
    lines = []
    for line in raw.strip().split("\n"):
        m = re.match(r"^(\d+)\s+(.+)$", line.strip())
        if m:
            lines.append(f"{int(m.group(1)) + 1}. {m.group(2)}")
        elif line.strip() and lines:
            lines[-1] = lines[-1] + " " + line.strip()
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Output parsing
# ─────────────────────────────────────────────────────────────────────────────

def split_thinking(content: str) -> Tuple[str, str]:
    """Split model thinking from the final answer.

    Handles explicit think tags emitted in text form.
    Returns (thinking, answer_after_think).
    """
    m = re.search(r"<think>(.*?)</think>\s*", content, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip(), content[m.end():].strip()
    return "", content


def parse_output(content: str) -> Tuple[str, Optional[int]]:
    """Return (label, sentence_id).

    label        : "CORRECT" | "ERROR" | "UNKNOWN"
    sentence_id  : int if label is ERROR, else None
    """
    return parse_assessor_answer(content)


def parse_output_candidates(*candidates: str) -> Tuple[str, Optional[int]]:
    """Try parsing multiple candidate texts, from most to least trusted."""
    for text in candidates:
        if not text or not text.strip():
            continue
        label, sid = parse_output(text)
        if label != "UNKNOWN":
            return label, sid

        # Last-line fallback: useful when the answer was serialized after the
        # thinking block but the main split missed it.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            last = lines[-1]
            if re.fullmatch(r"\d+", last):
                return "ERROR", int(last)
            if re.fullmatch(r"CORRECT", last, flags=re.IGNORECASE):
                return "CORRECT", None

    return "UNKNOWN", None


def build_generation_kwargs(
    tokenizer,
    *,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    min_p: float,
    presence_penalty: float,
    repetition_penalty: float,
) -> Dict:
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": True,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
        "min_p": min_p,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if presence_penalty > 0:
        kwargs["presence_penalty"] = presence_penalty
    return kwargs


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    model_type: str,
    prompt_config: Dict,
    *,
    use_thinking: bool = True,
    max_samples: Optional[int] = None,
    temperature: float = 0.7,
    top_p: float = 0.8,
    top_k: int = 20,
    min_p: float = 0.0,
    presence_penalty: float = 0.0,
    max_new_tokens: int = 16384,
    thinking_budget: int = 32768,
    batch_size: int = 8,
) -> List[Dict]:
    if max_samples:
        test_df = test_df.head(max_samples)

    model.eval()
    is_qwen = model_type == MODEL_TYPE_QWEN
    tokenizer.padding_side = "left"

    system_prompt = prompt_config["system_prompt"]
    user_template = prompt_config["user_template"]
    results: List[Dict] = []

    n = len(test_df)
    for batch_start in tqdm(range(0, n, batch_size), desc="Inference"):
        batch = test_df.iloc[batch_start : batch_start + batch_size]
        prompts, meta = [], []

        for _, row in batch.iterrows():
            error_flag = int(row["Error Flag"])
            sentences = sentences_to_1indexed(str(row.get("Sentences", row.get("Text", ""))))

            raw_sid = row.get("Error Sentence ID")
            if error_flag == 1 and pd.notna(raw_sid):
                gt_sid = int(raw_sid) + 1  # 0-indexed → 1-indexed
                gt_label = str(gt_sid)
            else:
                gt_sid, gt_label = None, "CORRECT"

            meta.append(
                dict(
                    text_id=str(row.get("Text ID", "")),
                    dataset=str(row.get("dataset", "")),
                    error_flag=error_flag,
                    gt_label=gt_label,
                    gt_sid=gt_sid,
                    error_type=str(row.get("Error Type", "")) if pd.notna(row.get("Error Type")) else "",
                    sentences=sentences,
                )
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_template.format(sentences=sentences)},
            ]
            prompt_kwargs = dict(tokenize=False, add_generation_prompt=True)
            if is_qwen:
                prompt_kwargs["enable_thinking"] = bool(use_thinking)
            prompts.append(tokenizer.apply_chat_template(messages, **prompt_kwargs))

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(model.device)
        gen_kwargs = build_generation_kwargs(
            tokenizer,
            max_new_tokens=(thinking_budget + 256) if (is_qwen and use_thinking) else max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repetition_penalty=1.05,
        )
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        orig_padded_len = inputs.input_ids.shape[1]

        for i, m in enumerate(meta):
            out_ids = outputs[i, orig_padded_len:].tolist()
            while out_ids and out_ids[-1] == tokenizer.pad_token_id:
                out_ids.pop()

            full_text = tokenizer.decode(out_ids, skip_special_tokens=True).strip()
            thinking, content = split_thinking(full_text)
            if not content:
                content = full_text

            pred_type, pred_sid = parse_output_candidates(content, full_text, thinking)
            pred_label = "CORRECT" if pred_type == "CORRECT" else (str(pred_sid) if pred_sid is not None else "UNKNOWN")

            detection_correct = (m["gt_label"] == "CORRECT" and pred_label == "CORRECT") or (
                m["gt_label"] != "CORRECT" and pred_label not in ("CORRECT", "UNKNOWN")
            )
            localization_correct = (
                m["gt_sid"] is not None and pred_sid is not None and m["gt_sid"] == pred_sid
            )

            results.append(
                dict(
                    text_id=m["text_id"],
                    dataset=m["dataset"],
                    note=m["sentences"],
                    error_type=m["error_type"],
                    gt_label=m["gt_label"],
                    gt_sid=m["gt_sid"],
                    pred_label=pred_label,
                    pred_sid=pred_sid,
                    detection_correct=detection_correct,
                    localization_correct=localization_correct,
                    thinking=thinking,
                    raw_output=content,
                )
            )

            if batch_start == 0 and i == 0:
                print(f"\n{'='*50}\nDEBUG first sample\nGT={m['gt_label']}  Pred={pred_label}\n--- THINKING ---\n{thinking}\n--- ANSWER ---\n{content}\n{'='*50}\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: List[Dict]) -> Dict:
    total = len(results)
    error_cases   = [r for r in results if r["gt_label"] != "CORRECT"]
    correct_cases = [r for r in results if r["gt_label"] == "CORRECT"]

    tp = sum(1 for r in error_cases   if r["pred_label"] not in ("CORRECT", "UNKNOWN"))
    fn = sum(1 for r in error_cases   if r["pred_label"] in  ("CORRECT", "UNKNOWN"))
    tn = sum(1 for r in correct_cases if r["pred_label"] == "CORRECT")
    fp = sum(1 for r in correct_cases if r["pred_label"] != "CORRECT")

    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    detected = [r for r in error_cases if r["pred_label"] not in ("CORRECT", "UNKNOWN")]
    sent_ok  = sum(1 for r in error_cases if r["localization_correct"])

    by_type: Dict = {}
    for r in error_cases:
        et = r["error_type"] or "unknown"
        s  = by_type.setdefault(et, {"total": 0, "detected": 0, "sentence_correct": 0})
        s["total"] += 1
        if r["pred_label"] not in ("CORRECT", "UNKNOWN"):
            s["detected"] += 1
        if r["localization_correct"]:
            s["sentence_correct"] += 1

    det_acc = sum(1 for r in results if r["detection_correct"]) / total if total else 0

    return dict(
        total_samples=total,
        detection=dict(
            accuracy=det_acc, precision=prec, recall=rec, f1=f1,
            tp=tp, fp=fp, tn=tn, fn=fn,
        ),
        sentence_extraction=dict(
            total_errors=len(error_cases),
            detected_errors=len(detected),
            exact_matches=sent_ok,
            accuracy=sent_ok / len(error_cases) if error_cases else 0,
        ),
        by_error_type=by_type,
    )


def print_metrics(m: Dict) -> None:
    det, sent = m["detection"], m["sentence_extraction"]
    print(f"\n{'='*50}")
    print("Detection (error vs no-error)")
    print(f"  Accuracy : {det['accuracy']:.3f}")
    print(f"  Precision: {det['precision']:.3f}  Recall: {det['recall']:.3f}  F1: {det['f1']:.3f}")
    print(f"  TP={det['tp']} FP={det['fp']} TN={det['tn']} FN={det['fn']}")
    print(f"\nSentence extraction (exact sentence match)")
    print(f"  Gold error cases : {sent['total_errors']}")
    print(f"  Predicted errors  : {sent['detected_errors']}")
    print(f"  Exact matches     : {sent['exact_matches']}")
    print(f"  Accuracy          : {sent['accuracy']:.3f}")
    print(f"\nBy error type")
    for et, s in sorted(m["by_error_type"].items(), key=lambda x: -x[1]["total"]):
        dr = s["detected"] / s["total"] if s["total"] else 0
        sr = s["sentence_correct"] / s["total"] if s["total"] else 0
        print(f"  {et}: n={s['total']}  det_recall={dr:.2f}  sent_acc={sr:.2f}")
    print(f"{'='*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Medical Error Detection + Localization")
    p.add_argument("--model_path",      required=True)
    p.add_argument("--prompt_config",   default=str(DEFAULT_PROMPT_CONFIG))
    p.add_argument("--dataset",         default="all", choices=["ms", "uw", "all"])
    p.add_argument("--max_samples",     type=int, default=None)
    p.add_argument("--batch_size",      type=int, default=8)
    p.add_argument("--temperature",     type=float, default=None)
    p.add_argument("--top_p",           type=float, default=None)
    p.add_argument("--top_k",           type=int, default=20)
    p.add_argument("--min_p",           type=float, default=0.0)
    p.add_argument("--presence_penalty", type=float, default=0.0)
    p.add_argument("--max_new_tokens",  type=int, default=None)
    p.add_argument("--thinking_budget", type=int, default=None)
    p.add_argument("--mode",            choices=["thinking", "no-thinking"], default=None)
    p.add_argument("--no_thinking",     action="store_true", help="Disable thinking mode")
    p.add_argument("--output_dir",      default="results/detection")
    p.add_argument("--base_model_path", default=None,
                   help="Override base-model path for LoRA adapters "
                        "(use when running offline and adapter_config "
                        "points to an HF hub ID)")
    p.add_argument("--shard_id",   type=int, default=0,
                   help="Index of this shard (0-based)")
    p.add_argument("--num_shards", type=int, default=1,
                   help="Total number of shards (1 = no sharding)")
    args = p.parse_args()
    if args.mode is None:
        args.mode = "no-thinking" if args.no_thinking else "thinking"
    use_thinking = args.mode == "thinking"
    if args.temperature is None:
        args.temperature = 0.6 if use_thinking else 0.7
    if args.top_p is None:
        args.top_p = 0.95 if use_thinking else 0.8
    if args.thinking_budget is None:
        args.thinking_budget = 32768
    if args.max_new_tokens is None:
        args.max_new_tokens = 16384

    model_type = detect_model_type(args.model_path)

    print(f"\n{'='*50}")
    print(f"Model   : {args.model_path}  ({model_type})")
    print(f"Dataset : {args.dataset}  |  Mode: {args.mode}  |  Thinking: {use_thinking}")
    print(
        f"Sampling: temperature={args.temperature} top_p={args.top_p} "
        f"top_k={args.top_k} min_p={args.min_p} presence_penalty={args.presence_penalty}"
    )
    if args.num_shards > 1:
        print(f"Shard   : {args.shard_id + 1}/{args.num_shards}")
    print(f"{'='*50}\n")

    model, tokenizer = load_model_and_tokenizer(
        args.model_path, model_type,
        base_model_override=args.base_model_path,
    )
    prompt_config    = load_prompt_config(Path(args.prompt_config))
    test_df          = load_test_data(args.dataset)

    if args.num_shards > 1:
        test_df = test_df.iloc[args.shard_id::args.num_shards].reset_index(drop=True)

    # ── Run inference ────────────────────────────────────────────────────────────
    results = run_inference(
        model,
        tokenizer,
        test_df,
        model_type,
        prompt_config,
        use_thinking=use_thinking,
        max_samples=args.max_samples,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        max_new_tokens=args.max_new_tokens,
        thinking_budget=args.thinking_budget,
        batch_size=args.batch_size,
    )

    # ── Evaluate and save results ────────────────────────────────────────────────
    df = pd.DataFrame(results)
    metrics = compute_metrics(results)
    print_metrics(metrics)

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    mode_slug = args.mode.replace("-", "_")
    sample_slug = f"n{len(results)}"
    shard_slug = f"_shard{args.shard_id}of{args.num_shards}" if args.num_shards > 1 else ""
    if use_thinking:
        generation_slug = f"{mode_slug}_tb{args.thinking_budget}_{sample_slug}{shard_slug}"
    else:
        generation_slug = f"{mode_slug}_max{args.max_new_tokens}_{sample_slug}{shard_slug}"

    results_file = f"{args.output_dir}/{args.dataset}_{generation_slug}_{ts}.jsonl"
    with open(results_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_file = f"{args.output_dir}/{args.dataset}_{generation_slug}_{ts}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            dict(model_path=args.model_path, dataset=args.dataset,
                 timestamp=ts, prompt_config=args.prompt_config,
                 generation=dict(
                     mode=args.mode,
                     thinking=use_thinking,
                     temperature=args.temperature,
                     top_p=args.top_p,
                     top_k=args.top_k,
                     min_p=args.min_p,
                     presence_penalty=args.presence_penalty,
                     max_new_tokens=args.max_new_tokens,
                     thinking_budget=args.thinking_budget,
                 ),
                 metrics=metrics),
            f, indent=2,
        )

    print(f"Results : {results_file}")
    print(f"Summary : {summary_file}")


if __name__ == "__main__":
    main()
