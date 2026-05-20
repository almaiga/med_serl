#!/usr/bin/env python3
"""
Medical Error Detection + Localization — vLLM backend

Mirrors inference_detection.py but uses vLLM offline inference for higher
throughput via continuous batching + PagedAttention.

Usage
-----
    python scripts/medrect/inference_detection_vllm.py \
        --model_path Abdine/medserl-qwen3-4b-medrect-mixed-selfplay-r2 \
        --mode thinking --max_samples 20

    # Multi-GPU tensor parallelism (all 4 GPUs, one process):
    python scripts/medrect/inference_detection_vllm.py \
        --model_path Abdine/medserl-qwen3-4b-medrect-mixed-selfplay-r2 \
        --mode thinking --tensor_parallel_size 4
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
from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from inference_error_detection import detect_model_type, load_test_data
from self_play.utils import parse_assessor_answer

DEFAULT_PROMPT_CONFIG = PROJECT_ROOT / "configs" / "prompts" / "detection_localization_prompts.json"


# ─────────────────────────────────────────────────────────────────────────────
# Helpers (identical to inference_detection.py)
# ─────────────────────────────────────────────────────────────────────────────

def load_prompt_config(path: Path = DEFAULT_PROMPT_CONFIG) -> Dict:
    with open(path) as f:
        return json.load(f)


def sentences_to_1indexed(raw: str) -> str:
    lines = []
    for line in raw.strip().split("\n"):
        m = re.match(r"^(\d+)\s+(.+)$", line.strip())
        if m:
            lines.append(f"{int(m.group(1)) + 1}. {m.group(2)}")
        elif line.strip() and lines:
            lines[-1] = lines[-1] + " " + line.strip()
    return "\n".join(lines)


def split_thinking(content: str) -> Tuple[str, str]:
    m = re.search(r"<think>(.*?)</think>\s*", content, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip(), content[m.end():].strip()
    return "", content


def parse_output(content: str) -> Tuple[str, Optional[int]]:
    return parse_assessor_answer(content)


def parse_output_candidates(*candidates: str) -> Tuple[str, Optional[int]]:
    for text in candidates:
        if not text or not text.strip():
            continue
        label, sid = parse_output(text)
        if label != "UNKNOWN":
            return label, sid
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            last = lines[-1]
            if re.fullmatch(r"\d+", last):
                return "ERROR", int(last)
            if re.fullmatch(r"CORRECT", last, flags=re.IGNORECASE):
                return "CORRECT", None
    return "UNKNOWN", None


# ─────────────────────────────────────────────────────────────────────────────
# Metrics (identical to inference_detection.py)
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
        detection=dict(accuracy=det_acc, precision=prec, recall=rec, f1=f1,
                       tp=tp, fp=fp, tn=tn, fn=fn),
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
    p = argparse.ArgumentParser(description="Medical Error Detection + Localization (vLLM)")
    p.add_argument("--model_path",         required=True)
    p.add_argument("--base_model_path",    default=None,
                   help="Base model path/ID; if set, model_path is treated as a LoRA adapter")
    p.add_argument("--prompt_config",      default=str(DEFAULT_PROMPT_CONFIG))
    p.add_argument("--dataset",            default="all", choices=["ms", "uw", "all"])
    p.add_argument("--max_samples",        type=int, default=None)
    p.add_argument("--temperature",        type=float, default=None)
    p.add_argument("--top_p",              type=float, default=None)
    p.add_argument("--top_k",              type=int, default=20)
    p.add_argument("--min_p",             type=float, default=0.0)
    p.add_argument("--presence_penalty",   type=float, default=0.0)
    p.add_argument("--thinking_budget",    type=int, default=4096)
    p.add_argument("--max_new_tokens",     type=int, default=4096)
    p.add_argument("--mode",               choices=["thinking", "no-thinking"], default=None)
    p.add_argument("--no_thinking",        action="store_true")
    p.add_argument("--tensor_parallel_size", type=int, default=1,
                   help="Number of GPUs for tensor parallelism (default: 1)")
    p.add_argument("--output_dir",         default="results/detection")
    p.add_argument("--shard_id",           type=int, default=0)
    p.add_argument("--num_shards",         type=int, default=1)
    args = p.parse_args()

    if args.mode is None:
        args.mode = "no-thinking" if args.no_thinking else "thinking"
    use_thinking = args.mode == "thinking"
    if args.temperature is None:
        args.temperature = 0.6 if use_thinking else 0.7
    if args.top_p is None:
        args.top_p = 0.95 if use_thinking else 0.8

    max_tokens = (args.thinking_budget + 256) if use_thinking else args.max_new_tokens

    print(f"\n{'='*50}")
    print(f"Model   : {args.model_path}")
    print(f"Backend : vLLM  |  TP size: {args.tensor_parallel_size}")
    print(f"Dataset : {args.dataset}  |  Mode: {args.mode}  |  Thinking: {use_thinking}")
    print(f"Sampling: temperature={args.temperature} top_p={args.top_p} "
          f"top_k={args.top_k} min_p={args.min_p}")
    print(f"Max tokens (total): {max_tokens}")
    if args.num_shards > 1:
        print(f"Shard   : {args.shard_id + 1}/{args.num_shards}")
    print(f"{'='*50}\n")

    # ── Load tokenizer (for chat template only) ───────────────────────────────
    # If base_model_path is given, model_path is a LoRA adapter — use base for tokenizer
    tokenizer_path = args.base_model_path if args.base_model_path else args.model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    # ── Load model via vLLM ───────────────────────────────────────────────────
    use_lora = bool(args.base_model_path)
    if use_lora:
        from vllm.lora.request import LoRARequest
        llm = LLM(
            model=args.base_model_path,
            dtype="bfloat16",
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=8192,
            trust_remote_code=True,
            enable_lora=True,
            max_lora_rank=64,
        )
        lora_request = LoRARequest("adapter", 1, args.model_path)
    else:
        llm = LLM(
            model=args.model_path,
            dtype="bfloat16",
            tensor_parallel_size=args.tensor_parallel_size,
            max_model_len=8192,
            trust_remote_code=True,
        )
        lora_request = None

    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        max_tokens=max_tokens,
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    prompt_config = load_prompt_config(Path(args.prompt_config))
    test_df = load_test_data(args.dataset)
    if args.max_samples:
        test_df = test_df.head(args.max_samples)
    if args.num_shards > 1:
        test_df = test_df.iloc[args.shard_id::args.num_shards].reset_index(drop=True)

    system_prompt  = prompt_config["system_prompt"]
    user_template  = prompt_config["user_template"]

    # ── Build all prompts upfront ─────────────────────────────────────────────
    prompts, meta = [], []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Building prompts"):
        error_flag = int(row["Error Flag"])
        sentences  = sentences_to_1indexed(str(row.get("Sentences", row.get("Text", ""))))

        raw_sid = row.get("Error Sentence ID")
        if error_flag == 1 and pd.notna(raw_sid):
            gt_sid, gt_label = int(raw_sid) + 1, str(int(raw_sid) + 1)
        else:
            gt_sid, gt_label = None, "CORRECT"

        meta.append(dict(
            text_id=str(row.get("Text ID", "")),
            dataset=str(row.get("dataset", "")),
            error_flag=error_flag,
            gt_label=gt_label,
            gt_sid=gt_sid,
            error_type=str(row.get("Error Type", "")) if pd.notna(row.get("Error Type")) else "",
            sentences=sentences,
        ))

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_template.format(sentences=sentences)},
        ]
        prompt_kwargs = dict(tokenize=False, add_generation_prompt=True)
        if use_thinking:
            prompt_kwargs["enable_thinking"] = True
        prompts.append(tokenizer.apply_chat_template(messages, **prompt_kwargs))

    # ── vLLM inference (all samples in one shot) ──────────────────────────────
    print(f"Running vLLM inference on {len(prompts)} samples...")
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)

    # ── Parse results ─────────────────────────────────────────────────────────
    results = []
    for i, (m, out) in enumerate(zip(meta, outputs)):
        full_text = out.outputs[0].text.strip()
        thinking, content = split_thinking(full_text)
        if not content:
            content = full_text

        pred_type, pred_sid = parse_output_candidates(content, full_text, thinking)
        pred_label = ("CORRECT" if pred_type == "CORRECT"
                      else (str(pred_sid) if pred_sid is not None else "UNKNOWN"))

        detection_correct = (
            (m["gt_label"] == "CORRECT" and pred_label == "CORRECT") or
            (m["gt_label"] != "CORRECT" and pred_label not in ("CORRECT", "UNKNOWN"))
        )
        localization_correct = (
            m["gt_sid"] is not None and pred_sid is not None and m["gt_sid"] == pred_sid
        )

        if i == 0:
            print(f"\n{'='*50}\nDEBUG first sample\nGT={m['gt_label']}  Pred={pred_label}")
            print(f"--- THINKING ---\n{thinking[:500]}")
            print(f"--- ANSWER ---\n{content}\n{'='*50}\n")

        results.append(dict(
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
        ))

    # ── Metrics + save ────────────────────────────────────────────────────────
    metrics = compute_metrics(results)
    print_metrics(metrics)

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_slug   = args.mode.replace("-", "_")
    sample_slug = f"n{len(results)}"
    shard_slug  = f"_shard{args.shard_id}of{args.num_shards}" if args.num_shards > 1 else ""
    gen_slug    = (f"{mode_slug}_tb{args.thinking_budget}_{sample_slug}{shard_slug}"
                   if use_thinking else
                   f"{mode_slug}_max{args.max_new_tokens}_{sample_slug}{shard_slug}")

    results_file = f"{args.output_dir}/{args.dataset}_{gen_slug}_{ts}.jsonl"
    with open(results_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_file = f"{args.output_dir}/{args.dataset}_{gen_slug}_{ts}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(dict(
            model_path=args.model_path,
            backend="vllm",
            dataset=args.dataset,
            timestamp=ts,
            prompt_config=args.prompt_config,
            generation=dict(
                mode=args.mode, thinking=use_thinking,
                temperature=args.temperature, top_p=args.top_p,
                top_k=args.top_k, min_p=args.min_p,
                presence_penalty=args.presence_penalty,
                max_tokens=max_tokens,
                thinking_budget=args.thinking_budget,
                tensor_parallel_size=args.tensor_parallel_size,
            ),
            metrics=metrics,
        ), f, indent=2)

    print(f"Results : {results_file}")
    print(f"Summary : {summary_file}")


if __name__ == "__main__":
    main()
