#!/usr/bin/env python3
"""
Medical Error Detection + Localization Inference (v2)

This variant is intended for MedRECT assessor-style evaluation on MEDEC test
data with clearer controls than scripts/medrect/inference_detection.py:

- explicit --max_input_length (default: 4096)
- explicit --answer_max_new_tokens for the final answer stage
- same prompt format as training
- supports merged full models or LoRA adapter directories

Example:
    python3 scripts/medrect/inference_detection_v2.py \
        --model_path outputs/local_training/qwen3-4b-medrect-assessor-v2 \
        --dataset all \
        --batch_size 32 \
        --temperature 0.7 \
        --thinking_budget 2048 \
        --answer_max_new_tokens 16 \
        --max_input_length 4096 \
        --output_dir results/detection/qwen3-4b-medrect-assessor-v2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from inference_error_detection import (  # noqa: E402
    detect_model_type,
    load_model_and_tokenizer,
    load_test_data,
)
from self_play.utils import parse_assessor_answer  # noqa: E402


DEFAULT_PROMPT_CONFIG = PROJECT_ROOT / "configs" / "prompts" / "detection_localization_prompts.json"

MODEL_TYPE_QWEN = "qwen"


def load_prompt_config(path: Path = DEFAULT_PROMPT_CONFIG) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sentences_to_1indexed(raw: str) -> str:
    """Convert MEDEC 0-indexed numbered lines to 1-indexed numbered lines."""
    lines = []
    for line in raw.strip().split("\n"):
        match = re.match(r"^(\d+)\s+(.+)$", line.strip())
        lines.append(f"{int(match.group(1)) + 1}. {match.group(2)}" if match else line)
    return "\n".join(lines)


def split_thinking(content: str) -> Tuple[str, str]:
    """Split visible thinking from final answer text."""
    match = re.search(r"<think>(.*?)</think>\s*", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip(), content[match.end():].strip()
    return "", content
def parse_output_candidates(*candidates: str) -> Tuple[str, Optional[int]]:
    """Parse final output from the most trusted candidate text available."""
    for text in candidates:
        if not text or not text.strip():
            continue

        label, sid = parse_assessor_answer(text)
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


def build_generation_kwargs(
    tokenizer,
    *,
    max_new_tokens: int,
    temperature: float,
    repetition_penalty: float,
) -> Dict:
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
        "repetition_penalty": repetition_penalty,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if temperature > 0:
        kwargs["temperature"] = temperature
        kwargs["top_p"] = 0.95
    return kwargs


def run_inference(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    model_type: str,
    prompt_config: Dict,
    *,
    use_thinking: bool,
    max_samples: Optional[int],
    temperature: float,
    thinking_budget: int,
    answer_max_new_tokens: int,
    batch_size: int,
    max_input_length: int,
    debug_first_n: int,
) -> List[Dict]:
    if max_samples:
        test_df = test_df.head(max_samples)

    model.eval()
    tokenizer.padding_side = "left"
    is_qwen = model_type == MODEL_TYPE_QWEN

    system_prompt = prompt_config["system_prompt"]
    user_template = prompt_config["user_template"]
    results: List[Dict] = []

    total = len(test_df)
    debug_printed = 0

    for batch_start in tqdm(range(0, total, batch_size), desc="Inference"):
        batch = test_df.iloc[batch_start: batch_start + batch_size]
        prompts: List[str] = []
        meta: List[Dict] = []

        for _, row in batch.iterrows():
            error_flag = int(row["Error Flag"])
            sentences = sentences_to_1indexed(str(row.get("Sentences", row.get("Text", ""))))

            raw_sid = row.get("Error Sentence ID")
            if error_flag == 1 and pd.notna(raw_sid):
                gt_sid = int(raw_sid) + 1
                gt_label = str(gt_sid)
            else:
                gt_sid = None
                gt_label = "CORRECT"

            meta.append(
                {
                    "text_id": str(row.get("Text ID", "")),
                    "dataset": str(row.get("dataset", "")),
                    "error_type": str(row.get("Error Type", "")) if pd.notna(row.get("Error Type")) else "",
                    "sentences": sentences,
                    "gt_sid": gt_sid,
                    "gt_label": gt_label,
                }
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_template.format(sentences=sentences)},
            ]
            prompts.append(
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(model.device)

        prompt_padded_len = inputs.input_ids.shape[1]
        max_new_tokens = (
            thinking_budget + answer_max_new_tokens
            if (is_qwen and use_thinking)
            else answer_max_new_tokens
        )
        gen_kwargs = build_generation_kwargs(
            tokenizer,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=1.05,
        )
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        for i, item in enumerate(meta):
            answer_ids = outputs[i, prompt_padded_len:].tolist()
            while answer_ids and answer_ids[-1] == tokenizer.pad_token_id:
                answer_ids.pop()

            full_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
            thinking, answer_text = split_thinking(full_text)
            if not answer_text:
                answer_text = full_text

            pred_type, pred_sid = parse_output_candidates(answer_text, full_text, thinking)
            if pred_type == "CORRECT":
                pred_label = "CORRECT"
            elif pred_sid is not None:
                pred_label = str(pred_sid)
            else:
                pred_label = "UNKNOWN"

            detection_correct = (
                (item["gt_label"] == "CORRECT" and pred_label == "CORRECT")
                or (item["gt_label"] != "CORRECT" and pred_label not in ("CORRECT", "UNKNOWN"))
            )
            localization_correct = (
                item["gt_sid"] is not None and pred_sid is not None and item["gt_sid"] == pred_sid
            )

            results.append(
                {
                    "text_id": item["text_id"],
                    "dataset": item["dataset"],
                    "note": item["sentences"],
                    "error_type": item["error_type"],
                    "gt_label": item["gt_label"],
                    "gt_sid": item["gt_sid"],
                    "pred_label": pred_label,
                    "pred_sid": pred_sid,
                    "detection_correct": detection_correct,
                    "localization_correct": localization_correct,
                    "thinking": thinking,
                    "raw_output": answer_text,
                    "full_text": full_text,
                }
            )

            if debug_printed < debug_first_n:
                print(
                    f"\n{'=' * 50}\n"
                    f"DEBUG sample {debug_printed + 1}\n"
                    f"GT={item['gt_label']}  Pred={pred_label}\n"
                    f"--- THINKING ---\n{thinking}\n"
                    f"--- ANSWER ---\n{answer_text}\n"
                    f"{'=' * 50}\n"
                )
                debug_printed += 1

    return results


def compute_metrics(results: List[Dict]) -> Dict:
    total = len(results)
    error_cases = [row for row in results if row["gt_label"] != "CORRECT"]
    correct_cases = [row for row in results if row["gt_label"] == "CORRECT"]

    tp = sum(1 for row in error_cases if row["pred_label"] not in ("CORRECT", "UNKNOWN"))
    fn = sum(1 for row in error_cases if row["pred_label"] in ("CORRECT", "UNKNOWN"))
    tn = sum(1 for row in correct_cases if row["pred_label"] == "CORRECT")
    fp = sum(1 for row in correct_cases if row["pred_label"] != "CORRECT")

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    exact_sentence_matches = sum(1 for row in error_cases if row["localization_correct"])
    detected_errors = sum(1 for row in error_cases if row["pred_label"] not in ("CORRECT", "UNKNOWN"))
    detection_accuracy = sum(1 for row in results if row["detection_correct"]) / total if total else 0.0

    by_error_type: Dict[str, Dict[str, int]] = {}
    for row in error_cases:
        error_type = row["error_type"] or "unknown"
        bucket = by_error_type.setdefault(
            error_type,
            {"total": 0, "detected": 0, "sentence_correct": 0},
        )
        bucket["total"] += 1
        if row["pred_label"] not in ("CORRECT", "UNKNOWN"):
            bucket["detected"] += 1
        if row["localization_correct"]:
            bucket["sentence_correct"] += 1

    return {
        "total_samples": total,
        "detection": {
            "accuracy": detection_accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
        "sentence_extraction": {
            "total_errors": len(error_cases),
            "detected_errors": detected_errors,
            "exact_matches": exact_sentence_matches,
            "accuracy": exact_sentence_matches / len(error_cases) if error_cases else 0.0,
        },
        "by_error_type": by_error_type,
    }


def print_metrics(metrics: Dict) -> None:
    detection = metrics["detection"]
    sentence = metrics["sentence_extraction"]

    print(f"\n{'=' * 50}")
    print("Detection (error vs no-error)")
    print(f"  Accuracy : {detection['accuracy']:.3f}")
    print(
        f"  Precision: {detection['precision']:.3f}  "
        f"Recall: {detection['recall']:.3f}  F1: {detection['f1']:.3f}"
    )
    print(
        f"  TP={detection['tp']} FP={detection['fp']} "
        f"TN={detection['tn']} FN={detection['fn']}"
    )

    print("\nSentence extraction (exact sentence match)")
    print(f"  Gold error cases : {sentence['total_errors']}")
    print(f"  Predicted errors  : {sentence['detected_errors']}")
    print(f"  Exact matches     : {sentence['exact_matches']}")
    print(f"  Accuracy          : {sentence['accuracy']:.3f}")

    print("\nBy error type")
    for error_type, bucket in sorted(metrics["by_error_type"].items(), key=lambda x: -x[1]["total"]):
        detection_recall = bucket["detected"] / bucket["total"] if bucket["total"] else 0.0
        sentence_accuracy = bucket["sentence_correct"] / bucket["total"] if bucket["total"] else 0.0
        print(
            f"  {error_type}: n={bucket['total']}  "
            f"det_recall={detection_recall:.2f}  sent_acc={sentence_accuracy:.2f}"
        )
    print(f"{'=' * 50}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Medical Error Detection + Localization (v2)")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt_config", default=str(DEFAULT_PROMPT_CONFIG))
    parser.add_argument("--dataset", default="all", choices=["ms", "uw", "all"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--thinking_budget", type=int, default=2048)
    parser.add_argument("--answer_max_new_tokens", type=int, default=16)
    parser.add_argument("--max_input_length", type=int, default=4096)
    parser.add_argument("--no_thinking", action="store_true")
    parser.add_argument("--output_dir", default="results/detection")
    parser.add_argument(
        "--base_model_path",
        default=None,
        help=(
            "Override base-model path for LoRA adapters when adapter_config.json "
            "points to a hub ID that should not be resolved directly."
        ),
    )
    parser.add_argument("--debug_first_n", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_type = detect_model_type(args.model_path)

    print(f"\n{'=' * 60}")
    print("  Medical Error Detection + Localization (v2)")
    print(f"{'=' * 60}")
    print(f"Model path         : {args.model_path}")
    print(f"Model type         : {model_type}")
    print(f"Dataset            : {args.dataset}")
    print(f"Thinking           : {not args.no_thinking}")
    print(f"Batch size         : {args.batch_size}")
    print(f"Temperature        : {args.temperature}")
    print(f"Thinking budget    : {args.thinking_budget}")
    print(f"Answer max tokens  : {args.answer_max_new_tokens}")
    print(f"Max input length   : {args.max_input_length}")
    print(f"Output dir         : {args.output_dir}")
    print(f"{'=' * 60}\n")

    model, tokenizer = load_model_and_tokenizer(
        args.model_path,
        model_type,
        base_model_override=args.base_model_path,
    )
    prompt_config = load_prompt_config(Path(args.prompt_config))
    test_df = load_test_data(args.dataset)

    results = run_inference(
        model,
        tokenizer,
        test_df,
        model_type,
        prompt_config,
        use_thinking=not args.no_thinking,
        max_samples=args.max_samples,
        temperature=args.temperature,
        thinking_budget=args.thinking_budget,
        answer_max_new_tokens=args.answer_max_new_tokens,
        batch_size=args.batch_size,
        max_input_length=args.max_input_length,
        debug_first_n=args.debug_first_n,
    )

    metrics = compute_metrics(results)
    print_metrics(metrics)

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_file = Path(args.output_dir) / f"{args.dataset}_{timestamp}.jsonl"
    with open(results_file, "w", encoding="utf-8") as handle:
        for row in results:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_file = Path(args.output_dir) / f"{args.dataset}_{timestamp}_summary.json"
    with open(summary_file, "w", encoding="utf-8") as handle:
        json.dump(
            {
                "model_path": args.model_path,
                "dataset": args.dataset,
                "timestamp": timestamp,
                "prompt_config": args.prompt_config,
                "thinking": not args.no_thinking,
                "temperature": args.temperature,
                "thinking_budget": args.thinking_budget,
                "answer_max_new_tokens": args.answer_max_new_tokens,
                "max_input_length": args.max_input_length,
                "metrics": metrics,
            },
            handle,
            indent=2,
        )

    print(f"Results : {results_file}")
    print(f"Summary : {summary_file}")


if __name__ == "__main__":
    main()
