#!/usr/bin/env python3
"""
Medical Error Detection + Localization Inference Script

Task: Given numbered clinical sentences, output CORRECT or the error sentence number.
Reuses model loading and Qwen3 generation from inference_error_detection.py.

Metrics:
- Detection accuracy: correctly identifying error vs no-error
- Localization accuracy: among detected errors, exact sentence number match

Usage:
    # Test fine-tuned LoRA adapter
    python scripts/medrect/inference_detection.py \
        --model_path outputs/local_training/qwen3-8b-medrect-sft \
        --dataset ms --max_samples 50

    # Test base Qwen3 model
    python scripts/medrect/inference_detection.py \
        --model_path Qwen/Qwen3-8B \
        --dataset all

    # Quick test
    python scripts/medrect/inference_detection.py \
        --model_path Qwen/Qwen3-4B \
        --max_samples 20 --batch_size 4
"""

import os
import sys
import json
import re
import argparse
import pandas as pd
import torch
from datetime import datetime
from typing import List, Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm

# Add parent dir so we can import from scripts/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inference_error_detection import (
    load_model_and_tokenizer,
    detect_model_type,
    load_test_data,
    MODEL_TYPE_QWEN,
    THINK_END_TOKEN_ID,
    IM_END_TOKEN_ID,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_CONFIG = PROJECT_ROOT / "configs" / "prompts" / "detection_localization_prompts.json"


# ── prompt / data helpers ────────────────────────────────────────────────────

def load_prompt_config(config_path: Path = DEFAULT_PROMPT_CONFIG) -> Dict:
    """Load detection prompt config from JSON."""
    with open(config_path, "r") as f:
        return json.load(f)


def convert_sentences_to_1indexed(sentences_str: str) -> str:
    """Convert 0-indexed MEDEC sentences to 1-indexed.

    Input:  "0 First sentence.\\n1 Second sentence."
    Output: "1. First sentence.\\n2. Second sentence."
    """
    lines = sentences_str.strip().split("\n")
    converted = []
    for line in lines:
        m = re.match(r"^(\d+)\s+(.+)$", line.strip())
        if m:
            converted.append(f"{int(m.group(1)) + 1}. {m.group(2)}")
        else:
            converted.append(line)
    return "\n".join(converted)


# ── output parsing ───────────────────────────────────────────────────────────

def parse_detection_output(content: str) -> Tuple[str, Optional[int]]:
    """Parse model output for detection task.

    Returns (label, sentence_id):
        label       – "CORRECT", "ERROR", or "UNKNOWN"
        sentence_id – int if error detected, None otherwise
    """
    content = content.strip()

    # Take the first non-empty line as the main answer
    main_answer = ""
    for line in content.split("\n"):
        line = line.strip()
        if line:
            main_answer = line
            break
    if not main_answer:
        main_answer = content

    # Strip common prefixes like "Answer:", "Label:", etc.
    main_answer = re.sub(
        r"^(answer|label|output|result)\s*:\s*", "", main_answer, flags=re.IGNORECASE
    )

    # Check for CORRECT (but not INCORRECT)
    if re.search(r"\bcorrect\b", main_answer, re.IGNORECASE) and not re.search(
        r"\bincorrect\b", main_answer, re.IGNORECASE
    ):
        return "CORRECT", None

    # Try to extract a sentence number
    num_match = re.search(r"\b(\d+)\b", main_answer)
    if num_match:
        return "ERROR", int(num_match.group(1))

    # Fallback heuristics
    if re.search(r"error|incorrect|mistake|wrong", content, re.IGNORECASE):
        return "ERROR", None

    return "UNKNOWN", None


# ── inference ────────────────────────────────────────────────────────────────

def run_detection_inference(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    model_type: str,
    prompt_config: Dict,
    use_thinking: bool = True,
    max_samples: int = None,
    temperature: float = 0.3,
    max_new_tokens: int = 256,
    thinking_budget: int = 512,
    batch_size: int = 1,
) -> List[Dict]:
    """Run detection + localization inference on MEDEC data."""
    results: List[Dict] = []

    if max_samples:
        test_df = test_df.head(max_samples)

    model.eval()
    is_qwen = model_type == MODEL_TYPE_QWEN
    tokenizer.padding_side = "left"

    system_prompt = prompt_config["system_prompt"]
    user_template = prompt_config["user_template"]

    num_batches = (len(test_df) + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Inference"):
        batch_start = batch_idx * batch_size
        batch_end = min(batch_start + batch_size, len(test_df))
        batch_df = test_df.iloc[batch_start:batch_end]

        batch_prompts: List[str] = []
        batch_metadata: List[Dict] = []

        for _, row in batch_df.iterrows():
            # ── ground truth ──
            error_flag = int(row["Error Flag"])
            sentences_raw = str(row.get("Sentences", row.get("Text", "")))
            sentences_1idx = convert_sentences_to_1indexed(sentences_raw)

            error_sid_0 = row.get("Error Sentence ID")
            if pd.notna(error_sid_0) and error_flag == 1:
                gt_sentence_id = int(error_sid_0) + 1  # 0→1 indexed
                gt_label = str(gt_sentence_id)
            else:
                gt_sentence_id = None
                gt_label = "CORRECT"

            batch_metadata.append(
                {
                    "text_id": str(row.get("Text ID", "")),
                    "dataset": str(row.get("dataset", "")),
                    "error_flag": error_flag,
                    "gt_label": gt_label,
                    "gt_sentence_id": gt_sentence_id,
                    "error_type": str(row.get("Error Type", ""))
                    if pd.notna(row.get("Error Type"))
                    else "",
                    "sentences_preview": sentences_1idx[:500],
                }
            )

            # ── prompt ──
            user_content = user_template.format(sentences=sentences_1idx)
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            if is_qwen and use_thinking:
                prompt = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=True,
                )
            else:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            batch_prompts.append(prompt)

        # ── tokenize ──
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
        ).to(model.device)
        input_lengths = inputs.attention_mask.sum(dim=1).tolist()

        # ── generate ──
        gen_kwargs = dict(
            max_new_tokens=thinking_budget if (is_qwen and use_thinking) else max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=0.95 if temperature > 0 else None,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        # ── parse outputs ──
        for i, meta in enumerate(batch_metadata):
            input_len = input_lengths[i]
            valid_tokens = outputs[i][outputs[i] != tokenizer.pad_token_id]
            output_ids = valid_tokens[input_len:].tolist()

            thinking_content = ""
            if is_qwen and use_thinking:
                try:
                    idx = len(output_ids) - output_ids[::-1].index(THINK_END_TOKEN_ID)
                    thinking_content = tokenizer.decode(
                        output_ids[:idx], skip_special_tokens=True
                    ).strip("\n")
                    content = tokenizer.decode(
                        output_ids[idx:], skip_special_tokens=True
                    ).strip("\n")
                except ValueError:
                    content = tokenizer.decode(
                        output_ids, skip_special_tokens=True
                    ).strip("\n")
            else:
                content = tokenizer.decode(
                    output_ids, skip_special_tokens=True
                ).strip("\n")

            pred_label_type, pred_sentence_id = parse_detection_output(content)

            if pred_label_type == "CORRECT":
                pred_label = "CORRECT"
            elif pred_sentence_id is not None:
                pred_label = str(pred_sentence_id)
            else:
                pred_label = "ERROR_UNKNOWN"

            detection_correct = (
                (meta["gt_label"] == "CORRECT" and pred_label == "CORRECT")
                or (meta["gt_label"] != "CORRECT" and pred_label not in ("CORRECT", "UNKNOWN"))
            )

            localization_correct = (
                meta["gt_sentence_id"] is not None
                and pred_sentence_id is not None
                and meta["gt_sentence_id"] == pred_sentence_id
            )

            results.append(
                {
                    "text_id": meta["text_id"],
                    "dataset": meta["dataset"],
                    "error_type": meta["error_type"],
                    "gt_label": meta["gt_label"],
                    "gt_sentence_id": meta["gt_sentence_id"],
                    "pred_label": pred_label,
                    "pred_sentence_id": pred_sentence_id,
                    "detection_correct": detection_correct,
                    "localization_correct": localization_correct,
                    "thinking": thinking_content[:500] if thinking_content else "",
                    "raw_output": content[:500],
                }
            )

            # Debug first sample
            if batch_idx == 0 and i == 0:
                print(f"\n{'='*60}")
                print("DEBUG: First sample")
                print(f"{'='*60}")
                print(f"GT: {meta['gt_label']}, Pred: {pred_label}")
                print(f"Raw output: {content[:300]}")
                print(f"{'='*60}\n")

    return results


# ── metrics ──────────────────────────────────────────────────────────────────

def calculate_detection_metrics(results: List[Dict]) -> Dict:
    """Calculate detection and localization metrics."""
    total = len(results)
    det_correct = sum(1 for r in results if r["detection_correct"])

    error_cases = [r for r in results if r["gt_label"] != "CORRECT"]
    correct_cases = [r for r in results if r["gt_label"] == "CORRECT"]

    tp = sum(1 for r in error_cases if r["pred_label"] not in ("CORRECT", "UNKNOWN"))
    fn = sum(1 for r in error_cases if r["pred_label"] in ("CORRECT", "UNKNOWN"))
    tn = sum(1 for r in correct_cases if r["pred_label"] == "CORRECT")
    fp = sum(1 for r in correct_cases if r["pred_label"] != "CORRECT")

    precision = tp / (tp + fp) if (tp + fp) else 0
    recall = tp / (tp + fn) if (tp + fn) else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0

    detected_errors = [
        r for r in error_cases if r["pred_label"] not in ("CORRECT", "UNKNOWN")
    ]
    loc_correct = sum(1 for r in detected_errors if r["localization_correct"])

    # per error-type breakdown
    by_type: Dict[str, Dict] = {}
    for r in error_cases:
        etype = r["error_type"] or "unknown"
        if etype not in by_type:
            by_type[etype] = {"total": 0, "detected": 0, "localized": 0}
        by_type[etype]["total"] += 1
        if r["pred_label"] not in ("CORRECT", "UNKNOWN"):
            by_type[etype]["detected"] += 1
            if r["localization_correct"]:
                by_type[etype]["localized"] += 1

    return {
        "total_samples": total,
        "detection": {
            "accuracy": det_correct / total if total else 0,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        },
        "localization": {
            "total_errors": len(error_cases),
            "detected_errors": len(detected_errors),
            "correctly_localized": loc_correct,
            "accuracy": loc_correct / len(detected_errors) if detected_errors else 0,
        },
        "by_error_type": by_type,
    }


def print_detection_metrics(metrics: Dict) -> None:
    """Pretty-print detection + localization metrics."""
    det = metrics["detection"]
    loc = metrics["localization"]

    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nDetection (error vs no-error):")
    print(f"  Accuracy:  {det['accuracy']:.3f}")
    print(f"  Precision: {det['precision']:.3f}")
    print(f"  Recall:    {det['recall']:.3f}")
    print(f"  F1:        {det['f1']:.3f}")
    print(f"  TP={det['tp']} FP={det['fp']} TN={det['tn']} FN={det['fn']}")

    print(f"\nLocalization (exact sentence match):")
    print(f"  Error cases: {loc['total_errors']}")
    print(f"  Detected:    {loc['detected_errors']}")
    print(f"  Correct loc: {loc['correctly_localized']}")
    print(f"  Accuracy:    {loc['accuracy']:.3f}")

    print(f"\nBy error type:")
    for etype, st in sorted(
        metrics["by_error_type"].items(), key=lambda x: -x[1]["total"]
    ):
        det_r = st["detected"] / st["total"] if st["total"] else 0
        loc_r = st["localized"] / st["detected"] if st["detected"] else 0
        print(f"  {etype}: {st['total']} total, det={det_r:.2f}, loc={loc_r:.2f}")
    print(f"{'='*60}\n")


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Medical Error Detection + Localization"
    )
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--prompt_config", type=str, default=str(DEFAULT_PROMPT_CONFIG)
    )
    parser.add_argument(
        "--dataset", type=str, default="all", choices=["ms", "uw", "all"]
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--thinking_budget", type=int, default=512)
    parser.add_argument("--no_thinking", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/detection")

    args = parser.parse_args()

    model_type = detect_model_type(args.model_path)

    print(f"\n{'='*60}")
    print("Medical Error Detection + Localization")
    print(f"{'='*60}")
    print(f"Model:    {args.model_path}")
    print(f"Type:     {model_type}")
    print(f"Dataset:  {args.dataset}")
    print(f"Thinking: {not args.no_thinking}")
    print(f"{'='*60}\n")

    # load
    model, tokenizer = load_model_and_tokenizer(args.model_path, model_type)
    prompt_config = load_prompt_config(Path(args.prompt_config))
    test_df = load_test_data(args.dataset)

    # run
    results = run_detection_inference(
        model,
        tokenizer,
        test_df,
        model_type,
        prompt_config,
        use_thinking=not args.no_thinking,
        max_samples=args.max_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        thinking_budget=args.thinking_budget,
        batch_size=args.batch_size,
    )

    # metrics
    metrics = calculate_detection_metrics(results)
    print_detection_metrics(metrics)

    # save
    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_tag = args.model_path.replace("/", "_")

    results_file = f"{args.output_dir}/{model_tag}_{args.dataset}_{ts}.jsonl"
    with open(results_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Results: {results_file}")

    summary_file = f"{args.output_dir}/{model_tag}_{args.dataset}_{ts}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            {
                "model_path": args.model_path,
                "dataset": args.dataset,
                "timestamp": ts,
                "prompt_config": args.prompt_config,
                "metrics": metrics,
            },
            f,
            indent=2,
        )
    print(f"Summary: {summary_file}")


if __name__ == "__main__":
    main()
