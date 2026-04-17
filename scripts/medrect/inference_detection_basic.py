#!/usr/bin/env python3
"""
Minimal medical error detection inference.

This script intentionally avoids the custom Qwen-specific generation paths used
by the larger inference wrappers. It applies the configured system/user prompt,
runs a single plain `generate()` call, and parses the visible text output.
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


def load_prompt_config(path: Path) -> Dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def sentences_to_1indexed(raw: str) -> str:
    lines = []
    for line in raw.strip().split("\n"):
        match = re.match(r"^(\d+)\s+(.+)$", line.strip())
        lines.append(f"{int(match.group(1)) + 1}. {match.group(2)}" if match else line)
    return "\n".join(lines)


def split_thinking(content: str) -> Tuple[str, str]:
    match = re.search(r"<think>(.*?)</think>\s*", content, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip(), content[match.end():].strip()
    return "", content.strip()


def parse_output_candidates(*candidates: str) -> Tuple[str, Optional[int]]:
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


def build_generation_kwargs(tokenizer, *, temperature: float, max_new_tokens: int) -> Dict:
    kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": temperature > 0,
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
    prompt_config: Dict,
    *,
    max_samples: Optional[int],
    batch_size: int,
    temperature: float,
    max_new_tokens: int,
    max_input_length: int,
    debug_first_n: int,
) -> List[Dict]:
    if max_samples:
        test_df = test_df.head(max_samples)

    model.eval()
    tokenizer.padding_side = "left"

    system_prompt = prompt_config["system_prompt"]
    user_template = prompt_config["user_template"]
    results: List[Dict] = []
    debug_printed = 0

    for batch_start in tqdm(range(0, len(test_df), batch_size), desc="Inference"):
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
            # enable_thinking=False: Qwen3 defaults to thinking mode which
            # requires sampling (temperature > 0).  This script uses greedy
            # decoding, so we explicitly disable thinking to get plain output.
            # Older tokenizers that don't support the kwarg fall back silently.
            try:
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
            except TypeError:
                prompt_text = tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
            prompts.append(prompt_text)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_input_length,
        ).to(model.device)

        prompt_len = inputs.input_ids.shape[1]
        gen_kwargs = build_generation_kwargs(
            tokenizer,
            temperature=temperature,
            max_new_tokens=max_new_tokens,
        )
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        for i, item in enumerate(meta):
            out_ids = outputs[i, prompt_len:].tolist()
            while out_ids and out_ids[-1] == tokenizer.pad_token_id:
                out_ids.pop()

            # Keep special tokens so split_thinking can find <think>...</think>.
            # Strip only the EOS / pad sentinels manually afterwards.
            full_text = tokenizer.decode(out_ids, skip_special_tokens=False).strip()
            # Remove padding / EOS artifacts that may appear at the boundaries.
            for _tok in (tokenizer.eos_token, tokenizer.pad_token):
                if _tok:
                    full_text = full_text.replace(_tok, "").strip()
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal medical error detection inference")
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--prompt_config", default=str(DEFAULT_PROMPT_CONFIG))
    parser.add_argument("--dataset", default="all", choices=["ms", "uw", "all"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--max_input_length", type=int, default=4096)
    parser.add_argument("--output_dir", default="results/detection/basic")
    parser.add_argument("--base_model_path", default=None)
    parser.add_argument("--debug_first_n", type=int, default=1)
    args = parser.parse_args()

    model_type = detect_model_type(args.model_path)
    print(f"\n{'=' * 60}")
    print("  Minimal Medical Error Detection Inference")
    print(f"{'=' * 60}")
    print(f"Model path         : {args.model_path}")
    print(f"Model type         : {model_type}")
    print(f"Dataset            : {args.dataset}")
    print(f"Batch size         : {args.batch_size}")
    print(f"Temperature        : {args.temperature}")
    print(f"Max new tokens     : {args.max_new_tokens}")
    print(f"Max input length   : {args.max_input_length}")
    print(f"Prompt config      : {args.prompt_config}")
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
        prompt_config,
        max_samples=args.max_samples,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        max_input_length=args.max_input_length,
        debug_first_n=args.debug_first_n,
    )

    metrics = compute_metrics(results)
    print(json.dumps(metrics, indent=2))

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
                "temperature": args.temperature,
                "max_new_tokens": args.max_new_tokens,
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
