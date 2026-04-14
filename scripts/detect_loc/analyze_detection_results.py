#!/usr/bin/env python3
"""Analyze saved detection JSONL outputs and diagnose parsing failures.

Usage:
    python scripts/detect_loc/analyze_detection_results.py \
        results/detection/Qwen3-4B_thinking/all_20260414_134021.jsonl

This script is intentionally read-only. It inspects a saved JSONL file,
highlights suspicious rows where the model answer may have been serialized into
the wrong field, and compares the current metrics to metrics recovered from the
stored text.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]


UNKNOWN_LABELS = {"UNKNOWN", "ERROR_UNKNOWN"}


def parse_assessor_answer(text: str) -> Tuple[str, Optional[int]]:
    """Canonical parser for assessor output.

    Returns:
        ("CORRECT", None)  — note is correct
        ("ERROR", sentence_id)  — error at sentence_id (1-indexed)
        ("UNKNOWN", None)  — could not parse
    """
    m = re.search(r"<think>(.*?)</think>\s*", text, re.DOTALL)
    if m:
        content = text[m.end():].strip()
    else:
        content = text

    answer = ""
    for line in content.split("\n"):
        line = line.strip()
        if line:
            answer = re.sub(
                r"^(answer|label|output|result|final_answer)\s*[:=]\s*",
                "",
                line,
                flags=re.IGNORECASE,
            )
            break

    if re.search(r"\bcorrect\b", answer, re.IGNORECASE) and not re.search(
        r"\bincorrect\b", answer, re.IGNORECASE
    ):
        return "CORRECT", None

    m = re.search(r"\b(\d+)\b", answer)
    if m:
        return "ERROR", int(m.group(1))

    if re.search(r"error|incorrect|mistake|wrong", answer, re.IGNORECASE):
        return "ERROR", None

    return "UNKNOWN", None


def load_rows(path: Path) -> List[Dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_metrics(rows: Iterable[Dict], *, use_recovered: bool = False) -> Dict:
    total = 0
    tp = fp = tn = fn = 0
    error_cases = 0
    sentence_matches = 0

    for row in rows:
        total += 1
        gt_has_error = row["gt_label"] != "CORRECT"
        if use_recovered:
            pred_label = row["recovered_pred_label"]
            pred_sid = row["recovered_pred_sid"]
        else:
            pred_label = row["pred_label"]
            pred_sid = row["pred_sid"]

        pred_has_error = pred_label not in {"CORRECT", "UNKNOWN", "ERROR_UNKNOWN"}

        if gt_has_error:
            error_cases += 1
            if pred_has_error:
                tp += 1
            else:
                fn += 1
            if row["gt_sid"] is not None and pred_sid == row["gt_sid"]:
                sentence_matches += 1
        else:
            if pred_has_error:
                fp += 1
            else:
                tn += 1

    accuracy = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    sentence_accuracy = sentence_matches / error_cases if error_cases else 0.0

    return {
        "total": total,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "error_cases": error_cases,
        "sentence_matches": sentence_matches,
        "sentence_accuracy": sentence_accuracy,
    }


def recover_prediction(row: Dict) -> Tuple[str, Optional[int], str]:
    raw_output = (row.get("raw_output") or "").strip()
    thinking = (row.get("thinking") or "").strip()

    fields_to_try = [
        ("raw_output", raw_output),
        ("thinking", thinking),
    ]

    for source, text in fields_to_try:
        if not text:
            continue
        label_type, pred_sid = parse_assessor_answer(text)
        if label_type == "CORRECT":
            return "CORRECT", None, source
        if label_type == "ERROR":
            return (str(pred_sid) if pred_sid is not None else "UNKNOWN"), pred_sid, source

    return "UNKNOWN", None, "none"


def enrich_rows(rows: List[Dict]) -> List[Dict]:
    enriched = []
    for row in rows:
        recovered_label, recovered_sid, recovered_source = recover_prediction(row)
        enriched.append(
            {
                **row,
                "recovered_pred_label": recovered_label,
                "recovered_pred_sid": recovered_sid,
                "recovered_source": recovered_source,
                "is_unknown_pred": row["pred_label"] in UNKNOWN_LABELS,
                "raw_output_empty": not (row.get("raw_output") or "").strip(),
                "thinking_nonempty": bool((row.get("thinking") or "").strip()),
                "recovery_changes_prediction": (
                    recovered_label != row["pred_label"] or recovered_sid != row["pred_sid"]
                ),
            }
        )
    return enriched


def print_metrics_block(title: str, metrics: Dict) -> None:
    print(title)
    print(
        f"  Accuracy={metrics['accuracy']:.3f}  Precision={metrics['precision']:.3f}  "
        f"Recall={metrics['recall']:.3f}  F1={metrics['f1']:.3f}"
    )
    print(f"  TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']}")
    print(
        f"  Sentence extraction accuracy={metrics['sentence_accuracy']:.3f}  "
        f"({metrics['sentence_matches']}/{metrics['error_cases']})"
    )


def print_example_rows(rows: List[Dict], limit: int) -> None:
    shown = 0
    for row in rows:
        if not row["recovery_changes_prediction"]:
            continue
        print("")
        print(f"text_id={row['text_id']} dataset={row['dataset']} gt={row['gt_label']}")
        print(
            f"  original:  pred_label={row['pred_label']} pred_sid={row['pred_sid']}"
        )
        print(
            f"  recovered: pred_label={row['recovered_pred_label']} "
            f"pred_sid={row['recovered_pred_sid']} source={row['recovered_source']}"
        )
        print(f"  raw_output={row.get('raw_output', '')[:160]!r}")
        print(f"  thinking={row.get('thinking', '')[:160]!r}")
        shown += 1
        if shown >= limit:
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved detection JSONL outputs.")
    parser.add_argument("jsonl_path", type=Path, help="Path to saved detection JSONL file")
    parser.add_argument("--show", type=int, default=10, help="Number of changed examples to print")
    args = parser.parse_args()

    rows = enrich_rows(load_rows(args.jsonl_path))

    pred_counts = Counter(row["pred_label"] for row in rows)
    recovered_counts = Counter(row["recovered_pred_label"] for row in rows)
    unknown_rows = [row for row in rows if row["is_unknown_pred"]]
    changed_rows = [row for row in rows if row["recovery_changes_prediction"]]

    print(f"File: {args.jsonl_path}")
    print(f"Rows: {len(rows)}")
    print("")
    print(f"Original pred_label counts : {dict(pred_counts.most_common())}")
    print(f"Recovered pred_label counts: {dict(recovered_counts.most_common())}")
    print("")
    print(f"Rows with UNKNOWN/ERROR_UNKNOWN : {len(unknown_rows)}")
    print(f"Rows with empty raw_output       : {sum(row['raw_output_empty'] for row in rows)}")
    print(f"Rows with non-empty thinking     : {sum(row['thinking_nonempty'] for row in rows)}")
    print(f"Rows changed by recovery         : {len(changed_rows)}")
    print(f"Recovered from thinking field    : {sum(row['recovered_source'] == 'thinking' for row in changed_rows)}")
    print("")

    print_metrics_block("Current metrics", compute_metrics(rows, use_recovered=False))
    print("")
    print_metrics_block("Recovered metrics", compute_metrics(rows, use_recovered=True))

    if changed_rows and args.show > 0:
        print("")
        print(f"Examples where recovery changes the prediction (showing up to {args.show})")
        print_example_rows(changed_rows, args.show)


if __name__ == "__main__":
    main()
