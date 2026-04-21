#!/usr/bin/env python3
"""Summarize simple judge traces from judge_interactions_*.jsonl."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def _iter_records(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"Skipping {path}:{line_no}: {exc}")


def _fmt_pct(n: int, total: int) -> str:
    return f"{(100.0 * n / total):.1f}%" if total else "0.0%"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "path",
        nargs="?",
        default="results/self_play/interactions",
        help="Judge trace file or directory containing judge_interactions_*.jsonl",
    )
    args = parser.parse_args()

    root = Path(args.path)
    if root.is_dir():
        files = sorted(root.glob("judge_interactions_*.jsonl"))
    else:
        files = [root]

    if not files:
        print("No judge_interactions_*.jsonl files found.")
        return 1

    latest = max(files, key=lambda p: p.stat().st_mtime)
    records = list(_iter_records(latest))
    if not records:
        print(f"No records found in {latest}")
        return 1

    verdicts = Counter()
    mode_verdicts = defaultdict(Counter)
    reasons = Counter()
    raw_mismatches = 0
    weighted_positive = 0
    weighted_negative = 0
    weighted_zero = 0
    total_weighted = 0.0
    total_base = 0.0
    total_final = 0.0

    for rec in records:
        verdict = str(rec.get("judge_verdict", "UNKNOWN"))
        mode = str(rec.get("mode", "unknown"))
        reason = str(rec.get("judge_reason", ""))
        verdicts[verdict] += 1
        mode_verdicts[mode][verdict] += 1
        reasons[reason] += 1

        if rec.get("ground_truth_raw") != rec.get("ground_truth_resolved"):
            raw_mismatches += 1

        weighted = float(rec.get("weighted_judge_score", 0.0))
        total_weighted += weighted
        total_base += float(rec.get("base_score", 0.0))
        total_final += float(rec.get("final_score", 0.0))
        if weighted > 0:
            weighted_positive += 1
        elif weighted < 0:
            weighted_negative += 1
        else:
            weighted_zero += 1

    total = len(records)
    print(f"File: {latest}")
    print(f"Records: {total}")
    print(f"Raw/resolved GT mismatches: {raw_mismatches} ({_fmt_pct(raw_mismatches, total)})")
    print(f"Avg base score:   {total_base / total:.3f}")
    print(f"Avg judge delta:  {total_weighted / total:.3f}")
    print(f"Avg final score:  {total_final / total:.3f}")
    print()
    print("Judge verdicts:")
    for verdict, count in verdicts.most_common():
        print(f"  {verdict:8s} {count:5d}  {_fmt_pct(count, total)}")
    print()
    print("Weighted judge contribution:")
    print(f"  positive {weighted_positive:5d}  {_fmt_pct(weighted_positive, total)}")
    print(f"  negative {weighted_negative:5d}  {_fmt_pct(weighted_negative, total)}")
    print(f"  zero     {weighted_zero:5d}  {_fmt_pct(weighted_zero, total)}")
    print()
    print("By mode:")
    for mode, counts in sorted(mode_verdicts.items()):
        mode_total = sum(counts.values())
        summary = ", ".join(
            f"{verdict}={count} ({_fmt_pct(count, mode_total)})"
            for verdict, count in counts.most_common()
        )
        print(f"  {mode:16s} {mode_total:5d}  {summary}")

    print()
    print("Top judge reasons:")
    for reason, count in reasons.most_common(10):
        label = reason if reason else "<empty>"
        print(f"  {label[:100]:100s} {count:5d}  {_fmt_pct(count, total)}")

    print()
    print("Sample ABSTAIN cases:")
    shown = 0
    for rec in records:
        if rec.get("judge_verdict") != "ABSTAIN":
            continue
        print(f"  note_id={rec.get('note_id','')}")
        print(f"    mode={rec.get('mode','')} gt={rec.get('ground_truth_resolved','')}")
        print(f"    reason={rec.get('judge_reason','')}")
        output = str(rec.get("judge_output", "")).replace("\n", " ")
        print(f"    output={output[:180]}")
        shown += 1
        if shown >= 5:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
