#!/usr/bin/env python3
"""Print misclassified cases from benchmark_simple_judge.py output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print only misclassified judge benchmark cases."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/self_play/judge_50_50_100.jsonl"),
        help="Benchmark JSONL file written by benchmark_simple_judge.py",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Maximum number of cases to print. 0 means no limit.",
    )
    parser.add_argument(
        "--mode",
        choices=["benign", "error_injection"],
        default=None,
        help="Optional mode filter.",
    )
    parser.add_argument(
        "--show-output",
        action="store_true",
        help="Also print the raw judge output.",
    )
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def format_float(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def main() -> int:
    args = parse_args()
    path = args.input.resolve()
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    rows = load_rows(path)
    bad = [row for row in rows if not row.get("matched_expected", False)]
    if args.mode is not None:
        bad = [row for row in bad if row.get("mode") == args.mode]

    print(f"Input file      : {path}")
    print(f"Total rows      : {len(rows)}")
    print(f"Misclassified   : {len(bad)}")
    if args.mode:
        print(f"Mode filter     : {args.mode}")

    if not bad:
        print("\nNo misclassified cases found.")
        return 0

    print("")
    count = 0
    for row in bad:
        count += 1
        print(f"[{count}] {row.get('note_id', '')}")
        print(f"  mode             : {row.get('mode', '')}")
        print(f"  expected         : {row.get('expected_relation', '')}")
        print(f"  judge verdict    : {row.get('judge_verdict', '')}")
        print(f"  judge score      : {format_float(row.get('judge_score'))}")
        print(f"  base rule score  : {format_float(row.get('base_rule_score'))}")
        print(f"  final score      : {format_float(row.get('final_score'))}")
        print(f"  latency ms       : {format_float(row.get('latency_ms'))}")
        print(f"  reason           : {row.get('judge_reason', '')}")
        if row.get("error"):
            print(f"  error            : {row.get('error')}")
        print("  original sentence:")
        print(f"    {row.get('original_sentence', '')}")
        print("  modified sentence:")
        print(f"    {row.get('modified_sentence', '')}")
        if args.show_output:
            print("  raw judge output:")
            print(f"    {row.get('judge_output', '')}")
        print("")

        if args.limit and count >= args.limit:
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
