#!/usr/bin/env python3
"""Inspect MedSeRL self-play game logs and verify two-turn execution."""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter


def resolve_input(path: str | None) -> str:
    if path:
        return path
    matches = sorted(
        glob.glob("results/self_play/interactions/game_*.jsonl"),
        key=os.path.getmtime,
    )
    if not matches:
        raise FileNotFoundError("No game_*.jsonl files found in results/self_play/interactions")
    return matches[-1]


def load_rows(path: str) -> list[dict]:
    rows: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def is_two_turn(row: dict) -> bool:
    spans = row.get("turn_reward_spans") or []
    roles = [span.get("role") for span in spans if isinstance(span, dict)]
    return (
        row.get("phase") == "game_complete"
        and bool(row.get("injector_output"))
        and bool(row.get("assessor_output"))
        and "injector" in roles
        and "assessor" in roles
    )


def print_example(row: dict, idx: int) -> None:
    spans = row.get("turn_reward_spans") or []
    print(f"\n=== example {idx} ===")
    print("note_id:", row.get("note_id", ""))
    print("phase:", row.get("phase", ""))
    print("mode:", row.get("mode", ""))
    print("judge:", row.get("judge_verdict", ""))
    print("two_turn:", is_two_turn(row))
    print("injector_output:", row.get("injector_output", ""))
    print("assessor_output:", row.get("assessor_output", ""))
    print("turn_reward_spans:", spans)
    print("injector_reward:", row.get("injector_reward"))
    print("assessor_reward:", row.get("assessor_reward"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Path to a specific game_*.jsonl file")
    parser.add_argument("--limit", type=int, default=5, help="How many examples to print")
    parser.add_argument(
        "--only-failures",
        action="store_true",
        help="Print only rows that are not full two-turn completed games",
    )
    args = parser.parse_args()

    path = resolve_input(args.input)
    rows = load_rows(path)

    if not rows:
        print("file:", path)
        print("rows: 0")
        return 0

    phase_counts = Counter(str(row.get("phase", "missing")) for row in rows)
    mode_counts = Counter(str(row.get("mode", "missing")) for row in rows)
    two_turn_rows = [row for row in rows if is_two_turn(row)]
    one_turn_rows = [row for row in rows if not is_two_turn(row)]

    print("file:", path)
    print("rows:", len(rows))
    print("two_turn_completed:", f"{len(two_turn_rows)}/{len(rows)}")
    print("injector_only_or_failed:", len(one_turn_rows))
    print("phase_counts:", dict(phase_counts))
    print("mode_counts:", dict(mode_counts))

    to_show = one_turn_rows if args.only_failures else rows
    for idx, row in enumerate(to_show[: args.limit], 1):
        print_example(row, idx)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
