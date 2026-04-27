#!/usr/bin/env python3
"""Inspect MedSeRL self-play game logs and verify two-turn execution."""

from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter
from textwrap import shorten


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


def clip_text(value: object, max_chars: int) -> str:
    text = "" if value is None else str(value)
    text = text.strip()
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return shorten(text, width=max_chars, placeholder="\n...[truncated]")


def print_block(title: str, value: object, max_chars: int) -> None:
    text = clip_text(value, max_chars)
    print(f"\n--- {title} ---")
    print(text if text else "(empty)")


def print_example(row: dict, idx: int, *, max_chars: int) -> None:
    spans = row.get("turn_reward_spans") or []
    print(f"\n=== example {idx} ===")
    print("note_id:", row.get("note_id", ""))
    print("phase:", row.get("phase", ""))
    print("mode:", row.get("mode", ""))
    print("judge:", row.get("judge_verdict", ""))
    print("judge_status:", row.get("judge_status", ""))
    print("ground_truth:", row.get("ground_truth", row.get("assessor_ground_truth", "")))
    print("assessor_label:", row.get("assessor_label", ""))
    print("two_turn:", is_two_turn(row))
    print("turn_reward_spans:", spans)
    print("injector_reward:", row.get("injector_reward"))
    print("assessor_reward:", row.get("assessor_reward"))
    print_block("original numbered note", row.get("original_sentences") or row.get("modified_sentences", ""), max_chars)
    if row.get("original_sentence") or row.get("modified_sentence"):
        print_block("changed sentence before", row.get("original_sentence", ""), max_chars)
        print_block("changed sentence after", row.get("modified_sentence", ""), max_chars)
    print_block("injector output", row.get("injector_output", ""), max_chars)
    print_block("modified numbered note", row.get("modified_sentences", ""), max_chars)
    print_block("judge reason", row.get("judge_reason", ""), max_chars)
    print_block("assessor output", row.get("assessor_output", ""), max_chars)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", help="Path to a specific game_*.jsonl file")
    parser.add_argument("--limit", type=int, default=5, help="How many examples to print")
    parser.add_argument("--max-chars", type=int, default=2000, help="Max characters per text block")
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
        print_example(row, idx, max_chars=args.max_chars)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
