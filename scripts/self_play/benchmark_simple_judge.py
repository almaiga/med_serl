#!/usr/bin/env python3
"""Benchmark the simple judge on a balanced benign/error sample.

This script uses the exact remote judge path from
scripts/self_play/simple_judge_reward.py. It samples 50/50 benign and error
injector outputs from data_processed, sends the same prompt payload used in
training, and reports latency plus verdict accuracy.
"""

from __future__ import annotations

import argparse
import atexit
import asyncio
import json
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.self_play.simple_judge_reward import (  # noqa: E402
    JUDGE_MODEL,
    JUDGE_TIMEOUT,
    JUDGE_URL,
    _judge_with_llm,
)
from scripts.self_play import reward_function as reward_function_module  # noqa: E402
from scripts.self_play.utils import parse_numbered_sentences  # noqa: E402

atexit.unregister(reward_function_module.print_summary)


DEFAULT_BENIGN = Path(
    "data_processed/medrect/injector_benign_chains_20260310_135156.jsonl"
)
DEFAULT_ERROR = Path(
    "data_processed/medrect/injector_error_chains_20260310_135156.jsonl"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the simple judge with a 50/50 benign-error mix."
    )
    parser.add_argument(
        "--benign-file",
        type=Path,
        default=DEFAULT_BENIGN,
        help="JSONL file with benign injector chains.",
    )
    parser.add_argument(
        "--error-file",
        type=Path,
        default=DEFAULT_ERROR,
        help="JSONL file with error injector chains.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Total number of examples to test. Rounded down to an even number.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for sampling.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Number of judge requests to run concurrently.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=5,
        help="How many mismatches to print.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSONL path for full per-example results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Build and validate the sampled examples without calling the judge.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def extract_numbered_note(user_prompt: str) -> str:
    lines = user_prompt.splitlines()
    note_lines: list[str] = []
    in_note = False
    for line in lines:
        if line.startswith("Output the changed sentence"):
            break
        if re.match(r"^\d+\.\s+", line.strip()):
            in_note = True
        if in_note and line.strip():
            note_lines.append(line.rstrip())
    return "\n".join(note_lines).strip()


def build_case(record: dict[str, Any], mode: str) -> dict[str, Any]:
    numbered_note = extract_numbered_note(str(record.get("user_prompt", "")))
    changed_sid = int(record.get("changed_sid") or record.get("error_sentence_id"))
    original_sentence = parse_numbered_sentences(numbered_note).get(changed_sid, "")
    modified_sentence = str(record.get("modified_text", ""))
    note_id = str(record.get("sample_id", ""))

    return {
        "note_id": note_id,
        "mode": mode,
        "expected_relation": "SAME" if mode == "benign" else "CHANGED",
        "solution_str": str(record.get("label") or f"{changed_sid}. {modified_sentence}"),
        "original_note": numbered_note,
        "original_sentence": original_sentence,
        "modified_sentence": modified_sentence,
        "extra_info": {
            "role": "injector",
            "note_id": note_id,
            "mode": "benign" if mode == "benign" else "error_injection",
            "sentences": numbered_note,
            "error_type": str(record.get("error_type") or ""),
        },
    }


def sample_cases(
    benign_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    if samples < 2:
        raise ValueError("--samples must be at least 2")

    if samples % 2 == 1:
        samples -= 1
        print(f"Adjusted sample count to {samples} to keep a 50/50 split.")

    per_class = samples // 2
    if len(benign_rows) < per_class:
        raise ValueError(
            f"Not enough benign rows in input: need {per_class}, found {len(benign_rows)}"
        )
    if len(error_rows) < per_class:
        raise ValueError(
            f"Not enough error rows in input: need {per_class}, found {len(error_rows)}"
        )

    rng = random.Random(seed)
    benign_cases = [build_case(row, "benign") for row in rng.sample(benign_rows, per_class)]
    error_cases = [build_case(row, "error_injection") for row in rng.sample(error_rows, per_class)]
    mixed = benign_cases + error_cases
    rng.shuffle(mixed)
    return mixed


async def run_case(case: dict[str, Any], semaphore: asyncio.Semaphore) -> dict[str, Any]:
    started = time.perf_counter()
    async with semaphore:
        judge_result = await _judge_with_llm(
            solution_str=case["solution_str"],
            extra_info=case["extra_info"],
            expected_relation=case["expected_relation"],
        )
    latency_ms = (time.perf_counter() - started) * 1000.0

    return {
        **case,
        "latency_ms": latency_ms,
        "judge_verdict": judge_result["verdict"],
        "judge_score": float(judge_result["judge_score"]),
        "judge_signed_score": float(judge_result["signed_score"]),
        "judge_reason": judge_result["reason"],
        "judge_output": judge_result["judge_output"],
        "matched_expected": judge_result["verdict"] == case["expected_relation"],
    }


def percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    idx = (len(values) - 1) * pct
    lo = int(idx)
    hi = min(lo + 1, len(values) - 1)
    frac = idx - lo
    return values[lo] * (1 - frac) + values[hi] * frac


def print_summary(results: list[dict[str, Any]], wall_time_s: float, show: int) -> None:
    latencies = sorted(r["latency_ms"] for r in results)
    matched = sum(1 for r in results if r["matched_expected"])
    abstains = sum(1 for r in results if r["judge_verdict"] == "ABSTAIN")

    benign = [r for r in results if r["mode"] == "benign"]
    error = [r for r in results if r["mode"] == "error_injection"]

    def acc(rows: list[dict[str, Any]]) -> float:
        return sum(1 for r in rows if r["matched_expected"]) / len(rows) if rows else 0.0

    print("")
    print("=== Simple Judge Benchmark Summary ===")
    print(f"Judge URL      : {JUDGE_URL}")
    print(f"Judge model    : {JUDGE_MODEL}")
    print(f"Judge timeout  : {JUDGE_TIMEOUT}s")
    print(f"Total samples  : {len(results)}")
    print(f"Benign / Error : {len(benign)} / {len(error)}")
    print(f"Accuracy       : {matched}/{len(results)} = {matched / len(results):.1%}")
    print(f"Benign acc     : {acc(benign):.1%}")
    print(f"Error acc      : {acc(error):.1%}")
    print(f"Abstains       : {abstains}")
    print(f"Wall time      : {wall_time_s:.2f}s")
    print(f"Throughput     : {len(results) / wall_time_s:.2f} req/s")
    print(f"Latency mean   : {statistics.mean(latencies):.1f} ms")
    print(f"Latency p50    : {percentile(latencies, 0.50):.1f} ms")
    print(f"Latency p95    : {percentile(latencies, 0.95):.1f} ms")
    print(f"Latency max    : {max(latencies):.1f} ms")

    mismatches = [r for r in results if not r["matched_expected"]]
    if not mismatches:
        print("")
        print("No mismatches found in sampled cases.")
        return

    print("")
    print(f"=== First {min(show, len(mismatches))} Mismatches ===")
    for row in mismatches[:show]:
        print(f"[{row['mode']}] {row['note_id']}")
        print(f"  expected: {row['expected_relation']}  got: {row['judge_verdict']}")
        print(f"  latency : {row['latency_ms']:.1f} ms")
        print(f"  reason  : {row['judge_reason']}")
        print(f"  original: {row['original_sentence']}")
        print(f"  modified: {row['modified_sentence']}")
        print("")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


async def async_main(args: argparse.Namespace) -> int:
    benign_path = (PROJECT_ROOT / args.benign_file).resolve()
    error_path = (PROJECT_ROOT / args.error_file).resolve()

    if not benign_path.exists():
        raise FileNotFoundError(f"Benign file not found: {benign_path}")
    if not error_path.exists():
        raise FileNotFoundError(f"Error file not found: {error_path}")

    benign_rows = load_jsonl(benign_path)
    error_rows = load_jsonl(error_path)
    cases = sample_cases(benign_rows, error_rows, args.samples, args.seed)

    print("=== Simple Judge Benchmark Setup ===")
    print(f"Benign file   : {benign_path}")
    print(f"Error file    : {error_path}")
    print(f"Judge URL     : {JUDGE_URL or '<unset>'}")
    print(f"Judge model   : {JUDGE_MODEL}")
    print(f"Samples       : {len(cases)}")
    print(f"Concurrency   : {args.concurrency}")
    print(f"Dry run       : {args.dry_run}")

    if args.dry_run:
        print("")
        print("Dry run complete. First sampled case:")
        preview = {k: v for k, v in cases[0].items() if k != "extra_info"}
        print(json.dumps(preview, indent=2, ensure_ascii=False)[:3000])
        return 0

    if not JUDGE_URL:
        print("ERROR: JUDGE_VLLM_URL is not set.", file=sys.stderr)
        print(
            "Start the server with: bash scripts/self_play/start_judge_server.sh",
            file=sys.stderr,
        )
        return 1

    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    results = await asyncio.gather(*(run_case(case, semaphore) for case in cases))
    wall_time_s = time.perf_counter() - started

    print_summary(results, wall_time_s, args.show)

    if args.output is not None:
        output_path = (PROJECT_ROOT / args.output).resolve()
        write_jsonl(output_path, results)
        print(f"Saved detailed results to {output_path}")

    return 0


def main() -> int:
    args = parse_args()
    try:
        return asyncio.run(async_main(args))
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
