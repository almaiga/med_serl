#!/usr/bin/env python3
"""Benchmark the simple judge in isolation on the judge server itself.

This script does not import the training reward path. It loads the same
`configs/prompts/simple_judge_prompts.json` prompt config used by training and
posts directly to the OpenAI-compatible vLLM judge endpoint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import statistics
import sys
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.self_play.utils import (  # noqa: E402
    parse_injector_compact,
    parse_numbered_sentences,
    strip_thinking,
)


DEFAULT_BENIGN = Path(
    "data_processed/medrect/injector_benign_chains_20260310_135156.jsonl"
)
DEFAULT_ERROR = Path(
    "data_processed/medrect/injector_error_chains_20260310_135156.jsonl"
)
PROMPT_PATH = PROJECT_ROOT / "configs" / "prompts" / "simple_judge_prompts.json"
DEFAULT_JUDGE_PORT = os.getenv("JUDGE_PORT", "8002")
DEFAULT_JUDGE_URL = os.getenv(
    "JUDGE_VLLM_URL",
    f"http://127.0.0.1:{DEFAULT_JUDGE_PORT}/v1/chat/completions",
)
DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "Qwen/Qwen3-8B")
DEFAULT_JUDGE_TIMEOUT = float(os.getenv("SIMPLE_JUDGE_TIMEOUT", "20"))
DEFAULT_JUDGE_WEIGHT = float(os.getenv("SIMPLE_JUDGE_WEIGHT", "0.3"))
REWARD_EXACT = 1.0
REWARD_PARTIAL = 0.3
FORMAT_BONUS = 0.2
REWARD_MISS = -1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the simple judge directly against the vLLM server."
    )
    parser.add_argument(
        "--judge-url",
        default=DEFAULT_JUDGE_URL,
        help="Judge endpoint. Defaults to local judge server on this machine.",
    )
    parser.add_argument(
        "--judge-model",
        default=DEFAULT_JUDGE_MODEL,
        help="Model name to send in the OpenAI-compatible payload.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_JUDGE_TIMEOUT,
        help="Per-request timeout in seconds.",
    )
    parser.add_argument(
        "--judge-weight",
        type=float,
        default=DEFAULT_JUDGE_WEIGHT,
        help="Weight used in final_score = base_rule_score + judge_weight * signed_judge_score.",
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
        help="Total examples to test. Rounded down to an even number.",
    )
    parser.add_argument(
        "--sample-id",
        default=None,
        help="Evaluate one specific sample_id instead of a random benchmark batch.",
    )
    parser.add_argument(
        "--sample-mode",
        choices=["benign", "error_injection"],
        default=None,
        help="Required with --sample-id so the script knows which source file to search.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling.",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent requests to the judge.",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=5,
        help="Number of mismatches to print.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSONL path for detailed results.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sample and build payloads without calling the judge.",
    )
    parser.add_argument(
        "--print-prompt",
        action="store_true",
        help="Print the exact prompt payload used for the selected case.",
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


def load_prompt_config() -> dict[str, Any]:
    with open(PROMPT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


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


def extract_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except (TypeError, ValueError, json.JSONDecodeError):
        pass

    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : idx + 1])
                    return parsed if isinstance(parsed, dict) else None
                except (TypeError, ValueError, json.JSONDecodeError):
                    return None
    return None


def build_messages(
    prompt_cfg: dict[str, Any],
    note_id: str,
    error_type: str,
    changed_sid: int,
    original_sentence: str,
    modified_sentence: str,
) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": prompt_cfg["system_prompt"]}]
    for ex in prompt_cfg.get("few_shot_examples", []):
        messages.append(
            {
                "role": "user",
                "content": prompt_cfg["user_template"].format(
                    note_id=ex.get("note_id", "example"),
                    error_type=ex.get("error_type", ""),
                    changed_sid=ex.get("changed_sid", "?"),
                    original_sentence=ex["original_sentence"],
                    modified_sentence=ex["modified_sentence"],
                ),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "verdict": ex["verdict"],
                        "score": ex["score"],
                        "reason": ex["reason"],
                    }
                ),
            }
        )

    messages.append(
        {
            "role": "user",
            "content": prompt_cfg["user_template"].format(
                note_id=note_id,
                error_type=error_type,
                changed_sid=changed_sid,
                original_sentence=original_sentence[:2000],
                modified_sentence=modified_sentence[:2000],
            ),
        }
    )
    return messages


def build_case(record: dict[str, Any], mode: str, prompt_cfg: dict[str, Any]) -> dict[str, Any]:
    numbered_note = extract_numbered_note(str(record.get("user_prompt", "")))
    changed_sid = int(record.get("changed_sid") or record.get("error_sentence_id"))
    original_sentence = parse_numbered_sentences(numbered_note).get(changed_sid, "")
    modified_sentence = str(record.get("modified_text", ""))
    note_id = str(record.get("sample_id", ""))
    error_type = str(record.get("error_type") or record.get("change_type") or "")
    expected_relation = "SAME" if mode == "benign" else "CHANGED"

    payload = {
        "model": DEFAULT_JUDGE_MODEL,
        "messages": build_messages(
            prompt_cfg=prompt_cfg,
            note_id=note_id,
            error_type=error_type,
            changed_sid=changed_sid,
            original_sentence=original_sentence,
            modified_sentence=modified_sentence,
        ),
        **prompt_cfg.get("sampling_params", {}),
    }

    return {
        "note_id": note_id,
        "mode": mode,
        "expected_relation": expected_relation,
        "changed_sid": changed_sid,
        "solution_str": str(record.get("label") or f"{changed_sid}. {modified_sentence}"),
        "expected_sid": record.get("error_sentence_id"),
        "error_type": error_type,
        "original_note": numbered_note,
        "original_sentence": original_sentence,
        "modified_sentence": modified_sentence,
        "payload": payload,
    }


def sample_cases(
    benign_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    prompt_cfg: dict[str, Any],
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
    benign_cases = [build_case(row, "benign", prompt_cfg) for row in rng.sample(benign_rows, per_class)]
    error_cases = [
        build_case(row, "error_injection", prompt_cfg)
        for row in rng.sample(error_rows, per_class)
    ]
    mixed = benign_cases + error_cases
    rng.shuffle(mixed)
    return mixed


def find_case_by_sample_id(
    benign_rows: list[dict[str, Any]],
    error_rows: list[dict[str, Any]],
    prompt_cfg: dict[str, Any],
    sample_id: str,
    sample_mode: str,
) -> dict[str, Any]:
    rows = benign_rows if sample_mode == "benign" else error_rows
    for row in rows:
        if str(row.get("sample_id", "")) == sample_id:
            return build_case(row, sample_mode, prompt_cfg)
    raise ValueError(f"sample_id not found in {sample_mode} file: {sample_id}")


def sync_post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


async def post_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    try:
        import aiohttp  # type: ignore
    except ImportError:
        return await asyncio.to_thread(sync_post_json, url, payload, timeout)

    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def run_case(
    case: dict[str, Any],
    judge_url: str,
    judge_model: str,
    timeout: float,
    judge_weight: float,
    semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    payload = dict(case["payload"])
    payload["model"] = judge_model

    base_rule_score, rule_outcome, has_valid_format, pred_sid = compute_base_injector_reward(case)

    started = time.perf_counter()
    content = ""
    error = ""
    verdict_obj: dict[str, Any] = {}
    verdict = "ABSTAIN"
    score = 0.0
    signed_judge_score = 0.0
    weighted_judge_score = 0.0
    final_score = base_rule_score

    try:
        async with semaphore:
            response = await post_json(judge_url, payload, timeout)
        try:
            content = response["choices"][0]["message"]["content"].strip()
        except Exception:
            content = ""

        _, stripped = strip_thinking(content)
        verdict_obj = extract_json_object(stripped) or extract_json_object(content) or {}
        verdict = str(verdict_obj.get("verdict", "ABSTAIN")).upper()
        try:
            score = float(verdict_obj.get("score", 0.0))
        except (TypeError, ValueError):
            score = 0.0
        score = max(0.0, min(1.0, score))
        signed_judge_score = compute_signed_judge_score(verdict, score, case["expected_relation"])
        weighted_judge_score = judge_weight * signed_judge_score
        final_score = base_rule_score + weighted_judge_score
    except Exception as exc:
        error = str(exc)

    latency_ms = (time.perf_counter() - started) * 1000.0
    matched_expected = verdict == case["expected_relation"]

    return {
        **{k: v for k, v in case.items() if k != "payload"},
        "latency_ms": latency_ms,
        "base_rule_score": base_rule_score,
        "rule_outcome": rule_outcome,
        "has_valid_format": has_valid_format,
        "pred_sid": pred_sid,
        "judge_verdict": verdict,
        "judge_score": score,
        "judge_signed_score": signed_judge_score,
        "weighted_judge_score": weighted_judge_score,
        "final_score": final_score,
        "judge_reason": str(verdict_obj.get("reason", ""))[:1000],
        "judge_output": content[:2000],
        "matched_expected": matched_expected,
        "error": error,
    }


def compute_base_injector_reward(case: dict[str, Any]) -> tuple[float, str, bool, int | None]:
    sid, modified_text = parse_injector_compact(case["solution_str"])
    if sid is None or modified_text is None:
        return REWARD_MISS, "invalid_format", False, None

    if case["mode"] == "benign":
        return REWARD_EXACT + FORMAT_BONUS, "exact_match", True, sid

    expected_sid = case.get("expected_sid")
    if expected_sid is not None and sid == int(expected_sid):
        return REWARD_EXACT + FORMAT_BONUS, "exact_match", True, sid
    if expected_sid is not None:
        return REWARD_PARTIAL + FORMAT_BONUS, "partial_match", True, sid
    return REWARD_PARTIAL + FORMAT_BONUS, "partial_match", True, sid


def compute_signed_judge_score(verdict: str, score: float, expected_relation: str) -> float:
    if verdict == expected_relation:
        return score
    if verdict in {"SAME", "CHANGED"}:
        return -score
    return 0.0


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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def print_summary(
    results: list[dict[str, Any]],
    wall_time_s: float,
    show: int,
    judge_url: str,
    judge_model: str,
    timeout: float,
) -> None:
    latencies = sorted(r["latency_ms"] for r in results)
    matched = sum(1 for r in results if r["matched_expected"])
    abstains = sum(1 for r in results if r["judge_verdict"] == "ABSTAIN")
    failures = sum(1 for r in results if r.get("error"))
    benign = [r for r in results if r["mode"] == "benign"]
    error = [r for r in results if r["mode"] == "error_injection"]

    def acc(rows: list[dict[str, Any]]) -> float:
        return sum(1 for r in rows if r["matched_expected"]) / len(rows) if rows else 0.0

    print("")
    print("=== Simple Judge Isolation Benchmark ===")
    print(f"Judge URL      : {judge_url}")
    print(f"Judge model    : {judge_model}")
    print(f"Judge timeout  : {timeout}s")
    print(f"Total samples  : {len(results)}")
    print(f"Benign / Error : {len(benign)} / {len(error)}")
    print(f"Accuracy       : {matched}/{len(results)} = {matched / len(results):.1%}")
    print(f"Benign acc     : {acc(benign):.1%}")
    print(f"Error acc      : {acc(error):.1%}")
    print(f"Abstains       : {abstains}")
    print(f"Failures       : {failures}")
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


async def async_main(args: argparse.Namespace) -> int:
    benign_path = (PROJECT_ROOT / args.benign_file).resolve()
    error_path = (PROJECT_ROOT / args.error_file).resolve()
    if not benign_path.exists():
        raise FileNotFoundError(f"Benign file not found: {benign_path}")
    if not error_path.exists():
        raise FileNotFoundError(f"Error file not found: {error_path}")

    prompt_cfg = load_prompt_config()
    benign_rows = load_jsonl(benign_path)
    error_rows = load_jsonl(error_path)

    if args.sample_id:
        if not args.sample_mode:
            raise ValueError("--sample-mode is required when using --sample-id")
        cases = [
            find_case_by_sample_id(
                benign_rows=benign_rows,
                error_rows=error_rows,
                prompt_cfg=prompt_cfg,
                sample_id=args.sample_id,
                sample_mode=args.sample_mode,
            )
        ]
    else:
        cases = sample_cases(
            benign_rows=benign_rows,
            error_rows=error_rows,
            prompt_cfg=prompt_cfg,
            samples=args.samples,
            seed=args.seed,
        )

    print("=== Simple Judge Isolation Setup ===")
    print(f"Prompt config  : {PROMPT_PATH}")
    print(f"Judge URL      : {args.judge_url}")
    print(f"Judge model    : {args.judge_model}")
    print(f"Judge weight   : {args.judge_weight}")
    print(f"Benign file    : {benign_path}")
    print(f"Error file     : {error_path}")
    print(f"Samples        : {len(cases)}")
    if args.sample_id:
        print(f"Sample id      : {args.sample_id}")
        print(f"Sample mode    : {args.sample_mode}")
    print(f"Concurrency    : {args.concurrency}")
    print(f"Dry run        : {args.dry_run}")

    if args.dry_run:
        preview = dict(cases[0])
        preview_payload = preview.pop("payload")
        print("")
        print("Dry run preview:")
        print(json.dumps(preview, indent=2, ensure_ascii=False)[:3000])
        print("")
        print("Payload preview:")
        print(json.dumps(preview_payload, indent=2, ensure_ascii=False)[:3000])
        return 0

    if args.print_prompt:
        print("")
        print("Prompt payload:")
        print(json.dumps(cases[0]["payload"], indent=2, ensure_ascii=False)[:12000])

    semaphore = asyncio.Semaphore(args.concurrency)
    started = time.perf_counter()
    results = await asyncio.gather(
        *(
            run_case(
                case=case,
                judge_url=args.judge_url,
                judge_model=args.judge_model,
                timeout=args.timeout,
                judge_weight=args.judge_weight,
                semaphore=semaphore,
            )
            for case in cases
        )
    )
    wall_time_s = time.perf_counter() - started

    if args.sample_id:
        row = results[0]
        print("")
        print("=== Simple Judge Single Case ===")
        print(f"Sample id        : {row['note_id']}")
        print(f"Mode             : {row['mode']}")
        print(f"Expected         : {row['expected_relation']}")
        print(f"Base rule score  : {row['base_rule_score']:.2f}")
        print(f"Rule outcome     : {row['rule_outcome']}")
        print(f"Judge verdict    : {row['judge_verdict']}")
        print(f"Judge score      : {row['judge_score']:.2f}")
        print(f"Judge signed     : {row['judge_signed_score']:.2f}")
        print(f"Judge weighted   : {row['weighted_judge_score']:.2f}")
        print(f"Final score      : {row['final_score']:.2f}")
        print(f"Matched expected : {row['matched_expected']}")
        print(f"Latency          : {row['latency_ms']:.1f} ms")
        print(f"Reason           : {row['judge_reason']}")
        if row.get("error"):
            print(f"Judge error      : {row['error']}")
            print("Judge server is not reachable at the configured URL.")
        print("")
        print("Original sentence:")
        print(row["original_sentence"])
        print("")
        print("Modified sentence:")
        print(row["modified_sentence"])
        print("")
        print("Raw judge output:")
        print(row["judge_output"])
        if args.output is not None:
            output_path = (PROJECT_ROOT / args.output).resolve()
            write_jsonl(output_path, results)
            print(f"Saved detailed results to {output_path}")
        return 0

    print_summary(
        results=results,
        wall_time_s=wall_time_s,
        show=args.show,
        judge_url=args.judge_url,
        judge_model=args.judge_model,
        timeout=args.timeout,
    )

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
