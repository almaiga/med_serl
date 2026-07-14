#!/usr/bin/env python3
"""Decisive test: does the judge catch plausible errors WITH thinking enabled?

The logprob pilot (thinking OFF, as used during v6 training) showed the judge
misses every medium/subtle clinical error while catching only blatant or
garbled ones. This script re-runs the same 48 synthetic cases and reports
per-category ACCURACY with thinking ON vs OFF.

If thinking-ON catches the medium/subtle errors that thinking-OFF missed, the
v6 self-play reward signal was crippled by the token-budget decision to
disable thinking in the judge -> fix is a thinking-enabled judge.
If thinking-ON still misses them, MedRECT-32B is fundamentally too weak ->
fix is a stronger judge (paper 3 co-evolution, R1, bigger model).

Usage (single VM, in-process):
    python3 scripts/self_play/test_judge_thinking_accuracy.py \
        --local-logs data_processed/synthetic_test/game_synthetic_v1.jsonl
    # add --no-thinking to reproduce the crippled baseline
    # add --tensor-parallel-size 2 for two GPUs
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# Reuse prompt construction from the pilot so both tests stay identical.
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_judge_logprobs_pilot import (  # noqa: E402
    SYS_PROMPT_BASE, HINT_V2_CARVEOUT, USER_HINT_V2,
    load_all_games_from_local, load_all_games_from_hf,
)


def build_messages(game: Dict, prompt_style: str) -> Optional[Dict]:
    modified_note = game.get("modified_sentences") or game.get("modified_note") or ""
    if not modified_note:
        return None
    orig = game.get("original_sentence") or ""
    mod = game.get("modified_sentence") or ""
    sid = game.get("changed_sid") or game.get("error_sentence_id") or ""
    if prompt_style == "hint_v2" and orig and mod and str(sid).strip():
        return dict(messages=[
            {"role": "system", "content": SYS_PROMPT_BASE + HINT_V2_CARVEOUT},
            {"role": "user", "content": USER_HINT_V2.format(
                changed_sid=sid, original_sentence=orig,
                modified_sentence=mod, modified_note=modified_note)},
        ])
    return dict(messages=[
        {"role": "system", "content": SYS_PROMPT_BASE},
        {"role": "user", "content": str(modified_note)},
    ])


def parse_answer(text: str) -> Dict:
    """Extract the judge's final verdict from generated text.

    Handles thinking output by taking whatever follows the last </think>.
    Returns {'verdict': 'CORRECT'|'CHANGED', 'sid': int|None}.
    """
    tail = text
    if "</think>" in text:
        tail = text.rsplit("</think>", 1)[1]
    tail = tail.strip()
    if re.search(r"\bCORRECT\b", tail, re.IGNORECASE):
        return dict(verdict="CORRECT", sid=None)
    m = re.search(r"\b(\d{1,3})\b", tail)
    if m:
        return dict(verdict="CHANGED", sid=int(m.group(1)))
    # Fallback: scan whole text
    if re.search(r"\bCORRECT\b", text, re.IGNORECASE):
        return dict(verdict="CORRECT", sid=None)
    m = re.search(r"\b(\d{1,3})\b", text)
    if m:
        return dict(verdict="CHANGED", sid=int(m.group(1)))
    return dict(verdict="UNPARSEABLE", sid=None)


def score(expected_verdict: str, expected_sid, parsed: Dict) -> str:
    """HIT / MISS / WRONG_SID / OVER_FLAG / UNPARSEABLE."""
    if parsed["verdict"] == "UNPARSEABLE":
        return "UNPARSEABLE"
    if expected_verdict == "CORRECT":
        return "HIT" if parsed["verdict"] == "CORRECT" else "OVER_FLAG"
    # expected CHANGED
    if parsed["verdict"] == "CORRECT":
        return "MISS"
    if expected_sid is not None and parsed["sid"] == int(expected_sid):
        return "HIT"
    return "WRONG_SID"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--local-logs", nargs="+", default=None)
    ap.add_argument("--dataset", default="Abdine/medserl-v6-run2-interactions")
    ap.add_argument("--split", default="train")
    ap.add_argument("--model", default="pfnet/Preferred-MedRECT-32B")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--prompt-style", default="hint_v2",
                    choices=["hint_v2", "native"])
    ap.add_argument("--no-thinking", action="store_true",
                    help="Disable thinking (reproduces the crippled baseline)")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--output", default="results/judge_thinking_accuracy.json")
    args = ap.parse_args()

    thinking = not args.no_thinking
    print(f"Model    : {args.model}")
    print(f"Thinking : {thinking}")
    print(f"Prompt   : {args.prompt_style}")
    print()

    if args.local_logs:
        games = load_all_games_from_local(args.local_logs)
    else:
        games = load_all_games_from_hf(args.dataset, args.split)

    cases = []
    for g in games:
        p = build_messages(g, args.prompt_style)
        if p is None:
            continue
        cases.append(dict(
            note_id=g.get("note_id", "?"),
            failure_mode=g.get("failure_mode", "unknown"),
            expected_verdict=g.get("expected_verdict"),
            expected_sid=g.get("expected_sid"),
            messages=p["messages"],
        ))
    print(f"Built {len(cases)} prompts\n")

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt_strs = []
    for c in cases:
        try:
            s = tok.apply_chat_template(
                c["messages"], tokenize=False, add_generation_prompt=True,
                enable_thinking=thinking)
        except TypeError:
            s = tok.apply_chat_template(
                c["messages"], tokenize=False, add_generation_prompt=True)
        prompt_strs.append(s)

    llm = LLM(model=args.model, dtype="bfloat16",
              tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.max_model_len, trust_remote_code=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.max_tokens)

    outputs = llm.generate(prompt_strs, sampling_params=sampling)

    by_cat = defaultdict(lambda: defaultdict(int))
    rows = []
    for c, out in zip(cases, outputs):
        text = out.outputs[0].text
        parsed = parse_answer(text)
        verdict = score(c["expected_verdict"], c["expected_sid"], parsed)
        by_cat[c["failure_mode"]][verdict] += 1
        by_cat[c["failure_mode"]]["_total"] += 1
        rows.append(dict(
            note_id=c["note_id"], failure_mode=c["failure_mode"],
            expected_verdict=c["expected_verdict"],
            expected_sid=c["expected_sid"],
            said_verdict=parsed["verdict"], said_sid=parsed["sid"],
            result=verdict,
            gen_tokens=len(out.outputs[0].token_ids),
        ))

    # ── Per-case table ──────────────────────────────────────────────────────
    print("=" * 92)
    print(f"{'note_id':<20} {'failure_mode':<26} {'exp':>8} {'said':>9} "
          f"{'result':>11}")
    print("=" * 92)
    for r in rows:
        exp = r["expected_verdict"][:8] if r["expected_verdict"] else "?"
        said = (r["said_verdict"] or "?")
        if r["said_sid"] is not None:
            said = f"{said}:{r['said_sid']}"
        mark = "OK " if r["result"] == "HIT" else "XX "
        print(f"{r['note_id']:<20} {r['failure_mode']:<26} {exp:>8} "
              f"{said:>9} {mark+r['result']:>11}")

    # ── Per-category accuracy ───────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"{'category':<28} {'HIT/total':>10} {'accuracy':>10}")
    print("=" * 60)
    total_hit = total = 0
    for cat in sorted(by_cat):
        d = by_cat[cat]
        hit = d.get("HIT", 0)
        tot = d["_total"]
        total_hit += hit
        total += tot
        print(f"{cat:<28} {f'{hit}/{tot}':>10} {hit/tot:>9.0%}")
    print("-" * 60)
    print(f"{'OVERALL':<28} {f'{total_hit}/{total}':>10} "
          f"{total_hit/total:>9.0%}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(dict(model=args.model, thinking=thinking,
                       overall_accuracy=total_hit/total,
                       by_category={k: dict(v) for k, v in by_cat.items()},
                       rows=rows), f, indent=2)
    print(f"\nWritten to {args.output}")


if __name__ == "__main__":
    main()
