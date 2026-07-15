#!/usr/bin/env python3
"""Diagnose a smoke game log: is the judge actually fine, and where is the
token budget being blown? Reads an existing game log — no GPU, no cost.

Answers three questions so we can pick the injector fix WITHOUT a wasted re-smoke:
  1. JUDGE HEALTH ON VALID INPUTS — when the injector hands the judge a real
     edit, does the judge rule OK? (If yes, the low "judge ok%" is an injector
     artifact, not a judge problem.)
  2. INJECTOR FAILURE ANATOMY — do the failures have <think> but no </think>?
     That means the injector ran OUT OF TOKENS mid-reasoning (budget too small),
     not that it produced a bad format.
  3. ASSESSOR HEADROOM — does the assessor close its </think> and sit well under
     its cap? If so, we can safely shave the assessor's budget and give it to the
     injector. If the assessor is itself near the edge, we can't — disable
     injector thinking instead.

Usage:
    python3 scripts/self_play/diagnose_smoke.py
    python3 scripts/self_play/diagnose_smoke.py results/self_play/interactions/game_XXXX.jsonl
"""
import collections
import glob
import json
import sys


INJ_KEYS = ("injector_output", "injector_raw", "raw_injector", "injector_text",
            "injector_response", "injector_completion")
ASS_KEYS = ("assessor_output", "assessor_raw", "raw_assessor", "assessor_text",
            "assessor_response", "assessor_completion", "assessor_think")


def _first(row, keys):
    for k in keys:
        v = row.get(k)
        if v:
            return v
    return ""


def _anatomy(texts, label):
    n = len(texts)
    if not n:
        print(f"  ({label}: none)")
        return
    n_open = sum("<think>" in t for t in texts)
    n_close = sum("</think>" in t for t in texts)
    lens = sorted(len(t) for t in texts)
    p50 = lens[len(lens) // 2]
    p90 = lens[min(len(lens) - 1, int(len(lens) * 0.9))]
    print(f"  {label}: n={n}")
    print(f"    have <think>  : {n_open}/{n}")
    print(f"    have </think> : {n_close}/{n}   "
          f"{'<-- ran OUT mid-think (budget too small)' if n_close < n_open else '(closed OK)'}")
    print(f"    char len p50/p90/max: {p50} / {p90} / {lens[-1]}")


def main() -> None:
    if len(sys.argv) > 1:
        log = sys.argv[1]
    else:
        logs = sorted(glob.glob("results/self_play/interactions/game_*.jsonl"))
        if not logs:
            print("no game logs found under results/self_play/interactions/")
            sys.exit(1)
        log = logs[-1]

    rows = [json.loads(l) for l in open(log) if l.strip()]
    complete = [r for r in rows if r.get("phase") == "game_complete"]
    print(f"log: {log}")
    print(f"rows: {len(rows)}   game_complete: {len(complete)}\n")

    # show available keys once so we can adapt if a field name differs
    sample = complete[0] if complete else (rows[0] if rows else {})
    print("available keys on a game_complete row:")
    print("  " + ", ".join(sorted(sample.keys())))
    print()

    # ── 1. JUDGE HEALTH ON VALID INPUTS ──────────────────────────────────────
    print("=" * 72)
    print("1. JUDGE HEALTH ON VALID INPUTS")
    valid_inj = {"exact_match", "wrong_edit_type", "partial_match"}
    got_edit = [r for r in complete if r.get("injector_outcome") in valid_inj]
    js_on_valid = collections.Counter(r.get("judge_status") for r in got_edit)
    n_valid = len(got_edit)
    ok_on_valid = js_on_valid.get("ok", 0)
    print(f"  games where injector produced a real edit: {n_valid}")
    print(f"  judge_status on those: {dict(js_on_valid)}")
    if n_valid:
        print(f"  judge ok on valid input: {ok_on_valid}/{n_valid} = "
              f"{100*ok_on_valid/n_valid:.0f}%   "
              f"{'<-- JUDGE IS FINE; low overall ok% is an injector artifact' if ok_on_valid/n_valid > 0.9 else '<-- judge itself is failing on valid input, investigate'}")
    print()

    # ── 2. INJECTOR FAILURE ANATOMY ──────────────────────────────────────────
    print("=" * 72)
    print("2. INJECTOR FAILURE ANATOMY")
    inj_fail = [r for r in complete
                if r.get("injector_outcome") in ("parse_failure", "truncation_filter")]
    _anatomy([_first(r, INJ_KEYS) for r in inj_fail], "injector failures")
    print()

    # ── 3. ASSESSOR HEADROOM ─────────────────────────────────────────────────
    print("=" * 72)
    print("3. ASSESSOR HEADROOM (can we donate its budget to the injector?)")
    ass_texts = [_first(r, ASS_KEYS) for r in complete]
    ass_texts = [t for t in ass_texts if t]
    _anatomy(ass_texts, "assessor outputs (all)")
    ass_out = collections.Counter(r.get("assessor_outcome") for r in complete)
    print(f"  assessor_outcome: {dict(ass_out)}")
    print("  (if assessor closes </think> on ~all and no 'truncation'/'invalid_format'"
          " spike, it HAS headroom to give tokens to the injector)")
    print()

    # ── RECOMMENDATION ───────────────────────────────────────────────────────
    print("=" * 72)
    print("RECOMMENDATION")
    inj_starved = inj_fail and (
        sum("<think>" in _first(r, INJ_KEYS) for r in inj_fail)
        > sum("</think>" in _first(r, INJ_KEYS) for r in inj_fail))
    ass_closes = ass_texts and (
        sum("</think>" in t for t in ass_texts) >= 0.9 * sum("<think>" in t for t in ass_texts)
        if any("<think>" in t for t in ass_texts) else True)
    ass_lens = sorted(len(t) for t in ass_texts) if ass_texts else [0]
    ass_p90 = ass_lens[min(len(ass_lens) - 1, int(len(ass_lens) * 0.9))]

    if inj_starved and ass_closes:
        print("  * Injector is starved mid-think AND assessor closes its think block.")
        print("  * FIX: rebalance budget — cut assessor_think_max_new_tokens by ~512,")
        print("    raise injector_max_new_tokens 1536 -> 2048. Total stays < 8192.")
        print(f"    (assessor p90 char len ~{ass_p90}; confirm it's not near its token cap)")
    elif inj_starved and not ass_closes:
        print("  * Injector is starved BUT assessor also near its edge — no spare budget.")
        print("  * FIX: disable injector thinking (enable_thinking=False for injector);")
        print("    a one-sentence edit doesn't need long CoT. Frees budget outright.")
    else:
        print("  * Injector failures are NOT simple mid-think starvation — inspect the")
        print("    LAST-chars dump (run inspect_injector_fails.py) before changing budgets.")


if __name__ == "__main__":
    main()
