#!/usr/bin/env python3
"""Diagnose a smoke game log using the REAL per-turn token telemetry.

Reads an existing game log — no GPU, no cost. Answers:
  1. JUDGE HEALTH ON VALID INPUTS — when the injector hands a real edit, does the
     judge rule OK? (If yes, low overall judge-ok% is an injector artifact.)
  2. TOKEN BUDGET — where do the 8192 tokens actually go? Uses injector_token_count,
     assessor_reasoning_token_count, assessor_final_token_count, prompt counts —
     NOT char length (injector_output is clipped to 4000 chars in the log, so char
     length is meaningless).
  3. WHO IS STARVED — for each role, how often does generation hit its cap (ran out
     of tokens), and does that line up with parse failures?
  4. HEADROOM — per-game total sequence length vs the 8192 response cap.

Usage:
    python3 scripts/self_play/diagnose_smoke.py
    python3 scripts/self_play/diagnose_smoke.py results/self_play/interactions/game_XXXX.jsonl
"""
import collections
import glob
import json
import sys

RESP_CAP = 8192  # ROLLOUT_RESPONSE_LENGTH


def _nums(rows, key):
    return [r[key] for r in rows if isinstance(r.get(key), (int, float))]


def _summ(vals, cap=None, label=""):
    if not vals:
        print(f"  {label:<34} (none)")
        return
    s = sorted(vals)
    p50 = s[len(s) // 2]
    p90 = s[min(len(s) - 1, int(len(s) * 0.9))]
    tail = ""
    if cap:
        at_cap = sum(1 for v in vals if v >= cap - 2)
        tail = f"   at-cap(>={cap}): {at_cap}/{len(vals)} = {100*at_cap/len(vals):.0f}%"
    print(f"  {label:<34} p50={p50:>5.0f}  p90={p90:>5.0f}  max={s[-1]:>5.0f}{tail}")


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

    # ── 1. JUDGE HEALTH ON VALID INPUTS ──────────────────────────────────────
    print("=" * 72)
    print("1. JUDGE HEALTH ON VALID INPUTS")
    valid_inj = {"exact_match", "wrong_edit_type", "partial_match"}
    got_edit = [r for r in complete if r.get("injector_outcome") in valid_inj]
    js = collections.Counter(r.get("judge_status") for r in got_edit)
    n = len(got_edit)
    ok = js.get("ok", 0)
    print(f"  games where injector produced a real edit: {n}")
    print(f"  judge_status on those: {dict(js)}")
    if n:
        verdict = ("JUDGE FINE; low overall ok% is an injector artifact"
                   if ok / n > 0.9 else "judge itself failing on valid input — investigate")
        print(f"  judge ok on valid input: {ok}/{n} = {100*ok/n:.0f}%   <-- {verdict}")
    print()

    # ── 2. TOKEN BUDGET (real counts) ────────────────────────────────────────
    print("=" * 72)
    print("2. TOKEN BUDGET — where the 8192 goes (real token counts)")
    inj_cap = int((_nums(complete, "injector_max_new_tokens") or [0])[0])
    ath_cap = int((_nums(complete, "assessor_think_max_new_tokens") or [0])[0])
    afn_cap = int((_nums(complete, "assessor_final_max_new_tokens") or [0])[0])
    print(f"  caps: injector={inj_cap}  assessor_think={ath_cap}  assessor_final={afn_cap}")
    _summ(_nums(complete, "injector_context_token_count"), label="prompt (injector_context)")
    _summ(_nums(complete, "injector_token_count"), cap=inj_cap, label="injector generation")
    _summ(_nums(complete, "assessor_prompt_token_count"), label="assessor prompt")
    _summ(_nums(complete, "assessor_reasoning_token_count"), cap=ath_cap, label="assessor reasoning")
    _summ(_nums(complete, "assessor_final_token_count"), cap=afn_cap, label="assessor final")
    print()

    # ── 3. WHO IS STARVED (cap-hit vs failure) ───────────────────────────────
    print("=" * 72)
    print("3. WHO IS STARVED — cap-hit rate and link to failures")
    inj_fail = [r for r in complete
                if r.get("injector_outcome") in ("parse_failure", "truncation_filter")]
    inj_fail_atcap = sum(1 for r in inj_fail
                         if isinstance(r.get("injector_token_count"), (int, float))
                         and r["injector_token_count"] >= inj_cap - 2)
    print(f"  injector failures: {len(inj_fail)}")
    if inj_fail:
        print(f"    of those, hit the injector cap ({inj_cap}): {inj_fail_atcap}/{len(inj_fail)}"
              f"  <-- these ran OUT of tokens mid-think")
    ath_hit = sum(1 for r in complete if r.get("assessor_think_cap_hit"))
    afn_hit = sum(1 for r in complete if r.get("assessor_final_cap_hit"))
    print(f"  assessor think_cap_hit : {ath_hit}/{len(complete)} = {100*ath_hit/max(len(complete),1):.0f}%")
    print(f"  assessor final_cap_hit : {afn_hit}/{len(complete)} = {100*afn_hit/max(len(complete),1):.0f}%")
    print()

    # ── 4. TOTAL SEQUENCE vs 8192 ────────────────────────────────────────────
    print("=" * 72)
    print(f"4. TOTAL RESPONSE LENGTH vs cap {RESP_CAP}")
    totals = []
    for r in complete:
        g = r.get("injector_token_count", 0) or 0
        ap = r.get("assessor_prompt_token_count", 0) or 0
        at = r.get("assessor_token_count",
                   (r.get("assessor_reasoning_token_count", 0) or 0)
                   + (r.get("assessor_final_token_count", 0) or 0)) or 0
        totals.append(g + ap + at)  # generated portion of the multi-turn response
    _summ(totals, cap=RESP_CAP, label="response tokens (inj+assessor)")
    if totals:
        over = sum(1 for t in totals if t >= RESP_CAP - 2)
        print(f"  games at/over the {RESP_CAP} response cap: {over}/{len(totals)}")
    print("  (this is the GENERATED span the 8192 cap governs; if p90 is already near")
    print("   8192, both roles are being truncated and the budget is the real limit)")
    print()

    # ── READOUT ──────────────────────────────────────────────────────────────
    print("=" * 72)
    print("READOUT")
    print("  - Section 1 tells us the judge is fine (or not).")
    print("  - If injector failures mostly hit the cap AND assessor think_cap_hit is")
    print("    high, BOTH roles are starved -> forced-close for the injector + decide")
    print("    whether 8192 must rise. Section 4 quantifies how tight it is.")


if __name__ == "__main__":
    main()
