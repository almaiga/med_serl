#!/usr/bin/env python3
"""Offline reward unit test: prove the v5 retune is wired correctly end-to-end.

This is the smoke run we cannot afford to do on 2xH100. It calls
compute_assessor_game_reward and compute_injector_game_reward from
scripts/self_play/game_reward.py with the (judge_verdict, assessor_output, mode)
tuples that the production agent loop would feed them, and asserts the returned
(reward, outcome) match the v5 expectations on every cell of the reward table.

A passing run means:
  - REWARD_PARTIAL = 0.5 (not r2's 0.3)
  - REWARD_MISS    = -1.5 (not r2's -1.0)
  - FORMAT_BONUS   = 0.2
  - the constants in game_reward.py actually take effect in the path the
    training loop calls — no stale duplicate is being imported instead.

Pre-flight Check 1 (scripts/self_play/reward_ev_check.py) checks the EV math
under the measured judge distribution. This script checks the per-cell reward
values that EV calc is built on. Together they confirm the reward + judge
combination is internally consistent before any GPU spends a token.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.self_play.game_reward import (  # noqa: E402
    FORMAT_BONUS,
    REWARD_EXACT,
    REWARD_MISS,
    REWARD_PARTIAL,
    compute_assessor_game_reward,
    compute_injector_game_reward,
)


# Expected v5 constants — assert them first so a stale checkout fails loudly.
EXPECTED = {
    "REWARD_EXACT":   1.0,
    "REWARD_PARTIAL": 0.5,
    "REWARD_MISS":   -1.5,
    "FORMAT_BONUS":   0.2,
}


def assert_close(actual, expected, label):
    if abs(actual - expected) > 1e-6:
        raise AssertionError(
            f"[{label}] expected {expected:+.3f}, got {actual:+.3f}"
        )


def main() -> int:
    fail = 0
    print("=" * 64)
    print("Constants in game_reward.py")
    print("=" * 64)
    actual = {
        "REWARD_EXACT":   REWARD_EXACT,
        "REWARD_PARTIAL": REWARD_PARTIAL,
        "REWARD_MISS":    REWARD_MISS,
        "FORMAT_BONUS":   FORMAT_BONUS,
    }
    for k, v in EXPECTED.items():
        match = "OK" if abs(actual[k] - v) < 1e-6 else "MISMATCH"
        marker = "[ OK ]" if match == "OK" else "[FAIL]"
        print(f"  {marker} {k:<16s} = {actual[k]:+.3f}  (expected {v:+.3f})")
        if match != "OK":
            fail += 1

    # ------------------------------------------------------------------
    # ASSESSOR REWARD: all eight cells of the (judge_verdict × mode × label) table.
    # exact_match / partial_match / miss / game_invalid
    # ------------------------------------------------------------------
    print()
    print("=" * 64)
    print("compute_assessor_game_reward (the path the RL loop calls)")
    print("=" * 64)

    EXACT_F = REWARD_EXACT + FORMAT_BONUS                 #  +1.2
    PART_F  = REWARD_PARTIAL + FORMAT_BONUS               #  +0.7
    MISS_F  = REWARD_MISS + FORMAT_BONUS                  #  -1.3
    PART    = REWARD_PARTIAL                              #  +0.5  (no format bonus)
    MISS    = REWARD_MISS                                 #  -1.5  (no format bonus)

    # Assessor output format is what parse_assessor_answer accepts: a bare
    # sentence number, "CORRECT", or "ERROR: <n>". UNKNOWN is anything else.
    cases = [
        # (label,                          judge,  assessor_output, changed_sid, expected_reward, expected_outcome)
        ("error/CHANGED, perfect detect",   "CHANGED", "7",            7,    EXACT_F, "exact_match"),
        ("error/CHANGED, wrong sid",        "CHANGED", "3",            7,    PART_F,  "partial_match"),
        ("error/CHANGED, says CORRECT",     "CHANGED", "CORRECT",      7,    MISS_F,  "miss"),
        ("error/CHANGED, garbage (UNKNOWN)","CHANGED", "blah blah",    7,    MISS,    "invalid_format"),
        ("benign/SAME, says CORRECT",       "SAME",    "CORRECT",      None, EXACT_F, "exact_match"),
        ("benign/SAME, flags a sentence",   "SAME",    "7",            None, MISS_F,  "miss"),
        ("benign/SAME, garbage (UNKNOWN)",  "SAME",    "blah blah",    None, MISS,    "miss"),
        ("ABSTAIN -> game neutral",         "ABSTAIN", "CORRECT",      7,    0.0,     "game_invalid"),
    ]

    for label, judge, ao, sid, want_r, want_out in cases:
        result = compute_assessor_game_reward(
            ao, judge_verdict=judge, changed_sid=sid
        )
        ok = (abs(result.reward - want_r) < 1e-6) and (result.outcome == want_out)
        marker = "[ OK ]" if ok else "[FAIL]"
        print(f"  {marker} {label:<36s}  reward={result.reward:+.3f} "
              f"outcome={result.outcome:<14s}  "
              f"want=({want_r:+.3f}, {want_out})")
        if not ok:
            fail += 1

    # ------------------------------------------------------------------
    # INJECTOR REWARD: judge agrees with requested mode -> exact; disagrees -> wrong_edit_type
    # ------------------------------------------------------------------
    print()
    print("=" * 64)
    print("compute_injector_game_reward (injector incentive)")
    print("=" * 64)

    inj_cases = [
        # (label,                          mode,              judge_verdict, parse_success, expected_reward, expected_outcome, expected_valid)
        ("error_injection, judge CHANGED",  "error_injection", "CHANGED", True,  EXACT_F, "exact_match",    True),
        ("error_injection, judge SAME",     "error_injection", "SAME",    True,  MISS_F,  "wrong_edit_type", True),
        ("benign, judge SAME",              "benign",          "SAME",    True,  EXACT_F, "exact_match",    True),
        ("benign, judge CHANGED",           "benign",          "CHANGED", True,  MISS_F,  "wrong_edit_type", True),
        ("parse failure",                   "error_injection", "CHANGED", False, MISS,    "parse_failure",   False),
    ]

    for label, mode, jv, parse_ok, want_r, want_out, want_valid in inj_cases:
        r, outcome, valid = compute_injector_game_reward(
            mode, jv, parse_success=parse_ok
        )
        ok = (
            abs(r - want_r) < 1e-6
            and outcome == want_out
            and valid == want_valid
        )
        marker = "[ OK ]" if ok else "[FAIL]"
        print(f"  {marker} {label:<32s}  reward={r:+.3f} outcome={outcome:<16s} "
              f"valid={valid}  want=({want_r:+.3f}, {want_out}, {want_valid})")
        if not ok:
            fail += 1

    # ------------------------------------------------------------------
    # Print the EV-relevant cell directly so we can sanity-check against §11.
    # ------------------------------------------------------------------
    print()
    print("=" * 64)
    print("Sanity: cells used by Pre-flight Check 1 (failure-analysis §11)")
    print("=" * 64)
    print(f"  EXACT + FORMAT (perfect detection)   = {EXACT_F:+.3f}   (want +1.200)")
    print(f"  PARTIAL + FORMAT (right kind, wrong sid) = {PART_F:+.3f}   (want +0.700)")
    print(f"  MISS  + FORMAT (false-neg or false-pos)  = {MISS_F:+.3f}   (want -1.300)")

    print()
    if fail == 0:
        print("ALL REWARD CHECKS PASS -- v5 constants are wired through the live "
              "reward path. Safe to launch the full run.")
        return 0
    else:
        print(f"{fail} REWARD CHECKS FAILED -- DO NOT launch until the divergence "
              "above is resolved.")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
