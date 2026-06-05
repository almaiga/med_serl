#!/usr/bin/env python3
"""Pre-flight Check 1: assessor reward EV under measured judge calibration.

The §3 / §4 of the failure analysis claimed v5's symmetric REWARD_MISS was
the wrong intervention because it amplified judge bias. The §5 + Exp 1 result
implies the *opposite* claim under a calibrated judge: with the judge fixed,
v5's reward retune is now the principled choice. This script makes that claim
quantitative.

We plug the verdict distributions measured in production into the assessor
reward formula and compute the expected per-game reward for three canonical
policies under each (judge, recipe) combination:

  Judges  (verdict_distribution per game mode)
    qwen3-8b   : from v5 game logs (4,951 games), the empirical judge bias
    medrect-32b: from Exp 1 (n=311 real errors) + Exp 2 Bucket C (n=30)

  Recipes  (REWARD_PARTIAL, REWARD_MISS)
    r2 : (0.3, -1.0) — the reward that produced r2's accidental 0.700
    v5 : (0.5, -1.5) — the reward retune we are about to commit GPU-days to

  Policies  (probabilities)
    always-CORRECT        : never flag anything; the conservatism trap
    perfect-discriminator : flags every real error, never flags benign,
                            always at the right sentence id; the ceiling
    r2-aggressive         : a stand-in for r2's actual policy
                            (high flag rate + high sid accuracy)

The headline we want to verify:
  Under (medrect-32b + v5 reward), EV(perfect-discriminator) substantially
  exceeds EV(always-CORRECT), so the gradient points toward the right
  policy. Under (qwen3-8b + v5 reward), the always-CORRECT trap is *not*
  closed — that is the failure mode of the v5 run.

Run with no arguments.
"""

# ────────────────────────────────────────────────────────────────────────────
# Empirical inputs
# ────────────────────────────────────────────────────────────────────────────

# Verdict distributions: P(judge_verdict | game_mode), per judge.
# Numbers are the actual fractions observed in the cited runs.
JUDGES = {
    # From v5 game logs (4,951 games). See docs/judge_bottleneck_failure_analysis.md §5 / App. A.
    "qwen3-8b (broken, v5 logs)": {
        "error":  {"CHANGED": 0.546, "SAME": 0.273, "ABSTAIN": 0.182},
        "benign": {"CHANGED": 0.116, "SAME": 0.823, "ABSTAIN": 0.061},
    },
    # From Exp 1 (n=311 real ms-test errors) for error mode, and Exp 2 Bucket C
    # (n=30 hand-crafted synonym/abbrev/brand-generic benigns) for benign mode.
    "medrect-32b + hint_v2 (Exp 1/2)": {
        "error":  {"CHANGED": 0.997, "SAME": 0.000, "ABSTAIN": 0.003},
        "benign": {"CHANGED": 0.233, "SAME": 0.767, "ABSTAIN": 0.000},
    },
}

# Reward recipes (from scripts/self_play/game_reward.py).
REWARDS = {
    "r2": {"EXACT": 1.0, "PARTIAL": 0.3, "MISS": -1.0, "FORMAT": 0.2},
    "v5": {"EXACT": 1.0, "PARTIAL": 0.5, "MISS": -1.5, "FORMAT": 0.2},
}

# 50/50 mix as in the live training config.
P_MODE = {"error": 0.5, "benign": 0.5}

# Assessor policies, parameterized as (p_flag in error mode, p_flag in benign mode,
# p_correct_sid given a flag). All three are interpreted independently of judge.
POLICIES = {
    "always-CORRECT":          (0.00, 0.00, 1.00),
    "perfect-discriminator":   (1.00, 0.00, 1.00),
    "r2-aggressive (high flag,high sid)": (0.90, 0.40, 0.85),
}


# ────────────────────────────────────────────────────────────────────────────
# Reward semantics — must mirror compute_assessor_game_reward in game_reward.py
# ────────────────────────────────────────────────────────────────────────────
#
# For each (judge_verdict, assessor_action) pair the reward outcome is:
#
#   judge=CHANGED, assessor flags the matched sid     -> EXACT  + FORMAT
#   judge=CHANGED, assessor flags a different sid     -> PARTIAL + FORMAT
#   judge=CHANGED, assessor says CORRECT              -> MISS   + FORMAT
#   judge=SAME,    assessor says CORRECT              -> EXACT  + FORMAT
#   judge=SAME,    assessor flags any sid             -> MISS   + FORMAT
#   judge=ABSTAIN, any assessor action                -> 0  (neutral)
#
# "Matched sid" in error mode = the real error sid (which the judge picked when
# it ruled CHANGED). In benign mode, judge=CHANGED is the judge OVER-FLAGGING,
# and the only sid it could be flagging is the synonym-swap sid; if the
# assessor flags the same sid, that's EXACT to the reward but is the mirror
# reward hack (assessor wrongly agreed with an over-flagging judge).
#
# We assume p_correct_sid is the same in both modes (the model doesn't know
# the mode and uses the same flagging policy).


def ev_per_game(judge, reward, p_flag_err, p_flag_ben, p_sid):
    EXACT = reward["EXACT"]   + reward["FORMAT"]
    PART  = reward["PARTIAL"] + reward["FORMAT"]
    MISS  = reward["MISS"]    + reward["FORMAT"]
    ev = 0.0
    for mode, p_mode in P_MODE.items():
        p_flag = p_flag_err if mode == "error" else p_flag_ben
        for verdict, p_v in judge[mode].items():
            if verdict == "ABSTAIN":
                continue
            if verdict == "CHANGED":
                ev += p_mode * p_v * (
                    p_flag * p_sid       * EXACT  +
                    p_flag * (1 - p_sid) * PART   +
                    (1 - p_flag)         * MISS
                )
            elif verdict == "SAME":
                ev += p_mode * p_v * (
                    p_flag       * MISS  +
                    (1 - p_flag) * EXACT
                )
    return ev


def main():
    width_judge = max(len(j) for j in JUDGES)
    width_pol   = max(len(p) for p in POLICIES)

    print("=" * 90)
    print("EV per game by (judge, reward, policy)")
    print("=" * 90)
    print(f"{'judge':<{width_judge}s}  {'rwd':3s}  {'policy':<{width_pol}s}  {'EV/game':>8s}")
    print("-" * 90)
    for jname, j in JUDGES.items():
        for rname, r in REWARDS.items():
            for pname, (pe, pb, ps) in POLICIES.items():
                ev = ev_per_game(j, r, pe, pb, ps)
                print(f"{jname:<{width_judge}s}  {rname:3s}  {pname:<{width_pol}s}  {ev:+8.3f}")
        print()

    print("=" * 90)
    print("Headline: EV(perfect-discriminator) - EV(always-CORRECT)")
    print("Gap > 0 means the reward gradient points at the right policy.")
    print("Gap >> 0 means strong signal. Gap ≈ 0 or < 0 means the trap is open.")
    print("=" * 90)
    print(f"{'judge':<{width_judge}s}  {'rwd':3s}  {'EV(perfect)':>11s}  {'EV(always-CORRECT)':>17s}  {'gap':>8s}")
    print("-" * 90)
    for jname, j in JUDGES.items():
        for rname, r in REWARDS.items():
            ev_perf = ev_per_game(j, r, 1.0, 0.0, 1.0)
            ev_safe = ev_per_game(j, r, 0.0, 0.0, 1.0)
            gap = ev_perf - ev_safe
            tag = ""
            if ev_safe > 0:
                tag = "  <-- always-CORRECT EV is POSITIVE: conservatism trap is OPEN"
            elif gap < 0.5:
                tag = "  <-- gap is small; signal may be drowned by policy noise"
            print(f"{jname:<{width_judge}s}  {rname:3s}  {ev_perf:>+11.3f}  {ev_safe:>+17.3f}  {gap:>+8.3f}{tag}")
        print()

    print("=" * 90)
    print("EV ceiling (perfect judge, perfect policy)")
    print("=" * 90)
    perfect_judge = {
        "error":  {"CHANGED": 1.0, "SAME": 0.0, "ABSTAIN": 0.0},
        "benign": {"CHANGED": 0.0, "SAME": 1.0, "ABSTAIN": 0.0},
    }
    for rname, r in REWARDS.items():
        ev = ev_per_game(perfect_judge, r, 1.0, 0.0, 1.0)
        print(f"  perfect judge + {rname} reward + perfect policy:  EV/game = {ev:+.3f}")
    print()


if __name__ == "__main__":
    main()
