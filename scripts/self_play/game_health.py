#!/usr/bin/env python3
"""Game-log health — jq-free replacement for the monitor's game section.

Reads the latest (or a given) self-play game log and prints the smoke gate
numbers: judge_status ok%, SAME-on-error%, game_invalid%, plus outcome/verdict
breakdowns and reward trend. Pure stdlib — no jq, no deps.

Usage:
    python3 scripts/self_play/game_health.py            # latest log
    python3 scripts/self_play/game_health.py <log.jsonl>
    python3 scripts/self_play/game_health.py --all      # all logs, chronological
                                                        # (full-run trend across restarts)
"""

import collections
import glob
import json
import sys


def _injector_trend(complete: list, n_windows: int = 6) -> None:
    """Chronological windows: is the injector learning to fit its budget?

    THE metric to watch during the v7 run: if -1.5 on parse_failure teaches
    the model to compress its thinking, waste%/at-cap% should FALL and mean
    injector tokens should drift DOWN over training. If they stay flat, the
    penalty is not shaping the injector and the wastage is a fixed tax.
    """
    if len(complete) < n_windows * 4:
        n_windows = max(1, len(complete) // 4)
    if n_windows < 2:
        return
    size = len(complete) / n_windows
    caps = [r.get("injector_max_new_tokens") for r in complete
            if isinstance(r.get("injector_max_new_tokens"), (int, float))]
    cap = int(caps[-1]) if caps else 0
    print()
    print(f"--- injector trend ({n_windows} chronological windows, cap={cap}) ---")
    print(f"  {'window':<8} {'games':>5} {'waste%':>7} {'at-cap%':>8} "
          f"{'mean-tok':>9} {'SAME-on-err%':>13} {'mean-reward':>12}")
    for w in range(n_windows):
        chunk = complete[int(w * size):int((w + 1) * size)]
        if not chunk:
            continue
        waste = sum(1 for r in chunk if r.get("injector_outcome")
                    in ("parse_failure", "truncation_filter"))
        toks = [r.get("injector_token_count") for r in chunk
                if isinstance(r.get("injector_token_count"), (int, float))]
        at_cap = sum(1 for t in toks if cap and t >= cap - 2)
        err = [r for r in chunk if r.get("mode") == "error_injection"
               and r.get("judge_verdict")]
        same = sum(1 for r in err if r.get("judge_verdict") == "SAME")
        rew = [r.get("assessor_reward") for r in chunk
               if r.get("assessor_reward") is not None]
        print(f"  {w + 1:<8} {len(chunk):>5} "
              f"{100 * waste / len(chunk):>6.0f}% "
              f"{100 * at_cap / max(len(toks), 1):>7.0f}% "
              f"{sum(toks) / max(len(toks), 1):>9.0f} "
              f"{100 * same / max(len(err), 1):>12.0f}% "
              f"{sum(rew) / max(len(rew), 1):>+12.3f}")
    print("  (want: waste%/at-cap% falling, mean-tok drifting down = model adapting)")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        game_dir = sys.argv[2] if len(sys.argv) > 2 else "results/self_play/interactions"
        logs = sorted(glob.glob(f"{game_dir}/game_*.jsonl"))
        if not logs:
            print(f"no game logs under {game_dir}/")
            sys.exit(1)
        log = f"{len(logs)} files: {logs[0]} .. {logs[-1]}"
        rows = []
        for lf in logs:
            rows.extend(json.loads(ln) for ln in open(lf) if ln.strip())
    elif len(sys.argv) > 1:
        log = sys.argv[1]
        rows = [json.loads(ln) for ln in open(log) if ln.strip()]
    else:
        logs = sorted(glob.glob("results/self_play/interactions/game_*.jsonl"))
        if not logs:
            print("no game logs under results/self_play/interactions/")
            sys.exit(1)
        log = logs[-1]
        rows = [json.loads(ln) for ln in open(log) if ln.strip()]

    complete = [r for r in rows if r.get("phase") == "game_complete"]

    print(f"log            : {log}")
    print(f"records        : {len(rows)}    game_complete: {len(complete)}")
    print()

    # ── judge health ────────────────────────────────────────────────────────
    # Judge ok% is measured ONLY on games where the injector produced a valid
    # edit — when the injector fails to parse, the judge is never really
    # consulted, so counting those as judge failures is an injector artifact
    # (verified 2026-07-15: judge was 26/26 = 100% on valid input while overall
    # ok% read 54%).
    js = collections.Counter(r.get("judge_status") for r in rows if r.get("judge_status"))
    tot = sum(js.values())
    print("judge_status   :", dict(js))
    valid_inj = {"exact_match", "wrong_edit_type", "partial_match"}
    got_edit = [r for r in complete if r.get("injector_outcome") in valid_inj]
    ok_on_valid = sum(1 for r in got_edit if r.get("judge_status") == "ok")
    ok_pct = 100 * ok_on_valid / len(got_edit) if got_edit else 0
    print(f"  judge ok     : {ok_on_valid}/{len(got_edit)} = {ok_pct:.0f}% on valid injector edits   (want > 90%)")

    inj_fail = [r for r in complete
                if r.get("injector_outcome") in ("parse_failure", "truncation_filter")]
    inj_fail_pct = 100 * len(inj_fail) / len(complete) if complete else 0
    print(f"injector waste : {len(inj_fail)}/{len(complete)} = {inj_fail_pct:.0f}%   "
          f"(informational; ~25-35% expected early, -1.5 should pressure it down — see trend)")

    err = [r for r in complete if r.get("mode") == "error_injection" and r.get("judge_verdict")]
    same = [r for r in err if r.get("judge_verdict") == "SAME"]
    same_pct = 100 * len(same) / len(err) if err else 0
    print(f"SAME-on-error  : {len(same)}/{len(err)} = {same_pct:.0f}%   (want < 15%)")

    gi = [r for r in complete if r.get("assessor_outcome") == "game_invalid"]
    gi_pct = 100 * len(gi) / len(complete) if complete else 0
    print(f"game_invalid   : {len(gi)}/{len(complete)} = {gi_pct:.0f}%   (want < 15%)")
    print()

    # ── breakdowns ──────────────────────────────────────────────────────────
    print("judge verdicts :", dict(collections.Counter(
        r.get("judge_verdict") for r in rows if r.get("judge_verdict"))))
    print("assessor out   :", dict(collections.Counter(
        r.get("assessor_outcome") for r in complete)))
    print("injector out   :", dict(collections.Counter(
        r.get("injector_outcome") for r in rows if r.get("injector_outcome"))))

    # ── reward trend ────────────────────────────────────────────────────────
    ar = [r.get("assessor_reward") for r in complete if r.get("assessor_reward") is not None]
    if ar:
        mean = sum(ar) / len(ar)
        pos = sum(1 for x in ar if x > 0.001)
        print(f"assessor_reward: mean {mean:+.3f}   positive {pos}/{len(ar)} "
              f"({100*pos/len(ar):.0f}%)")

    # ── injector adaptation trend ───────────────────────────────────────────
    _injector_trend(complete)

    # ── verdict ─────────────────────────────────────────────────────────────
    print()
    good = (ok_pct > 90) and (same_pct < 15) and (gi_pct < 15)
    if not tot:
        print("VERDICT: no judge calls logged — did the judge get hit? "
              "check the judge server for POST /v1/chat/completions")
    elif good:
        print("VERDICT: SMOKE PASS — judge healthy on live output. "
              "Safe to launch the full run.")
    else:
        print("VERDICT: SMOKE FAIL — one of ok%/SAME/game_invalid is off. "
              "Stop and inspect before the full run.")


if __name__ == "__main__":
    main()
