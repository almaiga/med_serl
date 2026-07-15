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


def _trend_verdict(vals: list, *, good: str) -> str:
    """Compare first-half mean vs second-half mean of a row's windows.

    good='down' → falling is healthy; good='up' → rising is healthy;
    good='info' → direction reported without judgement.
    """
    vals = [v for v in vals if v is not None]
    if len(vals) < 4:
        return ""
    half = len(vals) // 2
    a = sum(vals[:half]) / half
    b = sum(vals[half:]) / (len(vals) - half)
    spread = max(abs(a), abs(b), 1e-9)
    if abs(b - a) / spread < 0.15:
        arrow, direction = "→", "flat"
    elif b > a:
        arrow, direction = "↑", "up"
    else:
        arrow, direction = "↓", "down"
    if good == "info" or direction == "flat":
        return arrow
    return f"{arrow} {'good' if direction == good else 'BAD'}"


def _per_role_trend(complete: list, win_size: int = 50, max_windows: int = 8) -> None:
    """Transposed per-role trend: metrics as rows, time as columns (left→right).

    Fixed-size windows (default 50 games) so cells stay comparable as the run
    grows; shows the most recent `max_windows`. THE view for the v7 questions:
    is the injector compressing (waste/at-cap/think-tok), who is winning
    (inj-rew vs ass-rew), and is the judge holding (SAME-on-err)?
    """
    if len(complete) < win_size:
        win_size = max(10, len(complete) // 3)
    chunks = [complete[i:i + win_size]
              for i in range(0, len(complete), win_size)]
    chunks = [c for c in chunks if len(c) >= max(5, win_size // 3)][-max_windows:]
    if len(chunks) < 2:
        return
    caps = [r.get("injector_max_new_tokens") for r in complete
            if isinstance(r.get("injector_max_new_tokens"), (int, float))]
    cap = int(caps[-1]) if caps else 0

    def per_window(fn):
        return [fn(c) for c in chunks]

    def mean(xs):
        xs = [x for x in xs if x is not None]
        return sum(xs) / len(xs) if xs else None

    waste = per_window(lambda c: 100 * sum(
        1 for r in c if r.get("injector_outcome")
        in ("parse_failure", "truncation_filter")) / len(c))
    atcap = per_window(lambda c: (lambda t: 100 * sum(
        1 for x in t if cap and x >= cap - 2) / max(len(t), 1))(
        [r.get("injector_token_count") for r in c
         if isinstance(r.get("injector_token_count"), (int, float))]))
    itok = per_window(lambda c: mean(
        [r.get("injector_token_count") for r in c
         if isinstance(r.get("injector_token_count"), (int, float))]))
    irew = per_window(lambda c: mean([r.get("injector_reward") for r in c]))
    aexact = per_window(lambda c: 100 * sum(
        1 for r in c if r.get("assessor_outcome") == "exact_match") / len(c))
    arew = per_window(lambda c: mean([r.get("assessor_reward") for r in c]))

    def same_pct(c):
        err = [r for r in c if r.get("mode") == "error_injection"
               and r.get("judge_verdict")]
        if not err:
            return None
        return 100 * sum(1 for r in err
                         if r.get("judge_verdict") == "SAME") / len(err)
    same = per_window(same_pct)

    def fmt(v, kind):
        if v is None:
            return "    -"
        if kind == "pct":
            return f"{v:4.0f}%"
        if kind == "tok":
            return f"{v:5.0f}"
        return f"{v:+.2f}"

    def row(label, vals, kind, good):
        cells = " ".join(f"{fmt(v, kind):>6}" for v in vals)
        print(f"  {label:<15} {cells}   {_trend_verdict(vals, good=good)}")

    heads = " ".join(f"{'w' + str(i + 1):>6}" for i in range(len(chunks)))
    print()
    print(f"--- per-role trend (windows of {win_size} games, oldest→newest, "
          f"injector cap={cap}) ---")
    print(f"  {'':<15} {heads}   trend")
    print("  INJECTOR")
    row("  waste%", waste, "pct", "down")
    row("  at-cap%", atcap, "pct", "down")
    row("  think-tok", itok, "tok", "info")
    row("  reward", irew, "rew", "info")
    print("  ASSESSOR")
    row("  exact%", aexact, "pct", "up")
    row("  reward", arew, "rew", "up")
    print("  JUDGE")
    row("  SAME-on-err%", same, "pct", "down")

    ib = mean(irew[-3:])
    ab = mean(arew[-3:])
    if ib is not None and ab is not None:
        gap = ib - ab
        who = ("injector ahead" if gap > 0.15
               else "assessor ahead" if gap < -0.15 else "balanced")
        print(f"  balance (last 3 windows): inj {ib:+.2f} vs ass {ab:+.2f} → {who}")
    print("  (healthy: waste/at-cap ↓, exact%/ass-reward ↑, SAME-on-err < 15%,"
          " rewards oscillate rather than diverge)")


def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "--all":
        # --all [dir] [--since YYYYMMDD_HHMMSS]: scope to the current run so
        # old runs' logs don't pollute the trend windows.
        args = sys.argv[2:]
        since = ""
        if "--since" in args:
            i = args.index("--since")
            since = args[i + 1]
            args = args[:i] + args[i + 2:]
        game_dir = args[0] if args else "results/self_play/interactions"
        logs = sorted(glob.glob(f"{game_dir}/game_*.jsonl"))
        if since:
            logs = [p for p in logs
                    if p.split("game_")[-1].replace(".jsonl", "") >= since]
        if not logs:
            print(f"no game logs under {game_dir}/"
                  + (f" since {since}" if since else ""))
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
    ir = [r.get("injector_reward") for r in complete
          if r.get("injector_reward") is not None]
    ia = [r.get("injector_assigned_reward") for r in complete
          if r.get("injector_assigned_reward") is not None]
    if ir:
        ipos = sum(1 for x in ir if x > 0.001)
        line = (f"injector_reward: mean {sum(ir)/len(ir):+.3f}   "
                f"positive {ipos}/{len(ir)} ({100*ipos/len(ir):.0f}%)")
        if ia:
            line += f"   assigned(coupled) mean {sum(ia)/len(ia):+.3f}"
        print(line)

    # ── per-role adaptation trend ───────────────────────────────────────────
    _per_role_trend(complete)

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
