#!/usr/bin/env python3
"""Parse verl trainer stdout and print a multi-metric PPO trend table.

Pure stdlib, no jq. Merges metrics per step across split lines, shows the
last N steps side by side, and raises WARN flags tuned for this run:

  grad_norm > 10          instability
  response_length/max     within 300 tokens of ROLLOUT_RESPONSE_LENGTH (8192)
                          -> a game nearly overflowed; crash-margin warning
  actor/entropy < 0.05    policy collapse
  response/aborted_ratio  > 5% (vllm aborting generations)

Exit code 1 if any flag fired (so monitor_training.sh can aggregate).

Usage:
    python3 scripts/self_play/train_metrics.py <trainer.log> [--last N]
"""
import os
import re
import sys

PAIR = re.compile(r"([A-Za-z0-9_/@.]+):(-?[0-9.]+(?:[eE][-+]?[0-9]+)?)")
RESP_CAP = int(os.environ.get("ROLLOUT_RESPONSE_LENGTH", "8192"))

COLS = [
    # (header, metric key, format)
    ("score",  "critic/score/mean",       "{:+.3f}"),
    ("rew",    "critic/rewards/mean",     "{:+.3f}"),
    ("klpen",  "actor/reward_kl_penalty", "{:.4f}"),
    ("grad",   "actor/grad_norm",         "{:.2f}"),
    ("pgloss", "actor/pg_loss",           "{:+.4f}"),
    ("ent",    "actor/entropy",           "{:.3f}"),
    ("rlen",   "response_length/mean",    "{:.0f}"),
    ("rmax",   "response_length/max",     "{:.0f}"),
    ("tok/s",  "perf/throughput",         "{:.0f}"),
]


def main() -> None:
    if len(sys.argv) < 2:
        print("usage: train_metrics.py <trainer.log> [--last N]")
        sys.exit(2)
    log = sys.argv[1]
    last_n = 12
    if "--last" in sys.argv:
        last_n = int(sys.argv[sys.argv.index("--last") + 1])

    steps: dict[int, dict[str, float]] = {}
    with open(log, errors="replace") as f:
        for line in f:
            m = re.search(r"\bstep:(\d+)\b", line)
            if not m:
                continue
            step = int(m.group(1))
            d = steps.setdefault(step, {})
            for k, v in PAIR.findall(line):
                try:
                    d[k] = float(v)
                except ValueError:
                    pass

    if not steps:
        print("  (no step lines yet — trainer still warming up?)")
        sys.exit(0)

    order = sorted(steps)
    window = order[-last_n:]
    warn = []

    # ── trend table ──────────────────────────────────────────────────────────
    hdr = f"  {'step':>5} " + " ".join(f"{h:>8}" for h, _, _ in COLS)
    print(f"--- PPO trend (last {len(window)} steps of {len(order)}; "
          f"epoch {steps[order[-1]].get('training/epoch', '?')}) ---")
    print(hdr)
    for s in window:
        d = steps[s]
        cells = []
        for _, key, fmt in COLS:
            v = d.get(key)
            cells.append(fmt.format(v)[:8].rjust(8) if v is not None else "       -")
        print(f"  {s:>5} " + " ".join(cells))

    # ── flags on the latest step ─────────────────────────────────────────────
    d = steps[order[-1]]
    gn = d.get("actor/grad_norm")
    if gn is not None and gn > 10:
        warn.append(f"grad_norm {gn:.2f} > 10 (instability)")
    rmax = d.get("response_length/max")
    if rmax is not None and rmax > RESP_CAP - 300:
        warn.append(f"response_length/max {rmax:.0f} within 300 of cap {RESP_CAP} "
                    f"— a game nearly overflowed (crash margin)")
    ent = d.get("actor/entropy")
    if ent is not None and ent < 0.05:
        warn.append(f"entropy {ent:.3f} < 0.05 (policy collapse)")
    ab = d.get("response/aborted_ratio")
    if ab is not None and ab > 0.05:
        warn.append(f"aborted_ratio {ab:.2%} > 5% (vllm aborting generations)")

    # score direction over the window (informational)
    scores = [steps[s].get("critic/score/mean") for s in window]
    scores = [x for x in scores if x is not None]
    if len(scores) >= 6:
        head = sum(scores[:3]) / 3
        tail = sum(scores[-3:]) / 3
        arrow = "rising" if tail > head + 0.02 else ("falling" if tail < head - 0.02 else "flat")
        print(f"\n  score direction: {head:+.3f} -> {tail:+.3f} ({arrow} over window)")

    for w in warn:
        print(f"  ** WARN: {w}")
    sys.exit(1 if warn else 0)


if __name__ == "__main__":
    main()
