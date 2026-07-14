#!/usr/bin/env python3
"""Aggregate the memorization probe into a thinking-vs-no-thinking matrix.

For each model, shows F1 / recall / precision / sentence-accuracy under
thinking and no-thinking, plus the thinking-minus-nothinking delta.

Interpretation of the F1 delta:
  large positive (>= ~0.08)  -> model REASONS (thinking is doing real work)
  near zero / negative       -> model PATTERN-MATCHES or recalls answers
"""

import argparse
import glob
import json
import re
from pathlib import Path
from typing import Dict, Optional


def find_summary(eval_dir: Path) -> Optional[Path]:
    xs = sorted(glob.glob(str(eval_dir / "*_summary.json")))
    return Path(xs[-1]) if xs else None


def load(eval_dir: Path) -> Optional[Dict]:
    s = find_summary(eval_dir)
    if not s:
        return None
    m = json.loads(s.read_text())["metrics"]
    return dict(
        f1=m["detection"]["f1"],
        recall=m["detection"]["recall"],
        precision=m["detection"]["precision"],
        acc=m["detection"]["accuracy"],
        sent=m["sentence_extraction"]["accuracy"],
    )


def fmt(v: Optional[float]) -> str:
    return "—" if v is None else f"{v:.3f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-root", default="results/memorization_probe")
    args = ap.parse_args()
    root = Path(args.results_root)

    # discover model keys from directory names <key>__<mode>
    keys = []
    for d in sorted(root.glob("*__thinking")):
        m = re.match(r"(.+)__thinking$", d.name)
        if m and m.group(1) not in keys:
            keys.append(m.group(1))

    rows = []
    for key in keys:
        t = load(root / f"{key}__thinking")
        n = load(root / f"{key}__no-thinking")
        rows.append((key, t, n))

    # ── Accuracy + F1 matrix ─────────────────────────────────────────────────
    print()
    print("=" * 96)
    print(f"{'model':<20} {'acc think':>9} {'acc no-th':>9} {'Δacc':>7}   "
          f"{'F1 think':>9} {'F1 no-th':>9} {'ΔF1':>7}  {'verdict':<16}")
    print("=" * 96)
    for key, t, n in rows:
        if t is None or n is None:
            print(f"{key:<20} {'(incomplete — missing a run)':<40}")
            continue
        d_acc = t["acc"] - n["acc"]
        d_f1 = t["f1"] - n["f1"]
        # verdict keys on the larger of the two deltas (reasoning shows up in
        # whichever metric the model is being scored on)
        d_max = max(d_acc, d_f1)
        if d_max >= 0.08:
            verdict = "reasons"
        elif d_max <= 0.02:
            verdict = "pattern/recall"
        else:
            verdict = "mixed"
        print(f"{key:<20} {fmt(t['acc']):>9} {fmt(n['acc']):>9} "
              f"{d_acc:>+7.3f}   {fmt(t['f1']):>9} {fmt(n['f1']):>9} "
              f"{d_f1:>+7.3f}  {verdict:<16}")

    # ── Full breakdown ───────────────────────────────────────────────────────
    print()
    print("Full breakdown (t=thinking, n=no-thinking):")
    print("-" * 78)
    print(f"{'model':<20} {'mode':<12} {'F1':>6} {'rec':>6} {'prec':>6} "
          f"{'acc':>6} {'sent':>6}")
    for key, t, n in rows:
        for label, d in (("thinking", t), ("no-thinking", n)):
            if d is None:
                print(f"{key:<20} {label:<12} {'(missing)':>6}")
                continue
            print(f"{key:<20} {label:<12} {fmt(d['f1']):>6} "
                  f"{fmt(d['recall']):>6} {fmt(d['precision']):>6} "
                  f"{fmt(d['acc']):>6} {fmt(d['sent']):>6}")

    out = root / "memorization_matrix.json"
    out.write_text(json.dumps(
        [{"model": k, "thinking": t, "no_thinking": n} for k, t, n in rows],
        indent=2))
    print(f"\nWritten to {out}")


if __name__ == "__main__":
    main()
