#!/usr/bin/env python3
"""Aggregate Exp 2 results across every (style, model) JSONL under results/exp2/.

Prints:
1. Per-bucket accuracy table (each row = one style+model combo).
2. Per-bucket verdict mix (so you can see whether failures are localization
   misses (ABSTAIN), missed errors (SAME on a CHANGED bucket), or over-flags
   (CHANGED on a SAME bucket).
3. Ship-decision diff: current judge (qwen_pair@qwen8b) -> candidate judges.

Reads every file matching results/exp2/<style>__<model_tag>.jsonl produced by
exp2_run_probes.py.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_DIR = PROJECT_ROOT / "results" / "exp2"


def parse_filename(path: Path) -> tuple[str, str]:
    """results/exp2/<style>__<model_tag>.jsonl -> (style, model_tag)."""
    name = path.stem
    if "__" in name:
        style, model_tag = name.split("__", 1)
    else:
        style, model_tag = name, "?"
    return style, model_tag


def load_results(path: Path) -> list[dict]:
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def per_bucket_accuracy(rows: list[dict]) -> dict[str, tuple[int, int]]:
    buckets: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for r in rows:
        buckets[r["bucket"]][1] += 1
        if r["correct"]:
            buckets[r["bucket"]][0] += 1
    return {b: (c, n) for b, (c, n) in buckets.items()}


def per_bucket_verdicts(rows: list[dict]) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in rows:
        out[r["bucket"]][r["verdict"]] += 1
    return out


def fmt_pct(c: int, n: int) -> str:
    return f"{c/n:>3.0%}" if n else " -- "


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", type=Path, default=DEFAULT_DIR,
                    help="directory holding the exp2 result JSONLs")
    ap.add_argument("--baseline", default="qwen_pair__qwen8b",
                    help="style__model_tag to treat as the current judge "
                         "(used for the ship diff)")
    args = ap.parse_args()

    files = sorted(args.dir.glob("*.jsonl"))
    if not files:
        print(f"no result files under {args.dir}")
        return 1

    runs: dict[tuple[str, str], list[dict]] = {}
    for f in files:
        style, model_tag = parse_filename(f)
        runs[(style, model_tag)] = load_results(f)

    buckets = sorted({r["bucket"] for rs in runs.values() for r in rs})
    bucket_names: dict[str, str] = {}
    expected_by_bucket: dict[str, str] = {}
    for rs in runs.values():
        for r in rs:
            bucket_names[r["bucket"]] = r["bucket_name"]
            expected_by_bucket[r["bucket"]] = r["expected"]

    # --- (1) accuracy table -------------------------------------------------
    print("=" * 78)
    print("PER-BUCKET ACCURACY (rows = (style, model))")
    print("=" * 78)
    bucket_header = "  ".join(f"{b:>4s}" for b in buckets)
    expected_row = "  ".join(f"{expected_by_bucket[b][:4]:>4s}" for b in buckets)
    print(f"{'style':<17s} {'model':<12s}  {bucket_header}  {'all':>4s}  n")
    print(f"{'':<17s} {'(expected:)':<12s}  {expected_row}")
    print("-" * 78)
    for (style, model_tag), rs in sorted(runs.items()):
        accs = per_bucket_accuracy(rs)
        total_ok = sum(c for c, _ in accs.values())
        total_n = sum(n for _, n in accs.values())
        cells = [fmt_pct(*accs.get(b, (0, 0))) for b in buckets]
        line = (f"{style:<17s} {model_tag:<12s}  "
                + "  ".join(cells)
                + f"  {fmt_pct(total_ok, total_n)}  {total_n}")
        print(line)

    # --- (2) verdict mix per bucket -----------------------------------------
    print()
    print("=" * 78)
    print("PER-BUCKET VERDICT MIX  (expected verdict shown in [ ])")
    print("=" * 78)
    verdict_cols = ["SAME", "CHANGED", "ABSTAIN"]
    for b in buckets:
        exp = expected_by_bucket[b]
        print(f"\nbucket {b} [{exp}]  ({bucket_names[b]})")
        print(f"  {'style':<17s} {'model':<12s}  "
              + "  ".join(f"{v:>8s}" for v in verdict_cols))
        for (style, model_tag), rs in sorted(runs.items()):
            v = per_bucket_verdicts(rs).get(b, {})
            total = sum(v.values())
            if not total:
                continue
            cells = []
            for verdict in verdict_cols:
                n = v.get(verdict, 0)
                pct = n / total
                tag = ""
                if verdict == exp and n:
                    tag = "*"  # the right answer
                cells.append(f"{n:>3d} ({pct:>2.0%}){tag}")
            print(f"  {style:<17s} {model_tag:<12s}  " + "  ".join(cells))

    # --- (3) ship-decision diff ---------------------------------------------
    print()
    print("=" * 78)
    print(f"SHIP DIFF: baseline ({args.baseline}) vs other runs")
    print("=" * 78)
    base_style, _, base_model = args.baseline.partition("__")
    base_key = (base_style, base_model)
    if base_key not in runs:
        print(f"  baseline {args.baseline!r} not found in results dir; "
              f"available: {sorted(f'{s}__{m}' for s, m in runs)}")
        return 0
    base_accs = per_bucket_accuracy(runs[base_key])
    for (style, model_tag), rs in sorted(runs.items()):
        if (style, model_tag) == base_key:
            continue
        accs = per_bucket_accuracy(rs)
        print(f"\n  -> {style}__{model_tag}")
        for b in buckets:
            bc, bn = base_accs.get(b, (0, 0))
            cc, cn = accs.get(b, (0, 0))
            if not (bn and cn):
                continue
            delta = cc / cn - bc / bn
            arrow = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "=")
            print(f"    bucket {b} ({bucket_names[b]:18s})  "
                  f"{bc/bn:>3.0%} -> {cc/cn:>3.0%}  "
                  f"({arrow}{abs(delta):>3.0%})")
        # overall
        b_ok = sum(c for c, _ in base_accs.values())
        b_n = sum(n for _, n in base_accs.values())
        c_ok = sum(c for c, _ in accs.values())
        c_n = sum(n for _, n in accs.values())
        if b_n and c_n:
            delta = c_ok / c_n - b_ok / b_n
            arrow = "↑" if delta > 0.005 else ("↓" if delta < -0.005 else "=")
            print(f"    overall                            "
                  f"{b_ok/b_n:>3.0%} -> {c_ok/c_n:>3.0%}  "
                  f"({arrow}{abs(delta):>3.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
