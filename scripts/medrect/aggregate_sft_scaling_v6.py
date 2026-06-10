#!/usr/bin/env python3
"""Aggregate the v6 SFT-scaling experiment into a paper-ready F1 table.

Reads the per-shard summary JSONs that the scaling script wrote under
`results/sft_scaling_v6/{v6_mixed,v6_assessor_only}/` and produces:
  - results/sft_scaling_v6/scaling_v6_combined.csv
  - results/sft_scaling_v6/scaling_v6_combined.md  (markdown table for the paper)

Usage:
    python3 scripts/medrect/aggregate_sft_scaling_v6.py
    python3 scripts/medrect/aggregate_sft_scaling_v6.py --results-root results/sft_scaling_v6
"""

import argparse
import csv
import glob
import json
import re
from pathlib import Path
from typing import Dict, List, Optional


def find_summary(eval_dir: Path) -> Optional[Path]:
    summaries = sorted(eval_dir.glob("all_*_summary.json"))
    return summaries[-1] if summaries else None


def load_metrics(summary_path: Path) -> Dict:
    with open(summary_path) as f:
        data = json.load(f)
    m = data["metrics"]
    return dict(
        f1=m["detection"]["f1"],
        recall=m["detection"]["recall"],
        precision=m["detection"]["precision"],
        accuracy=m["detection"]["accuracy"],
        sent_acc=m["sentence_extraction"]["accuracy"],
    )


def load_variant(variant_root: Path) -> List[Dict]:
    rows: List[Dict] = []

    base_dir = variant_root / "eval_base_model"
    base_summary = find_summary(base_dir)
    if base_summary:
        rows.append(dict(
            label="base (0%)", count=0,
            **load_metrics(base_summary),
        ))

    for eval_dir in sorted(variant_root.glob("eval_frac_*_of_*")):
        m = re.search(r"frac_(\d+)_of_(\d+)", eval_dir.name)
        if not m:
            continue
        idx, total = int(m.group(1)), int(m.group(2))
        summary = find_summary(eval_dir)
        if not summary:
            continue

        # Pull row count from the manifest so the label is the real training size
        manifest = variant_root.parent.parent / "outputs" / "local_training" \
            / "sft_scaling_v6" / variant_root.name / "splits" / "manifest.json"
        count = None
        if manifest.exists():
            data = json.loads(manifest.read_text())
            for split in data["splits"]:
                if split["index"] == idx:
                    count = split["train_count"]
                    break

        rows.append(dict(
            label=f"frac {idx}/{total}",
            count=count if count is not None else "?",
            **load_metrics(summary),
        ))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results/sft_scaling_v6")
    parser.add_argument(
        "--sft-v2-f1", type=float, default=0.540,
        help="Published F1 of the SFT v2 baseline — used for reproducibility check.",
    )
    args = parser.parse_args()

    root = Path(args.results_root)
    if not root.exists():
        raise SystemExit(f"results root not found: {root}")

    variants = ["v6_mixed", "v6_assessor_only"]
    table: Dict[str, List[Dict]] = {}
    for variant in variants:
        variant_root = root / variant
        if not variant_root.exists():
            print(f"  warn: {variant_root} not found, skipping")
            continue
        table[variant] = load_variant(variant_root)

    # ── CSV ──────────────────────────────────────────────────────────────────
    csv_path = root / "scaling_v6_combined.csv"
    with open(csv_path, "w") as f:
        w = csv.writer(f)
        w.writerow(["variant", "label", "count", "f1", "recall",
                    "precision", "accuracy", "sent_acc"])
        for variant, rows in table.items():
            for r in rows:
                w.writerow([variant, r["label"], r["count"],
                            f"{r['f1']:.4f}", f"{r['recall']:.4f}",
                            f"{r['precision']:.4f}", f"{r['accuracy']:.4f}",
                            f"{r['sent_acc']:.4f}"])

    # ── Markdown ─────────────────────────────────────────────────────────────
    md_lines = ["# v6 SFT Scaling — Combined Results", ""]
    md_lines.append(f"Baseline (published SFT v2): F1 = {args.sft_v2_f1:.3f}")
    md_lines.append("")
    for variant in variants:
        if variant not in table:
            continue
        rows = table[variant]
        if not rows:
            continue
        md_lines.append(f"## {variant}")
        md_lines.append("")
        md_lines.append("| Shard | Rows | F1 | Recall | Precision | Sent acc |")
        md_lines.append("|---|---:|---:|---:|---:|---:|")
        for r in rows:
            md_lines.append(
                f"| {r['label']} | {r['count']} | "
                f"**{r['f1']:.3f}** | {r['recall']:.3f} | "
                f"{r['precision']:.3f} | {r['sent_acc']:.3f} |"
            )
        md_lines.append("")

    # Reproducibility check on the mixed 81.5 % point
    mixed_rows = table.get("v6_mixed", [])
    anchor = next(
        (r for r in mixed_rows if isinstance(r["count"], int) and r["count"] == 4424),
        None,
    )
    if anchor is not None:
        delta = anchor["f1"] - args.sft_v2_f1
        verdict = "PASS" if abs(delta) < 0.01 else "REVIEW"
        md_lines.append("## Reproducibility check")
        md_lines.append("")
        md_lines.append(
            f"Mixed 81.5 % shard (4,424 rows = `mixed_sft_train.jsonl`): "
            f"**F1 {anchor['f1']:.3f}**  vs  published SFT v2 F1 "
            f"{args.sft_v2_f1:.3f}  →  Δ {delta:+.3f}  →  **{verdict}**"
        )
        md_lines.append("")
        if abs(delta) >= 0.01:
            md_lines.append(
                "> Δ ≥ 0.01 — training config or seed may have drifted. "
                "Spot-check the LoRA hyperparams and inference seed before "
                "reporting the curve."
            )
            md_lines.append("")

    md_path = root / "scaling_v6_combined.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"\nWrote:\n  {csv_path}\n  {md_path}")
    print()
    print("\n".join(md_lines))


if __name__ == "__main__":
    main()
