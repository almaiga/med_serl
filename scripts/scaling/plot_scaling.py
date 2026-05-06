#!/usr/bin/env python3
"""Plot learning curve: detection performance vs. training data size.

Reads *_summary.json files from results/scaling/scale_*/
and produces a learning curve figure + printed table.

Usage
-----
    python scripts/scaling/plot_scaling.py
    python scripts/scaling/plot_scaling.py --results-dir results/scaling --out results/scaling/learning_curve.png
"""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


def load_results(results_dir: Path) -> list[dict]:
    rows = []
    for shard_dir in sorted(results_dir.glob("scale_*of*")):
        if not shard_dir.is_dir():
            continue
        m = re.search(r"scale_(\d+)of(\d+)", shard_dir.name)
        if not m:
            continue
        k, n_shards = int(m.group(1)), int(m.group(2))

        summaries = sorted(shard_dir.glob("*_summary.json"))
        if not summaries:
            print(f"  WARNING: no summary JSON in {shard_dir} — skipping")
            continue

        # Use the most recent summary if multiple exist
        with summaries[-1].open() as f:
            s = json.load(f)

        det  = s["metrics"]["detection"]
        sent = s["metrics"]["sentence_extraction"]

        # Recover training sample count from the shard JSONL if available
        scale_jsonl = Path("data_processed/scaling") / f"{shard_dir.name}.jsonl"
        n_train = None
        if scale_jsonl.exists():
            n_train = sum(1 for _ in scale_jsonl.open())

        rows.append({
            "shard":    k,
            "n_shards": n_shards,
            "n_train":  n_train,
            "frac":     k / n_shards,
            "f1":            det["f1"],
            "precision":     det["precision"],
            "recall":        det["recall"],
            "det_accuracy":  det["accuracy"],
            "loc_accuracy":  sent["accuracy"],
            "summary_file":  str(summaries[-1]),
        })

    rows.sort(key=lambda r: r["shard"])
    return rows


def print_table(rows: list[dict]) -> None:
    print(f"\n{'Shard':<8} {'N train':>8} {'Frac':>6}  {'Det F1':>7}  {'Det Acc':>8}  {'Loc Acc':>8}")
    print("-" * 58)
    for r in rows:
        n = str(r["n_train"]) if r["n_train"] else "?"
        print(f"{r['shard']}/{r['n_shards']:<5} {n:>8} {r['frac']:>6.1%}  "
              f"{r['f1']:>7.3f}  {r['det_accuracy']:>8.3f}  {r['loc_accuracy']:>8.3f}")
    print()


def plot(rows: list[dict], out: Path) -> None:
    xs = [r["n_train"] if r["n_train"] else r["frac"] for r in rows]
    xlabel = "Training samples" if rows[0]["n_train"] else "Training fraction"

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("Detection + Localization — Learning Curve", fontsize=13, fontweight="bold")

    metrics = [
        ("f1",           "Detection F1",          "#2563eb"),
        ("det_accuracy", "Detection Accuracy",     "#16a34a"),
        ("loc_accuracy", "Localization Accuracy",  "#dc2626"),
    ]

    for ax, xscale in zip(axes, ["linear", "log"]):
        for key, label, color in metrics:
            ys = [r[key] for r in rows]
            ax.plot(xs, ys, marker="o", label=label, color=color, linewidth=2, markersize=6)

        ax.set_xlabel(xlabel)
        ax.set_ylabel("Score")
        ax.set_xscale(xscale)
        ax.set_title(f"X scale: {xscale}")
        ax.set_ylim(0, 1.05)
        ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right")

    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved : {out}")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results/scaling")
    p.add_argument("--out",         default=None, help="Output PNG path")
    args = p.parse_args()

    results_dir = Path(args.results_dir)
    out = Path(args.out) if args.out else results_dir / "learning_curve.png"

    rows = load_results(results_dir)
    if not rows:
        print(f"No results found in {results_dir}. Run run_scaling_eval.sh first.")
        return

    print_table(rows)
    plot(rows, out)


if __name__ == "__main__":
    main()
