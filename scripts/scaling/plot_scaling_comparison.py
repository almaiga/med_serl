#!/usr/bin/env python3
"""
Plot learning curves for all scaling experiments + reference baselines.

Usage:
    python scripts/scaling/plot_scaling_comparison.py
    python scripts/scaling/plot_scaling_comparison.py --metric f1 --out results/scaling_comparison.png
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCALING_EXPERIMENTS = {
    "Assessor-only SFT": PROJECT_ROOT / "results/medrect_sft_scaling/assessor_only_scaling",
    "Mixed SFT (assessor+injector)": PROJECT_ROOT / "results/medrect_sft_scaling/mixed_scaling",
}

BASELINES = {
    "Base Qwen3-4B": PROJECT_ROOT / "results/medrect_sft_scaling/base_model_eval",
    "Selfplay r2 (SFT-mix + RL)": PROJECT_ROOT / "results/detection/4b_Abdine_medserl-qwen3-4b-medrect-mixed-selfplay-r2",
    "SFT assessor-only (published)": PROJECT_ROOT / "results/detection/4b_Abdine_qwen3-4b-medrect-assessor",
}

METRIC_KEYS = {
    "accuracy": ("detection", "accuracy"),
    "f1":       ("detection", "f1"),
    "precision":("detection", "precision"),
    "recall":   ("detection", "recall"),
    "sent_acc": ("sentence_extraction", "accuracy"),
}

COLORS = ["#2196F3", "#FF9800", "#9C27B0", "#4CAF50", "#F44336", "#00BCD4"]
BASELINE_COLORS = ["#333333", "#E53935", "#43A047", "#FB8C00"]


def read_summary(path: Path, metric_keys):
    with open(path) as f:
        s = json.load(f)
    m = s["metrics"]
    section, key = metric_keys
    return m[section][key], m["total_samples"]


def load_scaling_curve(exp_dir: Path, dataset: str, metric_keys):
    points = []
    for eval_dir in sorted(exp_dir.glob("eval_frac_*_of_*")):
        # read train_count from manifest
        frac_part = eval_dir.name  # e.g. eval_frac_03_of_8
        parts = frac_part.split("_")
        idx, total = int(parts[2]), int(parts[4])

        summaries = sorted(eval_dir.glob(f"{dataset}_*_summary.json"))
        if not summaries:
            continue
        vals = []
        for s in summaries:
            try:
                v, n = read_summary(s, metric_keys)
                vals.append(v)
            except Exception:
                pass
        if not vals:
            continue

        # get train_count from manifest — may be in outputs/ tree, not results/
        train_count = None
        for manifest_path in [
            exp_dir / "splits" / "manifest.json",
            PROJECT_ROOT / "outputs" / "local_training" / "medrect_sft_scaling" / exp_dir.name / "splits" / "manifest.json",
        ]:
            if manifest_path.exists():
                with open(manifest_path) as f:
                    manifest = json.load(f)
                for split in manifest["splits"]:
                    if split["index"] == idx:
                        train_count = split["train_count"]
                        break
                break
        if train_count is None:
            train_count = idx  # fallback

        mean = sum(vals) / len(vals)
        std = (sum((v - mean) ** 2 for v in vals) / len(vals)) ** 0.5 if len(vals) > 1 else 0
        points.append((train_count, mean, std))

    points.sort(key=lambda x: x[0])
    return points


def load_baseline(baseline_dir: Path, dataset: str, metric_keys):
    summaries = sorted(baseline_dir.glob(f"{dataset}_*_summary.json"))
    if not summaries:
        return None
    vals = []
    for s in summaries:
        try:
            v, _ = read_summary(s, metric_keys)
            vals.append(v)
        except Exception:
            pass
    return sum(vals) / len(vals) if vals else None


def plot(metric: str, dataset: str, out: Path):
    metric_keys = METRIC_KEYS[metric]
    fig, ax = plt.subplots(figsize=(9, 5.5))

    # --- scaling curves ---
    x_max = 0
    for i, (label, exp_dir) in enumerate(SCALING_EXPERIMENTS.items()):
        if not exp_dir.exists():
            print(f"  Skipping (not found): {exp_dir}")
            continue
        points = load_scaling_curve(exp_dir, dataset, metric_keys)
        if not points:
            print(f"  No data in: {exp_dir}")
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        stds = [p[2] for p in points]
        color = COLORS[i % len(COLORS)]
        ax.plot(xs, ys, marker="o", label=label, color=color, linewidth=2, markersize=6)
        if any(s > 0 for s in stds):
            ax.fill_between(xs,
                            [y - s for y, s in zip(ys, stds)],
                            [y + s for y, s in zip(ys, stds)],
                            alpha=0.15, color=color)
        x_max = max(x_max, max(xs))

    # --- baselines (horizontal lines) ---
    for i, (label, bdir) in enumerate(BASELINES.items()):
        if not bdir.exists():
            continue
        val = load_baseline(bdir, dataset, metric_keys)
        if val is None:
            continue
        color = BASELINE_COLORS[i % len(BASELINE_COLORS)]
        ax.axhline(val, linestyle="--", linewidth=1.5, color=color,
                   label=f"{label} ({val:.3f})", alpha=0.85)

    metric_label = {
        "accuracy": "Detection Accuracy",
        "f1": "Detection F1",
        "precision": "Detection Precision",
        "recall": "Detection Recall",
        "sent_acc": "Sentence Localization Accuracy",
    }[metric]

    ax.set_xlabel("SFT training examples", fontsize=12)
    ax.set_ylabel(metric_label, fontsize=12)
    ax.set_title(f"Scaling experiment — {metric_label} vs training data size\n(dataset={dataset})", fontsize=13)
    ax.set_ylim(0.3, 0.85)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180)
    plt.close(fig)
    print(f"Saved: {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", default="f1",
                        choices=list(METRIC_KEYS.keys()),
                        help="Metric to plot (default: f1)")
    parser.add_argument("--dataset", default="all", choices=["all", "ms", "uw"])
    parser.add_argument("--out", default=None,
                        help="Output PNG path (default: results/medrect_sft_scaling/scaling_comparison_{metric}.png)")
    args = parser.parse_args()

    out = Path(args.out) if args.out else (
        PROJECT_ROOT / f"results/medrect_sft_scaling/scaling_comparison_{args.metric}.png"
    )

    print(f"Plotting metric: {args.metric}, dataset: {args.dataset}")
    plot(args.metric, args.dataset, out)

    # also plot accuracy and sent_acc alongside f1
    if args.metric == "f1":
        for m in ("accuracy", "sent_acc"):
            extra_out = out.parent / f"scaling_comparison_{m}.png"
            print(f"Also plotting: {m}")
            plot(m, args.dataset, extra_out)


if __name__ == "__main__":
    main()
