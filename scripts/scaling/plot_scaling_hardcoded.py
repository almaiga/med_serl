#!/usr/bin/env python3
"""Quick plot from hardcoded scaling results."""

from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

# ── Data ─────────────────────────────────────────────────────────────────────

assessor = {
    "n":       [257,  514,  770,  1027, 1284, 1540, 1797, 2053],
    "f1":      [0.574,0.595,0.597,0.574,0.567,0.580,0.559,0.586],
    "acc":     [0.498,0.546,0.558,0.546,0.532,0.521,0.508,0.522],
    "sent_acc":[0.387,0.379,0.379,0.389,0.379,0.396,0.383,0.442],
}

mixed = {
    "n":       [619,  1237, 1855, 2473, 3092, 3710, 4328, 4946],
    "f1":      [0.583,0.549,0.599,0.562,0.533,0.553,0.557,0.568],
    "acc":     [0.523,0.506,0.509,0.504,0.492,0.504,0.502,0.499],
    "sent_acc":[0.373,0.373,0.476,0.429,0.379,0.398,0.413,0.451],
}

baselines = {
    "Base Qwen3-4B": {"f1": 0.651, "acc": 0.574, "sent_acc": 0.543,
                      "color": "#555555", "ls": "--"},
    "Selfplay r2":   {"f1": 0.691, "acc": 0.611, "sent_acc": 0.695,
                      "color": "#E53935", "ls": "-."},
}

METRICS = [
    ("f1",       "Detection F1"),
    ("acc",      "Detection Accuracy"),
    ("sent_acc", "Sentence Localization Accuracy"),
]

C_ASSESSOR = "#2196F3"
C_MIXED    = "#FF9800"

OUT = Path("results/medrect_sft_scaling/scaling_comparison.png")

# ── Plot ──────────────────────────────────────────────────────────────────────

fig = plt.figure(figsize=(15, 5))
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.32)

for col, (metric, ylabel) in enumerate(METRICS):
    ax = fig.add_subplot(gs[col])

    # scaling curves
    ax.plot(assessor["n"], assessor[metric],
            marker="o", color=C_ASSESSOR, linewidth=2, markersize=6,
            label="Assessor-only SFT")
    ax.plot(mixed["n"], mixed[metric],
            marker="s", color=C_MIXED, linewidth=2, markersize=6,
            label="Mixed SFT (assessor+injector)")

    # shade the region between curves
    # interpolate mixed to assessor x grid for comparison
    common_x = np.linspace(
        max(assessor["n"][0], mixed["n"][0]),
        min(assessor["n"][-1], mixed["n"][-1]), 200)
    a_interp = np.interp(common_x, assessor["n"], assessor[metric])
    m_interp = np.interp(common_x, mixed["n"],    mixed[metric])
    ax.fill_between(common_x, a_interp, m_interp, alpha=0.08, color="gray")

    # baselines
    for bname, bvals in baselines.items():
        ax.axhline(bvals[metric], color=bvals["color"], linestyle=bvals["ls"],
                   linewidth=1.6, label=f"{bname} ({bvals[metric]:.3f})", alpha=0.9)

    # best SFT marker
    best_a_idx = int(np.argmax(assessor[metric]))
    ax.scatter([assessor["n"][best_a_idx]], [assessor[metric][best_a_idx]],
               zorder=5, s=80, color=C_ASSESSOR, edgecolors="white", linewidth=1.5)

    ax.set_xlabel("SFT training examples", fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(ylabel, fontsize=12, fontweight="bold", pad=8)

    ymin = min(min(assessor[metric]), min(mixed[metric]),
               min(b[metric] for b in baselines.values())) - 0.05
    ymax = max(min(assessor[metric]), min(mixed[metric]),
               max(b[metric] for b in baselines.values())) + 0.12
    ax.set_ylim(max(0.3, ymin), min(1.0, ymax))
    ax.grid(True, alpha=0.3, linestyle=":")
    ax.tick_params(labelsize=9)

    if col == 0:
        ax.legend(fontsize=8.5, loc="lower right", framealpha=0.92)

fig.suptitle(
    "SFT Scaling Experiment — Assessor-only vs Mixed  |  Baselines: Base Qwen3-4B & Selfplay r2",
    fontsize=13, fontweight="bold", y=1.02
)

OUT.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT, dpi=180, bbox_inches="tight")
plt.close(fig)
print(f"Saved: {OUT}")
