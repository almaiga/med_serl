#!/usr/bin/env python3
"""
Plot self-play training dynamics for supervisor presentation.

Reads all game_*.jsonl files from results/self_play/interactions/ (sorted by
timestamp), treats each entry as one training step, and plots:
  - Average assessor reward vs injector reward
  - Assessor win rate (exact_match) by mode
  - Judge verdict distribution
  - Token usage (reasoning vs final answer)

Usage:
    python scripts/self_play/plot_training_dynamics.py
    python scripts/self_play/plot_training_dynamics.py \
        --log-dir results/self_play/interactions \
        --window 50 \
        --out results/self_play/training_dynamics.png
"""

import argparse
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = PROJECT_ROOT / "results/self_play/interactions"
DEFAULT_OUT = PROJECT_ROOT / "results/self_play/training_dynamics.png"


def load_all_games(log_dir: Path, latest_only: bool = True, gap_hours: float = 3.0, date_filter: str = None):
    """Load game entries. With latest_only=True, only files from the most
    recent continuous session are included. A new session starts when the gap
    between consecutive files exceeds gap_hours."""
    files = sorted(
        list(log_dir.glob("game_*.jsonl")) + list(log_dir.glob("interactions_*.jsonl")),
        key=lambda p: p.stat().st_mtime,
    )
    if not files:
        return []

    if date_filter:
        files = [f for f in files if date_filter in f.name]
        print(f"Date filter '{date_filter}': {len(files)} file(s)")

    if latest_only and not date_filter:
        # Walk backwards from the newest file; stop when gap > gap_hours
        gap_secs = gap_hours * 3600
        session_files = [files[-1]]
        for prev, curr in zip(reversed(files[:-1]), reversed(files[:-1])):
            # compare consecutive mtimes
            break
        # simpler: iterate in reverse and collect while gap is small
        session_files = [files[-1]]
        for i in range(len(files) - 2, -1, -1):
            if files[i + 1].stat().st_mtime - files[i].stat().st_mtime < gap_secs:
                session_files.append(files[i])
            else:
                break
        files = sorted(session_files, key=lambda p: p.stat().st_mtime)
        print(f"Latest run: {len(files)} file(s) from "
              f"{files[0].name} → {files[-1].name}")

    KEEP = {
        "phase", "mode", "error_type", "assessor_outcome", "assessor_reward",
        "injector_reward", "injector_assigned_reward", "injector_outcome",
        "judge_verdict", "judge_status", "game_valid",
        "assessor_reasoning_token_count", "assessor_final_token_count",
        "assessor_token_count", "injector_token_count",
        "assessor_think_cap_hit", "assessor_final_cap_hit",
        "timestamp", "note_id",
    }
    rows = []
    for f in files:
        with open(f) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if r.get("phase") == "game_complete" or "assessor_outcome" in r:
                        rows.append({k: v for k, v in r.items() if k in KEEP})
                except json.JSONDecodeError:
                    continue
    return rows


def rolling(arr, w):
    arr = np.array(arr, dtype=float)
    if len(arr) < w:
        return np.full(len(arr), np.nan)
    out = np.convolve(arr, np.ones(w) / w, mode="valid")
    pad = np.full(w - 1, np.nan)
    return np.concatenate([pad, out])


def plot(rows, window, out):
    if not rows:
        print("No game rows found.")
        return

    # Extract per-step signals
    assessor_rewards, injector_rewards = [], []
    assessor_wins, injector_wins = [], []
    benign_correct, error_correct = [], []
    judge_pass, judge_fail, judge_abstain = [], [], []
    reasoning_tokens, final_tokens = [], []

    for r in rows:
        outcome = r.get("assessor_outcome", r.get("outcome", ""))
        mode = r.get("mode", "")
        a_rew = float(r.get("assessor_reward", 0) or 0)
        i_rew = float(r.get("injector_assigned_reward", r.get("injector_reward", 0)) or 0)
        jv = (r.get("judge_verdict") or "").upper()

        assessor_rewards.append(a_rew)
        injector_rewards.append(i_rew)
        assessor_wins.append(1 if outcome == "exact_match" else 0)
        injector_wins.append(1 if outcome in ("miss", "invalid_format", "game_invalid") else 0)

        if mode == "benign":
            benign_correct.append(1 if outcome == "exact_match" else 0)
            error_correct.append(np.nan)
        elif mode == "error_injection":
            error_correct.append(1 if outcome == "exact_match" else 0)
            benign_correct.append(np.nan)
        else:
            benign_correct.append(np.nan)
            error_correct.append(np.nan)

        judge_pass.append(1 if jv == "PASS" else 0)
        judge_fail.append(1 if jv == "FAIL" else 0)
        judge_abstain.append(1 if jv == "ABSTAIN" else 0)

        rt = r.get("assessor_reasoning_token_count")
        ft = r.get("assessor_final_token_count")
        reasoning_tokens.append(float(rt) if rt is not None else np.nan)
        final_tokens.append(float(ft) if ft is not None else np.nan)

    steps = np.arange(len(rows))
    w = min(window, max(1, len(rows) // 5))

    # ── Figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.32)

    C_ASSESSOR = "#2196F3"
    C_INJECTOR = "#FF9800"
    C_BENIGN   = "#4CAF50"
    C_ERROR    = "#E53935"
    C_JUDGE    = "#9C27B0"

    def plot_with_raw(ax, xs, ys, color, label, ylabel, ylim=None):
        ys = np.array(ys, dtype=float)
        rm = rolling(ys, w)
        # raw scatter (very light)
        valid = ~np.isnan(ys)
        ax.scatter(xs[valid], ys[valid], color=color, alpha=0.08, s=8, zorder=1)
        ax.plot(xs, rm, color=color, linewidth=2, label=label, zorder=2)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_xlabel("Training step", fontsize=9)
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(True, alpha=0.25, linestyle=":")
        ax.legend(fontsize=9, loc="best")

    # 1. Assessor reward vs Injector reward
    ax1 = fig.add_subplot(gs[0, 0])
    rm_a = rolling(np.array(assessor_rewards, dtype=float), w)
    rm_i = rolling(np.array(injector_rewards, dtype=float), w)
    ax1.scatter(steps, assessor_rewards, color=C_ASSESSOR, alpha=0.08, s=8)
    ax1.scatter(steps, injector_rewards, color=C_INJECTOR, alpha=0.08, s=8)
    ax1.plot(steps, rm_a, color=C_ASSESSOR, linewidth=2, label="Assessor reward")
    ax1.plot(steps, rm_i, color=C_INJECTOR, linewidth=2, label="Injector reward")
    ax1.set_ylabel("Average reward", fontsize=10)
    ax1.set_xlabel("Training step", fontsize=9)
    ax1.set_title("Assessor vs Injector Reward", fontsize=11, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.25, linestyle=":")

    # 2. Win rates
    ax2 = fig.add_subplot(gs[0, 1])
    rm_aw = rolling(np.array(assessor_wins, dtype=float), w)
    rm_iw = rolling(np.array(injector_wins, dtype=float), w)
    ax2.plot(steps, rm_aw, color=C_ASSESSOR, linewidth=2, label="Assessor win rate")
    ax2.plot(steps, rm_iw, color=C_INJECTOR, linewidth=2, label="Injector win rate", linestyle="--")
    ax2.axhline(0.5, color="gray", linestyle=":", linewidth=1, alpha=0.7)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Win rate", fontsize=10)
    ax2.set_xlabel("Training step", fontsize=9)
    ax2.set_title("Assessor vs Injector Win Rate", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.25, linestyle=":")

    # 3. Accuracy by mode
    ax3 = fig.add_subplot(gs[1, 0])
    bc = np.array(benign_correct, dtype=float)
    ec = np.array(error_correct, dtype=float)
    rm_b = rolling(np.where(~np.isnan(bc), bc, np.nan), w)
    rm_e = rolling(np.where(~np.isnan(ec), ec, np.nan), w)
    ax3.plot(steps, rm_b, color=C_BENIGN, linewidth=2, label="Benign (no-error detection)")
    ax3.plot(steps, rm_e, color=C_ERROR, linewidth=2, label="Error injection (localization)")
    ax3.set_ylim(0, 1)
    ax3.set_ylabel("Accuracy", fontsize=10)
    ax3.set_xlabel("Training step", fontsize=9)
    ax3.set_title("Assessor Accuracy by Mode", fontsize=11, fontweight="bold")
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.25, linestyle=":")

    # 4. Judge verdict distribution (stacked area)
    ax4 = fig.add_subplot(gs[1, 1])
    rm_jp = rolling(np.array(judge_pass,    dtype=float), w)
    rm_jf = rolling(np.array(judge_fail,    dtype=float), w)
    rm_ja = rolling(np.array(judge_abstain, dtype=float), w)
    ax4.plot(steps, rm_jp, color=C_JUDGE,    linewidth=2, label="Judge PASS")
    ax4.plot(steps, rm_jf, color=C_ERROR,    linewidth=2, label="Judge FAIL",    linestyle="--")
    ax4.plot(steps, rm_ja, color="#888888",  linewidth=1.5, label="Judge ABSTAIN", linestyle=":")
    ax4.set_ylim(0, 1)
    ax4.set_ylabel("Rate", fontsize=10)
    ax4.set_xlabel("Training step", fontsize=9)
    ax4.set_title("Judge Verdict Distribution", fontsize=11, fontweight="bold")
    ax4.legend(fontsize=9)
    ax4.grid(True, alpha=0.25, linestyle=":")

    # 5. Token usage (reasoning vs final answer)
    ax5 = fig.add_subplot(gs[2, 0])
    rt = np.array(reasoning_tokens, dtype=float)
    ft = np.array(final_tokens, dtype=float)
    has_tokens = not np.all(np.isnan(rt))
    if has_tokens:
        rm_rt = rolling(rt, w)
        rm_ft = rolling(ft, w)
        ax5.plot(steps, rm_rt, color="#5C6BC0", linewidth=2, label="Reasoning tokens (<think>)")
        ax5.plot(steps, rm_ft, color="#26A69A", linewidth=2, label="Final answer tokens")
        ax5.set_ylabel("Avg token count", fontsize=10)
        ax5.set_title("Assessor Token Usage", fontsize=11, fontweight="bold")
    else:
        total_tokens = [float(r.get("assessor_token_count", 0) or 0) for r in rows]
        rm_tt = rolling(np.array(total_tokens, dtype=float), w)
        ax5.plot(steps, rm_tt, color="#5C6BC0", linewidth=2, label="Total assessor tokens")
        ax5.set_ylabel("Avg token count", fontsize=10)
        ax5.set_title("Assessor Token Usage", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Training step", fontsize=9)
    ax5.legend(fontsize=9)
    ax5.grid(True, alpha=0.25, linestyle=":")

    # 6. Outcome distribution bar (overall summary)
    ax6 = fig.add_subplot(gs[2, 1])
    from collections import Counter
    outcome_counts = Counter(r.get("assessor_outcome", r.get("outcome", "unknown")) for r in rows)
    labels = sorted(outcome_counts, key=lambda x: -outcome_counts[x])
    values = [outcome_counts[l] for l in labels]
    colors_bar = [C_ASSESSOR if l == "exact_match" else
                  C_INJECTOR if l in ("miss",) else
                  C_ERROR    if "invalid" in l or "game" in l else
                  "#888888" for l in labels]
    bars = ax6.barh(labels, values, color=colors_bar, edgecolor="white", linewidth=0.5)
    total = sum(values)
    for bar, val in zip(bars, values):
        ax6.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val} ({100*val/total:.1f}%)", va="center", fontsize=9)
    ax6.set_xlabel("Count", fontsize=9)
    ax6.set_title("Overall Outcome Distribution", fontsize=11, fontweight="bold")
    ax6.grid(True, alpha=0.25, linestyle=":", axis="x")

    fig.suptitle(
        f"MedSeRL Self-Play Training Dynamics  |  {len(rows)} games  |  "
        f"rolling window={w}",
        fontsize=13, fontweight="bold", y=1.01,
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    print(f"\nSummary ({len(rows)} games):")
    print(f"  Assessor win rate : {np.mean(assessor_wins):.1%}")
    print(f"  Avg assessor rew  : {np.nanmean(assessor_rewards):.3f}")
    print(f"  Avg injector rew  : {np.nanmean(injector_rewards):.3f}")
    print(f"  Judge PASS rate   : {np.mean(judge_pass):.1%}")
    print(f"  Benign accuracy   : {np.nanmean(bc):.1%}")
    print(f"  Error accuracy    : {np.nanmean(ec):.1%}")


def print_table(rows):
    from collections import Counter
    total = len(rows)
    outcomes = Counter(r.get("assessor_outcome", r.get("outcome", "?")) for r in rows)
    modes = Counter(r.get("mode", "?") for r in rows)
    judges = Counter((r.get("judge_verdict") or "N/A").upper() for r in rows)

    a_rews = [float(r.get("assessor_reward", 0) or 0) for r in rows]
    i_rews = [float(r.get("injector_assigned_reward", r.get("injector_reward", 0)) or 0) for r in rows]

    benign_rows = [r for r in rows if r.get("mode") == "benign"]
    error_rows  = [r for r in rows if r.get("mode") == "error_injection"]
    benign_acc  = sum(1 for r in benign_rows if r.get("assessor_outcome") == "exact_match") / max(len(benign_rows), 1)
    error_acc   = sum(1 for r in error_rows  if r.get("assessor_outcome") == "exact_match") / max(len(error_rows),  1)

    print(f"\n{'='*55}")
    print(f"  Self-Play Training Summary  ({total} games)")
    print(f"{'='*55}")
    print(f"  {'Metric':<30} {'Value':>10}")
    print(f"  {'-'*42}")
    print(f"  {'Assessor win rate':<30} {outcomes.get('exact_match',0)/total:>10.1%}")
    print(f"  {'Avg assessor reward':<30} {sum(a_rews)/total:>10.3f}")
    print(f"  {'Avg injector reward':<30} {sum(i_rews)/total:>10.3f}")
    print(f"  {'Benign accuracy':<30} {benign_acc:>10.1%}  (n={len(benign_rows)})")
    print(f"  {'Error injection accuracy':<30} {error_acc:>10.1%}  (n={len(error_rows)})")
    print(f"\n  {'Outcome':<25} {'Count':>6} {'%':>7}")
    print(f"  {'-'*40}")
    for k, v in sorted(outcomes.items(), key=lambda x: -x[1]):
        print(f"  {k:<25} {v:>6} {v/total:>7.1%}")
    print(f"\n  {'Judge verdict':<25} {'Count':>6} {'%':>7}")
    print(f"  {'-'*40}")
    for k, v in sorted(judges.items(), key=lambda x: -x[1]):
        print(f"  {k:<25} {v:>6} {v/total:>7.1%}")
    print(f"{'='*55}\n")


def export_data(rows, out_json: Path):
    """Export slim version of rows for local plotting."""
    slim = []
    for r in rows:
        slim.append({
            "mode":              r.get("mode"),
            "assessor_outcome":  r.get("assessor_outcome", r.get("outcome")),
            "assessor_reward":   float(r.get("assessor_reward", 0) or 0),
            "injector_reward":   float(r.get("injector_assigned_reward", r.get("injector_reward", 0)) or 0),
            "judge_verdict":     r.get("judge_verdict"),
            "error_type":        r.get("error_type"),
            "assessor_reasoning_token_count": r.get("assessor_reasoning_token_count"),
            "assessor_final_token_count":     r.get("assessor_final_token_count"),
            "assessor_token_count":           r.get("assessor_token_count"),
        })
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(slim, f)
    print(f"Data exported to: {out_json}")
    print(f"Copy to local with:")
    print(f"  scp root@<server>:{out_json} ~/Desktop/selfplay_data.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log-dir", type=Path, default=DEFAULT_LOG_DIR)
    parser.add_argument("--window", type=int, default=30,
                        help="Rolling average window size (default: 30)")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--all", action="store_true",
                        help="Use all logs instead of latest run only")
    parser.add_argument("--date", type=str, default=None,
                        help="Only load files containing this string in filename, e.g. 20260430")
    parser.add_argument("--gap-hours", type=float, default=3.0,
                        help="Max hours between files to be considered same run (default: 3)")
    parser.add_argument("--export", type=Path, default=None,
                        help="Export slim JSON for local plotting")
    parser.add_argument("--table-only", action="store_true",
                        help="Print table and export data without generating plot")
    args = parser.parse_args()

    print(f"Loading games from: {args.log_dir}")
    rows = load_all_games(args.log_dir, latest_only=not args.all, gap_hours=args.gap_hours, date_filter=args.date)
    print(f"Found {len(rows)} game entries")

    if not rows:
        print("No game entries found. Check --log-dir.")
        return

    print_table(rows)

    export_path = args.export or args.out.parent / "selfplay_data.json"
    export_data(rows, export_path)

    if not args.table_only:
        plot(rows, args.window, args.out)


if __name__ == "__main__":
    main()
