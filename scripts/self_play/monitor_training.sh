#!/usr/bin/env bash
# monitor_training.sh — one-shot health snapshot of a running self-play run.
#
# Combines PPO training metrics (from the trainer stdout log) with judge/game
# health (from the interaction logs) and flags anomalies. Read-only.
#
# Usage:
#   bash scripts/self_play/monitor_training.sh                      # auto-discover
#   bash scripts/self_play/monitor_training.sh <train.log> <gamedir>
#   watch -n 30 bash scripts/self_play/monitor_training.sh         # live
#
# Health gates (flags a WARN if):
#   - grad_norm > 10            (instability / spike)
#   - judge_status=ok  < 70%    (judge wiring / thinking problem)
#   - SAME-on-error    > 15%    (judge missing injected errors)
#   - game_invalid     > 25%    (rollouts wasted)

set -uo pipefail

TRAIN_LOG="${1:-}"
GAME_DIR="${2:-results/self_play/interactions}"

# ── Auto-discover the trainer log if not given ───────────────────────────────
if [[ -z "$TRAIN_LOG" ]]; then
    for d in logs/screen logs/online_autorestart logs; do
        cand=$(ls -t "$d"/*.log 2>/dev/null | head -1)
        [[ -n "$cand" ]] && { TRAIN_LOG="$cand"; break; }
    done
fi

echo "=========================================================="
echo " self-play training monitor"
echo "   train log : ${TRAIN_LOG:-<none found>}"
echo "   game dir  : ${GAME_DIR}"
echo "=========================================================="

WARN=0

# ── PPO metrics: multi-step, multi-metric trend (pure Python, no jq) ─────────
if [[ -n "${TRAIN_LOG:-}" && -f "$TRAIN_LOG" ]]; then
    echo
    python3 scripts/self_play/train_metrics.py "$TRAIN_LOG" --last 12 || WARN=1
else
    echo "  (no trainer log found)"
fi

# ── Judge / game health + injector adaptation trend (pure Python, no jq) ─────
echo
echo "--- game health (all logs, chronological) ---"
GH_OUT=$(python3 scripts/self_play/game_health.py --all "$GAME_DIR" 2>&1)
echo "$GH_OUT" | sed 's/^/  /'
echo "$GH_OUT" | grep -q "VERDICT: SMOKE FAIL" && WARN=1

# ── Checkpoints ──────────────────────────────────────────────────────────────
echo
echo "--- checkpoints on disk ---"
CKPTS=$(find outputs -type d -name "global_step_*" 2>/dev/null | sort -t_ -k3 -n)
if [[ -n "$CKPTS" ]]; then
    echo "$CKPTS" | sed 's/^/  /'
    echo "  (verify each pushed to HF: huggingface-cli repo-files list-tree <repo>)"
else
    echo "  none yet"
fi

echo
if [[ "$WARN" -eq 0 ]]; then
    echo "STATUS: healthy (no flags)"
else
    echo "STATUS: ** ${WARN} warning(s) above — investigate before trusting the run **"
fi
echo "(re-run: bash scripts/self_play/monitor_training.sh)"
