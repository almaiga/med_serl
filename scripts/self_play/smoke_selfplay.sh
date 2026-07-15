#!/usr/bin/env bash
# smoke_selfplay.sh — short self-play run to test the FIXED judge on the
# injector's ACTUAL output (not synthetic cases), then inspect the game log.
#
# This is the last cheap gate before the expensive full run. It runs a handful
# of real games through the thinking-on judge and checks:
#   - judge_status=ok is high         (thinking judge wired correctly live)
#   - SAME-on-error is low             (judge catching real injected errors)
#   - game_invalid is low              (rollouts not wasted)
#
# REQUIRED env:  JUDGE_VLLM_URL
# Usage:
#   JUDGE_VLLM_URL="http://<judge-host>:8002/v1/chat/completions" \
#     bash scripts/self_play/smoke_selfplay.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

: "${JUDGE_VLLM_URL:?set JUDGE_VLLM_URL to the running judge server}"

export ACTOR_MODEL="${ACTOR_MODEL:-Abdine/qwen3-4b-medrect-mixed-v2}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/smoke_v7}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-medserl_smoke_v7}"
export N_GPUS="${N_GPUS:-2}"
export JUDGE_MODEL="${JUDGE_MODEL:-pfnet/Preferred-MedRECT-32B}"
export JUDGE_TYPE=detection
export JUDGE_PROMPT_STYLE=hint_v2

# SMALL run: full code path (SMOKE=0), few pairs, one short epoch, no autorestart
export SMOKE=0
export MAX_PAIRS="${MAX_PAIRS:-24}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
export PPO_EPOCHS=1
export TOTAL_EPOCHS=1
export SAVE_FREQ=999999                 # don't bother saving in smoke
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"
export KL_COEF="${KL_COEF:-0.01}"
export KEEP_ONLY_LATEST_CHECKPOINT=1
export RESUME_MODE=disable
export WANDB=0
export AUTO_SCREEN=0

echo "=========================================================="
echo " SMOKE self-play (MAX_PAIRS=$MAX_PAIRS) — testing fixed judge live"
echo "=========================================================="

bash scripts/self_play/run_multiturn_training.sh || {
    echo "** smoke run exited non-zero — inspect the log above **"; exit 1; }

echo
echo "=== Smoke game-log health ==="
bash scripts/self_play/monitor_training.sh "" results/self_play/interactions

echo
echo "GATE READING:"
echo "  judge_status=ok should be HIGH (>90%)  — thinking judge wired live"
echo "  SAME-on-error should be LOW (<15%)      — judge catching real errors"
echo "  game_invalid should be LOW (<15%)       — rollouts not wasted"
echo "If all three look good, launch the full run. If not, STOP — cheaper to"
echo "find it here than in the \$6/hr full run."
