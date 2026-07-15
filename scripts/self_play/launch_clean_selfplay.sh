#!/usr/bin/env bash
# launch_clean_selfplay.sh — launch the clean v7 self-play run with the fixed judge.
#
# Bakes in the working v6 config PLUS the fixes learned the hard way:
#   - judge thinking ON (via judge_client.py / medrect_judge_prompts.json)
#   - RESUME_MODE=auto           (never restart from step 0 on autorestart)
#   - KEEP_ONLY_LATEST_CHECKPOINT=0  (keep every checkpoint)
#   - reminds you to run checkpoint_watcher.sh so checkpoints reach HF verified
#
# REQUIRED env:
#   JUDGE_VLLM_URL   full chat/completions URL of the running MedRECT-32B judge
#
# Usage:
#   JUDGE_VLLM_URL="http://<judge-host>:8002/v1/chat/completions" \
#     bash scripts/self_play/launch_clean_selfplay.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

: "${JUDGE_VLLM_URL:?set JUDGE_VLLM_URL to the running judge server}"

# ── Config (override any via env) ────────────────────────────────────────────
export ACTOR_MODEL="${ACTOR_MODEL:-Abdine/qwen3-4b-medrect-mixed-v2}"
export OUTPUT_DIR="${OUTPUT_DIR:-outputs/self_play_v7}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-medserl_selfplay_v7}"
export N_GPUS="${N_GPUS:-2}"
export JUDGE_MODEL="${JUDGE_MODEL:-pfnet/Preferred-MedRECT-32B}"
export JUDGE_TYPE="${JUDGE_TYPE:-detection}"
export JUDGE_PROMPT_STYLE="${JUDGE_PROMPT_STYLE:-hint_v2}"

export MAX_PAIRS="${MAX_PAIRS:-0}"            # 0 = full dataset
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-8}"
export PPO_EPOCHS="${PPO_EPOCHS:-2}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-5}"
export SAVE_FREQ="${SAVE_FREQ:-33}"
export ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"
export KL_COEF="${KL_COEF:-0.01}"

# The fixes vs v6:
export RESUME_MODE="${RESUME_MODE:-auto}"
export KEEP_ONLY_LATEST_CHECKPOINT="${KEEP_ONLY_LATEST_CHECKPOINT:-0}"

# Autorestart wrapper knobs
export SMOKE=0
export AUTO_SCREEN="${AUTO_SCREEN:-1}"
export WANDB="${WANDB:-1}"
export RESTART_AFTER_CHECKPOINT="${RESTART_AFTER_CHECKPOINT:-1}"
export RESTART_ON_FAILURE="${RESTART_ON_FAILURE:-1}"
export MAX_AUTORESTARTS="${MAX_AUTORESTARTS:-500}"
export CHECKPOINT_RESTART_GRACE_SEC="${CHECKPOINT_RESTART_GRACE_SEC:-30}"
export RUNNER_SCRIPT="${RUNNER_SCRIPT:-scripts/self_play/run_multiturn_training.sh}"

# Checkpoint watcher: auto-start in its own screen session (START_WATCHER=0 to skip)
export REPO_PREFIX="${REPO_PREFIX:-Abdine/qwen3-4b-medserl-v7-step}"
export PRIVATE="${PRIVATE:-1}"
START_WATCHER="${START_WATCHER:-1}"
WATCHER_SESSION="medserl_ckpt_watcher"

echo "=========================================================="
echo " LAUNCH clean self-play v7"
echo "   actor        : $ACTOR_MODEL"
echo "   judge        : $JUDGE_MODEL ($JUDGE_TYPE / $JUDGE_PROMPT_STYLE)"
echo "   judge url    : $JUDGE_VLLM_URL"
echo "   output       : $OUTPUT_DIR   resume=$RESUME_MODE  keep_all=$([[ $KEEP_ONLY_LATEST_CHECKPOINT == 0 ]] && echo yes)"
echo "   batch/mini   : $TRAIN_BATCH_SIZE / $PPO_MINI_BATCH_SIZE   kl=$KL_COEF"
echo "   ckpt watcher : ${REPO_PREFIX}<N>  private=$PRIVATE"
echo "=========================================================="

if [[ "$START_WATCHER" == "1" ]]; then
    if ! command -v screen >/dev/null 2>&1; then
        echo "ERROR: 'screen' not installed (apt-get install -y screen)"; exit 1
    fi
    if screen -list | grep -q "\.${WATCHER_SESSION}[[:space:]]"; then
        echo ">> checkpoint watcher already running (screen -r ${WATCHER_SESSION})"
    else
        mkdir -p logs/screen
        screen -L -Logfile "logs/screen/ckpt_watcher_$(date +%Y%m%d_%H%M%S).log" \
            -dmS "$WATCHER_SESSION" bash -lc \
            "OUTPUT_DIR=$(printf '%q' "$OUTPUT_DIR") REPO_PREFIX=$(printf '%q' "$REPO_PREFIX") PRIVATE=$(printf '%q' "$PRIVATE") bash scripts/self_play/checkpoint_watcher.sh"
        echo ">> checkpoint watcher started in screen '${WATCHER_SESSION}'"
    fi
fi

# Training: the autorestart wrapper self-launches into its own screen session
# (AUTO_SCREEN=1 above), so this command returns and both jobs run detached.
bash scripts/self_play/run_online_selfplay_autorestart.sh

echo
echo ">> Both running in screen. Useful commands:"
echo "   screen -ls                                   # list sessions"
echo "   screen -r ${WATCHER_SESSION}                 # attach watcher (Ctrl-A d to detach)"
echo "   bash scripts/self_play/monitor_training.sh   # health snapshot"
