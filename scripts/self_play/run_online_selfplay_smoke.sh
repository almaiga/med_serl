#!/bin/bash
# Tiny smoke wrapper for the real online self-play loop.
#
# Uses the same path as:
#   scripts/self_play/run_online_selfplay_training.sh
# but constrains it to a small single-round run so you can verify:
#   1. chained datagen works
#   2. judge connectivity works
#   3. VERL launches and saves an actor checkpoint
#   4. the resulting actor export is usable for the next round

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ACTOR_MODEL="${ACTOR_MODEL:-Abdine/qwen3-8b-medrect-mixed-sft}"
ONLINE_ROUNDS="${ONLINE_ROUNDS:-1}"
TRAIN_EPOCHS_PER_ROUND="${TRAIN_EPOCHS_PER_ROUND:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/self_play_online_vllm_smoke}"
EXPERIMENT_NAME_BASE="${EXPERIMENT_NAME_BASE:-medserl_selfplay_online_vllm_smoke}"

# Match the real path, but keep it tiny.
SMOKE="${SMOKE:-1}"
MAX_PAIRS="${MAX_PAIRS:-12}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
PPO_EPOCHS="${PPO_EPOCHS:-1}"
ROUND_SAVE_FREQ="${ROUND_SAVE_FREQ:-auto}"
TEST_FREQ="${TEST_FREQ:--1}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"

N_GPUS="${N_GPUS:-2}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-6}"
DATAGEN_GPU_MEMORY_UTILIZATION="${DATAGEN_GPU_MEMORY_UTILIZATION:-0.45}"
DATAGEN_MAX_TOKENS="${DATAGEN_MAX_TOKENS:-1024}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-8}"

REQUIRE_JUDGE="${REQUIRE_JUDGE:-1}"
RESUME_INCOMPLETE_ROUND="${RESUME_INCOMPLETE_ROUND:-0}"
KEEP_ONLY_LATEST_CHECKPOINT="${KEEP_ONLY_LATEST_CHECKPOINT:-0}"
WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-medserl-selfplay-smoke}"

echo "=================================================="
echo "Online Self-Play Smoke"
echo "=================================================="
echo "Actor model : ${ACTOR_MODEL}"
echo "Output root : ${OUTPUT_ROOT}"
echo "Rounds      : ${ONLINE_ROUNDS}"
echo "Max pairs   : ${MAX_PAIRS}"
echo "Judge req   : ${REQUIRE_JUDGE}"
echo "W&B         : ${WANDB}"
echo "=================================================="

AUTO_SCREEN="${AUTO_SCREEN:-1}" \
INITIAL_MODEL_PATH="${ACTOR_MODEL}" \
ONLINE_ROUNDS="${ONLINE_ROUNDS}" \
TRAIN_EPOCHS_PER_ROUND="${TRAIN_EPOCHS_PER_ROUND}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
EXPERIMENT_NAME_BASE="${EXPERIMENT_NAME_BASE}" \
SMOKE="${SMOKE}" \
MAX_PAIRS="${MAX_PAIRS}" \
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}" \
VAL_BATCH_SIZE="${VAL_BATCH_SIZE}" \
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE}" \
PPO_EPOCHS="${PPO_EPOCHS}" \
ROUND_SAVE_FREQ="${ROUND_SAVE_FREQ}" \
TEST_FREQ="${TEST_FREQ}" \
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN}" \
N_GPUS="${N_GPUS}" \
ROLLOUT_TP="${ROLLOUT_TP}" \
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION}" \
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS}" \
DATAGEN_GPU_MEMORY_UTILIZATION="${DATAGEN_GPU_MEMORY_UTILIZATION}" \
DATAGEN_MAX_TOKENS="${DATAGEN_MAX_TOKENS}" \
RAY_NUM_CPUS="${RAY_NUM_CPUS}" \
REQUIRE_JUDGE="${REQUIRE_JUDGE}" \
RESUME_INCOMPLETE_ROUND="${RESUME_INCOMPLETE_ROUND}" \
KEEP_ONLY_LATEST_CHECKPOINT="${KEEP_ONLY_LATEST_CHECKPOINT}" \
WANDB="${WANDB}" \
WANDB_PROJECT="${WANDB_PROJECT}" \
bash "${PROJECT_ROOT}/scripts/self_play/run_online_selfplay_training.sh"
