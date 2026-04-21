#!/bin/bash

set -euo pipefail

SCREEN_SESSION="${SCREEN_SESSION:-medserl_online_selfplay_push}"
AUTO_SCREEN="${AUTO_SCREEN:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCREEN_LOG_DIR="${SCREEN_LOG_DIR:-${REPO_ROOT}/logs/screen}"

if [ "$AUTO_SCREEN" = "1" ] && [ -z "${STY:-}" ]; then
    if ! command -v screen >/dev/null 2>&1; then
        echo "ERROR: 'screen' is required but not installed."
        exit 1
    fi

    if screen -list | grep -q "[[:space:]]${SCREEN_SESSION}[[:space:]]"; then
        echo "Screen session '${SCREEN_SESSION}' already exists."
        echo "Attach with: screen -r ${SCREEN_SESSION}"
        echo "Kill with:   screen -X -S ${SCREEN_SESSION} quit"
        exit 1
    fi

    mkdir -p "${SCREEN_LOG_DIR}"
    LOG_TS="$(date +%Y%m%d_%H%M%S)"
    SCREEN_LOG_FILE="${SCREEN_LOG_DIR}/${SCREEN_SESSION}_${LOG_TS}.log"
    SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
    QUOTED_ARGS=""
    for arg in "$@"; do
        QUOTED_ARGS+=" $(printf '%q' "$arg")"
    done

    echo "Launching ${SCRIPT_PATH} in screen session '${SCREEN_SESSION}'..."
    screen -L -Logfile "${SCREEN_LOG_FILE}" -dmS "${SCREEN_SESSION}" bash -lc "AUTO_SCREEN=0 bash $(printf '%q' "$SCRIPT_PATH")${QUOTED_ARGS}"
    echo "Attach with: screen -r ${SCREEN_SESSION}"
    echo "Kill with:   screen -X -S ${SCREEN_SESSION} quit"
    echo "Log file:    ${SCREEN_LOG_FILE}"
    exit 0
fi

PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/self_play_online_vllm}"
TRAINING_SCRIPT="${TRAINING_SCRIPT:-${SCRIPT_DIR}/run_online_selfplay_training.sh}"
AUTORESTART_SCRIPT="${AUTORESTART_SCRIPT:-${SCRIPT_DIR}/run_online_selfplay_autorestart.sh}"
UPLOAD_FINAL_TO_HF="${UPLOAD_FINAL_TO_HF:-1}"
LATEST_ACTOR_PATH_FILE="${LATEST_ACTOR_PATH_FILE:-${PROJECT_ROOT}/${OUTPUT_ROOT}/latest_actor_path.txt}"
ROUND_SAVE_FREQ="${ROUND_SAVE_FREQ:-34}"
KEEP_ONLY_LATEST_CHECKPOINT="${KEEP_ONLY_LATEST_CHECKPOINT:-1}"
RESTART_AFTER_CHECKPOINT="${RESTART_AFTER_CHECKPOINT:-1}"
CHECKPOINT_RESTART_GRACE_SEC="${CHECKPOINT_RESTART_GRACE_SEC:-20}"
RESTART_ON_FAILURE="${RESTART_ON_FAILURE:-1}"
MAX_AUTORESTARTS="${MAX_AUTORESTARTS:-100}"

echo "=================================================="
echo "MedSeRL Online Self-Play + HF Upload"
echo "=================================================="
echo "Project root      : ${PROJECT_ROOT}"
echo "Output root       : ${OUTPUT_ROOT}"
echo "Training script   : ${TRAINING_SCRIPT}"
echo "Autorestart script: ${AUTORESTART_SCRIPT}"
echo "Upload final to HF: ${UPLOAD_FINAL_TO_HF}"
echo "Latest actor file : ${LATEST_ACTOR_PATH_FILE}"
echo "Round save freq   : ${ROUND_SAVE_FREQ}"
echo "Keep latest ckpt  : ${KEEP_ONLY_LATEST_CHECKPOINT}"
echo "Restart on ckpt   : ${RESTART_AFTER_CHECKPOINT}"
echo "Restart grace sec : ${CHECKPOINT_RESTART_GRACE_SEC}"
echo "Restart on failure: ${RESTART_ON_FAILURE}"
echo "Max autorestarts  : ${MAX_AUTORESTARTS}"
echo "HF repo id        : ${HF_REPO_ID:-<unset>}"
echo "=================================================="

if [ "${UPLOAD_FINAL_TO_HF}" = "1" ]; then
    if [ -z "${HF_REPO_ID:-}" ]; then
        echo "ERROR: HF_REPO_ID must be set when UPLOAD_FINAL_TO_HF=1."
        exit 1
    fi
    if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
        echo "ERROR: set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before uploading."
        exit 1
    fi
fi

AUTO_SCREEN=0 \
RUNNER_SCRIPT="${TRAINING_SCRIPT}" \
OUTPUT_ROOT="${OUTPUT_ROOT}" \
ROUND_SAVE_FREQ="${ROUND_SAVE_FREQ}" \
KEEP_ONLY_LATEST_CHECKPOINT="${KEEP_ONLY_LATEST_CHECKPOINT}" \
RESTART_AFTER_CHECKPOINT="${RESTART_AFTER_CHECKPOINT}" \
CHECKPOINT_RESTART_GRACE_SEC="${CHECKPOINT_RESTART_GRACE_SEC}" \
RESTART_ON_FAILURE="${RESTART_ON_FAILURE}" \
MAX_AUTORESTARTS="${MAX_AUTORESTARTS}" \
UPLOAD_FINAL_TO_HF="${UPLOAD_FINAL_TO_HF}" \
bash "${AUTORESTART_SCRIPT}" "$@"
