#!/bin/bash

set -euo pipefail

SCREEN_SESSION="${SCREEN_SESSION:-medserl_online_selfplay_autorestart}"
AUTO_SCREEN="${AUTO_SCREEN:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCREEN_LOG_DIR="${SCREEN_LOG_DIR:-${REPO_ROOT}/logs/screen}"
RUN_LOG_DIR="${RUN_LOG_DIR:-${REPO_ROOT}/logs/online_autorestart}"
RUNNER_SCRIPT="${RUNNER_SCRIPT:-${SCRIPT_DIR}/run_online_selfplay_training.sh}"

if [ "$AUTO_SCREEN" = "1" ] && [ -z "${STY:-}" ]; then
    if ! command -v screen >/dev/null 2>&1; then
        echo "ERROR: 'screen' is required but not installed."
        exit 1
    fi

    if screen -list | grep -q "\.${SCREEN_SESSION}[[:space:]]"; then
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
RESTART_AFTER_CHECKPOINT="${RESTART_AFTER_CHECKPOINT:-1}"
CHECKPOINT_RESTART_GRACE_SEC="${CHECKPOINT_RESTART_GRACE_SEC:-20}"
RESTART_ON_FAILURE="${RESTART_ON_FAILURE:-1}"
MAX_AUTORESTARTS="${MAX_AUTORESTARTS:-100}"
UPLOAD_FINAL_TO_HF="${UPLOAD_FINAL_TO_HF:-1}"
WANDB="${WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-medserl-selfplay}"

mkdir -p "${RUN_LOG_DIR}"

echo "=================================================="
echo "MedSeRL Online Self-Play Auto-Restart"
echo "=================================================="
echo "Runner script          : ${RUNNER_SCRIPT}"
echo "Project root           : ${PROJECT_ROOT}"
echo "Output root            : ${OUTPUT_ROOT}"
echo "Restart after checkpoint: ${RESTART_AFTER_CHECKPOINT}"
echo "Restart grace (sec)    : ${CHECKPOINT_RESTART_GRACE_SEC}"
echo "Restart on failure     : ${RESTART_ON_FAILURE}"
echo "Max auto restarts      : ${MAX_AUTORESTARTS}"
echo "Upload final to HF     : ${UPLOAD_FINAL_TO_HF}"
echo "W&B                    : ${WANDB} (${WANDB_PROJECT})"
echo "=================================================="

restart_count=0

while true; do
    if [ "$restart_count" -gt "$MAX_AUTORESTARTS" ]; then
        echo "ERROR: exceeded MAX_AUTORESTARTS=${MAX_AUTORESTARTS}"
        exit 1
    fi

    run_ts="$(date +%Y%m%d_%H%M%S)"
    run_log="${RUN_LOG_DIR}/online_autorestart_${run_ts}.log"
    fifo_path="$(mktemp -u)"
    mkfifo "${fifo_path}"

    scheduled_restart_pid=""
    scheduled_restart_step=""
    child_pid=""

    cleanup_child() {
        if [ -n "${scheduled_restart_pid}" ]; then
            kill "${scheduled_restart_pid}" >/dev/null 2>&1 || true
            wait "${scheduled_restart_pid}" 2>/dev/null || true
            scheduled_restart_pid=""
        fi
        if [ -n "${child_pid}" ]; then
            kill -TERM -"${child_pid}" >/dev/null 2>&1 || true
            wait "${child_pid}" 2>/dev/null || true
            child_pid=""
        fi
        rm -f "${fifo_path}"
    }

    trap cleanup_child EXIT

    echo ""
    echo "=== Auto-restart attempt $((restart_count + 1)) ==="
    echo "Log: ${run_log}"

    setsid bash -lc "cd $(printf '%q' "${PROJECT_ROOT}") && AUTO_SCREEN=0 bash $(printf '%q' "${RUNNER_SCRIPT}")" >"${fifo_path}" 2>&1 &
    child_pid="$!"

    while IFS= read -r line; do
        printf '%s\n' "${line}"
        printf '%s\n' "${line}" >> "${run_log}"

        if [ "${RESTART_AFTER_CHECKPOINT}" = "1" ] && [ -z "${scheduled_restart_pid}" ]; then
            if [[ "${line}" =~ Saved\ hf_model\ to\ .*/global_step_([0-9]+)/actor/huggingface ]]; then
                scheduled_restart_step="${BASH_REMATCH[1]}"
                echo "Checkpoint ${scheduled_restart_step} fully saved. Scheduling restart in ${CHECKPOINT_RESTART_GRACE_SEC}s."
                printf '%s\n' "Checkpoint ${scheduled_restart_step} fully saved. Scheduling restart in ${CHECKPOINT_RESTART_GRACE_SEC}s." >> "${run_log}"
                (
                    sleep "${CHECKPOINT_RESTART_GRACE_SEC}"
                    if kill -0 "${child_pid}" >/dev/null 2>&1; then
                        kill -TERM -"${child_pid}" >/dev/null 2>&1 || true
                    fi
                ) &
                scheduled_restart_pid="$!"
            fi
        fi
    done < "${fifo_path}"

    child_status=0
    wait "${child_pid}" || child_status=$?

    if [ -n "${scheduled_restart_pid}" ]; then
        kill "${scheduled_restart_pid}" >/dev/null 2>&1 || true
        wait "${scheduled_restart_pid}" 2>/dev/null || true
        scheduled_restart_pid=""
    fi

    rm -f "${fifo_path}"
    trap - EXIT

    latest_actor_path_file="${PROJECT_ROOT}/${OUTPUT_ROOT}/latest_actor_path.txt"

    if [ "${child_status}" -eq 0 ]; then
        echo "Runner exited successfully."
        if [ "${UPLOAD_FINAL_TO_HF}" = "1" ]; then
            if [ -f "${latest_actor_path_file}" ]; then
                if [ -n "${HF_TOKEN:-}" ] || [ -n "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
                    ACTOR_DIR="$(cat "${latest_actor_path_file}")" \
                    bash "${SCRIPT_DIR}/push_selfplay_actor_to_hf.sh"
                else
                    echo "WARNING: UPLOAD_FINAL_TO_HF=1 but no HF token is set; skipping upload."
                fi
            else
                echo "WARNING: latest actor path file not found; skipping HF upload."
            fi
        fi
        exit 0
    fi

    if [ -n "${scheduled_restart_step}" ]; then
        restart_count=$((restart_count + 1))
        echo "Restarting after saved checkpoint at step ${scheduled_restart_step}."
        continue
    fi

    if [ "${RESTART_ON_FAILURE}" = "1" ]; then
        restart_count=$((restart_count + 1))
        echo "Runner exited with status ${child_status}. Restarting."
        continue
    fi

    echo "Runner exited with status ${child_status}. Not restarting."
    exit "${child_status}"
done
