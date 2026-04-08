#!/bin/bash
# MedSeRL online batched self-play loop using the vLLM chained datagen path.
#
# This wraps run_multiturn_training.sh in an outer loop:
#   round N model -> regenerate chained parquet -> short VERL train ->
#   save actor checkpoint -> use that checkpoint for round N+1 datagen.

set -e

SCREEN_SESSION="${SCREEN_SESSION:-medserl_online_selfplay}"
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
ONLINE_ROUNDS="${ONLINE_ROUNDS:-5}"
TRAIN_EPOCHS_PER_ROUND="${TRAIN_EPOCHS_PER_ROUND:-1}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/self_play_online_vllm}"
EXPERIMENT_NAME_BASE="${EXPERIMENT_NAME_BASE:-medserl_selfplay_online_vllm}"
INITIAL_MODEL_PATH="${INITIAL_MODEL_PATH:-${ACTOR_MODEL:-Qwen/Qwen3-4B}}"
REQUIRE_JUDGE="${REQUIRE_JUDGE:-1}"
KEEP_ONLY_LATEST_CHECKPOINT="${KEEP_ONLY_LATEST_CHECKPOINT:-1}"
ROUND_SAVE_FREQ="${ROUND_SAVE_FREQ:-60}"
RESUME_INCOMPLETE_ROUND="${RESUME_INCOMPLETE_ROUND:-1}"

find_latest_actor_checkpoint() {
    local round_dir="$1"
    find "$round_dir" -maxdepth 1 -type d -name "global_step_*" | sort -V | tail -1
}

resolve_actor_model_path() {
    local step_dir="$1"
    if [ -f "$step_dir/actor/huggingface/config.json" ]; then
        echo "$step_dir/actor/huggingface"
    elif [ -f "$step_dir/actor/config.json" ]; then
        echo "$step_dir/actor"
    else
        echo ""
    fi
}

if [ "$REQUIRE_JUDGE" = "1" ] && [ -z "${JUDGE_VLLM_URL:-}" ]; then
    echo "ERROR: JUDGE_VLLM_URL is not set."
    echo "Start the judge pod first and export JUDGE_VLLM_URL on this training pod."
    exit 1
fi

mkdir -p "$PROJECT_ROOT/$OUTPUT_ROOT"

echo "=================================================="
echo "MedSeRL Online Batched Self-Play (vLLM + VERL)"
echo "=================================================="
echo "Project root: $PROJECT_ROOT"
echo "Output root : $OUTPUT_ROOT"
echo "Rounds      : $ONLINE_ROUNDS"
echo "Train/round : $TRAIN_EPOCHS_PER_ROUND"
echo "Initial mdl : $INITIAL_MODEL_PATH"
echo "Save freq   : $ROUND_SAVE_FREQ"
echo "Judge req   : $REQUIRE_JUDGE"
echo "Reward mode : remote judge + rule reward"
echo "Resume inc. : $RESUME_INCOMPLETE_ROUND"
echo "=================================================="

CURRENT_MODEL_PATH="$INITIAL_MODEL_PATH"

for ROUND in $(seq 1 "$ONLINE_ROUNDS"); do
    ROUND_DIR="$PROJECT_ROOT/$OUTPUT_ROOT/round_${ROUND}"
    ROUND_NAME="${EXPERIMENT_NAME_BASE}_round_${ROUND}"
    ROUND_DATA_DIR="$ROUND_DIR/data"
    ROUND_COMPLETE_MARKER="$ROUND_DIR/round_complete.txt"

    echo ""
    echo "=================================================="
    echo "Round $ROUND / $ONLINE_ROUNDS"
    echo "Round output  : $ROUND_DIR"
    echo "=================================================="

    mkdir -p "$ROUND_DIR"

    if [ -f "$ROUND_COMPLETE_MARKER" ]; then
        CURRENT_MODEL_PATH="$(cat "$ROUND_COMPLETE_MARKER")"
        echo "Round $ROUND already complete."
        echo "Using saved actor checkpoint: $CURRENT_MODEL_PATH"
        continue
    fi

    ROUND_SKIP_DATAGEN=0
    ROUND_RESUME_MODE="disable"
    if [ "$RESUME_INCOMPLETE_ROUND" = "1" ] && [ -d "$ROUND_DIR" ]; then
        LATEST_STEP_DIR="$(find_latest_actor_checkpoint "$ROUND_DIR")"
        if [ -n "$LATEST_STEP_DIR" ] && [ -f "$ROUND_DATA_DIR/train_chained.parquet" ]; then
            ROUND_SKIP_DATAGEN=1
            ROUND_RESUME_MODE="auto"
            echo "Resuming incomplete round $ROUND from $LATEST_STEP_DIR"
        fi
    fi

    echo "Injector model: $CURRENT_MODEL_PATH"
    echo "Skip datagen : $ROUND_SKIP_DATAGEN"
    echo "Resume mode  : $ROUND_RESUME_MODE"

    AUTO_SCREEN=0 \
    ACTOR_MODEL="$CURRENT_MODEL_PATH" \
    OUTPUT_DIR="$ROUND_DIR" \
    DATA_DIR="$ROUND_DATA_DIR" \
    EXPERIMENT_NAME="$ROUND_NAME" \
    TOTAL_EPOCHS="$TRAIN_EPOCHS_PER_ROUND" \
    SAVE_FREQ="$ROUND_SAVE_FREQ" \
    REQUIRE_JUDGE="$REQUIRE_JUDGE" \
    RESUME_MODE="$ROUND_RESUME_MODE" \
    SKIP_DATAGEN="$ROUND_SKIP_DATAGEN" \
    bash "$PROJECT_ROOT/scripts/self_play/run_multiturn_training.sh"

    LATEST_STEP_DIR="$(find_latest_actor_checkpoint "$ROUND_DIR")"
    if [ -z "$LATEST_STEP_DIR" ] || [ ! -d "$LATEST_STEP_DIR/actor" ]; then
        echo "ERROR: No actor checkpoint found after round $ROUND."
        echo "Expected a directory like: $ROUND_DIR/global_step_*/actor"
        exit 1
    fi

    if [ "$KEEP_ONLY_LATEST_CHECKPOINT" = "1" ]; then
        while IFS= read -r step_dir; do
            [ "$step_dir" = "$LATEST_STEP_DIR" ] && continue
            rm -rf "$step_dir"
        done < <(find "$ROUND_DIR" -maxdepth 1 -type d -name "global_step_*" | sort -V)
    fi

    CURRENT_MODEL_PATH="$(resolve_actor_model_path "$LATEST_STEP_DIR")"
    if [ -z "$CURRENT_MODEL_PATH" ]; then
        echo "ERROR: No loadable Hugging Face actor export found after round $ROUND."
        echo "Expected config.json under $LATEST_STEP_DIR/actor/huggingface or $LATEST_STEP_DIR/actor"
        exit 1
    fi
    printf '%s\n' "$CURRENT_MODEL_PATH" > "$ROUND_COMPLETE_MARKER"
    printf '%s\n' "$CURRENT_MODEL_PATH" > "$PROJECT_ROOT/$OUTPUT_ROOT/latest_actor_path.txt"
    echo "Next round actor checkpoint: $CURRENT_MODEL_PATH"
done

echo ""
echo "=================================================="
echo "Online self-play complete"
echo "=================================================="
echo "Final actor: $CURRENT_MODEL_PATH"
echo "Saved at   : $PROJECT_ROOT/$OUTPUT_ROOT/latest_actor_path.txt"
