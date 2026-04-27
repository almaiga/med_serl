#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Launch detection inference for base model + fine-tuned model
#
# Usage:
#   bash scripts/detect_loc/launch_all.sh
#   MAX_SAMPLES=50 bash scripts/detect_loc/launch_all.sh   # quick test
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_detection.sh"

# Zero-shot inference defaults. All of these can still be overridden via env.
BASE_MODEL="${BASE_MODEL:-Qwen3-4B}"
FT_MODEL="${FT_MODEL:-medrect-sft}"
DATASET="${DATASET:-all}"
NO_THINKING="${NO_THINKING:-0}"
if [[ "${NO_THINKING}" == "1" ]]; then
    TEMPERATURE="${TEMPERATURE:-0.7}"
    TOP_P="${TOP_P:-0.8}"
else
    TEMPERATURE="${TEMPERATURE:-0.6}"
    TOP_P="${TOP_P:-0.95}"
fi
TOP_K="${TOP_K:-20}"
MIN_P="${MIN_P:-0}"
PRESENCE_PENALTY="${PRESENCE_PENALTY:-0}"
THINKING_BUDGET="${THINKING_BUDGET:-32768}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-16384}"

echo "============================================================"
echo "  Launching detection inference"
echo "  Base : ${BASE_MODEL}"
echo "  FT   : ${FT_MODEL}"
echo "  Data : ${DATASET}"
echo "  Temp : ${TEMPERATURE}"
echo "  TopP : ${TOP_P}"
echo "  TopK : ${TOP_K}"
echo "  Mode : $([ "${NO_THINKING}" == "1" ] && echo "zero-shot / no thinking" || echo "thinking enabled")"
echo "============================================================"
echo ""

run_and_wait() {
    local name="$1"
    local session="detect_${name}"
    DATASET="${DATASET}" TEMPERATURE="${TEMPERATURE}" TOP_P="${TOP_P}" TOP_K="${TOP_K}" MIN_P="${MIN_P}" PRESENCE_PENALTY="${PRESENCE_PENALTY}" THINKING_BUDGET="${THINKING_BUDGET}" MAX_NEW_TOKENS="${MAX_NEW_TOKENS}" NO_THINKING="${NO_THINKING}" bash "${RUNNER}" "${name}"
    echo "Waiting for ${session} to finish..."
    while screen -list 2>/dev/null | grep -q "${session}"; do sleep 15; done
    echo "${name} done."
    echo ""
}

run_and_wait "${BASE_MODEL}"
# run_and_wait "${FT_MODEL}"

echo "============================================================"
echo "  All done."
echo "============================================================"
