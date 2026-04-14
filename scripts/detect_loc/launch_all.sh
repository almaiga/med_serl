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
TEMPERATURE="${TEMPERATURE:-0}"
NO_THINKING="${NO_THINKING:-1}"

echo "============================================================"
echo "  Launching detection inference"
echo "  Base : ${BASE_MODEL}"
echo "  FT   : ${FT_MODEL}"
echo "  Temp : ${TEMPERATURE}"
echo "  Mode : $([ "${NO_THINKING}" == "1" ] && echo "zero-shot / no thinking" || echo "thinking enabled")"
echo "============================================================"
echo ""

run_and_wait() {
    local name="$1"
    local session="detect_${name}"
    TEMPERATURE="${TEMPERATURE}" NO_THINKING="${NO_THINKING}" bash "${RUNNER}" "${name}"
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
