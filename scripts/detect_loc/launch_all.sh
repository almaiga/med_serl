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

# Models to run (must exist in the MODELS registry in run_detection.sh)
BASE_MODEL="Qwen3-4B"
FT_MODEL="medrect-sft"

echo "============================================================"
echo "  Launching detection inference"
echo "  Base : ${BASE_MODEL}"
echo "  FT   : ${FT_MODEL}"
echo "============================================================"
echo ""

run_and_wait() {
    local name="$1"
    local session="detect_${name}"
    bash "${RUNNER}" "${name}"
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
