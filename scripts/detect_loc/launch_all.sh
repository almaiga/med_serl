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
BASE_MODEL="Qwen3-8B"
FT_MODEL="medrect-sft"

echo "============================================================"
echo "  Launching detection inference"
echo "  Base : ${BASE_MODEL}"
echo "  FT   : ${FT_MODEL}"
echo "============================================================"
echo ""

bash "${RUNNER}" "${BASE_MODEL}"
echo ""
bash "${RUNNER}" "${FT_MODEL}"
echo ""

echo "============================================================"
echo "  Both sessions started."
echo "  Attach  : screen -r detect_${BASE_MODEL}"
echo "            screen -r detect_${FT_MODEL}"
echo "  List    : screen -ls"
echo "============================================================"
