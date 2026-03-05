#!/usr/bin/env bash
# =============================================================================
# Qwen3-4B Baseline - Error Detection + Localization on MEDEC
# =============================================================================
# Tests Qwen3-4B's ability to detect and locate errors in medical notes
# without any fine-tuning (zero-shot with CoT reasoning)
#
# Usage:
#   bash scripts/run_baseline_qwen3_4b_detection.sh [dataset]
#   bash scripts/run_baseline_qwen3_4b_detection.sh ms
#   bash scripts/run_baseline_qwen3_4b_detection.sh uw
#   bash scripts/run_baseline_qwen3_4b_detection.sh all  # default
#
# Optional env overrides:
#   TEMPERATURE=0.3      # sampling temperature (default: 0.7)
#   MAX_NEW_TOKENS=512   # max tokens to generate (default: 512)
#   THINKING_BUDGET=1024 # thinking budget for Qwen (default: 1024)
#   BATCH_SIZE=4         # batch size (default: 1)
#   MAX_SAMPLES=50       # limit samples for quick test
#   NO_COT=1             # disable chain-of-thought reasoning
# =============================================================================

set -euo pipefail

# ── Configuration ───────────────────────────────────────────────────────────
DATASET="${1:-all}"
MODEL_PATH="Qwen/Qwen3-4B"
MODEL_NAME="qwen3-4b-baseline"
SESSION_NAME="baseline_qwen3_4b_detect"

# Inference parameters
TEMPERATURE="${TEMPERATURE:-0.7}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-1536}"
THINKING_BUDGET="${THINKING_BUDGET:-1024}"
BATCH_SIZE="${BATCH_SIZE:-8}"
MAX_SAMPLES="${MAX_SAMPLES:-}"
NO_COT="${NO_COT:-0}"

# Paths
PROMPT_CONFIG="configs/prompts/detection_localization_prompts.json"
OUTPUT_DIR="results/inference/${MODEL_NAME}"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${OUTPUT_DIR}/logs"
LOG_FILE="${LOG_DIR}/${DATASET}_${TIMESTAMP}.log"

# ── Kill existing session if running ────────────────────────────────────────
screen -X -S "${SESSION_NAME}" quit 2>/dev/null || true

# ── Create log directory ────────────────────────────────────────────────────
mkdir -p "${LOG_DIR}"

# ── Build Python command ────────────────────────────────────────────────────
PYTHON_CMD="python scripts/inference_error_detection.py"
PYTHON_CMD+=" --model_path ${MODEL_PATH}"
PYTHON_CMD+=" --model_name ${MODEL_NAME}"
PYTHON_CMD+=" --model_type qwen"
PYTHON_CMD+=" --dataset ${DATASET}"
PYTHON_CMD+=" --prompt_file ${PROMPT_CONFIG}"
PYTHON_CMD+=" --temperature ${TEMPERATURE}"
PYTHON_CMD+=" --max_new_tokens ${MAX_NEW_TOKENS}"
PYTHON_CMD+=" --thinking_budget ${THINKING_BUDGET}"
PYTHON_CMD+=" --batch_size ${BATCH_SIZE}"
PYTHON_CMD+=" --output_dir ${OUTPUT_DIR}"
[[ -n "${MAX_SAMPLES}" ]] && PYTHON_CMD+=" --max_samples ${MAX_SAMPLES}"
[[ "${NO_COT}" == "1" ]] && PYTHON_CMD+=" --no_cot"

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "🔬 Qwen3-4B Baseline: Error Detection + Localization"
echo "============================================================"
echo "Model:          ${MODEL_PATH}"
echo "Model Name:     ${MODEL_NAME}"
echo "Dataset:        ${DATASET}"
echo "CoT:            $([ "${NO_COT}" == "1" ] && echo "disabled" || echo "enabled")"
echo "Temperature:    ${TEMPERATURE}"
echo "Max Tokens:     ${MAX_NEW_TOKENS}"
echo "Thinking Budget:${THINKING_BUDGET}"
echo "Batch Size:     ${BATCH_SIZE}"
echo "Max Samples:    ${MAX_SAMPLES:-all}"
echo "Output:         ${OUTPUT_DIR}"
echo "Log:            ${LOG_FILE}"
echo "Screen Session: ${SESSION_NAME}"
echo "============================================================"
echo ""

# ── Launch in screen ────────────────────────────────────────────────────────
screen -dmS "${SESSION_NAME}" bash -c "
set -euo pipefail

# Activate environment (adjust path as needed)
if [ -f /workspace/miniconda3/bin/activate ]; then
    source /workspace/miniconda3/bin/activate
    conda activate med_serl
elif [ -f ~/miniconda3/bin/activate ]; then
    source ~/miniconda3/bin/activate
    conda activate med_serl
fi

# Run inference and log output
${PYTHON_CMD} 2>&1 | tee '${LOG_FILE}'

echo ''
echo '============================================================'
echo '✅ Baseline inference completed!'
echo '============================================================'
echo 'Results saved to: ${OUTPUT_DIR}'
echo 'Logs saved to: ${LOG_FILE}'
echo '============================================================'

exec bash
"

echo ""
echo "✅ Started in screen session: ${SESSION_NAME}"
echo ""
echo "To monitor:"
echo "  screen -r ${SESSION_NAME}"
echo "  tail -f ${LOG_FILE}"
echo ""
echo "To detach from screen: Ctrl+A, then D"
echo "To kill session: screen -X -S ${SESSION_NAME} quit"
echo ""
