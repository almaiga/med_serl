#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# DeepSeek-R1 Injector Reasoning Chain Synthesis
# =============================================================================
# Generates injector-perspective reasoning chains using DeepSeek-R1-0528.
# Output is medrect JSONL format, compatible with train_medrect_lora.py.
#
# Usage:
#   bash scripts/medrect/run_injector_synthesis.sh                   # all scenarios
#   bash scripts/medrect/run_injector_synthesis.sh injector_error    # error only
#   bash scripts/medrect/run_injector_synthesis.sh injector_benign   # benign only
#   bash scripts/medrect/run_injector_synthesis.sh all 5             # test with limit=5
#
# Environment:
#   DEEPSEEK_API_KEY  — required (or set in .env)
#   CONCURRENCY       — max parallel API calls (default: 3)
#   CHECKPOINT_EVERY  — save checkpoint interval (default: 25)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# ---- Arguments ----
SCENARIO="${1:-all}"
LIMIT="${2:-}"
RESUME="${3:-}"

# ---- Config (overridable via env) ----
CONCURRENCY="${CONCURRENCY:-50}"
CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-25}"
MAX_RETRIES="${MAX_RETRIES:-3}"

ERROR_DATA="${ERROR_DATA:-data_processed/medec_paired/train_val_split/sft_train.jsonl}"
BENIGN_DATA="${BENIGN_DATA:-data_processed/benign_changes/benign_train_clean.jsonl}"
PROMPT_CONFIG="configs/prompts/sft/injector_reasoning_prompts.json"
OUTPUT_DIR="data_processed/medrect"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="outputs/local_training/logs"
LOG_FILE="${LOG_DIR}/injector_synthesis_${TIMESTAMP}.log"

# ---- Setup ----
mkdir -p "${LOG_DIR}" "${OUTPUT_DIR}"

echo "============================================================"
echo "  DeepSeek-R1 Injector Chain Synthesis"
echo "============================================================"
echo "Scenario:     ${SCENARIO}"
echo "Concurrency:  ${CONCURRENCY}"
echo "Limit:        ${LIMIT:-none}"
echo "Resume:       ${RESUME:-no}"
echo "Error data:   ${ERROR_DATA}"
echo "Benign data:  ${BENIGN_DATA}"
echo "Output dir:   ${OUTPUT_DIR}"
echo "Log file:     ${LOG_FILE}"
echo "============================================================"

# ---- Verify data exists ----
if [[ "${SCENARIO}" == "all" || "${SCENARIO}" == "injector_error" ]]; then
    if [[ ! -f "${ERROR_DATA}" ]]; then
        echo "ERROR: Error data not found: ${ERROR_DATA}"
        exit 1
    fi
    echo "Error data:   $(wc -l < "${ERROR_DATA}") records"
fi

if [[ "${SCENARIO}" == "all" || "${SCENARIO}" == "injector_benign" ]]; then
    if [[ ! -f "${BENIGN_DATA}" ]]; then
        echo "ERROR: Benign data not found: ${BENIGN_DATA}"
        exit 1
    fi
    echo "Benign data:  $(wc -l < "${BENIGN_DATA}") records"
fi

if [[ ! -f "${PROMPT_CONFIG}" ]]; then
    echo "ERROR: Prompt config not found: ${PROMPT_CONFIG}"
    exit 1
fi

# ---- Build command ----
CMD="python scripts/medrect/generate_injector_chains.py"
CMD+=" --scenario ${SCENARIO}"
CMD+=" --error-data ${ERROR_DATA}"
CMD+=" --benign-data ${BENIGN_DATA}"
CMD+=" --prompt-config ${PROMPT_CONFIG}"
CMD+=" --output-dir ${OUTPUT_DIR}"
CMD+=" --concurrency ${CONCURRENCY}"
CMD+=" --max-retries ${MAX_RETRIES}"
CMD+=" --checkpoint-every ${CHECKPOINT_EVERY}"

if [[ -n "${LIMIT}" ]]; then
    CMD+=" --limit ${LIMIT}"
fi

if [[ -n "${RESUME}" ]]; then
    CMD+=" --resume"
fi

echo ""
echo "Command: ${CMD}"
echo ""

# ---- Run ----
${CMD} 2>&1 | tee "${LOG_FILE}"

echo ""
echo "============================================================"
echo "  Done. Log: ${LOG_FILE}"
echo "  Output:    ${OUTPUT_DIR}/injector_*_chains_*.jsonl"
echo "============================================================"
