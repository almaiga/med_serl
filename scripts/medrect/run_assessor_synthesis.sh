#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

SFT_PATH="${SFT_PATH:-data_processed/medec_paired/train_val_split/sft_train.jsonl}"
SCOPE="${SCOPE:-all}"
OUTPUT_DIR="${OUTPUT_DIR:-data_processed/medrect}"
PROMPT_CONFIG="${PROMPT_CONFIG:-configs/prompts/sft/medrect_assessor_reasoning_prompts.json}"
DETECTION_PROMPTS="${DETECTION_PROMPTS:-configs/prompts/sft/medrect_detection_prompts.json}"
MATCH_SUMMARY="${MATCH_SUMMARY:-data_processed/medrect/recovered_medrect_match_summary.json}"
CONCURRENCY="${CONCURRENCY:-40}"
MAX_RETRIES="${MAX_RETRIES:-3}"
SHOW_SAMPLES="${SHOW_SAMPLES:-3}"
LIMIT="${LIMIT:-}"
RESUME="${RESUME:-1}"

ACCEPTED_OUTPUT="${OUTPUT_DIR}/generated_assessor_${SCOPE}_accepted.jsonl"
SFT_OUTPUT="${OUTPUT_DIR}/generated_assessor_${SCOPE}_sft.jsonl"

GEN_CMD=(
    python3 scripts/medrect/generate_missing_assessor_chains.py
    --sft-path "${SFT_PATH}"
    --scope "${SCOPE}"
    --output-dir "${OUTPUT_DIR}"
    --prompt-config "${PROMPT_CONFIG}"
    --concurrency "${CONCURRENCY}"
    --max-retries "${MAX_RETRIES}"
)

if [[ "${SCOPE}" != "all" ]]; then
    GEN_CMD+=(--match-summary "${MATCH_SUMMARY}")
fi

if [[ -n "${LIMIT}" ]]; then
    GEN_CMD+=(--limit "${LIMIT}")
fi

if [[ "${RESUME}" == "1" ]]; then
    GEN_CMD+=(--resume)
fi

echo "============================================================"
echo "  Assessor Chain Synthesis"
echo "============================================================"
echo "SFT pairs:        ${SFT_PATH}"
echo "Scope:            ${SCOPE}"
echo "Output dir:       ${OUTPUT_DIR}"
echo "Accepted output:  ${ACCEPTED_OUTPUT}"
echo "SFT output:       ${SFT_OUTPUT}"
echo "Concurrency:      ${CONCURRENCY}"
echo "Max retries:      ${MAX_RETRIES}"
echo "Resume:           ${RESUME}"
if [[ -n "${LIMIT}" ]]; then
    echo "Limit:            ${LIMIT}"
fi
echo "============================================================"
echo ""

"${GEN_CMD[@]}"

python3 scripts/medrect/prepare_medrect_sft.py \
    --input "${ACCEPTED_OUTPUT}" \
    --output "${SFT_OUTPUT}" \
    --prompt-config "${DETECTION_PROMPTS}" \
    --language en \
    --show-samples "${SHOW_SAMPLES}"

echo ""
echo "Finished:"
echo "  Accepted raw chains: ${ACCEPTED_OUTPUT}"
echo "  Prepared SFT file:   ${SFT_OUTPUT}"
