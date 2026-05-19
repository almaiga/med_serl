#!/usr/bin/env bash
# Assessor reasoning pipeline — adapted from pfnet-research/medrect
#
# Three steps (mirrors MEDRECT exactly):
#   1. generate_assessor_reasoning.py  — call DeepSeek-R1 with cheat prompt
#   2. clean_reasoning_en.py           — remove meta-reference sentences
#   3. extract_correct_samples.py      — keep only correct predictions
#
# Only difference from upstream: label = sentence_number (not sentence_number: corrected_sentence)
#
# Usage:
#   bash scripts/medrect/pipeline/run_pipeline.sh ms_train   40
#   bash scripts/medrect/pipeline/run_pipeline.sh ms_val     40
#   bash scripts/medrect/pipeline/run_pipeline.sh uw_val     40

set -euo pipefail

SPLIT="${1:-ms_train}"       # ms_train | ms_val | uw_val
CONCURRENCY="${2:-40}"

SFT_PATH="data_processed/medec_v2/${SPLIT}.jsonl"
OUTPUT_DIR="data_processed/medrect_v2/${SPLIT}"
PIPELINE_DIR="scripts/medrect/pipeline"

echo "========================================"
echo "  Assessor pipeline — split: ${SPLIT}"
echo "========================================"

# ── Step 1: Generate ──────────────────────────────────────────────────────────
echo ""
echo "[1/3] Generating reasoning with DeepSeek-R1..."
python "${PIPELINE_DIR}/generate_assessor_reasoning.py" \
    --sft-path   "${SFT_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --concurrency "${CONCURRENCY}"

# Find the raw file that was just written
RAW_FILE=$(ls -t "${OUTPUT_DIR}"/raw_responses_*.jsonl | head -1)
echo "      Raw file: ${RAW_FILE}"

# ── Step 2: Clean ─────────────────────────────────────────────────────────────
echo ""
echo "[2/3] Cleaning meta-references..."
python "${PIPELINE_DIR}/clean_reasoning_en.py" \
    --input "${RAW_FILE}" \
    --overwrite

# ── Step 3: Extract ───────────────────────────────────────────────────────────
echo ""
echo "[3/3] Extracting correct samples..."
python "${PIPELINE_DIR}/extract_correct_samples.py" \
    --input "${RAW_FILE}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "Done. Accepted chains: ${OUTPUT_DIR}/generated_assessor_all_accepted.jsonl"
echo ""
echo "Next: run prepare_medrect_sft.py to format for training."
