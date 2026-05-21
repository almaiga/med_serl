#!/usr/bin/env bash
# Merge a LoRA adapter into the full base model and push it to the HF Hub.
#
# Usage:
#   bash scripts/medrect/merge_and_push.sh
#   bash scripts/medrect/merge_and_push.sh <adapter_dir> <merged_dir> <hf_repo>
#
# Defaults match the mixed SFT run. Requires you to be logged in: `hf auth login`.
# If the HF upload OOMs on a small box, prefix with HF_HUB_DISABLE_XET=1.
set -euo pipefail

ADAPTER_DIR="${1:-outputs/local_training/medrect_mixed_v2}"
MERGED_DIR="${2:-outputs/local_training/medrect_mixed_v2_merged}"
HF_REPO="${3:-Abdine/qwen3-4b-medrect-mixed-r2}"

if [ ! -f "${ADAPTER_DIR}/adapter_model.safetensors" ]; then
    echo "ERROR: no adapter_model.safetensors in ${ADAPTER_DIR}" >&2
    exit 1
fi

echo "=========================================================="
echo "  Merge + push"
echo "    adapter : ${ADAPTER_DIR}"
echo "    merged  : ${MERGED_DIR}"
echo "    HF repo : ${HF_REPO}"
echo "=========================================================="

echo "[1/2] Merging adapter into base model..."
python scripts/medrect/merge_medrect_lora.py \
    --adapter-dir "${ADAPTER_DIR}" \
    --output-dir  "${MERGED_DIR}"

echo "[2/2] Uploading merged model to https://huggingface.co/${HF_REPO}"
hf upload "${HF_REPO}" "${MERGED_DIR}"

echo "Done -> ${HF_REPO}"
