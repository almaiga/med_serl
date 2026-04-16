#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

ADAPTER_DIR="${ADAPTER_DIR:-outputs/local_training/qwen3-4b-medrect-mixed-sft}"
MERGED_DIR="${MERGED_DIR:-outputs/local_training/qwen3-4b-medrect-mixed}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-4B}"
HF_NAMESPACE="${HF_NAMESPACE:-Abdine}"
HF_MODEL_NAME="${HF_MODEL_NAME:-qwen3-4b-medrect-mixed}"
HF_REPO_ID="${HF_REPO_ID:-${HF_NAMESPACE}/${HF_MODEL_NAME}}"
HF_PRIVATE="${HF_PRIVATE:-0}"
SKIP_MERGE="${SKIP_MERGE:-0}"

if [[ "${SKIP_MERGE}" != "1" ]]; then
    if [[ ! -d "${ADAPTER_DIR}" ]]; then
        echo "ERROR: adapter directory not found: ${ADAPTER_DIR}"
        echo "Train first or set ADAPTER_DIR=/path/to/adapter"
        exit 1
    fi

    if [[ ! -f "${ADAPTER_DIR}/adapter_config.json" ]]; then
        echo "ERROR: ${ADAPTER_DIR} does not look like a LoRA adapter directory"
        exit 1
    fi

    echo "=================================================="
    echo "Merging MedRECT adapter into full model"
    echo "=================================================="
    echo "Adapter dir: ${ADAPTER_DIR}"
    echo "Merged dir : ${MERGED_DIR}"
    echo "Base model : ${BASE_MODEL}"
    echo "=================================================="

    python3 scripts/medrect/merge_medrect_lora.py \
        --adapter-dir "${ADAPTER_DIR}" \
        --output-dir "${MERGED_DIR}" \
        --base-model "${BASE_MODEL}"
fi

if [[ ! -d "${MERGED_DIR}" ]]; then
    echo "ERROR: merged model directory not found: ${MERGED_DIR}"
    echo "Set MERGED_DIR=/path/to/full/model or run without SKIP_MERGE=1"
    exit 1
fi

if [[ ! -f "${MERGED_DIR}/config.json" ]]; then
    echo "ERROR: ${MERGED_DIR} does not look like a merged Hugging Face model"
    exit 1
fi

VISIBILITY="public"
if [[ "${HF_PRIVATE}" == "1" ]]; then
    VISIBILITY="private"
fi

echo "=================================================="
echo "Push MedRECT Full Model To Hugging Face"
echo "=================================================="
echo "Merged dir : ${MERGED_DIR}"
echo "Repo id    : ${HF_REPO_ID}"
echo "Visibility : ${VISIBILITY}"
echo "Base model : ${BASE_MODEL}"
echo "=================================================="

python3 - "${HF_REPO_ID}" "${MERGED_DIR}" "${VISIBILITY}" "${BASE_MODEL}" <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

repo_id = sys.argv[1]
folder_path = Path(sys.argv[2])
visibility = sys.argv[3]
base_model = sys.argv[4]

token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
private = visibility == "private"

create_repo(
    repo_id=repo_id,
    repo_type="model",
    exist_ok=True,
    private=private,
    token=token,
)

readme_path = folder_path / "README.md"
if not readme_path.exists():
    readme_path.write_text(
        f"""---
library_name: transformers
base_model: {base_model}
tags:
- qwen3
- medrect
- medical
- sft
- merged-lora
---

# {repo_id}

Merged full model for MedRECT mixed SFT.

- Base model: `{base_model}`
- Source directory: `{folder_path}`
- Artifact type: merged full model
""",
        encoding="utf-8",
    )

api = HfApi(token=token)
api.upload_folder(
    repo_id=repo_id,
    repo_type="model",
    folder_path=str(folder_path),
)
print(f"Uploaded {folder_path} to {repo_id}")
PY

echo "Done: https://huggingface.co/${HF_REPO_ID}"
