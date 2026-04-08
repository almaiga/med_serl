#!/bin/bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

ACTOR_DIR="${ACTOR_DIR:-${REPO_ROOT}/outputs/self_play_online_vllm/round_1/global_step_202/actor/huggingface}"
HF_NAMESPACE="${HF_NAMESPACE:-Abdine}"
HF_MODEL_NAME="${HF_MODEL_NAME:-medserl-qwen3-4b-medrect-mixed-selfplay-r1}"
HF_REPO_ID="${HF_REPO_ID:-${HF_NAMESPACE}/${HF_MODEL_NAME}}"
HF_PRIVATE="${HF_PRIVATE:-0}"
MODEL_BASE="${MODEL_BASE:-Abdine/qwen3-4b-medrect-mixed}"

if [ ! -d "${ACTOR_DIR}" ]; then
    echo "ERROR: actor directory not found: ${ACTOR_DIR}"
    exit 1
fi

if [ ! -f "${ACTOR_DIR}/config.json" ]; then
    echo "ERROR: ${ACTOR_DIR} does not look like a Hugging Face export (missing config.json)"
    exit 1
fi

if [ -z "${HF_TOKEN:-}" ] && [ -z "${HUGGING_FACE_HUB_TOKEN:-}" ]; then
    echo "ERROR: set HF_TOKEN or HUGGING_FACE_HUB_TOKEN before uploading."
    exit 1
fi

VISIBILITY="public"
if [ "${HF_PRIVATE}" = "1" ]; then
    VISIBILITY="private"
fi

echo "=================================================="
echo "Push MedSeRL Actor To Hugging Face"
echo "=================================================="
echo "Actor dir : ${ACTOR_DIR}"
echo "Repo id   : ${HF_REPO_ID}"
echo "Visibility: ${VISIBILITY}"
echo "Base model: ${MODEL_BASE}"
echo "=================================================="

python3 - "${HF_REPO_ID}" "${ACTOR_DIR}" "${VISIBILITY}" "${MODEL_BASE}" <<'PY'
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo

repo_id = sys.argv[1]
folder_path = Path(sys.argv[2])
visibility = sys.argv[3]
model_base = sys.argv[4]
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
base_model: {model_base}
tags:
- medserl
- self-play
- reinforcement-learning
- qwen3
---

# {repo_id}

Round-1 MedSeRL self-play actor exported from VERL training.

- Base model: `{model_base}`
- Training recipe: batched injector -> assessor self-play
- Export path: `{folder_path}`

This artifact is intended for evaluation and manual testing.
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
