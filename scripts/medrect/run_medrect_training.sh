#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# MEDRECT Detection-Only LoRA SFT — RunPod Launcher
# =============================================================================
# Stage 0 of MedSeRL pipeline:
#   1. Prepare MEDRECT data → data_processed/medrect/medrect_sft_train.jsonl
#   2. Train Qwen3-8B LoRA on detection-only task
#
# Usage:
#   bash scripts/medrect/run_medrect_training.sh
#   bash scripts/medrect/run_medrect_training.sh --skip-prep     # skip data preparation
#   MODEL_PATH=/local/Qwen3-8B bash scripts/medrect/run_medrect_training.sh
#
# Environment overrides:
#   MODEL_PATH    — HF model name or local path (default: Qwen/Qwen3-8B)
#   MEDRECT_DATA  — path to medrect-en-train.json (default: data/medrect/medrect-en-train.json)
#   LORA_R        — LoRA rank (default: 64)
#   LORA_ALPHA    — LoRA alpha (default: 128)
#   LR            — learning rate (default: 1e-4)
#   EPOCHS        — training epochs (default: 3)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

# ---- Configuration (overridable via env) ----
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
MEDRECT_DATA="${MEDRECT_DATA:-/workspace/medrect/data/medrect/medrect-en-train.json}"
PROMPT_CONFIG="configs/prompts/sft/medrect_detection_prompts.json"
PREPARED_DATA="data_processed/medrect/medrect_sft_train.jsonl"

LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
LORA_DROPOUT="0.05"
LORA_TARGET="all-linear"
LR="${LR:-1e-4}"
EPOCHS="${EPOCHS:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-16}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
MODEL_SHORT=$(basename "${MODEL_PATH}")
OUTPUT_DIR="outputs/local_training/qwen3-8b-medrect-sft"
LOG_DIR="outputs/local_training/logs"
LOG_FILE="${LOG_DIR}/medrect_sft_${TIMESTAMP}.log"
SCREEN_NAME="medrect_sft"

SKIP_PREP=false
USE_WANDB=false

# Parse flags
for arg in "$@"; do
    case $arg in
        --skip-prep) SKIP_PREP=true ;;
        --wandb) USE_WANDB=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

# ---- Pre-flight checks ----

# Ensure screen is installed
if ! command -v screen &> /dev/null; then
    echo "Installing screen..."
    if command -v apt-get &> /dev/null; then
        apt-get update -qq && apt-get install -y screen
    elif command -v brew &> /dev/null; then
        brew install screen
    else
        echo "Cannot install screen. Please install manually."
        exit 1
    fi
fi

# Check for existing screen session
if screen -list | grep -q "${SCREEN_NAME}"; then
    echo "Screen session '${SCREEN_NAME}' already exists!"
    echo "  Attach: screen -r ${SCREEN_NAME}"
    echo "  Kill:   screen -X -S ${SCREEN_NAME} quit"
    exit 1
fi

# Check data files exist (MEDRECT_DATA may be space-separated)
if [[ "${SKIP_PREP}" == false ]]; then
    for _f in ${MEDRECT_DATA}; do
        if [[ ! -f "${_f}" ]]; then
            echo "MEDRECT data not found: ${_f}"
            echo "Set MEDRECT_DATA=/path/to/medrect-en-train.json"
            exit 1
        fi
    done
fi

# Check prompt config exists
if [[ ! -f "${PROMPT_CONFIG}" ]]; then
    echo "Prompt config not found: ${PROMPT_CONFIG}"
    exit 1
fi

# Create directories
mkdir -p "${LOG_DIR}" "$(dirname "${PREPARED_DATA}")" "${OUTPUT_DIR}"

# GPU info
echo ""
echo "============================================================"
echo "  MEDRECT Detection-Only LoRA SFT"
echo "============================================================"
echo "Model:        ${MODEL_PATH}"
echo "Data:         ${MEDRECT_DATA}"
echo "Prompt cfg:   ${PROMPT_CONFIG}"
echo "Output:       ${OUTPUT_DIR}"
echo "Log:          ${LOG_FILE}"
echo ""
echo "Hyperparameters:"
echo "  LoRA r=${LORA_R}, alpha=${LORA_ALPHA}, dropout=${LORA_DROPOUT}"
echo "  LR=${LR}, epochs=${EPOCHS}"
echo "  Batch=${BATCH_SIZE}, grad_accum=${GRAD_ACCUM}, effective=$((BATCH_SIZE * GRAD_ACCUM))"
echo "  Target modules: ${LORA_TARGET}"
echo "  Wandb: ${USE_WANDB}"
echo "  Skip prep: ${SKIP_PREP}"
echo "============================================================"

if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "GPU Info:"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# ---- Build pipeline function ----

run_pipeline() {
    echo "=================================================="
    echo "MEDRECT SFT Pipeline — Started $(date)"
    echo "=================================================="
    
    cd "${PROJECT_ROOT}"
    
    # Step 1: Prepare data
    if [[ "${SKIP_PREP}" == false ]]; then
        echo ""
        echo "=== Step 1: Preparing MEDRECT data ==="
        
        # Build input args (support multiple files via space-separated MEDRECT_DATA)
        INPUT_ARGS=""
        for f in ${MEDRECT_DATA}; do
            INPUT_ARGS="${INPUT_ARGS} ${f}"
        done
        
        python3 scripts/medrect/prepare_medrect_sft.py \
            --input ${INPUT_ARGS} \
            --output "${PREPARED_DATA}" \
            --prompt-config "${PROMPT_CONFIG}" \
            --show-samples 3
        
        echo "Data prepared: ${PREPARED_DATA}"
        echo "  Records: $(wc -l < "${PREPARED_DATA}")"
    else
        echo ""
        echo "=== Step 1: SKIPPED (--skip-prep) ==="
        if [[ ! -f "${PREPARED_DATA}" ]]; then
            echo "ERROR: Prepared data not found: ${PREPARED_DATA}"
            echo "Run without --skip-prep first."
            exit 1
        fi
        echo "Using existing: ${PREPARED_DATA} ($(wc -l < "${PREPARED_DATA}") records)"
    fi
    
    # Step 2: Train
    echo ""
    echo "=== Step 2: Training LoRA adapter ==="
    
    WANDB_FLAG=""
    if [[ "${USE_WANDB}" == true ]]; then
        WANDB_FLAG="--wandb --wandb-project medrect-sft"
    fi
    
    python3 scripts/medrect/train_medrect_lora.py \
        --train-file "${PREPARED_DATA}" \
        --model-name "${MODEL_PATH}" \
        --output-dir "${OUTPUT_DIR}" \
        --learning-rate "${LR}" \
        --num-train-epochs "${EPOCHS}" \
        --per-device-train-batch-size "${BATCH_SIZE}" \
        --gradient-accumulation-steps "${GRAD_ACCUM}" \
        --lora-r "${LORA_R}" \
        --lora-alpha "${LORA_ALPHA}" \
        --lora-dropout "${LORA_DROPOUT}" \
        --lora-target-modules "${LORA_TARGET}" \
        --eval-split 0 \
        --no-early-stopping \
        --bf16 \
        --debug-samples 2 \
        ${WANDB_FLAG}
    
    echo ""
    echo "=================================================="
    echo "Training Complete! $(date)"
    echo "=================================================="
    echo "Adapter saved: ${OUTPUT_DIR}"
    echo ""
    echo "Next steps:"
    echo "  1. Test: python -c \"from peft import AutoPeftModelForCausalLM; m = AutoPeftModelForCausalLM.from_pretrained('${OUTPUT_DIR}')\""
    echo "  2. Stage 1 (MedPRM): bash scripts/sft/launch_medprm_training.sh 8b"
    echo "     Set: --model-name ${OUTPUT_DIR}"
    echo "  3. Stage 2 (Self-play): bash scripts/self_play/run_multiturn_training.sh"
    echo "     Set: MODEL_PATH=${OUTPUT_DIR}"
}

# ---- Launch in screen ----

# Activate venv / conda
VENV_ACTIVATE=""
if [[ -f ".venv/bin/activate" ]]; then
    VENV_ACTIVATE="source .venv/bin/activate && "
elif [[ -f "venv/bin/activate" ]]; then
    VENV_ACTIVATE="source venv/bin/activate && "
fi

CONDA_ACTIVATE=""
CONDA_BASE="${CONDA_BASE:-/workspace/miniconda3}"
if [[ -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
    CONDA_ACTIVATE="source ${CONDA_BASE}/etc/profile.d/conda.sh && conda activate verl && "
fi

# Write pipeline to temp script for screen
PIPELINE_SCRIPT=$(mktemp /tmp/medrect_sft_XXXXXX.sh)
cat > "${PIPELINE_SCRIPT}" << HEREDOC
#!/bin/bash
set -e
cd ${PROJECT_ROOT}

# Restore environment
${CONDA_ACTIVATE}${VENV_ACTIVATE}true

export PATH="/usr/local/cuda/bin:\$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:\${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${PROJECT_ROOT}:\${PYTHONPATH:-}"

# Pass through all config as env vars
export MODEL_PATH="${MODEL_PATH}"
export MEDRECT_DATA="${MEDRECT_DATA}"
export PROMPT_CONFIG="${PROMPT_CONFIG}"
export PREPARED_DATA="${PREPARED_DATA}"
export LORA_R="${LORA_R}"
export LORA_ALPHA="${LORA_ALPHA}"
export LORA_DROPOUT="${LORA_DROPOUT}"
export LORA_TARGET="${LORA_TARGET}"
export LR="${LR}"
export EPOCHS="${EPOCHS}"
export BATCH_SIZE="${BATCH_SIZE}"
export GRAD_ACCUM="${GRAD_ACCUM}"
export OUTPUT_DIR="${OUTPUT_DIR}"
export SKIP_PREP="${SKIP_PREP}"
export USE_WANDB="${USE_WANDB}"
export PROJECT_ROOT="${PROJECT_ROOT}"

$(declare -f run_pipeline)
run_pipeline 2>&1 | tee "${LOG_FILE}"
HEREDOC

echo ""
echo "Launching in screen session '${SCREEN_NAME}'..."
screen -dmS "${SCREEN_NAME}" bash "${PIPELINE_SCRIPT}"

sleep 2

if screen -list | grep -q "${SCREEN_NAME}"; then
    echo ""
    echo "Training launched successfully!"
    echo ""
    echo "  Attach:   screen -r ${SCREEN_NAME}"
    echo "  Detach:   Ctrl+A then D"
    echo "  Log:      tail -f ${LOG_FILE}"
    echo "  Kill:     screen -X -S ${SCREEN_NAME} quit"
    echo ""
else
    echo "Failed to start screen session."
    exit 1
fi
