#!/usr/bin/env bash
set -eo pipefail

# =============================================================================
# Scaling experiment — training launcher
# Trains 8 LoRA models (one per data shard) from Qwen3-4B.
# Runs 4 jobs in parallel (one per GPU), then the next 4.
#
# Usage:
#   bash scripts/scaling/run_scaling_train.sh
#
# Optional env overrides:
#   DATA_DIR     — directory with scale_*of8.jsonl  (default: data_processed/scaling)
#   OUTPUT_DIR   — where models are saved           (default: outputs/scaling)
#   N_SHARDS     — number of shards                 (default: 8)
#   N_GPUS       — GPUs available                   (default: 4)
#   MODEL_NAME   — base model                       (default: Qwen/Qwen3-4B)
#   EPOCHS       — training epochs                  (default: 3)
#   LR           — learning rate                    (default: 1e-4)
#   LORA_R       — LoRA rank                        (default: 64)
#   LORA_ALPHA   — LoRA alpha                       (default: 128)
#   ONLY         — run only this shard index (1-8)  (default: all)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

DATA_DIR="${DATA_DIR:-data_processed/scaling}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/scaling}"
N_SHARDS="${N_SHARDS:-8}"
N_GPUS="${N_GPUS:-4}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3-4B}"
EPOCHS="${EPOCHS:-3}"
LR="${LR:-1e-4}"
LORA_R="${LORA_R:-64}"
LORA_ALPHA="${LORA_ALPHA:-128}"
ONLY="${ONLY:-}"
TRAIN_SCRIPT="scripts/medrect/train_medrect_lora.py"

echo ""
echo "============================================================"
echo "  Scaling Experiment — Training (${N_SHARDS} shards, ${N_GPUS} GPUs)"
echo "============================================================"
echo "Data dir   : ${DATA_DIR}"
echo "Output dir : ${OUTPUT_DIR}"
echo "Base model : ${MODEL_NAME}"
echo "Epochs     : ${EPOCHS}  |  LR: ${LR}  |  LoRA r=${LORA_R} α=${LORA_ALPHA}"
[[ -n "${ONLY}" ]] && echo "Only shard : ${ONLY}"
echo "============================================================"
echo ""

# ── Validate data files exist ─────────────────────────────────────────────────
for k in $(seq 1 "${N_SHARDS}"); do
    [[ -n "${ONLY}" && "${k}" != "${ONLY}" ]] && continue
    f="${DATA_DIR}/scale_${k}of${N_SHARDS}.jsonl"
    [[ -f "${f}" ]] || { echo "ERROR: ${f} not found. Run split_training_data.py first."; exit 1; }
done

# ── Ensure screen is available ────────────────────────────────────────────────
if ! command -v screen &>/dev/null; then
    apt-get update -qq && apt-get install -y screen 2>/dev/null || \
        { echo "ERROR: screen not found"; exit 1; }
fi

mkdir -p "${OUTPUT_DIR}/logs"

# ── Launch function ───────────────────────────────────────────────────────────
launch_shard() {
    local k=$1 gpu=$2
    local train_file="${DATA_DIR}/scale_${k}of${N_SHARDS}.jsonl"
    local out_dir="${OUTPUT_DIR}/scale_${k}of${N_SHARDS}"
    local screen_name="scale_train_${k}of${N_SHARDS}"
    local log="${OUTPUT_DIR}/logs/train_${k}of${N_SHARDS}.log"

    if screen -list 2>/dev/null | grep -q "${screen_name}"; then
        echo "  [shard ${k}] already running — skipping"
        return
    fi

    local n_samples
    n_samples=$(wc -l < "${train_file}")

    local cmd="CUDA_VISIBLE_DEVICES=${gpu} python ${TRAIN_SCRIPT}"
    cmd+=" --train-file ${train_file}"
    cmd+=" --output-dir ${out_dir}"
    cmd+=" --model-name ${MODEL_NAME}"
    cmd+=" --num-train-epochs ${EPOCHS}"
    cmd+=" --learning-rate ${LR}"
    cmd+=" --lora-r ${LORA_R}"
    cmd+=" --lora-alpha ${LORA_ALPHA}"

    echo "  [shard ${k}/${N_SHARDS}] GPU=${gpu}  n=${n_samples}  → ${out_dir}"
    screen -dmS "${screen_name}" bash -c \
        "${cmd} 2>&1 | tee -a '${log}'; echo '=== DONE shard ${k} (exit \$?) ==='"
}

# ── Run in waves of N_GPUS ────────────────────────────────────────────────────
if [[ -n "${ONLY}" ]]; then
    gpu=0
    launch_shard "${ONLY}" "${gpu}"
    echo ""
    echo "Launched shard ${ONLY} on GPU 0."
    echo "Logs: tail -f ${OUTPUT_DIR}/logs/train_${ONLY}of${N_SHARDS}.log"
    exit 0
fi

wave=1
gpu=0
pids=()
echo "Wave 1 (shards 1-${N_GPUS}):"
for k in $(seq 1 "${N_GPUS}"); do
    [[ "${k}" -gt "${N_SHARDS}" ]] && break
    launch_shard "${k}" "${gpu}"
    gpu=$(( (gpu + 1) % N_GPUS ))
    pids+=("scale_train_${k}of${N_SHARDS}")
done

echo ""
echo "Waiting for wave 1 to finish before launching wave 2..."
echo "(Check: screen -ls | grep scale_train)"
echo ""

# Wait for wave 1 screen sessions to exit
for name in "${pids[@]}"; do
    while screen -list 2>/dev/null | grep -q "${name}"; do
        sleep 60
    done
    echo "  [done] ${name}"
done

if [[ "${N_SHARDS}" -gt "${N_GPUS}" ]]; then
    echo ""
    echo "Wave 2 (shards $(( N_GPUS + 1 ))-${N_SHARDS}):"
    gpu=0
    for k in $(seq $(( N_GPUS + 1 )) "${N_SHARDS}"); do
        launch_shard "${k}" "${gpu}"
        gpu=$(( (gpu + 1) % N_GPUS ))
    done
fi

echo ""
echo "All training jobs launched."
echo "Monitor: screen -ls | grep scale_train"
echo "Logs:    ls ${OUTPUT_DIR}/logs/"
