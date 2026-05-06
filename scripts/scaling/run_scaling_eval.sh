#!/usr/bin/env bash
set -eo pipefail

# =============================================================================
# Scaling experiment — evaluation launcher
# Evaluates 8 trained models on the full test set using vLLM.
# Runs 4 jobs in parallel (one per GPU), then the next 4.
#
# Usage:
#   bash scripts/scaling/run_scaling_eval.sh
#
# Optional env overrides:
#   MODELS_DIR   — directory with scale_*of8/ model folders (default: outputs/scaling)
#   OUTPUT_DIR   — results destination                      (default: results/scaling)
#   N_SHARDS     — number of shards                         (default: 8)
#   N_GPUS       — GPUs available                           (default: 4)
#   MODE         — thinking | no-thinking                   (default: thinking)
#   THINKING_BUDGET  — token budget for thinking            (default: 4096)
#   DATASET      — ms | uw | all                            (default: all)
#   ONLY         — evaluate only this shard index (1-8)     (default: all)
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${PROJECT_ROOT}"

MODELS_DIR="${MODELS_DIR:-outputs/scaling}"
OUTPUT_DIR="${OUTPUT_DIR:-results/scaling}"
N_SHARDS="${N_SHARDS:-8}"
N_GPUS="${N_GPUS:-4}"
MODE="${MODE:-thinking}"
THINKING_BUDGET="${THINKING_BUDGET:-4096}"
DATASET="${DATASET:-all}"
ONLY="${ONLY:-}"
EVAL_SCRIPT="scripts/medrect/inference_detection_vllm.py"
PROMPT_CONFIG="configs/prompts/detection_localization_prompts.json"

if [[ "${MODE}" == "thinking" ]]; then
    TEMPERATURE="${TEMPERATURE:-0.6}"
    TOP_P="${TOP_P:-0.95}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4352}"
else
    TEMPERATURE="${TEMPERATURE:-0.7}"
    TOP_P="${TOP_P:-0.8}"
    MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-4096}"
fi

echo ""
echo "============================================================"
echo "  Scaling Experiment — Evaluation (${N_SHARDS} shards, ${N_GPUS} GPUs)"
echo "============================================================"
echo "Models dir : ${MODELS_DIR}"
echo "Output dir : ${OUTPUT_DIR}"
echo "Dataset    : ${DATASET}  |  Mode: ${MODE}"
echo "Budget     : ${THINKING_BUDGET}  |  Max tokens: ${MAX_NEW_TOKENS}"
[[ -n "${ONLY}" ]] && echo "Only shard : ${ONLY}"
echo "============================================================"
echo ""

# ── Validate ──────────────────────────────────────────────────────────────────
[[ -f "${EVAL_SCRIPT}" ]]    || { echo "ERROR: ${EVAL_SCRIPT} not found";    exit 1; }
[[ -f "${PROMPT_CONFIG}" ]]  || { echo "ERROR: ${PROMPT_CONFIG} not found";  exit 1; }

for k in $(seq 1 "${N_SHARDS}"); do
    [[ -n "${ONLY}" && "${k}" != "${ONLY}" ]] && continue
    m="${MODELS_DIR}/scale_${k}of${N_SHARDS}"
    [[ -d "${m}" ]] || { echo "ERROR: model dir ${m} not found. Run run_scaling_train.sh first."; exit 1; }
done

if ! command -v screen &>/dev/null; then
    apt-get update -qq && apt-get install -y screen 2>/dev/null || \
        { echo "ERROR: screen not found"; exit 1; }
fi

mkdir -p "${OUTPUT_DIR}/logs"

# ── Launch function ───────────────────────────────────────────────────────────
launch_eval() {
    local k=$1 gpu=$2
    local model_path="${MODELS_DIR}/scale_${k}of${N_SHARDS}"
    local out_dir="${OUTPUT_DIR}/scale_${k}of${N_SHARDS}"
    local screen_name="scale_eval_${k}of${N_SHARDS}"
    local log="${OUTPUT_DIR}/logs/eval_${k}of${N_SHARDS}.log"

    mkdir -p "${out_dir}"

    if screen -list 2>/dev/null | grep -q "${screen_name}"; then
        echo "  [shard ${k}] already running — skipping"
        return
    fi

    local cmd="CUDA_VISIBLE_DEVICES=${gpu} python ${EVAL_SCRIPT}"
    cmd+=" --model_path ${model_path}"
    cmd+=" --prompt_config ${PROMPT_CONFIG}"
    cmd+=" --dataset ${DATASET}"
    cmd+=" --mode ${MODE}"
    cmd+=" --temperature ${TEMPERATURE}"
    cmd+=" --top_p ${TOP_P}"
    cmd+=" --thinking_budget ${THINKING_BUDGET}"
    cmd+=" --max_new_tokens ${MAX_NEW_TOKENS}"
    cmd+=" --output_dir ${out_dir}"

    echo "  [shard ${k}/${N_SHARDS}] GPU=${gpu}  → ${out_dir}"
    screen -dmS "${screen_name}" bash -c \
        "${cmd} 2>&1 | tee -a '${log}'; echo '=== DONE eval ${k} (exit \$?) ==='"
}

# ── Run in waves ──────────────────────────────────────────────────────────────
if [[ -n "${ONLY}" ]]; then
    launch_eval "${ONLY}" 0
    echo ""
    echo "Launched eval for shard ${ONLY} on GPU 0."
    echo "Logs: tail -f ${OUTPUT_DIR}/logs/eval_${ONLY}of${N_SHARDS}.log"
    exit 0
fi

echo "Wave 1 (shards 1-${N_GPUS}):"
gpu=0
wave1_names=()
for k in $(seq 1 "${N_GPUS}"); do
    [[ "${k}" -gt "${N_SHARDS}" ]] && break
    launch_eval "${k}" "${gpu}"
    wave1_names+=("scale_eval_${k}of${N_SHARDS}")
    gpu=$(( (gpu + 1) % N_GPUS ))
done

echo ""
echo "Waiting for wave 1 to finish..."
for name in "${wave1_names[@]}"; do
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
        launch_eval "${k}" "${gpu}"
        gpu=$(( (gpu + 1) % N_GPUS ))
    done
fi

echo ""
echo "All evaluation jobs launched."
echo "Monitor : screen -ls | grep scale_eval"
echo "Results : ls ${OUTPUT_DIR}/"
echo ""
echo "Once done, plot with:"
echo "  python scripts/scaling/plot_scaling.py --results-dir ${OUTPUT_DIR}"
