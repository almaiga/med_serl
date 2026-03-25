#!/bin/bash
# MedSeRL Online Self-Play: "Chasing a Moving Target"
#
# Implements the MAGIC/selfplay-redteaming feedback loop:
#
#   Epoch 1: train on static MEDEC data (cold start)
#   Epoch 2: use Epoch-1 model to inject errors → train assessor on those
#   Epoch 3: use Epoch-2 model to inject errors → train assessor on those
#   ...
#
# Each epoch the injector improves. Each new training batch forces the
# assessor to detect the CURRENT model's best injections, not static MEDEC
# errors. This is the "moving target" dynamic.
#
# Usage:
#   export JUDGE_VLLM_URL=http://<pod-id>.runpod.internal:8002/v1/chat/completions
#   bash scripts/self_play/run_online_selfplay.sh
#
# Env overrides (all of run_agentic_train_small.sh's vars, plus):
#   ONLINE_EPOCHS     — total outer epochs of online data regeneration (default: 5)
#   TRAIN_EPOCHS_PER  — verl training epochs per data generation round (default: 1)
#   DATAGEN_GPU_MEM   — vLLM GPU memory util for online_datagen.py (default: 0.5)
#   DATAGEN_TP        — tensor parallel size for online_datagen.py (default: 1)

set -e

ONLINE_EPOCHS="${ONLINE_EPOCHS:-5}"
TRAIN_EPOCHS_PER="${TRAIN_EPOCHS_PER:-1}"
DATAGEN_GPU_MEM="${DATAGEN_GPU_MEM:-0.5}"
DATAGEN_TP="${DATAGEN_TP:-1}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MEDEC_TRAIN="$PROJECT_ROOT/data_processed/medec_paired/train_val_split/rl_train.jsonl"
TRAIN_PARQUET="$PROJECT_ROOT/data_processed/self_play/train.parquet"
INJECTION_PROMPTS="$PROJECT_ROOT/configs/prompts/error_injection_prompts_v4.json"
DETECTION_PROMPTS="$PROJECT_ROOT/configs/prompts/detection_localization_prompts.json"
MAX_PAIRS="${MAX_PAIRS:-100}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RUN_DIR="outputs/online_selfplay_${TIMESTAMP}"
mkdir -p "$RUN_DIR"

echo "======================================================"
echo "MedSeRL Online Self-Play — Chasing a Moving Target"
echo "======================================================"
echo "Outer epochs    : $ONLINE_EPOCHS"
echo "verl epochs/rnd : $TRAIN_EPOCHS_PER"
echo "Max pairs       : $MAX_PAIRS"
echo "Run dir         : $RUN_DIR"
echo "======================================================"

# ── Epoch 0: cold-start with static MEDEC data ────────────────────────────
echo ""
echo "=== Epoch 0 / $ONLINE_EPOCHS — Cold start (static MEDEC data) ==="

ACTOR_CKPT="${ACTOR_MODEL:-Qwen/Qwen3-4B}"  # initial model

EPOCHS=$TRAIN_EPOCHS_PER \
SKIP_DATAGEN=0 \
OUTPUT_BASE="$RUN_DIR/epoch_0" \
bash "$PROJECT_ROOT/scripts/self_play/run_agentic_train_small.sh"

# Find the actor checkpoint saved by verl
LATEST_CKPT=$(find "$RUN_DIR/epoch_0" -type d -name "global_step_*" | sort -V | tail -1)
if [ -z "$LATEST_CKPT" ]; then
    echo "WARNING: No checkpoint found after epoch 0. Using base model for next round."
    ACTOR_CKPT="${ACTOR_MODEL:-Qwen/Qwen3-4B}"
else
    ACTOR_CKPT="$LATEST_CKPT/actor"
    echo "  Checkpoint for online datagen: $ACTOR_CKPT"
fi

# ── Epochs 1..N: online data regeneration ─────────────────────────────────
for epoch in $(seq 1 "$ONLINE_EPOCHS"); do
    echo ""
    echo "=== Epoch $epoch / $ONLINE_EPOCHS — Online data regeneration ==="
    echo "  Injector model: $ACTOR_CKPT"

    # Regenerate train.parquet using current actor as injector
    python3 "$PROJECT_ROOT/scripts/self_play/online_datagen.py" \
        --model-path "$ACTOR_CKPT" \
        --medec-jsonl "$MEDEC_TRAIN" \
        --output "$TRAIN_PARQUET" \
        --injection-prompts "$INJECTION_PROMPTS" \
        --detection-prompts "$DETECTION_PROMPTS" \
        --max-pairs "$MAX_PAIRS" \
        --gpu-memory-utilization "$DATAGEN_GPU_MEM" \
        --tensor-parallel-size "$DATAGEN_TP"

    echo "  train.parquet regenerated from live model injections."
    echo "  Starting training on fresh data …"

    # Train for TRAIN_EPOCHS_PER epochs on the new data
    EPOCHS=$TRAIN_EPOCHS_PER \
    SKIP_DATAGEN=1 \
    OUTPUT_BASE="$RUN_DIR/epoch_${epoch}" \
    bash "$PROJECT_ROOT/scripts/self_play/run_agentic_train_small.sh"

    # Update checkpoint path for next round
    LATEST_CKPT=$(find "$RUN_DIR/epoch_${epoch}" -type d -name "global_step_*" | sort -V | tail -1)
    if [ -n "$LATEST_CKPT" ] && [ -d "$LATEST_CKPT/actor" ]; then
        ACTOR_CKPT="$LATEST_CKPT/actor"
        echo "  Updated actor checkpoint: $ACTOR_CKPT"
    else
        echo "  WARNING: No new checkpoint found — reusing previous: $ACTOR_CKPT"
    fi
done

echo ""
echo "======================================================"
echo "Online Self-Play Complete"
echo "======================================================"
echo "Run dir: $RUN_DIR"
echo "Final actor checkpoint: $ACTOR_CKPT"
