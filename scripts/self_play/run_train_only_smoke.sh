#!/bin/bash
# MedSeRL GRPO Chained Smoke Test (MAGIC-style)
#
# Sequential injection → assessment, assessor sees injector's ACTUAL model output.
#
# Phase A (offline, once): run the base model as injector → capture outputs →
#   build assessor prompts from those real outputs → save combined parquet.
# Phase B: standard single-turn vllm GRPO training on the combined parquet,
#   exactly like run_grpo_smoke.sh — no sglang, no multi-turn complexity.
#
# Usage:
#   bash scripts/self_play/run_train_only_smoke.sh
#
# Env overrides:
#   ACTOR_MODEL   — Model path (default: Qwen/Qwen3-4B)
#   SMOKE_STEPS   — Max training steps (default: 5)
#   MAX_PAIRS     — Pairs used for chained data generation (default: 20)
#   SKIP_DATAGEN  — Set to 1 to reuse existing chained parquet

# ─── Configuration ────────────────────────────────────────────────────────────
ACTOR_MODEL="${ACTOR_MODEL:-Qwen/Qwen3-4B}"
SMOKE_STEPS="${SMOKE_STEPS:-5}"
MAX_PAIRS="${MAX_PAIRS:-20}"
TRAIN_BATCH_SIZE=8
TRAIN_SAMPLES=$(( SMOKE_STEPS * TRAIN_BATCH_SIZE ))

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="grpo_chained_${TIMESTAMP}"
OUTPUT_DIR="outputs/self_play/grpo_chained_${TIMESTAMP}"

# All Ray temp under /workspace (persistent, large, won't break SSH)
RAY_TMPDIR_PATH="/workspace/ray_tmp"
mkdir -p "$RAY_TMPDIR_PATH"

# ── GPU visibility ──
export CUDA_VISIBLE_DEVICES=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

# ── Ray env vars for RunPod Docker ──
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_DEDUP_LOGS=0
export RAY_USE_MULTIPROCESSING_CPU_COUNT=1
export RAY_memory_monitor_refresh_ms=0
export RAY_raylet_start_wait_time_s=300
export RAY_TMPDIR="$RAY_TMPDIR_PATH"
export RAY_GCS_SERVER_REQUEST_TIMEOUT_S=60
export HYDRA_FULL_ERROR=1

# Paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/scripts/self_play/configs"
TRAIN_PARQUET="$PROJECT_ROOT/data_processed/self_play/train_chained.parquet"
VAL_PARQUET="$PROJECT_ROOT/data_processed/self_play/val_chained.parquet"
SMOKE_LOG="$OUTPUT_DIR/smoke_test.log"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_ROOT/results/self_play"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# ─── Cleanup: kill stale GPU / Ray processes ──────────────────────────────────
echo "=== Cleanup ==="

pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -f "vllm.worker" 2>/dev/null || true
sleep 2

if command -v nvidia-smi &>/dev/null; then
    GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u)
    if [ -n "$GPU_PIDS" ]; then
        echo "  Sending SIGTERM to stale GPU processes: $GPU_PIDS"
        for pid in $GPU_PIDS; do
            kill "$pid" 2>/dev/null || true
        done
        sleep 5
        REMAINING=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u)
        if [ -n "$REMAINING" ]; then
            echo "  SIGKILL remaining: $REMAINING"
            for pid in $REMAINING; do
                kill -9 "$pid" 2>/dev/null || true
            done
            sleep 3
        fi
    fi
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
fi
echo "  Cleanup done."

# ── CUDA health check ──
echo "  Verifying CUDA is functional ..."
CUDA_CHECK=$(python3 -c "
import torch, sys
print(f'torch {torch.__version__}, CUDA compiled: {torch.version.cuda}')
if not torch.cuda.is_available():
    print('FAIL: torch.cuda.is_available() == False', file=sys.stderr)
    sys.exit(1)
try:
    torch.cuda.init()
    d = torch.cuda.current_device()
    print(f'CUDA OK — device {d}: {torch.cuda.get_device_name(d)}')
except Exception as e:
    print(f'FAIL: {e}', file=sys.stderr)
    sys.exit(1)
" 2>&1)
CUDA_RC=$?
echo "  $CUDA_CHECK"
if [ $CUDA_RC -ne 0 ]; then
    echo "  ERROR: CUDA runtime cannot initialise. Restart the pod and re-run."
    exit 1
fi

# ─── Expand /dev/shm if too small ─────────────────────────────────────────────
if [ -d /dev/shm ]; then
    SHM_SIZE_KB=$(df /dev/shm 2>/dev/null | awk 'NR==2{print $2}')
    if [ -n "$SHM_SIZE_KB" ] && [ "$SHM_SIZE_KB" -lt 1048576 ]; then
        echo "  /dev/shm is only $((SHM_SIZE_KB/1024)) MB — expanding to 16 GB ..."
        mount -o remount,size=16G /dev/shm 2>/dev/null && echo "  /dev/shm expanded." \
            || echo "  WARNING: could not remount /dev/shm (not root?)."
    else
        echo "  /dev/shm is $((SHM_SIZE_KB/1024)) MB — OK."
    fi
fi

# ─── Trap ─────────────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Script finished. To free GPU memory: nvidia-smi then kill <pid>"
}
trap cleanup EXIT

echo "=================================================="
echo "MedSeRL GRPO Chained Smoke Test (MAGIC-style)"
echo "  Phase A: offline injector inference → chained parquet"
echo "  Phase B: single-turn vllm GRPO training"
echo "=================================================="
echo "Project root : $PROJECT_ROOT"
echo "Actor model  : $ACTOR_MODEL"
echo "Max pairs    : $MAX_PAIRS"
echo "Output dir   : $OUTPUT_DIR"
echo "=================================================="

# ─── Phase A: Generate chained data ──────────────────────────────────────────
echo ""
echo "=== Phase A: Chained Data Generation ==="

if [ "${SKIP_DATAGEN:-0}" = "1" ] && [ -f "$TRAIN_PARQUET" ]; then
    echo "SKIP_DATAGEN=1 — reusing existing $TRAIN_PARQUET"
else
    echo "Running generate_chained_data.py (model=$ACTOR_MODEL, max_pairs=$MAX_PAIRS) ..."
    python3 "$PROJECT_ROOT/scripts/self_play/generate_chained_data.py" \
        --model              "$ACTOR_MODEL" \
        --input              "$PROJECT_ROOT/data_processed/medec_paired/train_val_split/rl_train.jsonl" \
        --output             "$TRAIN_PARQUET" \
        --injection-prompts  "$PROJECT_ROOT/configs/prompts/error_injection_prompts_v4.json" \
        --detection-prompts  "$PROJECT_ROOT/configs/prompts/detection_localization_prompts.json" \
        --max-pairs          "$MAX_PAIRS"

    if [ $? -ne 0 ]; then
        echo "ERROR: generate_chained_data.py failed. Aborting."
        exit 1
    fi
fi

if [ ! -f "$VAL_PARQUET" ]; then
    echo "Copying train_chained.parquet → val_chained.parquet"
    cp "$TRAIN_PARQUET" "$VAL_PARQUET"
else
    echo "val_chained.parquet found: $VAL_PARQUET"
fi

# ─── Phase B: GRPO Training ───────────────────────────────────────────────────
echo ""
echo "=== Phase B: veRL GRPO Training (~${SMOKE_STEPS} steps) ==="
echo ""

GAME_LOG="$PROJECT_ROOT/$OUTPUT_DIR/game_interactions.jsonl"
export MEDSERL_GAME_LOG="$GAME_LOG"

rm -rf "$RAY_TMPDIR_PATH"/* 2>/dev/null || true
sleep 2

python3 "$PROJECT_ROOT/scripts/self_play/patch_verl_ray.py"

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_DIR" \
    --config-name="grpo_separated" \
    \
    algorithm.adv_estimator=grpo \
    \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=8 \
    data.train_max_samples=$TRAIN_SAMPLES \
    data.val_max_samples=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path="$ACTOR_MODEL" \
    "++actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    \
    critic.enable=false \
    reward_model.enable=False \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    trainer.total_epochs=1 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=medserl-grpo-chained \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.val_before_train=False \
    ++ray_kwargs.ray_init.include_dashboard=False \
    ++ray_kwargs.ray_init.num_cpus=8 \
    "++ray_kwargs.ray_init._temp_dir=$RAY_TMPDIR_PATH" \
    ++ray_kwargs.ray_init.object_store_memory=1000000000 \
    "++ray_kwargs.runtime_env.env_vars.MEDSERL_GAME_LOG=$GAME_LOG" \
    custom_reward_function.path="$PROJECT_ROOT/scripts/self_play/reward_function.py" \
    custom_reward_function.name="compute_score" \
    2>&1 | tee "$SMOKE_LOG"

TRAIN_EXIT=${PIPESTATUS[0]}

# ─── Verification ─────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "=== Verification ==="
echo "=================================================="

REWARD_HITS=$(grep -c "compute_score called" "$SMOKE_LOG" 2>/dev/null || true)
REWARD_HITS=${REWARD_HITS:-0}

echo ""
echo "compute_score calls in log : $REWARD_HITS  (>0 confirms rewards reached)"
echo "Training exit code         : $TRAIN_EXIT"
echo ""

if [ "$TRAIN_EXIT" -eq 0 ] && [ "$REWARD_HITS" -gt 0 ]; then
    echo "SMOKE TEST PASSED: Chained GRPO training completed with rewards."
elif [ "$TRAIN_EXIT" -eq 0 ]; then
    echo "SMOKE TEST PARTIAL: Training completed but reward calls not found in log."
else
    echo "SMOKE TEST FAILED: Training exited with code $TRAIN_EXIT."
    echo "Check: $SMOKE_LOG"
fi

echo ""
echo "Full log : $SMOKE_LOG"
echo "Outputs  : $OUTPUT_DIR"

# ─── Response Quality Analysis ────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "=== Response Quality Analysis ==="
echo "=================================================="

python3 "$PROJECT_ROOT/scripts/self_play/analyze_smoke_quality.py" \
    --project-root        "$PROJECT_ROOT" \
    --output-dir          "$OUTPUT_DIR" \
    --smoke-log           "$SMOKE_LOG" \
    --max-response-length 3072

exit $TRAIN_EXIT
