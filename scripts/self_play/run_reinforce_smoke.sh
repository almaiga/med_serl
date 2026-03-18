#!/bin/bash
# MedSeRL REINFORCE++ Smoke Test (MAGIC-inspired separated training)
#
# Single-turn REINFORCE++, no sglang multi-turn.
# Mixed batches: injector + assessor examples in the same training pass.
# compute_score dispatches rewards by extra_info["role"].
#
# Why REINFORCE++ over GRPO for small experiments:
#   - n=1 rollout per prompt (no group sampling needed) → simpler, cheaper
#   - Stable advantage estimates via running baseline, not group comparison
#   - Less sensitive to batch size (GRPO needs many groups; REINFORCE++ does not)
#
# Usage:
#   bash scripts/self_play/run_reinforce_smoke.sh
#
# Env overrides:
#   ACTOR_MODEL   — Model path (default: Qwen/Qwen3-4B)
#   SMOKE_STEPS   — Max training steps (default: 5)
#   ROLES         — Data roles: injector / assessor / mixed (default: mixed)

# ─── Configuration ────────────────────────────────────────────────────────────
ACTOR_MODEL="${ACTOR_MODEL:-Qwen/Qwen3-4B}"
SMOKE_STEPS="${SMOKE_STEPS:-5}"
ROLES="${ROLES:-mixed}"
TRAIN_BATCH_SIZE=16   # 1 GPU
TRAIN_SAMPLES=$(( SMOKE_STEPS * TRAIN_BATCH_SIZE ))

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="reinforce_smoke_${TIMESTAMP}"
OUTPUT_DIR="outputs/self_play/reinforce_smoke_${TIMESTAMP}"

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
TRAIN_PARQUET="$PROJECT_ROOT/data_processed/self_play/train_grpo.parquet"
VAL_PARQUET="$PROJECT_ROOT/data_processed/self_play/val_grpo.parquet"
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
echo "MedSeRL REINFORCE++ Smoke Test (separated, no multi-turn)"
echo "=================================================="
echo "Project root : $PROJECT_ROOT"
echo "Actor model  : $ACTOR_MODEL"
echo "Roles        : $ROLES"
echo "Output dir   : $OUTPUT_DIR"
echo "=================================================="

# ─── Step 0: Ensure data exists ──────────────────────────────────────────────
echo ""
echo "=== Step 0: Checking Data ==="

if [ ! -f "$TRAIN_PARQUET" ]; then
    echo "train_grpo.parquet missing — generating with --roles=$ROLES, MAX_PAIRS=20 ..."
    python3 "$PROJECT_ROOT/scripts/self_play/preprocess_medec.py" \
        --input "$PROJECT_ROOT/data_processed/medec_paired/train_val_split/rl_train.jsonl" \
        --output "$TRAIN_PARQUET" \
        --injection-prompts "$PROJECT_ROOT/configs/prompts/error_injection_prompts_v4.json" \
        --detection-prompts "$PROJECT_ROOT/configs/prompts/detection_localization_prompts.json" \
        --roles "$ROLES" \
        --max-pairs 20
else
    echo "train_grpo.parquet found: $TRAIN_PARQUET"
fi

if [ ! -f "$VAL_PARQUET" ]; then
    echo "val_grpo.parquet missing — copying train_grpo.parquet"
    cp "$TRAIN_PARQUET" "$VAL_PARQUET"
else
    echo "val_grpo.parquet found: $VAL_PARQUET"
fi

# ─── Step 1: Clean Ray state & launch training ───────────────────────────────
echo ""
echo "=== Step 1: veRL REINFORCE++ Training (~${SMOKE_STEPS} steps) ==="
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
    algorithm.adv_estimator=reinforce_plus_plus \
    \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=8 \
    data.train_max_samples=$TRAIN_SAMPLES \
    data.val_max_samples=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=6144 \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path="$ACTOR_MODEL" \
    "++actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
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
    actor_rollout_ref.rollout.n=1 \
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
    \
    trainer.total_epochs=1 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=medserl-reinforce \
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

# ─── Step 2: Verification ────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "=== Step 2: Verification ==="
echo "=================================================="

SCORE_VALS=$(grep -o "critic/score/mean:[0-9.-]*" "$SMOKE_LOG" 2>/dev/null | cut -d: -f2 | grep -v '^0\.0*$' | wc -l || true)
SCORE_VALS=${SCORE_VALS:-0}
LAST_SCORE=$(grep -o "critic/score/mean:[0-9.-]*" "$SMOKE_LOG" 2>/dev/null | tail -1 | cut -d: -f2 || true)

echo ""
echo "Last critic/score/mean : ${LAST_SCORE:-(not found)}"
echo "Training exit code     : $TRAIN_EXIT"
echo ""

if [ "$TRAIN_EXIT" -eq 0 ] && [ "$SCORE_VALS" -gt 0 ]; then
    echo "SMOKE TEST PASSED: REINFORCE++ training completed with non-zero rewards."
elif [ "$TRAIN_EXIT" -eq 0 ]; then
    echo "SMOKE TEST PARTIAL: Training completed — check critic/score/mean in log."
else
    echo "SMOKE TEST FAILED: Training exited with code $TRAIN_EXIT."
    echo "Check: $SMOKE_LOG"
fi

echo ""
echo "Full log      : $SMOKE_LOG"
echo "Interactions  : $GAME_LOG"
echo "Outputs       : $OUTPUT_DIR"

# ─── Step 3: Response Quality Analysis ───────────────────────────────────────
echo ""
echo "=================================================="
echo "=== Step 3: Response Quality Analysis ==="
echo "=================================================="

python3 "$PROJECT_ROOT/scripts/self_play/analyze_smoke_quality.py" \
    --project-root        "$PROJECT_ROOT" \
    --output-dir          "$OUTPUT_DIR" \
    --smoke-log           "$SMOKE_LOG" \
    --max-response-length 6144

exit $TRAIN_EXIT
