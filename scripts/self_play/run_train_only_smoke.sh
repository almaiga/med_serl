#!/bin/bash
# MedSeRL Training-Only Smoke Test (NO judge server)
#
# Purpose: Fast iteration on the veRL training loop without waiting 14+ min
#          for the judge server to start. Uses pure rule-based reward only.
#
# How it works:
#   - Sets UMLS_WEIGHT=0 so agentic_reward.py short-circuits to rule-based scoring
#   - Cleans stale Ray state to avoid GCS timeout
#   - Runs veRL PPO/REINFORCE++ for ~5 training steps
#   - NO judge server, NO UMLS lookups
#
# Usage:
#   bash scripts/self_play/run_train_only_smoke.sh
#
# Env overrides:
#   ACTOR_MODEL   — Actor model path (default: Qwen/Qwen3-4B)
#   SMOKE_STEPS   — Max training steps (default: 5)

# ─── Configuration ────────────────────────────────────────────────────────────
ACTOR_MODEL="${ACTOR_MODEL:-Qwen/Qwen3-4B}"
SMOKE_STEPS="${SMOKE_STEPS:-5}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="smoke_trainonly_${TIMESTAMP}"
OUTPUT_DIR="outputs/self_play/smoke_trainonly_${TIMESTAMP}"

# NOTE: Do NOT set VLLM_USE_V1=0 here — veRL's internal vLLM rollout requires V1.
# No judge server in this script, so no V0 override needed at all.

# Skip judge entirely — pure rule-based reward
export UMLS_WEIGHT=0

# All Ray temp under /workspace (persistent, large, won't break SSH)
RAY_TMPDIR_PATH="/workspace/ray_tmp"
mkdir -p "$RAY_TMPDIR_PATH"

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
TRAIN_PARQUET="$PROJECT_ROOT/data_processed/self_play/train.parquet"
VAL_PARQUET="$PROJECT_ROOT/data_processed/self_play/val.parquet"
SMOKE_LOG="$OUTPUT_DIR/smoke_test.log"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_ROOT/results/self_play"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# ─── Cleanup: kill stale GPU / Ray processes ──────────────────────────────────
echo "=== Cleanup ==="
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -f "vllm.worker" 2>/dev/null || true
ray stop --force 2>/dev/null || true
pkill -9 -f "ray::" 2>/dev/null || true
rm -rf "$RAY_TMPDIR_PATH"/* /dev/shm/ray /tmp/ray 2>/dev/null || true
sleep 2

# Kill any remaining GPU processes
if command -v nvidia-smi &>/dev/null; then
    GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u)
    if [ -n "$GPU_PIDS" ]; then
        echo "  Killing stale GPU processes: $GPU_PIDS"
        for pid in $GPU_PIDS; do
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 3
    fi
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
fi
echo "  Cleanup done."

# ─── Expand /dev/shm if too small (Docker default is 64MB) ────────────────────
# Ray's object store uses /dev/shm for shared-memory IPC.  If it's tiny the GCS
# server deadlocks during startup → "Deadline Exceeded".
if [ -d /dev/shm ]; then
    SHM_SIZE_KB=$(df /dev/shm 2>/dev/null | awk 'NR==2{print $2}')
    if [ -n "$SHM_SIZE_KB" ] && [ "$SHM_SIZE_KB" -lt 1048576 ]; then
        echo "  /dev/shm is only $((SHM_SIZE_KB/1024)) MB — expanding to 16 GB ..."
        mount -o remount,size=16G /dev/shm 2>/dev/null && echo "  /dev/shm expanded." \
            || echo "  WARNING: could not remount /dev/shm (not root?). Will cap object_store_memory instead."
    else
        echo "  /dev/shm is $((SHM_SIZE_KB/1024)) MB — OK."
    fi
fi

# ─── Trap: cleanup on exit ───────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Cleaning up GPU processes..."
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "vllm.worker" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    if command -v nvidia-smi &>/dev/null; then
        GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u)
        for pid in $GPU_PIDS; do
            kill -9 "$pid" 2>/dev/null || true
        done
    fi
}
trap cleanup EXIT

echo "=================================================="
echo "MedSeRL Training-Only Smoke Test (NO JUDGE)"
echo "=================================================="
echo "Project root : $PROJECT_ROOT"
echo "Actor model  : $ACTOR_MODEL"
echo "UMLS_WEIGHT  : $UMLS_WEIGHT (rule-based only)"
echo "Output dir   : $OUTPUT_DIR"
echo "=================================================="

# ─── Step 0: Ensure data exists ──────────────────────────────────────────────
echo ""
echo "=== Step 0: Checking Data ==="

if [ ! -f "$TRAIN_PARQUET" ]; then
    echo "train.parquet missing — running preprocess_medec.py with MAX_PAIRS=20 ..."
    python3 "$PROJECT_ROOT/scripts/self_play/preprocess_medec.py" \
        --input "$PROJECT_ROOT/data_processed/medec_paired/train_val_split/rl_train.jsonl" \
        --output "$TRAIN_PARQUET" \
        --injection-prompts "$PROJECT_ROOT/configs/prompts/error_injection_prompts_v4.json" \
        --max-pairs 20
else
    echo "train.parquet found: $TRAIN_PARQUET"
fi

if [ ! -f "$VAL_PARQUET" ]; then
    echo "val.parquet missing — copying train.parquet"
    cp "$TRAIN_PARQUET" "$VAL_PARQUET"
else
    echo "val.parquet found: $VAL_PARQUET"
fi

# ─── Step 1: Clean Ray state & launch training ───────────────────────────────
echo ""
echo "=== Step 1: veRL Training (rule-based reward, ~${SMOKE_STEPS} steps) ==="
echo ""

# Final Ray state cleanup right before launch
ray stop --force 2>/dev/null || true
rm -rf "$RAY_TMPDIR_PATH"/* /dev/shm/ray /tmp/ray 2>/dev/null || true
sleep 2

# ── Patch veRL Ray init for Docker ──
# This must handle THREE cases:
#   a) Fresh main_ppo.py (never patched) — apply full patch
#   b) Already patched but missing object_store_memory — add it
#   c) Fully patched — no-op
python3 << 'PATCH_RAY'
import pathlib, re
fpath = pathlib.Path("/workspace/verl/verl/trainer/main_ppo.py")
if not fpath.exists():
    print("SKIP: main_ppo.py not found")
else:
    code = fpath.read_text()
    fresh_target = "ray.init(**OmegaConf.to_container(ray_init_kwargs))"
    full_replacement = """_ray_kw = OmegaConf.to_container(ray_init_kwargs)
    # ── MedSeRL Docker fix ──
    import os as _os
    if _ray_kw.get('num_cpus') is None:
        _ray_kw['num_cpus'] = min(_os.cpu_count() or 4, 8)
    _ray_kw.setdefault('include_dashboard', False)
    _ray_kw.setdefault('_temp_dir', '/workspace/ray_tmp')
    _ray_kw.setdefault('_node_ip_address', '127.0.0.1')
    _ray_kw.setdefault('object_store_memory', 1_000_000_000)  # 1 GB — safe for small /dev/shm
    _ray_kw.setdefault('_plasma_directory', '/workspace/ray_tmp')  # bypass /dev/shm entirely
    print(f"ray init kwargs (patched): {_ray_kw}")
    ray.init(**_ray_kw)"""

    if "_ray_kw" not in code and fresh_target in code:
        # Case (a): never patched
        code = code.replace(fresh_target, full_replacement)
        fpath.write_text(code)
        print("PATCHED: main_ppo.py — full Docker-safe Ray defaults")
    elif "_ray_kw" in code and "object_store_memory" not in code:
        # Case (b): patched before but missing object_store_memory
        # Insert the two new lines right before the print(f"ray init kwargs
        insert_before = '    print(f"ray init kwargs (patched):'
        new_lines = (
            "    _ray_kw.setdefault('object_store_memory', 1_000_000_000)  # 1 GB — safe for small /dev/shm\n"
            "    _ray_kw.setdefault('_plasma_directory', '/workspace/ray_tmp')  # bypass /dev/shm entirely\n"
        )
        if insert_before in code:
            code = code.replace(insert_before, new_lines + insert_before)
            fpath.write_text(code)
            print("PATCHED: added object_store_memory + _plasma_directory to existing patch")
        else:
            print("WARNING: could not find insertion point — manual check needed")
    else:
        print("main_ppo.py already fully patched")
PATCH_RAY

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_DIR" \
    --config-name="ppo_agentic" \
    \
    algorithm.adv_estimator=reinforce_plus_plus \
    \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data.train_batch_size=8 \
    data.val_batch_size=8 \
    data.train_max_samples=20 \
    data.val_max_samples=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    \
    actor_rollout_ref.model.path="$ACTOR_MODEL" \
    "++actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.40 \
    actor_rollout_ref.rollout.max_model_len=2048 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.prompt_length=1024 \
    actor_rollout_ref.rollout.response_length=1024 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    \
    critic.enable=false \
    \
    reward_model.enable=False \
    \
    custom_reward_function.path="$PROJECT_ROOT/scripts/self_play/agentic_reward.py" \
    custom_reward_function.name=async_compute_score \
    \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.001 \
    \
    trainer.total_epochs=1 \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=medserl-smoke \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=99999 \
    trainer.val_before_train=False \
    ++ray_kwargs.ray_init.include_dashboard=False \
    ++ray_kwargs.ray_init.num_cpus=8 \
    "++ray_kwargs.ray_init._temp_dir=$RAY_TMPDIR_PATH" \
    ++ray_kwargs.ray_init.object_store_memory=1000000000 \
    2>&1 | tee "$SMOKE_LOG"

TRAIN_EXIT=${PIPESTATUS[0]}

# ─── Step 2: Verification ────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "=== Step 2: Verification ==="
echo "=================================================="

CS_CALLS=$(grep -c "compute_score called" "$SMOKE_LOG" 2>/dev/null || true); CS_CALLS=${CS_CALLS:-0}

echo ""
echo "compute_score invocations : $CS_CALLS"
echo "Training exit code        : $TRAIN_EXIT"
echo ""

if [ "$TRAIN_EXIT" -eq 0 ] && [ "$CS_CALLS" -gt 0 ]; then
    echo "SMOKE TEST PASSED: Training loop ran with rule-based reward."
elif [ "$TRAIN_EXIT" -eq 0 ]; then
    echo "SMOKE TEST PARTIAL: Training completed but no compute_score logs found."
    echo "This may be a logging level issue — check agentic_reward.py logger config."
else
    echo "SMOKE TEST FAILED: Training exited with code $TRAIN_EXIT."
    echo "Check the log for errors: $SMOKE_LOG"
fi

echo ""
echo "Full log: $SMOKE_LOG"
echo "Outputs : $OUTPUT_DIR"

exit $TRAIN_EXIT
