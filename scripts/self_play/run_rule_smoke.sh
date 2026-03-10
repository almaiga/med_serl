#!/bin/bash
# MedSeRL Rule-Based Smoke Test — NO JUDGE SERVER
#
# Purpose: Validate veRL training pipeline end-to-end using only the
#          rule-based reward function (reward_function.py).  No vLLM
#          judge, no UMLS, no 14-min judge startup.
#
# Modeled after official veRL examples:
#   https://github.com/verl-project/verl/tree/main/examples/sglang_multiturn
#
# Key differences from run_agentic_train_smoke.sh:
#   - No judge server (skip Step 1 entirely)
#   - Points custom_reward_function at reward_function.py (pure rule-based)
#   - No LoRA (matches official veRL examples)
#   - gpu_memory_utilization=0.85 (no judge competing for GPU)
#   - Adds ulimit -n 65535 (matches official examples)
#
# Usage:
#   bash scripts/self_play/run_rule_smoke.sh
#
# Env overrides:
#   ACTOR_MODEL   — Actor model path (default: Qwen/Qwen3-4B)
#   SMOKE_STEPS   — Max training steps (default: 5)

set -x                    # echo every command (matches official veRL examples)
ulimit -n 65535           # prevent "too many open files" (matches official examples)

# ─── Configuration ────────────────────────────────────────────────────────────
ACTOR_MODEL="${ACTOR_MODEL:-Qwen/Qwen3-4B}"
SMOKE_STEPS="${SMOKE_STEPS:-5}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="smoke_rule_${TIMESTAMP}"
OUTPUT_DIR="outputs/self_play/smoke_rule_${TIMESTAMP}"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/scripts/self_play/configs"
TRAIN_PARQUET="$PROJECT_ROOT/data_processed/self_play/train.parquet"
VAL_PARQUET="$PROJECT_ROOT/data_processed/self_play/val.parquet"
SMOKE_LOG="$OUTPUT_DIR/smoke_test.log"

mkdir -p "$OUTPUT_DIR"

# ─── Environment ──────────────────────────────────────────────────────────────
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export CUDA_VISIBLE_DEVICES=0
export HYDRA_FULL_ERROR=1

# CUDA / torch startup acceleration
export TORCH_CUDA_ARCH_LIST="8.0"         # A100 = sm_80 — skip JIT for other archs
export CUDA_MODULE_LOADING=LAZY           # lazy-load CUDA modules

# Ray Docker fixes
export RAY_memory_monitor_refresh_ms=0
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_DEDUP_LOGS=0
export RAY_USE_MULTIPROCESSING_CPU_COUNT=1
export RAY_raylet_start_wait_time_s=300
export RAY_TMPDIR=/dev/shm/ray
export RAY_GCS_SERVER_REQUEST_TIMEOUT_S=60

# ─── Cleanup ──────────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Trap: cleaning up..."
    ray stop --force 2>/dev/null || true
    pkill -9 -f "gcs_server|raylet|plasma_store" 2>/dev/null || true
    if command -v nvidia-smi &>/dev/null; then
        nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null \
            | tr -d ' ' | sort -u | while read pid; do
                [ -n "$pid" ] && kill -9 "$pid" 2>/dev/null || true
            done
    fi
    echo "  Cleanup complete."
}
trap cleanup EXIT

# Kill stale GPU processes from previous runs
echo "=== Cleanup: killing stale processes ==="
ray stop --force 2>/dev/null || true
pkill -9 -f "ray::|vllm.entrypoints|vllm.worker|from multiprocessing" 2>/dev/null || true
if command -v nvidia-smi &>/dev/null; then
    GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u)
    if [ -n "$GPU_PIDS" ]; then
        echo "  Killing GPU processes: $GPU_PIDS"
        for pid in $GPU_PIDS; do
            kill -9 "$pid" 2>/dev/null || true
        done
    else
        echo "  No stale GPU processes found."
    fi
fi
sleep 2
echo "  Cleanup done."

# ─── Pre-flight: GPU check ───────────────────────────────────────────────────
echo ""
echo "=== GPU Memory ==="
nvidia-smi --query-gpu=memory.free,memory.total --format=csv,noheader 2>/dev/null

# ─── Pre-flight: Patch veRL for Docker ────────────────────────────────────────
echo ""
echo "=== Patching veRL ==="

# Patch 1: GLIBC workaround (widen except ImportError → except (ImportError, OSError))
python3 << 'PATCH1'
import pathlib, re
fpath = pathlib.Path("/workspace/verl/verl/workers/engine/__init__.py")
if fpath.exists():
    code = fpath.read_text()
    new_code, n = re.subn(r'except ImportError:', 'except (ImportError, OSError):', code)
    if n > 0:
        fpath.write_text(new_code)
        print(f"  GLIBC: patched {n} exceptions")
    else:
        print("  GLIBC: already patched")
else:
    print("  GLIBC: file not found, skipping")
PATCH1

# Patch 2: Ray init Docker fix (the critical one — injects num_cpus, dashboard=False, loopback IP)
python3 << 'PATCH2'
import pathlib
fpath = pathlib.Path("/workspace/verl/verl/trainer/main_ppo.py")
if fpath.exists():
    code = fpath.read_text()
    target = "ray.init(**OmegaConf.to_container(ray_init_kwargs))"
    if "_ray_kw" not in code and target in code:
        replacement = """_ray_kw = OmegaConf.to_container(ray_init_kwargs)
    # ── MedSeRL Docker fix: inject Ray init defaults ──
    import os as _os
    if _ray_kw.get('num_cpus') is None:
        _ray_kw['num_cpus'] = _os.cpu_count() or 4
    _ray_kw.setdefault('include_dashboard', False)
    _ray_kw.setdefault('_temp_dir', '/dev/shm/ray')
    _ray_kw.setdefault('_node_ip_address', '127.0.0.1')
    print(f"ray init kwargs (patched): {_ray_kw}")
    ray.init(**_ray_kw)"""
        code = code.replace(target, replacement)
        fpath.write_text(code)
        print("  Ray init: PATCHED")
    else:
        print("  Ray init: already patched")
else:
    print("  Ray init: file not found, skipping")
PATCH2

# ─── Data check ──────────────────────────────────────────────────────────────
echo ""
echo "=== Data Check ==="
if [ ! -f "$TRAIN_PARQUET" ]; then
    echo "ERROR: $TRAIN_PARQUET not found. Run preprocess first."
    exit 1
fi
echo "  train.parquet: $TRAIN_PARQUET"

if [ ! -f "$VAL_PARQUET" ]; then
    cp "$TRAIN_PARQUET" "$VAL_PARQUET"
    echo "  val.parquet: created from train.parquet"
else
    echo "  val.parquet: $VAL_PARQUET"
fi

# ─── Clean Ray state ─────────────────────────────────────────────────────────
echo ""
echo "=== Cleaning Ray state ==="
ray stop --force 2>/dev/null || true
pkill -9 -f "gcs_server|raylet|plasma_store" 2>/dev/null || true
rm -rf /tmp/ray /dev/shm/ray 2>/dev/null || true
fuser -k 6379/tcp 2>/dev/null || true
mkdir -p /dev/shm/ray
sleep 2
echo "  Done."

# ─── Training ────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "MedSeRL Rule-Based Smoke Test"
echo "=================================================="
echo "Actor model  : $ACTOR_MODEL"
echo "Reward       : reward_function.py (rule-based, no judge)"
echo "Output       : $OUTPUT_DIR"
echo "=================================================="
echo ""

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_DIR" \
    --config-name="ppo_agentic" \
    \
    algorithm.adv_estimator=reinforce_plus_plus \
    \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data.train_batch_size=8 \
    data.train_max_samples=20 \
    data.val_max_samples=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    \
    actor_rollout_ref.model.path="$ACTOR_MODEL" \
    "++actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
    actor_rollout_ref.rollout.max_num_batched_tokens=8192 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.prompt_length=1024 \
    actor_rollout_ref.rollout.response_length=1024 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    \
    critic.enable=false \
    reward_model.enable=False \
    \
    custom_reward_function.path="$PROJECT_ROOT/scripts/self_play/reward_function.py" \
    custom_reward_function.name=compute_score \
    \
    algorithm.use_kl_in_reward=False \
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
    2>&1 | tee "$SMOKE_LOG"

TRAIN_EXIT=${PIPESTATUS[0]}

# ─── Verification ─────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "=== Smoke Test Verification ==="
echo "=================================================="

CS_CALLS=$(grep -c "compute_score called" "$SMOKE_LOG" 2>/dev/null || echo 0)
REWARD_LINES=$(grep -c "reward=" "$SMOKE_LOG" 2>/dev/null || echo 0)

echo ""
echo "compute_score invocations : $CS_CALLS"
echo "reward log lines          : $REWARD_LINES"
echo ""

if [ "$CS_CALLS" -gt 0 ]; then
    echo "SMOKE TEST PASSED: compute_score called $CS_CALLS times with real model rollouts."
    echo ""
    echo "Sample calls:"
    grep "compute_score called" "$SMOKE_LOG" | head -5
else
    echo "SMOKE TEST FAILED: No compute_score calls found."
    echo "Training exit code: $TRAIN_EXIT"
fi

echo ""
echo "Full log : $SMOKE_LOG"
echo "Outputs  : $OUTPUT_DIR"

exit $TRAIN_EXIT
