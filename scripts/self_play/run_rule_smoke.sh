#!/bin/bash
# MedSeRL SGLang Multi-Turn Smoke Test
#
# Modeled directly on the official veRL sglang multiturn examples:
#   https://github.com/verl-project/verl/tree/main/examples/sglang_multiturn
#   → run_qwen3-4b_gsm8k_multiturn.sh
#
# Key design choices (matching official examples):
#   - rollout engine:  sglang  (required for multi-turn)
#   - rollout mode:    async   (recommended for sglang)
#   - algorithm:       REINFORCE++ (no critic needed)
#   - reward:          reward_function.py (rule-based, no judge server)
#   - multi_stage_wake_up: True (sglang → FSDP share the same GPU)
#   - gpu_memory_utilization: 0.5 (matches official — sglang shares GPU)
#   - ref param_offload: True (save GPU memory for 1-GPU setup)
#
# Usage:
#   bash scripts/self_play/run_rule_smoke.sh
#
# Env overrides:
#   ACTOR_MODEL   — Actor model path   (default: Qwen/Qwen3-4B)
#   N_GPUS        — Number of GPUs     (default: 1)
#   TRAIN_EPOCHS  — Training epochs    (default: 1)

set -x
ulimit -n 65535

# ── Configuration ─────────────────────────────────────────────────────────────
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_PATH="$PROJECT_DIR/scripts/self_play/configs"

ACTOR_MODEL="${ACTOR_MODEL:-Qwen/Qwen3-4B}"
N_GPUS="${N_GPUS:-1}"
TRAIN_EPOCHS="${TRAIN_EPOCHS:-1}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="smoke_sglang_${TIMESTAMP}"
OUTPUT_DIR="$PROJECT_DIR/outputs/self_play/smoke_sglang_${TIMESTAMP}"
SMOKE_LOG="$OUTPUT_DIR/smoke_test.log"

TRAIN_PARQUET="$PROJECT_DIR/data_processed/self_play/train.parquet"
VAL_PARQUET="$PROJECT_DIR/data_processed/self_play/val.parquet"

mkdir -p "$OUTPUT_DIR"

# ── Environment ───────────────────────────────────────────────────────────────
export PYTHONPATH="$PROJECT_DIR:$PYTHONPATH"
export HYDRA_FULL_ERROR=1
export CUDA_MODULE_LOADING=LAZY

# ── Cleanup trap ──────────────────────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Trap: cleaning up..."
    ray stop --force 2>/dev/null || true
    pkill -9 -f "gcs_server|raylet|plasma_store|sglang" 2>/dev/null || true
    echo "  Cleanup complete."
}
trap cleanup EXIT

# ── Kill stale processes ──────────────────────────────────────────────────────
echo "=== Cleanup: killing stale processes ==="
ray stop --force 2>/dev/null || true
pkill -9 -f "ray::|sglang|vllm|from multiprocessing" 2>/dev/null || true
sleep 2
echo "  Done."

# ── GPU check ─────────────────────────────────────────────────────────────────
echo ""
echo "=== GPU Memory ==="
nvidia-smi --query-gpu=index,memory.free,memory.total --format=csv,noheader 2>/dev/null || echo "  No GPU found"

# ── Patch veRL (Docker only — skip on bare metal) ────────────────────────────
echo ""
echo "=== Patching veRL (Docker fixes) ==="

# Patch: GLIBC workaround (widen except ImportError → except (ImportError, OSError))
python3 << 'PATCH_GLIBC'
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
    print("  GLIBC: file not found (not Docker?), skipping")
PATCH_GLIBC

# ── Data check ────────────────────────────────────────────────────────────────
echo ""
echo "=== Data Check ==="
if [ ! -f "$TRAIN_PARQUET" ]; then
    echo "ERROR: $TRAIN_PARQUET not found."
    echo "  Run:  python3 scripts/self_play/preprocess_medec.py"
    exit 1
fi
echo "  train: $TRAIN_PARQUET"

if [ ! -f "$VAL_PARQUET" ]; then
    cp "$TRAIN_PARQUET" "$VAL_PARQUET"
    echo "  val:   created from train.parquet"
else
    echo "  val:   $VAL_PARQUET"
fi

# ── Start Ray head ────────────────────────────────────────────────────────────
# Pre-start Ray with bounded resources so GCS doesn't timeout in Docker.
# RAY_ADDRESS=auto tells veRL's ray.init() to connect here instead of spawning.
echo ""
echo "=== Starting Ray head node ==="
rm -rf /tmp/ray /dev/shm/ray 2>/dev/null || true
mkdir -p /dev/shm/ray

ray start --head \
    --num-cpus=8 \
    --num-gpus="$N_GPUS" \
    --temp-dir=/dev/shm/ray \
    --node-ip-address=127.0.0.1 \
    --include-dashboard=false \
    --port=6379 \
    --object-store-memory=10000000000

export RAY_ADDRESS=auto
sleep 3
echo "  Ray head ready (RAY_ADDRESS=$RAY_ADDRESS)"

# ── Training ──────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "MedSeRL SGLang Smoke Test"
echo "=================================================="
echo "  Actor model  : $ACTOR_MODEL"
echo "  Rollout      : sglang (async, multi_stage_wake_up)"
echo "  Algorithm    : REINFORCE++"
echo "  Reward       : reward_function.py (rule-based)"
echo "  GPUs         : $N_GPUS"
echo "  Output       : $OUTPUT_DIR"
echo "=================================================="
echo ""

python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="ppo_sglang_smoke" \
    \
    algorithm.adv_estimator=reinforce_plus_plus \
    algorithm.use_kl_in_reward=False \
    \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data.train_batch_size=8 \
    data.train_max_samples=20 \
    data.val_max_samples=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path="$ACTOR_MODEL" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.multi_stage_wake_up=True \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=2 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    \
    critic.enable=false \
    reward_model.enable=False \
    \
    custom_reward_function.path="$PROJECT_DIR/scripts/self_play/reward_function.py" \
    custom_reward_function.name=compute_score \
    \
    trainer.total_epochs="$TRAIN_EPOCHS" \
    trainer.critic_warmup=0 \
    trainer.logger=console \
    trainer.project_name=medserl-smoke \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=99999 \
    trainer.val_before_train=False \
    "$@" \
    2>&1 | tee "$SMOKE_LOG"

TRAIN_EXIT=${PIPESTATUS[0]}

# ── Verification ──────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "=== Smoke Test Verification ==="
echo "=================================================="

CS_CALLS=$(grep -c "compute_score called" "$SMOKE_LOG" 2>/dev/null || true)
CS_CALLS="${CS_CALLS:-0}"

echo ""
echo "  compute_score invocations : $CS_CALLS"
echo "  Training exit code        : $TRAIN_EXIT"
echo ""

if [ "$CS_CALLS" -gt 0 ] 2>/dev/null; then
    echo "SMOKE TEST PASSED: compute_score called $CS_CALLS times."
    grep "compute_score called" "$SMOKE_LOG" | head -3
elif [ "$TRAIN_EXIT" -eq 0 ]; then
    echo "SMOKE TEST PASSED: Training completed successfully (exit 0)."
else
    echo "SMOKE TEST FAILED."
fi

echo ""
echo "  Log     : $SMOKE_LOG"
echo "  Outputs : $OUTPUT_DIR"

exit $TRAIN_EXIT
