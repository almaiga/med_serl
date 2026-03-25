#!/bin/bash
# MedSeRL Minimal Smoke Test
#
# Follows the blog-post pattern (huggingface.co/blog/Weyaxi/engineering-handbook-grpo-lora-with-verl):
#   - NO custom --config-path / --config-name → uses verl's built-in ppo_trainer directly
#   - ALL config via CLI overrides
#   - Phase A (datagen) only runs if parquet is missing or SKIP_DATAGEN=0
#
# Usage:
#   bash scripts/self_play/run_smoke_minimal.sh
#
# Env overrides:
#   ACTOR_MODEL    — model path or HF hub id  (default: Qwen/Qwen3-4B)
#   SMOKE_STEPS    — max training steps        (default: 3)
#   MAX_PAIRS      — pairs for datagen         (default: 20)
#   SKIP_DATAGEN   — 1=reuse parquet if exists (default: auto)
#   N_GPUS         — number of GPUs to use     (default: 2)

set -e

ACTOR_MODEL="${ACTOR_MODEL:-Qwen/Qwen3-4B}"
SMOKE_STEPS="${SMOKE_STEPS:-3}"
MAX_PAIRS="${MAX_PAIRS:-20}"
N_GPUS="${N_GPUS:-2}"
# Scale batch size with GPUs so per-GPU load stays constant
TRAIN_BATCH_SIZE=$(( 16 * N_GPUS ))
TRAIN_SAMPLES=$(( SMOKE_STEPS * TRAIN_BATCH_SIZE ))

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_PARQUET="$PROJECT_ROOT/data_processed/self_play/train_chained.parquet"
VAL_PARQUET="$PROJECT_ROOT/data_processed/self_play/val_chained.parquet"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="outputs/self_play/smoke_minimal_${TIMESTAMP}"
SMOKE_LOG="$OUTPUT_DIR/smoke.log"

mkdir -p "$OUTPUT_DIR"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# ── GPU env ──────────────────────────────────────────────────────────────────
# Expose all GPUs; Ray + verl allocate them automatically
unset CUDA_VISIBLE_DEVICES
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_DEDUP_LOGS=0
export RAY_memory_monitor_refresh_ms=0
RAY_TMPDIR_PATH="/workspace/ray_tmp"
mkdir -p "$RAY_TMPDIR_PATH"
export RAY_TMPDIR="$RAY_TMPDIR_PATH"
export HYDRA_FULL_ERROR=1

echo "=============================="
echo "MedSeRL Minimal Smoke Test"
echo "  Model  : $ACTOR_MODEL"
echo "  Steps  : $SMOKE_STEPS"
echo "  Pairs  : $MAX_PAIRS"
echo "  GPUs   : $N_GPUS"
echo "  Batch  : $TRAIN_BATCH_SIZE"
echo "  Output : $OUTPUT_DIR"
echo "=============================="

# ── Phase A: datagen (skip if parquet already exists and SKIP_DATAGEN≠0) ────
if [ "${SKIP_DATAGEN:-auto}" != "1" ] && [ ! -f "$TRAIN_PARQUET" ]; then
    echo ""
    echo "=== Phase A: generating training data ==="
    python3 "$PROJECT_ROOT/scripts/self_play/generate_chained_data.py" \
        --model             "$ACTOR_MODEL" \
        --input             "$PROJECT_ROOT/data_processed/medec_paired/train_val_split/rl_train.jsonl" \
        --output            "$TRAIN_PARQUET" \
        --injection-prompts "$PROJECT_ROOT/configs/prompts/error_injection_prompts_v4.json" \
        --detection-prompts "$PROJECT_ROOT/configs/prompts/detection_localization_prompts.json" \
        --max-pairs         "$MAX_PAIRS"
    cp "$TRAIN_PARQUET" "$VAL_PARQUET"
    echo "  Datagen done."
else
    echo "Phase A: reusing $TRAIN_PARQUET"
    [ -f "$TRAIN_PARQUET" ] || { echo "ERROR: $TRAIN_PARQUET not found. Run without SKIP_DATAGEN=1 first."; exit 1; }
    [ -f "$VAL_PARQUET" ]   || cp "$TRAIN_PARQUET" "$VAL_PARQUET"
fi

# ── Phase B: verl training — NO custom config file ───────────────────────────
echo ""
echo "=== Phase B: GRPO training ($SMOKE_STEPS steps, no custom YAML) ==="

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.val_batch_size=8 \
    data.train_max_samples=$TRAIN_SAMPLES \
    data.val_max_samples=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.return_raw_chat=True \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    \
    actor_rollout_ref.model.path="$ACTOR_MODEL" \
    actor_rollout_ref.model.trust_remote_code=True \
    "++actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=$(( 4 * N_GPUS )) \
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
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$N_GPUS \
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
    trainer.project_name=medserl-smoke \
    trainer.experiment_name="smoke_minimal_${TIMESTAMP}" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.n_gpus_per_node=$N_GPUS \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=-1 \
    trainer.val_before_train=False \
    \
    ++ray_kwargs.ray_init.include_dashboard=False \
    ++ray_kwargs.ray_init.num_cpus=8 \
    "+ray_kwargs.ray_init._temp_dir=$RAY_TMPDIR_PATH" \
    ++ray_kwargs.ray_init.object_store_memory=1000000000 \
    \
    custom_reward_function.path="$PROJECT_ROOT/scripts/self_play/reward_function.py" \
    custom_reward_function.name="compute_score" \
    2>&1 | tee "$SMOKE_LOG"

TRAIN_EXIT=${PIPESTATUS[0]}

echo ""
echo "=============================="
[ "$TRAIN_EXIT" -eq 0 ] && echo "PASSED (exit 0)" || echo "FAILED (exit $TRAIN_EXIT)"
echo "Log: $SMOKE_LOG"
echo "=============================="
exit $TRAIN_EXIT
