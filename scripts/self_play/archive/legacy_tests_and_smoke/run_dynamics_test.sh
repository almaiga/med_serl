#!/bin/bash
# MedSeRL Training Dynamics Test
# Inspired by: github.com/mickelliu/selfplay-redteaming
#
# Runs veRL REINFORCE++ for ROUNDS × STEPS_PER_ROUND steps.
# val_before_train=True + test_freq=1 gives a reward reading at step 0
# and after every update step — confirms policy is actually moving.
#
# Key hyperparams aligned with selfplay-redteaming:
#   - lr           = 5e-7      (their actor_learning_rate)
#   - kl_coef      = 0.01      (their init_kl_coef)
#   - reward_type  = general_sum (each role gets independent reward, not negated)
#   - normalize_reward via reinforce_plus_plus whitening (their normalize_reward)
#   - enforce_eager = True     (avoid CUDA graph issues, their --enforce_eager)
#   - No SFT aux loss          (model is already fine-tuned)
#
# Usage:
#   bash scripts/self_play/run_dynamics_test.sh
#
# Env overrides:
#   ACTOR_MODEL      — model path (default: Qwen/Qwen3-4B)
#   ROUNDS           — train→eval cycles (default: 3)
#   STEPS_PER_ROUND  — verl steps per round (default: 5)
#   SKIP_DATAGEN     — set to 1 to reuse existing parquet

# ─── Configuration ────────────────────────────────────────────────────────────
ACTOR_MODEL="${ACTOR_MODEL:-Qwen/Qwen3-4B}"
ROUNDS="${ROUNDS:-3}"
STEPS_PER_ROUND="${STEPS_PER_ROUND:-5}"
TRAIN_BATCH_SIZE=16   # 1 GPU — keep small for speed
TRAIN_SAMPLES=$(( STEPS_PER_ROUND * TRAIN_BATCH_SIZE ))

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="dynamics_test_${TIMESTAMP}"
OUTPUT_DIR="outputs/self_play/dynamics_test_${TIMESTAMP}"

RAY_TMPDIR_PATH="/workspace/ray_tmp"
mkdir -p "$RAY_TMPDIR_PATH"

# ── 1 GPU: start here, scale to 2 once dynamics are verified ──
export CUDA_VISIBLE_DEVICES=0
export RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1

export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_DEDUP_LOGS=0
export RAY_USE_MULTIPROCESSING_CPU_COUNT=1
export RAY_memory_monitor_refresh_ms=0
export RAY_raylet_start_wait_time_s=300
export RAY_TMPDIR="$RAY_TMPDIR_PATH"
export RAY_GCS_SERVER_REQUEST_TIMEOUT_S=60
export HYDRA_FULL_ERROR=1

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/scripts/self_play/configs"
TRAIN_PARQUET="$PROJECT_ROOT/data_processed/self_play/train_grpo.parquet"
VAL_PARQUET="$PROJECT_ROOT/data_processed/self_play/val_grpo.parquet"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_ROOT/results/self_play"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# ─── Cleanup ──────────────────────────────────────────────────────────────────
echo "=== Cleanup ==="
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -f "vllm.worker"      2>/dev/null || true
sleep 2
rm -rf "$RAY_TMPDIR_PATH"/* 2>/dev/null || true
echo "  done."

# ─── Data ─────────────────────────────────────────────────────────────────────
if [ "${SKIP_DATAGEN:-0}" != "1" ] || [ ! -f "$TRAIN_PARQUET" ]; then
    echo "=== Generating training data ==="
    python3 "$PROJECT_ROOT/scripts/self_play/preprocess_medec.py" \
        --input "$PROJECT_ROOT/data_processed/medec_paired/train_val_split/rl_train.jsonl" \
        --output "$TRAIN_PARQUET" \
        --injection-prompts "$PROJECT_ROOT/configs/prompts/error_injection_prompts_v4.json" \
        --detection-prompts "$PROJECT_ROOT/configs/prompts/detection_localization_prompts.json" \
        --roles mixed \
        --max-pairs 20
    [ $? -ne 0 ] && { echo "ERROR: data generation failed"; exit 1; }
else
    echo "=== Reusing $TRAIN_PARQUET ==="
fi
[ ! -f "$VAL_PARQUET" ] && cp "$TRAIN_PARQUET" "$VAL_PARQUET"

# ─── Patch verl ───────────────────────────────────────────────────────────────
python3 "$PROJECT_ROOT/scripts/self_play/patch_verl_ray.py"

# ─── Multi-round training loop ────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  Dynamics Test: $ROUNDS rounds × $STEPS_PER_ROUND steps"
echo "  Model        : $ACTOR_MODEL"
echo "  LR           : 5e-7  (selfplay-redteaming aligned)"
echo "  KL coef      : 0.01  (selfplay-redteaming aligned)"
echo "  Reward type  : general_sum (independent per-role rewards)"
echo "  GPU          : 1 (scale to 2 after verification)"
echo "=================================================="

ROUND_SCORES=()

for ROUND in $(seq 1 "$ROUNDS"); do
    echo ""
    echo "=================================================="
    echo "  ROUND $ROUND / $ROUNDS"
    echo "=================================================="

    ROUND_LOG="$OUTPUT_DIR/round_${ROUND}.log"
    GAME_LOG="$OUTPUT_DIR/round_${ROUND}_interactions.jsonl"
    export MEDSERL_GAME_LOG="$GAME_LOG"

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
        data.max_prompt_length=2048 \
        data.max_response_length=2048 \
        data.filter_overlong_prompts=False \
        data.truncation=error \
        data.return_raw_chat=True \
        \
        actor_rollout_ref.model.path="$ACTOR_MODEL" \
        "++actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
        actor_rollout_ref.model.use_remove_padding=False \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=5e-7 \
        actor_rollout_ref.actor.ppo_mini_batch_size=4 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=True \
        actor_rollout_ref.actor.kl_loss_coef=0.01 \
        actor_rollout_ref.actor.kl_loss_type=low_var_kl \
        actor_rollout_ref.actor.entropy_coeff=0 \
        actor_rollout_ref.actor.fsdp_config.param_offload=True \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.actor.strategy=fsdp2 \
        \
        actor_rollout_ref.rollout.name=vllm \
        actor_rollout_ref.rollout.temperature=0.7 \
        actor_rollout_ref.rollout.top_p=1.0 \
        actor_rollout_ref.rollout.top_k=-1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
        actor_rollout_ref.rollout.n=1 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
        "++actor_rollout_ref.rollout.enforce_eager=True" \
        \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        actor_rollout_ref.ref.strategy=fsdp2 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
        \
        critic.enable=false \
        reward_model.enable=False \
        algorithm.use_kl_in_reward=False \
        algorithm.kl_ctrl.kl_coef=0.01 \
        \
        trainer.total_epochs=1 \
        trainer.critic_warmup=0 \
        trainer.logger=console \
        trainer.project_name=medserl-dynamics-test \
        trainer.experiment_name="${EXPERIMENT_NAME}_r${ROUND}" \
        trainer.default_local_dir="$OUTPUT_DIR/round_${ROUND}" \
        trainer.n_gpus_per_node=1 \
        trainer.nnodes=1 \
        trainer.save_freq=-1 \
        trainer.test_freq=1 \
        trainer.val_before_train=True \
        ++ray_kwargs.ray_init.include_dashboard=False \
        ++ray_kwargs.ray_init.num_cpus=4 \
        "++ray_kwargs.ray_init._temp_dir=$RAY_TMPDIR_PATH" \
        ++ray_kwargs.ray_init.object_store_memory=500000000 \
        "++ray_kwargs.runtime_env.env_vars.MEDSERL_GAME_LOG=$GAME_LOG" \
        custom_reward_function.path="$PROJECT_ROOT/scripts/self_play/reward_function.py" \
        custom_reward_function.name="compute_score" \
        2>&1 | tee "$ROUND_LOG"

    ROUND_EXIT=${PIPESTATUS[0]}

    LAST_SCORE=$(grep -o "critic/score/mean:[0-9. -]*" "$ROUND_LOG" 2>/dev/null \
        | tail -1 | cut -d: -f2 || true)
    ROUND_SCORES+=("Round $ROUND: last_score=${LAST_SCORE:-(not found)}  exit=$ROUND_EXIT")

    if [ $ROUND_EXIT -ne 0 ]; then
        echo "ERROR: Round $ROUND failed (exit=$ROUND_EXIT). Stopping."
        break
    fi

    # Free GPU memory between rounds
    pkill -9 -f "ray::" 2>/dev/null || true
    pkill -9 -f "gcs_server" 2>/dev/null || true
    sleep 5
    rm -rf "$RAY_TMPDIR_PATH"/* 2>/dev/null || true
done

# ─── Results ──────────────────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "  DYNAMICS TEST RESULTS"
echo "=================================================="
for SUMMARY in "${ROUND_SCORES[@]}"; do
    echo "  $SUMMARY"
done

echo ""
echo "  Per-step critic/score/mean across all rounds:"
for ROUND in $(seq 1 "$ROUNDS"); do
    ROUND_LOG="$OUTPUT_DIR/round_${ROUND}.log"
    [ -f "$ROUND_LOG" ] || continue
    echo "  --- Round $ROUND ---"
    grep -o "critic/score/mean:[0-9. -]*" "$ROUND_LOG" 2>/dev/null | while read -r line; do
        echo "    $line"
    done
done

echo ""
echo "  KL divergence per step (confirms policy is moving):"
for ROUND in $(seq 1 "$ROUNDS"); do
    ROUND_LOG="$OUTPUT_DIR/round_${ROUND}.log"
    [ -f "$ROUND_LOG" ] || continue
    echo "  --- Round $ROUND ---"
    grep -o "actor/kl[^|]*" "$ROUND_LOG" 2>/dev/null | head -20 | while read -r line; do
        echo "    $line"
    done
done

LAST_ROUND_LOG="$OUTPUT_DIR/round_${ROUNDS}.log"
if [ -f "$LAST_ROUND_LOG" ]; then
    echo ""
    echo "  Quality analysis (last round):"
    python3 "$PROJECT_ROOT/scripts/self_play/analyze_smoke_quality.py" \
        --project-root "$PROJECT_ROOT" \
        --output-dir   "$OUTPUT_DIR/round_${ROUNDS}" \
        --smoke-log    "$LAST_ROUND_LOG" \
        --max-response-length 2048
fi

echo ""
echo "  Full logs: $OUTPUT_DIR"
echo "=================================================="
