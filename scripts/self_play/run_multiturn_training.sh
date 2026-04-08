#!/bin/bash
# MedSeRL self-play training launcher.
#
# Active path: batched chained injector -> assessor self-play using offline vLLM
# data generation plus standard single-turn VERL training.
#
# This aligns with the simpler official VERL rollout setup for vLLM and avoids
# the SGLang interaction-system runtime for this experiment.

set -e

SCREEN_SESSION="${SCREEN_SESSION:-medserl_selfplay_train}"
AUTO_SCREEN="${AUTO_SCREEN:-1}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
SCREEN_LOG_DIR="${SCREEN_LOG_DIR:-${REPO_ROOT}/logs/screen}"

if [ "$AUTO_SCREEN" = "1" ] && [ -z "${STY:-}" ]; then
    if ! command -v screen >/dev/null 2>&1; then
        echo "ERROR: 'screen' is required but not installed."
        exit 1
    fi

    if screen -list | grep -q "[[:space:]]${SCREEN_SESSION}[[:space:]]"; then
        echo "Screen session '${SCREEN_SESSION}' already exists."
        echo "Attach with: screen -r ${SCREEN_SESSION}"
        echo "Kill with:   screen -X -S ${SCREEN_SESSION} quit"
        exit 1
    fi

    mkdir -p "${SCREEN_LOG_DIR}"
    LOG_TS="$(date +%Y%m%d_%H%M%S)"
    SCREEN_LOG_FILE="${SCREEN_LOG_DIR}/${SCREEN_SESSION}_${LOG_TS}.log"
    SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
    QUOTED_ARGS=""
    for arg in "$@"; do
        QUOTED_ARGS+=" $(printf '%q' "$arg")"
    done

    echo "Launching ${SCRIPT_PATH} in screen session '${SCREEN_SESSION}'..."
    screen -L -Logfile "${SCREEN_LOG_FILE}" -dmS "${SCREEN_SESSION}" bash -lc "AUTO_SCREEN=0 bash $(printf '%q' "$SCRIPT_PATH")${QUOTED_ARGS}"
    echo "Attach with: screen -r ${SCREEN_SESSION}"
    echo "Kill with:   screen -X -S ${SCREEN_SESSION} quit"
    echo "Log file:    ${SCREEN_LOG_FILE}"
    exit 0
fi

SMOKE="${SMOKE:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/self_play_chained_vllm}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-medserl_selfplay_chained_vllm}"
MODEL_PATH="${ACTOR_MODEL:-Abdine/qwen3-4b-medrect-mixed}"
N_GPUS="${N_GPUS:-2}"
ROLLOUT_TP="${ROLLOUT_TP:-1}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.45}"
ROLLOUT_MAX_NUM_SEQS="${ROLLOUT_MAX_NUM_SEQS:-8}"
DATAGEN_GPU_MEMORY_UTILIZATION="${DATAGEN_GPU_MEMORY_UTILIZATION:-0.45}"
DATAGEN_MAX_TOKENS="${DATAGEN_MAX_TOKENS:-1024}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-8}"
RAY_CLEAN_START="${RAY_CLEAN_START:-1}"
ZERO_SUM="${ZERO_SUM:-1}"
SKIP_DATAGEN="${SKIP_DATAGEN:-0}"
REQUIRE_JUDGE="${REQUIRE_JUDGE:-1}"
RESUME_MODE="${RESUME_MODE:-disable}"
RESUME_FROM_PATH="${RESUME_FROM_PATH:-}"
WANDB="${WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-medserl-selfplay}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_BASE_URL="${WANDB_BASE_URL:-}"
WANDB_MODE="${WANDB_MODE:-online}"

export JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-8B}"
export SIMPLE_JUDGE_WEIGHT="${SIMPLE_JUDGE_WEIGHT:-0.3}"
export VLLM_USE_V1="${VLLM_USE_V1:-1}"

if [ "$SMOKE" = "1" ]; then
    MAX_PAIRS="${MAX_PAIRS:-12}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
    VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-8}"
    TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
    PPO_EPOCHS="${PPO_EPOCHS:-1}"
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1280}"
    SAVE_FREQ="${SAVE_FREQ:-auto}"
    TEST_FREQ="${TEST_FREQ:--1}"
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
    ACTOR_CKPT_SAVE_CONTENTS="${ACTOR_CKPT_SAVE_CONTENTS:-[model,hf_model,extra]}"
    ACTOR_CKPT_LOAD_CONTENTS="${ACTOR_CKPT_LOAD_CONTENTS:-[model,extra]}"
else
    MAX_PAIRS="${MAX_PAIRS:-}"
    TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
    VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-16}"
    TOTAL_EPOCHS="${TOTAL_EPOCHS:-3}"
    PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}"
    PPO_EPOCHS="${PPO_EPOCHS:-2}"
    MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
    MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1280}"
    SAVE_FREQ="${SAVE_FREQ:--1}"
    TEST_FREQ="${TEST_FREQ:--1}"
    VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-false}"
    ACTOR_CKPT_SAVE_CONTENTS="${ACTOR_CKPT_SAVE_CONTENTS:-[model,optimizer,extra,hf_model]}"
    ACTOR_CKPT_LOAD_CONTENTS="${ACTOR_CKPT_LOAD_CONTENTS:-[model,optimizer,extra]}"
fi

if [ -d "/workspace/med_serl" ]; then
    PROJECT_ROOT="/workspace/med_serl"
    CONDA_BASE="/workspace/miniconda3"
else
    PROJECT_ROOT="$REPO_ROOT"
    CONDA_BASE="${HOME}/miniconda3"
fi

DATA_DIR="${DATA_DIR:-$PROJECT_ROOT/data_processed/self_play}"
TRAIN_PARQUET="$DATA_DIR/train_chained.parquet"
VAL_PARQUET="$DATA_DIR/val_chained.parquet"

TRAINER_LOGGER="console"
if [ "$WANDB" = "1" ]; then
    TRAINER_LOGGER="[console,wandb]"
fi

echo "=================================================="
echo "MedSeRL Self-Play Training (Chained vLLM)"
echo "=================================================="
echo "Project root: $PROJECT_ROOT"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "Smoke mode: $SMOKE"
echo "GPUs: $N_GPUS"
echo "Rollout TP: $ROLLOUT_TP"
echo "Rollout GPU mem util: $ROLLOUT_GPU_MEMORY_UTILIZATION"
echo "Rollout max num seqs: $ROLLOUT_MAX_NUM_SEQS"
echo "Datagen GPU mem util: $DATAGEN_GPU_MEMORY_UTILIZATION"
echo "Datagen max tokens: $DATAGEN_MAX_TOKENS"
if [ -n "$MAX_PAIRS" ]; then
    echo "Max pairs: $MAX_PAIRS"
else
    echo "Max pairs: ALL"
fi
echo "Train batch size: $TRAIN_BATCH_SIZE"
echo "Total epochs: $TOTAL_EPOCHS"
echo "Max response length: $MAX_RESPONSE_LENGTH"
echo "Ray CPUs: $RAY_NUM_CPUS"
echo "Save freq: $SAVE_FREQ"
echo "Test freq: $TEST_FREQ"
echo "Val before train: $VAL_BEFORE_TRAIN"
echo "Zero-sum pass: $ZERO_SUM"
echo "Resume mode: $RESUME_MODE"
echo "W&B: $WANDB"
echo "Logger: $TRAINER_LOGGER"
echo "Judge URL: ${JUDGE_VLLM_URL:-<disabled - rule reward only>}"
echo "Require judge: $REQUIRE_JUDGE"
echo "=================================================="

if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate med_serl
    echo "✓ Conda environment 'med_serl' activated"
elif [ -f "$PROJECT_ROOT/med_serl/bin/activate" ]; then
    source "$PROJECT_ROOT/med_serl/bin/activate"
    echo "✓ Virtual environment activated"
fi

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export RAY_DISABLE_DOCKER_CPU_WARNING=1
export RAY_USE_MULTIPROCESSING_CPU_COUNT=1
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_DEDUP_LOGS=0
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export PYTHONUNBUFFERED=1
export PYTHONFAULTHANDLER=1
export HYDRA_FULL_ERROR=1
export WANDB_PROJECT
export WANDB_MODE
if [ -n "${WANDB_API_KEY:-}" ]; then
    export WANDB_API_KEY
fi
if [ -n "$WANDB_ENTITY" ]; then
    export WANDB_ENTITY
fi
if [ -n "$WANDB_BASE_URL" ]; then
    export WANDB_BASE_URL
fi
unset TORCH_NCCL_AVOID_RECORD_STREAMS
echo "✓ PYTHONPATH set to include: $PROJECT_ROOT"

if [ "$REQUIRE_JUDGE" = "1" ] && [ -z "${JUDGE_VLLM_URL:-}" ]; then
    echo "ERROR: JUDGE_VLLM_URL is not set."
    echo "Start the judge server and export JUDGE_VLLM_URL before training."
    exit 1
fi

python3 - <<'PY'
import ray, torch

print(f"Version check: ray={ray.__version__}")
print(f"Version check: torch={torch.__version__}")
try:
    import vllm
    print(f"Version check: vllm={vllm.__version__}")
except Exception as exc:
    print(f"Version check: vllm import failed: {exc}")
PY

cd "$PROJECT_ROOT"
mkdir -p "$OUTPUT_DIR"
mkdir -p "$DATA_DIR"

echo ""
echo "=== Phase A: Chained Data Generation ==="
DATAGEN_MAX_PAIRS_ARGS=()
if [ -n "$MAX_PAIRS" ] && [ "$MAX_PAIRS" != "0" ]; then
    DATAGEN_MAX_PAIRS_ARGS=(--max-pairs "$MAX_PAIRS")
fi

if [ "$SKIP_DATAGEN" = "1" ] && [ -f "$TRAIN_PARQUET" ]; then
    echo "SKIP_DATAGEN=1 — reusing existing $TRAIN_PARQUET"
else
    ZERO_SUM_FLAG=""
    if [ "$ZERO_SUM" = "1" ]; then
        ZERO_SUM_FLAG="--zero-sum"
    fi

    python3 scripts/self_play/generate_chained_data.py \
        --model "$MODEL_PATH" \
        --input "$PROJECT_ROOT/data_processed/medec_paired/train_val_split/rl_train.jsonl" \
        --output "$TRAIN_PARQUET" \
        --injection-prompts "$PROJECT_ROOT/configs/prompts/error_injection_prompts_v4.json" \
        --detection-prompts "$PROJECT_ROOT/configs/prompts/detection_localization_prompts.json" \
        --gpu-memory-utilization "$DATAGEN_GPU_MEMORY_UTILIZATION" \
        --max-tokens "$DATAGEN_MAX_TOKENS" \
        "${DATAGEN_MAX_PAIRS_ARGS[@]}" \
        $ZERO_SUM_FLAG
fi

if [ -f "$PROJECT_ROOT/data_processed/medec_paired/train_val_split/rl_val.jsonl" ] && [ "$SKIP_DATAGEN" != "1" ]; then
    ZERO_SUM_FLAG=""
    if [ "$ZERO_SUM" = "1" ]; then
        ZERO_SUM_FLAG="--zero-sum"
    fi
    python3 scripts/self_play/generate_chained_data.py \
        --model "$MODEL_PATH" \
        --input "$PROJECT_ROOT/data_processed/medec_paired/train_val_split/rl_val.jsonl" \
        --output "$VAL_PARQUET" \
        --injection-prompts "$PROJECT_ROOT/configs/prompts/error_injection_prompts_v4.json" \
        --detection-prompts "$PROJECT_ROOT/configs/prompts/detection_localization_prompts.json" \
        --gpu-memory-utilization "$DATAGEN_GPU_MEMORY_UTILIZATION" \
        --max-tokens "$DATAGEN_MAX_TOKENS" \
        "${DATAGEN_MAX_PAIRS_ARGS[@]}" \
        $ZERO_SUM_FLAG
elif [ ! -f "$VAL_PARQUET" ]; then
    echo "Warning: No separate validation parquet, copying training parquet"
    cp "$TRAIN_PARQUET" "$VAL_PARQUET"
fi

echo ""
echo "=== Verifying Chained Data Format ==="
TRAIN_EXAMPLE_COUNT="$(python3 - <<PY
import pyarrow.parquet as pq
table = pq.read_table("$TRAIN_PARQUET")
df = table.to_pandas()
print(f"Total examples: {len(df)}")
print("Roles:", df["extra_info"].apply(lambda x: x.get("role", "?")).value_counts().to_dict())
print("Chained assessor rows:", int(df["extra_info"].apply(lambda x: bool(x.get("chained"))).sum()))
print("Prompt type:", type(df.iloc[0]["prompt"]))
print("✓ Chained data ready")
print(len(df))
PY
) "
printf '%s\n' "$TRAIN_EXAMPLE_COUNT" | sed '$d'
TRAIN_EXAMPLE_COUNT="$(printf '%s\n' "$TRAIN_EXAMPLE_COUNT" | tail -n 1)"

RESOLVED_SAVE_FREQ="$SAVE_FREQ"
STEPS_PER_EPOCH=1
if [ -n "$TRAIN_EXAMPLE_COUNT" ] && [ "$TRAIN_EXAMPLE_COUNT" -gt 0 ] 2>/dev/null; then
    STEPS_PER_EPOCH=$(( (TRAIN_EXAMPLE_COUNT + TRAIN_BATCH_SIZE - 1) / TRAIN_BATCH_SIZE ))
    [ "$STEPS_PER_EPOCH" -lt 1 ] && STEPS_PER_EPOCH=1
fi
if [ "$SAVE_FREQ" = "auto" ]; then
    RESOLVED_SAVE_FREQ="$STEPS_PER_EPOCH"
elif [[ "$SAVE_FREQ" =~ ^[0-9]+$ ]] && [ "$SAVE_FREQ" -gt 0 ]; then
    if [ "$SAVE_FREQ" -gt "$STEPS_PER_EPOCH" ]; then
        RESOLVED_SAVE_FREQ="$STEPS_PER_EPOCH"
    fi
fi
echo "Resolved save freq: $RESOLVED_SAVE_FREQ"

echo ""
echo "=== Phase B: vLLM REINFORCE++ Training ==="
echo "Stage 1: offline injector batch via vLLM"
echo "Stage 2: offline assessor batch via vLLM"
echo "Stage 3: standard single-turn VERL training on chained parquet"

if [ "$RAY_CLEAN_START" = "1" ]; then
    if command -v ray >/dev/null 2>&1; then
        echo ""
        echo "=== Cleaning Existing Ray Processes ==="
        ray stop --force >/dev/null 2>&1 || true
        sleep 2
    else
        echo "Warning: 'ray' command not found; skipping Ray cleanup"
    fi
fi

RESUME_ARGS=()
RESUME_ARGS+=("trainer.resume_mode=$RESUME_MODE")
if [ "$RESUME_MODE" = "resume_path" ]; then
    if [ -z "$RESUME_FROM_PATH" ]; then
        echo "ERROR: RESUME_MODE=resume_path requires RESUME_FROM_PATH"
        exit 1
    fi
    RESUME_ARGS+=("trainer.resume_from_path=$RESUME_FROM_PATH")
fi

python3 -m verl.trainer.main_ppo \
    --config-name="ppo_trainer" \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files="$TRAIN_PARQUET" \
    data.val_files="$VAL_PARQUET" \
    data.return_raw_chat=True \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.val_batch_size="$VAL_BATCH_SIZE" \
    data.max_prompt_length="$MAX_PROMPT_LENGTH" \
    data.max_response_length="$MAX_RESPONSE_LENGTH" \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.shuffle=True \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs="$PPO_EPOCHS" \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.checkpoint.save_contents="$ACTOR_CKPT_SAVE_CONTENTS" \
    actor_rollout_ref.actor.checkpoint.load_contents="$ACTOR_CKPT_LOAD_CONTENTS" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.disable_log_stats=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size="$ROLLOUT_TP" \
    actor_rollout_ref.rollout.gpu_memory_utilization="$ROLLOUT_GPU_MEMORY_UTILIZATION" \
    actor_rollout_ref.rollout.max_num_seqs="$ROLLOUT_MAX_NUM_SEQS" \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    critic.enable=false \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    algorithm.use_kl_in_reward=False \
    reward_model.enable=False \
    custom_reward_function.path="$PROJECT_ROOT/scripts/self_play/simple_judge_reward.py" \
    custom_reward_function.name=async_compute_score \
    trainer.logger="$TRAINER_LOGGER" \
    trainer.project_name="$WANDB_PROJECT" \
    trainer.experiment_name="$EXPERIMENT_NAME" \
    trainer.default_local_dir="$OUTPUT_DIR" \
    trainer.n_gpus_per_node="$N_GPUS" \
    trainer.nnodes=1 \
    trainer.total_epochs="$TOTAL_EPOCHS" \
    trainer.save_freq="$RESOLVED_SAVE_FREQ" \
    "${RESUME_ARGS[@]}" \
    trainer.test_freq="$TEST_FREQ" \
    trainer.val_before_train="$VAL_BEFORE_TRAIN" \
    "++ray_kwargs.ray_init.num_cpus=$RAY_NUM_CPUS" \
    "++ray_kwargs.runtime_env.working_dir=$PROJECT_ROOT" \
    "++ray_kwargs.runtime_env.env_vars.PYTHONPATH=$PYTHONPATH" \
    "++ray_kwargs.runtime_env.env_vars.PYTHONUNBUFFERED=1" \
    "++ray_kwargs.runtime_env.env_vars.PYTHONFAULTHANDLER=1" \
    "++ray_kwargs.runtime_env.env_vars.TOKENIZERS_PARALLELISM=false" \
    "++ray_kwargs.runtime_env.env_vars.TRANSFORMERS_NO_ADVISORY_WARNINGS=1" \
    "++ray_kwargs.runtime_env.env_vars.RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0" \
    "++ray_kwargs.runtime_env.env_vars.RAY_DEDUP_LOGS=0" \
    "++ray_kwargs.runtime_env.env_vars.HYDRA_FULL_ERROR=1" \
    "++ray_kwargs.runtime_env.env_vars.VLLM_USE_V1=$VLLM_USE_V1" \
    "++ray_kwargs.runtime_env.env_vars.WANDB_PROJECT=$WANDB_PROJECT" \
    "++ray_kwargs.runtime_env.env_vars.WANDB_MODE=$WANDB_MODE" \
    "++ray_kwargs.runtime_env.env_vars.WANDB_API_KEY=${WANDB_API_KEY:-}" \
    "++ray_kwargs.runtime_env.env_vars.WANDB_ENTITY=$WANDB_ENTITY" \
    "++ray_kwargs.runtime_env.env_vars.WANDB_BASE_URL=$WANDB_BASE_URL"

echo ""
echo "=================================================="
echo "Training Complete!"
echo "=================================================="
echo "Outputs: $OUTPUT_DIR"
echo "Train parquet: $TRAIN_PARQUET"
echo "Val parquet: $VAL_PARQUET"
