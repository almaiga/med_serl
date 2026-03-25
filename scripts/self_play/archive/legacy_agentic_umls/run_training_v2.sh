#!/bin/bash
# MedSeRL Self-Play Training Script (v2)
# Two-phase game: Injector modifies note → Assessor classifies
# Uses SFT-aligned prompts (v4) and launches in screen session.
#
# Usage:
#   bash scripts/self_play/run_training.sh          # default: SFT 4B model
#   bash scripts/self_play/run_training.sh 8b       # use SFT 8B model
#   MODEL_PATH=/path/to/model bash scripts/self_play/run_training.sh  # custom model

set -e

# Ensure screen is installed
if ! command -v screen &> /dev/null; then
    echo "screen not found — installing via apt-get..."
    apt-get update -qq && apt-get install -y screen
    echo "screen installed."
fi

# Get project root
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$PROJECT_ROOT"

# Model selection: use SFT-trained model by default
MODEL_SIZE="${1:-4b}"
if [ -z "$MODEL_PATH" ]; then
    SFT_MODEL="$PROJECT_ROOT/outputs/local_training/qwen3-${MODEL_SIZE}-medprm-v2"
    if [ -d "$SFT_MODEL" ]; then
        MODEL_PATH="$SFT_MODEL"
        echo "Using SFT-trained model: $MODEL_PATH"
    else
        # Fallback to base model
        if [ "$MODEL_SIZE" == "8b" ]; then
            MODEL_PATH="Qwen/Qwen3-8B"
        else
            MODEL_PATH="Qwen/Qwen3-4B"
        fi
        echo "WARNING: SFT model not found at $SFT_MODEL, falling back to $MODEL_PATH"
    fi
fi

MODEL_SHORT=$(basename "$MODEL_PATH")
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="medserl_selfplay_${MODEL_SHORT}_${TIMESTAMP}"
OUTPUT_DIR="$PROJECT_ROOT/outputs/self_play/${MODEL_SHORT}_${TIMESTAMP}"
CONFIG_DIR="$PROJECT_ROOT/scripts/self_play/configs"
LOG_DIR="$PROJECT_ROOT/outputs/logs"
LOG_FILE="$LOG_DIR/selfplay_${TIMESTAMP}.log"
SESSION="selfplay"

# Configurable training size
MAX_PAIRS=${MAX_PAIRS:-50}

# Check if screen session already exists
if screen -list | grep -q "${SESSION}"; then
    echo "Screen session '${SESSION}' already exists!"
    echo "Attach:  screen -r ${SESSION}"
    echo "Kill:    screen -S ${SESSION} -X quit"
    exit 1
fi

mkdir -p "$OUTPUT_DIR" "$LOG_DIR" results/self_play

# The pipeline function
run_selfplay() {
    echo "=================================================="
    echo "MedSeRL Self-Play Training"
    echo "=================================================="
    echo "Project root: $PROJECT_ROOT"
    echo "Model: $MODEL_PATH"
    echo "Model size: $MODEL_SIZE"
    echo "Output: $OUTPUT_DIR"
    echo "Max pairs: $MAX_PAIRS"
    echo "Started: $(date)"
    echo "=================================================="

    cd "$PROJECT_ROOT"

    # Step 1: Preprocess data with v4 prompts (SFT-aligned)
    echo ""
    echo "=== Step 1: Preprocessing MEDEC data (v4 prompts) ==="
    python3 scripts/self_play/preprocess_medec.py \
        --input data_processed/medec_paired/train_val_split/rl_train.jsonl \
        --output data_processed/self_play/train.parquet \
        --injection-prompts configs/prompts/error_injection_prompts_v4.json \
        --max-pairs "$MAX_PAIRS"

    VAL_FILE="data_processed/self_play/val.parquet"
    if [ -f "data_processed/medec_paired/train_val_split/rl_val.jsonl" ]; then
        python3 scripts/self_play/preprocess_medec.py \
            --input data_processed/medec_paired/train_val_split/rl_val.jsonl \
            --output "$VAL_FILE" \
            --injection-prompts configs/prompts/error_injection_prompts_v4.json \
            --max-pairs 50
    else
        echo "Warning: No separate validation file, using training file"
        VAL_FILE="data_processed/self_play/train.parquet"
    fi

    # Step 2: Launch self-play training
    echo ""
    echo "=== Step 2: Starting Self-Play Training ==="
    echo "Turn 1: Injector modifies note"
    echo "Turn 2: Assessor classifies note"
    echo ""

    export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
    export SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK=True
    export TRANSFORMERS_ATTN_IMPLEMENTATION=sdpa

    python3 -m verl.trainer.main_ppo \
        --config-path="$CONFIG_DIR" \
        --config-name="ppo_multiturn" \
        algorithm.adv_estimator=reinforce_plus_plus \
        data.train_files="$PROJECT_ROOT/data_processed/self_play/train.parquet" \
        data.val_files="$PROJECT_ROOT/$VAL_FILE" \
        data.filter_overlong_prompts=True \
        data.truncation='error' \
        data.max_prompt_length=1024 \
        data.max_response_length=2048 \
        actor_rollout_ref.model.path="$MODEL_PATH" \
        +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
        actor_rollout_ref.model.use_remove_padding=False \
        actor_rollout_ref.model.enable_gradient_checkpointing=True \
        actor_rollout_ref.actor.optim.lr=1e-6 \
        actor_rollout_ref.actor.ppo_mini_batch_size=16 \
        actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
        actor_rollout_ref.actor.use_kl_loss=False \
        actor_rollout_ref.actor.entropy_coeff=0.01 \
        actor_rollout_ref.actor.fsdp_config.param_offload=False \
        actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
        actor_rollout_ref.actor.strategy=fsdp2 \
        actor_rollout_ref.rollout.name=sglang \
        actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
        actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
        actor_rollout_ref.rollout.temperature=0.7 \
        actor_rollout_ref.rollout.top_p=0.95 \
        actor_rollout_ref.rollout.top_k=20 \
        +actor_rollout_ref.rollout.repetition_penalty=1.1 \
        actor_rollout_ref.rollout.multi_turn.enable=True \
        actor_rollout_ref.rollout.multi_turn.max_user_turns=2 \
        actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
        actor_rollout_ref.rollout.multi_turn.interaction_config_path="$CONFIG_DIR/interaction_config.yaml" \
        actor_rollout_ref.rollout.multi_turn.use_inference_chat_template=True \
        actor_rollout_ref.ref.strategy=fsdp2 \
        actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
        actor_rollout_ref.ref.fsdp_config.param_offload=True \
        algorithm.use_kl_in_reward=True \
        algorithm.kl_ctrl.kl_coef=0.001 \
        reward_model.enable=False \
        trainer.critic_warmup=0 \
        trainer.logger=console \
        trainer.project_name='medserl-selfplay' \
        trainer.experiment_name="$EXPERIMENT_NAME" \
        trainer.default_local_dir="$OUTPUT_DIR" \
        trainer.n_gpus_per_node=1 \
        trainer.nnodes=1 \
        trainer.total_epochs=3 \
        trainer.save_freq=-1 \
        trainer.test_freq=10 \
        trainer.val_before_train=True \
        +trainer.enable_save=False \
        custom_reward_function.path="$PROJECT_ROOT/scripts/self_play/reward_function.py" \
        custom_reward_function.name=compute_score

    echo ""
    echo "=================================================="
    echo "Training Complete! $(date)"
    echo "=================================================="
    echo "Outputs: $OUTPUT_DIR"

    # Step 3: Analyze results
    echo ""
    echo "=== Step 3: Analyzing Training Results ==="
    if [ -d "results/self_play/interactions" ]; then
        python3 scripts/self_play/analyze_training.py \
            --log-dir results/self_play/interactions \
            --samples 3
    fi
    echo "=================================================="
}

# Write pipeline to temp script for screen
PIPELINE_SCRIPT=$(mktemp /tmp/selfplay_XXXXXX.sh)
cat > "$PIPELINE_SCRIPT" << HEREDOC
#!/bin/bash
set -e
cd $PROJECT_ROOT

# Activate verl conda environment (same as original run_training.sh)
CONDA_BASE=\${CONDA_BASE:-/workspace/miniconda3}
if [ -f "\$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "\$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate verl
    echo "✓ Activated conda env: verl"
else
    echo "WARNING: conda not found at \$CONDA_BASE, using current environment"
fi

# Restore CUDA environment (screen subshell doesn't inherit these)
export PATH="/usr/local/cuda/bin:\$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:\${LD_LIBRARY_PATH:-}"
# Ensure GPU 0 is visible (unset = all GPUs, empty string = no GPUs)
if [ -z "\${CUDA_VISIBLE_DEVICES+x}" ] || [ "\$CUDA_VISIBLE_DEVICES" = "" ]; then
    unset CUDA_VISIBLE_DEVICES
fi
echo "✓ CUDA env restored (LD_LIBRARY_PATH, PATH)"

export MODEL_PATH="$MODEL_PATH"
export MODEL_SIZE="$MODEL_SIZE"
export MODEL_SHORT="$MODEL_SHORT"
export TIMESTAMP="$TIMESTAMP"
export EXPERIMENT_NAME="$EXPERIMENT_NAME"
export OUTPUT_DIR="$OUTPUT_DIR"
export CONFIG_DIR="$CONFIG_DIR"
export MAX_PAIRS="$MAX_PAIRS"
export PROJECT_ROOT="$PROJECT_ROOT"

$(declare -f run_selfplay)
run_selfplay 2>&1 | tee "$LOG_FILE"
HEREDOC

echo "Launching self-play in screen session '${SESSION}'..."
echo "Log: $LOG_FILE"
echo "Model: $MODEL_PATH"
echo ""
screen -dmS "$SESSION" bash "$PIPELINE_SCRIPT"

echo "Screen session started."
echo ""
echo "Attach:   screen -r $SESSION"
echo "Detach:   Ctrl-A then D"
echo "Monitor:  tail -f $LOG_FILE"
echo "Kill:     screen -S $SESSION -X quit"
