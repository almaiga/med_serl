#!/bin/bash
# MedSeRL True Self-Play Training Script
# Two-phase game with multi-turn enabled:
# Turn 1: Injector modifies note → Turn 2: Assessor classifies
#
# This script properly activates the conda environment and sets PYTHONPATH
# so that verl can import MedicalGameInteraction from scripts.self_play.interactions

set -e

# Configuration
OUTPUT_DIR="outputs/self_play_multiturn"
EXPERIMENT_NAME="medserl_selfplay_multiturn"
MODEL_PATH="Qwen/Qwen3-4B"

# Detect environment (runpod vs local)
if [ -d "/workspace/med_serl" ]; then
    PROJECT_ROOT="/workspace/med_serl"
    CONDA_BASE="/workspace/miniconda3"
else
    PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
    CONDA_BASE="${HOME}/miniconda3"
fi

CONFIG_DIR="$PROJECT_ROOT/scripts/self_play/configs"
DATA_DIR="$PROJECT_ROOT/data_processed/self_play"

# Activate conda environment
echo "=================================================="
echo "MedSeRL True Self-Play Training (Multi-Turn)"
echo "=================================================="
echo "Project root: $PROJECT_ROOT"
echo "Model: $MODEL_PATH"
echo "Output: $OUTPUT_DIR"
echo "=================================================="

# Source conda
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate med_serl
    echo "✓ Conda environment 'med_serl' activated"
elif [ -f "$PROJECT_ROOT/med_serl/bin/activate" ]; then
    source "$PROJECT_ROOT/med_serl/bin/activate"
    echo "✓ Virtual environment activated"
fi

# Set PYTHONPATH so verl can import MedicalGameInteraction
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
echo "✓ PYTHONPATH set to include: $PROJECT_ROOT"

# Change to project directory
cd "$PROJECT_ROOT"

# Create output directories
mkdir -p "$OUTPUT_DIR"
mkdir -p results/self_play/interactions

# Step 1: Regenerate training data with updated prompts (no answer leakage)
echo ""
echo "=== Step 1: Preprocessing MEDEC data with updated prompts ==="
echo "Prompts no longer include final_answer (Injector outputs only generated_note)"

python3 scripts/self_play/preprocess_medec.py \
    --input data_processed/medec_paired/train_val_split/rl_train.jsonl \
    --output "$DATA_DIR/train.parquet" \
    --injection-prompts configs/prompts/error_injection_prompts_v2.json \
    --max-pairs 50

# Validation data
VAL_FILE="$DATA_DIR/val.parquet"
if [ -f "data_processed/medec_paired/train_val_split/rl_val.jsonl" ]; then
    python3 scripts/self_play/preprocess_medec.py \
        --input data_processed/medec_paired/train_val_split/rl_val.jsonl \
        --output "$VAL_FILE" \
        --injection-prompts configs/prompts/error_injection_prompts_v2.json \
        --max-pairs 50
else
    echo "Warning: No separate validation file, using training file"
    VAL_FILE="$DATA_DIR/train.parquet"
fi

# Verify data
echo ""
echo "=== Verifying training data format ==="
python3 -c "
import pyarrow.parquet as pq
table = pq.read_table('$DATA_DIR/train.parquet')
df = table.to_pandas()
print(f'Total examples: {len(df)}')
# Check prompts don't contain final_answer in instructions
for i in range(min(2, len(df))):
    prompt = df.iloc[i]['prompt']
    if isinstance(prompt, list):
        content = ' '.join(m.get('content','') for m in prompt)
    else:
        content = str(prompt)
    has_final_in_template = 'final_answer:' in content and 'OUTPUT' in content.upper()
    mode = df.iloc[i]['extra_info'].get('mode', '')
    print(f'  [{mode}] Template contains final_answer: {has_final_in_template}')
print('✓ Data preprocessing complete')
"

# Step 2: Launch multi-turn training with verl
echo ""
echo "=== Step 2: Starting Multi-Turn Self-Play Training ==="
echo "Turn 1: Model as Injector (generates modified note)"
echo "Turn 2: Model as Assessor (classifies the note)"
echo "MedicalGameInteraction orchestrates the game"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    data.train_files="$DATA_DIR/train.parquet" \
    data.val_files="$VAL_FILE" \
    data.train_batch_size=16 \
    data.max_prompt_length=1024 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=8 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ppo_epochs=2 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=2 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$CONFIG_DIR/interaction_config.yaml" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    critic.optim.lr=1e-5 \
    critic.model.path=$MODEL_PATH \
    critic.ppo_micro_batch_size_per_gpu=2 \
    critic.ppo_mini_batch_size=8 \
    algorithm.gamma=1.0 \
    algorithm.lam=0.95 \
    algorithm.use_kl_in_reward=True \
    algorithm.kl_ctrl.kl_coef=0.001 \
    algorithm.kl_ctrl.type=fixed \
    reward_model.enable=False \
    custom_reward_function.path="$PROJECT_ROOT/scripts/self_play/reward_function.py" \
    custom_reward_function.name=compute_score \
    trainer.logger=console \
    trainer.project_name='medserl-selfplay' \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.default_local_dir=$OUTPUT_DIR \
    trainer.n_gpus_per_node=1 \
    trainer.nnodes=1 \
    trainer.total_epochs=3 \
    trainer.save_freq=50 \
    trainer.test_freq=10 \
    trainer.val_before_train=True

echo ""
echo "=================================================="
echo "Training Complete!"
echo "=================================================="
echo "Outputs: $OUTPUT_DIR"
echo "Logs: results/self_play/interactions/"

# Step 3: Analyze results
echo ""
echo "=== Step 3: Analyzing Training Results ==="
if [ -d "results/self_play/interactions" ]; then
    python3 scripts/self_play/analyze_training.py \
        --log-dir results/self_play/interactions \
        --samples 3
fi

echo "=================================================="
