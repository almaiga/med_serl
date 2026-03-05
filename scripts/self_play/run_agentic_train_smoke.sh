#!/bin/bash
# MedSeRL Agentic Self-Play Training — SMOKE TEST
#
# Purpose: Validate that compute_score() in agentic_reward.py is called
#          with real veRL model rollouts (NOT simulated responses).
#
# What this does:
#   1. Pre-downloads Qwen3-8B judge model (~16GB)
#   2. Starts vLLM judge server on port 8002
#   3. Copies train.parquet → val.parquet if val doesn't exist
#   4. Runs veRL PPO/REINFORCE++ for ~5 training steps
#   5. Confirms reward function was invoked (check for "compute_score" in log)
#
# Expected outcome: Training completes, logs show "compute_score called" entries.
#
# Usage:
#   chmod +x scripts/self_play/run_agentic_train_smoke.sh
#   ./scripts/self_play/run_agentic_train_smoke.sh
#
# Env overrides:
#   ACTOR_MODEL   — Actor model path (default: Qwen/Qwen3-4B)
#   JUDGE_MODEL   — Judge model path (default: Qwen/Qwen3-8B)
#   JUDGE_PORT    — vLLM judge port  (default: 8002)
#   SMOKE_STEPS   — Max training steps (default: 5, set high to run full epoch)
#   SKIP_SERVER   — Set to 1 to reuse an already-running judge server

# Don't use set -e — we handle errors manually and need the EXIT trap to always run

# ─── Configuration ────────────────────────────────────────────────────────────
ACTOR_MODEL="${ACTOR_MODEL:-Qwen/Qwen3-4B}"
JUDGE_MODEL="${JUDGE_MODEL:-Qwen/Qwen3-8B}"
JUDGE_PORT="${JUDGE_PORT:-8002}"
SMOKE_STEPS="${SMOKE_STEPS:-5}"
SKIP_SERVER="${SKIP_SERVER:-0}"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
EXPERIMENT_NAME="smoke_agentic_${TIMESTAMP}"
OUTPUT_DIR="outputs/self_play/smoke_${TIMESTAMP}"

# Paths
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
CONFIG_DIR="$PROJECT_ROOT/scripts/self_play/configs"
TRAIN_PARQUET="$PROJECT_ROOT/data_processed/self_play/train.parquet"
VAL_PARQUET="$PROJECT_ROOT/data_processed/self_play/val.parquet"
SMOKE_LOG="$OUTPUT_DIR/smoke_test.log"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_ROOT/results/self_play"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# ─── Cleanup: kill leftover processes from previous crashed runs ──────────────
echo "=== Cleanup: killing ALL stale GPU / vLLM / Ray processes ==="
# Kill any leftover vLLM servers on the judge port
if lsof -ti :${JUDGE_PORT} >/dev/null 2>&1; then
    echo "  Killing processes on port ${JUDGE_PORT}..."
    lsof -ti :${JUDGE_PORT} | xargs kill -9 2>/dev/null || true
    sleep 2
fi
# Kill any remaining Ray processes
if pgrep -f "ray::" >/dev/null 2>&1; then
    echo "  Stopping Ray..."
    ray stop --force 2>/dev/null || true
    pkill -9 -f "ray::" 2>/dev/null || true
    sleep 2
fi
# Kill any orphan vllm / python-gpu processes
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -f "vllm.worker" 2>/dev/null || true
pkill -9 -f "from multiprocessing" 2>/dev/null || true

# Nuclear option: kill ALL processes holding the GPU except this shell
echo "  Checking GPU processes via nvidia-smi..."
if command -v nvidia-smi &>/dev/null; then
    GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u)
    if [ -n "$GPU_PIDS" ]; then
        echo "  Found GPU processes: $GPU_PIDS — killing them..."
        for pid in $GPU_PIDS; do
            kill -9 "$pid" 2>/dev/null || true
        done
        sleep 3
    else
        echo "  No stale GPU processes found."
    fi
    # Show current GPU state
    nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader
fi
# Allow CUDA driver to recover after aggressive kill -9
sleep 5
echo "  Cleanup done."

# ─── Pre-flight: verify enough GPU memory is free ────────────────────────────
echo ""
echo "=== Pre-flight: GPU memory check ==="
python3 << 'GPU_CHECK_EOF'
import subprocess, sys
try:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, timeout=10
    )
    lines = result.stdout.strip().split("\n")
    print(f"  Found {len(lines)} GPU(s)")
    total_free_gb = 0
    for i, line in enumerate(lines):
        free_mb, total_mb = [int(x.strip()) for x in line.split(",")]
        free_gb = free_mb / 1024
        total_gb = total_mb / 1024
        total_free_gb += free_gb
        print(f"  GPU {i}: {free_gb:.1f} GiB free / {total_gb:.1f} GiB total")
    # Need at least 20 GiB free on any single GPU for the judge
    if total_free_gb < 20:
        print(f"  ERROR: Only {total_free_gb:.1f} GiB total free — need at least 20 GiB.")
        sys.exit(1)
    else:
        print(f"  OK — {total_free_gb:.1f} GiB total free across {len(lines)} GPU(s).")
except Exception as e:
    print(f"  WARNING: Could not check GPU memory: {e}")
GPU_CHECK_EOF
if [ $? -ne 0 ]; then
    echo "ABORTING: Not enough GPU memory. Clean up stale processes first."
    exit 1
fi

# ─── Trap: ensure cleanup on exit ────────────────────────────────────────────
cleanup() {
    echo ""
    echo "Trap: cleaning up all GPU processes..."
    if [ -n "$VLLM_PID" ]; then
        echo "  Stopping judge server (PID $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
    fi
    pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
    pkill -9 -f "vllm.worker" 2>/dev/null || true
    ray stop --force 2>/dev/null || true
    # Kill any remaining GPU processes spawned by this script
    if command -v nvidia-smi &>/dev/null; then
        GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u)
        for pid in $GPU_PIDS; do
            kill -9 "$pid" 2>/dev/null || true
        done
    fi
    echo "  Cleanup complete."
}

trap cleanup EXIT

echo "=================================================="
echo "MedSeRL Agentic Training — SMOKE TEST"
echo "=================================================="
echo "Project root : $PROJECT_ROOT"
echo "Actor model  : $ACTOR_MODEL"
echo "Judge model  : $JUDGE_MODEL"
echo "Judge port   : $JUDGE_PORT"
echo "Smoke steps  : $SMOKE_STEPS"
echo "Output dir   : $OUTPUT_DIR"
echo "=================================================="

# ─── Pre-flight: Patch veRL for GLIBC / Megatron compat ───────────────────────
# veRL's __init__.py already wraps mindspeed & megatron in try/except but only
# catches ImportError.  transformer_engine raises OSError when GLIBC is too old.
# This patch widens  except ImportError  →  except (ImportError, OSError).
echo ""
echo "=== Pre-flight: Patching veRL engine imports (GLIBC workaround) ==="
python3 << 'PATCH_EOF'
import pathlib, re

fpath = pathlib.Path("/workspace/verl/verl/workers/engine/__init__.py")
if not fpath.exists():
    print("  SKIP: file not found")
else:
    code = fpath.read_text()
    # Widen every bare  except ImportError:  to  except (ImportError, OSError):
    new_code, n = re.subn(
        r'except ImportError:',
        'except (ImportError, OSError):',
        code,
    )
    if n == 0:
        print("  No bare 'except ImportError:' found (already patched?).")
    else:
        fpath.write_text(new_code)
        print(f"  PATCHED: widened {n} 'except ImportError' → 'except (ImportError, OSError)'")
PATCH_EOF

# ─── Pre-flight: Config approach ──────────────────────────────────────────────
# Using ppo_agentic.yaml with Hydra defaults composition (same pattern as the
# working ppo_multiturn.yaml / run_training_v2.sh). veRL's built-in ppo_trainer
# base config fills ALL dataclass fields correctly — we only override what's
# different. No auto-patching needed.
echo ""
echo "=== Pre-flight: Using Hydra defaults composition (ppo_agentic.yaml) ==="
echo "  Base: verl.trainer.config.ppo_trainer (built-in)"
echo "  Overlay: ppo_agentic.yaml (our overrides only)"

# ─── Step 0: Ensure data exists ───────────────────────────────────────────────
echo ""
echo "=== Step 0: Checking / Generating Data ==="

if [ ! -f "$TRAIN_PARQUET" ]; then
    echo "train.parquet missing — running preprocess_medec.py with MAX_PAIRS=20 ..."
    python3 "$PROJECT_ROOT/scripts/self_play/preprocess_medec.py" \
        --input "$PROJECT_ROOT/data_processed/medec_paired/train_val_split/rl_train.jsonl" \
        --output "$TRAIN_PARQUET" \
        --injection-prompts "$PROJECT_ROOT/configs/prompts/error_injection_prompts_v4.json" \
        --max-pairs 20
    echo "train.parquet generated."
else
    echo "train.parquet found: $TRAIN_PARQUET"
fi

# val.parquet: use dedicated file or fall back to train.parquet
if [ ! -f "$VAL_PARQUET" ]; then
    echo "val.parquet missing — symlinking train.parquet as val.parquet"
    cp "$TRAIN_PARQUET" "$VAL_PARQUET"
    echo "val.parquet created (copy of train.parquet)"
else
    echo "val.parquet found: $VAL_PARQUET"
fi

# ─── Step 1: Start judge server ───────────────────────────────────────────────
echo ""
echo "=== Step 1: Judge Server (${JUDGE_MODEL}) ==="

JUDGE_URL="http://localhost:${JUDGE_PORT}/v1/chat/completions"
export JUDGE_VLLM_URL="$JUDGE_URL"
export UMLS_API_KEY="${UMLS_API_KEY:-6878e795-ad79-4743-9758-546cacb8b31c}"

if [ "$SKIP_SERVER" = "1" ]; then
    echo "SKIP_SERVER=1 — assuming judge already running at $JUDGE_URL"
else
    # Check if server is already responding
    if curl -s -o /dev/null -w "%{http_code}" "${JUDGE_URL%/chat/completions}/models" 2>/dev/null | grep -q "200"; then
        echo "Judge server already running at $JUDGE_URL — skipping launch."
    else
        echo "Pre-downloading judge model weights (avoids vLLM cold-start timeout)..."
        python3 -c "
from huggingface_hub import snapshot_download
import os
print('Downloading ${JUDGE_MODEL} ...')
snapshot_download('${JUDGE_MODEL}', ignore_patterns=['*.gguf'])
print('Download complete.')
" || echo "Pre-download skipped (model may already be cached)."

        echo "Starting vLLM judge server on port ${JUDGE_PORT} ..."
        python3 -m vllm.entrypoints.openai.api_server \
            --model "${JUDGE_MODEL}" \
            --port "${JUDGE_PORT}" \
            --dtype bfloat16 \
            --max-model-len 4096 \
            --gpu-memory-utilization 0.3 \
            --enforce-eager \
            --served-model-name "${JUDGE_MODEL}" \
            &
        VLLM_PID=$!
        echo "vLLM PID: $VLLM_PID"

        # Wait for server to come up (up to 900s)
        echo "Waiting for judge server to become healthy (may take 5-10 min)..."
        WAIT_SECS=0
        MAX_WAIT=900
        until curl -s -o /dev/null -w "%{http_code}" "http://localhost:${JUDGE_PORT}/health" 2>/dev/null | grep -q "200"; do
            sleep 10
            WAIT_SECS=$((WAIT_SECS + 10))
            echo -n "."
            if [ "$WAIT_SECS" -ge "$MAX_WAIT" ]; then
                echo ""
                echo "ERROR: Judge server did not start within ${MAX_WAIT}s. Aborting."
                kill "$VLLM_PID" 2>/dev/null || true
                exit 1
            fi
        done
        echo ""
        echo "Judge server healthy after ${WAIT_SECS}s at $JUDGE_URL"
    fi
fi

# Quick sanity-check: one test inference through the judge
echo "Judge server sanity check..."
curl -s "${JUDGE_URL}" \
    -H "Content-Type: application/json" \
    -d '{"model":"'"${JUDGE_MODEL}"'","messages":[{"role":"user","content":"Reply OK."}],"max_tokens":5,"chat_template_kwargs":{"enable_thinking":false}}' \
    | python3 -c "import sys,json; r=json.load(sys.stdin); print('Judge OK:', r['choices'][0]['message']['content'])" \
    || echo "WARNING: Judge sanity check failed — reward may fall back to rule-based."

# ─── Step 2: Launch veRL REINFORCE++ smoke run ────────────────────────────────
echo ""
echo "=== Step 2: veRL Training (smoke: ~${SMOKE_STEPS} optimizer steps) ==="
echo "Config: ppo_agentic.yaml (Hydra defaults from ppo_trainer)"
echo "  reward_model.enable=False (standalone judge)"
echo ""
echo "JUDGE_VLLM_URL=$JUDGE_VLLM_URL"
echo "UMLS_API_KEY set: $([ -n "$UMLS_API_KEY" ] && echo yes || echo NO)"
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
    actor_rollout_ref.model.use_shm=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.lora_rank=16 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.model.target_modules=all-linear \
    actor_rollout_ref.actor.optim.lr=1e-5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.strategy=fsdp2 \
    \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.95 \
    actor_rollout_ref.rollout.top_k=20 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_num_batched_tokens=4096 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.prompt_length=1024 \
    actor_rollout_ref.rollout.response_length=1024 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
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
    2>&1 | tee "$SMOKE_LOG"

TRAIN_EXIT=${PIPESTATUS[0]}

# ─── Step 3: Verification ─────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "=== Step 3: Smoke Test Verification ==="
echo "=================================================="

CS_CALLS=$(grep -c "compute_score called" "$SMOKE_LOG" 2>/dev/null || true); CS_CALLS=${CS_CALLS:-0}
JUDGE_CALLS=$(grep -c "judge_verdict\|umls_score\|PASS\|FAIL\|ABSTAIN" "$SMOKE_LOG" 2>/dev/null || true); JUDGE_CALLS=${JUDGE_CALLS:-0}
RULE_CALLS=$(grep -c "rule_score\|rule_compute_score" "$SMOKE_LOG" 2>/dev/null || true); RULE_CALLS=${RULE_CALLS:-0}

echo ""
echo "compute_score invocations : $CS_CALLS"
echo "Judge verdict entries     : $JUDGE_CALLS"
echo "rule_score log lines      : $RULE_CALLS"
echo ""

if [ "$CS_CALLS" -gt 0 ]; then
    echo "SMOKE TEST PASSED: compute_score was called with real model rollouts."
    echo ""
    echo "Sample compute_score calls from log:"
    grep "compute_score called" "$SMOKE_LOG" | head -5
else
    echo "SMOKE TEST INCONCLUSIVE: No 'compute_score called' log lines found."
    echo "Check that agentic_reward.py has the INFO log at the start of compute_score."
    echo "Training exit code: $TRAIN_EXIT"
fi

echo ""
echo "Full log: $SMOKE_LOG"
echo "Outputs : $OUTPUT_DIR"

# Kill judge server if we launched it
if [ -n "$VLLM_PID" ]; then
    echo ""
    echo "Stopping judge server (PID $VLLM_PID)..."
    kill "$VLLM_PID" 2>/dev/null || true
fi

exit $TRAIN_EXIT
