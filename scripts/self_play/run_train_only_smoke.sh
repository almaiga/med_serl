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
#   - Thinking budget: ~2048 tokens, Answer: ~1024 tokens (total: 3072)
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

# ── GPU visibility — pin GPU 0 and prevent Ray from clearing it in workers ──
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
TRAIN_PARQUET="$PROJECT_ROOT/data_processed/self_play/train.parquet"
VAL_PARQUET="$PROJECT_ROOT/data_processed/self_play/val.parquet"
SMOKE_LOG="$OUTPUT_DIR/smoke_test.log"

mkdir -p "$OUTPUT_DIR"
mkdir -p "$PROJECT_ROOT/results/self_play"

export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

# ─── Cleanup: kill stale GPU / Ray processes ──────────────────────────────────
echo "=== Cleanup ==="

# Kill RunPod's built-in vLLM API server (auto-starts on boot, hogs ~90% GPU)
pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
pkill -9 -f "vllm.worker" 2>/dev/null || true
sleep 2

# Kill stale GPU processes — use SIGTERM first, then SIGKILL only if needed.
# Immediate SIGKILL can corrupt the CUDA runtime context inside Docker.
if command -v nvidia-smi &>/dev/null; then
    GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ' | sort -u)
    if [ -n "$GPU_PIDS" ]; then
        echo "  Sending SIGTERM to stale GPU processes: $GPU_PIDS"
        for pid in $GPU_PIDS; do
            kill "$pid" 2>/dev/null || true
        done
        sleep 5
        # Only SIGKILL stragglers
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
    echo ""
    echo "  ERROR: CUDA runtime cannot initialise (driver is fine, runtime is not)."
    echo "  This usually happens after kill -9 of GPU processes in Docker."
    echo "  Fix: restart the RunPod pod, then re-run this script."
    exit 1
fi

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
# NOTE: Aggressive cleanup (ray stop, GPU kill) is disabled — it kills RunPod SSH.
# If you need to free GPU memory manually, run: nvidia-smi and kill specific PIDs.
cleanup() {
    echo ""
    echo "Script finished. To free GPU memory, run: nvidia-smi then kill <pid>"
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

# Export run-specific game log path so ALL Ray workers write to the same file
GAME_LOG="$PROJECT_ROOT/$OUTPUT_DIR/game_interactions.jsonl"
export MEDSERL_GAME_LOG="$GAME_LOG"

# Final Ray state cleanup right before launch
# NOTE: Only clean workspace ray_tmp — do NOT touch /dev/shm or /tmp (kills RunPod SSH)
rm -rf "$RAY_TMPDIR_PATH"/* 2>/dev/null || true
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
    --config-name="ppo_multiturn" \
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
    data.max_response_length=3072 \
    data.filter_overlong_prompts=False \
    data.truncation=error \
    data.return_raw_chat=True \
    \
    actor_rollout_ref.model.path="$ACTOR_MODEL" \
    "++actor_rollout_ref.model.override_config.attn_implementation=sdpa" \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=0.85 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.40 \
    actor_rollout_ref.rollout.max_model_len=4096 \
    actor_rollout_ref.rollout.max_num_batched_tokens=5120 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.prompt_length=1024 \
    actor_rollout_ref.rollout.response_length=3072 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=2 \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=2 \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path="$CONFIG_DIR/interaction_config.yaml" \
    \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    \
    critic.enable=false \
    \
    reward_model.enable=False \
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
    custom_reward_function.path="$PROJECT_ROOT/scripts/self_play/agentic_reward.py" \
    custom_reward_function.name="compute_score" \
    2>&1 | tee "$SMOKE_LOG"

TRAIN_EXIT=${PIPESTATUS[0]}

# ─── Step 2: Verification ────────────────────────────────────────────────────
echo ""
echo "=================================================="
echo "=== Step 2: Verification ==="
echo "=================================================="

TURN2=$(grep -c "num_turns" "$SMOKE_LOG" 2>/dev/null || true); TURN2=${TURN2:-0}

echo ""
echo "num_turns lines in log : $TURN2  (>0 confirms multi-turn ran)"
echo "Training exit code     : $TRAIN_EXIT"
echo ""

if [ "$TRAIN_EXIT" -eq 0 ] && [ "$TURN2" -gt 0 ]; then
    echo "SMOKE TEST PASSED: Multi-turn training completed. Reward from interaction."
elif [ "$TRAIN_EXIT" -eq 0 ]; then
    echo "SMOKE TEST PARTIAL: Training completed but num_turns not found in log."
    echo "Check for 'num_turns/mean:2.0' in: $SMOKE_LOG"
else
    echo "SMOKE TEST FAILED: Training exited with code $TRAIN_EXIT."
    echo "Check the log for errors: $SMOKE_LOG"
fi

echo ""
echo "Full log: $SMOKE_LOG"
echo "Outputs : $OUTPUT_DIR"

# ─── Step 3: Response Quality Analysis ───────────────────────────────────────
echo ""
echo "=================================================="
echo "=== Step 3: Response Quality Analysis ==="
echo "=================================================="

python3 << PYEOF
import json, re, sys, os
from collections import defaultdict
from pathlib import Path

# Records are written by agentic_reward.py to results/self_play/interactions/
TRACE_DIR = Path("$PROJECT_ROOT/results/self_play/interactions")
SMOKE_LOG  = Path("$SMOKE_LOG")

# Find the interaction's JSONL for this run.
# Primary: run-specific path set via MEDSERL_GAME_LOG env var → game_interactions.jsonl in OUTPUT_DIR
# Fallback: game_*.jsonl in TRACE_DIR, then oldest interactions_*.jsonl in 5-min window
GAME_LOG = Path("$PROJECT_ROOT/$OUTPUT_DIR/game_interactions.jsonl")

if GAME_LOG.exists() and GAME_LOG.stat().st_size > 0:
    src = GAME_LOG
    label = "game (interaction, run-specific)"
else:
    if TRACE_DIR.exists():
        game_files = sorted(TRACE_DIR.glob("game_*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
        all_files  = sorted(TRACE_DIR.glob("*.jsonl"),      key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        game_files, all_files = [], []
    if game_files:
        src = game_files[0]; label = "game (interaction)"
    elif all_files:
        latest = all_files[0].stat().st_mtime
        run_files = [f for f in all_files if latest - f.stat().st_mtime < 300]
        src = run_files[-1]; label = "interactions (oldest-in-run fallback)"
    else:
        src = None; label = None

if src is None:
    print("  No JSONL found — checking smoke log for inline records ...")
    src = SMOKE_LOG
else:
    try:
        rel = src.relative_to(Path("$PROJECT_ROOT"))
    except ValueError:
        rel = src.name
    print(f"  Reading ({label}): {rel}")

records = []
for line in src.read_text().splitlines():
    if line.startswith("{") and '"timestamp"' in line:
        try:
            records.append(json.loads(line))
        except Exception:
            pass

if not records:
    print("  No JSONL records found — model may not have produced any rewards yet")
    sys.exit(0)

print(f"  Total records : {len(records)}")

def strip_thinking(text):
    m = re.search(r"<think>(.*?)</think>\s*", text, re.DOTALL)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return "", text

def tokens(text):
    return int(len(text) / 1.4)  # ~1.4 chars/token for medical text

# Per-mode breakdown
by_mode = defaultdict(list)
for r in records:
    by_mode[r.get("mode", "?")].append(r)
for mode, recs in sorted(by_mode.items()):
    print(f"  Mode '{mode}': {len(recs)} records")

# Multi-turn health check
# assessor_actually_ran is set by reward_function.py after content-based turn split
assessor_ran = sum(1 for r in records if r.get("assessor_actually_ran", False))
phases_sep   = sum(1 for r in records if r.get("phases_separated", False))
# Fallback: if split was not possible, infer from outcome being non-trivial
# (the reward computation itself works even without the split)
outcomes_found = sum(1 for r in records if r.get("outcome") not in (None, "invalid_format"))
print(f"  Assessor ran (split)  : {assessor_ran}/{len(records)}")
print(f"  Phases separated      : {phases_sep}/{len(records)}")
print(f"  Valid reward outcomes : {outcomes_found}/{len(records)}")
if assessor_ran == 0 and outcomes_found == 0:
    print("  WARNING: No assessor responses AND no valid outcomes — multi-turn may be broken")
elif assessor_ran == 0:
    print("  NOTE: Turn split not found in solution_str (verl strips chat tokens).")
    print("        Rewards computed correctly from full response. Use model_response_full for debug.")

# Analyze full response (injector_response contains full solution_str when split fails)
inj_stats = []
for r in records:
    resp = r.get("injector_response", "") or r.get("model_response_full", "")
    if not resp:
        continue
    has_open  = "<think>" in resp
    has_close = "</think>" in resp
    # Strict truncation: opened <think> but never closed it
    truncated = has_open and not has_close
    think_count = resp.count("</think>")
    thinking_part, answer = strip_thinking(resp)
    inj_stats.append({
        "total_tok":    tokens(resp),
        "think_tok":    tokens(thinking_part),
        "answer_tok":   tokens(answer),
        "truncated":    truncated,
        "think_count":  think_count,
        "answer_empty": len(answer.strip()) == 0,
    })

# Analyze assessor responses if the split worked
assess_stats = []
for r in records:
    resp = r.get("assessor_response", "")
    if not resp:
        continue
    thinking, answer = strip_thinking(resp)
    assess_stats.append({
        "total_tok":    tokens(resp),
        "think_tok":    tokens(thinking),
        "answer_tok":   tokens(answer),
        "has_think":    "<think>" in resp,
        "answer":       answer.strip(),
    })

if inj_stats:
    n = len(inj_stats)
    avg_tot   = sum(s["total_tok"]  for s in inj_stats) / n
    avg_think = sum(s["think_tok"]  for s in inj_stats) / n
    avg_ans   = sum(s["answer_tok"] for s in inj_stats) / n
    max_tot   = max(s["total_tok"]  for s in inj_stats)
    truncated = sum(1 for s in inj_stats if s["truncated"])
    no_answer = sum(1 for s in inj_stats if s["answer_empty"])
    over_bud  = sum(1 for s in inj_stats if s["total_tok"] > 3072)
    two_turns = sum(1 for s in inj_stats if s["think_count"] >= 2)

    print(f"""
  ── Full response (injector + assessor, or assessor-only) ──
  Samples           : {n}
  Tokens  avg/max   : {avg_tot:,.0f} / {max_tot:,}  (budget: 3072)
    Thinking avg    : {avg_think:,.0f}
    Answer   avg    : {avg_ans:,.0f}
  Over budget       : {over_bud}/{n}
  Inj truncated (no </think>) : {truncated}/{n}  {'WARNING ' if truncated > n//4 else 'OK'}
  Two </think> blocks (2 turns visible) : {two_turns}/{n}""")

if assess_stats:
    n = len(assess_stats)
    avg_tot   = sum(s["total_tok"]  for s in assess_stats) / n
    avg_think = sum(s["think_tok"]  for s in assess_stats) / n
    avg_ans   = sum(s["answer_tok"] for s in assess_stats) / n
    max_tot   = max(s["total_tok"]  for s in assess_stats)

    # assessor may use /no_think so "answer" = the full stripped response
    correct_answers = sum(1 for s in assess_stats if re.search(r'\bCORRECT\b', s["answer"], re.IGNORECASE))
    numeric_answers = sum(1 for s in assess_stats if re.search(r'^\s*\d+\s*$', s["answer"]))
    empty_answers   = sum(1 for s in assess_stats if not s["answer"].strip())

    print(f"""
  ── Assessor (Phase 2 split succeeded) ────────────────
  Samples           : {n}
  Tokens  avg/max   : {avg_tot:,.0f} / {max_tot:,}
    Thinking avg    : {avg_think:,.0f}
    Answer   avg    : {avg_ans:,.0f}
  Answer patterns:
    'CORRECT'       : {correct_answers}/{n}
    Sentence nums   : {numeric_answers}/{n}
    Empty           : {empty_answers}/{n}""")

# Outcome breakdown (always available — computed from full solution_str)
from collections import Counter
outcome_counts = Counter(r.get("outcome", "unknown") for r in records)
rewards = [r.get("reward", 0.0) for r in records]
avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
invalid = sum(1 for r in records if not r.get("has_valid_format", True))

print(f"\n  ── Reward outcomes ──────────────────────────────────")
for oc, cnt in sorted(outcome_counts.items(), key=lambda x: -x[1]):
    print(f"  {oc:20s}: {cnt}")
print(f"  Avg reward        : {avg_reward:.3f}")
print(f"  Invalid format    : {invalid}/{len(records)}")

# Training metrics from smoke log
import subprocess
try:
    grep_out = subprocess.run(
        ["grep", "-oP", r"critic/score/mean:[0-9e.+\-]+", str(SMOKE_LOG)],
        capture_output=True, text=True
    )
    scores = [float(x.split(":")[1]) for x in grep_out.stdout.strip().split("\n") if x]
    if scores:
        print(f"\n  Training critic/score/mean : {scores[0]:.4f} (step 1) → {scores[-1]:.4f} (last step)")
except Exception:
    pass

# Sample responses (first 3 records with injector_response or assessor_response)
sample_recs = [r for r in records if r.get("injector_response") or r.get("assessor_response")][:3]
if sample_recs:
    print(f"\n  ── Sample game records ──────────────────────────────")
    for i, r in enumerate(sample_recs):
        gt  = r.get("ground_truth", "?")
        oc  = r.get("outcome", "?")
        rwd = r.get("reward", 0.0)
        inj = (r.get("injector_response") or "")[:150].replace("\n", " ↵ ")
        asm = (r.get("assessor_response") or "")[:80].replace("\n", " ↵ ")
        print(f"  [{i+1}] gt={gt!r:8} outcome={oc:15} reward={rwd:+.2f}")
        if inj: print(f"       INJ: {inj!r}")
        if asm: print(f"       ASM: {asm!r}")
print()
PYEOF

exit $TRAIN_EXIT