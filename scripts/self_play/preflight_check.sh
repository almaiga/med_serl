#!/usr/bin/env bash
# preflight_check.sh
#
# Comprehensive offline pre-flight for self-play training.
#
# Purpose: prevent burning money on a multi-GPU launch that will fail on a
# wiring bug we could have caught locally. Every failure we have hit so far —
# the Qwen3-thinking judge bug, the JUDGE_VLLM_URL=/chat/completions doubling,
# the env var dropping out of the shell — would have been caught here.
#
# This script:
#   1) Tier 1 — pure offline: reward constants, reward semantics, prompt JSON,
#                judge_client.py source check, EV math.
#   2) Tier 2 — judge server: HTTP ping, end-to-end verdict parse, no <think>.
#   3) Tier 3 — artefacts: actor model on HF, dataset files, output dir
#                writable.
#   4) Tier 4 — env vars: JUDGE_VLLM_URL, ACTOR_MODEL, JUDGE_TYPE,
#                JUDGE_PROMPT_STYLE, KL_COEF, ROLLOUT_GPU_MEMORY_UTILIZATION,
#                RAM headroom, GPU free.
#
# Exit 0 iff every gate is green.
# Exit 1 with a numbered list of failures otherwise.
#
# Usage:
#   bash scripts/self_play/preflight_check.sh
#
# Required env (set BEFORE running):
#   JUDGE_VLLM_URL    e.g. http://judge-host:8000/v1  (with or without /chat/completions)
#
# Optional env (defaults shown):
#   ACTOR_MODEL=Abdine/qwen3-4b-medrect-mixed-v2
#   JUDGE_MODEL=pfnet/Preferred-MedRECT-32B
#   JUDGE_TYPE=detection
#   JUDGE_PROMPT_STYLE=hint_v2
#   KL_COEF=0.01
#   ROLLOUT_GPU_MEMORY_UTILIZATION=0.5

set -uo pipefail

ACTOR_MODEL="${ACTOR_MODEL:-Abdine/qwen3-4b-medrect-mixed-v2}"
JUDGE_MODEL="${JUDGE_MODEL:-pfnet/Preferred-MedRECT-32B}"
JUDGE_TYPE="${JUDGE_TYPE:-detection}"
JUDGE_PROMPT_STYLE="${JUDGE_PROMPT_STYLE:-hint_v2}"
KL_COEF="${KL_COEF:-0.01}"
ROLLOUT_GPU_MEMORY_UTILIZATION="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.5}"

PASS=0
FAIL=0
FAIL_LIST=()

check() {
    local name="$1"; shift
    printf "  %-58s " "$name"
    if "$@" >/tmp/preflight_last.out 2>&1; then
        echo "OK"
        PASS=$((PASS + 1))
    else
        echo "FAIL"
        FAIL=$((FAIL + 1))
        FAIL_LIST+=("$name")
        sed 's/^/      /' /tmp/preflight_last.out | head -6
    fi
}

echo "============================================================"
echo "  PRE-FLIGHT CHECK — self-play launch gate"
echo "============================================================"
echo "  ACTOR_MODEL    : $ACTOR_MODEL"
echo "  JUDGE_MODEL    : $JUDGE_MODEL"
echo "  JUDGE_TYPE     : $JUDGE_TYPE"
echo "  JUDGE_PROMPT   : $JUDGE_PROMPT_STYLE"
echo "  KL_COEF        : $KL_COEF"
echo "  JUDGE_VLLM_URL : ${JUDGE_VLLM_URL:-<UNSET>}"
echo "============================================================"

# =============================================================================
# Tier 1 — offline source / config sanity
# =============================================================================
echo
echo "--- Tier 1: offline source + config ---"

check "medrect_judge_prompts.json parses" \
    python3 -c "import json; json.load(open('configs/prompts/medrect_judge_prompts.json'))"

check "medrect_judge_prompts.json has hint_v2 block" \
    python3 -c "
import json
cfg = json.load(open('configs/prompts/medrect_judge_prompts.json'))
assert 'hint_v2' in cfg, 'hint_v2 missing'
assert 'system_prompt' in cfg['hint_v2']
assert '{original_sentence}' in cfg['hint_v2']['user_template']
assert '{modified_sentence}' in cfg['hint_v2']['user_template']
assert '{modified_note}' in cfg['hint_v2']['user_template']
assert '{changed_sid}' in cfg['hint_v2']['user_template']
"

check "judge_client.py detection branch has enable_thinking=False" \
    bash -c "grep -A6 'is_detection = JUDGE_TYPE' scripts/self_play/judge_client.py \
        | grep -q 'enable_thinking.*False'"

check "judge_client.py legacy branch still has enable_thinking=False" \
    bash -c "grep -A12 'else:' scripts/self_play/judge_client.py \
        | grep -q 'enable_thinking.*False'"

check "game_reward.py v5 constants (PARTIAL=0.5, MISS=-1.5)" \
    bash -c "grep -E '^REWARD_PARTIAL = 0.5' scripts/self_play/game_reward.py >/dev/null \
        && grep -E '^REWARD_MISS = -1.5' scripts/self_play/game_reward.py >/dev/null"

check "reward unit test (13 cells)" \
    bash -c "python3 scripts/self_play/test_game_reward_v5.py 2>&1 | grep -q 'ALL REWARD CHECKS PASS'"

check "reward EV math: v5 gives the widest gradient" \
    bash -c "python3 scripts/self_play/reward_ev_check.py 2>&1 | grep -q '+1.246'"

check "run_multiturn_training.sh defaults JUDGE_TYPE=detection" \
    bash -c "grep -E 'JUDGE_TYPE.*detection' scripts/self_play/run_multiturn_training.sh >/dev/null"

check "run_multiturn_training.sh defaults JUDGE_PROMPT_STYLE=hint_v2" \
    bash -c "grep -E 'JUDGE_PROMPT_STYLE.*hint_v2' scripts/self_play/run_multiturn_training.sh >/dev/null"

check "run_multiturn_training.sh propagates JUDGE_TYPE to Ray" \
    bash -c "grep -E 'env_vars.JUDGE_TYPE' scripts/self_play/run_multiturn_training.sh >/dev/null"

check "run_multiturn_training.sh propagates JUDGE_PROMPT_STYLE to Ray" \
    bash -c "grep -E 'env_vars.JUDGE_PROMPT_STYLE' scripts/self_play/run_multiturn_training.sh >/dev/null"

# =============================================================================
# Tier 2 — judge server end-to-end
# =============================================================================
echo
echo "--- Tier 2: judge server (network) ---"

if [[ -z "${JUDGE_VLLM_URL:-}" ]]; then
    echo "  FAIL: JUDGE_VLLM_URL not set (export it before running this script)"
    FAIL=$((FAIL + 1))
    FAIL_LIST+=("JUDGE_VLLM_URL env var")
else
    BASE_URL="${JUDGE_VLLM_URL%/chat/completions}"
    BASE_URL="${BASE_URL%/}"

    check "judge server /v1/models reachable" \
        bash -c "curl -sf -m 10 '${BASE_URL}/models' >/dev/null"

    check "judge server is serving MedRECT-32B" \
        bash -c "curl -sf -m 10 '${BASE_URL}/models' | grep -q '$JUDGE_MODEL'"

    check "judge: end-to-end verdict, no <think>, parseable as sid" \
        python3 - <<PY
import os, json, requests, sys, re
url = os.environ['JUDGE_VLLM_URL'].rstrip('/')
if not url.endswith('/chat/completions'):
    url = url + '/chat/completions'
cfg = json.load(open('configs/prompts/medrect_judge_prompts.json'))
sub = cfg['hint_v2']
payload = {
    'model': '${JUDGE_MODEL}',
    'messages': [
        {'role':'system','content':sub['system_prompt']},
        {'role':'user','content':sub['user_template'].format(
            changed_sid=4,
            original_sentence='Her medications include lisinopril, metoprolol, and metformin.',
            modified_sentence='Her medications include lisinopril, atenolol, and metformin.',
            modified_note=(
                '1. A 50yo woman presents for follow-up.\n'
                '2. PMH: HTN, T2DM.\n'
                '3. Vitals stable.\n'
                '4. Her medications include lisinopril, atenolol, and metformin.\n'
                '5. Labs unchanged.'))},
    ],
    'chat_template_kwargs': {'enable_thinking': False},
    **cfg['sampling_params'],
}
r = requests.post(url, json=payload, timeout=60)
data = r.json()
assert 'choices' in data, f'bad response: {data}'
v = data['choices'][0]['message']['content'].strip()
assert '<think>' not in v, f'thinking is ON; verdict={v!r}'
assert len(v) < 50, f'verdict too long; verdict={v!r}'
assert re.match(r'^(CORRECT|\d+)$', v), f'unparseable verdict: {v!r}'
print(f'  verdict={v!r}')
PY

    check "judge_client end-to-end (production code path)" \
        env JUDGE_TYPE=detection JUDGE_PROMPT_STYLE=hint_v2 python3 - <<PY
import asyncio, os, sys
sys.path.insert(0, '.')
from scripts.self_play.judge_client import judge_sentence_pair
result = asyncio.run(judge_sentence_pair(
    original_sentence='Her medications include metoprolol.',
    modified_sentence='Her medications include atenolol.',
    modified_note=(
        '1. Patient overview.\n'
        '4. Her medications include atenolol.\n'
        '5. Vitals stable.'),
    changed_sid=4,
    judge_model='${JUDGE_MODEL}',
))
status = result.get('judge_status')
verdict = result.get('verdict')
assert status == 'ok', f'judge_status={status!r}, expected ok'
assert verdict in ('SAME', 'CHANGED', 'ABSTAIN'), f'bad verdict: {verdict!r}'
print(f'  verdict={verdict}  status={status}')
PY
fi

# =============================================================================
# Tier 3 — artefacts
# =============================================================================
echo
echo "--- Tier 3: artefacts ---"

check "actor model reachable on HuggingFace: $ACTOR_MODEL" \
    python3 -c "from huggingface_hub import HfApi; HfApi().model_info('$ACTOR_MODEL')"

check "training data: train_injector_seed.parquet" \
    test -f data_processed/self_play_vllm_agent_loop/train_injector_seed.parquet

check "val data: val_injector_seed.parquet" \
    test -f data_processed/self_play_vllm_agent_loop/val_injector_seed.parquet

check "outputs/ writable" \
    bash -c "mkdir -p outputs/_preflight_test && rmdir outputs/_preflight_test"

check "RESUME_MODE is 'auto' or 'disable' (NOT 'none', which crashes verl)" \
    bash -c "test \"\${RESUME_MODE:-disable}\" = auto -o \"\${RESUME_MODE:-disable}\" = disable"

# =============================================================================
# Tier 4 — runtime env
# =============================================================================
echo
echo "--- Tier 4: runtime env ---"

check "JUDGE_VLLM_URL set in current shell" \
    bash -c "test -n \"\${JUDGE_VLLM_URL:-}\""

if command -v free >/dev/null; then
    # Need at least ~80 GB free for actor + ref + vLLM rollout + Ray overhead.
    check "system RAM >= 80 GB available" \
        bash -c "test \$(free -g | awk '/^Mem:/ {print \$7}') -ge 80"
fi

if command -v nvidia-smi >/dev/null; then
    check "GPU 0 mostly free (<10 GB used)" \
        bash -c "test \$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1) -lt 10240"
fi

# =============================================================================
# Verdict + emit launch command
# =============================================================================
echo
echo "============================================================"
echo "  PRE-FLIGHT: $PASS passed, $FAIL failed"
echo "============================================================"

if [[ $FAIL -gt 0 ]]; then
    echo "FAIL: do NOT launch self-play. Issues:"
    for r in "${FAIL_LIST[@]}"; do
        echo "  ✗ $r"
    done
    exit 1
fi

echo "ALL GATES GREEN — safe to launch."
echo
echo "============================================================"
echo "  Copy-paste this command to launch (all env vars hardcoded)"
echo "============================================================"
cat <<LAUNCH
JUDGE_VLLM_URL="${JUDGE_VLLM_URL}" \\
SMOKE=0 \\
AUTO_SCREEN=1 \\
N_GPUS=2 \\
ACTOR_MODEL=${ACTOR_MODEL} \\
JUDGE_MODEL=${JUDGE_MODEL} \\
JUDGE_TYPE=${JUDGE_TYPE} \\
JUDGE_PROMPT_STYLE=${JUDGE_PROMPT_STYLE} \\
KL_COEF=${KL_COEF} \\
ROLLOUT_GPU_MEMORY_UTILIZATION=${ROLLOUT_GPU_MEMORY_UTILIZATION} \\
OUTPUT_DIR=outputs/self_play_v6_final \\
EXPERIMENT_NAME=medserl_selfplay_v6_final \\
MAX_PAIRS=0 \\
TRAIN_BATCH_SIZE=16 \\
PPO_MINI_BATCH_SIZE=8 \\
PPO_EPOCHS=2 \\
TOTAL_EPOCHS=5 \\
SAVE_FREQ=33 \\
KEEP_ONLY_LATEST_CHECKPOINT=0 \\
WANDB=1 \\
RESUME_MODE=disable \\
RESTART_AFTER_CHECKPOINT=0 \\
RESTART_ON_FAILURE=0 \\
MAX_AUTORESTARTS=1 \\
RUNNER_SCRIPT=scripts/self_play/run_multiturn_training.sh \\
bash scripts/self_play/run_online_selfplay_autorestart.sh
LAUNCH

echo
echo "  ALSO: open second terminal and run:"
echo "    bash scripts/self_play/monitor_live.sh"
echo "    # at step ~20: confirm judge_status=ok > 95% and recent reward > early reward"
exit 0
