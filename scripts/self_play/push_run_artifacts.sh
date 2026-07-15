#!/usr/bin/env bash
# push_run_artifacts.sh — one-shot VERIFIED push of a finished run to HF:
#   1. every checkpoint's HF-format actor weights -> Abdine/<prefix><N> (model repos)
#   2. game logs + trainer logs                   -> Abdine/<logs repo> (dataset repo)
# Each push is verified by re-listing the repo (hf_push_verified.py); the script
# exits non-zero and says FAIL loudly if anything did not land.
#
# Usage (on the training pod, after stopping the run):
#   bash scripts/self_play/push_run_artifacts.sh
#   OUTPUT_DIR=outputs/self_play_v7 REPO_PREFIX=Abdine/qwen3-4b-medserl-v7-step \
#     LOGS_REPO=Abdine/medserl-v7-run-logs bash scripts/self_play/push_run_artifacts.sh

set -uo pipefail
cd "$(dirname "$0")/../.."

OUTPUT_DIR="${OUTPUT_DIR:-outputs/self_play_v7}"
REPO_PREFIX="${REPO_PREFIX:-Abdine/qwen3-4b-medserl-v7-step}"
LOGS_REPO="${LOGS_REPO:-Abdine/medserl-v7-run-logs}"
GAME_DIR="${GAME_DIR:-results/self_play/interactions}"
TRAIN_LOG_DIR="${TRAIN_LOG_DIR:-logs/screen}"

FAILURES=0
PUSHED=""

echo "=========================================================="
echo " push run artifacts (verified)"
echo "   checkpoints : $OUTPUT_DIR -> ${REPO_PREFIX}<N> (private)"
echo "   logs        : $GAME_DIR, $TRAIN_LOG_DIR -> $LOGS_REPO (private dataset)"
echo "=========================================================="

# ── sanity: is training really stopped? ──────────────────────────────────────
if screen -list 2>/dev/null | grep -qE "medserl.*(selfplay|autorestart)"; then
    echo "WARN: a medserl screen session is still running:"
    screen -list | grep -E "medserl" | sed 's/^/  /'
    echo "  (checkpoints already on disk are static, so pushing is safe;"
    echo "   but if you meant to stop training, do that too.)"
fi

# ── 1. checkpoints ────────────────────────────────────────────────────────────
CKPTS=$(find "$OUTPUT_DIR" -maxdepth 1 -type d -name "global_step_*" 2>/dev/null | sort -t_ -k3 -n)
if [[ -z "$CKPTS" ]]; then
    echo "FAIL: no checkpoints under $OUTPUT_DIR"
    FAILURES=$((FAILURES + 1))
fi
for ck in $CKPTS; do
    step=$(basename "$ck" | sed 's/global_step_//')
    hfdir="$ck/actor/huggingface"
    if [[ ! -d "$hfdir" ]]; then
        echo "FAIL: $ck has no actor/huggingface dir (hf_model not saved?)"
        FAILURES=$((FAILURES + 1)); continue
    fi
    # require config.json plus whatever safetensors file actually exists
    # (sharded models have model-0000X-of-0000Y.safetensors, not model.safetensors)
    st=$(ls "$hfdir"/*.safetensors 2>/dev/null | head -1)
    if [[ -z "$st" ]]; then
        echo "FAIL: $hfdir contains no .safetensors file"
        FAILURES=$((FAILURES + 1)); continue
    fi
    echo
    echo "--- pushing checkpoint step $step -> ${REPO_PREFIX}${step} ---"
    if python3 scripts/self_play/hf_push_verified.py \
        --local "$hfdir" \
        --repo "${REPO_PREFIX}${step}" --repo-type model --private \
        --require config.json --require "$(basename "$st")" \
        --commit-message "self-play checkpoint global_step_${step}"; then
        touch "$ck/.hf_pushed"
        PUSHED="$PUSHED ${REPO_PREFIX}${step}"
    else
        FAILURES=$((FAILURES + 1))
    fi
done

# ── 2. game logs ──────────────────────────────────────────────────────────────
LATEST_GAME=$(ls -t "$GAME_DIR"/game_*.jsonl 2>/dev/null | head -1)
if [[ -n "${LATEST_GAME:-}" ]]; then
    echo
    echo "--- pushing game logs -> ${LOGS_REPO}/interactions ---"
    if python3 scripts/self_play/hf_push_verified.py \
        --local "$GAME_DIR" \
        --repo "$LOGS_REPO" --repo-type dataset --private \
        --path-in-repo interactions \
        --allow-patterns "*.jsonl,*.json" \
        --require "$(basename "$LATEST_GAME")" \
        --commit-message "game logs"; then
        PUSHED="$PUSHED ${LOGS_REPO}(interactions)"
    else
        FAILURES=$((FAILURES + 1))
    fi
else
    echo "FAIL: no game logs under $GAME_DIR"
    FAILURES=$((FAILURES + 1))
fi

# ── 3. trainer logs ───────────────────────────────────────────────────────────
LATEST_TRAIN=$(ls -t "$TRAIN_LOG_DIR"/*.log 2>/dev/null | head -1)
if [[ -n "${LATEST_TRAIN:-}" ]]; then
    echo
    echo "--- pushing trainer logs -> ${LOGS_REPO}/trainer_logs ---"
    if python3 scripts/self_play/hf_push_verified.py \
        --local "$TRAIN_LOG_DIR" \
        --repo "$LOGS_REPO" --repo-type dataset --private \
        --path-in-repo trainer_logs \
        --allow-patterns "*.log" \
        --require "$(basename "$LATEST_TRAIN")" \
        --commit-message "trainer logs"; then
        PUSHED="$PUSHED ${LOGS_REPO}(trainer_logs)"
    else
        FAILURES=$((FAILURES + 1))
    fi
else
    echo "WARN: no trainer logs under $TRAIN_LOG_DIR (skipping)"
fi

# ── summary ──────────────────────────────────────────────────────────────────
echo
echo "=========================================================="
if [[ "$FAILURES" -eq 0 ]]; then
    echo " ALL PUSHES VERIFIED OK:"
    for p in $PUSHED; do echo "   - $p"; done
else
    echo " ${FAILURES} FAILURE(S) — the items marked FAIL above are NOT on HF."
    echo " Do not delete anything local until this reports ALL OK."
fi
echo "=========================================================="
exit "$FAILURES"
