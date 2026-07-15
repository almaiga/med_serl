#!/usr/bin/env bash
# checkpoint_watcher.sh — auto-push every new self-play checkpoint to HF, VERIFIED.
#
# Runs alongside training. Polls the output dir for new global_step_* folders
# and pushes each one's HF-format actor weights to its own repo, then verifies
# the push landed (via hf_push_verified.py). This is the safeguard that would
# have prevented the v6 loss — checkpoints reach HF as they are produced, and
# a failed push exits loudly instead of being silently lost.
#
# Usage:
#   OUTPUT_DIR=outputs/self_play_v7 REPO_PREFIX=Abdine/qwen3-4b-medserl-v7-step \
#     bash scripts/self_play/checkpoint_watcher.sh
#
#   # run in background next to training:
#   OUTPUT_DIR=... REPO_PREFIX=... nohup bash scripts/self_play/checkpoint_watcher.sh \
#     > logs/checkpoint_watcher.log 2>&1 &

set -uo pipefail
cd "$(dirname "$0")/../.."

OUTPUT_DIR="${OUTPUT_DIR:?set OUTPUT_DIR (e.g. outputs/self_play_v7)}"
REPO_PREFIX="${REPO_PREFIX:?set REPO_PREFIX (e.g. Abdine/qwen3-4b-medserl-v7-step)}"
POLL_SECONDS="${POLL_SECONDS:-60}"
PRIVATE="${PRIVATE:-0}"                # 1 = push private repos
PUSHED_MARK=".hf_pushed"              # marker file dropped in a ckpt once pushed

priv_flag=""
[[ "$PRIVATE" == "1" ]] && priv_flag="--private"

echo "=========================================================="
echo " checkpoint watcher"
echo "   output dir  : $OUTPUT_DIR"
echo "   repo prefix : ${REPO_PREFIX}<N>"
echo "   poll        : ${POLL_SECONDS}s   private=${PRIVATE}"
echo "=========================================================="

while true; do
    if [[ -d "$OUTPUT_DIR" ]]; then
        for ck in "$OUTPUT_DIR"/global_step_*; do
            [[ -d "$ck" ]] || continue
            step=$(basename "$ck" | grep -oE '[0-9]+$')
            hf_dir="$ck/actor/huggingface"
            # only push once the HF export exists and we haven't pushed it yet
            if [[ -f "$hf_dir/config.json" && ! -f "$ck/$PUSHED_MARK" ]]; then
                repo="${REPO_PREFIX}${step}"
                echo "[$(date -u +%H:%M:%S)] new checkpoint step $step -> $repo"
                if python3 scripts/self_play/hf_push_verified.py \
                        --local "$hf_dir" --repo "$repo" --repo-type model \
                        $priv_flag \
                        --require config.json --require model.safetensors \
                        --commit-message "self-play checkpoint step ${step}"; then
                    touch "$ck/$PUSHED_MARK"
                    echo "[$(date -u +%H:%M:%S)] step $step VERIFIED on HF"
                else
                    echo "[$(date -u +%H:%M:%S)] ** step $step push FAILED — will retry next poll **"
                fi
            fi
        done
    fi
    sleep "$POLL_SECONDS"
done
