#!/usr/bin/env bash
# monitor_training.sh — one-shot health snapshot of a running self-play run.
#
# Combines PPO training metrics (from the trainer stdout log) with judge/game
# health (from the interaction logs) and flags anomalies. Read-only.
#
# Usage:
#   bash scripts/self_play/monitor_training.sh                      # auto-discover
#   bash scripts/self_play/monitor_training.sh <train.log> <gamedir>
#   watch -n 30 bash scripts/self_play/monitor_training.sh         # live
#
# Health gates (flags a WARN if):
#   - grad_norm > 10            (instability / spike)
#   - judge_status=ok  < 70%    (judge wiring / thinking problem)
#   - SAME-on-error    > 15%    (judge missing injected errors)
#   - game_invalid     > 25%    (rollouts wasted)

set -uo pipefail

TRAIN_LOG="${1:-}"
GAME_DIR="${2:-results/self_play/interactions}"

# ── Auto-discover the trainer log if not given ───────────────────────────────
if [[ -z "$TRAIN_LOG" ]]; then
    for d in logs/screen logs/online_autorestart logs; do
        cand=$(ls -t "$d"/*.log 2>/dev/null | head -1)
        [[ -n "$cand" ]] && { TRAIN_LOG="$cand"; break; }
    done
fi

echo "=========================================================="
echo " self-play training monitor"
echo "   train log : ${TRAIN_LOG:-<none found>}"
echo "   game dir  : ${GAME_DIR}"
echo "=========================================================="

WARN=0

# ── PPO metrics from the latest step line ────────────────────────────────────
if [[ -n "${TRAIN_LOG:-}" && -f "$TRAIN_LOG" ]]; then
    LAST=$(grep -oE "step:[0-9]+ - .*perf/throughput:[0-9.]+" "$TRAIN_LOG" | tail -1)
    if [[ -n "$LAST" ]]; then
        get() { echo "$LAST" | grep -oE "$1:[-0-9.eE]+" | head -1 | cut -d: -f2; }
        echo
        echo "--- PPO training (latest step) ---"
        printf "  global_step        : %s\n" "$(get 'training/global_step')"
        printf "  epoch              : %s\n" "$(get 'training/epoch')"
        printf "  critic/score/mean  : %s\n" "$(get 'critic/score/mean')"
        printf "  critic/rewards/mean: %s\n" "$(get 'critic/rewards/mean')"
        printf "  reward_kl_penalty  : %s\n" "$(get 'actor/reward_kl_penalty')"
        printf "  grad_norm          : %s\n" "$(get 'actor/grad_norm')"
        printf "  pg_loss            : %s\n" "$(get 'actor/pg_loss')"
        printf "  lr                 : %s\n" "$(get 'actor/lr')"
        printf "  response_len/mean  : %s\n" "$(get 'response_length/mean')"
        printf "  throughput tok/s   : %s\n" "$(get 'perf/throughput')"

        GN=$(get 'actor/grad_norm')
        if [[ -n "$GN" ]] && awk "BEGIN{exit !($GN>10)}"; then
            echo "  ** WARN: grad_norm ${GN} > 10 (instability)"; WARN=1
        fi

        # score/mean trend over last 10 steps
        echo
        echo "--- critic/score/mean trend (last 10 steps) ---"
        grep -oE "training/global_step:[0-9]+|critic/score/mean:[-0-9.eE]+" "$TRAIN_LOG" \
          | paste - - | tail -10 \
          | sed -E 's#training/global_step:([0-9]+).*critic/score/mean:([-0-9.eE]+)#  step \1  score \2#'
    else
        echo "  (no step lines yet — trainer still warming up?)"
    fi
else
    echo "  (no trainer log found)"
fi

# ── Judge / game health from interaction logs ────────────────────────────────
LOG=$(ls -t "${GAME_DIR}"/game_*.jsonl 2>/dev/null | head -1)
if [[ -n "${LOG:-}" ]] && command -v jq >/dev/null 2>&1; then
    N=$(jq -c 'select(.phase=="game_complete")' "$LOG" 2>/dev/null | wc -l | tr -d ' ')
    echo
    echo "--- game health : $(basename "$LOG") ($N complete games) ---"

    echo "  judge_status:"
    jq -r 'select(.judge_status) | .judge_status' "$LOG" | sort | uniq -c | sed 's/^/    /'
    OKN=$(jq -c 'select((.judge_status//"")=="ok")' "$LOG" | wc -l | tr -d ' ')
    ALLJ=$(jq -c 'select(.judge_status)' "$LOG" | wc -l | tr -d ' ')
    if [[ "$ALLJ" -gt 0 ]]; then
        PCT=$(awk -v a="$OKN" -v b="$ALLJ" 'BEGIN{printf "%.0f",100*a/b}')
        echo "    judge_status=ok : ${PCT}%"
        [[ "$PCT" -lt 70 ]] && { echo "    ** WARN: judge ok < 70% (wiring/thinking?)"; WARN=1; }
    fi

    ET=$(jq -c 'select((.mode//"")=="error_injection" and .judge_verdict)' "$LOG" | wc -l | tr -d ' ')
    ES=$(jq -c 'select((.mode//"")=="error_injection" and (.judge_verdict//"")=="SAME")' "$LOG" | wc -l | tr -d ' ')
    if [[ "$ET" -gt 0 ]]; then
        SP=$(awk -v s="$ES" -v t="$ET" 'BEGIN{printf "%.1f",100*s/t}')
        echo "  judge SAME-on-error : ${ES}/${ET} = ${SP}%  (want < 15% now that thinking is on)"
        awk "BEGIN{exit !($SP>15)}" && { echo "    ** WARN: SAME-on-error > 15% (judge missing errors)"; WARN=1; }
    fi

    echo "  assessor outcomes:"
    jq -r 'select(.phase=="game_complete") | (.assessor_outcome // "?")' "$LOG" \
      | sort | uniq -c | sort -rn | sed 's/^/    /'
    GI=$(jq -c 'select(.phase=="game_complete" and (.assessor_outcome//"")=="game_invalid")' "$LOG" | wc -l | tr -d ' ')
    if [[ "$N" -gt 0 ]]; then
        GIP=$(awk -v g="$GI" -v n="$N" 'BEGIN{printf "%.0f",100*g/n}')
        [[ "$GIP" -gt 25 ]] && { echo "    ** WARN: game_invalid ${GIP}% > 25% (wasted rollouts)"; WARN=1; }
    fi

    echo "  assessor_reward (early vs recent, complete games):"
    ROWS=$(jq -c 'select(.phase=="game_complete")' "$LOG")
    echo "$ROWS" | head -50 | jq -r '.assessor_reward // empty' | awk '{s+=$1;n++} END{if(n)printf "    early 50 : %+.3f\n",s/n}'
    echo "$ROWS" | tail -50 | jq -r '.assessor_reward // empty' | awk '{s+=$1;n++} END{if(n)printf "    recent 50: %+.3f\n",s/n}'
else
    echo
    echo "  (no game logs under ${GAME_DIR}, or jq missing)"
fi

# ── Checkpoints ──────────────────────────────────────────────────────────────
echo
echo "--- checkpoints on disk ---"
CKPTS=$(find outputs -type d -name "global_step_*" 2>/dev/null | sort -t_ -k3 -n)
if [[ -n "$CKPTS" ]]; then
    echo "$CKPTS" | sed 's/^/  /'
    echo "  (verify each pushed to HF: huggingface-cli repo-files list-tree <repo>)"
else
    echo "  none yet"
fi

echo
if [[ "$WARN" -eq 0 ]]; then
    echo "STATUS: healthy (no flags)"
else
    echo "STATUS: ** ${WARN} warning(s) above — investigate before trusting the run **"
fi
echo "(re-run: bash scripts/self_play/monitor_training.sh)"
