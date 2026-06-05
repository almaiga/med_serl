#!/usr/bin/env bash
# monitor_live.sh
#
# Live snapshot of the running self-play. Uses the same field names that
# plot_training_dynamics.py / analyze_training.py / analyze_judge_traces.py
# already know about — judge_verdict, judge_status, assessor_outcome,
# assessor_reward, injector_reward, injector_outcome, phase, etc.
#
# Usage:
#   bash scripts/self_play/monitor_live.sh                                 # default log dir
#   bash scripts/self_play/monitor_live.sh /workspace/runs/v6/logs         # custom
#   K=10 bash scripts/self_play/monitor_live.sh                            # show 10 examples
#
# Exit 0 if logs found and parsed, 1 otherwise.

set -uo pipefail

K="${K:-5}"
LOG_DIR="${1:-results/self_play/interactions}"

if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq required (apt-get install -y jq)" >&2
    exit 1
fi

if [[ -n "${MEDSERL_GAME_LOG:-}" && -f "${MEDSERL_GAME_LOG}" ]]; then
    LOG="${MEDSERL_GAME_LOG}"
else
    LOG=$(ls -t "${LOG_DIR}"/game_*.jsonl 2>/dev/null | head -n1 || true)
    if [[ -z "${LOG:-}" ]]; then
        LOG=$(ls -t "${LOG_DIR}"/{interactions,smoke}_*.jsonl 2>/dev/null | head -n1 || true)
    fi
fi

echo "=========================================================="
echo " self-play live monitor"
echo "   log : ${LOG:-<none>}"
echo "=========================================================="

if [[ -z "${LOG:-}" ]]; then
    echo "no log found under ${LOG_DIR}"
    exit 1
fi

N_ALL=$(wc -l < "$LOG" | tr -d ' ')
N_COMPLETE=$(jq -c 'select(.phase == "game_complete")' "$LOG" 2>/dev/null | wc -l | tr -d ' ')
echo "Total records       : $N_ALL"
echo "phase=game_complete : $N_COMPLETE   <-- the rows the reward fields use"

if [[ "$N_ALL" -eq 0 ]]; then
    exit 0
fi

# ─── games by mode (count completed games only) ─────────────────────────────
echo
echo "--- games by mode (phase=game_complete) ---"
jq -r 'select(.phase=="game_complete") | .mode // "?"' "$LOG" | sort | uniq -c | sort -rn | sed 's/^/  /'

# ─── judge verdict per mode ─────────────────────────────────────────────────
echo
echo "--- judge verdict per mode ---"
jq -r 'select(.judge_verdict) | [.mode // "?", .judge_verdict] | @tsv' "$LOG" \
    | sort | uniq -c | sort -rn | sed 's/^/  /'
# v5 watchdog cell
ERR_TOT=$(jq -c 'select(.judge_verdict and (.mode // "")=="error_injection")' "$LOG" | wc -l | tr -d ' ')
ERR_SAME=$(jq -c 'select((.mode // "")=="error_injection" and (.judge_verdict // "")=="SAME")' "$LOG" | wc -l | tr -d ' ')
if [[ "$ERR_TOT" -gt 0 ]]; then
    PCT=$(awk -v s="$ERR_SAME" -v t="$ERR_TOT" 'BEGIN{printf "%.1f", 100*s/t}')
    echo "  >> judge SAME on error_injection = ${ERR_SAME}/${ERR_TOT} = ${PCT}%   (v5: 27%; want < 5%)"
fi

# ─── judge status mix ───────────────────────────────────────────────────────
echo
echo "--- judge_status mix ---"
jq -r 'select(.judge_status) | .judge_status' "$LOG" | sort | uniq -c | sort -rn | sed 's/^/  /'

# ─── assessor outcome distribution ──────────────────────────────────────────
echo
echo "--- assessor outcome (phase=game_complete) ---"
jq -r 'select(.phase=="game_complete") | (.assessor_outcome // .outcome // "?")' "$LOG" \
    | sort | uniq -c | sort -rn | sed 's/^/  /'

# ─── injector outcome distribution ──────────────────────────────────────────
echo
echo "--- injector outcome ---"
jq -r '.injector_outcome // "?"' "$LOG" | sort | uniq -c | sort -rn | sed 's/^/  /'

# ─── reward trend (assessor) — early vs recent vs overall ───────────────────
echo
echo "--- assessor_reward trend (mean over phase=game_complete) ---"
EARLY=$(( N_COMPLETE < 100 ? N_COMPLETE / 4 + 1 : 100 ))
RECENT=$(( N_COMPLETE < 100 ? N_COMPLETE / 4 + 1 : 100 ))
COMPLETE_ROWS=$(jq -c 'select(.phase=="game_complete")' "$LOG")
{
    echo -n "  early  ${EARLY}: "
    echo "$COMPLETE_ROWS" | head -n "$EARLY"  | jq -r '(.assessor_reward // .reward // empty)' \
        | awk '{s+=$1;n++} END{if(n)printf "%.3f  (n=%d)\n",s/n,n; else print "no data"}'
    echo -n "  recent ${RECENT}: "
    echo "$COMPLETE_ROWS" | tail -n "$RECENT" | jq -r '(.assessor_reward // .reward // empty)' \
        | awk '{s+=$1;n++} END{if(n)printf "%.3f  (n=%d)\n",s/n,n; else print "no data"}'
    echo -n "  overall   : "
    echo "$COMPLETE_ROWS" | jq -r '(.assessor_reward // .reward // empty)' \
        | awk '{s+=$1;n++} END{if(n)printf "%.3f  (n=%d)\n",s/n,n; else print "no data"}'
}

echo
echo "--- injector_reward trend (mean over ALL records) ---"
{
    echo -n "  early  ${EARLY}: "
    head -n "$EARLY"  "$LOG" | jq -r '(.injector_reward // empty)' \
        | awk '{s+=$1;n++} END{if(n)printf "%.3f  (n=%d)\n",s/n,n; else print "no data"}'
    echo -n "  recent ${RECENT}: "
    tail -n "$RECENT" "$LOG" | jq -r '(.injector_reward // empty)' \
        | awk '{s+=$1;n++} END{if(n)printf "%.3f  (n=%d)\n",s/n,n; else print "no data"}'
    echo -n "  overall   : "
    jq -r '(.injector_reward // empty)' "$LOG" \
        | awk '{s+=$1;n++} END{if(n)printf "%.3f  (n=%d)\n",s/n,n; else print "no data"}'
}

# ─── per-mode reward stats ──────────────────────────────────────────────────
echo
echo "--- per-mode mean rewards ---"
for MODE in error_injection benign; do
    A=$(echo "$COMPLETE_ROWS" | jq -r "select(.mode==\"${MODE}\") | (.assessor_reward // .reward // empty)" \
        | awk '{s+=$1;n++} END{if(n)printf "%.3f (n=%d)",s/n,n; else print "no data"}')
    I=$(jq -r "select(.mode==\"${MODE}\") | .injector_reward // empty" "$LOG" \
        | awk '{s+=$1;n++} END{if(n)printf "%.3f (n=%d)",s/n,n; else print "no data"}')
    echo "  ${MODE}:   assessor=${A}   injector=${I}"
done

# ─── who's winning ──────────────────────────────────────────────────────────
echo
echo "--- assessor reward sign (phase=game_complete) ---"
echo "$COMPLETE_ROWS" | jq -r '(.assessor_reward // .reward // empty)' | awk '
BEGIN{pos=0;zer=0;neg=0;n=0}
{n++; if($1>0.001)pos++; else if($1<-0.001)neg++; else zer++}
END{
    if(n==0){print "  no rewards"; exit}
    printf "  positive : %4d  (%5.1f%%)\n", pos, 100*pos/n
    printf "  zero     : %4d  (%5.1f%%)\n", zer, 100*zer/n
    printf "  negative : %4d  (%5.1f%%)\n", neg, 100*neg/n
}'

# ─── K most recent complete games ───────────────────────────────────────────
echo
echo "--- ${K} most recent complete games ---"
echo "$COMPLETE_ROWS" | tail -n "$K" | jq -r '
  . as $g
  | "----------------------------------------------------------------"
    + "\n  note=\($g.note_id // "?")  mode=\($g.mode // "?")  type=\($g.error_type // "?")"
    + "\n  judge verdict  : \($g.judge_verdict // "?")  status=\($g.judge_status // "?")"
    + "\n  injector edit  : \(($g.modified_sentences // "")[0:160] | gsub("\n";" "))"
    + "\n  assessor label : \($g.assessor_label // "?")  sid=\($g.assessor_pred_sid // "?")"
    + "\n  assessor outcome: \($g.assessor_outcome // $g.outcome // "?")"
    + "\n  reward (assess): \($g.assessor_reward // $g.reward // "?")"
    + "\n  reward (inject): \($g.injector_reward // "?")   inj_outcome=\($g.injector_outcome // "?")"'
echo "----------------------------------------------------------------"
echo
echo "(re-run any time: bash scripts/self_play/monitor_live.sh)"
