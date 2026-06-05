#!/usr/bin/env bash
# monitor_live.sh
#
# Live snapshot of the running self-play. Reads the most recent game-log JSONL
# (written by medical_game_interaction.py) and the most recent judge-trace
# JSONL (written by simple_judge_reward.py), prints:
#
#   1. game counts (total, by mode, by outcome)
#   2. reward trend: early window vs recent window vs latest
#   3. judge verdict distribution per mode
#   4. judge status mix (ok / parse failures / etc.)
#   5. per-mode reward stats (assessor and injector)
#   6. who's winning: distribution of (assessor_reward, injector_reward)
#   7. K most recent full game examples (mode, edit, judge verdict, response, rewards)
#
# Usage:
#   bash scripts/self_play/monitor_live.sh                       # default 5 examples
#   bash scripts/self_play/monitor_live.sh /workspace/runs/v6    # custom log dir
#   K=10 bash scripts/self_play/monitor_live.sh                  # show 10 examples
#
# Exit codes: 0 if logs found and parsed, 1 otherwise.

set -uo pipefail

K="${K:-5}"
LOG_DIR="${1:-results/self_play/interactions}"

if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq required (apt-get install -y jq)" >&2
    exit 1
fi

# If MEDSERL_GAME_LOG points to a specific file, use its dir.
if [[ -n "${MEDSERL_GAME_LOG:-}" && -f "${MEDSERL_GAME_LOG}" ]]; then
    GAME_LOG="${MEDSERL_GAME_LOG}"
    LOG_DIR=$(dirname "${MEDSERL_GAME_LOG}")
else
    # Most recent game_*.jsonl in LOG_DIR. The agent loop writes these.
    GAME_LOG=$(ls -t "${LOG_DIR}"/game_*.jsonl 2>/dev/null | head -n1 || true)
    if [[ -z "${GAME_LOG:-}" ]]; then
        GAME_LOG=$(ls -t "${LOG_DIR}"/{interactions,smoke}_*.jsonl 2>/dev/null | head -n1 || true)
    fi
fi
JUDGE_LOG=$(ls -t "${LOG_DIR}"/*judge*trace*.jsonl 2>/dev/null | head -n1 || true)

echo "=========================================================="
echo " self-play live monitor"
echo "   game log  : ${GAME_LOG:-<none>}"
echo "   judge log : ${JUDGE_LOG:-<none>}"
echo "=========================================================="

if [[ -z "${GAME_LOG:-}" ]]; then
    echo "no game log found under ${LOG_DIR}; check MEDSERL_GAME_LOG env or pass path"
    exit 1
fi

N=$(wc -l < "$GAME_LOG" | tr -d ' ')
echo "Total games logged: $N"
if [[ "$N" -eq 0 ]]; then
    echo "(empty log)"
    exit 0
fi

# ─── 1. games by mode + outcome ─────────────────────────────────────────────
echo
echo "--- games by mode ---"
jq -r '.mode // "?"' "$GAME_LOG" | sort | uniq -c | sort -rn | sed 's/^/  /'

echo
echo "--- games by outcome ---"
jq -r '.outcome // "?"' "$GAME_LOG" | sort | uniq -c | sort -rn | sed 's/^/  /'

# ─── 2. reward trend: early vs recent ───────────────────────────────────────
echo
echo "--- assessor reward trend (mean) ---"
EARLY_N=$(( N < 100 ? N / 4 + 1 : 100 ))
RECENT_N=$(( N < 100 ? N / 4 + 1 : 100 ))
{
    echo -n "  early ${EARLY_N} : "
    head -n "$EARLY_N"  "$GAME_LOG" | jq -r '.reward // empty' | awk '{s+=$1;n++} END{if(n)printf "%.3f  (n=%d)\n",s/n,n; else print "no data"}'
    echo -n "  recent ${RECENT_N}: "
    tail -n "$RECENT_N" "$GAME_LOG" | jq -r '.reward // empty' | awk '{s+=$1;n++} END{if(n)printf "%.3f  (n=%d)\n",s/n,n; else print "no data"}'
    echo -n "  overall    : "
    jq -r '.reward // empty' "$GAME_LOG"            | awk '{s+=$1;n++} END{if(n)printf "%.3f  (n=%d)\n",s/n,n; else print "no data"}'
} 2>/dev/null

# ─── 3 & 4. judge verdict + status mix ──────────────────────────────────────
if [[ -n "${JUDGE_LOG:-}" ]]; then
    JN=$(wc -l < "$JUDGE_LOG" | tr -d ' ')
    echo
    echo "--- judge verdict per mode (from $JN judge calls) ---"
    jq -r '[.mode // "?", .judge_verdict // "?"] | @tsv' "$JUDGE_LOG" \
        | sort | uniq -c | sort -rn | sed 's/^/  /'
    # Special focus: SAME rate on error_injection (the v5 failure cell)
    ERR_TOT=$(jq -c 'select((.mode // "")=="error_injection")' "$JUDGE_LOG" | wc -l | tr -d ' ')
    ERR_SAME=$(jq -c 'select((.mode // "")=="error_injection" and (.judge_verdict // "")=="SAME")' "$JUDGE_LOG" | wc -l | tr -d ' ')
    if [[ "$ERR_TOT" -gt 0 ]]; then
        PCT=$(awk -v s="$ERR_SAME" -v t="$ERR_TOT" 'BEGIN{printf "%.1f", 100*s/t}')
        echo "  >> judge SAME on error_injection = ${ERR_SAME}/${ERR_TOT} = ${PCT}%   (v5: 27%; want < 5%)"
    fi
else
    echo
    echo "(no judge trace log found — skipping verdict mix)"
fi

# ─── 5. per-mode reward stats ───────────────────────────────────────────────
echo
echo "--- per-mode reward (mean) — assessor side ---"
for MODE in error_injection benign; do
    MEAN=$(jq -r "select(.mode==\"${MODE}\") | .reward // empty" "$GAME_LOG" \
        | awk '{s+=$1;n++} END{if(n)printf "%.3f (n=%d)",s/n,n; else print "no games"}')
    echo "  ${MODE}: ${MEAN}"
done

echo
echo "--- per-mode reward (mean) — injector side (if logged) ---"
for MODE in error_injection benign; do
    MEAN=$(jq -r "select(.mode==\"${MODE}\") | .injector_reward // empty" "$GAME_LOG" \
        | awk '{s+=$1;n++} END{if(n)printf "%.3f (n=%d)",s/n,n; else print "no games"}')
    echo "  ${MODE}: ${MEAN}"
done

# ─── 6. who is winning — assessor reward sign distribution ──────────────────
echo
echo "--- who's winning — assessor reward sign (overall) ---"
jq -r '.reward // empty' "$GAME_LOG" | awk '
BEGIN{pos=0;zer=0;neg=0;n=0}
{n++; if($1>0.001)pos++; else if($1<-0.001)neg++; else zer++}
END{
    if(n==0){print "  no rewards"; exit}
    printf "  positive : %4d  (%5.1f%%)\n", pos, 100*pos/n
    printf "  zero     : %4d  (%5.1f%%)\n", zer, 100*zer/n
    printf "  negative : %4d  (%5.1f%%)\n", neg, 100*neg/n
}'

# ─── 7. K recent game examples ─────────────────────────────────────────────
echo
echo "--- ${K} most recent games (full context) ---"
tail -n "$K" "$GAME_LOG" | jq -r --arg JL "${JUDGE_LOG:-}" '
  . as $g
  | "----------------------------------------------------------------"
    + "\n  note=\($g.note_id // "?")  mode=\($g.mode // "?")  type=\($g.error_type // "?")"
    + "\n  ground_truth   : \($g.ground_truth // "?")"
    + "\n  injector edit  : \(($g.modified_sentences // "")[0:160] | gsub("\n";" "))"
    + "\n  assessor label : \($g.assessor_label // "?")  sid=\($g.assessor_pred_sid // "?")  valid_fmt=\($g.has_valid_format // "?")"
    + "\n  outcome        : \($g.outcome // "?")"
    + "\n  reward (assess): \($g.reward // "?")"
    + "\n  reward (inject): \($g.injector_reward // "?")  retroactive=\($g.injector_retroactive_reward // "?")"'
echo "----------------------------------------------------------------"
echo
echo "(re-run any time: bash scripts/self_play/monitor_live.sh)"
