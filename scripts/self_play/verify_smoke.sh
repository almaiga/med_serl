#!/usr/bin/env bash
# verify_smoke.sh
#
# Sanity-check a smoke training run against the chosen launch config:
#   - judge   = MedRECT-32B + hint_v2  (JUDGE_TYPE=detection, JUDGE_PROMPT_STYLE=hint_v2)
#   - reward  = v5 retune              (REWARD_MISS=-1.5, FORMAT_BONUS=0.2 -> miss reward = -1.3)
#
# Reads the MOST RECENT game-log JSONL under the given directory and runs
# three pass/fail checks that, together, cover the wiring + judge behaviour
# we cannot verify offline:
#
#   1. Judge SAME rate on error_injection < 5 %   -- judge is calibrated on
#      the live injector distribution (Exp 1 baseline: 0 %; we allow drift).
#   2. Non-ok judge_status rate              < 10 %  -- parser + judge
#      integration is healthy on real injector output.
#   3. assessor_reward on outcome=miss        = -1.3  -- v5 reward retune
#      actually propagated to the running reward function.
#
# Usage:
#   bash scripts/self_play/verify_smoke.sh [log_dir]
#
# Defaults log_dir = scripts/self_play/logs/
#
# Exit codes:
#   0  all checks pass -- safe to launch the full run
#   1  could not find / read game logs
#   2  check 1 failed (judge misbehaviour)
#   3  check 2 failed (parser / integration issue)
#   4  check 3 failed (v5 reward did not propagate)

set -uo pipefail

LOG_DIR="${1:-scripts/self_play/logs}"

if ! command -v jq >/dev/null 2>&1; then
    echo "ERROR: jq is required (apt-get install jq)" >&2
    exit 1
fi

LATEST=$(ls -t "${LOG_DIR}"/game_*.jsonl 2>/dev/null | head -n1 || true)
if [[ -z "${LATEST:-}" ]]; then
    echo "ERROR: no game_*.jsonl under ${LOG_DIR}" >&2
    echo "       run the smoke training first: SMOKE=1 bash scripts/self_play/run_multiturn_training.sh" >&2
    exit 1
fi

TOTAL=$(wc -l < "$LATEST" | tr -d ' ')
if [[ "$TOTAL" -eq 0 ]]; then
    echo "ERROR: $LATEST is empty" >&2
    exit 1
fi

echo "============================================================"
echo " Smoke verify"
echo "   log    : $LATEST"
echo "   games  : $TOTAL"
echo "============================================================"

# ─── Diagnostics: verdict mix per mode ──────────────────────────────────
echo
echo "--- Verdict mix per game mode ---"
echo "error_injection:"
jq -r 'select(.requested_mode=="error_injection") | .judge_verdict // "<missing>"' "$LATEST" \
    | sort | uniq -c | sed 's/^/  /'
echo "benign:"
jq -r 'select(.requested_mode=="benign") | .judge_verdict // "<missing>"' "$LATEST" \
    | sort | uniq -c | sed 's/^/  /'

# ─── Diagnostics: judge_status mix ──────────────────────────────────────
echo
echo "--- judge_status mix ---"
jq -r '.judge_status // "<missing>"' "$LATEST" | sort | uniq -c | sed 's/^/  /'

# ─── Diagnostics: assessor reward by outcome ────────────────────────────
echo
echo "--- assessor_reward by outcome (count : outcome : reward) ---"
jq -r '[.assessor_outcome // "<missing>", (.assessor_reward // "<missing>" | tostring)] | @tsv' "$LATEST" \
    | sort | uniq -c | sed 's/^/  /'

# ─── Aggregations for the three checks ──────────────────────────────────
ERR_TOTAL=$(jq -c 'select(.requested_mode=="error_injection")' "$LATEST" | wc -l | tr -d ' ')
ERR_SAME=$(jq -c 'select(.requested_mode=="error_injection" and .judge_verdict=="SAME")' "$LATEST" | wc -l | tr -d ' ')
NON_OK=$(jq -c 'select((.judge_status // "ok") != "ok")' "$LATEST" | wc -l | tr -d ' ')

# Most common miss-reward value (so we judge by the mode, not by outliers).
MISS_COUNT=$(jq -c 'select(.assessor_outcome=="miss")' "$LATEST" | wc -l | tr -d ' ')
MISS_MODE=$(jq -r 'select(.assessor_outcome=="miss") | (.assessor_reward // empty) | tostring' "$LATEST" \
    | sort | uniq -c | sort -rn | head -n1 | awk '{print $2}')

# ─── Pass/fail logic ────────────────────────────────────────────────────
echo
echo "============================================================"
echo " VERDICTS"
echo "============================================================"

EXIT=0

# Check 1 -- judge SAME rate on error_injection
if [[ "$ERR_TOTAL" -gt 0 ]]; then
    SAME_PCT=$(awk -v s="$ERR_SAME" -v t="$ERR_TOTAL" 'BEGIN{printf "%.1f", 100*s/t}')
    if awk -v s="$ERR_SAME" -v t="$ERR_TOTAL" 'BEGIN{exit !((s/t) < 0.05)}'; then
        echo "[ OK ]  Check 1  judge SAME on error_injection = ${ERR_SAME}/${ERR_TOTAL} = ${SAME_PCT}%  (< 5%)"
    else
        echo "[FAIL]  Check 1  judge SAME = ${SAME_PCT}% >= 5%  -- judge mislabels real errors on live distribution"
        [[ "$EXIT" -eq 0 ]] && EXIT=2
    fi
else
    echo "[SKIP]  Check 1  no error_injection games in this log yet"
fi

# Check 2 -- non-ok judge_status rate
NON_OK_PCT=$(awk -v n="$NON_OK" -v t="$TOTAL" 'BEGIN{printf "%.1f", 100*n/t}')
if awk -v n="$NON_OK" -v t="$TOTAL" 'BEGIN{exit !((n/t) < 0.10)}'; then
    echo "[ OK ]  Check 2  non-ok judge_status = ${NON_OK}/${TOTAL} = ${NON_OK_PCT}%  (< 10%)"
else
    echo "[FAIL]  Check 2  non-ok judge_status ${NON_OK_PCT}% >= 10%  -- parser or judge integration issue"
    [[ "$EXIT" -eq 0 ]] && EXIT=3
fi

# Check 3 -- miss reward magnitude
if [[ "$MISS_COUNT" -eq 0 ]]; then
    echo "[SKIP]  Check 3  no miss outcomes yet (need more smoke games)"
elif [[ -z "${MISS_MODE:-}" ]]; then
    echo "[FAIL]  Check 3  assessor_reward field missing on miss outcomes"
    [[ "$EXIT" -eq 0 ]] && EXIT=4
elif awk -v m="$MISS_MODE" 'BEGIN{exit !(m > -1.35 && m < -1.25)}'; then
    echo "[ OK ]  Check 3  miss reward = ${MISS_MODE}  (matches v5: REWARD_MISS=-1.5 + FORMAT=0.2 = -1.3)"
elif awk -v m="$MISS_MODE" 'BEGIN{exit !(m > -0.85 && m < -0.75)}'; then
    echo "[FAIL]  Check 3  miss reward = ${MISS_MODE}  (matches r2: -1.0 + 0.2 = -0.8); v5 retune NOT propagated"
    [[ "$EXIT" -eq 0 ]] && EXIT=4
else
    echo "[FAIL]  Check 3  miss reward = ${MISS_MODE}  -- unexpected; expected -1.3 under v5 reward"
    [[ "$EXIT" -eq 0 ]] && EXIT=4
fi

echo
if [[ "$EXIT" -eq 0 ]]; then
    echo "ALL CHECKS PASS  -- safe to launch the full run"
else
    echo "VERIFY FAILED  -- exit code $EXIT"
    case "$EXIT" in
        2)
            echo "  judge is mislabeling real errors at >= 5 % on the live distribution;"
            echo "  inspect 'modified_note' / 'modified_sentence' shape in the log"
            ;;
        3)
            echo "  parser or judge integration is unhealthy. Inspect raw judge outputs:"
            echo "  jq -r 'select(.judge_status!=\"ok\") | .judge_output // empty' \"$LATEST\" | head"
            ;;
        4)
            echo "  v5 reward constants did not propagate. Sync game_reward.py + duplicates"
            echo "  (interactions/medical_game_interaction.py, rewards/zero_sum_reward.py)"
            ;;
    esac
fi
exit $EXIT
