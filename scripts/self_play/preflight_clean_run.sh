#!/usr/bin/env bash
# preflight_clean_run.sh — GATE before spending money on the clean self-play run.
#
# Runs every cheap check and STOPS at the first failure. Only prints the launch
# command if ALL gates pass. Tiers get progressively more expensive:
#   Tier 1  offline source/config/reward  (instant, free)
#   Tier 2  HF auth + verified write      (seconds, free)
#   Tier 3  judge thinking end-to-end     (needs GPU + judge model; ~2 min)
#
# Usage:
#   bash scripts/self_play/preflight_clean_run.sh            # tiers 1-2 only
#   RUN_JUDGE_TEST=1 bash scripts/self_play/preflight_clean_run.sh   # + tier 3
#
# Exit 0 = all gates passed; non-zero = a gate failed (message says which).

set -uo pipefail
cd "$(dirname "$0")/../.."

fail() { echo; echo "GATE FAILED: $1"; echo "Fix it before spending GPU."; exit 1; }
ok()   { echo "  [OK] $1"; }

echo "=========================================================="
echo " PREFLIGHT — clean self-play run"
echo "=========================================================="

# ── Tier 1: offline source / config / reward ─────────────────────────────────
echo
echo "-- Tier 1: offline (config, reward, data) --"

# judge thinking ON in the code
grep -q '"enable_thinking": True' scripts/self_play/judge_client.py \
    || fail "judge_client.py detection branch is not enable_thinking=True"
ok "judge_client.py: enable_thinking=True"

# judge token budget large enough for thinking
MT=$(python3 -c "import json;print(json.load(open('configs/prompts/medrect_judge_prompts.json'))['sampling_params']['max_tokens'])")
[[ "$MT" -ge 1024 ]] || fail "judge max_tokens=$MT too small for thinking (want >=2048)"
ok "judge max_tokens=$MT"

# prompt JSON valid
python3 -c "import json;json.load(open('configs/prompts/medrect_judge_prompts.json'))" \
    || fail "medrect_judge_prompts.json invalid"
ok "judge prompt config valid JSON"

# reward unit tests
python3 scripts/self_play/test_game_reward_v5.py >/dev/null 2>&1 \
    || fail "reward unit test (v5) failed"
ok "reward unit tests pass"

# training data present
for f in mixed_sft_train.jsonl mixed_sft_heldout_rl.jsonl; do
    [[ -f "data_processed/medrect_v2/$f" ]] || fail "missing data_processed/medrect_v2/$f"
done
ok "SFT/RL training data present"

# MEDEC test present
[[ -d data_raw/MEDEC/MEDEC-MS && -d data_raw/MEDEC/MEDEC-UW ]] \
    || fail "MEDEC test set missing under data_raw/MEDEC/"
ok "MEDEC test set present"

# ── Tier 2: HF auth + verified write ─────────────────────────────────────────
echo
echo "-- Tier 2: HuggingFace auth + write path --"
python3 - <<'PY' || exit 1
import io, sys
from huggingface_hub import HfApi
api = HfApi()
try:
    who = api.whoami(); print(f"  [OK] HF auth: {who['name']}")
except Exception as e:
    print(f"GATE FAILED: HF auth: {e!r}"); sys.exit(1)
# verified round-trip write
repo = f"{who['name']}/_preflight_writetest"
try:
    api.create_repo(repo, repo_type="dataset", private=True, exist_ok=True)
    api.upload_file(path_or_fileobj=io.BytesIO(b"ok\n"),
                    path_in_repo="OK.txt", repo_id=repo, repo_type="dataset")
    landed = "OK.txt" in api.list_repo_files(repo, repo_type="dataset")
    api.delete_repo(repo, repo_type="dataset")
    if not landed:
        print("GATE FAILED: HF write did not land"); sys.exit(1)
    print("  [OK] HF write verified (upload -> list-tree -> delete)")
except Exception as e:
    print(f"GATE FAILED: HF write test: {e!r}"); sys.exit(1)
PY
[[ $? -eq 0 ]] || fail "HF auth/write"

# SFT start model resolves
python3 - <<'PY' || exit 1
import sys
from huggingface_hub import HfApi
api = HfApi()
repo = "Abdine/qwen3-4b-medrect-mixed-v2"
try:
    files = api.list_repo_files(repo)
    if "config.json" in files and any(f.endswith(".safetensors") for f in files):
        print(f"  [OK] SFT start model resolves: {repo}")
    else:
        print(f"GATE FAILED: {repo} missing weights/config"); sys.exit(1)
except Exception as e:
    print(f"GATE FAILED: {repo}: {e!r}"); sys.exit(1)
PY
[[ $? -eq 0 ]] || fail "SFT start model on HF"

# ── Tier 3: judge thinking end-to-end (GPU) ──────────────────────────────────
echo
if [[ "${RUN_JUDGE_TEST:-0}" == "1" ]]; then
    echo "-- Tier 3: judge thinking end-to-end (GPU) --"
    JUDGE_PROMPT_STYLE="${JUDGE_PROMPT_STYLE:-hint_v2}" \
      python3 scripts/self_play/test_judge_thinking_endtoend.py \
      || fail "judge end-to-end thinking test did not return GO"
    ok "judge thinking end-to-end: GO"
else
    echo "-- Tier 3: judge thinking test SKIPPED (set RUN_JUDGE_TEST=1 to run on GPU) --"
fi

echo
echo "=========================================================="
echo " ALL PREFLIGHT GATES PASSED"
echo "=========================================================="
echo "Next:"
echo "  1. Back up data:      bash scripts/self_play/backup_data_to_hf.sh"
echo "  2. Smoke self-play:   bash scripts/self_play/smoke_selfplay.sh"
echo "  3. Launch full run:   bash scripts/self_play/launch_clean_selfplay.sh"
echo "  (start checkpoint_watcher.sh in the background alongside step 3)"
