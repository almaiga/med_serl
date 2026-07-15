#!/usr/bin/env bash
# backup_data_to_hf.sh — back up the training data to a private HF dataset, VERIFIED.
#
# So a wiped pod is one `hf download` away instead of a regeneration. Uploads
# the SFT/RL data + the synthetic test set, then confirms each landed.
#
# Usage:
#   bash scripts/self_play/backup_data_to_hf.sh
#   REPO=Abdine/medserl-data bash scripts/self_play/backup_data_to_hf.sh

set -euo pipefail
cd "$(dirname "$0")/../.."

REPO="${REPO:-Abdine/medserl-training-data}"

echo "=== Backing up training data to ${REPO} (private, verified) ==="

# medrect_v2 SFT/RL data
python3 scripts/self_play/hf_push_verified.py \
    --local data_processed/medrect_v2 \
    --repo "$REPO" --repo-type dataset --private \
    --path-in-repo medrect_v2 \
    --allow-patterns "*.jsonl" \
    --require mixed_sft_train.jsonl \
    --require mixed_sft_heldout_rl.jsonl \
    --commit-message "backup: medrect_v2 SFT/RL data"

# synthetic judge test set
python3 scripts/self_play/hf_push_verified.py \
    --local data_processed/synthetic_test \
    --repo "$REPO" --repo-type dataset --private \
    --path-in-repo synthetic_test \
    --allow-patterns "*.jsonl" \
    --require game_synthetic_v1.jsonl \
    --commit-message "backup: synthetic judge test set"

echo
echo "=== DATA BACKUP COMPLETE + VERIFIED ==="
echo "Restore on a fresh pod with:"
echo "  hf download ${REPO} --repo-type dataset --local-dir data_processed/_restored"
