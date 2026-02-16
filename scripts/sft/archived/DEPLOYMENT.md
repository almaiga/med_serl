# Deployment Instructions

## Files to Upload to Remote Server

When running on a remote server (RunPod, etc.), make sure to upload both files to the same directory:

```
scripts/sft/
├── quick_infer_qwen3_4b_lora.py
└── parse_changes.py
```

## Quick Deploy Command

From your local machine:

```bash
# Copy both files to remote server
scp -i ~/.ssh/id_ed25519 \
  scripts/sft/quick_infer_qwen3_4b_lora.py \
  scripts/sft/parse_changes.py \
  root@YOUR_RUNPOD_ID:/workspace/med_serl/scripts/sft/
```

Or if already connected via SSH:

```bash
# On your local machine, from the project root
cd /Users/josmaiga/Documents/GitHub/med_serl

# Then on the remote server, create the directory and copy
ssh -i ~/.ssh/id_ed25519 root@YOUR_RUNPOD_ID "mkdir -p /workspace/med_serl/scripts/sft"

scp -i ~/.ssh/id_ed25519 \
  scripts/sft/*.py \
  root@YOUR_RUNPOD_ID:/workspace/med_serl/scripts/sft/
```

## Verify Installation

On the remote server:

```bash
cd /workspace/med_serl/scripts/sft
ls -la *.py

# Should show:
# parse_changes.py
# quick_infer_qwen3_4b_lora.py
```

## Run Test

```bash
python quick_infer_qwen3_4b_lora.py --help
# Should NOT show the parse_changes warning if both files are present
```
