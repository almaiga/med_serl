#!/bin/bash
# Fast config validation — tests veRL config loading WITHOUT starting the judge server.
# Usage: bash scripts/self_play/test_config_only.sh
#
# This runs the veRL config through Hydra + migrate_legacy_reward_impl to catch
# OmegaConf/Hydra errors in seconds instead of waiting 5-10 min for the judge.

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"

ACTOR_MODEL="${ACTOR_MODEL:-Qwen/Qwen3-4B}"
CONFIG_DIR="$PROJECT_ROOT/scripts/self_play/configs"
TRAIN_PARQUET="$PROJECT_ROOT/data_processed/self_play/train.parquet"
VAL_PARQUET="$PROJECT_ROOT/data_processed/self_play/val.parquet"

echo "=== Fast Config Validation ==="
echo "Config dir: $CONFIG_DIR"
echo ""

# Step 1: Apply the GLIBC patch if needed
echo "--- Applying GLIBC patch (if needed) ---"
python3 << 'PATCH_EOF'
import pathlib, re
fpath = pathlib.Path("/workspace/verl/verl/workers/engine/__init__.py")
if not fpath.exists():
    print("  SKIP: veRL not installed at /workspace/verl")
else:
    code = fpath.read_text()
    new_code, n = re.subn(r'except ImportError:', 'except (ImportError, OSError):', code)
    if n == 0:
        print("  Already patched.")
    else:
        fpath.write_text(new_code)
        print(f"  Patched {n} except clauses.")
PATCH_EOF

# Step 2: Test config loading + migration
echo ""
echo "--- Testing Hydra config load + migrate_legacy_reward_impl ---"
python3 << 'TEST_EOF'
import sys, os
os.environ.setdefault("HYDRA_FULL_ERROR", "1")

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

config_dir = os.environ.get("CONFIG_DIR", "scripts/self_play/configs")
abs_config_dir = os.path.abspath(config_dir)

print(f"  Loading config from: {abs_config_dir}")

# Initialize Hydra with our config directory
with initialize_config_dir(config_dir=abs_config_dir, version_base=None):
    # Compose with the same overrides the smoke script uses
    overrides = [
        "algorithm.adv_estimator=reinforce_plus_plus",
        f"data.train_files={os.environ.get('TRAIN_PARQUET', 'data_processed/self_play/train.parquet')}",
        f"data.val_files={os.environ.get('VAL_PARQUET', 'data_processed/self_play/val.parquet')}",
        "data.train_batch_size=4",
        "data.val_batch_size=4",
        "data.train_max_samples=20",
        "data.max_prompt_length=1024",
        "data.max_response_length=512",
        "data.filter_overlong_prompts=False",
        "data.truncation=error",
        f"actor_rollout_ref.model.path={os.environ.get('ACTOR_MODEL', 'Qwen/Qwen3-4B')}",
        "++actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        "actor_rollout_ref.model.use_remove_padding=False",
        "actor_rollout_ref.actor.ppo_mini_batch_size=4",
        "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1",
        "actor_rollout_ref.actor.use_kl_loss=False",
        "actor_rollout_ref.actor.strategy=fsdp2",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.gpu_memory_utilization=0.45",
        "actor_rollout_ref.rollout.enforce_eager=True",
        "actor_rollout_ref.rollout.response_length=512",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.ref.fsdp_config.param_offload=True",
        f"critic.model.path={os.environ.get('ACTOR_MODEL', 'Qwen/Qwen3-4B')}",
        "critic.ppo_micro_batch_size_per_gpu=1",
        "critic.fsdp_config.param_offload=True",
        "critic.fsdp_config.optimizer_offload=True",
        "reward_model.enable=False",
        f"custom_reward_function.path={os.path.abspath('scripts/self_play/agentic_reward.py')}",
        "custom_reward_function.name=async_compute_score",
        "trainer.total_epochs=1",
        "trainer.n_gpus_per_node=1",
        "trainer.nnodes=1",
        "trainer.save_freq=-1",
    ]
    
    config = compose(config_name="self_play", overrides=overrides)

print("  ✓ Hydra config loaded successfully")
print()

# Check top-level keys
top_keys = list(config.keys())
print(f"  Top-level keys: {top_keys}")

# Check for required sections
for section in ["reward_model", "custom_reward_function", "sandbox_fusion", "reward"]:
    if section in config:
        print(f"  ✓ '{section}' section exists")
    else:
        print(f"  ✗ '{section}' section MISSING")

print()

# Now test migrate_legacy_reward_impl
print("--- Testing migrate_legacy_reward_impl ---")
try:
    from verl.experimental.reward_loop.reward_loop import migrate_legacy_reward_impl
    config = migrate_legacy_reward_impl(config)
    print("  ✓ Migration succeeded!")
    print()
    
    # Show the resulting reward config
    if "reward" in config:
        reward_keys = list(config.reward.keys())
        print(f"  reward.* keys after migration: {reward_keys}")
    
    # Verify legacy keys are deleted
    for old_key in ["reward_model", "custom_reward_function", "sandbox_fusion"]:
        if old_key in config:
            print(f"  ⚠ Legacy key '{old_key}' still present (should have been deleted)")
        else:
            print(f"  ✓ Legacy key '{old_key}' deleted")
            
except Exception as e:
    print(f"  ✗ Migration FAILED: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()
print("=== Config validation PASSED ===")
TEST_EOF

echo ""
echo "Done."
