# Legacy Self-Play Files

These files were archived from the active `scripts/self_play/` surface to reduce confusion.

They belong to the older agentic/UMLS workflow and deprecated standalone config path, including:

- `agentic_reward.py`
- `judge_prompts.py`
- `umls_async.py`
- `run_agentic_*`
- `run_training.sh`
- `run_training_v2.sh`
- `run_online_selfplay.sh`
- `run_mvp.sh`
- `test_agentic_judge.py`
- `test_config_only.sh`
- `configs/self_play.yaml`
- `configs/self_play.yaml.bak`
- `configs/ppo_agentic.yaml`

Current active path:

- `scripts/self_play/run_multiturn_training.sh`
- `scripts/self_play/simple_judge_reward.py`
- `scripts/self_play/interactions/medical_game_interaction.py`
- `scripts/self_play/preprocess_medec.py`
- `scripts/self_play/configs/ppo_multiturn.yaml`
- `scripts/self_play/configs/interaction_config.yaml`

Nothing in this folder is part of the recommended training flow now.
