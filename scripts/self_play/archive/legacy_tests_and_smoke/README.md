# Legacy Tests And Smoke Files

These files were archived from the active `scripts/self_play/` folder because they are not part of the current recommended training flow.

Archived here:

- smoke launchers such as `run_grpo_smoke.sh`, `run_reinforce_smoke.sh`, `run_rule_smoke.sh`, `run_smoke_minimal.sh`, `run_train_only_smoke.sh`, `run_dynamics_test.sh`
- experimental smoke configs such as `configs/grpo_separated.yaml` and `configs/ppo_sglang_smoke.yaml`
- local test utilities such as `test_parquet_format.py`, `test_selfplay_loop.py`, `test_reward.py`, `test_parser.py`, `test_note_extraction.py`, `test_game_tool.py`, `test_training_dynamics.py`
- smoke-analysis helpers such as `analyze_smoke_quality.py` and `analyze_training_log.py`

Current active path remains:

- `scripts/self_play/run_multiturn_training.sh`
- `scripts/self_play/simple_judge_reward.py`
- `scripts/self_play/interactions/medical_game_interaction.py`
- `scripts/self_play/preprocess_medec.py`
- `scripts/self_play/configs/ppo_multiturn.yaml`
- `scripts/self_play/configs/interaction_config.yaml`

These archived files are preserved for reference and can be restored later if needed.
