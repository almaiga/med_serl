# MedSeRL Training Module
# Contains training orchestrator, reward engine, and OpenRLHF integration

from src.training.reward_engine import (
    calculate_reward,
    calculate_reward_with_metadata,
    has_thinking_section,
    parse_verdict,
    RewardMetadata,
    STRUCTURAL_REWARD,
    CORRECT_CLASSIFICATION_REWARD,
    FALSE_NEGATIVE_PENALTY,
    FALSE_POSITIVE_PENALTY
)

from src.training.train_serl import (
    prepare_sft_data,
    format_sft_input,
    format_sft_target,
    sft_warmup,
    rl_training_loop,
    main as train_main,
    SFTExample,
    SFTConfig,
    RLTrainingConfig,
    MedSERLTrainer
)

from src.training.checkpoint import (
    save_checkpoint,
    load_checkpoint,
    load_checkpoint_metadata,
    list_checkpoints,
    get_latest_checkpoint,
    cleanup_old_checkpoints,
    get_checkpoint_filename,
    CheckpointMetadata
)

from src.training.logging_utils import (
    WandbLogger,
    WandbConfig,
    create_wandb_logger,
    log_metrics_if_configured
)

__all__ = [
    # Reward Engine
    "calculate_reward",
    "calculate_reward_with_metadata",
    "has_thinking_section",
    "parse_verdict",
    "RewardMetadata",
    "STRUCTURAL_REWARD",
    "CORRECT_CLASSIFICATION_REWARD",
    "FALSE_NEGATIVE_PENALTY",
    "FALSE_POSITIVE_PENALTY",
    # Training
    "prepare_sft_data",
    "format_sft_input",
    "format_sft_target",
    "sft_warmup",
    "rl_training_loop",
    "train_main",
    "SFTExample",
    "SFTConfig",
    "RLTrainingConfig",
    "MedSERLTrainer",
    # Checkpoint
    "save_checkpoint",
    "load_checkpoint",
    "load_checkpoint_metadata",
    "list_checkpoints",
    "get_latest_checkpoint",
    "cleanup_old_checkpoints",
    "get_checkpoint_filename",
    "CheckpointMetadata",
    # Logging
    "WandbLogger",
    "WandbConfig",
    "create_wandb_logger",
    "log_metrics_if_configured"
]
