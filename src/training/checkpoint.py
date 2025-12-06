"""
Checkpoint Module - Model checkpoint saving and loading for MedSeRL.

This module provides functionality to:
- Save model weights at configured intervals
- Include episode number in checkpoint filenames
- Load checkpoints for resuming training

Requirements: 8.5, 8.6
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """
    Metadata stored with each checkpoint.
    
    Attributes:
        episode: Episode number when checkpoint was saved
        total_episodes: Total episodes in training run
        mean_reward: Mean reward at checkpoint time
        accuracy: Evaluation accuracy at checkpoint time
        model_path: Original model path
        timestamp: ISO format timestamp
    """
    episode: int
    total_episodes: int
    mean_reward: float = 0.0
    accuracy: float = 0.0
    model_path: str = ""
    timestamp: str = ""


def get_checkpoint_filename(episode: int) -> str:
    """
    Generate checkpoint filename with episode number.
    
    Args:
        episode: Episode number (1-indexed for display)
        
    Returns:
        Checkpoint filename string
        
    Requirements: 8.6 - Include episode number in filename
    """
    return f"checkpoint_episode_{episode}"


def save_checkpoint(
    checkpoint_dir: str,
    episode: int,
    model: Optional[Any] = None,
    tokenizer: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
    total_episodes: int = 0,
    model_path: str = ""
) -> str:
    """
    Save a training checkpoint with episode number in filename.
    
    Saves model weights (if available), configuration, and metadata
    to a checkpoint directory named with the episode number.
    
    Args:
        checkpoint_dir: Base directory for checkpoints
        episode: Current episode number (0-indexed, will be displayed as 1-indexed)
        model: Model to save (HuggingFace model or None)
        tokenizer: Tokenizer to save (optional)
        config: Training configuration dict
        metrics: Current training metrics
        total_episodes: Total episodes in training run
        model_path: Original model path
        
    Returns:
        Path to the saved checkpoint directory
        
    Requirements: 8.5, 8.6
    """
    from datetime import datetime
    
    # Create checkpoint directory with episode number
    # Requirements: 8.6 - Include episode number in filename
    checkpoint_name = get_checkpoint_filename(episode + 1)  # 1-indexed for display
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)
    
    # Save model weights if provided
    # Requirements: 8.5 - Save model weights at configured intervals
    if model is not None:
        try:
            model_save_path = os.path.join(checkpoint_path, "model")
            model.save_pretrained(model_save_path)
            logger.info(f"Model weights saved to {model_save_path}")
        except Exception as e:
            logger.warning(f"Failed to save model weights: {e}")
            # Save a marker file indicating model save failed
            marker_path = os.path.join(checkpoint_path, "model_save_failed.txt")
            with open(marker_path, "w") as f:
                f.write(f"Model save failed: {str(e)}\n")
    
    # Save tokenizer if provided
    if tokenizer is not None:
        try:
            tokenizer_save_path = os.path.join(checkpoint_path, "tokenizer")
            tokenizer.save_pretrained(tokenizer_save_path)
            logger.info(f"Tokenizer saved to {tokenizer_save_path}")
        except Exception as e:
            logger.warning(f"Failed to save tokenizer: {e}")
    
    # Create metadata
    metadata = CheckpointMetadata(
        episode=episode + 1,  # 1-indexed for display
        total_episodes=total_episodes,
        mean_reward=metrics.get("mean_reward", 0.0) if metrics else 0.0,
        accuracy=metrics.get("accuracy", 0.0) if metrics else 0.0,
        model_path=model_path,
        timestamp=datetime.now().isoformat()
    )
    
    # Save metadata
    metadata_path = os.path.join(checkpoint_path, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(asdict(metadata), f, indent=2)
    
    # Save config if provided
    if config is not None:
        config_path = os.path.join(checkpoint_path, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2, default=str)
    
    # Save metrics if provided
    if metrics is not None:
        metrics_path = os.path.join(checkpoint_path, "metrics.json")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
    
    logger.info(f"Checkpoint saved: {checkpoint_path}")
    print(f"Checkpoint saved: {checkpoint_path}")
    
    return checkpoint_path


def load_checkpoint_metadata(checkpoint_path: str) -> Optional[CheckpointMetadata]:
    """
    Load checkpoint metadata from a checkpoint directory.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        
    Returns:
        CheckpointMetadata or None if not found
    """
    metadata_path = os.path.join(checkpoint_path, "metadata.json")
    
    if not os.path.exists(metadata_path):
        logger.warning(f"Metadata not found at {metadata_path}")
        return None
    
    try:
        with open(metadata_path, "r") as f:
            data = json.load(f)
        return CheckpointMetadata(**data)
    except Exception as e:
        logger.error(f"Failed to load metadata: {e}")
        return None


def load_checkpoint(
    checkpoint_path: str,
    load_model: bool = True,
    load_tokenizer: bool = True,
    device_map: str = "auto"
) -> Dict[str, Any]:
    """
    Load a checkpoint for resuming training or inference.
    
    Args:
        checkpoint_path: Path to checkpoint directory
        load_model: Whether to load the model
        load_tokenizer: Whether to load the tokenizer
        device_map: Device map for model loading
        
    Returns:
        Dictionary with loaded components:
            - model: Loaded model (if load_model=True)
            - tokenizer: Loaded tokenizer (if load_tokenizer=True)
            - metadata: CheckpointMetadata
            - config: Training config dict
            - metrics: Metrics dict
    """
    result = {
        "model": None,
        "tokenizer": None,
        "metadata": None,
        "config": None,
        "metrics": None
    }
    
    # Load metadata
    result["metadata"] = load_checkpoint_metadata(checkpoint_path)
    
    # Load config
    config_path = os.path.join(checkpoint_path, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r") as f:
            result["config"] = json.load(f)
    
    # Load metrics
    metrics_path = os.path.join(checkpoint_path, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            result["metrics"] = json.load(f)
    
    # Load model
    if load_model:
        model_path = os.path.join(checkpoint_path, "model")
        if os.path.exists(model_path):
            try:
                from transformers import AutoModelForCausalLM
                result["model"] = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device_map,
                    torch_dtype="auto"
                )
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
    
    # Load tokenizer
    if load_tokenizer:
        tokenizer_path = os.path.join(checkpoint_path, "tokenizer")
        if os.path.exists(tokenizer_path):
            try:
                from transformers import AutoTokenizer
                result["tokenizer"] = AutoTokenizer.from_pretrained(tokenizer_path)
                logger.info(f"Tokenizer loaded from {tokenizer_path}")
            except Exception as e:
                logger.error(f"Failed to load tokenizer: {e}")
    
    return result


def list_checkpoints(checkpoint_dir: str) -> list:
    """
    List all checkpoints in a directory, sorted by episode number.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        List of (episode_number, checkpoint_path) tuples, sorted by episode
    """
    checkpoints = []
    
    if not os.path.exists(checkpoint_dir):
        return checkpoints
    
    for name in os.listdir(checkpoint_dir):
        path = os.path.join(checkpoint_dir, name)
        if os.path.isdir(path) and name.startswith("checkpoint_episode_"):
            try:
                # Extract episode number from filename
                episode = int(name.replace("checkpoint_episode_", ""))
                checkpoints.append((episode, path))
            except ValueError:
                continue
    
    # Sort by episode number
    checkpoints.sort(key=lambda x: x[0])
    
    return checkpoints


def get_latest_checkpoint(checkpoint_dir: str) -> Optional[str]:
    """
    Get the path to the latest checkpoint (highest episode number).
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None if no checkpoints exist
    """
    checkpoints = list_checkpoints(checkpoint_dir)
    
    if not checkpoints:
        return None
    
    return checkpoints[-1][1]


def cleanup_old_checkpoints(
    checkpoint_dir: str,
    keep_last_n: int = 3
) -> list:
    """
    Remove old checkpoints, keeping only the most recent N.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last_n: Number of recent checkpoints to keep
        
    Returns:
        List of removed checkpoint paths
    """
    import shutil
    
    checkpoints = list_checkpoints(checkpoint_dir)
    removed = []
    
    if len(checkpoints) <= keep_last_n:
        return removed
    
    # Remove oldest checkpoints
    to_remove = checkpoints[:-keep_last_n]
    
    for episode, path in to_remove:
        try:
            shutil.rmtree(path)
            removed.append(path)
            logger.info(f"Removed old checkpoint: {path}")
        except Exception as e:
            logger.warning(f"Failed to remove checkpoint {path}: {e}")
    
    return removed
