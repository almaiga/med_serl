"""
Logging Utilities - Weights & Biases integration for MedSeRL.

This module provides functionality to:
- Initialize W&B logging if configured
- Log training and evaluation metrics
- Log model artifacts and checkpoints

Requirements: 8.4
"""

import logging
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class WandbConfig:
    """
    Configuration for Weights & Biases logging.
    
    Attributes:
        enabled: Whether W&B logging is enabled
        project: W&B project name
        run_name: Optional run name
        entity: Optional W&B entity (team/user)
        tags: Optional list of tags for the run
        notes: Optional notes for the run
        config: Training configuration to log
    """
    enabled: bool = False
    project: str = "medserl"
    run_name: Optional[str] = None
    entity: Optional[str] = None
    tags: Optional[List[str]] = None
    notes: Optional[str] = None
    config: Optional[Dict[str, Any]] = None


class WandbLogger:
    """
    Weights & Biases logger for MedSeRL training.
    
    Provides methods to log metrics, artifacts, and manage W&B runs.
    Gracefully handles cases where W&B is not installed or configured.
    
    Requirements: 8.4
    """
    
    def __init__(self, config: WandbConfig):
        """
        Initialize the W&B logger.
        
        Args:
            config: WandbConfig with logging settings
        """
        self.config = config
        self.run = None
        self._wandb = None
        self._initialized = False
        
        if config.enabled:
            self._initialize()
    
    def _initialize(self) -> bool:
        """
        Initialize W&B run.
        
        Returns:
            True if initialization successful, False otherwise
        """
        try:
            import wandb
            self._wandb = wandb
        except ImportError:
            logger.warning(
                "wandb not installed. Install with: pip install wandb"
            )
            return False
        
        try:
            # Initialize W&B run
            init_kwargs = {
                "project": self.config.project,
                "name": self.config.run_name,
                "config": self.config.config,
            }
            
            if self.config.entity:
                init_kwargs["entity"] = self.config.entity
            if self.config.tags:
                init_kwargs["tags"] = self.config.tags
            if self.config.notes:
                init_kwargs["notes"] = self.config.notes
            
            self.run = self._wandb.init(**init_kwargs)
            self._initialized = True
            
            logger.info(f"W&B initialized: {self.config.project}/{self.run.name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize W&B: {e}")
            return False
    
    @property
    def is_enabled(self) -> bool:
        """Check if W&B logging is enabled and initialized."""
        return self._initialized and self.run is not None
    
    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        prefix: str = ""
    ) -> None:
        """
        Log metrics to W&B.
        
        Args:
            metrics: Dictionary of metric names to values
            step: Optional step number (episode, iteration, etc.)
            prefix: Optional prefix for metric names (e.g., "train/", "eval/")
            
        Requirements: 8.4
        """
        if not self.is_enabled:
            return
        
        # Add prefix to metric names
        if prefix:
            metrics = {f"{prefix}{k}": v for k, v in metrics.items()}
        
        try:
            if step is not None:
                self._wandb.log(metrics, step=step)
            else:
                self._wandb.log(metrics)
        except Exception as e:
            logger.warning(f"Failed to log metrics to W&B: {e}")
    
    def log_training_metrics(
        self,
        episode: int,
        mean_reward: float,
        max_reward: float,
        min_reward: float,
        batch_size: int,
        **kwargs
    ) -> None:
        """
        Log training metrics for an episode.
        
        Args:
            episode: Current episode number
            mean_reward: Mean reward for the episode
            max_reward: Maximum reward in the batch
            min_reward: Minimum reward in the batch
            batch_size: Number of samples in the batch
            **kwargs: Additional metrics to log
        """
        metrics = {
            "train/mean_reward": mean_reward,
            "train/max_reward": max_reward,
            "train/min_reward": min_reward,
            "train/batch_size": batch_size,
            "train/episode": episode,
            **{f"train/{k}": v for k, v in kwargs.items()}
        }
        self.log_metrics(metrics, step=episode)
    
    def log_evaluation_metrics(
        self,
        episode: int,
        accuracy: float,
        mean_reward: float,
        precision: float = 0.0,
        recall: float = 0.0,
        f1: float = 0.0,
        per_error_type_accuracy: Optional[Dict[str, float]] = None,
        **kwargs
    ) -> None:
        """
        Log evaluation metrics.
        
        Args:
            episode: Current episode number
            accuracy: Overall accuracy
            mean_reward: Mean reward on evaluation set
            precision: Precision score
            recall: Recall score
            f1: F1 score
            per_error_type_accuracy: Accuracy per error type
            **kwargs: Additional metrics to log
        """
        metrics = {
            "eval/accuracy": accuracy,
            "eval/mean_reward": mean_reward,
            "eval/precision": precision,
            "eval/recall": recall,
            "eval/f1": f1,
            **{f"eval/{k}": v for k, v in kwargs.items()}
        }
        
        # Add per-error-type accuracy
        if per_error_type_accuracy:
            for error_type, acc in per_error_type_accuracy.items():
                # Sanitize error type name for W&B
                safe_name = error_type.lower().replace(" ", "_")
                metrics[f"eval/accuracy_{safe_name}"] = acc
        
        self.log_metrics(metrics, step=episode)
    
    def log_checkpoint(
        self,
        checkpoint_path: str,
        episode: int,
        metrics: Optional[Dict[str, float]] = None
    ) -> None:
        """
        Log a checkpoint as a W&B artifact.
        
        Args:
            checkpoint_path: Path to the checkpoint directory
            episode: Episode number of the checkpoint
            metrics: Optional metrics associated with the checkpoint
        """
        if not self.is_enabled:
            return
        
        try:
            artifact = self._wandb.Artifact(
                name=f"checkpoint-episode-{episode}",
                type="model",
                metadata={
                    "episode": episode,
                    **(metrics or {})
                }
            )
            artifact.add_dir(checkpoint_path)
            self.run.log_artifact(artifact)
            logger.info(f"Logged checkpoint artifact for episode {episode}")
        except Exception as e:
            logger.warning(f"Failed to log checkpoint artifact: {e}")
    
    def log_config(self, config: Dict[str, Any]) -> None:
        """
        Update the run configuration.
        
        Args:
            config: Configuration dictionary to log
        """
        if not self.is_enabled:
            return
        
        try:
            self._wandb.config.update(config)
        except Exception as e:
            logger.warning(f"Failed to update W&B config: {e}")
    
    def log_summary(self, summary: Dict[str, Any]) -> None:
        """
        Log summary metrics (final results).
        
        Args:
            summary: Dictionary of summary metrics
        """
        if not self.is_enabled:
            return
        
        try:
            for key, value in summary.items():
                self.run.summary[key] = value
        except Exception as e:
            logger.warning(f"Failed to log summary: {e}")
    
    def finish(self) -> None:
        """Finish the W&B run."""
        if self.is_enabled and self.run:
            try:
                self.run.finish()
                logger.info("W&B run finished")
            except Exception as e:
                logger.warning(f"Failed to finish W&B run: {e}")
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - finish the run."""
        self.finish()
        return False


def create_wandb_logger(
    enabled: bool = False,
    project: str = "medserl",
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    **kwargs
) -> WandbLogger:
    """
    Factory function to create a WandbLogger.
    
    Args:
        enabled: Whether to enable W&B logging
        project: W&B project name
        run_name: Optional run name
        config: Training configuration to log
        **kwargs: Additional WandbConfig parameters
        
    Returns:
        Configured WandbLogger instance
        
    Requirements: 8.4
    """
    wandb_config = WandbConfig(
        enabled=enabled,
        project=project,
        run_name=run_name,
        config=config,
        **{k: v for k, v in kwargs.items() if hasattr(WandbConfig, k)}
    )
    return WandbLogger(wandb_config)


def log_metrics_if_configured(
    wandb_logger: Optional[WandbLogger],
    metrics: Dict[str, float],
    step: Optional[int] = None,
    prefix: str = ""
) -> None:
    """
    Convenience function to log metrics if W&B is configured.
    
    Args:
        wandb_logger: Optional WandbLogger instance
        metrics: Metrics to log
        step: Optional step number
        prefix: Optional prefix for metric names
        
    Requirements: 8.4
    """
    if wandb_logger and wandb_logger.is_enabled:
        wandb_logger.log_metrics(metrics, step=step, prefix=prefix)
