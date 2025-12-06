"""
MedSeRL Configuration Schema and Validation

Defines dataclasses for training arguments, model paths, and hyperparameters.
Implements path validation logic per Requirements 10.1-10.5.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List


class ConfigurationError(Exception):
    """Raised when configuration validation fails."""
    pass


@dataclass
class ModelPaths:
    """Configuration for model paths.
    
    Attributes:
        base_model_path: Path to the base MedGemma-4B model
        scribe_model_path: Path to the Scribe agent model (defaults to base_model_path)
        doctor_model_path: Path to the Doctor agent model (defaults to base_model_path)
        reference_model_path: Path to the reference model for KL divergence
    """
    base_model_path: str
    scribe_model_path: Optional[str] = None
    doctor_model_path: Optional[str] = None
    reference_model_path: Optional[str] = None
    
    def __post_init__(self):
        # Default scribe and doctor to base model if not specified
        if self.scribe_model_path is None:
            self.scribe_model_path = self.base_model_path
        if self.doctor_model_path is None:
            self.doctor_model_path = self.base_model_path
        if self.reference_model_path is None:
            self.reference_model_path = self.base_model_path


@dataclass
class DataPaths:
    """Configuration for data paths.
    
    Attributes:
        medec_data_path: Path to MEDEC dataset directory
        output_dir: Directory for saving checkpoints and logs
        checkpoint_dir: Directory for model checkpoints (defaults to output_dir/checkpoints)
        log_dir: Directory for training logs (defaults to output_dir/logs)
    """
    medec_data_path: str
    output_dir: str
    checkpoint_dir: Optional[str] = None
    log_dir: Optional[str] = None
    
    def __post_init__(self):
        if self.checkpoint_dir is None:
            self.checkpoint_dir = os.path.join(self.output_dir, "checkpoints")
        if self.log_dir is None:
            self.log_dir = os.path.join(self.output_dir, "logs")


@dataclass
class TrainingHyperparameters:
    """Hyperparameters for RL training.
    
    Attributes:
        learning_rate: Learning rate for policy optimization
        batch_size: Number of samples per training batch (must be divisible by 4)
        kl_coef: KL divergence coefficient for PPO/Reinforce++
        gamma: Discount factor for rewards
        clip_range: PPO clip range
        max_grad_norm: Maximum gradient norm for clipping
        warmup_steps: Number of warmup steps for learning rate scheduler
    """
    learning_rate: float = 1e-5
    batch_size: int = 16
    kl_coef: float = 0.1
    gamma: float = 0.99
    clip_range: float = 0.2
    max_grad_norm: float = 1.0
    warmup_steps: int = 100


@dataclass
class TrainingSchedule:
    """Configuration for training schedule.
    
    Attributes:
        num_episodes: Total number of training episodes
        eval_frequency: Evaluate every N episodes
        checkpoint_frequency: Save checkpoint every N episodes
        sft_epochs: Number of epochs for SFT warm-up phase
        max_steps_per_episode: Maximum steps per episode
    """
    num_episodes: int = 1000
    eval_frequency: int = 50
    checkpoint_frequency: int = 100
    sft_epochs: int = 1
    max_steps_per_episode: int = 512


@dataclass
class InfrastructureConfig:
    """Configuration for distributed training infrastructure.
    
    Attributes:
        num_gpus: Number of GPUs to use
        num_vllm_engines: Number of vLLM engines for rollout generation
        tensor_parallel_size: Tensor parallelism size for vLLM
        ray_num_cpus: Number of CPUs for Ray workers
        use_wandb: Whether to log to Weights & Biases
        wandb_project: W&B project name
    """
    num_gpus: int = 1
    num_vllm_engines: int = 1
    tensor_parallel_size: int = 1
    ray_num_cpus: int = 4
    use_wandb: bool = False
    wandb_project: str = "medserl"


@dataclass
class RewardConfig:
    """Configuration for reward calculation.
    
    Attributes:
        structural_reward: Reward for having <thinking> section
        correct_classification_reward: Reward for correct classification
        false_negative_penalty: Penalty for missing an error
        false_positive_penalty: Penalty for false alarm
    """
    structural_reward: float = 0.1
    correct_classification_reward: float = 1.0
    false_negative_penalty: float = -1.0
    false_positive_penalty: float = -1.5


@dataclass
class MedSERLConfig:
    """Main configuration class for MedSeRL training.
    
    Combines all configuration components and provides validation.
    """
    model_paths: ModelPaths
    data_paths: DataPaths
    hyperparameters: TrainingHyperparameters = field(default_factory=TrainingHyperparameters)
    schedule: TrainingSchedule = field(default_factory=TrainingSchedule)
    infrastructure: InfrastructureConfig = field(default_factory=InfrastructureConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    
    def validate(self) -> List[str]:
        """Validate the configuration and return list of errors.
        
        Returns:
            List of error messages. Empty list if configuration is valid.
        """
        errors = []
        
        # Validate model paths exist
        errors.extend(self._validate_model_paths())
        
        # Validate data paths exist
        errors.extend(self._validate_data_paths())
        
        # Validate hyperparameters
        errors.extend(self._validate_hyperparameters())
        
        # Validate schedule
        errors.extend(self._validate_schedule())
        
        # Validate infrastructure
        errors.extend(self._validate_infrastructure())
        
        return errors
    
    def _validate_model_paths(self) -> List[str]:
        """Validate that model paths exist."""
        errors = []
        
        if not self._path_exists(self.model_paths.base_model_path):
            errors.append(
                f"Base model path does not exist: {self.model_paths.base_model_path}"
            )
        
        if self.model_paths.scribe_model_path != self.model_paths.base_model_path:
            if not self._path_exists(self.model_paths.scribe_model_path):
                errors.append(
                    f"Scribe model path does not exist: {self.model_paths.scribe_model_path}"
                )
        
        if self.model_paths.doctor_model_path != self.model_paths.base_model_path:
            if not self._path_exists(self.model_paths.doctor_model_path):
                errors.append(
                    f"Doctor model path does not exist: {self.model_paths.doctor_model_path}"
                )
        
        return errors
    
    def _validate_data_paths(self) -> List[str]:
        """Validate that data paths exist."""
        errors = []
        
        if not self._path_exists(self.data_paths.medec_data_path):
            errors.append(
                f"MEDEC data path does not exist: {self.data_paths.medec_data_path}"
            )
        
        # Output directory can be created, but parent must exist
        output_parent = Path(self.data_paths.output_dir).parent
        if not output_parent.exists() and str(output_parent) != ".":
            errors.append(
                f"Parent directory for output does not exist: {output_parent}"
            )
        
        return errors
    
    def _validate_hyperparameters(self) -> List[str]:
        """Validate hyperparameter values."""
        errors = []
        hp = self.hyperparameters
        
        if hp.learning_rate <= 0:
            errors.append(
                f"Learning rate must be positive, got: {hp.learning_rate}"
            )
        
        if hp.batch_size <= 0:
            errors.append(
                f"Batch size must be positive, got: {hp.batch_size}"
            )
        
        if hp.batch_size % 4 != 0:
            errors.append(
                f"Batch size must be divisible by 4 for quadrant strategy, got: {hp.batch_size}"
            )
        
        if hp.kl_coef < 0:
            errors.append(
                f"KL coefficient must be non-negative, got: {hp.kl_coef}"
            )
        
        if not 0 <= hp.gamma <= 1:
            errors.append(
                f"Gamma must be between 0 and 1, got: {hp.gamma}"
            )
        
        if hp.clip_range <= 0:
            errors.append(
                f"Clip range must be positive, got: {hp.clip_range}"
            )
        
        return errors
    
    def _validate_schedule(self) -> List[str]:
        """Validate training schedule values."""
        errors = []
        sched = self.schedule
        
        if sched.num_episodes <= 0:
            errors.append(
                f"Number of episodes must be positive, got: {sched.num_episodes}"
            )
        
        if sched.eval_frequency <= 0:
            errors.append(
                f"Evaluation frequency must be positive, got: {sched.eval_frequency}"
            )
        
        if sched.checkpoint_frequency <= 0:
            errors.append(
                f"Checkpoint frequency must be positive, got: {sched.checkpoint_frequency}"
            )
        
        if sched.sft_epochs <= 0:
            errors.append(
                f"SFT epochs must be positive, got: {sched.sft_epochs}"
            )
        
        return errors
    
    def _validate_infrastructure(self) -> List[str]:
        """Validate infrastructure configuration."""
        errors = []
        infra = self.infrastructure
        
        if infra.num_gpus <= 0:
            errors.append(
                f"Number of GPUs must be positive, got: {infra.num_gpus}"
            )
        
        if infra.num_vllm_engines <= 0:
            errors.append(
                f"Number of vLLM engines must be positive, got: {infra.num_vllm_engines}"
            )
        
        if infra.tensor_parallel_size <= 0:
            errors.append(
                f"Tensor parallel size must be positive, got: {infra.tensor_parallel_size}"
            )
        
        return errors
    
    @staticmethod
    def _path_exists(path: str) -> bool:
        """Check if a path exists (file or directory)."""
        return Path(path).exists()
    
    def validate_or_raise(self) -> None:
        """Validate configuration and raise ConfigurationError if invalid.
        
        Raises:
            ConfigurationError: If any validation errors are found.
        """
        errors = self.validate()
        if errors:
            error_msg = "Configuration validation failed:\n" + "\n".join(
                f"  - {error}" for error in errors
            )
            raise ConfigurationError(error_msg)
    
    def ensure_directories(self) -> None:
        """Create output directories if they don't exist."""
        Path(self.data_paths.output_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data_paths.checkpoint_dir).mkdir(parents=True, exist_ok=True)
        Path(self.data_paths.log_dir).mkdir(parents=True, exist_ok=True)


def validate_path_exists(path: str, path_name: str) -> None:
    """Validate that a path exists and raise ConfigurationError if not.
    
    Args:
        path: The path to validate
        path_name: Human-readable name for error messages
        
    Raises:
        ConfigurationError: If the path does not exist
    """
    if not Path(path).exists():
        raise ConfigurationError(f"{path_name} does not exist: {path}")


def create_config_from_args(
    base_model_path: str,
    medec_data_path: str,
    output_dir: str,
    learning_rate: float = 1e-5,
    batch_size: int = 16,
    kl_coef: float = 0.1,
    num_episodes: int = 1000,
    eval_frequency: int = 50,
    num_gpus: int = 1,
    use_wandb: bool = False,
    **kwargs
) -> MedSERLConfig:
    """Create a MedSERLConfig from command-line style arguments.
    
    This is a convenience function for creating configurations from
    argparse or similar command-line argument parsers.
    
    Args:
        base_model_path: Path to the base MedGemma-4B model
        medec_data_path: Path to MEDEC dataset
        output_dir: Directory for outputs
        learning_rate: Learning rate for training
        batch_size: Batch size (must be divisible by 4)
        kl_coef: KL divergence coefficient
        num_episodes: Number of training episodes
        eval_frequency: Evaluate every N episodes
        num_gpus: Number of GPUs to use
        use_wandb: Whether to use Weights & Biases
        **kwargs: Additional arguments (ignored)
        
    Returns:
        Configured MedSERLConfig instance
    """
    return MedSERLConfig(
        model_paths=ModelPaths(base_model_path=base_model_path),
        data_paths=DataPaths(
            medec_data_path=medec_data_path,
            output_dir=output_dir
        ),
        hyperparameters=TrainingHyperparameters(
            learning_rate=learning_rate,
            batch_size=batch_size,
            kl_coef=kl_coef
        ),
        schedule=TrainingSchedule(
            num_episodes=num_episodes,
            eval_frequency=eval_frequency
        ),
        infrastructure=InfrastructureConfig(
            num_gpus=num_gpus,
            use_wandb=use_wandb
        )
    )
