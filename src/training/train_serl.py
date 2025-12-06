"""
MedSeRL Training Orchestrator

Main training loop integrating all components with OpenRLHF.
Includes SFT warm-up phase and RL training loop.

Requirements: 6.1-6.5, 7.1-7.6
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any

from src.data_processor import MedicalDataProcessor


# =============================================================================
# SFT Data Preparation
# Requirements: 6.1, 6.2, 6.5
# =============================================================================

@dataclass
class SFTExample:
    """A single SFT training example with input and target."""
    input_text: str
    target_text: str
    error_type: str
    text_id: str


def prepare_sft_data(
    data_processor: MedicalDataProcessor,
    system_prompt: Optional[str] = None
) -> List[SFTExample]:
    """
    Prepare SFT training data from MEDEC error samples.
    
    Filters MEDEC training data for error_flag=1 only and formats
    targets with <thinking> and <verdict> sections.
    
    Args:
        data_processor: MedicalDataProcessor with loaded MEDEC data
        system_prompt: Optional system prompt to prepend to inputs
        
    Returns:
        List of SFTExample objects ready for training
        
    Requirements:
        - 6.1: Use only MEDEC training samples with error_flag=1
        - 6.2: Format targets with <thinking> and <verdict> sections
        - 6.5: Include analysis of specific error type in thinking section
    """
    sft_examples = []
    
    # Get error pool (already filtered for error_flag=1 by data processor)
    # Requirements 6.1: Use only samples with error_flag=1
    error_pool = data_processor.get_error_pool()
    
    for entry in error_pool:
        text = entry.get('text', '')
        error_type = entry.get('error_type', 'Unknown')
        text_id = entry.get('text_id', '')
        
        if not text.strip():
            continue
        
        # Format input prompt
        input_text = format_sft_input(text, system_prompt)
        
        # Format target with <thinking> and <verdict> sections
        # Requirements 6.2, 6.5
        target_text = format_sft_target(error_type)
        
        sft_examples.append(SFTExample(
            input_text=input_text,
            target_text=target_text,
            error_type=error_type,
            text_id=text_id
        ))
    
    return sft_examples


def format_sft_input(
    clinical_note: str,
    system_prompt: Optional[str] = None
) -> str:
    """
    Format a clinical note as an SFT input prompt.
    
    Args:
        clinical_note: The clinical note text to analyze
        system_prompt: Optional system prompt to prepend
        
    Returns:
        Formatted input string for the model
    """
    default_system = (
        "You are a medical error detection assistant. Analyze the following "
        "clinical note for potential medical errors. Examine each of the five "
        "error types: Diagnosis, Management, Treatment, Pharmacotherapy, and "
        "Causal Organism. Provide your analysis in a <thinking> section, then "
        "give your verdict in a <verdict> section."
    )
    
    system = system_prompt or default_system
    
    return f"{system}\n\nClinical Note:\n{clinical_note}\n\nAnalysis:"


def format_sft_target(error_type: str) -> str:
    """
    Format the target output with <thinking> and <verdict> sections.
    
    The thinking section includes analysis of the specific error type
    as required by Requirement 6.5.
    
    Args:
        error_type: The type of error present in the note
        
    Returns:
        Formatted target string with thinking and verdict sections
        
    Requirements:
        - 6.2: Format with <thinking> and <verdict> sections
        - 6.5: Include analysis of specific error type in thinking
    """
    # Normalize error type for consistent output
    normalized_type = _normalize_error_type_for_output(error_type)
    
    # Generate thinking section that analyzes the specific error type
    # Requirements 6.5: Include analysis of specific error type
    thinking_content = _generate_thinking_content(normalized_type)
    
    return f"<thinking>\n{thinking_content}\n</thinking>\n<verdict>Error: {normalized_type}</verdict>"


def _normalize_error_type_for_output(error_type: str) -> str:
    """
    Normalize error type string for consistent output format.
    
    Handles variations like 'causalOrganism' -> 'Causal Organism'.
    """
    if not error_type:
        return "Unknown"
    
    # Map of known variations to canonical forms
    type_map = {
        'diagnosis': 'Diagnosis',
        'management': 'Management',
        'treatment': 'Treatment',
        'pharmacotherapy': 'Pharmacotherapy',
        'causalorganism': 'Causal Organism',
        'causal organism': 'Causal Organism',
        'causal_organism': 'Causal Organism',
    }
    
    normalized = error_type.lower().replace(' ', '').replace('_', '')
    return type_map.get(normalized, error_type)


def _generate_thinking_content(error_type: str) -> str:
    """
    Generate the content for the <thinking> section.
    
    Creates analysis text that examines each error type and identifies
    the specific error type present.
    
    Args:
        error_type: The normalized error type
        
    Returns:
        Thinking section content string
    """
    all_types = ['Diagnosis', 'Management', 'Treatment', 'Pharmacotherapy', 'Causal Organism']
    
    lines = []
    for et in all_types:
        if et == error_type:
            lines.append(f"Checking for {et} errors... Found a {et} error in this note.")
        else:
            lines.append(f"Checking for {et} errors... None identified.")
    
    return '\n'.join(lines)


def convert_sft_examples_to_hf_format(
    examples: List[SFTExample],
    tokenizer: Any = None
) -> List[Dict[str, str]]:
    """
    Convert SFT examples to HuggingFace dataset format.
    
    Args:
        examples: List of SFTExample objects
        tokenizer: Optional tokenizer for chat template formatting
        
    Returns:
        List of dicts with 'text' key containing full training text
    """
    hf_data = []
    
    for example in examples:
        # Combine input and target for causal LM training
        full_text = f"{example.input_text}\n{example.target_text}"
        
        hf_data.append({
            'text': full_text,
            'input': example.input_text,
            'output': example.target_text,
            'error_type': example.error_type,
            'text_id': example.text_id
        })
    
    return hf_data


# =============================================================================
# SFT Warm-Up Training
# Requirements: 6.3, 6.4
# =============================================================================

@dataclass
class SFTConfig:
    """Configuration for SFT warm-up training."""
    output_dir: str
    num_epochs: int = 1
    learning_rate: float = 2e-5
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    max_seq_length: int = 2048
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 10
    save_strategy: str = "epoch"
    fp16: bool = True
    bf16: bool = False


def sft_warmup(
    model_path: str,
    train_data: List[SFTExample],
    config: SFTConfig,
    tokenizer_path: Optional[str] = None,
    use_peft: bool = True,
    peft_config: Optional[Dict] = None
) -> str:
    """
    Perform SFT warm-up training on MedGemma-4B.
    
    Trains the model for exactly one epoch on error samples to learn
    the Chain of Thought format and error type definitions.
    
    Args:
        model_path: Path to the base model (MedGemma-4B)
        train_data: List of SFTExample objects for training
        config: SFT training configuration
        tokenizer_path: Path to tokenizer (defaults to model_path)
        use_peft: Whether to use PEFT/LoRA for efficient fine-tuning
        peft_config: Optional PEFT configuration dict
        
    Returns:
        Path to the saved checkpoint
        
    Requirements:
        - 6.3: Train for exactly one epoch
        - 6.4: Save the fine-tuned model checkpoint
    """
    try:
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            TrainingArguments,
            Trainer,
            DataCollatorForLanguageModeling
        )
        from datasets import Dataset
    except ImportError as e:
        raise ImportError(
            f"Required packages not installed: {e}. "
            "Install with: pip install transformers datasets"
        )
    
    # Load tokenizer
    tokenizer_path = tokenizer_path or model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto"
    )
    
    # Apply PEFT if requested
    if use_peft:
        try:
            from peft import LoraConfig, get_peft_model, TaskType
            
            default_peft_config = {
                'r': 16,
                'lora_alpha': 32,
                'lora_dropout': 0.05,
                'target_modules': ['q_proj', 'v_proj', 'k_proj', 'o_proj'],
                'task_type': TaskType.CAUSAL_LM
            }
            
            peft_cfg = peft_config or default_peft_config
            lora_config = LoraConfig(**peft_cfg)
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()
        except ImportError:
            print("Warning: PEFT not installed, training full model")
    
    # Convert training data to HuggingFace format
    hf_data = convert_sft_examples_to_hf_format(train_data)
    
    # Create dataset
    dataset = Dataset.from_list(hf_data)
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            truncation=True,
            max_length=config.max_seq_length,
            padding='max_length'
        )
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )
    
    # Create data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )
    
    # Create output directory
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Configure training arguments
    # Requirements 6.3: Train for exactly one epoch
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,  # Exactly 1 epoch per requirement
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        fp16=config.fp16,
        bf16=config.bf16,
        report_to="none",  # Disable W&B for SFT phase
        remove_unused_columns=False
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )
    
    # Train
    print(f"Starting SFT warm-up training for {config.num_epochs} epoch(s)...")
    print(f"Training on {len(train_data)} error samples")
    trainer.train()
    
    # Save checkpoint
    # Requirements 6.4: Save the fine-tuned model checkpoint
    checkpoint_path = os.path.join(config.output_dir, "sft_checkpoint")
    trainer.save_model(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    
    print(f"SFT warm-up complete. Checkpoint saved to: {checkpoint_path}")
    
    return checkpoint_path


def run_sft_warmup_from_medec(
    model_path: str,
    medec_data_path: str,
    output_dir: str,
    num_epochs: int = 1,
    **kwargs
) -> str:
    """
    Convenience function to run SFT warm-up directly from MEDEC data.
    
    Args:
        model_path: Path to base model
        medec_data_path: Path to MEDEC dataset
        output_dir: Directory for outputs
        num_epochs: Number of training epochs (default: 1)
        **kwargs: Additional arguments passed to SFTConfig
        
    Returns:
        Path to saved checkpoint
    """
    # Load training data (excludes test files)
    print(f"Loading MEDEC training data from {medec_data_path}...")
    data_processor = MedicalDataProcessor.load_training_data(
        data_path=medec_data_path
    )
    
    # Prepare SFT examples
    print("Preparing SFT training examples...")
    sft_examples = prepare_sft_data(data_processor)
    print(f"Prepared {len(sft_examples)} SFT examples from error pool")
    
    # Create config
    config = SFTConfig(
        output_dir=output_dir,
        num_epochs=num_epochs,
        **{k: v for k, v in kwargs.items() if hasattr(SFTConfig, k)}
    )
    
    # Run SFT warm-up
    return sft_warmup(
        model_path=model_path,
        train_data=sft_examples,
        config=config
    )


# =============================================================================
# RL Training Configuration
# Requirements: 7.6, 9.5
# =============================================================================

@dataclass
class RLTrainingConfig:
    """Configuration for RL training loop."""
    # Model paths
    model_path: str
    scribe_model_path: Optional[str] = None
    reference_model_path: Optional[str] = None

    # Data paths
    medec_data_path: str = "data_raw/MEDEC"
    output_dir: str = "outputs/rl"

    # Training hyperparameters
    num_episodes: int = 1000
    batch_size: int = 16  # Must be divisible by 4
    learning_rate: float = 1e-5
    kl_coef: float = 0.1
    gamma: float = 0.99
    clip_range: float = 0.2
    max_grad_norm: float = 1.0

    # Schedule
    eval_frequency: int = 50
    checkpoint_frequency: int = 100
    max_steps_per_episode: int = 512

    # Infrastructure
    num_gpus: int = 1
    num_vllm_engines: int = 1
    tensor_parallel_size: int = 1

    # Logging
    use_wandb: bool = False
    wandb_project: str = "medserl"
    wandb_run_name: Optional[str] = None

    # Flags
    use_mock_agents: bool = False  # For testing without GPU

    def __post_init__(self):
        if self.scribe_model_path is None:
            self.scribe_model_path = self.model_path
        if self.reference_model_path is None:
            self.reference_model_path = self.model_path
        if self.batch_size % 4 != 0:
            raise ValueError(
                f"batch_size must be divisible by 4, got {self.batch_size}"
            )


# =============================================================================
# Training Orchestrator
# Requirements: 7.6, 9.5
# =============================================================================

class MedSERLTrainer:
    """
    Main training orchestrator for MedSeRL.

    Initializes all components and manages the training loop including
    SFT warm-up and RL training phases.

    Requirements: 7.6, 9.5
    """

    def __init__(self, config: RLTrainingConfig):
        """
        Initialize the MedSeRL trainer with all components.

        Args:
            config: Training configuration

        Requirements: 7.6
        """
        self.config = config
        self.episode = 0

        # Initialize components
        self._init_data_processor()
        self._init_agents()
        self._init_logging()

        print("MedSeRL Trainer initialized")
        print(f"  Model: {config.model_path}")
        print(f"  Data: {config.medec_data_path}")
        print(f"  Output: {config.output_dir}")

    def _init_data_processor(self) -> None:
        """Initialize the data processor for MEDEC dataset."""
        print("Loading MEDEC training data...")
        self.data_processor = MedicalDataProcessor.load_training_data(
            data_path=self.config.medec_data_path
        )
        print(f"  Error pool: {len(self.data_processor.error_pool)} samples")
        print(f"  Clean pool: {len(self.data_processor.clean_pool)} samples")

    def _init_agents(self) -> None:
        """Initialize Scribe and Doctor agents."""
        from src.agents.scribe_agent import create_scribe_agent
        from src.agents.doctor_agent import create_doctor_agent

        print("Initializing agents...")

        # Initialize Scribe Agent
        self.scribe_agent = create_scribe_agent(
            model_path=self.config.scribe_model_path,
            use_mock=self.config.use_mock_agents,
            lazy_init=True  # Defer GPU allocation
        )

        # Initialize Doctor Agent
        self.doctor_agent = create_doctor_agent(
            model_path=self.config.model_path,
            use_mock=self.config.use_mock_agents,
            lazy_init=True
        )

        print(f"  Scribe: {self.scribe_agent}")
        print(f"  Doctor: {self.doctor_agent}")

    def _init_logging(self) -> None:
        """Initialize logging (W&B if configured).
        
        Requirements: 8.4
        """
        from src.training.logging_utils import create_wandb_logger

        self.wandb_logger = create_wandb_logger(
            enabled=self.config.use_wandb,
            project=self.config.wandb_project,
            run_name=self.config.wandb_run_name,
            config=vars(self.config)
        )

        # Keep backward compatibility
        self.wandb_run = self.wandb_logger.run if self.wandb_logger.is_enabled else None

        if self.wandb_logger.is_enabled:
            print(f"  W&B: {self.config.wandb_project}")
        elif self.config.use_wandb:
            print("  W&B: Not available (wandb not installed)")

    def run_sft_warmup(self) -> str:
        """
        Run SFT warm-up phase.

        Returns:
            Path to the saved SFT checkpoint
        """
        print("\n" + "=" * 60)
        print("Starting SFT Warm-Up Phase")
        print("=" * 60)

        sft_examples = prepare_sft_data(self.data_processor)
        print(f"Prepared {len(sft_examples)} SFT examples")

        sft_config = SFTConfig(
            output_dir=os.path.join(self.config.output_dir, "sft"),
            num_epochs=1
        )

        checkpoint_path = sft_warmup(
            model_path=self.config.model_path,
            train_data=sft_examples,
            config=sft_config
        )

        # Update model path to use SFT checkpoint
        self.config.model_path = checkpoint_path
        print(f"SFT complete. Using checkpoint: {checkpoint_path}")

        return checkpoint_path

    def run_rl_training(self) -> None:
        """
        Run the main RL training loop.

        Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
        """
        print("\n" + "=" * 60)
        print("Starting RL Training Loop")
        print("=" * 60)

        # Create output directories
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        checkpoint_dir = os.path.join(self.config.output_dir, "checkpoints")
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

        for episode in range(self.config.num_episodes):
            self.episode = episode

            # Run one training episode
            metrics = self._run_episode()

            # Log metrics
            self._log_metrics(metrics, episode)

            # Evaluation
            if (episode + 1) % self.config.eval_frequency == 0:
                eval_metrics = self._evaluate()
                self._log_metrics(eval_metrics, episode, prefix="eval/")

            # Checkpoint
            if (episode + 1) % self.config.checkpoint_frequency == 0:
                self._save_checkpoint(episode, checkpoint_dir, metrics)

            # Progress update
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1}/{self.config.num_episodes} - "
                      f"Reward: {metrics.get('mean_reward', 0):.3f}")

        print("\nRL Training complete!")

    def _run_episode(self) -> Dict[str, float]:
        """
        Run a single training episode.

        Returns:
            Dictionary of training metrics

        Requirements: 7.1, 7.2, 7.3, 7.4
        """
        from src.training.reward_engine import calculate_reward

        # 7.1: Generate batch using Scribe Agent
        batch_prompts = self.data_processor.get_quadrant_batch(
            batch_size=self.config.batch_size
        )

        # Transform with Scribe Agent
        transformed_batch = self.scribe_agent.transform_batch(batch_prompts)

        # 7.2: Execute rollouts with Doctor Agent
        notes = [item["transformed_text"] for item in transformed_batch]
        doctor_outputs = self.doctor_agent.analyze_batch(notes)

        # 7.3: Compute rewards
        rewards = []
        for output, item in zip(doctor_outputs, transformed_batch):
            ground_truth = item["ground_truth"]
            reward = calculate_reward(output, ground_truth)
            rewards.append(reward)

        # 7.4: Weight update would happen here via OpenRLHF
        # This is handled by the OpenRLHF integration script

        # Compute metrics
        metrics = {
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0,
            "max_reward": max(rewards) if rewards else 0,
            "min_reward": min(rewards) if rewards else 0,
            "batch_size": len(rewards),
            "episode": self.episode
        }

        return metrics

    def _evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on test data.

        Returns:
            Dictionary of evaluation metrics
        """
        # Load test data
        test_processor = MedicalDataProcessor.load_test_data(
            data_path=self.config.medec_data_path
        )

        # Sample from test data
        test_samples = test_processor.error_pool[:50] + test_processor.clean_pool[:50]

        from src.training.reward_engine import calculate_reward

        correct = 0
        total = 0
        rewards = []

        for sample in test_samples:
            note = sample["text"]
            has_error = sample.get("label") == "Error"
            error_type = sample.get("error_type")

            output = self.doctor_agent.analyze_note(note)
            ground_truth = {"has_error": has_error, "error_type": error_type}

            reward = calculate_reward(output, ground_truth)
            rewards.append(reward)

            # Check if prediction is correct
            prediction = self.doctor_agent.parse_prediction(output)
            pred_has_error = prediction["predicted_label"] == "Error"
            if pred_has_error == has_error:
                correct += 1
            total += 1

        return {
            "accuracy": correct / total if total > 0 else 0,
            "mean_reward": sum(rewards) / len(rewards) if rewards else 0,
            "num_samples": total
        }

    def _log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "train/"
    ) -> None:
        """Log metrics to W&B if configured.
        
        Requirements: 8.4
        """
        if hasattr(self, 'wandb_logger') and self.wandb_logger:
            self.wandb_logger.log_metrics(metrics, step=step, prefix=prefix)
        elif self.wandb_run:
            # Backward compatibility
            import wandb
            wandb.log({f"{prefix}{k}": v for k, v in metrics.items()}, step=step)

    def _save_checkpoint(
        self,
        episode: int,
        checkpoint_dir: str,
        metrics: Optional[Dict[str, float]] = None
    ) -> str:
        """
        Save a training checkpoint with episode number in filename.

        Args:
            episode: Current episode number
            checkpoint_dir: Directory to save checkpoint
            metrics: Optional metrics to save with checkpoint

        Returns:
            Path to saved checkpoint

        Requirements: 8.5, 8.6
        """
        from src.training.checkpoint import save_checkpoint

        # Get model and tokenizer if available (for real training)
        model = None
        tokenizer = None

        # Try to get model from doctor agent if it has one
        if hasattr(self.doctor_agent, 'llm') and self.doctor_agent.llm is not None:
            try:
                # For vLLM, we can't directly save - would need the underlying model
                # This is a placeholder for when using HuggingFace models directly
                pass
            except Exception:
                pass

        return save_checkpoint(
            checkpoint_dir=checkpoint_dir,
            episode=episode,
            model=model,
            tokenizer=tokenizer,
            config=vars(self.config),
            metrics=metrics,
            total_episodes=self.config.num_episodes,
            model_path=self.config.model_path
        )


# =============================================================================
# Standalone RL Training Loop Function
# Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
# =============================================================================

def rl_training_loop(
    data_processor: MedicalDataProcessor,
    scribe_agent: Any,
    doctor_agent: Any,
    reward_fn: callable,
    num_episodes: int = 1000,
    batch_size: int = 16,
    eval_frequency: int = 50,
    checkpoint_frequency: int = 100,
    output_dir: str = "outputs/rl",
    use_wandb: bool = False
) -> Dict[str, List[float]]:
    """
    Standalone RL training loop function.

    This function implements the core RL training loop that:
    1. Generates batches using the Scribe Agent
    2. Executes rollouts with the Doctor Agent
    3. Computes rewards using the reward function
    4. (Weight updates are handled by OpenRLHF integration)

    Args:
        data_processor: MedicalDataProcessor with loaded MEDEC data
        scribe_agent: ScribeAgent for data generation
        doctor_agent: DoctorAgent for policy rollouts
        reward_fn: Reward function (calculate_reward)
        num_episodes: Number of training episodes
        batch_size: Batch size (must be divisible by 4)
        eval_frequency: Evaluate every N episodes
        checkpoint_frequency: Save checkpoint every N episodes
        output_dir: Output directory for checkpoints
        use_wandb: Whether to log to Weights & Biases

    Returns:
        Dictionary containing training history (rewards, metrics)

    Requirements: 7.1, 7.2, 7.3, 7.4, 7.5
    """
    # Create output directories
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    # Training history
    history = {
        "episode": [],
        "mean_reward": [],
        "max_reward": [],
        "min_reward": [],
        "eval_accuracy": [],
        "eval_reward": []
    }

    # Initialize W&B if requested
    wandb_run = None
    if use_wandb:
        try:
            import wandb
            wandb_run = wandb.init(project="medserl", name="rl_training")
        except ImportError:
            print("Warning: wandb not installed, skipping logging")

    print(f"Starting RL training for {num_episodes} episodes")
    print(f"Batch size: {batch_size}, Eval frequency: {eval_frequency}")

    for episode in range(num_episodes):
        # 7.1: Generate batch using Scribe Agent
        batch_prompts = data_processor.get_quadrant_batch(batch_size=batch_size)

        # Transform with Scribe Agent
        transformed_batch = scribe_agent.transform_batch(batch_prompts)

        # 7.2: Execute rollouts with Doctor Agent
        notes = [item["transformed_text"] for item in transformed_batch]
        doctor_outputs = doctor_agent.analyze_batch(notes)

        # 7.3: Compute rewards
        rewards = []
        for output, item in zip(doctor_outputs, transformed_batch):
            ground_truth = item["ground_truth"]
            reward = reward_fn(output, ground_truth)
            rewards.append(reward)

        # 7.4, 7.5: Weight updates handled by OpenRLHF
        # In standalone mode, we just track metrics

        # Record metrics
        mean_reward = sum(rewards) / len(rewards) if rewards else 0
        history["episode"].append(episode)
        history["mean_reward"].append(mean_reward)
        history["max_reward"].append(max(rewards) if rewards else 0)
        history["min_reward"].append(min(rewards) if rewards else 0)

        # Log to W&B
        if wandb_run:
            import wandb
            wandb.log({
                "train/mean_reward": mean_reward,
                "train/max_reward": max(rewards) if rewards else 0,
                "train/min_reward": min(rewards) if rewards else 0,
                "train/episode": episode
            }, step=episode)

        # Progress update
        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{num_episodes} - "
                  f"Mean Reward: {mean_reward:.3f}")

        # 7.5: Increment episode counter (implicit in loop)

    print(f"\nRL training complete! Final mean reward: {history['mean_reward'][-1]:.3f}")

    if wandb_run:
        wandb_run.finish()

    return history


def main(
    model_path: str,
    medec_path: str = "data_raw/MEDEC",
    output_dir: str = "outputs/medserl",
    num_episodes: int = 1000,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    kl_coef: float = 0.1,
    eval_frequency: int = 50,
    checkpoint_frequency: int = 100,
    use_wandb: bool = False,
    use_mock: bool = False,
    skip_sft: bool = False,
    **kwargs
) -> MedSERLTrainer:
    """
    Main entry point for MedSeRL training.

    Initializes all components and runs the training pipeline including
    SFT warm-up and RL training phases.

    Args:
        model_path: Path to base model (MedGemma-4B)
        medec_path: Path to MEDEC dataset
        output_dir: Output directory for checkpoints
        num_episodes: Number of RL training episodes
        batch_size: Batch size (must be divisible by 4)
        learning_rate: Learning rate for policy optimization
        kl_coef: KL divergence coefficient
        eval_frequency: Evaluate every N episodes
        checkpoint_frequency: Save checkpoint every N episodes
        use_wandb: Whether to log to Weights & Biases
        use_mock: Use mock agents for testing
        skip_sft: Skip SFT warm-up phase
        **kwargs: Additional arguments

    Returns:
        The trainer instance

    Requirements: 7.6, 9.5
    """
    # Create configuration
    config = RLTrainingConfig(
        model_path=model_path,
        medec_data_path=medec_path,
        output_dir=output_dir,
        num_episodes=num_episodes,
        batch_size=batch_size,
        learning_rate=learning_rate,
        kl_coef=kl_coef,
        eval_frequency=eval_frequency,
        checkpoint_frequency=checkpoint_frequency,
        use_wandb=use_wandb,
        use_mock_agents=use_mock
    )

    # Initialize trainer
    trainer = MedSERLTrainer(config)

    # Run SFT warm-up (unless skipped)
    if not skip_sft:
        trainer.run_sft_warmup()

    # Run RL training
    trainer.run_rl_training()

    return trainer


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="MedSeRL Training")
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to base model (MedGemma-4B)")
    parser.add_argument("--medec_path", type=str, default="data_raw/MEDEC",
                        help="Path to MEDEC dataset")
    parser.add_argument("--output_dir", type=str, default="outputs/medserl",
                        help="Output directory for checkpoints")

    # Training mode
    parser.add_argument("--sft_only", action="store_true",
                        help="Run only SFT warm-up phase")
    parser.add_argument("--skip_sft", action="store_true",
                        help="Skip SFT warm-up phase")

    # Hyperparameters
    parser.add_argument("--num_episodes", type=int, default=1000,
                        help="Number of RL training episodes")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size (must be divisible by 4)")
    parser.add_argument("--learning_rate", type=float, default=1e-5,
                        help="Learning rate")
    parser.add_argument("--kl_coef", type=float, default=0.1,
                        help="KL divergence coefficient")

    # Schedule
    parser.add_argument("--eval_frequency", type=int, default=50,
                        help="Evaluate every N episodes")
    parser.add_argument("--checkpoint_frequency", type=int, default=100,
                        help="Save checkpoint every N episodes")

    # Logging
    parser.add_argument("--use_wandb", action="store_true",
                        help="Log to Weights & Biases")

    # Testing
    parser.add_argument("--use_mock", action="store_true",
                        help="Use mock agents for testing")

    args = parser.parse_args()

    if args.sft_only:
        # Run only SFT warm-up
        checkpoint = run_sft_warmup_from_medec(
            model_path=args.model_path,
            medec_data_path=args.medec_path,
            output_dir=args.output_dir
        )
        print(f"SFT checkpoint saved to: {checkpoint}")
    else:
        # Run full training pipeline
        main(
            model_path=args.model_path,
            medec_path=args.medec_path,
            output_dir=args.output_dir,
            num_episodes=args.num_episodes,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            kl_coef=args.kl_coef,
            eval_frequency=args.eval_frequency,
            checkpoint_frequency=args.checkpoint_frequency,
            use_wandb=args.use_wandb,
            use_mock=args.use_mock,
            skip_sft=args.skip_sft
        )
