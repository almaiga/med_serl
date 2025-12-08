#!/usr/bin/env python3
"""
MedSeRL GPU Training Script

Real training script for RunPod/GPU environments.
Supports configurable dataset sizes for quick testing before full runs.

Usage:
    # Quick test with 256 samples
    python scripts/train_gpu.py --num_samples 256 --sft_epochs 1 --rl_episodes 10

    # Full training
    python scripts/train_gpu.py --num_samples -1 --sft_epochs 1 --rl_episodes 100

    # Resume from checkpoint
    python scripts/train_gpu.py --resume_from outputs/gpu_training/sft/sft_checkpoint
"""

import argparse
import os
import sys
import warnings
from pathlib import Path
from datetime import datetime

# Suppress warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.*deprecated.*")

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def check_gpu():
    """Check GPU availability and print info."""
    import torch

    print("=" * 60)
    print("GPU Check")
    print("=" * 60)

    if torch.cuda.is_available():
        print("CUDA available: Yes")
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024**3
            print(f"  GPU {i}: {props.name} ({mem_gb:.1f} GB)")
        return "cuda"
    else:
        print("CUDA not available, using CPU (training will be slow)")
        return "cpu"


def load_model_and_tokenizer(model_path: str, use_4bit: bool = False):
    """Load model with appropriate quantization for GPU memory."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Quantization config for memory efficiency
    if use_4bit:
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

    print(f"Model loaded: {model.config._name_or_path}")
    print(f"Parameters: {model.num_parameters():,}")

    return model, tokenizer


def apply_lora(model, r: int = 16, alpha: int = 32):
    """Apply LoRA for efficient fine-tuning."""
    from peft import LoraConfig, get_peft_model, TaskType

    config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        task_type=TaskType.CAUSAL_LM,
    )

    model = get_peft_model(model, config)
    model.print_trainable_parameters()
    return model


def run_sft_phase(
    model,
    tokenizer,
    processor,
    output_dir: str,
    num_samples: int = -1,
    num_epochs: int = 1,
    batch_size: int = 2,
    gradient_accumulation: int = 8,
    learning_rate: float = 2e-5,
    max_length: int = 1024,
):
    """Run SFT warm-up phase."""
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from datasets import Dataset
    from src.training.train_serl import prepare_sft_data, convert_sft_examples_to_hf_format

    print("\n" + "=" * 60)
    print("Phase 1: SFT Warm-Up")
    print("=" * 60)

    # Prepare data with contrastive pairs (INCORRECT + CORRECT from same note)
    sft_examples = prepare_sft_data(processor)
    print(f"Total contrastive pairs: {len(sft_examples)}")

    # Count CORRECT vs INCORRECT
    correct_count = sum(1 for ex in sft_examples if ex.error_type == "Correct")
    incorrect_count = len(sft_examples) - correct_count
    print(f"  INCORRECT samples: {incorrect_count}")
    print(f"  CORRECT samples: {correct_count}")

    if num_samples > 0 and num_samples < len(sft_examples):
        # Sample subset - balance if both types available, otherwise use what we have
        import random
        incorrect = [ex for ex in sft_examples if ex.error_type != "Correct"]
        correct = [ex for ex in sft_examples if ex.error_type == "Correct"]
        
        if correct and incorrect:
            # Balance between correct and incorrect
            half = num_samples // 2
            sft_examples = random.sample(incorrect, min(half, len(incorrect))) + \
                           random.sample(correct, min(half, len(correct)))
        else:
            # Only one type available - use all of that type up to num_samples
            available = incorrect or correct
            sft_examples = random.sample(available, min(num_samples, len(available)))
        
        random.shuffle(sft_examples)
        print(f"Using {len(sft_examples)} samples")

    # Convert to HF format
    hf_data = convert_sft_examples_to_hf_format(sft_examples)
    dataset = Dataset.from_list(hf_data)

    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

    # Training args
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=gradient_accumulation,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
        dataloader_num_workers=4,
        remove_unused_columns=False,
    )

    # Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("\nStarting SFT training...")
    trainer.train()

    # Save
    checkpoint_path = os.path.join(output_dir, "sft_checkpoint")
    trainer.save_model(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    print(f"SFT checkpoint saved: {checkpoint_path}")

    return checkpoint_path


def run_rl_phase(
    model,
    tokenizer,
    processor,
    output_dir: str,
    num_episodes: int = 100,
    batch_size: int = 16,
    learning_rate: float = 1e-5,
    max_length: int = 1024,
    log_interactions: bool = True,
):
    """
    Run RL training phase with actual gradient updates.

    Uses a simplified REINFORCE implementation for single-GPU training.
    For multi-GPU, use the OpenRLHF script instead.
    """
    import torch
    from torch.optim import AdamW
    from torch.amp import autocast, GradScaler
    from src.training.reward_engine import calculate_reward
    from src.training.interaction_logger import InteractionLogger

    print("\n" + "=" * 60)
    print("Phase 2: RL Training (REINFORCE)")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Batch size: {batch_size}")

    device = next(model.parameters()).device

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    scaler = GradScaler('cuda')

    # Setup interaction logger
    interaction_logger = None
    if log_interactions:
        interaction_logger = InteractionLogger(
            output_dir=os.path.join(output_dir, "interactions"),
            session_name=datetime.now().strftime("%Y%m%d_%H%M%S"),
        )

    # Training loop
    model.train()
    total_rewards = []

    for episode in range(num_episodes):
        # Get batch from balanced quadrant
        batch = processor.get_quadrant_batch(batch_size=batch_size)

        episode_rewards = []
        episode_loss = 0.0

        for i, sample in enumerate(batch):
            note = sample.get("original_text", sample.get("text", ""))
            ground_truth = sample.get("meta", sample.get("ground_truth", {}))
            quadrant = sample.get("quadrant", "unknown")

            # Format prompt
            prompt = format_prompt(note)

            # Tokenize
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            ).to(device)

            # Generate with gradient tracking
            with autocast('cuda', dtype=torch.bfloat16):
                # Forward pass to get logits
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=256,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                )

            # Decode output
            generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)

            # Calculate reward
            reward = calculate_reward(generated_text, ground_truth)
            episode_rewards.append(reward)

            # Log interaction
            if interaction_logger:
                interaction_logger.log_interaction(
                    episode=episode,
                    batch_idx=i,
                    note=note[:500],
                    quadrant=quadrant,
                    ground_truth=ground_truth,
                    model_output=generated_text,
                    reward=reward,
                )

            # REINFORCE loss: -reward * log_prob
            # Simplified: use cross-entropy on generated tokens weighted by reward
            if reward != 0:
                with autocast('cuda', dtype=torch.bfloat16):
                    # Get log probs for generated sequence
                    labels = outputs.sequences.clone()
                    labels[:, :inputs["input_ids"].shape[1]] = -100  # Mask prompt

                    loss_outputs = model(
                        input_ids=outputs.sequences,
                        labels=labels,
                    )
                    # Weight loss by negative reward (maximize reward)
                    weighted_loss = -reward * loss_outputs.loss

                scaler.scale(weighted_loss).backward()
                episode_loss += weighted_loss.item()

        # Update weights
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

        # Track metrics
        mean_reward = sum(episode_rewards) / len(episode_rewards)
        total_rewards.append(mean_reward)

        # Log progress
        if (episode + 1) % 10 == 0 or episode == 0:
            print(
                f"Episode {episode + 1}/{num_episodes} | "
                f"Mean Reward: {mean_reward:.3f} | "
                f"Loss: {episode_loss:.4f}"
            )

        # Save checkpoint periodically
        if (episode + 1) % 50 == 0:
            ckpt_path = os.path.join(output_dir, f"rl_checkpoint_ep{episode + 1}")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    # Final save
    final_path = os.path.join(output_dir, "rl_final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    if interaction_logger:
        interaction_logger.close()

    print(f"\nRL training complete. Final checkpoint: {final_path}")
    print(f"Mean reward over training: {sum(total_rewards) / len(total_rewards):.3f}")

    return final_path


def format_prompt(note: str) -> str:
    """Format clinical note as prompt."""
    return f"""You are a medical error detection assistant. Analyze the following clinical note and determine if it contains any medical errors. Consider diagnosis, management, treatment, pharmacotherapy, and causal organism. Provide your reasoning in <think> tags, then your final answer as CORRECT or INCORRECT in <answer> tags.

Clinical Note:
{note}

"""


def main():
    parser = argparse.ArgumentParser(description="MedSeRL GPU Training")

    # Model
    parser.add_argument(
        "--model_path",
        type=str,
        default="google/medgemma-4b-it",
        help="Model path or HuggingFace name",
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Resume from checkpoint path",
    )

    # Data
    parser.add_argument(
        "--num_samples",
        type=int,
        default=512,
        help="Number of SFT samples (-1 for all contrastive pairs)",
    )
    parser.add_argument(
        "--medec_path",
        type=str,
        default="data_raw/MEDEC",
        help="Path to MEDEC dataset",
    )

    # Training
    parser.add_argument("--sft_epochs", type=int, default=3, help="SFT epochs (default: 3)")
    parser.add_argument("--rl_episodes", type=int, default=100, help="RL episodes")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")

    # Options
    parser.add_argument("--use_4bit", action="store_true", help="Use 4-bit quantization")
    parser.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA")
    parser.add_argument("--skip_sft", action="store_true", help="Skip SFT phase")
    parser.add_argument("--skip_rl", action="store_true", help="Skip RL phase")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/gpu_training",
        help="Output directory",
    )

    args = parser.parse_args()

    # Check GPU
    device = check_gpu()

    # Create output dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    print("\nLoading MEDEC data...")
    from src.data_processor import MedicalDataProcessor

    processor = MedicalDataProcessor.load_training_data(data_path=args.medec_path)
    print(f"Error pool: {len(processor.error_pool)} samples")
    print(f"Clean pool: {len(processor.clean_pool)} samples")

    total_available = len(processor.error_pool) + len(processor.clean_pool)
    if args.num_samples > 0:
        print(f"Using {args.num_samples} samples (of {total_available} available)")
    else:
        print(f"Using all {total_available} samples")

    # Load model
    model_path = args.resume_from or args.model_path
    model, tokenizer = load_model_and_tokenizer(model_path, use_4bit=args.use_4bit)

    # Apply LoRA
    if args.use_lora and not args.resume_from:
        print("\nApplying LoRA...")
        model = apply_lora(model)

    # SFT Phase
    if not args.skip_sft and not args.resume_from:
        sft_output = os.path.join(args.output_dir, "sft")
        run_sft_phase(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            output_dir=sft_output,
            num_samples=args.num_samples,
            num_epochs=args.sft_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
        )

    # RL Phase
    if not args.skip_rl:
        rl_output = os.path.join(args.output_dir, "rl")
        run_rl_phase(
            model=model,
            tokenizer=tokenizer,
            processor=processor,
            output_dir=rl_output,
            num_episodes=args.rl_episodes,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate / 10,  # Lower LR for RL
        )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
