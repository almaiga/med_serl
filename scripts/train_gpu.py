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


def compute_approx_kl(log_probs, log_probs_base, kl_estimator="k1"):
    """
    Compute the approximate KL divergence between two distributions.
    Copied from SeRL: openrlhf/models/utils.py
    Schulman blog: http://joschu.net/blog/kl-approx.html

    Args:
        log_probs: Log probabilities of the new distribution.
        log_probs_base: Log probabilities of the base distribution.
    """
    if kl_estimator == "k1":
        # k1: Simple log ratio (can be negative for individual tokens)
        log_ratio = log_probs.float() - log_probs_base.float()

    # The k2 estimator is the non negative kl approximation
    elif kl_estimator == "k2":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = log_ratio**2 / 2.0

    # The k3 estimator is the non negative kl approximation
    # This is always >= 0
    elif kl_estimator == "k3":
        log_ratio = log_probs.float() - log_probs_base.float()
        log_ratio = -log_ratio
        log_ratio = log_ratio.exp() - 1 - log_ratio
    else:
        raise ValueError(f"Unknown kl_estimator: {kl_estimator}")

    return log_ratio


def log_probs_from_logits(logits, labels, temperature=1.0):
    """
    Compute log probabilities from logits for given labels.
    Copied from SeRL: openrlhf/models/utils.py
    """
    import torch.nn.functional as F

    if temperature != 1.0:
        logits = logits / temperature

    # Compute log softmax and gather the log probs for the labels
    log_probs = F.log_softmax(logits, dim=-1)
    log_probs_labels = log_probs.gather(
        dim=-1, index=labels.unsqueeze(-1)
    ).squeeze(-1)

    return log_probs_labels


def masked_mean(tensor, mask, dim=None):
    """
    Compute masked mean.
    Copied from SeRL: openrlhf/models/utils.py
    """
    if mask is None:
        return tensor.mean(dim=dim)
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1e-8)


def run_rl_phase(
    model,
    tokenizer,
    processor,
    output_dir: str,
    num_episodes: int = 100,
    batch_size: int = 16,
    learning_rate: float = 5e-7,
    kl_coef: float = 1e-4,
    max_length: int = 1024,
    log_interactions: bool = True,
):
    """
    Run RL training with REINFORCE++ (baseline + KL penalty).

    Based on SeRL implementation with proper:
    1. Per-token log probability computation
    2. KL divergence using log_probs - ref_log_probs (k1 estimator)
    3. KL penalty integrated into reward signal
    4. Advantage normalization across batch

    SeRL parameters: lr=5e-7, kl_coef=1e-4
    """
    import json
    import time
    import torch
    import copy
    from torch.optim import AdamW
    from tqdm import tqdm
    from src.training.reward_engine import calculate_reward
    from src.training.interaction_logger import InteractionLogger

    print("\n" + "=" * 60)
    print("Phase 2: RL Training (REINFORCE++)")
    print("=" * 60)
    print(f"Episodes: {num_episodes}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"KL coefficient: {kl_coef}")

    device = next(model.parameters()).device

    # Create reference model (frozen copy for KL penalty)
    print("Creating reference model for KL penalty...")
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    # Setup optimizer
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # Setup interaction logger
    interaction_logger = None
    if log_interactions:
        interaction_logger = InteractionLogger(
            output_dir=os.path.join(output_dir, "interactions"),
            session_name=datetime.now().strftime("%Y%m%d_%H%M%S"),
        )

    # Setup metrics logger
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metrics_file = os.path.join(output_dir, "rl_metrics.jsonl")
    metrics_summary_file = os.path.join(output_dir, "rl_summary.json")

    # Training state
    model.train()
    total_rewards = []
    all_metrics = []
    start_time = time.time()

    # Progress bar for episodes
    pbar = tqdm(range(num_episodes), desc="RL Training", unit="ep")

    for episode in pbar:
        episode_start = time.time()
        batch = processor.get_quadrant_batch(batch_size=batch_size)

        episode_rewards = []
        batch_data = []

        # === ROLLOUT PHASE: Generate responses and compute ref log probs ===
        for i, sample in enumerate(batch):
            note = sample.get("original_text", sample.get("text", ""))
            ground_truth = sample.get("meta", sample.get("ground_truth", {}))
            quadrant = sample.get("quadrant", "unknown")

            prompt = format_prompt_for_chat(note, tokenizer)
            inputs = tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding=True,
            ).to(device)

            prompt_len = inputs["input_ids"].shape[1]

            # Get stop token IDs for </answer>
            stop_strings = ["</answer>", "```", "\n\n\n"]
            
            # Generate response
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,  # Increased for reasoning
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    stop_strings=stop_strings,
                    tokenizer=tokenizer,  # Required for stop_strings
                )

            sequences = outputs
            generated_ids = sequences[0][prompt_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
            # Clean up generated text - stop at </answer> if present
            if "</answer>" in generated_text:
                generated_text = generated_text.split("</answer>")[0] + "</answer>"
            elif "```" in generated_text:
                generated_text = generated_text.split("```")[0].strip()

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

            # Create action mask
            seq_len = sequences.shape[1]
            action_mask = torch.zeros(1, seq_len - 1, device=device)
            response_len = seq_len - prompt_len
            if response_len > 0:
                action_mask[0, prompt_len - 1:seq_len - 1] = 1.0

            # Compute reference log probs (no grad, done during rollout)
            input_seq = sequences[:, :-1]
            target_seq = sequences[:, 1:]

            with torch.no_grad():
                ref_outputs = ref_model(input_ids=input_seq)
                ref_logits = ref_outputs.logits.float()
                ref_log_probs = log_probs_from_logits(ref_logits, target_seq)

            batch_data.append({
                'sequences': sequences,
                'action_mask': action_mask,
                'ref_log_probs': ref_log_probs,
                'reward': reward,
            })

            # Clear cache after each sample
            torch.cuda.empty_cache()

        # === COMPUTE ADVANTAGES ===
        rewards_tensor = torch.tensor(
            [d['reward'] for d in batch_data],
            device=device, dtype=torch.float32
        )
        advantages = rewards_tensor - rewards_tensor.mean()
        if len(advantages) > 1:
            adv_std = advantages.std().clamp(min=1e-8)
            advantages = advantages / adv_std

        # === POLICY UPDATE (one sample at a time to save memory) ===
        optimizer.zero_grad()
        total_kl_value = 0.0

        for idx, (data, advantage) in enumerate(zip(batch_data, advantages)):
            sequences = data['sequences']
            action_mask = data['action_mask']
            ref_log_probs = data['ref_log_probs']

            # Forward pass
            input_seq = sequences[:, :-1]
            target_seq = sequences[:, 1:]

            policy_outputs = model(input_ids=input_seq)
            policy_logits = policy_outputs.logits.float()
            policy_log_probs = log_probs_from_logits(policy_logits, target_seq)

            # Compute KL
            kl = compute_approx_kl(policy_log_probs, ref_log_probs, kl_estimator="k1")

            # Masked log probs
            masked_log_probs = policy_log_probs * action_mask
            response_log_prob = masked_log_probs.sum()

            # Mean KL
            kl_mean = masked_mean(kl, action_mask, dim=-1).mean()

            # REINFORCE loss with KL penalty
            loss = (-advantage * response_log_prob + kl_coef * kl_mean) / batch_size
            loss.backward()

            total_kl_value += kl_mean.item()

        # Gradient clipping and update
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Clear cache
        torch.cuda.empty_cache()

        # === METRICS TRACKING ===
        episode_time = time.time() - episode_start
        mean_reward = sum(episode_rewards) / len(episode_rewards)
        total_rewards.append(mean_reward)
        avg_kl = total_kl_value / batch_size
        running_avg_reward = sum(total_rewards) / len(total_rewards)

        # Log metrics
        episode_metrics = {
            "episode": episode,
            "mean_reward": mean_reward,
            "avg_kl": avg_kl,
            "running_avg_reward": running_avg_reward,
            "episode_time_sec": episode_time,
            "total_time_sec": time.time() - start_time,
            "rewards": episode_rewards,
        }
        all_metrics.append(episode_metrics)

        # Append to metrics file
        with open(metrics_file, "a") as f:
            f.write(json.dumps(episode_metrics) + "\n")

        # Update progress bar
        pbar.set_postfix({
            'reward': f'{mean_reward:.3f}',
            'kl': f'{avg_kl:.4f}',
            'avg_r': f'{running_avg_reward:.3f}'
        })

        # Save checkpoint periodically
        if (episode + 1) % 50 == 0:
            ckpt_path = os.path.join(output_dir, f"rl_checkpoint_ep{episode + 1}")
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    # Close progress bar
    pbar.close()

    # Final save
    final_path = os.path.join(output_dir, "rl_final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    if interaction_logger:
        interaction_logger.close()

    # Save summary metrics
    total_time = time.time() - start_time
    summary = {
        "total_episodes": num_episodes,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "kl_coef": kl_coef,
        "final_avg_reward": sum(total_rewards) / len(total_rewards) if total_rewards else 0,
        "best_episode_reward": max(total_rewards) if total_rewards else 0,
        "total_time_sec": total_time,
        "avg_time_per_episode": total_time / num_episodes,
        "reward_history": total_rewards,
    }
    with open(metrics_summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    # Clean up reference model
    del ref_model
    torch.cuda.empty_cache()

    avg_reward = summary["final_avg_reward"]
    print(f"\nRL training complete. Final checkpoint: {final_path}")
    print(f"Mean reward over training: {avg_reward:.3f}")
    print(f"Metrics saved to: {metrics_file}")

    return final_path


def format_prompt_for_chat(note: str, tokenizer) -> str:
    """Format clinical note using MedGemma chat template."""
    messages = [
        {
            "role": "system",
            "content": "You are a medical error detection assistant. Analyze clinical notes for errors in diagnosis, management, treatment, pharmacotherapy, or causal organism identification. Respond with your reasoning in <think> tags, then your verdict in <answer> tags as CORRECT or INCORRECT."
        },
        {
            "role": "user", 
            "content": f"""Analyze this clinical note for medical errors:

{note}

Respond with:
<think>[Your analysis]</think>
<answer>CORRECT or INCORRECT</answer>"""
        }
    ]
    
    # Use chat template if available
    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
    else:
        # Fallback for tokenizers without chat template
        return f"""<start_of_turn>system
{messages[0]['content']}<end_of_turn>
<start_of_turn>user
{messages[1]['content']}<end_of_turn>
<start_of_turn>model
"""


def format_prompt(note: str) -> str:
    """Legacy format - kept for compatibility but prefer format_prompt_for_chat."""
    return f"""You are a medical error detection assistant. Analyze the following clinical note for medical errors.

Clinical Note:
{note}

Respond with <think>[analysis]</think> then <answer>CORRECT</answer> or <answer>INCORRECT</answer>:
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
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="SFT learning rate")
    parser.add_argument("--rl_lr", type=float, default=5e-7, help="RL learning rate (SeRL uses 5e-7)")

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
            learning_rate=args.rl_lr,
            kl_coef=1e-4,  # SeRL default
        )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
