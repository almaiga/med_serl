#!/usr/bin/env python3
"""
MedSeRL Multi-GPU Training Script with Accelerate

Supports 1 or more GPUs using HuggingFace Accelerate.

Usage:
    # Single GPU
    python scripts/train_gpu_multi.py --num_samples 512

    # Multi-GPU with Accelerate
    accelerate launch --config_file configs/accelerate_config.yaml scripts/train_gpu_multi.py --num_samples 512
"""

import argparse
import os
import sys
import warnings
import json
import time
import copy
from pathlib import Path
from datetime import datetime

os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", message=".*pynvml.*deprecated.*")
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.*deprecated.*")

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.optim import AdamW
from tqdm import tqdm
from accelerate import Accelerator
from accelerate.utils import gather_object


def check_gpu(accelerator):
    """Check GPU availability."""
    print("=" * 60)
    print("GPU Check")
    print("=" * 60)
    print(f"Accelerator device: {accelerator.device}")
    print(f"Num processes: {accelerator.num_processes}")
    print(f"Process index: {accelerator.process_index}")
    print(f"Is main process: {accelerator.is_main_process}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_gb = props.total_memory / 1024**3
            print(f"  GPU {i}: {props.name} ({mem_gb:.1f} GB)")


def load_model_and_tokenizer(model_path: str, use_4bit: bool = False):
    """Load model without device_map for Accelerate compatibility."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"\nLoading model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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
        # Don't use device_map with Accelerate
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )

    print(f"Model loaded: {model.config._name_or_path}")
    print(f"Parameters: {model.num_parameters():,}")
    return model, tokenizer


def apply_lora(model, r: int = 16, alpha: int = 32):
    """Apply LoRA."""
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
    model, tokenizer, processor, output_dir: str,
    num_samples: int = -1, num_epochs: int = 1,
    batch_size: int = 2, gradient_accumulation: int = 8,
    learning_rate: float = 2e-5, max_length: int = 1024,
):
    """SFT phase - HuggingFace Trainer handles multi-GPU automatically."""
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from datasets import Dataset
    from src.training.train_serl import prepare_sft_data, convert_sft_examples_to_hf_format
    import random

    print("\n" + "=" * 60)
    print("Phase 1: SFT Warm-Up (Multi-GPU)")
    print("=" * 60)

    sft_examples = prepare_sft_data(processor)
    print(f"Total contrastive pairs: {len(sft_examples)}")

    if num_samples > 0 and num_samples < len(sft_examples):
        incorrect = [ex for ex in sft_examples if ex.error_type != "Correct"]
        correct = [ex for ex in sft_examples if ex.error_type == "Correct"]
        half = num_samples // 2
        sft_examples = random.sample(incorrect, min(half, len(incorrect))) + \
                       random.sample(correct, min(half, len(correct)))
        random.shuffle(sft_examples)
        print(f"Using {len(sft_examples)} samples")

    hf_data = convert_sft_examples_to_hf_format(sft_examples)
    dataset = Dataset.from_list(hf_data)

    def tokenize(examples):
        return tokenizer(examples["text"], truncation=True, max_length=max_length, padding="max_length")

    tokenized = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)

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
        ddp_find_unused_parameters=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    )

    print("\nStarting SFT training...")
    trainer.train()

    checkpoint_path = os.path.join(output_dir, "sft_checkpoint")
    trainer.save_model(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    print(f"SFT checkpoint saved: {checkpoint_path}")
    return checkpoint_path


def compute_approx_kl(log_probs, log_probs_base):
    """K1 KL estimator."""
    return log_probs.float() - log_probs_base.float()


def log_probs_from_logits(logits, labels):
    """Compute log probs from logits."""
    import torch.nn.functional as F
    log_probs = F.log_softmax(logits, dim=-1)
    return log_probs.gather(dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def masked_mean(tensor, mask, dim=None):
    """Masked mean."""
    if mask is None:
        return tensor.mean(dim=dim)
    return (tensor * mask).sum(dim=dim) / mask.sum(dim=dim).clamp(min=1e-8)


def format_prompt_for_chat(note: str, tokenizer) -> str:
    """Format prompt."""
    system_content = """You are a healthcare professional specializing in analyzing medical notes.
Important: Medical notes should be presumed CORRECT unless there is an obvious, significant error.
Classification: INCORRECT (clinically significant error) or CORRECT (default)."""

    user_content = f"""Analyze this clinical note:

{note}

Provide brief reasoning in <think> tags, then your classification in <answer> tags."""

    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_content}
    ]
    
    if hasattr(tokenizer, 'apply_chat_template'):
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return f"System: {system_content}\nUser: {user_content}\nAssistant:"


def run_rl_phase(
    accelerator, model, ref_model, tokenizer, processor, output_dir: str,
    num_episodes: int = 100, batch_size: int = 16,
    learning_rate: float = 5e-7, kl_coef: float = 1e-4,
    max_length: int = 1024, use_self_instruction: bool = False,
    self_instruction_config: dict = None,
):
    """RL phase with Accelerate for multi-GPU."""
    from src.training.reward_engine import calculate_reward
    from src.training.interaction_logger import InteractionLogger

    print("\n" + "=" * 60)
    print("Phase 2: RL Training (REINFORCE++) - Multi-GPU")
    print("=" * 60)
    print(f"Episodes: {num_episodes}, Batch: {batch_size}, LR: {learning_rate}")
    print(f"Num GPUs: {accelerator.num_processes}")

    # Self-instruction setup
    si_state, si_config = None, None
    if use_self_instruction:
        from src.training.self_instruction import (
            initialize_self_instruction, generate_and_filter_batch,
            get_self_instruction_stats, DEFAULT_CONFIG
        )
        si_config = self_instruction_config or DEFAULT_CONFIG.copy()
        si_state, si_config = initialize_self_instruction(
            processor.error_pool[:200], processor.clean_pool[:200], si_config
        )

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    model, optimizer = accelerator.prepare(model, optimizer)

    # Interaction logger (main process only)
    interaction_logger = None
    if accelerator.is_main_process:
        interaction_logger = InteractionLogger(
            output_dir=os.path.join(output_dir, "interactions"),
            session_name=datetime.now().strftime("%Y%m%d_%H%M%S"),
        )

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    metrics_file = os.path.join(output_dir, "rl_metrics.jsonl")

    model.train()
    total_rewards = []
    start_time = time.time()

    # Samples per GPU
    samples_per_gpu = batch_size // accelerator.num_processes
    if samples_per_gpu < 1:
        samples_per_gpu = 1

    pbar = tqdm(range(num_episodes), desc="RL Training", disable=not accelerator.is_main_process)

    for episode in pbar:
        episode_start = time.time()

        # Get batch (each GPU gets different samples)
        if use_self_instruction and si_state is not None:
            batch = generate_and_filter_batch(
                model=accelerator.unwrap_model(model),
                tokenizer=tokenizer,
                seed_error_samples=processor.error_pool[:200],
                seed_correct_samples=processor.clean_pool[:200],
                state=si_state, config=si_config,
                target_batch_size=samples_per_gpu,
                current_step=episode,
                device=accelerator.device,
                max_attempts=samples_per_gpu * 3,
                verbose=False,
            )
        else:
            batch = processor.get_quadrant_batch(batch_size=samples_per_gpu)

        episode_rewards = []
        batch_data = []

        # Rollout phase
        for i, sample in enumerate(batch):
            note = sample.get("original_text", sample.get("text", ""))
            ground_truth = sample.get("meta", sample.get("ground_truth", {}))

            prompt = format_prompt_for_chat(note, tokenizer)
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=max_length, padding=True).to(accelerator.device)
            prompt_len = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = accelerator.unwrap_model(model).generate(
                    **inputs, max_new_tokens=512, do_sample=True,
                    temperature=0.7, top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

            sequences = outputs
            generated_text = tokenizer.decode(sequences[0][prompt_len:], skip_special_tokens=True)
            if "</answer>" in generated_text:
                generated_text = generated_text.split("</answer>")[0] + "</answer>"

            reward = calculate_reward(generated_text, ground_truth)
            episode_rewards.append(reward)

            # Log interaction
            if interaction_logger:
                interaction_logger.log_interaction(
                    episode=episode, batch_idx=i, note=note[:500],
                    quadrant=sample.get("quadrant", "unknown"),
                    ground_truth=ground_truth, model_output=generated_text, reward=reward,
                )

            # Compute ref log probs
            seq_len = sequences.shape[1]
            action_mask = torch.zeros(1, seq_len - 1, device=accelerator.device)
            if seq_len > prompt_len:
                action_mask[0, prompt_len - 1:seq_len - 1] = 1.0

            input_seq = sequences[:, :-1]
            target_seq = sequences[:, 1:]

            with torch.no_grad():
                ref_outputs = ref_model(input_ids=input_seq.to(ref_model.device))
                ref_log_probs = log_probs_from_logits(ref_outputs.logits.float(), target_seq.to(ref_model.device))

            batch_data.append({
                'sequences': sequences, 'action_mask': action_mask,
                'ref_log_probs': ref_log_probs.to(accelerator.device), 'reward': reward,
            })
            torch.cuda.empty_cache()

        # Gather rewards across GPUs
        all_rewards = gather_object(episode_rewards)
        if accelerator.is_main_process:
            flat_rewards = [r for sublist in all_rewards for r in (sublist if isinstance(sublist, list) else [sublist])]
        else:
            flat_rewards = episode_rewards

        # Compute advantages (local)
        rewards_tensor = torch.tensor([d['reward'] for d in batch_data], device=accelerator.device)
        advantages = rewards_tensor - rewards_tensor.mean()
        if len(advantages) > 1:
            advantages = advantages / advantages.std().clamp(min=1e-8)

        # Policy update
        optimizer.zero_grad()
        total_kl = 0.0

        for idx, (data, adv) in enumerate(zip(batch_data, advantages)):
            sequences = data['sequences']
            action_mask = data['action_mask']
            ref_log_probs = data['ref_log_probs']

            input_seq = sequences[:, :-1]
            target_seq = sequences[:, 1:]

            policy_outputs = model(input_ids=input_seq)
            policy_log_probs = log_probs_from_logits(policy_outputs.logits.float(), target_seq)

            kl = compute_approx_kl(policy_log_probs, ref_log_probs)
            masked_log_probs = policy_log_probs * action_mask
            response_log_prob = masked_log_probs.sum()
            kl_mean = masked_mean(kl, action_mask, dim=-1).mean()

            loss = (-adv * response_log_prob + kl_coef * kl_mean) / len(batch_data)
            accelerator.backward(loss)
            total_kl += kl_mean.item()

        accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        torch.cuda.empty_cache()

        # Metrics (main process)
        if accelerator.is_main_process:
            mean_reward = sum(flat_rewards) / len(flat_rewards) if flat_rewards else 0
            total_rewards.append(mean_reward)
            
            metrics = {
                "episode": episode, "mean_reward": mean_reward,
                "avg_kl": total_kl / len(batch_data),
                "running_avg": sum(total_rewards) / len(total_rewards),
                "time": time.time() - episode_start,
            }
            with open(metrics_file, "a") as f:
                f.write(json.dumps(metrics) + "\n")

            pbar.set_postfix({'r': f'{mean_reward:.3f}', 'avg': f'{metrics["running_avg"]:.3f}'})

            if (episode + 1) % 50 == 0:
                ckpt = os.path.join(output_dir, f"rl_checkpoint_ep{episode + 1}")
                accelerator.unwrap_model(model).save_pretrained(ckpt)
                tokenizer.save_pretrained(ckpt)

        accelerator.wait_for_everyone()

    # Final save
    if accelerator.is_main_process:
        final_path = os.path.join(output_dir, "rl_final")
        accelerator.unwrap_model(model).save_pretrained(final_path)
        tokenizer.save_pretrained(final_path)
        if interaction_logger:
            interaction_logger.close()
        print(f"\nTraining complete. Model: {final_path}")
        print(f"Avg reward: {sum(total_rewards)/len(total_rewards):.3f}")

    return os.path.join(output_dir, "rl_final")


def main():
    parser = argparse.ArgumentParser(description="MedSeRL Multi-GPU Training")
    parser.add_argument("--model_path", type=str, default="google/medgemma-4b-it")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=512)
    parser.add_argument("--medec_path", type=str, default="data_raw/MEDEC")
    parser.add_argument("--sft_epochs", type=int, default=3)
    parser.add_argument("--rl_episodes", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--rl_lr", type=float, default=5e-7)
    parser.add_argument("--use_4bit", action="store_true")
    parser.add_argument("--use_lora", action="store_true", default=True)
    parser.add_argument("--skip_sft", action="store_true")
    parser.add_argument("--skip_rl", action="store_true")
    parser.add_argument("--use_self_instruction", action="store_true")
    parser.add_argument("--output_dir", type=str, default="outputs/gpu_training_multi")
    args = parser.parse_args()

    # Initialize Accelerator
    accelerator = Accelerator(mixed_precision="bf16")
    
    check_gpu(accelerator)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data (main process)
    if accelerator.is_main_process:
        print("\nLoading MEDEC data...")
    from src.data_processor import MedicalDataProcessor
    processor = MedicalDataProcessor.load_training_data(data_path=args.medec_path)
    
    if accelerator.is_main_process:
        print(f"Error pool: {len(processor.error_pool)}, Clean pool: {len(processor.clean_pool)}")

    # Load model
    model_path = args.resume_from or args.model_path
    model, tokenizer = load_model_and_tokenizer(model_path, use_4bit=args.use_4bit)

    if args.use_lora and not args.resume_from:
        if accelerator.is_main_process:
            print("\nApplying LoRA...")
        model = apply_lora(model)

    # Create reference model before accelerator.prepare
    ref_model = copy.deepcopy(model)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    ref_model = ref_model.to(accelerator.device)

    # SFT Phase
    if not args.skip_sft and not args.resume_from:
        run_sft_phase(
            model=model, tokenizer=tokenizer, processor=processor,
            output_dir=os.path.join(args.output_dir, "sft"),
            num_samples=args.num_samples, num_epochs=args.sft_epochs,
            batch_size=args.batch_size, learning_rate=args.learning_rate,
        )

    # RL Phase
    if not args.skip_rl:
        si_config = None
        if args.use_self_instruction:
            si_config = {"rouge_threshold": 0.7, "difficulty_lower": 0.2, "difficulty_upper": 0.8,
                        "num_few_shot": 4, "num_from_generated": 2, "expiration_steps": 2, "n_samples_for_difficulty": 4}
        
        run_rl_phase(
            accelerator=accelerator, model=model, ref_model=ref_model,
            tokenizer=tokenizer, processor=processor,
            output_dir=os.path.join(args.output_dir, "rl"),
            num_episodes=args.rl_episodes, batch_size=args.batch_size,
            learning_rate=args.rl_lr, use_self_instruction=args.use_self_instruction,
            self_instruction_config=si_config,
        )

    if accelerator.is_main_process:
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()
