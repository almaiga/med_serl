#!/usr/bin/env python3
"""
MedSeRL Local Training Script for Apple Silicon

Runs actual SFT and RL training with MedGemma-4B-IT on Mac with MPS backend.
Designed for M3 Max with 48GB unified memory.

Usage:
    python scripts/train_local.py --num_samples 16 --sft_epochs 1 --rl_episodes 2
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Set environment for Apple Silicon
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"


def check_dependencies():
    """Check if required packages are installed."""
    missing = []
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        if torch.backends.mps.is_available():
            print("MPS (Metal) backend: Available ✓")
        else:
            print("MPS backend: Not available (will use CPU)")
    except ImportError:
        missing.append("torch")
    
    try:
        import transformers
        print(f"Transformers: {transformers.__version__}")
    except ImportError:
        missing.append("transformers")
    
    try:
        import peft
        print(f"PEFT: {peft.__version__}")
    except ImportError:
        missing.append("peft")
    
    if missing:
        print(f"\nMissing packages: {missing}")
        print("Install with: pip install torch transformers peft accelerate")
        return False
    
    return True


def get_device():
    """Get the best available device."""
    import torch
    
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def load_model_and_tokenizer(model_name: str, device: str, use_4bit: bool = False):
    """
    Load MedGemma model and tokenizer.
    
    Args:
        model_name: HuggingFace model name or local path
        device: Device to load model on
        use_4bit: Use 4-bit quantization (not supported on MPS)
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    
    print(f"\nLoading model: {model_name}")
    print(f"Device: {device}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Ensure pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate settings for device
    if device == "mps":
        # MPS doesn't support 4-bit, use float16
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            trust_remote_code=True
        )
    
    print(f"Model loaded: {model.config.name_or_path}")
    print(f"Parameters: {model.num_parameters():,}")
    
    return model, tokenizer


def apply_lora(model, r: int = 8, alpha: int = 16):
    """Apply LoRA adapters for efficient fine-tuning."""
    from peft import LoraConfig, get_peft_model, TaskType
    
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        task_type=TaskType.CAUSAL_LM
    )
    
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    return model


def run_sft_training(
    model,
    tokenizer,
    train_data: list,
    output_dir: str,
    num_epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 2e-5,
    max_length: int = 512
):
    """
    Run SFT warm-up training.
    
    Args:
        model: The model to train
        tokenizer: Tokenizer
        train_data: List of SFT examples
        output_dir: Where to save checkpoints
        num_epochs: Number of training epochs
        batch_size: Batch size (keep small for memory)
        learning_rate: Learning rate
        max_length: Max sequence length
    """
    from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
    from datasets import Dataset
    
    print(f"\n{'='*60}")
    print("Starting SFT Training")
    print(f"{'='*60}")
    print(f"Samples: {len(train_data)}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch size: {batch_size}")
    
    # Prepare dataset
    def format_example(example):
        return {"text": f"{example['input']}\n{example['output']}"}
    
    formatted_data = [format_example(ex) for ex in train_data]
    dataset = Dataset.from_list(formatted_data)
    
    # Tokenize
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding="max_length"
        )
    
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=4,
        learning_rate=learning_rate,
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy="epoch",
        fp16=False,  # MPS doesn't support fp16 training well
        bf16=False,
        report_to="none",
        dataloader_pin_memory=False,  # Required for MPS
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=data_collator,
    )
    
    # Train
    print("\nTraining...")
    trainer.train()
    
    # Save
    checkpoint_path = os.path.join(output_dir, "sft_checkpoint")
    trainer.save_model(checkpoint_path)
    tokenizer.save_pretrained(checkpoint_path)
    
    print(f"\nSFT complete. Saved to: {checkpoint_path}")
    return checkpoint_path


def run_rl_episode(
    model,
    tokenizer,
    batch: list,
    device: str
) -> dict:
    """
    Run a single RL episode (inference + reward calculation).
    
    For simplicity, this does inference only (no gradient updates).
    Full RL would require PPO/REINFORCE implementation.
    """
    import torch
    from src.training.reward_engine import calculate_reward
    
    rewards = []
    outputs = []
    
    model.eval()
    
    for sample in batch:
        # Get the note text
        note = sample.get("original_text", "")
        ground_truth = sample.get("meta", {})
        
        # Format prompt
        prompt = f"""You are a medical error detection assistant. Analyze this clinical note for errors.

Clinical Note:
{note}

Provide your analysis in <thinking> tags, then your verdict in <verdict> tags."""
        
        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id
            )
        
        # Decode
        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Extract just the generated part
        generated = output_text[len(prompt):].strip()
        outputs.append(generated)
        
        # Calculate reward
        reward = calculate_reward(generated, ground_truth)
        rewards.append(reward)
    
    return {
        "outputs": outputs,
        "rewards": rewards,
        "mean_reward": sum(rewards) / len(rewards) if rewards else 0
    }


def main():
    parser = argparse.ArgumentParser(description="MedSeRL Local Training")
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-2b-it",  # Smaller model for testing, change to medgemma-4b-it
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=16,
        help="Number of training samples"
    )
    parser.add_argument(
        "--sft_epochs",
        type=int,
        default=1,
        help="Number of SFT epochs"
    )
    parser.add_argument(
        "--rl_episodes",
        type=int,
        default=2,
        help="Number of RL episodes"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for RL (must be divisible by 4)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/local_training",
        help="Output directory"
    )
    parser.add_argument(
        "--skip_sft",
        action="store_true",
        help="Skip SFT phase"
    )
    parser.add_argument(
        "--use_lora",
        action="store_true",
        default=True,
        help="Use LoRA for efficient training"
    )
    
    args = parser.parse_args()
    
    print("="*60)
    print("MedSeRL Local Training")
    print("="*60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load data
    print("\nLoading MEDEC data...")
    from src.data_processor import MedicalDataProcessor
    
    processor = MedicalDataProcessor.load_training_data()
    print(f"Error pool: {len(processor.error_pool)} samples")
    print(f"Clean pool: {len(processor.clean_pool)} samples")
    
    # Load model
    model, tokenizer = load_model_and_tokenizer(args.model_name, device)
    
    # Apply LoRA if requested
    if args.use_lora:
        print("\nApplying LoRA adapters...")
        model = apply_lora(model)
    
    # SFT Phase
    if not args.skip_sft:
        print("\n" + "="*60)
        print("Phase 1: SFT Warm-Up")
        print("="*60)
        
        # Prepare SFT data
        from src.training.train_serl import prepare_sft_data, convert_sft_examples_to_hf_format
        
        sft_examples = prepare_sft_data(processor)
        
        # Limit to num_samples
        sft_examples = sft_examples[:args.num_samples]
        print(f"Using {len(sft_examples)} SFT samples")
        
        # Convert to HF format
        sft_data = convert_sft_examples_to_hf_format(sft_examples)
        
        # Run SFT
        sft_checkpoint = run_sft_training(
            model=model,
            tokenizer=tokenizer,
            train_data=sft_data,
            output_dir=os.path.join(args.output_dir, "sft"),
            num_epochs=args.sft_epochs,
            batch_size=1  # Keep small for memory
        )
    
    # RL Phase
    print("\n" + "="*60)
    print("Phase 2: RL Training (Inference Only)")
    print("="*60)
    print("Note: This runs inference and calculates rewards.")
    print("Full RL gradient updates require OpenRLHF setup.")
    
    for episode in range(args.rl_episodes):
        print(f"\nEpisode {episode + 1}/{args.rl_episodes}")
        
        # Generate batch
        batch = processor.get_quadrant_batch(batch_size=args.batch_size)
        print(f"  Generated batch: {len(batch)} samples")
        
        # Run episode
        result = run_rl_episode(model, tokenizer, batch, device)
        
        print(f"  Mean reward: {result['mean_reward']:.3f}")
        print(f"  Rewards: {result['rewards']}")
        
        # Show sample output
        if result['outputs']:
            print(f"\n  Sample output:")
            print(f"  {result['outputs'][0][:300]}...")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
