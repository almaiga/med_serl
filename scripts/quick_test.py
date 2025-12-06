#!/usr/bin/env python3
"""
MedSeRL Quick Test Script

Runs a quick test of the MedSeRL pipeline with a small number of samples.
This tests the data loading, batch generation, and mock agent pipeline
without requiring GPU or actual model inference.

Usage:
    python scripts/quick_test.py --batch_size 16 --num_episodes 2
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_quick_test(batch_size: int = 16, num_episodes: int = 2, verbose: bool = True):
    """
    Run a quick test of the MedSeRL pipeline.
    
    Args:
        batch_size: Number of samples per batch (must be divisible by 4)
        num_episodes: Number of training episodes to simulate
        verbose: Print detailed output
    """
    print("=" * 60)
    print("MedSeRL Quick Test")
    print("=" * 60)
    print(f"Batch size: {batch_size}")
    print(f"Episodes: {num_episodes}")
    print()
    
    # Step 1: Test Data Loading
    print("Step 1: Loading MEDEC data...")
    try:
        from src.data_processor import MedicalDataProcessor
        
        processor = MedicalDataProcessor.load_training_data(
            data_path="data_raw/MEDEC"
        )
        print(f"  ✓ Loaded {len(processor.error_pool)} error samples")
        print(f"  ✓ Loaded {len(processor.clean_pool)} clean samples")
    except Exception as e:
        print(f"  ✗ Failed to load data: {e}")
        return False
    
    # Step 2: Test Batch Generation
    print("\nStep 2: Testing batch generation...")
    try:
        batch = processor.get_quadrant_batch(batch_size=batch_size)
        print(f"  ✓ Generated batch with {len(batch)} samples")
        
        # Verify quadrant distribution
        modes = {}
        for sample in batch:
            mode = sample.get('mode', 'unknown')
            modes[mode] = modes.get(mode, 0) + 1
        
        print(f"  ✓ Quadrant distribution: {modes}")
        
        # Check that each quadrant has batch_size/4 samples
        expected_per_quadrant = batch_size // 4
        for mode, count in modes.items():
            if count != expected_per_quadrant:
                print(f"  ⚠ Warning: {mode} has {count} samples, expected {expected_per_quadrant}")
    except Exception as e:
        print(f"  ✗ Failed to generate batch: {e}")
        return False
    
    # Step 3: Test Mock Agents
    print("\nStep 3: Testing mock agents...")
    try:
        from src.agents.scribe_agent import create_scribe_agent
        from src.agents.doctor_agent import create_doctor_agent
        
        # Create mock agents (no GPU required)
        scribe = create_scribe_agent(model_path="mock", use_mock=True)
        doctor = create_doctor_agent(model_path="mock", use_mock=True)
        
        print(f"  ✓ Created mock Scribe agent")
        print(f"  ✓ Created mock Doctor agent")
        
        # Test Scribe transformation
        transformed = scribe.transform_batch(batch[:4])
        print(f"  ✓ Scribe transformed {len(transformed)} samples")
        
        # Test Doctor analysis
        notes = [t.get('transformed_text', t.get('original_text', '')) for t in transformed]
        outputs = doctor.analyze_batch(notes)
        print(f"  ✓ Doctor analyzed {len(outputs)} notes")
        
        if verbose and outputs:
            print(f"\n  Sample Doctor output:")
            print(f"  {outputs[0][:200]}...")
    except Exception as e:
        print(f"  ✗ Failed to test agents: {e}")
        return False
    
    # Step 4: Test Reward Calculation
    print("\nStep 4: Testing reward calculation...")
    try:
        from src.training.reward_engine import calculate_reward, calculate_reward_with_metadata
        
        # Test with a sample output and ground truth
        sample_output = "<thinking>\nChecking for errors...\n</thinking>\n<verdict>Error: Diagnosis</verdict>"
        ground_truth = {"has_error": True, "error_type": "Diagnosis"}
        
        reward = calculate_reward(sample_output, ground_truth)
        print(f"  ✓ Calculated reward: {reward}")
        
        # Test with metadata
        reward_meta = calculate_reward_with_metadata(sample_output, ground_truth)
        print(f"  ✓ Reward breakdown: structural={reward_meta.structural_reward}, "
              f"outcome={reward_meta.outcome_reward}")
    except Exception as e:
        print(f"  ✗ Failed to test reward: {e}")
        return False
    
    # Step 5: Simulate Training Loop
    print(f"\nStep 5: Simulating {num_episodes} training episodes...")
    try:
        from src.training.reward_engine import calculate_reward
        
        total_rewards = []
        
        for episode in range(num_episodes):
            # Generate batch
            batch = processor.get_quadrant_batch(batch_size=batch_size)
            
            # Transform with Scribe
            transformed = scribe.transform_batch(batch)
            
            # Analyze with Doctor
            notes = [t.get('transformed_text', t.get('original_text', '')) for t in transformed]
            outputs = doctor.analyze_batch(notes)
            
            # Calculate rewards
            rewards = []
            for output, item in zip(outputs, transformed):
                ground_truth = item.get('ground_truth', item.get('meta', {}))
                reward = calculate_reward(output, ground_truth)
                rewards.append(reward)
            
            mean_reward = sum(rewards) / len(rewards) if rewards else 0
            total_rewards.append(mean_reward)
            
            print(f"  Episode {episode + 1}: mean_reward = {mean_reward:.3f}")
        
        avg_reward = sum(total_rewards) / len(total_rewards) if total_rewards else 0
        print(f"  ✓ Average reward across episodes: {avg_reward:.3f}")
    except Exception as e:
        print(f"  ✗ Failed to simulate training: {e}")
        return False
    
    # Step 6: Test SFT Data Preparation
    print("\nStep 6: Testing SFT data preparation...")
    try:
        from src.training.train_serl import prepare_sft_data
        
        sft_examples = prepare_sft_data(processor)
        print(f"  ✓ Prepared {len(sft_examples)} SFT examples")
        
        if verbose and sft_examples:
            example = sft_examples[0]
            print(f"\n  Sample SFT target:")
            print(f"  {example.target_text[:200]}...")
    except Exception as e:
        print(f"  ✗ Failed to prepare SFT data: {e}")
        return False
    
    # Step 7: Test Checkpoint Utilities
    print("\nStep 7: Testing checkpoint utilities...")
    try:
        from src.training.checkpoint import get_checkpoint_filename
        
        filename = get_checkpoint_filename(episode=42)
        print(f"  ✓ Checkpoint filename: {filename}")
        
        # Verify episode number is in filename
        assert "42" in filename, "Episode number not in filename"
        print(f"  ✓ Episode number correctly included in filename")
    except Exception as e:
        print(f"  ✗ Failed to test checkpoint: {e}")
        return False
    
    print("\n" + "=" * 60)
    print("✓ All quick tests passed!")
    print("=" * 60)
    print()
    print("Next steps:")
    print("  1. Run full unit tests: python3 -m pytest tests/ -v")
    print("  2. For GPU training, see scripts/train_medserl_openrlhf.sh")
    print()
    
    return True


def main():
    parser = argparse.ArgumentParser(description="MedSeRL Quick Test")
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=16,
        help="Batch size (must be divisible by 4)"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=2,
        help="Number of episodes to simulate"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    if args.batch_size % 4 != 0:
        print(f"Error: batch_size must be divisible by 4, got {args.batch_size}")
        sys.exit(1)
    
    success = run_quick_test(
        batch_size=args.batch_size,
        num_episodes=args.num_episodes,
        verbose=not args.quiet
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
