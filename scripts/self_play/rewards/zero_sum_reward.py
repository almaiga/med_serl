"""Zero-sum reward function for medical error detection self-play.

Implements the adversarial reward structure:
- Injector wins (+1) if Assessor is fooled
- Assessor wins (+1) if classification is correct
- Zero-sum: one player's gain is the other's loss

Compatible with both verl and OpenRLHF reward interfaces.
"""

import json
import re
from typing import Optional

import torch

# Reward values
REWARD_WIN = 1.0
REWARD_LOSE = -1.0
FORMAT_REWARD = 0.1  # Small bonus for correct output format


def parse_final_answer(response: str) -> Optional[str]:
    """Extract final_answer from response."""
    pattern = r'final_answer:\s*["\']?(CORRECT|INCORRECT)["\']?'
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return None


def compute_game_rewards(
    mode: str,
    ground_truth: str,
    injector_output: str,
    assessor_output: str,
) -> tuple[float, float]:
    """Compute zero-sum rewards for Injector and Assessor.
    
    Args:
        mode: "benign" or "error_injection"
        ground_truth: "CORRECT" (benign) or "INCORRECT" (error)
        injector_output: Full Injector response
        assessor_output: Full Assessor response
        
    Returns:
        (injector_reward, assessor_reward)
    """
    # Parse Assessor's classification
    assessor_answer = parse_final_answer(assessor_output)
    injector_answer = parse_final_answer(injector_output)
    
    # Default rewards
    injector_reward = 0.0
    assessor_reward = 0.0
    
    # Check for format compliance
    if injector_answer is not None:
        injector_reward += FORMAT_REWARD
    if assessor_answer is not None:
        assessor_reward += FORMAT_REWARD
    
    # If Assessor failed to produce valid output, Injector wins by default
    if assessor_answer is None:
        return injector_reward + REWARD_WIN, assessor_reward + REWARD_LOSE
    
    # Zero-sum game outcome
    assessor_correct = (assessor_answer == ground_truth)
    
    if assessor_correct:
        # Assessor wins
        assessor_reward += REWARD_WIN
        injector_reward += REWARD_LOSE
    else:
        # Injector wins (fooled the Assessor)
        injector_reward += REWARD_WIN
        assessor_reward += REWARD_LOSE
    
    return injector_reward, assessor_reward


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: Optional[dict] = None,
) -> dict:
    """verl-compatible reward function.
    
    Called after rollout completes with the full transcript.
    
    Args:
        data_source: Dataset identifier
        solution_str: Full rollout transcript (both turns)
        ground_truth: Dict with game metadata
        extra_info: Additional data including game_result
        
    Returns:
        Dict with reward breakdown
    """
    # Extract game result from extra_info (populated by tool)
    game_result = extra_info.get("game_result", {}) if extra_info else {}
    
    mode = game_result.get("mode", "unknown")
    gt = game_result.get("ground_truth", "CORRECT")
    injector_output = game_result.get("injector_output", "")
    assessor_output = game_result.get("assessor_output", "")
    
    injector_reward, assessor_reward = compute_game_rewards(
        mode=mode,
        ground_truth=gt,
        injector_output=injector_output,
        assessor_output=assessor_output,
    )
    
    # Combined reward (for single-model training)
    # Both roles are the same model, so we sum rewards
    total_reward = injector_reward + assessor_reward
    
    return {
        "reward": total_reward,
        "injector_reward": injector_reward,
        "assessor_reward": assessor_reward,
        "mode": mode,
        "ground_truth": gt,
        "assessor_answer": parse_final_answer(assessor_output),
        "assessor_correct": parse_final_answer(assessor_output) == gt,
    }


def reward_func(
    queries: list[str],
    prompts: list[str],
    labels: list[str],
) -> torch.Tensor:
    """OpenRLHF-compatible reward function.
    
    Args:
        queries: List of full model outputs (transcripts)
        prompts: List of initial prompts
        labels: List of JSON-encoded label dicts
        
    Returns:
        Tensor of reward values
    """
    rewards = []
    
    for query, prompt, label_str in zip(queries, prompts, labels):
        try:
            label = json.loads(label_str) if isinstance(label_str, str) else label_str
        except json.JSONDecodeError:
            label = {}
        
        # Parse the multi-turn transcript to extract turns
        # This depends on how verl formats the transcript
        # Simplified: assume game_result is embedded or we parse manually
        
        extra_info = label.get("extra_info", {})
        game_result = extra_info.get("game_result", {})
        
        if game_result:
            result = compute_score(
                data_source="medec_selfplay",
                solution_str=query,
                ground_truth=label.get("reward_model", {}).get("ground_truth", {}),
                extra_info=extra_info,
            )
            rewards.append(result["reward"])
        else:
            # Fallback: try to parse transcript directly
            # This handles cases where game_result wasn't populated
            rewards.append(0.0)
    
    return torch.tensor(rewards, dtype=torch.float32)


# Logging utilities for analysis
def log_game_outcome(
    game_result: dict,
    injector_reward: float,
    assessor_reward: float,
    output_path: Optional[str] = None,
) -> dict:
    """Create a log entry for game analysis."""
    log_entry = {
        "note_id": game_result.get("note_data", {}).get("note_id", ""),
        "mode": game_result.get("mode", ""),
        "ground_truth": game_result.get("ground_truth", ""),
        "assessor_answer": parse_final_answer(game_result.get("assessor_output", "")),
        "assessor_correct": parse_final_answer(game_result.get("assessor_output", "")) == game_result.get("ground_truth", ""),
        "injector_reward": injector_reward,
        "assessor_reward": assessor_reward,
        "injector_output_length": len(game_result.get("injector_output", "")),
        "assessor_output_length": len(game_result.get("assessor_output", "")),
    }
    
    if output_path:
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    return log_entry
