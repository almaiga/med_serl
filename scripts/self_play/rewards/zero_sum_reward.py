"""Zero-sum reward function for medical error detection + localization self-play.

Implements the 3-tier adversarial reward structure:
- Exact match (CORRECT↔CORRECT, or same sentence number): +1.0
- Partial match (error detected but wrong sentence): +0.3
- Miss (wrong classification or unparseable): -1.0
- Format bonus: +0.1 for parseable output

Compatible with both verl and OpenRLHF reward interfaces.
"""

import json
import re
from typing import Optional, Tuple

import torch

from scripts.self_play.utils import parse_assessor_answer

# Reward values (3-tier)
REWARD_EXACT = 1.0     # Exact match
REWARD_PARTIAL = 0.3   # Detected error, wrong sentence
REWARD_MISS = -1.0     # Missed or invalid
FORMAT_REWARD = 0.1    # Small bonus for correct output format


def parse_final_answer(response: str) -> Tuple[str, Optional[int]]:
    """Extract assessor answer from response.
    
    Returns:
        (label, sentence_id) where label is CORRECT/ERROR/UNKNOWN
    """
    if not response:
        return ("UNKNOWN", None)
    return parse_assessor_answer(response)


def compute_game_rewards(
    mode: str,
    ground_truth: str,
    injector_output: str,
    assessor_output: str,
) -> tuple[float, float]:
    """Compute 3-tier zero-sum rewards for Injector and Assessor.
    
    Args:
        mode: "benign" or "error_injection"
        ground_truth: "CORRECT" or str(sentence_id)
        injector_output: Full Injector response
        assessor_output: Full Assessor response
        
    Returns:
        (injector_reward, assessor_reward)
    """
    # Parse Assessor's answer
    label, pred_sid = parse_final_answer(assessor_output)
    
    # Default rewards
    injector_reward = 0.0
    assessor_reward = 0.0
    
    # Format compliance
    if label != "UNKNOWN":
        assessor_reward += FORMAT_REWARD
    # Injector format bonus (compact N. <text> format)
    if injector_output and re.search(r'^\d+\.\s', injector_output.strip()):
        injector_reward += FORMAT_REWARD
    
    # If Assessor failed to produce valid output, Injector wins by default
    if label == "UNKNOWN":
        return injector_reward + abs(REWARD_MISS), assessor_reward + REWARD_MISS
    
    gt_is_correct = (ground_truth == "CORRECT")
    
    # 3-tier reward
    if gt_is_correct and label == "CORRECT":
        # Exact match: both agree note is correct → Assessor wins
        assessor_reward += REWARD_EXACT
        injector_reward += REWARD_MISS
    elif gt_is_correct and label != "CORRECT":
        # False positive: note is correct but assessor flagged → Injector wins
        assessor_reward += REWARD_MISS
        injector_reward += REWARD_EXACT
    elif not gt_is_correct and label == "CORRECT":
        # Missed error: Assessor fooled → Injector wins
        assessor_reward += REWARD_MISS
        injector_reward += REWARD_EXACT
    elif not gt_is_correct and label == "ERROR":
        # Assessor detected error — check sentence number
        pred_str = str(pred_sid) if pred_sid else ""
        if pred_str == ground_truth:
            # Exact sentence match → Assessor wins decisively
            assessor_reward += REWARD_EXACT
            injector_reward += REWARD_MISS
        elif pred_sid is not None:
            # Partial: detected error, wrong sentence → partial credit both ways
            assessor_reward += REWARD_PARTIAL
            injector_reward += -REWARD_PARTIAL
        else:
            # Detected but no sentence → mostly Assessor partial win
            assessor_reward += REWARD_PARTIAL
            injector_reward += -REWARD_PARTIAL
    
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
    game_result = extra_info.get("game_result", {}) if extra_info else {}
    
    mode = game_result.get("mode", "unknown")
    gt = str(game_result.get("ground_truth", "CORRECT"))
    injector_output = game_result.get("injector_output", "")
    assessor_output = game_result.get("assessor_output", "")
    
    injector_reward, assessor_reward = compute_game_rewards(
        mode=mode,
        ground_truth=gt,
        injector_output=injector_output,
        assessor_output=assessor_output,
    )
    
    # Combined reward (for single-model training)
    total_reward = injector_reward + assessor_reward
    
    label, pred_sid = parse_final_answer(assessor_output)
    
    return {
        "reward": total_reward,
        "injector_reward": injector_reward,
        "assessor_reward": assessor_reward,
        "mode": mode,
        "ground_truth": gt,
        "assessor_label": label,
        "assessor_pred_sid": pred_sid,
        "assessor_exact": (
            (gt == "CORRECT" and label == "CORRECT") or 
            (gt != "CORRECT" and label == "ERROR" and str(pred_sid) == gt)
        ),
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
            rewards.append(0.0)
    
    return torch.tensor(rewards, dtype=torch.float32)


def log_game_outcome(
    game_result: dict,
    injector_reward: float,
    assessor_reward: float,
    output_path: Optional[str] = None,
) -> dict:
    """Create a log entry for game analysis."""
    label, pred_sid = parse_final_answer(game_result.get("assessor_output", ""))
    gt = str(game_result.get("ground_truth", ""))
    
    log_entry = {
        "note_id": game_result.get("note_data", {}).get("note_id", ""),
        "mode": game_result.get("mode", ""),
        "ground_truth": gt,
        "assessor_label": label,
        "assessor_pred_sid": pred_sid,
        "assessor_exact": (
            (gt == "CORRECT" and label == "CORRECT") or
            (gt != "CORRECT" and label == "ERROR" and str(pred_sid) == gt)
        ),
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
