"""Custom reward function for MedSeRL self-play training.

Following verl documentation format:
https://verl.readthedocs.io/en/latest/preparation/reward_function.html

Implements zero-sum reward:
- Assessor correct: +1.0 (Injector gets -1.0)
- Assessor wrong: -1.0 (Injector gets +1.0)
- Format bonus: +0.2 for following output format
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Optional


# Global log file path
LOG_DIR = Path(__file__).parent.parent.parent / "results" / "self_play" / "interactions"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"interactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

# Reward values
REWARD_WIN = 1.0
REWARD_LOSE = -1.0
FORMAT_BONUS = 0.2


def parse_final_answer(response: str) -> Optional[str]:
    """Extract final_answer from model response.
    
    Expected format: final_answer: "CORRECT" or "INCORRECT"
    """
    pattern = r'final_answer:\s*["\']?(CORRECT|INCORRECT)["\']?'
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback: check for words in output
    upper = response.upper()
    if "INCORRECT" in upper:
        return "INCORRECT"
    elif "CORRECT" in upper:
        return "CORRECT"
    
    return None


def check_format_compliance(response: str) -> bool:
    """Check if response follows required format.
    
    Required: final_answer: "..." and Explanation: ...
    """
    has_final_answer = bool(re.search(
        r'final_answer:\s*["\']?(CORRECT|INCORRECT)["\']?',
        response,
        re.IGNORECASE
    ))
    has_explanation = bool(re.search(
        r'Explanation:\s*\S+',
        response,
        re.IGNORECASE
    ))
    return has_final_answer and has_explanation


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Compute reward score for medical error detection self-play.
    
    This is called by verl's RewardManager after rollout.
    
    Args:
        data_source (str): Dataset identifier (e.g., 'medec_selfplay')
        solution_str (str): The model's generated response
        ground_truth (str): "CORRECT" or "INCORRECT" from data
        extra_info (dict): Additional information from dataset
        
    Returns:
        float: Reward score
    """
    # Parse model's classification
    model_answer = parse_final_answer(solution_str)
    
    # Check format compliance
    has_valid_format = check_format_compliance(solution_str)
    format_bonus = FORMAT_BONUS if has_valid_format else 0.0
    
    # Compute reward based on classification accuracy
    if model_answer == ground_truth:
        # Model correctly classified the note
        reward = REWARD_WIN + format_bonus
    elif model_answer in ["CORRECT", "INCORRECT"]:
        # Model classified but was wrong
        reward = REWARD_LOSE + format_bonus
    else:
        # Model failed to produce valid classification
        reward = REWARD_LOSE  # No format bonus for invalid output
    
    # Convert extra_info to JSON-serializable types (handles numpy int64, etc.)
    def make_serializable(obj):
        if hasattr(obj, 'item'):  # numpy types
            return obj.item()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [make_serializable(item) for item in obj]
        return obj
    
    # Log the interaction to file
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "data_source": data_source,
        "ground_truth": ground_truth,
        "model_answer": model_answer,
        "model_response": solution_str[:500],  # Truncate for logging
        "reward": float(reward),
        "has_valid_format": has_valid_format,
        "mode": extra_info.get("mode", "unknown") if extra_info else "unknown",
        "note_id": extra_info.get("note_id", "") if extra_info else "",
    }
    
    # Append to log file
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    except Exception as e:
        pass  # Don't fail training if logging fails
    
    return reward
