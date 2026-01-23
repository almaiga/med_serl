"""Custom reward function for MedSeRL self-play training.

Following verl documentation format:
https://verl.readthedocs.io/en/latest/preparation/reward_function.html
"""

import json
import os
from pathlib import Path
from datetime import datetime


# Global log file path
LOG_DIR = Path(__file__).parent.parent.parent / "results" / "self_play" / "interactions"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"interactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Compute reward score for medical error detection self-play.
    
    For now, this is a placeholder that returns a simple reward.
    In full self-play implementation, this will evaluate game outcomes.
    
    Args:
        data_source (str): Dataset identifier (e.g., 'medec_selfplay')
        solution_str (str): The model's generated response
        ground_truth (str): Ground truth note_id from data
        extra_info (dict): Additional information from dataset
        
    Returns:
        float: Reward score (0.0 to 1.0)
    """
    # For now, return neutral reward to allow training to proceed
    # TODO: Implement proper game-based reward evaluation
    reward = 0.5
    
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
        "model_response": solution_str,
        "reward": float(reward),
        "extra_info": make_serializable(extra_info) if extra_info else None
    }
    
    # Append to log file
    with open(LOG_FILE, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')
    
    return reward
