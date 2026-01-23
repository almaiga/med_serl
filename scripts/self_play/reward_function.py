"""Custom reward function for MedSeRL self-play training.

Following verl documentation format:
https://verl.readthedocs.io/en/latest/preparation/reward_function.html

Medical Error Detection Self-Play Game (adapted from SeRL paper):
- Single model plays both roles (Injector and Assessor)
- Injector: Modifies clinical note (benign edit OR error injection)
- Assessor: Classifies modified note as CORRECT or INCORRECT

Zero-sum rewards:
- Assessor correct: +1.0
- Assessor wrong: -1.0
- Format bonus: +0.2 for following output format
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


# Global log file path - creates new file per training run
LOG_DIR = Path(__file__).parent.parent.parent / "results" / "self_play" / "interactions"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"interactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

# Reward values (following SeRL paper's zero-sum formulation)
REWARD_WIN = 1.0
REWARD_LOSE = -1.0
FORMAT_BONUS = 0.2


def parse_final_answer(response: str) -> Optional[str]:
    """Extract final_answer from model response.
    
    Expected format: final_answer: "CORRECT" or "INCORRECT"
    """
    if not response:
        return None
        
    pattern = r'final_answer:\s*["\']?(CORRECT|INCORRECT)["\']?'
    match = re.search(pattern, response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback: check for words anywhere in output
    upper = response.upper()
    if "INCORRECT" in upper:
        return "INCORRECT"
    elif "CORRECT" in upper:
        return "CORRECT"
    
    return None


def extract_generated_note(response: str) -> Optional[str]:
    """Extract the generated_note section from Injector's response.
    
    Expected format:
    generated_note:
    [the modified note]
    
    final_answer: ...
    """
    if not response:
        return None
    
    # Remove <think> tags first
    clean = re.sub(r'<think>.*?</think>', '', response, flags=re.DOTALL)
    
    # Try to find generated_note section
    match = re.search(
        r'generated_note:\s*\n(.*?)(?=\n\s*final_answer:|$)', 
        clean, 
        re.DOTALL | re.IGNORECASE
    )
    if match:
        return match.group(1).strip()
    
    return None


def check_format_compliance(response: str) -> bool:
    """Check if response follows required format.
    
    For Injector: generated_note: + final_answer:
    For Assessor: final_answer: + Explanation:
    """
    if not response:
        return False
        
    has_final_answer = bool(re.search(
        r'final_answer:\s*["\']?(CORRECT|INCORRECT)["\']?',
        response,
        re.IGNORECASE
    ))
    
    # Check for either generated_note (Injector) or Explanation (Assessor)
    has_structure = bool(re.search(
        r'(generated_note:|Explanation:)\s*\S+',
        response,
        re.IGNORECASE
    ))
    
    return has_final_answer and has_structure


def make_serializable(obj: Any) -> Any:
    """Convert numpy types and other non-JSON types to Python natives."""
    if obj is None:
        return None
    if hasattr(obj, 'item'):  # numpy types
        return obj.item()
    elif isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif isinstance(obj, bytes):
        return obj.decode('utf-8', errors='replace')
    return obj


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Compute reward score for medical error detection self-play.
    
    This is called by verl's RewardManager after rollout.
    
    Args:
        data_source (str): Dataset identifier (e.g., 'medec_selfplay')
        solution_str (str): The model's generated response
        ground_truth (str): "CORRECT" or "INCORRECT" from data
        extra_info (dict): Additional information from dataset including:
            - correct_note: The original correct clinical note
            - incorrect_note: The error version (for error mode)
            - mode: "benign" or "error_injection"
            - note_id: Unique identifier
        
    Returns:
        float: Reward score
    """
    # Ensure we have valid inputs
    solution_str = solution_str or ""
    ground_truth = ground_truth or ""
    extra_info = make_serializable(extra_info) if extra_info else {}
    
    # Parse model's classification
    model_answer = parse_final_answer(solution_str)
    
    # Extract generated note (for logging/analysis)
    generated_note = extract_generated_note(solution_str)
    
    # Check format compliance
    has_valid_format = check_format_compliance(solution_str)
    format_bonus = FORMAT_BONUS if has_valid_format else 0.0
    
    # Compute reward based on classification accuracy
    if model_answer == ground_truth:
        # Model correctly classified the note
        reward = REWARD_WIN + format_bonus
        outcome = "correct"
    elif model_answer in ["CORRECT", "INCORRECT"]:
        # Model classified but was wrong
        reward = REWARD_LOSE + format_bonus
        outcome = "wrong"
    else:
        # Model failed to produce valid classification
        reward = REWARD_LOSE  # No format bonus for invalid output
        outcome = "invalid_format"
    
    # Build comprehensive log entry for failure analysis
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "data_source": str(data_source),
        
        # Game outcome
        "ground_truth": str(ground_truth),
        "model_answer": model_answer,
        "outcome": outcome,
        "reward": float(reward),
        "has_valid_format": has_valid_format,
        
        # Mode info
        "mode": extra_info.get("mode", "unknown"),
        "note_id": extra_info.get("note_id", ""),
        
        # CRITICAL FOR ANALYSIS: The actual clinical notes
        "original_correct_note": extra_info.get("correct_note", "")[:1000],  # Truncate
        "original_incorrect_note": extra_info.get("incorrect_note", "")[:1000],
        "error_type": extra_info.get("error_type", ""),
        "error_sentence": extra_info.get("error_sentence", ""),
        "corrected_sentence": extra_info.get("corrected_sentence", ""),
        
        # Model outputs
        "generated_note": (generated_note or "")[:1000],  # What Injector produced
        "model_response_full": solution_str[:2000],  # Full response (truncated)
    }
    
    # Append to log file
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        # Log error but don't fail training
        try:
            error_log = LOG_DIR / "reward_errors.log"
            with open(error_log, 'a') as f:
                f.write(f"{datetime.now().isoformat()} Error: {e}\n")
        except:
            pass
    
    return reward
