"""Custom reward function for MedSeRL self-play training.

Following verl documentation format:
https://verl.readthedocs.io/en/latest/preparation/reward_function.html

Medical Error Detection Self-Play Game (adapted from SeRL paper arXiv:2506.07468):
- Single model plays both roles (Injector and Assessor)
- Injector: Modifies clinical note (benign edit OR error injection)
- Assessor: Classifies modified note as CORRECT or INCORRECT

Zero-sum rewards (following SeRL Algorithm 1):
- Assessor correct → Assessor wins: RA=-1.0, RD=+1.0
- Assessor wrong → Injector wins: RA=+1.0, RD=-1.0
- Format bonus: +0.2 for following output format

Note: In our implementation, each example is assessed independently.
The "Injector" already produced its output (the modified note in training data).
The model acts as "Assessor" and classifies - this is what we reward.
"""

import json
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any
from collections import defaultdict
import threading
from difflib import SequenceMatcher


# Global log file path - creates new file per training run
LOG_DIR = Path(__file__).parent.parent.parent / "results" / "self_play" / "interactions"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"interactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
SUMMARY_FILE = LOG_DIR / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Reward values (following SeRL paper's zero-sum formulation)
REWARD_WIN = 1.0
REWARD_LOSE = -1.0
FORMAT_BONUS = 0.2

# Global statistics tracker (thread-safe)
_stats_lock = threading.Lock()
_stats = {
    "total_interactions": 0,
    "correct_classifications": 0,
    "wrong_classifications": 0,
    "invalid_format": 0,
    # By mode
    "benign_correct": 0,
    "benign_wrong": 0,
    "error_correct": 0,
    "error_wrong": 0,
    # Rewards
    "total_reward": 0.0,
    "benign_reward": 0.0,
    "error_reward": 0.0,
    # For computing averages
    "benign_count": 0,
    "error_count": 0,
    # Token metrics
    "total_response_chars": 0,
    "total_response_tokens_approx": 0,  # Rough estimate: chars / 4
    "min_response_chars": float('inf'),
    "max_response_chars": 0,
    "truncated_responses": 0,  # Responses that appear cut off
    "responses_with_think_tags": 0,
    "responses_missing_closing_think": 0,
    # Note similarity metrics (Injector modifications)
    "total_similarity_benign": 0.0,
    "benign_similarity_count": 0,
    "total_similarity_error": 0.0,
    "error_similarity_count": 0,
}


def get_summary_stats() -> Dict[str, Any]:
    """Get summary statistics for the training run."""
    with _stats_lock:
        total = _stats["total_interactions"]
        if total == 0:
            return {"message": "No interactions yet"}
        
        correct = _stats["correct_classifications"]
        wrong = _stats["wrong_classifications"]
        invalid = _stats["invalid_format"]
        
        benign_total = _stats["benign_count"]
        error_total = _stats["error_count"]
        
        # Token metrics
        total_chars = _stats["total_response_chars"]
        min_chars = _stats["min_response_chars"] if _stats["min_response_chars"] != float('inf') else 0
        max_chars = _stats["max_response_chars"]
        truncated = _stats["truncated_responses"]
        
        return {
            "total_interactions": total,
            "accuracy": correct / total if total > 0 else 0,
            "win_rate_assessor": correct / total if total > 0 else 0,  # Assessor wins when correct
            "win_rate_injector": wrong / total if total > 0 else 0,   # Injector wins when wrong
            "invalid_format_rate": invalid / total if total > 0 else 0,
            
            # Average rewards
            "avg_reward": _stats["total_reward"] / total if total > 0 else 0,
            "avg_reward_benign": _stats["benign_reward"] / benign_total if benign_total > 0 else 0,
            "avg_reward_error": _stats["error_reward"] / error_total if error_total > 0 else 0,
            
            # By mode breakdown
            "benign_accuracy": _stats["benign_correct"] / benign_total if benign_total > 0 else 0,
            "error_accuracy": _stats["error_correct"] / error_total if error_total > 0 else 0,
            
            # Raw counts
            "benign_count": benign_total,
            "error_count": error_total,
            "correct_classifications": correct,
            "wrong_classifications": wrong,
            
            # Token/Generation metrics
            "avg_response_chars": total_chars / total if total > 0 else 0,
            "avg_response_tokens_approx": (total_chars / 4) / total if total > 0 else 0,  # ~4 chars per token
            "min_response_chars": min_chars,
            "max_response_chars": max_chars,
            "truncation_rate": truncated / total if total > 0 else 0,
            "truncated_responses": truncated,
            "responses_with_think_tags": _stats["responses_with_think_tags"],
            "responses_missing_closing_think": _stats["responses_missing_closing_think"],
            
            # Note similarity metrics
            "avg_similarity_benign": _stats["total_similarity_benign"] / max(_stats["benign_similarity_count"], 1),
            "similarity_benign_count": _stats["benign_similarity_count"],
            "avg_similarity_error": _stats["total_similarity_error"] / max(_stats["error_similarity_count"], 1),
            "similarity_error_count": _stats["error_similarity_count"],
        }


def save_summary():
    """Save summary statistics to file."""
    summary = get_summary_stats()
    summary["timestamp"] = datetime.now().isoformat()
    try:
        with open(SUMMARY_FILE, 'w') as f:
            json.dump(summary, f, indent=2)
    except Exception:
        pass


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


def compute_similarity(text1: str, text2: str) -> float:
    """Compute similarity ratio between two texts using SequenceMatcher.
    
    Args:
        text1: First text (e.g., original note)
        text2: Second text (e.g., generated/modified note)
        
    Returns:
        float: Similarity ratio (0.0=completely different, 1.0=identical)
    """
    if not text1 or not text2:
        return 0.0
    
    # Normalize whitespace for fair comparison
    text1_norm = ' '.join(text1.split())
    text2_norm = ' '.join(text2.split())
    
    return SequenceMatcher(None, text1_norm, text2_norm).ratio()


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
    
    For Injector: generated_note: + final_answer: + changes_made:
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
        r'(generated_note:|Explanation:|changes_made:)\s*\S+',
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


def detect_truncation(response: str) -> dict:
    """Detect if response was truncated due to token limits.
    
    Returns dict with:
        - is_truncated: bool
        - has_think_tag: bool
        - missing_closing_think: bool
        - response_chars: int
        - response_tokens_approx: int
    """
    if not response:
        return {
            "is_truncated": False,
            "has_think_tag": False,
            "missing_closing_think": False,
            "response_chars": 0,
            "response_tokens_approx": 0,
        }
    
    response_chars = len(response)
    response_tokens_approx = response_chars // 4  # Rough estimate
    
    has_think_tag = "<think>" in response.lower()
    has_closing_think = "</think>" in response.lower()
    missing_closing_think = has_think_tag and not has_closing_think
    
    # Truncation indicators:
    # 1. Has opening <think> but no closing </think>
    # 2. Ends mid-word (no punctuation or whitespace at end)
    # 3. No final_answer after opening <think>
    # 4. Ends with incomplete sentence
    
    ends_cleanly = response.rstrip().endswith(('.', '!', '?', '"', "'", ')', ']', '}', '>'))
    has_final_answer = bool(re.search(r'final_answer:', response, re.IGNORECASE))
    
    is_truncated = (
        missing_closing_think or
        (has_think_tag and not has_final_answer) or
        (not ends_cleanly and response_chars > 100)  # Long response that doesn't end cleanly
    )
    
    return {
        "is_truncated": is_truncated,
        "has_think_tag": has_think_tag,
        "missing_closing_think": missing_closing_think,
        "response_chars": response_chars,
        "response_tokens_approx": response_tokens_approx,
        "has_final_answer": has_final_answer,
        "ends_cleanly": ends_cleanly,
    }


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Compute reward score for medical error detection self-play.
    
    This is called by verl's RewardManager after rollout.
    
    Following SeRL paper (arXiv:2506.07468) Algorithm 1:
    - The model acts as Assessor, classifying the note
    - If Assessor is correct: Assessor wins (reward = +1.0)
    - If Assessor is wrong: Injector wins (reward = -1.0)
    
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
        float: Reward score (RD for Assessor perspective)
    
    NOTE: In multi-turn mode, MedicalGameInteraction handles the rewards.
    This function is used for single-turn training or fallback.
    """
    global _stats
    
    # Ensure we have valid inputs
    solution_str = solution_str or ""
    ground_truth = ground_truth or ""
    extra_info = make_serializable(extra_info) if extra_info else {}
    mode = extra_info.get("mode", "unknown")
    
    # Detect truncation and get token metrics
    truncation_info = detect_truncation(solution_str)
    
    # Parse model's classification
    model_answer = parse_final_answer(solution_str)
    
    # Extract generated note (for logging/analysis)
    generated_note = extract_generated_note(solution_str)
    
    # Calculate similarity between original and generated note
    original_note = extra_info.get("correct_note", "")
    similarity = compute_similarity(original_note, generated_note or "")
    
    # Check format compliance
    has_valid_format = check_format_compliance(solution_str)
    format_bonus = FORMAT_BONUS if has_valid_format else 0.0
    
    # Compute reward based on classification accuracy
    # This is RD (Defender/Assessor reward) in SeRL terminology
    if model_answer == ground_truth:
        # Assessor correctly classified the note - Assessor wins
        reward = REWARD_WIN + format_bonus
        outcome = "correct"
    elif model_answer in ["CORRECT", "INCORRECT"]:
        # Assessor classified but was wrong - Injector wins
        reward = REWARD_LOSE + format_bonus
        outcome = "wrong"
    else:
        # Model failed to produce valid classification
        reward = REWARD_LOSE  # No format bonus for invalid output
        outcome = "invalid_format"
    
    # Update global statistics (thread-safe)
    with _stats_lock:
        _stats["total_interactions"] += 1
        _stats["total_reward"] += reward
        
        # Token/truncation metrics
        resp_chars = truncation_info["response_chars"]
        _stats["total_response_chars"] += resp_chars
        _stats["total_response_tokens_approx"] += truncation_info["response_tokens_approx"]
        if resp_chars < _stats["min_response_chars"]:
            _stats["min_response_chars"] = resp_chars
        if resp_chars > _stats["max_response_chars"]:
            _stats["max_response_chars"] = resp_chars
        if truncation_info["is_truncated"]:
            _stats["truncated_responses"] += 1
        if truncation_info["has_think_tag"]:
            _stats["responses_with_think_tags"] += 1
        if truncation_info["missing_closing_think"]:
            _stats["responses_missing_closing_think"] += 1
        
        # Track similarity metrics by mode
        if generated_note and original_note:
            if mode == "benign":
                _stats["total_similarity_benign"] += similarity
                _stats["benign_similarity_count"] += 1
            else:
                _stats["total_similarity_error"] += similarity
                _stats["error_similarity_count"] += 1
        
        if outcome == "correct":
            _stats["correct_classifications"] += 1
            if mode == "benign":
                _stats["benign_correct"] += 1
            else:
                _stats["error_correct"] += 1
        elif outcome == "wrong":
            _stats["wrong_classifications"] += 1
            if mode == "benign":
                _stats["benign_wrong"] += 1
            else:
                _stats["error_wrong"] += 1
        else:
            _stats["invalid_format"] += 1
        
        if mode == "benign":
            _stats["benign_count"] += 1
            _stats["benign_reward"] += reward
        else:
            _stats["error_count"] += 1
            _stats["error_reward"] += reward
        
        # Save summary every 100 interactions
        if _stats["total_interactions"] % 100 == 0:
            save_summary()
    
    # Build comprehensive log entry for failure analysis
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "data_source": str(data_source),
        
        # Game outcome (from Assessor perspective)
        "ground_truth": str(ground_truth),
        "model_answer": model_answer,
        "outcome": outcome,
        "reward": float(reward),
        "reward_injector": -float(reward) if outcome != "invalid_format" else 0.0,  # Zero-sum
        "has_valid_format": has_valid_format,
        
        # Mode info
        "mode": mode,
        "note_id": extra_info.get("note_id", ""),
        
        # Similarity metrics (Injector modifications)
        "note_similarity": float(similarity),
        "has_generated_note": bool(generated_note),
        
        # Token/Generation metrics
        "response_chars": truncation_info["response_chars"],
        "response_tokens_approx": truncation_info["response_tokens_approx"],
        "is_truncated": truncation_info["is_truncated"],
        "has_think_tag": truncation_info["has_think_tag"],
        "missing_closing_think": truncation_info["missing_closing_think"],
        
        # CRITICAL FOR ANALYSIS: The actual clinical notes
        "original_correct_note": extra_info.get("correct_note", "")[:1000],
        "original_incorrect_note": extra_info.get("incorrect_note", "")[:1000],
        "error_type": extra_info.get("error_type", ""),
        "error_sentence": extra_info.get("error_sentence", ""),
        "corrected_sentence": extra_info.get("corrected_sentence", ""),
        
        # Model outputs
        "generated_note": (generated_note or "")[:1000],
        "model_response_full": solution_str[:3000],  # Increased to see more truncation
    }
    
    # Append to log file
    try:
        with open(LOG_FILE, 'a') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        try:
            error_log = LOG_DIR / "reward_errors.log"
            with open(error_log, 'a') as f:
                f.write(f"{datetime.now().isoformat()} Error: {e}\n")
        except:
            pass
    
    return reward


def print_summary():
    """Print summary statistics to stdout."""
    summary = get_summary_stats()
    print("\n" + "="*70)
    print("SELF-PLAY TRAINING SUMMARY")
    print("="*70)
    print(f"Total Interactions: {summary.get('total_interactions', 0)}")
    print(f"Overall Accuracy: {summary.get('accuracy', 0):.2%}")
    print(f"Assessor Win Rate: {summary.get('win_rate_assessor', 0):.2%}")
    print(f"Injector Win Rate: {summary.get('win_rate_injector', 0):.2%}")
    print(f"Invalid Format Rate: {summary.get('invalid_format_rate', 0):.2%}")
    print("-"*70)
    print(f"Avg Reward (Overall): {summary.get('avg_reward', 0):.3f}")
    print(f"Avg Reward (Benign): {summary.get('avg_reward_benign', 0):.3f}")
    print(f"Avg Reward (Error): {summary.get('avg_reward_error', 0):.3f}")
    print("-"*70)
    print(f"Benign Accuracy: {summary.get('benign_accuracy', 0):.2%} ({summary.get('benign_count', 0)} samples)")
    print(f"Error Accuracy: {summary.get('error_accuracy', 0):.2%} ({summary.get('error_count', 0)} samples)")
    print("-"*70)
    print("TOKEN/GENERATION METRICS:")
    print(f"  Avg Response Length: {summary.get('avg_response_chars', 0):.0f} chars (~{summary.get('avg_response_tokens_approx', 0):.0f} tokens)")
    print(f"  Min Response: {summary.get('min_response_chars', 0)} chars")
    print(f"  Max Response: {summary.get('max_response_chars', 0)} chars")
    print(f"  Truncation Rate: {summary.get('truncation_rate', 0):.2%} ({summary.get('truncated_responses', 0)} truncated)")
    print(f"  Responses with <think>: {summary.get('responses_with_think_tags', 0)}")
    print(f"  Missing </think>: {summary.get('responses_missing_closing_think', 0)}")
    print("-"*70)
    print("NOTE SIMILARITY (Original vs Generated):")
    benign_sim_count = summary.get('similarity_benign_count', 0)
    error_sim_count = summary.get('similarity_error_count', 0)
    if benign_sim_count > 0:
        print(f"  Benign mode: {summary.get('avg_similarity_benign', 0):.1%} similarity ({benign_sim_count} samples)")
    if error_sim_count > 0:
        print(f"  Error mode: {summary.get('avg_similarity_error', 0):.1%} similarity ({error_sim_count} samples)")
    print("="*70 + "\n")
    
    # Save final summary
    save_summary()


# Register cleanup to print summary at exit
import atexit
atexit.register(print_summary)
