"""Custom reward function for MedSeRL self-play training.

Following verl documentation format:
https://verl.readthedocs.io/en/latest/preparation/reward_function.html

Medical Error Detection + Localization Self-Play Game (adapted from SeRL paper arXiv:2506.07468):
- Single model plays both roles (Injector and Assessor)
- Injector: Modifies clinical note (benign edit OR error injection)
- Assessor: Outputs CORRECT (no error) or an integer sentence number (error location)

3-tier rewards:
- Exact match (CORRECT↔CORRECT, or same sentence number): +1.0
- Detection only (error detected but wrong sentence number): +0.3
- Miss (wrong classification or unparseable): -1.0
- Format bonus: +0.2 for valid output format (CORRECT or bare integer)

Note: In our implementation, each example is assessed independently.
The "Injector" already produced its output (the modified note in training data).
The model acts as "Assessor" and classifies/localizes - this is what we reward.
"""

import json
import logging
import os
import re
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from collections import defaultdict
import threading
from difflib import SequenceMatcher

from scripts.self_play.utils import parse_assessor_answer, strip_thinking

logger = logging.getLogger(__name__)


# Global log file path - creates new file per training run
LOG_DIR = Path(__file__).parent.parent.parent / "results" / "self_play" / "interactions"
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / f"interactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
SUMMARY_FILE = LOG_DIR / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

# Reward values (3-tier)
REWARD_EXACT = 1.0     # Exact match (correct label + sentence)
REWARD_PARTIAL = 0.3   # Detected error but wrong sentence number
REWARD_MISS = -1.0     # Missed, false positive, or invalid
FORMAT_BONUS = 0.2     # Bonus for parseable output format

# Global statistics tracker (thread-safe)
_stats_lock = threading.Lock()
_stats = {
    "total_interactions": 0,
    "exact_match": 0,
    "partial_match": 0,
    "miss": 0,
    "invalid_format": 0,
    # By mode
    "benign_correct": 0,     # CORRECT on benign note
    "benign_false_pos": 0,   # flagged sentence on benign note
    "error_exact": 0,        # correct sentence number
    "error_partial": 0,      # detected error, wrong sentence
    "error_missed": 0,       # said CORRECT on error note
    # Rewards
    "total_reward": 0.0,
    "benign_reward": 0.0,
    "error_reward": 0.0,
    # For computing averages
    "benign_count": 0,
    "error_count": 0,
    # Token metrics
    "total_response_chars": 0,
    "total_response_tokens_approx": 0,
    "min_response_chars": float('inf'),
    "max_response_chars": 0,
    "truncated_responses": 0,
    "responses_with_think_tags": 0,
    "responses_missing_closing_think": 0,
    # Phase separation metrics
    "phases_separated_count": 0,
    "injector_produced_note_count": 0,
    "assessor_actually_ran_count": 0,
    "injector_truncated_count": 0,
    "assessor_truncated_count": 0,
    "injector_total_chars": 0,
    "assessor_total_chars": 0,
}


def get_summary_stats() -> Dict[str, Any]:
    """Get summary statistics for the training run."""
    with _stats_lock:
        total = _stats["total_interactions"]
        if total == 0:
            return {"message": "No interactions yet"}
        
        exact = _stats["exact_match"]
        partial = _stats["partial_match"]
        miss = _stats["miss"]
        invalid = _stats["invalid_format"]
        
        benign_total = _stats["benign_count"]
        error_total = _stats["error_count"]
        
        # Token metrics
        total_chars = _stats["total_response_chars"]
        min_chars = _stats["min_response_chars"] if _stats["min_response_chars"] != float('inf') else 0
        max_chars = _stats["max_response_chars"]
        truncated = _stats["truncated_responses"]
        
        # Phase separation metrics
        phases_sep = _stats["phases_separated_count"]
        injector_notes = _stats["injector_produced_note_count"]
        assessor_ran = _stats["assessor_actually_ran_count"]
        
        return {
            "total_interactions": total,
            "exact_match_rate": exact / total if total > 0 else 0,
            "partial_match_rate": partial / total if total > 0 else 0,
            "miss_rate": miss / total if total > 0 else 0,
            "invalid_format_rate": invalid / total if total > 0 else 0,
            
            # Average rewards
            "avg_reward": _stats["total_reward"] / total if total > 0 else 0,
            "avg_reward_benign": _stats["benign_reward"] / benign_total if benign_total > 0 else 0,
            "avg_reward_error": _stats["error_reward"] / error_total if error_total > 0 else 0,
            
            # Benign breakdown
            "benign_correct": _stats["benign_correct"],
            "benign_false_pos": _stats["benign_false_pos"],
            "benign_accuracy": _stats["benign_correct"] / benign_total if benign_total > 0 else 0,
            
            # Error breakdown
            "error_exact": _stats["error_exact"],
            "error_partial": _stats["error_partial"],
            "error_missed": _stats["error_missed"],
            "error_localize_accuracy": _stats["error_exact"] / error_total if error_total > 0 else 0,
            "error_detect_rate": (_stats["error_exact"] + _stats["error_partial"]) / error_total if error_total > 0 else 0,
            
            # Counts
            "benign_count": benign_total,
            "error_count": error_total,
            "exact_match": exact,
            "partial_match": partial,
            "miss": miss,
            
            # Token metrics
            "avg_response_chars": total_chars / total if total > 0 else 0,
            "avg_response_tokens_approx": (total_chars / 4) / total if total > 0 else 0,
            "min_response_chars": min_chars,
            "max_response_chars": max_chars,
            "truncation_rate": truncated / total if total > 0 else 0,
            "truncated_responses": truncated,
            "responses_with_think_tags": _stats["responses_with_think_tags"],
            "responses_missing_closing_think": _stats["responses_missing_closing_think"],
            
            # Phase separation
            "phases_separated_rate": phases_sep / total if total > 0 else 0,
            "phases_separated_count": phases_sep,
            "injector_produced_note_rate": injector_notes / total if total > 0 else 0,
            "assessor_actually_ran_rate": assessor_ran / total if total > 0 else 0,
            "injector_truncated_count": _stats["injector_truncated_count"],
            "assessor_truncated_count": _stats["assessor_truncated_count"],
            "avg_injector_chars": _stats["injector_total_chars"] / total if total > 0 else 0,
            "avg_assessor_chars": _stats["assessor_total_chars"] / total if total > 0 else 0,
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


def parse_final_answer(response: str) -> Tuple[str, Optional[int]]:
    """Extract assessor answer from model response.
    
    Uses shared parse_assessor_answer from utils.py which handles:
    - "CORRECT" → ("CORRECT", None)
    - Bare integer "3" → ("ERROR", 3)
    - "final_answer: 5" → ("ERROR", 5)
    - Invalid → ("UNKNOWN", None)
    
    Returns:
        (label, sentence_id) where label is CORRECT/ERROR/UNKNOWN
    """
    if not response:
        return ("UNKNOWN", None)
    return parse_assessor_answer(response)


def check_format_compliance(response: str) -> bool:
    """Check if response follows required format.
    
    Valid formats:
    - "CORRECT" (bare word, optionally with CoT before)
    - A bare integer like "3" or "12"
    - With optional <think>...</think> block before the answer
    """
    if not response:
        return False
    label, _ = parse_assessor_answer(response)
    return label != "UNKNOWN"


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
    
    # Primary truncation signal: opened <think> but never closed it.
    # Secondary: very long response that doesn't end with punctuation.
    # NOTE: We no longer check for 'final_answer:' — injector outputs
    # never contain that string, causing false-positive truncation.
    is_truncated = (
        missing_closing_think or
        (not ends_cleanly and response_chars > 100)  # Long response that doesn't end cleanly
    )
    
    return {
        "is_truncated": is_truncated,
        "has_think_tag": has_think_tag,
        "missing_closing_think": missing_closing_think,
        "response_chars": response_chars,
        "response_tokens_approx": response_tokens_approx,
        "ends_cleanly": ends_cleanly,
    }


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Compute 3-tier reward score for medical error detection + localization.
    
    This is called by verl's RewardManager after rollout.
    
    3-tier reward (adapted from SeRL paper arXiv:2506.07468):
    - Exact match (CORRECT↔CORRECT, or same sentence number): +1.0
    - Detection only (error detected but wrong sentence number): +0.3
    - Miss (wrong classification or unparseable): -1.0
    - Format bonus: +0.2 for valid output format
    
    Args:
        data_source (str): Dataset identifier (e.g., 'medec_selfplay')
        solution_str (str): The model's generated response
        ground_truth (str): "CORRECT" or str(sentence_id) from data
        extra_info (dict): Additional information from dataset
        
    Returns:
        float: Reward score (Assessor perspective)
    """
    global _stats
    
    # Smoke-test sentinel — confirms real model rollouts reach the reward function
    logger.info(
        "compute_score called: source=%s gt=%s solution=%r...",
        data_source, ground_truth, (solution_str or "")[:80],
    )
    
    # Ensure we have valid inputs
    solution_str = solution_str or ""
    ground_truth = str(ground_truth) if ground_truth else ""
    extra_info = make_serializable(extra_info) if extra_info else {}
    mode = extra_info.get("mode", "unknown")
    
    # Detect truncation and get token metrics
    truncation_info = detect_truncation(solution_str)
    
    # Parse model's output using shared parser
    label, pred_sid = parse_final_answer(solution_str)
    
    # Check format compliance
    has_valid_format = check_format_compliance(solution_str)
    format_bonus = FORMAT_BONUS if has_valid_format else 0.0
    
    # ── Guard: If this is a single-turn rollout where only the injector ran,
    # the solution_str contains a clinical note (not an assessor answer).
    # Numbers in clinical text (e.g., Na=134) would be mis-parsed as sentence IDs.
    # Detect this by checking if the response looks like an injector compact output.
    _looks_like_injector = bool(re.search(
        r'(?:^|\n)\s*\d+\.\s+\S', solution_str.split('</think>')[-1] if '</think>' in solution_str.lower() else solution_str
    ))
    if _looks_like_injector and label == "ERROR" and pred_sid is not None and pred_sid > 20:
        # Sentence IDs in real notes are 1–~15.  A pred_sid > 20 almost certainly
        # came from a clinical value (lab result, dose, etc.), not an assessor answer.
        logger.warning(
            "Guard: pred_sid=%d looks like a clinical value, not a sentence ID. "
            "Resetting to UNKNOWN (likely single-turn rollout without assessor).",
            pred_sid,
        )
        label, pred_sid = "UNKNOWN", None
        has_valid_format = False
        format_bonus = 0.0
    
    # --- 3-tier reward computation ---
    gt_is_correct = (ground_truth == "CORRECT")
    
    if gt_is_correct and label == "CORRECT":
        # Exact match: note is correct, assessor says CORRECT
        reward = REWARD_EXACT + format_bonus
        outcome = "exact_match"
    elif gt_is_correct and label != "CORRECT":
        # False positive: note is correct, assessor flagged a sentence
        reward = REWARD_MISS + format_bonus
        outcome = "miss"
    elif not gt_is_correct and label == "CORRECT":
        # Missed error: note has error, assessor said CORRECT
        reward = REWARD_MISS + format_bonus
        outcome = "miss"
    elif not gt_is_correct and label == "ERROR":
        # Detected error - check sentence number
        pred_str = str(pred_sid) if pred_sid else ""
        if pred_str == ground_truth:
            # Exact sentence match
            reward = REWARD_EXACT + format_bonus
            outcome = "exact_match"
        elif pred_sid is not None:
            # Detected error but wrong sentence number → partial credit
            reward = REWARD_PARTIAL + format_bonus
            outcome = "partial_match"
        else:
            # Detected error but no sentence number
            reward = REWARD_PARTIAL
            outcome = "partial_match"
    else:
        # UNKNOWN / unparseable
        reward = REWARD_MISS
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
        
        # Outcome tracking
        if outcome == "exact_match":
            _stats["exact_match"] += 1
        elif outcome == "partial_match":
            _stats["partial_match"] += 1
        elif outcome == "miss":
            _stats["miss"] += 1
        else:
            _stats["invalid_format"] += 1
        
        # Mode-specific
        if mode == "benign":
            _stats["benign_count"] += 1
            _stats["benign_reward"] += reward
            if outcome == "exact_match":
                _stats["benign_correct"] += 1
            else:
                _stats["benign_false_pos"] += 1
        else:
            _stats["error_count"] += 1
            _stats["error_reward"] += reward
            if outcome == "exact_match":
                _stats["error_exact"] += 1
            elif outcome == "partial_match":
                _stats["error_partial"] += 1
            else:
                _stats["error_missed"] += 1
        
        # Save summary every 100 interactions
        if _stats["total_interactions"] % 100 == 0:
            save_summary()
    
    # =========================================================================
    # MULTI-TURN RESPONSE PARSING (for concatenated rollouts)
    # =========================================================================
    injector_response = solution_str
    assessor_response = ""
    
    # Try to find the boundary between turns
    assessor_markers = [
        r'<\|im_start\|>user\s*\n',
        r'\nuser\s*\n',
    ]
    
    turn_boundary = None
    for marker in assessor_markers:
        match = re.search(marker, solution_str, re.IGNORECASE | re.DOTALL)
        if match:
            turn_boundary = match.start()
            break
    
    if turn_boundary:
        injector_response = solution_str[:turn_boundary].strip()
        assessor_response = solution_str[turn_boundary:].strip()
    else:
        assistant_blocks = re.findall(
            r'<\|im_start\|>assistant(.*?)(?=<\|im_end\|>|<\|im_start\|>|$)', 
            solution_str, re.DOTALL
        )
        if len(assistant_blocks) >= 2:
            injector_response = assistant_blocks[0].strip()
            assessor_response = assistant_blocks[-1].strip()
    
    injector_truncation = detect_truncation(injector_response)
    assessor_truncation = detect_truncation(assessor_response) if assessor_response else {
        "is_truncated": False, "has_think_tag": False, "missing_closing_think": False,
        "response_chars": 0, "response_tokens_approx": 0
    }
    
    assessor_actually_ran = len(assessor_response) > 10
    
    with _stats_lock:
        if turn_boundary is not None or assessor_actually_ran:
            _stats["phases_separated_count"] += 1
        if assessor_actually_ran:
            _stats["assessor_actually_ran_count"] += 1
        if injector_truncation.get("is_truncated", False):
            _stats["injector_truncated_count"] += 1
        if assessor_truncation.get("is_truncated", False):
            _stats["assessor_truncated_count"] += 1
        _stats["injector_total_chars"] += injector_truncation.get("response_chars", 0)
        _stats["assessor_total_chars"] += assessor_truncation.get("response_chars", 0)
    
    # Build log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "data_source": str(data_source),
        
        # Game outcome
        "ground_truth": ground_truth,
        "assessor_label": label,
        "assessor_pred_sid": pred_sid,
        "outcome": outcome,
        "reward": float(reward),
        "has_valid_format": has_valid_format,
        
        # Mode info
        "mode": mode,
        "note_id": extra_info.get("note_id", ""),
        "error_type": extra_info.get("error_type", ""),
        
        # Token metrics
        "response_chars": truncation_info["response_chars"],
        "response_tokens_approx": truncation_info["response_tokens_approx"],
        "is_truncated": truncation_info["is_truncated"],
        "has_think_tag": truncation_info["has_think_tag"],
        
        # Phase separation
        "phases_separated": turn_boundary is not None or assessor_actually_ran,
        "assessor_actually_ran": assessor_actually_ran,
        "injector_response_chars": injector_truncation["response_chars"],
        "assessor_response_chars": assessor_truncation["response_chars"],
        
        # Truncated responses for debugging
        "injector_response": injector_response[:4000],
        "assessor_response": assessor_response[:2000],
        "model_response_full": solution_str[:8000],
    }
    
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
    print("SELF-PLAY TRAINING SUMMARY (3-Tier Rewards)")
    print("="*70)
    print(f"Total Interactions: {summary.get('total_interactions', 0)}")
    print(f"Exact Match Rate: {summary.get('exact_match_rate', 0):.2%}")
    print(f"Partial Match Rate: {summary.get('partial_match_rate', 0):.2%}")
    print(f"Miss Rate: {summary.get('miss_rate', 0):.2%}")
    print(f"Invalid Format Rate: {summary.get('invalid_format_rate', 0):.2%}")
    print("-"*70)
    print(f"Avg Reward (Overall): {summary.get('avg_reward', 0):.3f}")
    print(f"Avg Reward (Benign): {summary.get('avg_reward_benign', 0):.3f}")
    print(f"Avg Reward (Error): {summary.get('avg_reward_error', 0):.3f}")
    print("-"*70)
    print(f"Benign Accuracy: {summary.get('benign_accuracy', 0):.2%} ({summary.get('benign_count', 0)} samples)")
    print(f"  Correct: {summary.get('benign_correct', 0)} | False Positive: {summary.get('benign_false_pos', 0)}")
    print("-"*70)
    print(f"Error Detection Rate: {summary.get('error_detect_rate', 0):.2%} ({summary.get('error_count', 0)} samples)")
    print(f"Error Localization Accuracy: {summary.get('error_localize_accuracy', 0):.2%}")
    print(f"  Exact: {summary.get('error_exact', 0)} | Partial: {summary.get('error_partial', 0)} | Missed: {summary.get('error_missed', 0)}")
    print("-"*70)
    print("PHASE SEPARATION (Multi-Turn):")
    print(f"  Phases Separated: {summary.get('phases_separated_rate', 0):.2%}")
    print(f"  Assessor Actually Ran: {summary.get('assessor_actually_ran_rate', 0):.2%}")
    print(f"  Injector Truncated: {summary.get('injector_truncated_count', 0)}")
    print(f"  Assessor Truncated: {summary.get('assessor_truncated_count', 0)}")
    print("-"*70)
    print("TOKEN/GENERATION METRICS:")
    print(f"  Avg Response Length: {summary.get('avg_response_chars', 0):.0f} chars (~{summary.get('avg_response_tokens_approx', 0):.0f} tokens)")
    print(f"  Truncation Rate: {summary.get('truncation_rate', 0):.2%} ({summary.get('truncated_responses', 0)} truncated)")
    print(f"  Responses with <think>: {summary.get('responses_with_think_tags', 0)}")
    print("="*70 + "\n")
    
    save_summary()


# Register cleanup to print summary at exit
import atexit
atexit.register(print_summary)
