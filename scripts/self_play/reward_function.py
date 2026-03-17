"""Custom reward function for MedSeRL self-play training.

Following verl documentation format:
https://verl.readthedocs.io/en/latest/preparation/reward_function.html

Medical Error Detection + Localization Self-Play Game (adapted from SeRL paper arXiv:2506.07468):
- Single model plays both roles (Injector and Assessor) in separate GRPO passes
- Injector: Modifies clinical note (benign edit OR error injection)
- Assessor: Outputs CORRECT (no error) or an integer sentence number (error location)

Role dispatch in compute_score (via extra_info["role"]):
  "assessor" (default): 3-tier reward based on detection accuracy
  "injector": format + sentence-placement reward using MEDEC ground truth

3-tier assessor rewards:
- Exact match (CORRECT↔CORRECT, or same sentence number): +1.0
- Detection only (error detected but wrong sentence number): +0.3
- Miss (wrong classification or unparseable): -1.0
- Format bonus: +0.2 for valid output format

Injector rewards (proxy using MEDEC ground truth):
- Correct sentence + valid format: +1.2
- Wrong sentence + valid format:   +0.5
- Invalid format:                  -1.0
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

from scripts.self_play.utils import parse_assessor_answer, strip_thinking, parse_injector_compact

logger = logging.getLogger(__name__)


# Log file resolved lazily on first write so MEDSERL_GAME_LOG (set via Ray
# runtime_env.env_vars) is always visible — same pattern as medical_game_interaction.py.
_DEFAULT_LOG_DIR = Path(__file__).parent.parent.parent / "results" / "self_play" / "interactions"
_DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
_resolved_log_file: "Path | None" = None
_resolved_summary_file: "Path | None" = None


def _get_log_files():
    global _resolved_log_file, _resolved_summary_file
    if _resolved_log_file is not None:
        return _resolved_log_file, _resolved_summary_file
    env_log = os.environ.get("MEDSERL_GAME_LOG")
    if env_log:
        base = Path(env_log).parent
        try:
            base.mkdir(parents=True, exist_ok=True)
            _resolved_log_file = base / f"interactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
            _resolved_summary_file = base / f"summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            return _resolved_log_file, _resolved_summary_file
        except Exception:
            pass
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    _resolved_log_file = _DEFAULT_LOG_DIR / f"interactions_{ts}.jsonl"
    _resolved_summary_file = _DEFAULT_LOG_DIR / f"summary_{ts}.json"
    return _resolved_log_file, _resolved_summary_file


# Keep module-level aliases for backward compat with any code that reads LOG_FILE directly
def _log_file() -> Path:
    return _get_log_files()[0]


LOG_DIR = _DEFAULT_LOG_DIR  # kept for any external imports

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
        with open(_get_log_files()[1], 'w') as f:
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


def _parse_answer_from_tail(text: str) -> Tuple[str, Optional[int]]:
    """Extract assessor's answer from the LAST part of solution_str.

    In multi-turn rollouts, solution_str = injector_output + assessor_output.
    The assessor (with /no_think) outputs only "CORRECT" or a bare integer.
    These appear at the END; the injector's "N. full sentence" is in the middle.
    Scanning from the tail avoids mis-parsing the injector's sentence number.
    Falls back to parse_assessor_answer on the full text if nothing found.
    """
    if not text:
        return ("UNKNOWN", None)
    lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
    for line in reversed(lines[-10:]):
        line = re.sub(r'</?think>', '', line, flags=re.IGNORECASE).strip()
        if not line:
            continue
        line = re.sub(
            r'^(answer|label|output|result|final_answer)\s*[:=]\s*',
            '', line, flags=re.IGNORECASE
        ).strip()
        if re.fullmatch(r'correct', line, re.IGNORECASE):
            return "CORRECT", None
        m = re.fullmatch(r'\d+', line)
        if m and 1 <= int(m.group()) <= 50:
            return "ERROR", int(m.group())
    return parse_assessor_answer(text)


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


def _compute_assessor_reward(
    solution_str: str,
    ground_truth: str,
) -> tuple:
    """3-tier reward for assessor role. Returns (reward, outcome, label, pred_sid)."""
    label, pred_sid = _parse_answer_from_tail(solution_str)
    has_valid_format = check_format_compliance(solution_str)
    format_bonus = FORMAT_BONUS if has_valid_format else 0.0

    gt_is_correct = (ground_truth == "CORRECT")

    if gt_is_correct and label == "CORRECT":
        reward, outcome = REWARD_EXACT + format_bonus, "exact_match"
    elif gt_is_correct and label != "CORRECT":
        reward, outcome = REWARD_MISS + format_bonus, "miss"
    elif not gt_is_correct and label == "CORRECT":
        reward, outcome = REWARD_MISS + format_bonus, "miss"
    elif not gt_is_correct and label == "ERROR":
        pred_str = str(pred_sid) if pred_sid else ""
        if pred_str == ground_truth:
            reward, outcome = REWARD_EXACT + format_bonus, "exact_match"
        elif pred_sid is not None:
            reward, outcome = REWARD_PARTIAL + format_bonus, "partial_match"
        else:
            reward, outcome = REWARD_PARTIAL, "partial_match"
    else:
        reward, outcome = REWARD_MISS, "invalid_format"
        has_valid_format = False

    return reward, outcome, label, pred_sid, has_valid_format


def _compute_injector_reward(solution_str: str, extra_info: dict) -> tuple:
    """Proxy reward for injector role using MEDEC ground-truth sentence IDs.

    Returns (reward, outcome, pred_sid, has_valid_format).
    """
    sid, modified_text = parse_injector_compact(solution_str)

    if sid is None or modified_text is None:
        return REWARD_MISS, "invalid_format", None, False

    mode = extra_info.get("mode", "error_injection")
    if mode == "benign":
        # Any valid benign edit is rewarded
        return REWARD_EXACT + FORMAT_BONUS, "exact_match", sid, True

    expected_sid = extra_info.get("error_sentence_id")
    if expected_sid is not None and sid == int(expected_sid):
        return REWARD_EXACT + FORMAT_BONUS, "exact_match", sid, True
    elif expected_sid is not None:
        return REWARD_PARTIAL + FORMAT_BONUS, "partial_match", sid, True
    else:
        # No ground-truth sentence ID available — reward valid format
        return REWARD_PARTIAL + FORMAT_BONUS, "partial_match", sid, True


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Compute reward for medical self-play training.

    Dispatches by extra_info["role"]:
      "assessor" (default): 3-tier detection/localization reward
      "injector":           format + sentence-placement proxy reward

    Called by verl's RewardManager after each rollout.
    """
    global _stats

    logger.info(
        "compute_score called: source=%s role=%s gt=%s solution=%r...",
        data_source,
        (extra_info or {}).get("role", "assessor"),
        ground_truth,
        (solution_str or "")[:80],
    )

    solution_str = solution_str or ""
    ground_truth = str(ground_truth) if ground_truth else ""
    extra_info = make_serializable(extra_info) if extra_info else {}
    role = extra_info.get("role", "assessor")
    mode = extra_info.get("mode", "unknown")

    truncation_info = detect_truncation(solution_str)

    # --- Role dispatch ---
    if role == "injector":
        reward, outcome, pred_sid, has_valid_format = _compute_injector_reward(solution_str, extra_info)
        label = "injector"
    else:
        reward, outcome, label, pred_sid, has_valid_format = _compute_assessor_reward(
            solution_str, ground_truth
        )

    # --- Update statistics (thread-safe) ---
    with _stats_lock:
        _stats["total_interactions"] += 1
        _stats["total_reward"] += reward

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

        if outcome == "exact_match":
            _stats["exact_match"] += 1
        elif outcome == "partial_match":
            _stats["partial_match"] += 1
        elif outcome == "miss":
            _stats["miss"] += 1
        else:
            _stats["invalid_format"] += 1

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

        if _stats["total_interactions"] % 100 == 0:
            save_summary()

    # --- Log interaction ---
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "data_source": str(data_source),
        "role": role,
        "ground_truth": ground_truth,
        "pred_label": label,
        "pred_sid": pred_sid,
        "outcome": outcome,
        "reward": float(reward),
        "has_valid_format": has_valid_format,
        "mode": mode,
        "note_id": extra_info.get("note_id", ""),
        "error_type": extra_info.get("error_type", ""),
        "response_chars": truncation_info["response_chars"],
        "is_truncated": truncation_info["is_truncated"],
        "has_think_tag": truncation_info["has_think_tag"],
        "model_response": solution_str[:4000],
    }

    try:
        with open(_get_log_files()[0], 'a') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
    except Exception as e:
        try:
            with open(LOG_DIR / "reward_errors.log", 'a') as f:
                f.write(f"{datetime.now().isoformat()} Error: {e}\n")
        except Exception:
            pass

    return reward


def interaction_reward_passthrough(
    data_source, solution_str, ground_truth, extra_info=None
):
    """Reward function for multi-turn MedicalGameInteraction training.

    The patched tool_agent_loop.py stores the interaction's final info
    dict in agent_data.extra_fields, which verl maps to extra_info here.

    If extra_info contains phase='game_complete' (written by
    MedicalGameInteraction._process_assessor_turn), we read the assessor
    verdict directly from there.  Otherwise we fall back to parsing the
    tail of solution_str for the assessor's answer.
    """
    extra_info = extra_info or {}
    ground_truth = str(ground_truth) if ground_truth else ""

    # Primary path: interaction stored its verdict in extra_fields
    if extra_info.get("phase") == "game_complete":
        label = extra_info.get("assessor_label", "UNKNOWN")
        has_fmt = extra_info.get("has_valid_format", False)
        pred_sid = extra_info.get("assessor_pred_sid")
        fmt_bonus = FORMAT_BONUS if has_fmt else 0.0
        gt_correct = ground_truth == "CORRECT"

        if label == "UNKNOWN":
            return REWARD_MISS
        if gt_correct and label == "CORRECT":
            return REWARD_EXACT + fmt_bonus
        if gt_correct and label != "CORRECT":
            return REWARD_MISS + fmt_bonus
        if not gt_correct and label == "CORRECT":
            return REWARD_MISS + fmt_bonus
        # not gt_correct and error detected
        pred_str = str(pred_sid) if pred_sid is not None else "?"
        if pred_str == ground_truth:
            return REWARD_EXACT + fmt_bonus
        if pred_sid is not None:
            return REWARD_PARTIAL + fmt_bonus
        return REWARD_PARTIAL

    # Fallback: parse the last assessor answer out of solution_str
    # (multi-turn solution_str contains both turns concatenated)
    label, pred_sid = parse_final_answer(solution_str or "")
    has_fmt = label != "UNKNOWN"
    fmt_bonus = FORMAT_BONUS if has_fmt else 0.0
    gt_correct = ground_truth == "CORRECT"

    if label == "UNKNOWN":
        return REWARD_MISS
    if gt_correct and label == "CORRECT":
        return REWARD_EXACT + fmt_bonus
    if gt_correct and label != "CORRECT":
        return REWARD_MISS + fmt_bonus
    if not gt_correct and label == "CORRECT":
        return REWARD_MISS + fmt_bonus
    pred_str = str(pred_sid) if pred_sid is not None else "?"
    if pred_str == ground_truth:
        return REWARD_EXACT + fmt_bonus
    if pred_sid is not None:
        return REWARD_PARTIAL + fmt_bonus
    return REWARD_PARTIAL


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
    print("TOKEN/GENERATION METRICS:")
    print(f"  Avg Response Length: {summary.get('avg_response_chars', 0):.0f} chars (~{summary.get('avg_response_tokens_approx', 0):.0f} tokens)")
    print(f"  Truncation Rate: {summary.get('truncation_rate', 0):.2%} ({summary.get('truncated_responses', 0)} truncated)")
    print(f"  Responses with <think>: {summary.get('responses_with_think_tags', 0)}")
    print("="*70 + "\n")
    
    save_summary()


# Register cleanup to print summary at exit
import atexit
atexit.register(print_summary)
