"""Shared game-level reward logic for batched self-play.

This module is used by the vLLM agent loop path and by legacy reward wrappers.
It keeps the judge-derived game truth and the assessor grading logic in one
place so both rollout and offline analysis agree on semantics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from scripts.self_play.utils import parse_assessor_answer

# Keep these values aligned with reward_function.py.
REWARD_EXACT = 1.0
REWARD_PARTIAL = 0.3
REWARD_MISS = -1.0
FORMAT_BONUS = 0.2

VALID_JUDGE_VERDICTS = {"SAME", "CHANGED"}


@dataclass(frozen=True)
class AssessorRewardResult:
    reward: float
    outcome: str
    label: str
    pred_sid: Optional[int]
    ground_truth: Optional[str]
    has_valid_format: bool


def derive_assessor_ground_truth(
    judge_verdict: str,
    changed_sid: Optional[int],
) -> Optional[str]:
    """Map judge truth to assessor target.

    SAME -> CORRECT
    CHANGED -> changed sentence id
    ABSTAIN / invalid -> None
    """
    verdict = str(judge_verdict or "").upper()
    if verdict == "SAME":
        return "CORRECT"
    if verdict == "CHANGED" and changed_sid is not None:
        return str(int(changed_sid))
    return None


def compute_injector_game_reward(
    mode: str,
    judge_verdict: str,
    *,
    parse_success: bool,
) -> tuple[float, str, bool]:
    """Reward injector for achieving the requested edit type.

    Returns (reward, outcome, game_valid).
    """
    if not parse_success:
        return REWARD_MISS, "parse_failure", False

    verdict = str(judge_verdict or "").upper()
    if verdict not in VALID_JUDGE_VERDICTS:
        return REWARD_MISS, "judge_invalid", False

    expected = "SAME" if mode == "benign" else "CHANGED"
    if verdict == expected:
        return REWARD_EXACT + FORMAT_BONUS, "exact_match", True
    return REWARD_MISS + FORMAT_BONUS, "wrong_edit_type", True


def compute_assessor_game_reward(
    assessor_output: str,
    *,
    judge_verdict: str,
    changed_sid: Optional[int],
    ground_truth_override: Optional[str] = None,
) -> AssessorRewardResult:
    """Grade assessor against the judge-derived truth.

    SAME: only CORRECT is rewarded.
    CHANGED: exact sentence gets full reward, wrong sentence partial reward.
    """
    label, pred_sid = parse_assessor_answer(assessor_output or "")
    has_valid_format = label != "UNKNOWN"
    format_bonus = FORMAT_BONUS if has_valid_format else 0.0
    ground_truth = ground_truth_override or derive_assessor_ground_truth(judge_verdict, changed_sid)

    if ground_truth is None:
        return AssessorRewardResult(
            reward=0.0,
            outcome="game_invalid",
            label=label,
            pred_sid=pred_sid,
            ground_truth=None,
            has_valid_format=has_valid_format,
        )

    if ground_truth == "CORRECT":
        if label == "CORRECT":
            return AssessorRewardResult(
                reward=REWARD_EXACT + format_bonus,
                outcome="exact_match",
                label=label,
                pred_sid=pred_sid,
                ground_truth=ground_truth,
                has_valid_format=has_valid_format,
            )
        return AssessorRewardResult(
            reward=REWARD_MISS + format_bonus,
            outcome="miss",
            label=label,
            pred_sid=pred_sid,
            ground_truth=ground_truth,
            has_valid_format=has_valid_format,
        )

    # judge says CHANGED
    if label == "CORRECT":
        return AssessorRewardResult(
            reward=REWARD_MISS + format_bonus,
            outcome="miss",
            label=label,
            pred_sid=pred_sid,
            ground_truth=ground_truth,
            has_valid_format=has_valid_format,
        )

    if label == "ERROR":
        pred_str = str(pred_sid) if pred_sid is not None else ""
        if pred_str == ground_truth:
            reward = REWARD_EXACT + format_bonus
            outcome = "exact_match"
        elif pred_sid is not None:
            reward = REWARD_PARTIAL + format_bonus
            outcome = "partial_match"
        else:
            reward = REWARD_PARTIAL
            outcome = "partial_match"
        return AssessorRewardResult(
            reward=reward,
            outcome=outcome,
            label=label,
            pred_sid=pred_sid,
            ground_truth=ground_truth,
            has_valid_format=has_valid_format,
        )

    return AssessorRewardResult(
        reward=REWARD_MISS,
        outcome="invalid_format",
        label=label,
        pred_sid=pred_sid,
        ground_truth=ground_truth,
        has_valid_format=False,
    )
