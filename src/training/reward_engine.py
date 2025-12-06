"""
Reward Engine - Deterministic reward calculation for MedSeRL.

This module computes rewards for the Doctor Agent's predictions based on
ground truth labels. The reward function is deterministic and does not
require external annotation.

Reward Components:
- Structural Reward (+0.1): Output contains <thinking> section
- Outcome Reward (+1.0): Correct classification
- False Negative Penalty (-1.0): Missed error
- False Positive Penalty (-1.5): Incorrect error flag

Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
"""

import re
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# Reward constants
STRUCTURAL_REWARD = 0.1  # Reward for having <thinking> section
CORRECT_CLASSIFICATION_REWARD = 1.0  # Reward for correct classification
FALSE_NEGATIVE_PENALTY = -1.0  # Penalty for missing an error
FALSE_POSITIVE_PENALTY = -1.5  # Higher penalty to reduce alert fatigue


@dataclass
class RewardMetadata:
    """
    Detailed breakdown of reward calculation.

    Attributes:
        structural_reward: +0.1 if output has <thinking> section
        outcome_reward: +1.0 for correct, -1.0/-1.5 for incorrect
        total_reward: Sum of all components
        correct_classification: Whether prediction matched ground truth
        false_positive: Whether clean note was incorrectly flagged as error
        false_negative: Whether an error note was missed
    """
    structural_reward: float
    outcome_reward: float
    total_reward: float
    correct_classification: bool
    false_positive: bool
    false_negative: bool


def has_thinking_section(output: str) -> bool:
    """
    Check if the model output contains a <thinking> section.

    Args:
        output: The raw model output string

    Returns:
        True if output contains <thinking>...</thinking> tags

    Requirements: 5.1
    """
    if not output:
        return False

    # Check for <thinking> tag (case-insensitive)
    pattern = r'<thinking>.*?</thinking>'
    match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)
    return match is not None


def parse_verdict(output: str) -> Optional[str]:
    """
    Extract the verdict content from model output.

    Args:
        output: The raw model output string

    Returns:
        The content of the <verdict> section, or None if not found

    Requirements: 5.1
    """
    if not output:
        return None

    # Extract content between <verdict> tags (case-insensitive)
    pattern = r'<verdict>(.*?)</verdict>'
    match = re.search(pattern, output, re.DOTALL | re.IGNORECASE)

    if match:
        return match.group(1).strip()
    return None


def _is_error_prediction(verdict: str) -> bool:
    """
    Determine if a verdict indicates an error was detected.

    Args:
        verdict: The verdict string content

    Returns:
        True if verdict indicates an error, False otherwise
    """
    if not verdict:
        return False

    verdict_lower = verdict.lower()

    # Check for "No Clinical Error" or "No Error" first
    if "no clinical error" in verdict_lower or "no error" in verdict_lower:
        return False

    # Check if it contains "error" (indicating an error was found)
    return "error" in verdict_lower


def _extract_error_type_from_verdict(verdict: str) -> Optional[str]:
    """
    Extract the specific error type from a verdict string.

    Args:
        verdict: The verdict string content

    Returns:
        The error type if found, None otherwise
    """
    if not verdict:
        return None

    # Standard error types
    error_types = [
        "Diagnosis",
        "Management",
        "Treatment",
        "Pharmacotherapy",
        "Causal Organism"
    ]

    verdict_lower = verdict.lower()

    # Check for each error type
    for error_type in error_types:
        if error_type.lower() in verdict_lower:
            return error_type

    # Try to extract from "Error: [Type]" format
    error_match = re.search(
        r'error:\s*(\w+(?:\s+\w+)?)',
        verdict,
        re.IGNORECASE
    )
    if error_match:
        extracted = error_match.group(1).strip()
        # Normalize to standard error type names
        for error_type in error_types:
            if error_type.lower() == extracted.lower():
                return error_type
        return extracted

    return None


def calculate_reward(
    model_output: str,
    ground_truth: Dict[str, Any]
) -> float:
    """
    Calculate the deterministic reward for a model prediction.

    The reward is computed based on:
    1. Structural reward (+0.1): If output contains <thinking> section
    2. Outcome reward (+1.0): If classification is correct
    3. Penalties:
       - False negative (-1.0): Error note predicted as clean
       - False positive (-1.5): Clean note predicted as error

    Args:
        model_output: Raw model output with <thinking> and <verdict>
        ground_truth: Dictionary containing:
            - has_error (bool): Whether note actually contains an error
            - error_type (str, optional): Specific error type if has_error

    Returns:
        The total reward as a float (sum of all components)

    Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6
    """
    # Initialize reward components
    structural_reward = 0.0
    outcome_reward = 0.0

    # 1. Structural reward: Check for <thinking> section
    # Requirements: 5.1
    if has_thinking_section(model_output):
        structural_reward = STRUCTURAL_REWARD

    # 2. Parse the verdict to determine prediction
    verdict = parse_verdict(model_output)
    predicted_error = _is_error_prediction(verdict) if verdict else False

    # Get ground truth
    actual_has_error = ground_truth.get("has_error", False)
    actual_error_type = ground_truth.get("error_type")

    # 3. Calculate outcome reward based on classification correctness
    if actual_has_error:
        # Ground truth: Note contains an error
        if predicted_error:
            # Predicted error - check if error type matches
            predicted_type = None
            if verdict:
                predicted_type = _extract_error_type_from_verdict(verdict)

            # Requirements: 5.2 - Award +1.0 if identifies specific error type
            if predicted_type and actual_error_type:
                if predicted_type.lower() == actual_error_type.lower():
                    outcome_reward = CORRECT_CLASSIFICATION_REWARD
                else:
                    # Detected error but wrong type - no reward, no penalty
                    outcome_reward = 0.0
            else:
                # Detected error but no type specified - credit for detection
                outcome_reward = CORRECT_CLASSIFICATION_REWARD
        else:
            # Requirements: 5.3 - Penalize -1.0 if prediction states "No Error"
            outcome_reward = FALSE_NEGATIVE_PENALTY
    else:
        # Ground truth: Note is clean (no error)
        if not predicted_error:
            # Requirements: 5.4 - Award +1.0 if states "No Clinical Error"
            outcome_reward = CORRECT_CLASSIFICATION_REWARD
        else:
            # Requirements: 5.5 - Penalize -1.5 if incorrectly identifies error
            outcome_reward = FALSE_POSITIVE_PENALTY

    # Requirements: 5.6 - Return sum of structural, outcome, and penalty
    total_reward = structural_reward + outcome_reward

    logger.debug(
        f"Reward: structural={structural_reward}, "
        f"outcome={outcome_reward}, total={total_reward}"
    )

    return total_reward


def calculate_reward_with_metadata(
    model_output: str,
    ground_truth: Dict[str, Any]
) -> RewardMetadata:
    """
    Calculate reward with detailed metadata breakdown.

    Same as calculate_reward but returns a RewardMetadata object with
    full breakdown of reward components.

    Args:
        model_output: The raw model output string
        ground_truth: Dictionary with has_error and optional error_type

    Returns:
        RewardMetadata with detailed breakdown
    """
    # Initialize components
    structural_reward = 0.0
    outcome_reward = 0.0
    correct_classification = False
    false_positive = False
    false_negative = False

    # Structural reward
    if has_thinking_section(model_output):
        structural_reward = STRUCTURAL_REWARD

    # Parse prediction
    verdict = parse_verdict(model_output)
    predicted_error = _is_error_prediction(verdict) if verdict else False

    # Ground truth
    actual_has_error = ground_truth.get("has_error", False)
    actual_error_type = ground_truth.get("error_type")

    # Calculate outcome
    if actual_has_error:
        if predicted_error:
            predicted_type = None
            if verdict:
                predicted_type = _extract_error_type_from_verdict(verdict)
            if predicted_type and actual_error_type:
                if predicted_type.lower() == actual_error_type.lower():
                    outcome_reward = CORRECT_CLASSIFICATION_REWARD
                    correct_classification = True
            else:
                outcome_reward = CORRECT_CLASSIFICATION_REWARD
                correct_classification = True
        else:
            outcome_reward = FALSE_NEGATIVE_PENALTY
            false_negative = True
    else:
        if not predicted_error:
            outcome_reward = CORRECT_CLASSIFICATION_REWARD
            correct_classification = True
        else:
            outcome_reward = FALSE_POSITIVE_PENALTY
            false_positive = True

    total_reward = structural_reward + outcome_reward

    return RewardMetadata(
        structural_reward=structural_reward,
        outcome_reward=outcome_reward,
        total_reward=total_reward,
        correct_classification=correct_classification,
        false_positive=false_positive,
        false_negative=false_negative
    )
