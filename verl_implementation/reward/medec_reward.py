#!/usr/bin/env python3
"""
Custom Reward Function for Medical Error Detection

This module provides reward computation for verl REINFORCE++/GRPO training
on the MEDEC medical error detection task.

The reward function evaluates:
1. Error detection accuracy (binary: error vs correct)
2. Error localization (sentence-level match)
3. Error correction quality (text similarity)
4. Error type classification

Usage with verl:
    Set in config:
        custom_reward_function.path: reward/medec_reward.py
        custom_reward_function.name: compute_score
"""

import re
import json
from typing import Optional, Any
from difflib import SequenceMatcher


def extract_json_from_response(response: str) -> Optional[dict]:
    """
    Extract JSON object from model response.
    Handles markdown code blocks and partial JSON.
    """
    # Try to find JSON in code blocks
    json_patterns = [
        r'```json\s*(.*?)\s*```',
        r'```\s*(.*?)\s*```',
        r'\{[^{}]*\}',
    ]
    
    for pattern in json_patterns:
        matches = re.findall(pattern, response, re.DOTALL)
        for match in matches:
            try:
                # Clean up the match
                text = match.strip()
                if not text.startswith('{'):
                    # Find the JSON object
                    start = text.find('{')
                    end = text.rfind('}') + 1
                    if start >= 0 and end > start:
                        text = text[start:end]
                
                return json.loads(text)
            except json.JSONDecodeError:
                continue
    
    # Try direct parsing
    try:
        return json.loads(response.strip())
    except json.JSONDecodeError:
        pass
    
    return None


def compute_text_similarity(text1: str, text2: str) -> float:
    """Compute similarity between two texts using SequenceMatcher."""
    if not text1 or not text2:
        return 0.0
    
    # Normalize texts
    text1 = text1.lower().strip()
    text2 = text2.lower().strip()
    
    return SequenceMatcher(None, text1, text2).ratio()


def normalize_error_type(error_type: str) -> str:
    """Normalize error type strings for comparison."""
    if not error_type:
        return ""
    
    error_type = error_type.lower().strip()
    
    # Map common variations
    type_map = {
        "dx": "diagnosis",
        "diag": "diagnosis",
        "diagnostic": "diagnosis",
        "mgmt": "management",
        "manage": "management",
        "tx": "treatment",
        "therapy": "treatment",
        "therapeutic": "treatment",
        "pharma": "pharmacotherapy",
        "medication": "pharmacotherapy",
        "drug": "pharmacotherapy",
        "med": "pharmacotherapy",
        "causal": "causalorganism",
        "organism": "causalorganism",
        "pathogen": "causalorganism",
        "infection": "causalorganism",
    }
    
    for key, value in type_map.items():
        if key in error_type:
            return value
    
    return error_type


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: dict,
    extra_info: Optional[dict] = None
) -> float:
    """
    Compute reward score for medical error detection task.
    
    This function is called by verl's reward computation pipeline.
    
    Args:
        data_source: Source dataset identifier (e.g., "medec_ms", "medec_uw")
        solution_str: Model's response string
        ground_truth: Dictionary containing:
            - has_error: bool
            - error_sentence: str (if has_error)
            - corrected_sentence: str (if has_error)
            - error_type: str (if has_error)
            - corrected_text: str (optional, full corrected note)
        extra_info: Optional metadata (note_id, split, etc.)
    
    Returns:
        float: Reward score in range [-1.0, 1.0]
            - 1.0: Perfect detection, localization, and correction
            - 0.7: Correct detection and localization
            - 0.4: Correct detection only
            - 0.0: Partial/unclear response
            - -0.5: Wrong detection (FP or FN)
            - -1.0: Completely wrong or invalid response
    """
    # Extract JSON from model response
    parsed = extract_json_from_response(solution_str)
    
    if parsed is None:
        # Could not parse response - small negative reward
        return -0.2
    
    # Get ground truth values
    gt_has_error = ground_truth.get('has_error', False)
    gt_error_sentence = ground_truth.get('error_sentence', '')
    gt_corrected_sentence = ground_truth.get('corrected_sentence', '')
    gt_error_type = normalize_error_type(ground_truth.get('error_type', ''))
    
    # Extract model predictions
    assessment = str(parsed.get('assessment', '')).upper()
    pred_has_error = assessment == 'ERROR' or 'ERROR' in assessment
    pred_error_sentence = str(parsed.get('error_sentence', '') or '')
    pred_corrected_sentence = str(parsed.get('corrected_sentence', '') or '')
    pred_error_type = normalize_error_type(str(parsed.get('error_type', '') or ''))
    
    # Initialize reward components
    reward = 0.0
    
    # Component 1: Detection accuracy (most important)
    if pred_has_error == gt_has_error:
        reward += 0.4  # Correct detection
    else:
        # Wrong detection is heavily penalized
        if gt_has_error and not pred_has_error:
            # False negative - missed an error (dangerous in medical context)
            return -0.7
        else:
            # False positive - hallucinated an error
            return -0.5
    
    # If correctly identified no error, we're done
    if not gt_has_error and not pred_has_error:
        return 0.6  # Good job identifying correct note
    
    # Component 2: Error localization
    if gt_error_sentence and pred_error_sentence:
        localization_sim = compute_text_similarity(
            gt_error_sentence, 
            pred_error_sentence
        )
        
        if localization_sim > 0.8:
            reward += 0.3  # Good localization
        elif localization_sim > 0.5:
            reward += 0.2  # Partial localization
        elif localization_sim > 0.3:
            reward += 0.1  # Weak localization
    
    # Component 3: Error correction
    if gt_corrected_sentence and pred_corrected_sentence:
        correction_sim = compute_text_similarity(
            gt_corrected_sentence,
            pred_corrected_sentence
        )
        
        if correction_sim > 0.8:
            reward += 0.2  # Good correction
        elif correction_sim > 0.5:
            reward += 0.1  # Partial correction
    
    # Component 4: Error type classification (bonus)
    if gt_error_type and pred_error_type:
        if gt_error_type == pred_error_type:
            reward += 0.1  # Correct error type
    
    # Ensure reward is in valid range
    return min(1.0, max(-1.0, reward))


def compute_score_batch(
    data_sources: list[str],
    solution_strs: list[str],
    ground_truths: list[dict],
    extra_infos: Optional[list[dict]] = None
) -> list[float]:
    """
    Batch version of compute_score for efficiency.
    
    Args:
        data_sources: List of data source identifiers
        solution_strs: List of model responses
        ground_truths: List of ground truth dictionaries
        extra_infos: Optional list of metadata dictionaries
    
    Returns:
        list[float]: List of reward scores
    """
    if extra_infos is None:
        extra_infos = [None] * len(data_sources)
    
    rewards = []
    for ds, sol, gt, info in zip(data_sources, solution_strs, ground_truths, extra_infos):
        reward = compute_score(ds, sol, gt, info)
        rewards.append(reward)
    
    return rewards


# Aliases for verl compatibility
reward_fn = compute_score
batch_reward_fn = compute_score_batch


# ============================================================================
# Self-Play Reward Functions (for Phase 2)
# ============================================================================

def compute_injector_reward(
    injector_action: str,  # "error" or "benign"
    injected_note: str,
    assessor_prediction: dict,
    mnv_passed: bool,
    ground_truth_action: Optional[str] = None
) -> float:
    """
    Compute reward for the Injector in self-play.
    
    The Injector is rewarded for:
    - Creating notes that pass MNV filters (+0.1)
    - Fooling the Assessor (+0.5)
    - Penalized when Assessor correctly identifies (+/-0.3)
    
    Args:
        injector_action: What the injector did ("error" or "benign")
        injected_note: The modified clinical note
        assessor_prediction: Assessor's detection result
        mnv_passed: Whether the note passed Medical Note Verifiers
        ground_truth_action: Optional ground truth for curriculum
    
    Returns:
        float: Injector reward
    """
    reward = 0.0
    
    # MNV filter bonus
    if mnv_passed:
        reward += 0.1
    else:
        # Failed MNV means bad injection - penalize
        return -0.3
    
    # Get assessor's prediction
    assessor_says_error = assessor_prediction.get('has_error', False)
    
    # Adversarial game outcome
    if injector_action == "error":
        # Injector created an error
        if not assessor_says_error:
            # Assessor missed it - Injector wins!
            reward += 0.5
        else:
            # Assessor caught it - Injector loses
            reward -= 0.2
    else:
        # Injector made benign change
        if assessor_says_error:
            # Assessor false positive - Injector wins!
            reward += 0.4
        else:
            # Assessor correctly identified as benign - push
            reward += 0.1
    
    return reward


def compute_assessor_reward(
    assessor_prediction: dict,
    ground_truth: dict,
    injector_action: str
) -> float:
    """
    Compute reward for the Assessor in self-play.
    
    The Assessor is rewarded for:
    - Correct detection (+0.4)
    - Correct localization (+0.3)
    - Correct correction (+0.2)
    - Penalized for FP/FN (-0.5/-0.7)
    
    Args:
        assessor_prediction: Assessor's full prediction dict
        ground_truth: Ground truth error information
        injector_action: What the injector did ("error" or "benign")
    
    Returns:
        float: Assessor reward
    """
    # For self-play, we use the injector action as ground truth
    gt_has_error = (injector_action == "error")
    
    modified_ground_truth = {
        **ground_truth,
        'has_error': gt_has_error
    }
    
    # Use the standard reward function
    solution_str = json.dumps(assessor_prediction)
    return compute_score("self_play", solution_str, modified_ground_truth)


if __name__ == "__main__":
    # Test the reward function
    print("Testing reward function...")
    
    # Test case 1: Correct detection of error
    gt1 = {
        "has_error": True,
        "error_sentence": "The patient was prescribed aspirin for bacterial infection.",
        "corrected_sentence": "The patient was prescribed amoxicillin for bacterial infection.",
        "error_type": "pharmacotherapy"
    }
    
    response1 = """```json
{
    "assessment": "ERROR",
    "reasoning": "The prescription of aspirin for a bacterial infection is incorrect.",
    "error_sentence": "The patient was prescribed aspirin for bacterial infection.",
    "corrected_sentence": "The patient was prescribed amoxicillin for bacterial infection.",
    "error_type": "pharmacotherapy"
}
```"""
    
    score1 = compute_score("test", response1, gt1)
    print(f"Test 1 (perfect detection): {score1}")  # Should be ~1.0
    
    # Test case 2: Missed error (false negative)
    response2 = """```json
{
    "assessment": "CORRECT",
    "reasoning": "The note appears correct.",
    "error_sentence": null,
    "corrected_sentence": null,
    "error_type": null
}
```"""
    
    score2 = compute_score("test", response2, gt1)
    print(f"Test 2 (missed error): {score2}")  # Should be -0.7
    
    # Test case 3: Correctly identified no error
    gt3 = {
        "has_error": False,
        "error_sentence": None,
        "corrected_sentence": None,
        "error_type": None
    }
    
    response3 = """```json
{
    "assessment": "CORRECT",
    "reasoning": "The clinical note is accurate.",
    "error_sentence": null,
    "corrected_sentence": null,
    "error_type": null
}
```"""
    
    score3 = compute_score("test", response3, gt3)
    print(f"Test 3 (correct = no error): {score3}")  # Should be ~0.6
    
    # Test case 4: False positive
    response4 = """```json
{
    "assessment": "ERROR",
    "reasoning": "I think there's an error.",
    "error_sentence": "Some sentence",
    "corrected_sentence": "Fixed sentence",
    "error_type": "diagnosis"
}
```"""
    
    score4 = compute_score("test", response4, gt3)
    print(f"Test 4 (false positive): {score4}")  # Should be -0.5
    
    print("\nAll tests completed!")
