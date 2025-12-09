"""
MEDEC Rule-Based Reward Function for OpenRLHF.

Evaluates model responses for medical error detection using deterministic rules.

Interface: reward_func(queries, prompts, labels) -> torch.Tensor
"""

import re
import torch
from typing import Optional

STRUCTURAL_REWARD = 0.1
CORRECT_CLASSIFICATION_REWARD = 1.0
FALSE_NEGATIVE_PENALTY = -1.0
FALSE_POSITIVE_PENALTY = -1.5


def has_thinking_section(output: str) -> bool:
    """Check if output contains <think> section."""
    if not output:
        return False
    pattern = r'<think(?:ing)?>.*?</think(?:ing)?>'
    return bool(re.search(pattern, output, re.DOTALL | re.IGNORECASE))


def parse_answer(output: str) -> Optional[str]:
    """Extract CORRECT/INCORRECT from model output."""
    if not output:
        return None
    
    # Try <answer> tags
    match = re.search(r'<answer>\s*(CORRECT|INCORRECT)\s*</answer>', output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Try Answer: format
    match = re.search(r'Answer:\s*(CORRECT|INCORRECT)', output, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback patterns
    output_lower = output.lower()
    
    no_error_patterns = [
        r'no\s+(?:obvious\s+)?errors?\s+(?:detected|found)',
        r'appears?\s+(?:to\s+be\s+)?correct',
    ]
    for pattern in no_error_patterns:
        if re.search(pattern, output_lower):
            return 'CORRECT'
    
    error_patterns = [
        r'error\s+(?:detected|found)',
        r'contains?\s+(?:a\s+)?error',
    ]
    for pattern in error_patterns:
        if re.search(pattern, output_lower):
            return 'INCORRECT'
    
    return None


def parse_label(label: str) -> bool:
    """Parse ground truth label. Returns True if note has error."""
    if not label:
        return False
    label_upper = label.upper().strip()
    if label_upper == "INCORRECT" or label_upper.startswith("ERROR"):
        return True
    return False


def reward_func(queries: list[str], prompts: list[str], labels: list[str]) -> torch.Tensor:
    """
    Compute rewards for MEDEC task.
    
    Args:
        queries: Model responses
        prompts: Input prompts (unused)
        labels: Ground truth ("CORRECT" or "INCORRECT")
    
    Returns:
        torch.Tensor of rewards
    """
    reward_list = []
    
    for query, label in zip(queries, labels):
        reward = 0.0
        
        if has_thinking_section(query):
            reward += STRUCTURAL_REWARD
        
        predicted_answer = parse_answer(query)
        actual_has_error = parse_label(label)
        
        if predicted_answer is None:
            reward += -0.5
        else:
            predicted_error = (predicted_answer == "INCORRECT")
            
            if actual_has_error:
                if predicted_error:
                    reward += CORRECT_CLASSIFICATION_REWARD
                else:
                    reward += FALSE_NEGATIVE_PENALTY
            else:
                if not predicted_error:
                    reward += CORRECT_CLASSIFICATION_REWARD
                else:
                    reward += FALSE_POSITIVE_PENALTY
        
        reward_list.append(reward)
    
    return torch.tensor(reward_list)


if __name__ == '__main__':
    queries = [
        "<think>Checking...</think>\n<answer>CORRECT</answer>",
        "<think>Found issue</think>\n<answer>INCORRECT</answer>",
    ]
    labels = ["CORRECT", "INCORRECT"]
    print(f"Rewards: {reward_func(queries, [], labels)}")
