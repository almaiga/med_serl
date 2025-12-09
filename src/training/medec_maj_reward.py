"""
MEDEC Majority Voting Reward Function for OpenRLHF.

Uses majority voting across multiple samples when ground truth is unavailable
(e.g., for self-generated instructions in SeRL's self-instruction mode).

Interface: reward_func(queries, prompts, labels) -> torch.Tensor
"""

import re
import torch
from typing import Optional


def parse_answer(response: str) -> Optional[str]:
    """Extract CORRECT/INCORRECT from response."""
    if not response:
        return None
    
    if isinstance(response, list) and len(response) > 0:
        response = response[0]
    response = str(response)
    
    # Try <answer> tags
    match = re.search(r'<answer>\s*(CORRECT|INCORRECT)\s*</answer>', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Try Answer: format
    match = re.search(r'Answer:\s*(CORRECT|INCORRECT)', response, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback
    response_lower = response.lower()
    
    no_error = ['no error', 'no errors', 'appears correct', 'is correct']
    for indicator in no_error:
        if indicator in response_lower:
            return 'CORRECT'
    
    has_error = ['error detected', 'error found', 'contains error', 'has error']
    for indicator in has_error:
        if indicator in response_lower:
            return 'INCORRECT'
    
    return None


def reward_func(queries: list[str], prompts: list[str], labels: list[str]) -> torch.Tensor:
    """
    Majority voting reward for MEDEC.
    
    Compares all responses to find majority answer, rewards agreement.
    Labels parameter is ignored - uses majority voting instead.
    
    Args:
        queries: Model responses (n_samples_per_prompt)
        prompts: Input prompts (unused)
        labels: Ground truth (unused in majority voting)
    
    Returns:
        torch.Tensor (1.0 for majority agreement, 0.0 otherwise)
    """
    n = len(queries)
    
    if n == 0:
        return torch.tensor([])
    if n == 1:
        return torch.tensor([0.5])
    
    # Extract answers
    extracted = [parse_answer(q) for q in queries]
    
    # Build equality matrix
    equal_matrix = [[False] * n for _ in range(n)]
    for i in range(n):
        for j in range(i, n):
            if i == j:
                equal_matrix[i][j] = True
            else:
                is_equal = extracted[i] and extracted[j] and extracted[i] == extracted[j]
                equal_matrix[i][j] = is_equal
                equal_matrix[j][i] = is_equal
    
    # Count agreements
    maj_count = [sum(equal_matrix[i]) for i in range(n)]
    majority_idx = maj_count.index(max(maj_count))
    
    # Assign rewards
    reward_list = [1.0 if equal_matrix[majority_idx][i] else 0.0 for i in range(n)]
    
    return torch.tensor(reward_list)


if __name__ == '__main__':
    queries = [
        "<answer>CORRECT</answer>",
        "<answer>CORRECT</answer>",
        "<answer>INCORRECT</answer>",
        "<answer>CORRECT</answer>",
    ]
    print(f"Rewards: {reward_func(queries, [], [])}")  # [1.0, 1.0, 0.0, 1.0]
