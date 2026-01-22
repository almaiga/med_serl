"""Reward functions for self-play training."""

from .zero_sum_reward import reward_func, compute_score

__all__ = ["reward_func", "compute_score"]
