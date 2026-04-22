"""Unit tests for shared game reward logic."""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.self_play.game_reward import (
    REWARD_EXACT,
    REWARD_MISS,
    REWARD_PARTIAL,
    FORMAT_BONUS,
    compute_assessor_game_reward,
    compute_injector_game_reward,
    derive_assessor_ground_truth,
)


class GameRewardLogicTest(unittest.TestCase):
    def test_ground_truth_from_judge(self):
        self.assertEqual(derive_assessor_ground_truth("SAME", 4), "CORRECT")
        self.assertEqual(derive_assessor_ground_truth("CHANGED", 4), "4")
        self.assertIsNone(derive_assessor_ground_truth("ABSTAIN", 4))

    def test_injector_reward_benign(self):
        reward, outcome, valid = compute_injector_game_reward("benign", "SAME", parse_success=True)
        self.assertEqual(reward, REWARD_EXACT + FORMAT_BONUS)
        self.assertEqual(outcome, "exact_match")
        self.assertTrue(valid)

    def test_injector_reward_wrong_edit_type(self):
        reward, outcome, valid = compute_injector_game_reward("benign", "CHANGED", parse_success=True)
        self.assertEqual(reward, REWARD_MISS + FORMAT_BONUS)
        self.assertEqual(outcome, "wrong_edit_type")
        self.assertTrue(valid)

    def test_injector_reward_parse_failure(self):
        reward, outcome, valid = compute_injector_game_reward("error_injection", "CHANGED", parse_success=False)
        self.assertEqual(reward, REWARD_MISS)
        self.assertEqual(outcome, "parse_failure")
        self.assertFalse(valid)

    def test_assessor_same_correct(self):
        result = compute_assessor_game_reward("CORRECT", judge_verdict="SAME", changed_sid=5)
        self.assertEqual(result.reward, REWARD_EXACT + FORMAT_BONUS)
        self.assertEqual(result.outcome, "exact_match")
        self.assertEqual(result.ground_truth, "CORRECT")

    def test_assessor_same_false_positive(self):
        result = compute_assessor_game_reward("5", judge_verdict="SAME", changed_sid=5)
        self.assertEqual(result.reward, REWARD_MISS + FORMAT_BONUS)
        self.assertEqual(result.outcome, "miss")

    def test_assessor_changed_exact(self):
        result = compute_assessor_game_reward("5", judge_verdict="CHANGED", changed_sid=5)
        self.assertEqual(result.reward, REWARD_EXACT + FORMAT_BONUS)
        self.assertEqual(result.outcome, "exact_match")

    def test_assessor_changed_partial(self):
        result = compute_assessor_game_reward("3", judge_verdict="CHANGED", changed_sid=5)
        self.assertEqual(result.reward, REWARD_PARTIAL + FORMAT_BONUS)
        self.assertEqual(result.outcome, "partial_match")

    def test_assessor_changed_miss(self):
        result = compute_assessor_game_reward("CORRECT", judge_verdict="CHANGED", changed_sid=5)
        self.assertEqual(result.reward, REWARD_MISS + FORMAT_BONUS)
        self.assertEqual(result.outcome, "miss")


if __name__ == "__main__":
    unittest.main()
