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

    def test_injector_reward_judge_unavailable_is_neutral(self):
        reward, outcome, valid = compute_injector_game_reward(
            "benign",
            "ABSTAIN",
            parse_success=True,
            judge_status="request_failed",
        )
        self.assertEqual(reward, 0.0)
        self.assertEqual(outcome, "judge_unavailable")
        self.assertFalse(valid)

    def test_injector_reward_semantic_abstain_penalizes_injector(self):
        reward, outcome, valid = compute_injector_game_reward(
            "benign",
            "ABSTAIN",
            parse_success=True,
            judge_status="semantic_abstain",
        )
        self.assertEqual(reward, REWARD_MISS + FORMAT_BONUS)
        self.assertEqual(outcome, "judge_semantic_abstain")
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

    def test_assessor_override_for_injector_parse_failure(self):
        result = compute_assessor_game_reward(
            "CORRECT",
            judge_verdict="ABSTAIN",
            changed_sid=None,
            ground_truth_override="CORRECT",
        )
        self.assertEqual(result.reward, REWARD_EXACT + FORMAT_BONUS)
        self.assertEqual(result.outcome, "exact_match")
        self.assertEqual(result.ground_truth, "CORRECT")

    def test_assessor_neutral_when_ground_truth_unknown(self):
        result = compute_assessor_game_reward(
            "CORRECT",
            judge_verdict="ABSTAIN",
            changed_sid=None,
        )
        self.assertEqual(result.reward, 0.0)
        self.assertEqual(result.outcome, "game_invalid")

    def test_retuned_constants(self):
        # Locked-in retuned values (conservatism fix). Update intentionally if changed.
        self.assertEqual(REWARD_EXACT, 1.0)
        self.assertEqual(REWARD_PARTIAL, 0.5)
        self.assertEqual(REWARD_MISS, -1.5)
        self.assertEqual(FORMAT_BONUS, 0.2)

    def test_always_correct_ev_nonpositive_at_50_50(self):
        # The whole point of the retune: "always say CORRECT" must NOT be a safe
        # positive-EV strategy at a 50/50 benign/error mix (it was +0.2 before).
        benign_correct = compute_assessor_game_reward(
            "CORRECT", judge_verdict="SAME", changed_sid=5
        ).reward
        error_miss = compute_assessor_game_reward(
            "CORRECT", judge_verdict="CHANGED", changed_sid=5
        ).reward
        ev_always_correct = 0.5 * benign_correct + 0.5 * error_miss
        self.assertLessEqual(ev_always_correct, 0.0)
        # And attempting detection (even partial) must beat missing.
        partial = compute_assessor_game_reward(
            "3", judge_verdict="CHANGED", changed_sid=5
        ).reward
        self.assertGreater(partial, error_miss)


if __name__ == "__main__":
    unittest.main()
