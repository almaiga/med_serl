"""Test script for reward function."""

import sys
sys.path.insert(0, '/Users/josmaiga/Documents/GitHub/med_serl')

from scripts.self_play.rewards.zero_sum_reward import compute_game_rewards, parse_final_answer

print("=== Testing Reward Function ===\n")

# Test case 1: Benign mode, Assessor correct
print("Test 1: Benign mode, Assessor says CORRECT (correct)")
injector_r, assessor_r = compute_game_rewards(
    mode="benign",
    ground_truth="CORRECT",
    injector_output='final_answer: "CORRECT"',
    assessor_output='final_answer: "CORRECT"\nExplanation: No errors found.'
)
print(f"  Injector reward: {injector_r}")
print(f"  Assessor reward: {assessor_r}")
print(f"  Sum (should be ~0): {injector_r + assessor_r}")

# Test case 2: Benign mode, Assessor wrong
print("\nTest 2: Benign mode, Assessor says INCORRECT (wrong - Injector wins)")
injector_r, assessor_r = compute_game_rewards(
    mode="benign",
    ground_truth="CORRECT",
    injector_output='final_answer: "CORRECT"',
    assessor_output='final_answer: "INCORRECT"\nExplanation: Found an error.'
)
print(f"  Injector reward: {injector_r}")
print(f"  Assessor reward: {assessor_r}")
print(f"  Sum (should be ~0): {injector_r + assessor_r}")

# Test case 3: Error mode, Assessor correct
print("\nTest 3: Error mode, Assessor says INCORRECT (correct)")
injector_r, assessor_r = compute_game_rewards(
    mode="error_injection",
    ground_truth="INCORRECT",
    injector_output='final_answer: "INCORRECT"',
    assessor_output='final_answer: "INCORRECT"\nExplanation: Found management error.'
)
print(f"  Injector reward: {injector_r}")
print(f"  Assessor reward: {assessor_r}")
print(f"  Sum (should be ~0): {injector_r + assessor_r}")

# Test case 4: Error mode, Assessor wrong (Injector wins)
print("\nTest 4: Error mode, Assessor says CORRECT (wrong - Injector wins)")
injector_r, assessor_r = compute_game_rewards(
    mode="error_injection",
    ground_truth="INCORRECT",
    injector_output='final_answer: "INCORRECT"',
    assessor_output='final_answer: "CORRECT"\nExplanation: No errors found.'
)
print(f"  Injector reward: {injector_r}")
print(f"  Assessor reward: {assessor_r}")
print(f"  Sum (should be ~0): {injector_r + assessor_r}")

# Test case 5: Invalid Assessor output
print("\nTest 5: Invalid Assessor output (Injector wins by default)")
injector_r, assessor_r = compute_game_rewards(
    mode="benign",
    ground_truth="CORRECT",
    injector_output='final_answer: "CORRECT"',
    assessor_output='I am not sure about this note.'
)
print(f"  Injector reward: {injector_r}")
print(f"  Assessor reward: {assessor_r}")

print("\n=== Reward Function Test Complete! ===")
