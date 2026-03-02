"""Test script for reward function (3-tier sentence-level localization)."""

import sys
sys.path.insert(0, '/Users/josmaiga/Documents/GitHub/med_serl')

from scripts.self_play.rewards.zero_sum_reward import compute_game_rewards, parse_final_answer

print("=== Testing Reward Function (3-Tier Localization) ===\n")

# Test case 1: Benign mode, Assessor says CORRECT → exact match
print("Test 1: Benign mode, Assessor says CORRECT (exact match)")
injector_r, assessor_r = compute_game_rewards(
    mode="benign",
    ground_truth="CORRECT",
    injector_output='1. A patient presents to the clinic.',
    assessor_output='CORRECT'
)
print(f"  Injector reward: {injector_r:.2f}")
print(f"  Assessor reward: {assessor_r:.2f}")
assert assessor_r > 0, "Assessor should win"
print("  PASS")

# Test case 2: Benign mode, Assessor flags sentence (false positive)
print("\nTest 2: Benign mode, Assessor says '3' (false positive - Injector wins)")
injector_r, assessor_r = compute_game_rewards(
    mode="benign",
    ground_truth="CORRECT",
    injector_output='1. A patient comes to the clinic.',
    assessor_output='3'
)
print(f"  Injector reward: {injector_r:.2f}")
print(f"  Assessor reward: {assessor_r:.2f}")
assert assessor_r < 0, "Assessor should lose (false positive)"
assert injector_r > 0, "Injector should win"
print("  PASS")

# Test case 3: Error mode, Assessor exact sentence match
print("\nTest 3: Error mode, Assessor says '7' (exact match, gt=7)")
injector_r, assessor_r = compute_game_rewards(
    mode="error_injection",
    ground_truth="7",
    injector_output='7. Azithromycin and ceftriaxone are prescribed.',
    assessor_output='7'
)
print(f"  Injector reward: {injector_r:.2f}")
print(f"  Assessor reward: {assessor_r:.2f}")
assert assessor_r > 0, "Assessor should win (exact match)"
print("  PASS")

# Test case 4: Error mode, Assessor wrong sentence (partial credit)
print("\nTest 4: Error mode, Assessor says '5' but error at '7' (partial)")
injector_r, assessor_r = compute_game_rewards(
    mode="error_injection",
    ground_truth="7",
    injector_output='7. Azithromycin and ceftriaxone are prescribed.',
    assessor_output='5'
)
print(f"  Injector reward: {injector_r:.2f}")
print(f"  Assessor reward: {assessor_r:.2f}")
assert 0 < assessor_r < 1.0, "Assessor should get partial credit"
print("  PASS")

# Test case 5: Error mode, Assessor says CORRECT (missed error)
print("\nTest 5: Error mode, Assessor says CORRECT (missed - Injector wins)")
injector_r, assessor_r = compute_game_rewards(
    mode="error_injection",
    ground_truth="7",
    injector_output='7. Azithromycin and ceftriaxone are prescribed.',
    assessor_output='CORRECT'
)
print(f"  Injector reward: {injector_r:.2f}")
print(f"  Assessor reward: {assessor_r:.2f}")
assert assessor_r < 0, "Assessor should lose (missed error)"
assert injector_r > 0, "Injector should win"
print("  PASS")

# Test case 6: Invalid Assessor output
print("\nTest 6: Invalid Assessor output (Injector wins by default)")
injector_r, assessor_r = compute_game_rewards(
    mode="benign",
    ground_truth="CORRECT",
    injector_output='1. A patient presents to the clinic.',
    assessor_output='I am not sure about this note.'
)
print(f"  Injector reward: {injector_r:.2f}")
print(f"  Assessor reward: {assessor_r:.2f}")
assert assessor_r < 0, "Assessor should lose (invalid format)"
print("  PASS")

# Test parse_final_answer
print("\n=== Parse Tests ===")
label, sid = parse_final_answer("CORRECT")
print(f'  "CORRECT" → label={label}, sid={sid}')
assert label == "CORRECT" and sid is None

label, sid = parse_final_answer("7")
print(f'  "7" → label={label}, sid={sid}')
assert label == "ERROR" and sid == 7

label, sid = parse_final_answer("")
print(f'  "" → label={label}, sid={sid}')
assert label == "UNKNOWN" and sid is None

print("\n=== Reward Function Test Complete! ===")
