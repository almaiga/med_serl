"""Test script for Medical Game Tool."""

import sys
sys.path.insert(0, '/Users/josmaiga/Documents/GitHub/med_serl')

from scripts.self_play.tools.medical_game_tool import MedicalGameTool, GameMode

# Sample data from MEDEC
sample_note_data = {
    "note_id": "ms-train-1591",
    "correct_note": "A 26-year-old immigrant from Mexico presents to your clinic for a physical. He tells you that several weeks ago, he noticed a lesion on his penis which went away after several weeks. It was nontender and did not bother him. He currently does not have any complaints. His temperature is 97.9 F (36.6 C), blood pressure is 139/91 mmHg, pulse is 87/min, respirations are 14/min, and oxygen saturation is 98% on room air. Physical exam is unremarkable and shows no evidence of any rash. A VDRL and FTA-ABS test are both positive. Penicillin is prescribed.",
    "incorrect_note": "A 26-year-old immigrant from Mexico presents to your clinic for a physical. He tells you that several weeks ago, he noticed a lesion on his penis which went away after several weeks. It was nontender and did not bother him. He currently does not have any complaints. His temperature is 97.9 F (36.6 C), blood pressure is 139/91 mmHg, pulse is 87/min, respirations are 14/min, and oxygen saturation is 98% on room air. Physical exam is unremarkable and shows no evidence of any rash. A VDRL and FTA-ABS test are both positive. Azithromycin and ceftriaxone are prescribed.",
    "error_type": "management",
    "error_sentence": "Azithromycin and ceftriaxone are prescribed.",
    "corrected_sentence": "Penicillin is prescribed."
}

# Initialize tool
config = {
    "benign_ratio": 0.5,
    "injection_prompts_path": "/Users/josmaiga/Documents/GitHub/med_serl/configs/prompts/error_injection_prompts_v2.json",
    "detection_prompts_path": "/Users/josmaiga/Documents/GitHub/med_serl/configs/prompts/error_detection_prompts.json",
}

tool = MedicalGameTool(config)

print("=== Testing Game Tool ===\n")

# Test game initialization
session_id = "test_session_1"
result = tool(session_id=session_id, action="", turn=0, extra_info=sample_note_data)

print(f"Turn 0 (Init):")
print(f"  Mode: {tool.game_states[session_id].mode.value}")
print(f"  Ground truth: {tool.game_states[session_id].ground_truth}")
print(f"  Done: {result['done']}")
print(f"  System prompt (first 100 chars): {result['observation']['system'][:100]}...")
print(f"  User prompt (first 200 chars): {result['observation']['user'][:200]}...")

# Simulate Injector response
mock_injector_response = """<think>I need to make a surface edit to preserve meaning</think>

generated_note:
A 26-year-old immigrant from Mexico comes to your clinic for a physical. He tells you that several weeks ago, he noticed a lesion on his penis which went away after several weeks. It was nontender and did not bother him. He currently does not have any complaints. His temperature is 97.9 F (36.6 C), blood pressure is 139/91 mmHg, pulse is 87/min, respirations are 14/min, and oxygen saturation is 98% on room air. Physical exam is unremarkable and shows no evidence of any rash. A VDRL and FTA-ABS test are both positive. Penicillin is prescribed.

final_answer: "CORRECT"

changes_made:
{"original_sentence": "presents to your clinic", "modified_sentence": "comes to your clinic", "words_changed": "presents → comes"}"""

result2 = tool(session_id=session_id, action=mock_injector_response, turn=1, extra_info=None)

print(f"\nTurn 1 (After Injector):")
print(f"  Done: {result2['done']}")
print(f"  Generated note extracted: {tool.game_states[session_id].generated_note[:100] if tool.game_states[session_id].generated_note else 'None'}...")
print(f"  Assessor system prompt: {result2['observation']['system'][:100]}...")
print(f"  Assessor user prompt: {result2['observation']['user'][:100]}...")

# Simulate Assessor response
mock_assessor_response = """<think>Checking for clinical errors...</think>

final_answer: "CORRECT"
Explanation: The clinical note contains appropriate diagnosis and treatment for syphilis."""

result3 = tool(session_id=session_id, action=mock_assessor_response, turn=2, extra_info=None)

print(f"\nTurn 2 (After Assessor):")
print(f"  Done: {result3['done']}")
print(f"  Game result keys: {list(result3['game_result'].keys())}")
print(f"  Mode: {result3['game_result']['mode']}")
print(f"  Ground truth: {result3['game_result']['ground_truth']}")

print("\n=== Game Tool Test Complete! ===")
