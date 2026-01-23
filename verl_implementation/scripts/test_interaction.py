#!/usr/bin/env python3
"""
Test script for MedicalGameInteraction

Validates the two-phase game flow works correctly:
1. Injector generates modified note
2. Assessor classifies the note
3. Reward is computed correctly
"""

import asyncio
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.self_play.interactions.medical_game_interaction import MedicalGameInteraction


async def test_benign_correct():
    """Test benign mode with correct Assessor classification."""
    print("\n" + "="*80)
    print("TEST 1: Benign mode - Assessor correctly identifies as CORRECT")
    print("="*80)
    
    config = {
        "name": "medical_game",
        "detection_prompts_path": "configs/prompts/error_detection_prompts.json"
    }
    
    interaction = MedicalGameInteraction(config)
    
    # Start interaction
    instance_id = await interaction.start_interaction(
        ground_truth="CORRECT",
        mode="benign",
        note_data={"correct_note": "Test note"}
    )
    
    print(f"✓ Started interaction: {instance_id}")
    
    # Simulate Injector output (Phase 1)
    injector_messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User prompt"},
        {"role": "assistant", "content": """
<think>I'll change "started" to "initiated"</think>

generated_note:
The patient was initiated on antibiotics.

final_answer: "CORRECT"

changes_made:
{"original_sentence": "The patient was started on antibiotics.", "modified_sentence": "The patient was initiated on antibiotics.", "words_changed": "started → initiated"}
"""}
    ]
    
    should_terminate, response, score, metadata = await interaction.generate_response(
        instance_id, injector_messages
    )
    
    print(f"✓ Injector phase complete")
    print(f"  - Should terminate: {should_terminate}")
    print(f"  - Score: {score}")
    print(f"  - Assessor prompt preview: {response[:100]}...")
    
    assert not should_terminate, "Should not terminate after Injector phase"
    assert "generated_note" not in response.lower() or "classify" in response.lower(), "Should be Assessor prompt"
    
    # Simulate Assessor output (Phase 2) - CORRECT classification
    assessor_messages = injector_messages + [
        {"role": "user", "content": response},
        {"role": "assistant", "content": """
final_answer: "CORRECT"
Explanation: The note shows proper clinical documentation with no medical errors detected.
"""}
    ]
    
    should_terminate, feedback, score, metadata = await interaction.generate_response(
        instance_id, assessor_messages
    )
    
    print(f"✓ Assessor phase complete")
    print(f"  - Should terminate: {should_terminate}")
    print(f"  - Feedback: {feedback}")
    print(f"  - Score: {score}")
    print(f"  - Metadata: {metadata}")
    
    assert should_terminate, "Should terminate after Assessor phase"
    assert score > 0, f"Expected positive score for correct classification, got {score}"
    assert score >= 1.2, f"Expected 1.0 + 0.2 format bonus = 1.2, got {score}"
    
    await interaction.finalize_interaction(instance_id)
    
    print("✓ TEST 1 PASSED")


async def test_benign_incorrect():
    """Test benign mode with incorrect Assessor classification (Injector wins)."""
    print("\n" + "="*80)
    print("TEST 2: Benign mode - Assessor incorrectly identifies as INCORRECT")
    print("="*80)
    
    config = {
        "name": "medical_game",
        "detection_prompts_path": "configs/prompts/error_detection_prompts.json"
    }
    
    interaction = MedicalGameInteraction(config)
    
    instance_id = await interaction.start_interaction(
        ground_truth="CORRECT",
        mode="benign"
    )
    
    # Injector phase
    injector_messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "User"},
        {"role": "assistant", "content": """
generated_note:
Patient shows normal vital signs.

final_answer: "CORRECT"
"""}
    ]
    
    should_terminate, response, score, metadata = await interaction.generate_response(
        instance_id, injector_messages
    )
    
    # Assessor phase - WRONG classification
    assessor_messages = injector_messages + [
        {"role": "user", "content": response},
        {"role": "assistant", "content": """
final_answer: "INCORRECT"
Explanation: I detected an error in the vital signs documentation.
"""}
    ]
    
    should_terminate, feedback, score, metadata = await interaction.generate_response(
        instance_id, assessor_messages
    )
    
    print(f"✓ Assessor misclassified")
    print(f"  - Score: {score}")
    print(f"  - Feedback: {feedback}")
    
    assert score < 0, f"Expected negative score for wrong classification, got {score}"
    assert score >= -0.8, f"Expected -1.0 + 0.2 format bonus = -0.8, got {score}"
    
    await interaction.finalize_interaction(instance_id)
    
    print("✓ TEST 2 PASSED")


async def test_error_correct():
    """Test error mode with correct Assessor classification."""
    print("\n" + "="*80)
    print("TEST 3: Error mode - Assessor correctly identifies as INCORRECT")
    print("="*80)
    
    config = {
        "name": "medical_game",
        "detection_prompts_path": "configs/prompts/error_detection_prompts.json"
    }
    
    interaction = MedicalGameInteraction(config)
    
    instance_id = await interaction.start_interaction(
        ground_truth="INCORRECT",
        mode="error_injection"
    )
    
    # Injector phase
    injector_messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "User"},
        {"role": "assistant", "content": """
generated_note:
Patient diagnosed with pneumonia. Started on aspirin.

final_answer: "INCORRECT"
"""}
    ]
    
    should_terminate, response, score, metadata = await interaction.generate_response(
        instance_id, injector_messages
    )
    
    # Assessor phase - CORRECT classification
    assessor_messages = injector_messages + [
        {"role": "user", "content": response},
        {"role": "assistant", "content": """
final_answer: "INCORRECT"
Explanation: Aspirin is not appropriate treatment for pneumonia; antibiotics should be used.
"""}
    ]
    
    should_terminate, feedback, score, metadata = await interaction.generate_response(
        instance_id, assessor_messages
    )
    
    print(f"✓ Assessor correctly identified error")
    print(f"  - Score: {score}")
    print(f"  - Feedback: {feedback}")
    
    assert score > 0, f"Expected positive score for correct classification, got {score}"
    assert score >= 1.2, f"Expected 1.0 + 0.2 format bonus = 1.2, got {score}"
    
    await interaction.finalize_interaction(instance_id)
    
    print("✓ TEST 3 PASSED")


async def test_invalid_format():
    """Test handling of invalid format (no format bonus)."""
    print("\n" + "="*80)
    print("TEST 4: Invalid format - No format bonus")
    print("="*80)
    
    config = {
        "name": "medical_game",
        "detection_prompts_path": "configs/prompts/error_detection_prompts.json"
    }
    
    interaction = MedicalGameInteraction(config)
    
    instance_id = await interaction.start_interaction(
        ground_truth="CORRECT",
        mode="benign"
    )
    
    # Injector phase
    injector_messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "User"},
        {"role": "assistant", "content": "generated_note:\nNormal patient"}
    ]
    
    should_terminate, response, score, metadata = await interaction.generate_response(
        instance_id, injector_messages
    )
    
    # Assessor phase - Missing "Explanation:" field
    assessor_messages = injector_messages + [
        {"role": "user", "content": response},
        {"role": "assistant", "content": 'final_answer: "CORRECT"'}  # No explanation
    ]
    
    should_terminate, feedback, score, metadata = await interaction.generate_response(
        instance_id, assessor_messages
    )
    
    print(f"✓ Invalid format handled")
    print(f"  - Score: {score}")
    print(f"  - Has valid format: {metadata.get('has_valid_format')}")
    
    assert score == 1.0, f"Expected 1.0 (no format bonus), got {score}"
    assert not metadata.get('has_valid_format'), "Should detect invalid format"
    
    await interaction.finalize_interaction(instance_id)
    
    print("✓ TEST 4 PASSED")


async def main():
    """Run all tests."""
    print("\n" + "="*80)
    print("MEDICAL GAME INTERACTION TEST SUITE")
    print("="*80)
    
    try:
        await test_benign_correct()
        await test_benign_incorrect()
        await test_error_correct()
        await test_invalid_format()
        
        print("\n" + "="*80)
        print("ALL TESTS PASSED ✓")
        print("="*80)
        print("\nInteraction system is working correctly!")
        print("Ready to run full training with verl.")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
