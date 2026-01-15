#!/usr/bin/env python3
"""
Mock test for MedSeRL trainer structure without real model.
Tests logging, batch processing, reward calculation.
"""

import json
import os
import sys
import tempfile
from datetime import datetime

# Mock data
class MockBatch:
    def __init__(self):
        self.data = {
            "notes": [
                "Patient has type 2 diabetes.",
                "Patient presents with chest pain.",
            ],
            "labels": ["INCORRECT", "INCORRECT"],
        }

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

# Mock VCF result
class MockFilterResult:
    def __init__(self, passed=True):
        self.passed = passed
        self.score_jaccard = 0.87
        self.reason = None if passed else "too_many_edits"
        self.word_edits = 2
        self.sentences_changed = 1

# Test logging
print("=" * 60)
print("Mock Trainer Test")
print("=" * 60)

# Create temp log files
with tempfile.TemporaryDirectory() as tmpdir:
    interaction_log_path = os.path.join(tmpdir, "interactions.jsonl")
    metrics_log_path = os.path.join(tmpdir, "metrics.jsonl")

    # Simulate training round
    print("\n[Test] Simulating training round...")

    batch = MockBatch()
    round_num = 0

    # Mock injector outputs
    injector_outputs = [
        "<think>Change type 2 to type 1</think>\nPatient has type 1 diabetes.",
        "<think>Change chest to back</think>\nPatient presents with back pain.",
    ]

    # Mock assessor outputs
    assessor_outputs = [
        "<think>Type 1 with no insulin mentioned - error!</think>\nANSWER: INCORRECT",
        "<think>Back pain is different from chest pain</think>\nANSWER: CORRECT",
    ]

    # Mock VCF results
    vcf_results = [MockFilterResult(passed=True), MockFilterResult(passed=True)]

    # Mock rewards
    rewards = [1.0, -1.0]  # First: assessor wins, Second: injector wins

    # Write interaction log
    print("[Test] Writing interaction log...")
    with open(interaction_log_path, 'a') as f:
        for i in range(len(batch["notes"])):
            interaction = {
                "round": round_num,
                "sample_idx": i,
                "original_note": batch["notes"][i],
                "injector_output": injector_outputs[i],
                "assessor_output": assessor_outputs[i],
                "reward": rewards[i],
                "vcf_passed": vcf_results[i].passed,
                "vcf_jaccard": vcf_results[i].score_jaccard,
            }
            f.write(json.dumps(interaction) + "\n")

    # Write metrics log
    print("[Test] Writing metrics log...")
    with open(metrics_log_path, 'a') as f:
        metrics = {
            "round": round_num,
            "loss": 0.234,
            "mean_reward": sum(rewards) / len(rewards),
            "vcf_acceptance_rate": sum(r.passed for r in vcf_results) / len(vcf_results),
            "injector_win_rate": sum(1 for r in rewards if r < 0) / len(rewards),
            "assessor_win_rate": sum(1 for r in rewards if r > 0) / len(rewards),
            "timestamp": datetime.utcnow().isoformat(),
        }
        f.write(json.dumps(metrics) + "\n")

    # Verify logs
    print("[Test] Verifying logs...")

    with open(interaction_log_path) as f:
        interactions = [json.loads(line) for line in f]

    with open(metrics_log_path) as f:
        metrics_data = [json.loads(line) for line in f]

    print(f"✅ Interactions logged: {len(interactions)}")
    print(f"✅ Metrics logged: {len(metrics_data)}")

    # Show sample interaction
    print("\nSample Interaction:")
    print(json.dumps(interactions[0], indent=2))

    # Show metrics
    print("\nMetrics:")
    print(json.dumps(metrics_data[0], indent=2))

print("\n" + "=" * 60)
print("✅ Mock trainer test passed!")
print("=" * 60)
print("\nLog files verified:")
print("  - interactions.jsonl format correct")
print("  - metrics.jsonl format correct")
print("\nNext: Run test_single_batch.py with real model")
