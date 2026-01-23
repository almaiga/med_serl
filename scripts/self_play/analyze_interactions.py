#!/usr/bin/env python3
"""Analyze self-play interaction logs for failure diagnosis.

Usage:
    python scripts/self_play/analyze_interactions.py [log_file]
    
If no log file specified, uses the most recent one in results/self_play/interactions/
"""

import json
import sys
from pathlib import Path
from collections import Counter
from typing import Optional


def load_interactions(log_path: Path) -> list:
    """Load interaction logs from JSONL file."""
    interactions = []
    with open(log_path, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    interactions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return interactions


def analyze_failures(interactions: list) -> dict:
    """Analyze patterns in failures."""
    outcomes = Counter()
    modes = Counter()
    failure_patterns = Counter()
    
    for ix in interactions:
        outcome = ix.get("outcome", "unknown")
        mode = ix.get("mode", "unknown")
        
        outcomes[outcome] += 1
        modes[mode] += 1
        
        # Analyze failure patterns
        if outcome in ["wrong", "invalid_format"]:
            response = ix.get("model_response_full", "")
            
            # Check for common failure patterns
            if "incomplete" in response.lower() or "clarify" in response.lower():
                failure_patterns["confused_about_input"] += 1
            elif not response.strip():
                failure_patterns["empty_response"] += 1
            elif "CORRECT" not in response.upper() and "INCORRECT" not in response.upper():
                failure_patterns["no_classification"] += 1
            elif ix.get("model_answer") != ix.get("ground_truth"):
                failure_patterns["wrong_classification"] += 1
    
    return {
        "outcomes": dict(outcomes),
        "modes": dict(modes),
        "failure_patterns": dict(failure_patterns),
    }


def print_sample_interaction(ix: dict, label: str = "Sample"):
    """Print a formatted interaction for analysis."""
    print(f"\n{'='*60}")
    print(f"{label}")
    print(f"{'='*60}")
    print(f"Note ID: {ix.get('note_id', 'N/A')}")
    print(f"Mode: {ix.get('mode', 'N/A')}")
    print(f"Ground Truth: {ix.get('ground_truth', 'N/A')}")
    print(f"Model Answer: {ix.get('model_answer', 'N/A')}")
    print(f"Outcome: {ix.get('outcome', 'N/A')}")
    print(f"Reward: {ix.get('reward', 'N/A')}")
    print(f"Valid Format: {ix.get('has_valid_format', 'N/A')}")
    
    print(f"\n--- Original Correct Note (first 500 chars) ---")
    print(ix.get('original_correct_note', 'N/A')[:500])
    
    if ix.get('mode') == 'error_injection':
        print(f"\n--- Error Type ---")
        print(ix.get('error_type', 'N/A'))
        print(f"\n--- Error Sentence ---")
        print(ix.get('error_sentence', 'N/A'))
    
    print(f"\n--- Generated Note (Injector Output) ---")
    print(ix.get('generated_note', 'N/A')[:500] or "(not extracted)")
    
    print(f"\n--- Full Model Response (first 1000 chars) ---")
    print(ix.get('model_response_full', 'N/A')[:1000])
    print()


def main():
    # Find log file
    if len(sys.argv) > 1:
        log_path = Path(sys.argv[1])
    else:
        log_dir = Path(__file__).parent.parent.parent / "results" / "self_play" / "interactions"
        log_files = sorted(log_dir.glob("interactions_*.jsonl"), reverse=True)
        if not log_files:
            print("No interaction log files found in results/self_play/interactions/")
            print("Run training first to generate logs.")
            return
        log_path = log_files[0]
    
    print(f"Analyzing: {log_path}")
    
    interactions = load_interactions(log_path)
    print(f"Total interactions: {len(interactions)}")
    
    if not interactions:
        print("No interactions found in log file.")
        return
    
    # Analyze
    analysis = analyze_failures(interactions)
    
    print(f"\n{'='*60}")
    print("OUTCOME SUMMARY")
    print(f"{'='*60}")
    for outcome, count in sorted(analysis["outcomes"].items()):
        pct = count / len(interactions) * 100
        print(f"  {outcome}: {count} ({pct:.1f}%)")
    
    print(f"\n{'='*60}")
    print("MODE DISTRIBUTION")
    print(f"{'='*60}")
    for mode, count in sorted(analysis["modes"].items()):
        pct = count / len(interactions) * 100
        print(f"  {mode}: {count} ({pct:.1f}%)")
    
    print(f"\n{'='*60}")
    print("FAILURE PATTERNS")
    print(f"{'='*60}")
    for pattern, count in sorted(analysis["failure_patterns"].items(), key=lambda x: -x[1]):
        print(f"  {pattern}: {count}")
    
    # Show sample failures
    failures = [ix for ix in interactions if ix.get("outcome") in ["wrong", "invalid_format"]]
    successes = [ix for ix in interactions if ix.get("outcome") == "correct"]
    
    if failures:
        print_sample_interaction(failures[0], "SAMPLE FAILURE #1")
        if len(failures) > 1:
            print_sample_interaction(failures[1], "SAMPLE FAILURE #2")
    
    if successes:
        print_sample_interaction(successes[0], "SAMPLE SUCCESS")
    
    # Check for the "confused about input" issue
    confused = [
        ix for ix in interactions 
        if "incomplete" in ix.get("model_response_full", "").lower()
        or "clarify" in ix.get("model_response_full", "").lower()
    ]
    if confused:
        print(f"\n{'='*60}")
        print(f"WARNING: {len(confused)} responses show model confusion about input!")
        print("This suggests prompts are not being passed correctly to the model.")
        print(f"{'='*60}")
        print_sample_interaction(confused[0], "CONFUSED RESPONSE EXAMPLE")


if __name__ == "__main__":
    main()
