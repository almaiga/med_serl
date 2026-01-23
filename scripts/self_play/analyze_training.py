#!/usr/bin/env python3
"""Analyze self-play training results following SeRL paper metrics.

Generates summary statistics like:
- Win rates (Assessor vs Injector)
- Accuracy by mode (benign vs error_injection)
- Average rewards
- Error analysis

Usage:
    python scripts/self_play/analyze_training.py [--log-dir results/self_play/interactions]
"""

import json
import argparse
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import sys


def load_interactions(log_dir: Path) -> list:
    """Load all interaction logs from directory."""
    interactions = []
    
    # Find all interaction log files
    log_files = sorted(log_dir.glob("interactions_*.jsonl"), reverse=True)
    
    if not log_files:
        print(f"No interaction logs found in {log_dir}")
        return []
    
    # Use most recent log file
    log_file = log_files[0]
    print(f"Loading from: {log_file}")
    
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                try:
                    interactions.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    
    return interactions


def compute_statistics(interactions: list) -> dict:
    """Compute comprehensive statistics following SeRL paper format."""
    
    if not interactions:
        return {"error": "No interactions to analyze"}
    
    stats = {
        "total": len(interactions),
        "by_outcome": defaultdict(int),
        "by_mode": defaultdict(lambda: defaultdict(int)),
        "rewards": {
            "total": 0.0,
            "by_mode": defaultdict(float),
            "by_outcome": defaultdict(float),
        },
        "format_compliance": {
            "valid": 0,
            "invalid": 0,
        },
        "error_types": defaultdict(lambda: {"correct": 0, "wrong": 0}),
        # Token/generation metrics
        "token_metrics": {
            "total_chars": 0,
            "min_chars": float('inf'),
            "max_chars": 0,
            "truncated": 0,
            "with_think_tags": 0,
            "missing_closing_think": 0,
        },
    }
    
    for ix in interactions:
        outcome = ix.get("outcome", "unknown")
        mode = ix.get("mode", "unknown")
        reward = ix.get("reward", 0.0)
        has_format = ix.get("has_valid_format", False)
        error_type = ix.get("error_type", "") or "none"
        
        # Token metrics (if available)
        resp_chars = ix.get("response_chars", 0)
        if resp_chars:
            stats["token_metrics"]["total_chars"] += resp_chars
            if resp_chars < stats["token_metrics"]["min_chars"]:
                stats["token_metrics"]["min_chars"] = resp_chars
            if resp_chars > stats["token_metrics"]["max_chars"]:
                stats["token_metrics"]["max_chars"] = resp_chars
        if ix.get("is_truncated", False):
            stats["token_metrics"]["truncated"] += 1
        if ix.get("has_think_tag", False):
            stats["token_metrics"]["with_think_tags"] += 1
        if ix.get("missing_closing_think", False):
            stats["token_metrics"]["missing_closing_think"] += 1
        
        # Overall outcomes
        stats["by_outcome"][outcome] += 1
        
        # By mode
        stats["by_mode"][mode][outcome] += 1
        stats["by_mode"][mode]["total"] = stats["by_mode"][mode].get("total", 0) + 1
        
        # Rewards
        stats["rewards"]["total"] += reward
        stats["rewards"]["by_mode"][mode] += reward
        stats["rewards"]["by_outcome"][outcome] += reward
        
        # Format compliance
        if has_format:
            stats["format_compliance"]["valid"] += 1
        else:
            stats["format_compliance"]["invalid"] += 1
        
        # Error type analysis (for error_injection mode)
        if mode == "error_injection" and error_type:
            if outcome == "correct":
                stats["error_types"][error_type]["correct"] += 1
            else:
                stats["error_types"][error_type]["wrong"] += 1
    
    # Compute derived metrics
    total = stats["total"]
    correct = stats["by_outcome"].get("correct", 0)
    wrong = stats["by_outcome"].get("wrong", 0)
    invalid = stats["by_outcome"].get("invalid_format", 0)
    
    # Token metrics
    token_metrics = stats["token_metrics"]
    total_chars = token_metrics["total_chars"]
    min_chars = token_metrics["min_chars"] if token_metrics["min_chars"] != float('inf') else 0
    max_chars = token_metrics["max_chars"]
    truncated = token_metrics["truncated"]
    
    stats["metrics"] = {
        # Overall
        "accuracy": correct / total if total > 0 else 0,
        "win_rate_assessor": correct / total if total > 0 else 0,
        "win_rate_injector": wrong / total if total > 0 else 0,
        "invalid_rate": invalid / total if total > 0 else 0,
        "format_compliance_rate": stats["format_compliance"]["valid"] / total if total > 0 else 0,
        
        # Average rewards
        "avg_reward": stats["rewards"]["total"] / total if total > 0 else 0,
        
        # Token metrics
        "avg_response_chars": total_chars / total if total > 0 else 0,
        "avg_response_tokens_approx": (total_chars / 4) / total if total > 0 else 0,
        "min_response_chars": min_chars,
        "max_response_chars": max_chars,
        "truncation_rate": truncated / total if total > 0 else 0,
        "truncated_count": truncated,
        "with_think_tags": token_metrics["with_think_tags"],
        "missing_closing_think": token_metrics["missing_closing_think"],
    }
    
    # Per-mode metrics
    for mode in stats["by_mode"]:
        mode_total = stats["by_mode"][mode]["total"]
        mode_correct = stats["by_mode"][mode].get("correct", 0)
        mode_reward = stats["rewards"]["by_mode"].get(mode, 0)
        
        stats["metrics"][f"{mode}_accuracy"] = mode_correct / mode_total if mode_total > 0 else 0
        stats["metrics"][f"{mode}_avg_reward"] = mode_reward / mode_total if mode_total > 0 else 0
        stats["metrics"][f"{mode}_count"] = mode_total
    
    return stats


def print_report(stats: dict):
    """Print formatted report."""
    
    if "error" in stats:
        print(stats["error"])
        return
    
    print("\n" + "="*70)
    print("MEDSERL SELF-PLAY TRAINING ANALYSIS")
    print("Following SeRL paper (arXiv:2506.07468) metrics")
    print("="*70)
    
    metrics = stats.get("metrics", {})
    
    print(f"\n📊 OVERALL STATISTICS (n={stats['total']})")
    print("-"*50)
    print(f"  Accuracy:              {metrics.get('accuracy', 0):.2%}")
    print(f"  Assessor Win Rate:     {metrics.get('win_rate_assessor', 0):.2%}")
    print(f"  Injector Win Rate:     {metrics.get('win_rate_injector', 0):.2%}")
    print(f"  Invalid Format Rate:   {metrics.get('invalid_rate', 0):.2%}")
    print(f"  Format Compliance:     {metrics.get('format_compliance_rate', 0):.2%}")
    print(f"  Average Reward:        {metrics.get('avg_reward', 0):.3f}")
    
    print(f"\n🎯 BY MODE")
    print("-"*50)
    
    for mode in ["benign", "error_injection"]:
        count = metrics.get(f"{mode}_count", 0)
        if count > 0:
            acc = metrics.get(f"{mode}_accuracy", 0)
            avg_r = metrics.get(f"{mode}_avg_reward", 0)
            print(f"  {mode.upper():20} (n={count})")
            print(f"    Accuracy:          {acc:.2%}")
            print(f"    Avg Reward:        {avg_r:.3f}")
    
    print(f"\n📈 OUTCOME DISTRIBUTION")
    print("-"*50)
    by_outcome = stats.get("by_outcome", {})
    for outcome, count in sorted(by_outcome.items()):
        pct = count / stats["total"] * 100
        bar = "█" * int(pct / 2)
        print(f"  {outcome:15} {count:5} ({pct:5.1f}%) {bar}")
    
    # Token/Generation metrics
    print(f"\n📏 TOKEN/GENERATION METRICS")
    print("-"*50)
    print(f"  Avg Response:          {metrics.get('avg_response_chars', 0):.0f} chars (~{metrics.get('avg_response_tokens_approx', 0):.0f} tokens)")
    print(f"  Min Response:          {metrics.get('min_response_chars', 0)} chars")
    print(f"  Max Response:          {metrics.get('max_response_chars', 0)} chars")
    print(f"  Truncation Rate:       {metrics.get('truncation_rate', 0):.2%} ({metrics.get('truncated_count', 0)} truncated)")
    print(f"  With <think> tags:     {metrics.get('with_think_tags', 0)}")
    print(f"  Missing </think>:      {metrics.get('missing_closing_think', 0)}")
    
    # Error type analysis
    error_types = stats.get("error_types", {})
    if error_types and any(v["correct"] + v["wrong"] > 0 for v in error_types.values() if v != "none"):
        print(f"\n🔬 ERROR TYPE ANALYSIS (error_injection mode)")
        print("-"*50)
        for etype, counts in sorted(error_types.items()):
            if etype == "none":
                continue
            total_type = counts["correct"] + counts["wrong"]
            if total_type > 0:
                acc = counts["correct"] / total_type
                print(f"  {etype:20} {total_type:4} samples, {acc:.1%} accuracy")
    
    print("\n" + "="*70)


def print_sample_interactions(interactions: list, n: int = 3):
    """Print sample interactions for review."""
    
    if not interactions:
        return
    
    # Get samples from different categories
    correct_samples = [ix for ix in interactions if ix.get("outcome") == "correct"][:n]
    wrong_samples = [ix for ix in interactions if ix.get("outcome") == "wrong"][:n]
    
    print(f"\n📝 SAMPLE CORRECT CLASSIFICATIONS")
    print("-"*50)
    for ix in correct_samples[:2]:
        print(f"  Note ID: {ix.get('note_id', 'N/A')}")
        print(f"  Mode: {ix.get('mode', 'N/A')}, GT: {ix.get('ground_truth', 'N/A')}")
        print(f"  Note preview: {ix.get('original_correct_note', '')[:100]}...")
        print()
    
    print(f"\n📝 SAMPLE WRONG CLASSIFICATIONS")
    print("-"*50)
    for ix in wrong_samples[:2]:
        print(f"  Note ID: {ix.get('note_id', 'N/A')}")
        print(f"  Mode: {ix.get('mode', 'N/A')}")
        print(f"  Ground Truth: {ix.get('ground_truth', 'N/A')}")
        print(f"  Model Answer: {ix.get('model_answer', 'N/A')}")
        if ix.get("mode") == "error_injection":
            print(f"  Error Type: {ix.get('error_type', 'N/A')}")
            print(f"  Error Sentence: {ix.get('error_sentence', 'N/A')[:100]}...")
        print()


def main():
    parser = argparse.ArgumentParser(description="Analyze self-play training results")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("results/self_play/interactions"),
        help="Directory containing interaction logs",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2,
        help="Number of sample interactions to show",
    )
    parser.add_argument(
        "--export-json",
        type=Path,
        default=None,
        help="Export statistics to JSON file",
    )
    
    args = parser.parse_args()
    
    # Load interactions
    interactions = load_interactions(args.log_dir)
    
    if not interactions:
        print("No interactions found. Run training first.")
        sys.exit(1)
    
    # Compute statistics
    stats = compute_statistics(interactions)
    
    # Print report
    print_report(stats)
    
    # Print samples
    if args.samples > 0:
        print_sample_interactions(interactions, args.samples)
    
    # Export if requested
    if args.export_json:
        # Convert defaultdicts to regular dicts for JSON
        def convert_defaultdict(obj):
            if isinstance(obj, defaultdict):
                return {k: convert_defaultdict(v) for k, v in obj.items()}
            elif isinstance(obj, dict):
                return {k: convert_defaultdict(v) for k, v in obj.items()}
            return obj
        
        export_stats = convert_defaultdict(stats)
        export_stats["exported_at"] = datetime.now().isoformat()
        
        with open(args.export_json, 'w') as f:
            json.dump(export_stats, f, indent=2)
        print(f"\nExported statistics to: {args.export_json}")


if __name__ == "__main__":
    main()
