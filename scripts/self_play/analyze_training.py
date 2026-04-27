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
import re
import subprocess


def is_game_log(path: Path) -> bool:
    """Return True if a JSONL file appears to contain self-play game rows."""
    try:
        with open(path, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                return bool(
                    row.get("phase") == "game_complete"
                    or row.get("turn_reward_spans")
                    or row.get("injector_output")
                    or row.get("assessor_output")
                )
    except (OSError, json.JSONDecodeError):
        return False
    return False


def git_tracked_files(root: Path) -> set[Path]:
    """Return repo-tracked files under root, or an empty set outside git."""
    try:
        proc = subprocess.run(
            ["git", "ls-files", "--", str(root)],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return set()

    repo_root = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=False,
        capture_output=True,
        text=True,
    )
    base = Path(repo_root.stdout.strip()) if repo_root.returncode == 0 else Path.cwd()
    return {(base / line.strip()).resolve() for line in proc.stdout.splitlines() if line.strip()}


def count_rows(path: Path) -> int:
    try:
        with open(path, "r") as f:
            return sum(1 for line in f if line.strip())
    except OSError:
        return 0


def find_game_logs(log_dir: Path) -> list[Path]:
    logs = [p for p in log_dir.glob("*.jsonl") if is_game_log(p)]
    tracked = git_tracked_files(log_dir)
    local_logs = [p for p in logs if p.resolve() not in tracked]
    selected = local_logs if local_logs else logs
    return sorted(selected, key=lambda p: p.stat().st_mtime, reverse=True)


def print_log_listing(log_dir: Path, limit: int = 30) -> None:
    logs = find_game_logs(log_dir)
    if not logs:
        print(f"No self-play game logs found in {log_dir}")
        return
    print(f"Self-play game logs in {log_dir}:")
    for path in logs[:limit]:
        ts = datetime.fromtimestamp(path.stat().st_mtime).strftime("%Y-%m-%d %H:%M")
        print(f"  {ts} rows={count_rows(path):5d} {path}")


def load_interactions(log_path: Path) -> list:
    """Load interactions from a specific file or a directory.

    When given a directory, choose the newest JSONL file that looks like a
    self-play game log. This includes custom MEDSERL_GAME_LOG names such as
    `smoke_*.jsonl`, not only the default `game_*.jsonl` files.
    """
    interactions = []

    if log_path.is_file():
        log_file = log_path
    else:
        log_files = find_game_logs(log_path)
        if not log_files:
            print(f"No interaction logs found in {log_path}")
            return []
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
            "assessor_token_total": 0,
            "assessor_token_count": 0,
            "assessor_cap_hits": 0,
        },
        # Note similarity metrics
        "similarity": {
            "benign_total": 0.0,
            "benign_count": 0,
            "error_total": 0.0,
            "error_count": 0,
        },
    }
    
    for ix in interactions:
        # Fallbacks for both single-turn and our new multi-turn selfplay logging
        outcome = ix.get("assessor_outcome", ix.get("outcome", "unknown"))
        mode = ix.get("mode", "unknown")
        reward = float(ix.get("assessor_reward", ix.get("reward", 0.0)))
        
        # Format compliance fallback
        has_format = ix.get("has_valid_format", outcome != "invalid_format" and outcome != "parse_failure")
        error_type = ix.get("error_type", "") or "none"
        
        # Token metrics (if available)
        resp_chars = ix.get("response_chars", 0)
        if not resp_chars:
            # Multi-turn logs store per-phase responses instead of a single field.
            inj = ix.get("injector_output", ix.get("injector_response", "")) or ""
            asm = ix.get("assessor_output", ix.get("assessor_response", "")) or ""
            resp_chars = len(inj) + len(asm)
        if resp_chars:
            stats["token_metrics"]["total_chars"] += resp_chars
            if resp_chars < stats["token_metrics"]["min_chars"]:
                stats["token_metrics"]["min_chars"] = resp_chars
            if resp_chars > stats["token_metrics"]["max_chars"]:
                stats["token_metrics"]["max_chars"] = resp_chars
        if ix.get("is_truncated", False):
            stats["token_metrics"]["truncated"] += 1
        elif ix.get("outcome") == "invalid_format":
            # In the current setup, most invalid formats are practical truncation
            # events from overlong assessor generations.
            stats["token_metrics"]["truncated"] += 1

        has_think = ix.get("has_think_tag", False)
        if not has_think:
            joined = f"{ix.get('injector_output', ix.get('injector_response', ''))}\n{ix.get('assessor_output', ix.get('assessor_response', ''))}"
            has_think = "<think>" in joined.lower()
        if has_think:
            stats["token_metrics"]["with_think_tags"] += 1
            
        missing_closing = ix.get("missing_closing_think", False)
        if not missing_closing:
            joined = f"{ix.get('injector_output', ix.get('injector_response', ''))}\n{ix.get('assessor_output', ix.get('assessor_response', ''))}".lower()
            missing_closing = ("<think>" in joined) and ("</think>" not in joined)
        if missing_closing:
            stats["token_metrics"]["missing_closing_think"] += 1

        assessor_tokens = ix.get("assessor_token_count")
        assessor_max_tokens = ix.get("assessor_max_new_tokens")
        if assessor_tokens is not None:
            try:
                assessor_tokens = int(assessor_tokens)
                stats["token_metrics"]["assessor_token_total"] += assessor_tokens
                stats["token_metrics"]["assessor_token_count"] += 1
                if assessor_max_tokens is not None and assessor_tokens >= int(assessor_max_tokens):
                    stats["token_metrics"]["assessor_cap_hits"] += 1
            except (TypeError, ValueError):
                pass
        
        # Similarity metrics (if available)
        similarity = ix.get("note_similarity")
        if similarity is not None and ix.get("has_generated_note", False):
            if mode == "benign":
                stats["similarity"]["benign_total"] += similarity
                stats["similarity"]["benign_count"] += 1
            else:
                stats["similarity"]["error_total"] += similarity
                stats["similarity"]["error_count"] += 1
        
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
            if outcome == "exact_match":
                stats["error_types"][error_type]["correct"] += 1
            else:
                stats["error_types"][error_type]["wrong"] += 1
    
    # Compute derived metrics
    total = stats["total"]
    exact = stats["by_outcome"].get("exact_match", 0)
    partial = stats["by_outcome"].get("partial_match", 0)
    miss = stats["by_outcome"].get("miss", 0)
    invalid = stats["by_outcome"].get("invalid_format", 0)
    
    # Token metrics
    token_metrics = stats["token_metrics"]
    total_chars = token_metrics["total_chars"]
    min_chars = token_metrics["min_chars"] if token_metrics["min_chars"] != float('inf') else 0
    max_chars = token_metrics["max_chars"]
    truncated = token_metrics["truncated"]
    
    stats["metrics"] = {
        # Overall
        "accuracy": exact / total if total > 0 else 0,
        "win_rate_assessor": (exact + partial) / total if total > 0 else 0,
        "win_rate_injector": (miss + invalid) / total if total > 0 else 0,
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
        "avg_assessor_tokens": token_metrics["assessor_token_total"] / max(token_metrics["assessor_token_count"], 1),
        "assessor_cap_hits": token_metrics["assessor_cap_hits"],
        "assessor_token_count": token_metrics["assessor_token_count"],
        
        # Similarity metrics
        "avg_similarity_benign": stats["similarity"]["benign_total"] / max(stats["similarity"]["benign_count"], 1),
        "similarity_benign_count": stats["similarity"]["benign_count"],
        "avg_similarity_error": stats["similarity"]["error_total"] / max(stats["similarity"]["error_count"], 1),
        "similarity_error_count": stats["similarity"]["error_count"],
    }
    
    # Per-mode metrics
    for mode in stats["by_mode"]:
        mode_total = stats["by_mode"][mode]["total"]
        mode_correct = stats["by_mode"][mode].get("exact_match", 0)
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
    if metrics.get("assessor_token_count", 0):
        print(f"  Avg Assessor Tokens:   {metrics.get('avg_assessor_tokens', 0):.0f}")
        print(f"  Assessor Cap Hits:     {metrics.get('assessor_cap_hits', 0)}")
    
    # Note similarity metrics
    print(f"\n📝 NOTE SIMILARITY (Original vs Generated)")
    print("-"*50)
    benign_sim_count = metrics.get('similarity_benign_count', 0)
    error_sim_count = metrics.get('similarity_error_count', 0)
    if benign_sim_count > 0:
        print(f"  Benign mode:           {metrics.get('avg_similarity_benign', 0):.1%} similarity ({benign_sim_count} samples)")
    if error_sim_count > 0:
        print(f"  Error mode:            {metrics.get('avg_similarity_error', 0):.1%} similarity ({error_sim_count} samples)")
    
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

    def public_answer(text: str) -> str:
        text = text or ""
        m = re.search(r"<think>.*?</think>\s*(.*)", text, flags=re.DOTALL)
        if m:
            text = m.group(1)
        text = text.strip()
        if not text:
            return ""
        for line in text.splitlines():
            line = line.strip()
            if line:
                return line[:160]
        return ""

    def outcome(ix: dict) -> str:
        return ix.get("assessor_outcome", ix.get("outcome", "unknown"))

    def reward_span(ix: dict, role: str) -> dict:
        for span in ix.get("turn_reward_spans") or []:
            if isinstance(span, dict) and span.get("role") == role:
                return span
        return {}

    def print_turns(ix: dict) -> None:
        injector_span = reward_span(ix, "injector")
        assessor_span = reward_span(ix, "assessor")
        injector_reward = ix.get("injector_reward", injector_span.get("raw_reward", "N/A"))
        assigned_injector_reward = injector_span.get("reward", injector_reward)
        injector_target_return = injector_span.get("target_return", ix.get("injector_target_return", "N/A"))
        coupling_mode = injector_span.get("coupling_mode", ix.get("injector_coupling_mode", "N/A"))
        assessor_reward = ix.get("assessor_reward", assessor_span.get("reward", "N/A"))
        print(f"  Judge: {ix.get('judge_verdict', 'N/A')} ({ix.get('judge_status', '')})")
        print(
            f"  Injector reward: raw={injector_reward}, assigned={assigned_injector_reward}, "
            f"target_return={injector_target_return}, coupling={coupling_mode}"
        )
        print(f"  Assessor reward: {assessor_reward}")
        if ix.get("assessor_token_count") is not None:
            print(
                f"  Assessor tokens: {ix.get('assessor_token_count')}/{ix.get('assessor_max_new_tokens')} "
                f"cap_hit={ix.get('assessor_cap_hit', '')}"
            )
        print(f"  Injector output: {public_answer(ix.get('injector_output', ix.get('injector_response', '')))}")
        print(f"  Changed sentence: {ix.get('changed_sid', 'N/A')}")
        if ix.get("original_sentence") or ix.get("modified_sentence"):
            print(f"    Before: {str(ix.get('original_sentence', ''))[:180]}")
            print(f"    After : {str(ix.get('modified_sentence', ''))[:180]}")
        print(f"  Assessor output: {public_answer(ix.get('assessor_output', ix.get('assessor_response', '')))}")

    # Get samples from different categories. New game logs use
    # `assessor_outcome`; older reward-function logs use `outcome`.
    correct_samples = [ix for ix in interactions if outcome(ix) == "exact_match"][:n]
    wrong_samples = [
        ix for ix in interactions
        if outcome(ix) in ("miss", "invalid_format", "parse_failure", "game_invalid", "wrong")
    ][:n]

    print(f"\n📝 SAMPLE CORRECT CLASSIFICATIONS")
    print("-"*50)
    for ix in correct_samples[:n]:
        print(f"  Note ID: {ix.get('note_id', 'N/A')}")
        print(f"  Mode: {ix.get('mode', 'N/A')}, GT: {ix.get('ground_truth', 'N/A')}")
        note_preview = ix.get('modified_sentences') or public_answer(ix.get('injector_response', '')) or ix.get('model_response') or ''
        assessor_summary = ix.get('assessor_public_answer') or public_answer(ix.get('assessor_response', '')) or ix.get('model_response', 'N/A')
        print(f"  Note preview: {note_preview[:100]}...")
        print(f"  Assessor: {public_answer(ix.get('assessor_output', '')) or assessor_summary[:120]}")
        if ix.get("assessor_label") == "ERROR":
            print(f"  Parsed answer: sentence {ix.get('assessor_pred_sid', 'N/A')}")
        elif ix.get("assessor_label"):
            print(f"  Parsed answer: {ix.get('assessor_label')}")
        print_turns(ix)
        print()

    print(f"\n📝 SAMPLE WRONG CLASSIFICATIONS")
    print("-"*50)
    for ix in wrong_samples[:n]:
        print(f"  Note ID: {ix.get('note_id', 'N/A')}")
        print(f"  Mode: {ix.get('mode', 'N/A')}")
        print(f"  Ground Truth: {ix.get('ground_truth', 'N/A')}")
        assessor_summary = ix.get('assessor_public_answer') or public_answer(ix.get('assessor_response', '')) or ix.get('model_response', 'N/A')
        print(f"  Assessor Output: {public_answer(ix.get('assessor_output', '')) or assessor_summary[:160]}")
        if ix.get("mode") == "error_injection":
            print(f"  Error Type: {ix.get('error_type', 'N/A')}")
            print(f"  Predicted SID: {ix.get('assessor_pred_sid', 'N/A')}")
        print_turns(ix)
        print()


def main():
    parser = argparse.ArgumentParser(description="Analyze self-play training results")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("results/self_play/interactions"),
        help="Directory containing interaction logs, or a specific jsonl log file",
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
    parser.add_argument(
        "--list-logs",
        action="store_true",
        help="List discovered local self-play game logs and row counts, then exit",
    )
    
    args = parser.parse_args()

    if args.list_logs:
        print_log_listing(args.log_dir)
        return
    
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
