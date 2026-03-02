"""
Test harness for the Agentic UMLS Judge.

Loads N examples from the self-play Parquet dataset, simulates assessor
responses, runs both the rule-based and agentic reward functions, and
compares the scores. Also validates the 3-step pipeline (Extract → Retrieve
→ Adjudicate) with detailed traces.

Usage:
    # Requires: UMLS_API_KEY env var, vLLM judge server running on JUDGE_VLLM_URL

    # Quick smoke test (5 examples, no UMLS/LLM — tests plumbing only)
    python -m scripts.self_play.test_agentic_judge --dry-run --n 5

    # Full test with UMLS + judge LLM (50 examples)
    python -m scripts.self_play.test_agentic_judge --n 50

    # Custom vLLM endpoint
    JUDGE_VLLM_URL=http://localhost:8001/v1/chat/completions \\
    python -m scripts.self_play.test_agentic_judge --n 50
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.self_play.reward_function import compute_score as rule_compute_score
from scripts.self_play.utils import parse_assessor_answer, strip_thinking, split_sentences
from scripts.self_play.judge_prompts import (
    get_config,
    get_extraction_system_prompt,
    get_adjudication_system_prompt,
    format_evidence_for_prompt,
)


# =============================================================================
# Data loading
# =============================================================================

def load_parquet_examples(path: str, n: int = 50) -> List[Dict[str, Any]]:
    """Load N examples from a verl Parquet dataset."""
    try:
        from datasets import load_dataset
        ds = load_dataset("parquet", data_files=path, split="train")
    except ImportError:
        import pandas as pd
        df = pd.read_parquet(path)
        ds = df.to_dict(orient="records")

    examples = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        # Extract fields - handle both dict and HF Dataset rows
        if hasattr(row, "keys"):
            ex = dict(row)
        else:
            ex = row

        extra_info = ex.get("extra_info", {})
        if isinstance(extra_info, str):
            extra_info = json.loads(extra_info)

        reward_model = ex.get("reward_model", {})
        if isinstance(reward_model, str):
            reward_model = json.loads(reward_model)

        examples.append({
            "data_source": ex.get("data_source", "medec_selfplay"),
            "prompt": ex.get("prompt", []),
            "ground_truth": reward_model.get("ground_truth", "CORRECT"),
            "extra_info": extra_info,
            "mode": extra_info.get("mode", "unknown"),
            "note_id": extra_info.get("note_id", f"test-{i}"),
        })

    return examples


# =============================================================================
# Simulated assessor responses
# =============================================================================

def simulate_assessor_responses(examples: List[Dict]) -> List[str]:
    """Generate simulated assessor responses for testing.

    For each example, creates a response that is:
    - Correct ~60% of the time ("CORRECT" for benign, right sentence for error)
    - Partially correct ~20% (wrong sentence number for errors)
    - Wrong ~20%
    """
    import random
    random.seed(42)
    responses = []

    for ex in examples:
        gt = ex["ground_truth"]
        r = random.random()

        if gt == "CORRECT":
            if r < 0.7:
                resp = "CORRECT"
            else:
                resp = str(random.randint(1, 10))
        else:
            try:
                sid = int(gt)
            except (ValueError, TypeError):
                sid = 3

            if r < 0.5:
                resp = str(sid)  # Exact match
            elif r < 0.75:
                resp = str(max(1, sid + random.choice([-1, 1])))  # Wrong sentence
            else:
                resp = "CORRECT"  # Missed error

        # Optionally wrap in <think> tags
        if random.random() < 0.3:
            resp = f"<think>Analyzing the note carefully...</think>\n{resp}"

        responses.append(resp)

    return responses


# =============================================================================
# Dry run (tests plumbing without UMLS/LLM)
# =============================================================================

def run_dry_test(examples: List[Dict], responses: List[str]) -> Dict[str, Any]:
    """Test the rule-based reward and plumbing only (no UMLS/LLM calls)."""
    print("\n" + "=" * 70)
    print("DRY RUN — testing plumbing (rule-based only, no UMLS/LLM)")
    print("=" * 70)

    results = []
    for i, (ex, resp) in enumerate(zip(examples, responses)):
        rule_score = rule_compute_score(
            data_source=ex["data_source"],
            solution_str=resp,
            ground_truth=ex["ground_truth"],
            extra_info=ex["extra_info"],
        )
        label, pred_sid = parse_assessor_answer(resp)
        results.append({
            "idx": i,
            "note_id": ex["note_id"],
            "mode": ex["mode"],
            "ground_truth": ex["ground_truth"],
            "assessor_label": label,
            "assessor_pred_sid": pred_sid,
            "rule_score": rule_score,
        })
        print(f"  [{i:3d}] {ex['mode']:16s} gt={ex['ground_truth']:8s} "
              f"pred={label:8s}({pred_sid}) rule={rule_score:+.2f}")

    # Summary
    scores = [r["rule_score"] for r in results]
    benign = [r for r in results if r["mode"] == "benign"]
    error = [r for r in results if r["mode"] != "benign"]
    print(f"\n  Avg rule score:   {sum(scores)/len(scores):.3f}")
    if benign:
        print(f"  Avg benign score: {sum(r['rule_score'] for r in benign)/len(benign):.3f}")
    if error:
        print(f"  Avg error score:  {sum(r['rule_score'] for r in error)/len(error):.3f}")

    # Test prompt loading
    print("\n--- Prompt config validation ---")
    config = get_config()
    print(f"  Config keys: {list(config.keys())}")
    print(f"  Extraction system prompt length: {len(get_extraction_system_prompt())} chars")
    print(f"  Adjudication system prompt length: {len(get_adjudication_system_prompt())} chars")
    print(f"  Evidence format: {format_evidence_for_prompt([])}")

    # Test with sample evidence
    sample_evidence = [
        {
            "entity_name": "metformin",
            "cui": "C0025598",
            "semantic_type": "Pharmacologic Substance",
            "synonyms": ["Glucophage", "metformin hydrochloride"],
            "relations": [
                {"relation": "treats", "related_name": "type 2 diabetes mellitus", "related_cui": "C0011860"}
            ],
            "source": "UMLS",
            "found": True,
        }
    ]
    formatted = format_evidence_for_prompt(sample_evidence)
    print(f"  Sample evidence formatting:\n    {formatted[:200]}...")

    return {"status": "ok", "n_tested": len(results), "results": results}


# =============================================================================
# Full test (with UMLS + LLM)
# =============================================================================

async def run_full_test(examples: List[Dict], responses: List[str]) -> Dict[str, Any]:
    """Test the full agentic reward pipeline (UMLS + judge LLM)."""
    # Import agentic reward (lazy to avoid import errors in dry-run mode)
    from scripts.self_play.agentic_reward import (
        compute_score as agentic_compute_score,
        async_compute_score,
        _extract_entities,
        _retrieve_evidence,
        _adjudicate,
        _identify_changed_sentences,
        JUDGE_URL,
        JUDGE_MODEL,
        RULE_WEIGHT,
        UMLS_WEIGHT,
    )

    print("\n" + "=" * 70)
    print("FULL TEST — Agentic UMLS Judge Pipeline")
    print("=" * 70)
    print(f"  Judge URL:    {JUDGE_URL}")
    print(f"  Judge Model:  {JUDGE_MODEL}")
    print(f"  UMLS API Key: {'set' if os.getenv('UMLS_API_KEY') else 'NOT SET'}")
    print(f"  Rule Weight:  {RULE_WEIGHT}")
    print(f"  UMLS Weight:  {UMLS_WEIGHT}")
    print(f"  Examples:     {len(examples)}")

    # Pre-flight: check judge server
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                JUDGE_URL.replace("/chat/completions", "/models"),
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    print("  Judge server: ONLINE")
                else:
                    print(f"  Judge server: returned {resp.status} (may still work)")
    except Exception as e:
        print(f"  Judge server: UNREACHABLE ({e})")
        print("  WARNING: Judge LLM calls will fail. Run with --dry-run instead.")

    results = []
    start_time = time.time()

    for i, (ex, resp) in enumerate(zip(examples, responses)):
        t0 = time.time()

        # Rule-based score
        rule_score = rule_compute_score(
            data_source=ex["data_source"],
            solution_str=resp,
            ground_truth=ex["ground_truth"],
            extra_info=ex["extra_info"],
        )

        # Agentic score (sync wrapper — what verl actually calls)
        agentic_score = agentic_compute_score(
            data_source=ex["data_source"],
            solution_str=resp,
            ground_truth=ex["ground_truth"],
            extra_info=ex["extra_info"],
        )

        elapsed = time.time() - t0
        label, pred_sid = parse_assessor_answer(resp)

        results.append({
            "idx": i,
            "note_id": ex["note_id"],
            "mode": ex["mode"],
            "ground_truth": ex["ground_truth"],
            "assessor_label": label,
            "assessor_pred_sid": pred_sid,
            "rule_score": rule_score,
            "agentic_score": agentic_score,
            "delta": agentic_score - rule_score,
            "elapsed_s": elapsed,
        })

        print(f"  [{i:3d}] {ex['mode']:16s} gt={ex['ground_truth']:8s} "
              f"rule={rule_score:+.2f} agentic={agentic_score:+.2f} "
              f"Δ={agentic_score - rule_score:+.3f} ({elapsed:.1f}s)")

    total_time = time.time() - start_time

    # Summary
    print("\n" + "-" * 70)
    print("SUMMARY")
    print("-" * 70)
    n = len(results)
    rule_avg = sum(r["rule_score"] for r in results) / n
    agent_avg = sum(r["agentic_score"] for r in results) / n
    delta_avg = sum(r["delta"] for r in results) / n
    time_avg = sum(r["elapsed_s"] for r in results) / n

    print(f"  Examples tested: {n}")
    print(f"  Avg rule score:    {rule_avg:.3f}")
    print(f"  Avg agentic score: {agent_avg:.3f}")
    print(f"  Avg delta:         {delta_avg:+.3f}")
    print(f"  Avg time/example:  {time_avg:.1f}s")
    print(f"  Total time:        {total_time:.1f}s")

    # Breakdown by mode
    for mode in ("benign", "error_injection"):
        subset = [r for r in results if r["mode"] == mode]
        if subset:
            rule_m = sum(r["rule_score"] for r in subset) / len(subset)
            agent_m = sum(r["agentic_score"] for r in subset) / len(subset)
            print(f"\n  [{mode}] n={len(subset)}")
            print(f"    rule:    {rule_m:.3f}")
            print(f"    agentic: {agent_m:.3f}")
            print(f"    delta:   {agent_m - rule_m:+.3f}")

    # Save results
    output_path = PROJECT_ROOT / "results" / "self_play" / "judge_test_results.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, default=str) + "\n")
    print(f"\n  Results saved to {output_path}")

    return {"status": "ok", "n_tested": n, "results": results}


# =============================================================================
# Standalone entity extraction test
# =============================================================================

async def test_extraction_only(examples: List[Dict]) -> None:
    """Test just Step 1 (entity extraction) on a few examples."""
    from scripts.self_play.agentic_reward import _extract_entities, _identify_changed_sentences

    print("\n" + "=" * 70)
    print("EXTRACTION TEST — Step 1 Only")
    print("=" * 70)

    for i, ex in enumerate(examples[:10]):
        if ex["mode"] == "benign":
            continue
        orig, mod = _identify_changed_sentences(ex["extra_info"])
        if not mod:
            print(f"  [{i}] Skipping — no modified sentence found")
            continue

        print(f"\n  [{i}] note_id={ex['note_id']}")
        print(f"    Original:  {orig[:100]}...")
        print(f"    Modified:  {mod[:100]}...")

        entities = await _extract_entities(mod)
        print(f"    Entities:  {json.dumps(entities, indent=2)[:300]}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test the Agentic UMLS Judge")
    parser.add_argument("--n", type=int, default=50, help="Number of examples to test")
    parser.add_argument("--data", type=str,
                        default=str(PROJECT_ROOT / "data_processed" / "self_play" / "train.parquet"),
                        help="Path to Parquet dataset")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test plumbing only (no UMLS/LLM calls)")
    parser.add_argument("--extraction-only", action="store_true",
                        help="Test Step 1 (entity extraction) only")
    args = parser.parse_args()

    print(f"Loading {args.n} examples from {args.data}...")
    examples = load_parquet_examples(args.data, n=args.n)
    print(f"Loaded {len(examples)} examples")

    # Show mode distribution
    modes = {}
    for ex in examples:
        modes[ex["mode"]] = modes.get(ex["mode"], 0) + 1
    print(f"Mode distribution: {modes}")

    if args.dry_run:
        responses = simulate_assessor_responses(examples)
        run_dry_test(examples, responses)
    elif args.extraction_only:
        asyncio.run(test_extraction_only(examples))
    else:
        responses = simulate_assessor_responses(examples)
        asyncio.run(run_full_test(examples, responses))


if __name__ == "__main__":
    main()
