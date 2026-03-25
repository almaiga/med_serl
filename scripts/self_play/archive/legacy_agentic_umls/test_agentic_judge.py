"""
Test harness for the Agentic UMLS Judge.

Loads N examples from either:
  a) A self-play Parquet dataset (simulated responses), or
  b) A real game_interactions.jsonl from a training run (--from-interactions)

Runs both rule-based and agentic reward functions and compares scores.

Usage:
    # Quick dry-run (no UMLS/LLM — tests plumbing only)
    python -m scripts.self_play.test_agentic_judge --dry-run --n 10

    # Full test on real smoke-test outputs (judge server on separate instance)
    JUDGE_VLLM_URL=http://<judge-pod-ip>:8002/v1/chat/completions \\
    python -m scripts.self_play.test_agentic_judge \\
        --from-interactions outputs/self_play/reinforce_smoke_*/interactions_*.jsonl \\
        --n 30

    # Dry-run on real outputs (no judge server needed)
    python -m scripts.self_play.test_agentic_judge \\
        --from-interactions outputs/self_play/reinforce_smoke_*/interactions_*.jsonl \\
        --dry-run --n 20
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
from scripts.self_play.utils import (
    parse_assessor_answer,
    parse_injector_compact,
    split_sentences,
    find_error_sentence_id,
)
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

    # Debug: print schema of first row
    if len(ds) > 0:
        row0 = dict(ds[0]) if hasattr(ds[0], "keys") else ds[0]
        print(f"  Parquet columns: {list(row0.keys())}")
        for k, v in row0.items():
            vtype = type(v).__name__
            if isinstance(v, dict):
                print(f"    {k}: dict with keys {list(v.keys())}")
            elif isinstance(v, list):
                print(f"    {k}: list[{len(v)}]")
            elif isinstance(v, str) and len(v) > 80:
                print(f"    {k}: str[{len(v)} chars]")
                try:
                    parsed = json.loads(v)
                    if isinstance(parsed, dict):
                        print(f"      -> parsed dict keys: {list(parsed.keys())}")
                        for ik, iv in parsed.items():
                            print(f"         {ik}: {type(iv).__name__} = {str(iv)[:100]}")
                except json.JSONDecodeError:
                    pass
            else:
                print(f"    {k}: {vtype} = {str(v)[:120]}")

    examples = []
    for i, row in enumerate(ds):
        if i >= n:
            break
        ex = dict(row) if hasattr(row, "keys") else row

        # Robustly extract nested fields — HF datasets may return Arrow structs
        extra_info = ex.get("extra_info", {})
        if isinstance(extra_info, str):
            try:
                extra_info = json.loads(extra_info)
            except json.JSONDecodeError:
                extra_info = {}
        if not isinstance(extra_info, dict):
            extra_info = {}

        reward_model = ex.get("reward_model", {})
        if isinstance(reward_model, str):
            try:
                reward_model = json.loads(reward_model)
            except json.JSONDecodeError:
                reward_model = {}
        if not isinstance(reward_model, dict):
            reward_model = {}

        # --- Derive ground_truth and mode ---
        # The parquet extra_info may lack 'mode' and 'error_sentence_id'.
        # We reconstruct them from the available fields:
        #   - mode: 'benign' if no incorrect_note/error_sentence, else 'error_injection'
        #   - error_sentence_id: computed via find_error_sentence_id()
        #   - ground_truth: 'CORRECT' for benign, str(sid) for error

        mode = extra_info.get("mode", "")
        if not isinstance(mode, str):
            mode = str(mode) if mode is not None else ""

        # Infer mode if missing
        if not mode:
            if extra_info.get("incorrect_note") or extra_info.get("error_sentence"):
                mode = "error_injection"
            else:
                mode = "benign"

        # Compute error_sentence_id if missing
        error_sid = extra_info.get("error_sentence_id")
        if error_sid is None and mode == "error_injection":
            error_text = extra_info.get("error_sentence", "")
            note_text = extra_info.get("incorrect_note", extra_info.get("correct_note", ""))
            if error_text and note_text:
                error_sid = find_error_sentence_id(note_text, error_text)
            # Store back into extra_info so downstream code can use it
            extra_info["error_sentence_id"] = error_sid
            extra_info["mode"] = mode

        # Derive ground_truth
        if mode == "benign":
            gt = "CORRECT"
        elif error_sid is not None:
            gt = str(error_sid)
        else:
            # Fallback: check reward_model
            gt_raw = reward_model.get("ground_truth", "CORRECT")
            if isinstance(gt_raw, dict):
                gt = "CORRECT"  # nested dict — not a real label
            else:
                gt = str(gt_raw) if gt_raw is not None else "CORRECT"

        # Also check interaction_kwargs as last resort
        interaction_kwargs = ex.get("interaction_kwargs", {})
        if isinstance(interaction_kwargs, str):
            try:
                interaction_kwargs = json.loads(interaction_kwargs)
            except json.JSONDecodeError:
                interaction_kwargs = {}
        if isinstance(interaction_kwargs, dict):
            if mode in ("", "unknown") and "mode" in interaction_kwargs:
                mode = str(interaction_kwargs["mode"])
            if gt == "CORRECT" and mode != "benign" and "ground_truth" in interaction_kwargs:
                gt_ik = interaction_kwargs["ground_truth"]
                if gt_ik is not None and str(gt_ik) != "CORRECT":
                    gt = str(gt_ik)

        note_id = str(extra_info.get("note_id", f"test-{i}"))

        examples.append({
            "data_source": str(ex.get("data_source", "medec_selfplay")),
            "prompt": ex.get("prompt", []),
            "ground_truth": gt,
            "extra_info": extra_info,
            "mode": mode,
            "note_id": note_id,
        })

    return examples


# =============================================================================
# Load from real game_interactions.jsonl
# =============================================================================

def _load_parquet_note_map(parquet_path: str) -> dict:
    """Load note_id → {correct_note, incorrect_note, error_sentence, corrected_sentence, error_type}
    from a verl Parquet dataset.  Returns empty dict on failure.
    """
    note_map: dict = {}
    if not parquet_path:
        return note_map
    p = Path(parquet_path)
    if not p.exists():
        # Try to find the parquet automatically
        candidates = sorted(
            (PROJECT_ROOT / "data_processed" / "self_play").glob("*.parquet")
        )
        if candidates:
            p = candidates[0]
            print(f"  Auto-detected parquet: {p}")
        else:
            print(f"  WARNING: Parquet not found at {parquet_path} and no auto-detect hit")
            return note_map
    try:
        import pandas as pd
        df = pd.read_parquet(p)
        print(f"  Loading note context from {p} ({len(df)} rows) ...")
        for _, row in df.iterrows():
            ei = row.get("extra_info", {})
            if isinstance(ei, str):
                try:
                    ei = json.loads(ei)
                except json.JSONDecodeError:
                    ei = {}
            if not isinstance(ei, dict):
                ei = {}
            nid = str(ei.get("note_id", ""))
            if not nid or nid in note_map:
                continue
            note_map[nid] = {
                "correct_note": ei.get("correct_note", ""),
                "incorrect_note": ei.get("incorrect_note", ""),
                "error_sentence": ei.get("error_sentence", ""),
                "corrected_sentence": ei.get("corrected_sentence", ""),
                "error_type": ei.get("error_type", ""),
            }
        print(f"  Loaded {len(note_map)} unique note contexts from parquet")
    except Exception as e:
        print(f"  WARNING: Could not load parquet ({e})")
    return note_map


def load_interactions(
    path: str,
    n: int = 50,
    role_filter: str = "assessor",
    parquet_path: str = None,
) -> tuple:
    """Load real model responses from a game_interactions.jsonl produced by training.

    For assessor role: joins each assessor record with its matching injector
    record (same note_id) to reconstruct the full context needed by the UMLS
    judge pipeline:
        - error_sentence: the injector's modified sentence text
        - corrected_sentence: the original sentence from the parquet correct_note
        - correct_note / incorrect_note: full notes from parquet

    Returns (examples, responses) in the same format expected by run_dry_test /
    run_full_test, so all existing test logic works unchanged.

    Args:
        path: Path (or glob) to one or more interactions_*.jsonl files.
        n: Max records to load.
        role_filter: Only load records with this role ("assessor" or "injector").
        parquet_path: Path to the verl Parquet dataset for note context.
                      Auto-detected from data_processed/self_play/ if None.
    """
    import glob

    # Support glob patterns
    paths = sorted(glob.glob(path)) if "*" in path or "?" in path else [path]
    if not paths:
        raise FileNotFoundError(f"No files matched: {path}")

    # Load ALL records (both roles) so we can join injector ↔ assessor
    all_records: list = []
    for p in paths:
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    all_records.append(rec)
                except json.JSONDecodeError:
                    continue

    # Split by role
    assessor_records = [r for r in all_records if r.get("role", "assessor") == "assessor"]
    injector_records = [r for r in all_records if r.get("role") == "injector"]

    # Build injector lookup: note_id → most-recent injector record
    injector_by_note: dict = {}
    for rec in injector_records:
        nid = str(rec.get("note_id", ""))
        injector_by_note[nid] = rec  # last one wins

    # Diagnostic: sample note_ids
    sample_assess_nids = list({str(r.get("note_id", "")) for r in assessor_records[:20]})
    sample_inj_nids = list(injector_by_note.keys())[:5]
    print(
        f"  Interactions file: {len(all_records)} total records "
        f"(assessor={len(assessor_records)}, injector={len(injector_records)})"
    )
    print(f"  Sample assessor note_ids: {sample_assess_nids[:5]}")
    print(f"  Sample injector note_ids: {sample_inj_nids[:5]}")

    # Deduplicate by note_id — take the last record per unique note (most recent training step).
    # Without this, GRPO rollouts repeat the same note up to n_steps times, inflating counts
    # and making metrics meaningless (e.g. 100% detection on 6 notes × 10 rollouts each).
    role_records = [r for r in all_records if r.get("role", "assessor") == role_filter]
    by_note: dict = {}
    anon_count = 0
    for rec in role_records:
        nid = str(rec.get("note_id", ""))
        if not nid:
            nid = f"__anon_{anon_count}"
            anon_count += 1
        by_note[nid] = rec  # last seen wins (most recent training step)
    target_records = list(by_note.values())[:n]
    print(f"  Using {len(target_records)} '{role_filter}' records (deduplicated from {len(role_records)}, n={n})")

    # Load note context from parquet (for UMLS judge enrichment)
    note_map = _load_parquet_note_map(parquet_path or "")

    # Diagnostic: check note_id overlap between interactions and parquet
    parquet_nids = set(note_map.keys())
    interact_nids = {str(r.get("note_id", "")) for r in target_records}
    overlap = parquet_nids & interact_nids
    print(
        f"  Parquet note_ids: {len(parquet_nids)} | "
        f"Interactions note_ids: {len(interact_nids)} | "
        f"Overlap: {len(overlap)}"
    )
    if not overlap and parquet_nids and interact_nids:
        print(f"  WARNING: No note_id overlap! Parquet samples: {list(parquet_nids)[:3]} "
              f"vs Interactions samples: {list(interact_nids)[:3]}")
        print("  Judge will fall back to parquet's first error_sentence as best-effort.")

    # If no overlap, build a fallback pool from parquet for error records
    fallback_parquet_entries = list(note_map.values()) if not overlap else []

    examples, responses = [], []
    n_enriched = 0

    for i, rec in enumerate(target_records):
        note_id = str(rec.get("note_id", f"rec-{i}"))
        mode = rec.get("mode", "error_injection")
        gt = rec.get("ground_truth", "CORRECT")
        model_response = rec.get("model_response", "")
        data_source = rec.get("data_source", "medec_selfplay")

        extra_info: dict = {
            "role": role_filter,
            "mode": mode,
            "note_id": note_id,
            "error_type": rec.get("error_type", ""),
            "_logged_rule_reward": rec.get("reward"),
            "_logged_outcome": rec.get("outcome"),
        }

        # error_sentence_id from ground_truth for rule scorer
        error_sid = None
        if mode not in ("benign", "") and gt not in ("CORRECT", "", None):
            try:
                error_sid = int(gt)
                extra_info["error_sentence_id"] = error_sid
            except (ValueError, TypeError):
                pass

        # Enrich assessor records with full note context for the UMLS judge.
        # Always fetch correct_note — needed for benign false-positive verification
        # (judge checks if the sentence the model flagged is actually clinically wrong).
        if role_filter == "assessor":
            # Try exact note_id match first, then fallback pool
            parquet_ctx = note_map.get(note_id, {})
            if not parquet_ctx and fallback_parquet_entries:
                # Use a cycling entry from the parquet as best-effort context
                parquet_ctx = fallback_parquet_entries[i % len(fallback_parquet_entries)]

            correct_note = parquet_ctx.get("correct_note", "")
            if correct_note:
                extra_info["correct_note"] = correct_note

        # Error-specific enrichment (modified sentence + original sentence)
        if role_filter == "assessor" and mode not in ("benign", ""):
            incorrect_note = parquet_ctx.get("incorrect_note", "")

            # Modified sentence: parse the injector's compact output "N. <text>"
            modified_sentence = ""
            injector_rec = injector_by_note.get(note_id, {})
            if injector_rec:
                inj_resp = injector_rec.get("model_response", "")
                _, modified_sentence = parse_injector_compact(inj_resp)
                modified_sentence = modified_sentence or ""
            # Fallback to parquet error_sentence
            if not modified_sentence:
                modified_sentence = parquet_ctx.get("error_sentence", "")

            # Original (corrected) sentence at error_sentence_id
            corrected_sentence = parquet_ctx.get("corrected_sentence", "")
            if not corrected_sentence and correct_note and error_sid is not None:
                sents = split_sentences(correct_note)
                if 1 <= error_sid <= len(sents):
                    corrected_sentence = sents[error_sid - 1]

            if incorrect_note:
                extra_info["incorrect_note"] = incorrect_note
            if modified_sentence:
                extra_info["error_sentence"] = modified_sentence
            if corrected_sentence:
                extra_info["corrected_sentence"] = corrected_sentence

            if modified_sentence or corrected_sentence:
                n_enriched += 1

        examples.append({
            "data_source": data_source,
            "prompt": [],
            "ground_truth": str(gt) if gt is not None else "CORRECT",
            "extra_info": extra_info,
            "mode": mode,
            "note_id": note_id,
        })
        responses.append(model_response)

    print(
        f"  Judge context enriched: {n_enriched}/{len(target_records)} records "
        f"have modified_sentence for UMLS analysis"
    )

    return examples, responses


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
        logged = ex["extra_info"].get("_logged_rule_reward")
        logged_str = f"  logged={logged:+.2f}" if logged is not None else ""
        print(f"  [{i:3d}] {str(ex['mode']):16s} gt={str(ex['ground_truth']):8s} "
              f"pred={label:8s}({pred_sid}) rule={rule_score:+.2f}{logged_str}")

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
        cleanup as agentic_cleanup,
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

    # Pre-flight: check judge server + verify POST endpoint
    import aiohttp
    try:
        async with aiohttp.ClientSession() as session:
            # Check /v1/models is reachable
            async with session.get(
                JUDGE_URL.replace("/chat/completions", "/models"),
                timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status == 200:
                    models_data = await resp.json()
                    model_ids = [m.get("id", "?") for m in models_data.get("data", [])]
                    print(f"  Judge server: ONLINE (models: {model_ids})")
                else:
                    print(f"  Judge server: /v1/models returned {resp.status}")

            # Verify POST to /v1/chat/completions works (minimal probe)
            probe_payload = {
                "model": JUDGE_MODEL,
                "messages": [{"role": "user", "content": "test"}],
                "max_tokens": 32,
            }
            async with session.post(
                JUDGE_URL,
                json=probe_payload,
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status == 200:
                    print(f"  POST {JUDGE_URL}: OK")
                elif resp.status == 405:
                    body = await resp.text()
                    print(f"  POST {JUDGE_URL}: 405 Method Not Allowed")
                    print(f"    Response: {body[:200]}")
                    print("  >>> Your vLLM server may not support /v1/chat/completions.")
                    print("  >>> Restart it with: python -m vllm.entrypoints.openai.api_server")
                    print("  >>> Or set JUDGE_VLLM_URL to the correct endpoint.")
                    # Try /v1/completions as fallback
                    alt_url = JUDGE_URL.replace("/chat/completions", "/completions")
                    async with session.post(
                        alt_url,
                        json={"model": JUDGE_MODEL, "prompt": "test", "max_tokens": 1},
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as alt_resp:
                        if alt_resp.status == 200:
                            print(f"  >>> /v1/completions works! Set JUDGE_VLLM_URL={alt_url}")
                        else:
                            print(f"  >>> /v1/completions also returned {alt_resp.status}")
                else:
                    body = await resp.text()
                    print(f"  POST {JUDGE_URL}: {resp.status} — {body[:200]}")
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

        # Agentic score (async — avoids threading issues from sync wrapper)
        agentic_result = await async_compute_score(
            data_source=ex["data_source"],
            solution_str=resp,
            ground_truth=ex["ground_truth"],
            extra_info=ex["extra_info"],
        )
        # Extract score from dict (veRL Reward Loop returns {"score": float})
        agentic_score = agentic_result["score"] if isinstance(agentic_result, dict) else agentic_result

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

        print(f"  [{i:3d}] {str(ex['mode']):16s} gt={str(ex['ground_truth']):8s} "
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

    # Cleanup aiohttp session
    await agentic_cleanup()

    return {"status": "ok", "n_tested": n, "results": results}


# =============================================================================
# Standalone entity extraction test
# =============================================================================

async def test_extraction_only(examples: List[Dict]) -> None:
    """Test just Step 1 (entity extraction) on a few examples."""
    from scripts.self_play.agentic_reward import _extract_entities, _identify_changed_sentences, cleanup as agentic_cleanup

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

    await agentic_cleanup()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Test the Agentic UMLS Judge")
    parser.add_argument("--n", type=int, default=50, help="Number of examples to test")
    parser.add_argument("--data", type=str,
                        default=str(PROJECT_ROOT / "data_processed" / "self_play" / "train.parquet"),
                        help="Path to Parquet dataset (used when --from-interactions is not set)")
    parser.add_argument("--from-interactions", type=str, default=None,
                        help="Path (or glob) to game_interactions.jsonl from a training run. "
                             "Uses real model responses instead of simulated ones.")
    parser.add_argument("--role", type=str, default="assessor",
                        help="Role to load when using --from-interactions (default: assessor)")
    parser.add_argument("--parquet", type=str, default=None,
                        help="Path to verl Parquet dataset for note context enrichment "
                             "(auto-detected from data_processed/self_play/ if not set)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Test plumbing only (no UMLS/LLM calls)")
    parser.add_argument("--extraction-only", action="store_true",
                        help="Test Step 1 (entity extraction) only")
    args = parser.parse_args()

    # ── Load data ──────────────────────────────────────────────────────────────
    if args.from_interactions:
        print(f"Loading real model outputs from {args.from_interactions} (role={args.role}) ...")
        examples, responses = load_interactions(
            args.from_interactions,
            n=args.n,
            role_filter=args.role,
            parquet_path=args.parquet,
        )
        print(f"  {len(examples)} records loaded")
    else:
        print(f"Loading {args.n} examples from {args.data} (simulated responses) ...")
        examples = load_parquet_examples(args.data, n=args.n)
        print(f"Loaded {len(examples)} examples")
        responses = simulate_assessor_responses(examples)

    # Show mode distribution
    modes: dict = {}
    for ex in examples:
        modes[ex["mode"]] = modes.get(ex["mode"], 0) + 1
    print(f"Mode distribution: {modes}")

    # ── Run selected test mode ─────────────────────────────────────────────────
    if args.dry_run:
        run_dry_test(examples, responses)
    elif args.extraction_only:
        asyncio.run(test_extraction_only(examples))
    else:
        asyncio.run(run_full_test(examples, responses))


if __name__ == "__main__":
    main()
