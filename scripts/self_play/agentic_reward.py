"""
Agentic UMLS Judge reward function for MedSeRL self-play.

Wraps the rule-based reward_function.compute_score with an evidence-based
3-step pipeline:
  Step 1 (Extract): LLM extracts medical entities from the changed sentence.
  Step 2 (Retrieve): Async UMLS/RxNorm lookup for each entity.
  Step 3 (Adjudicate): LLM produces verdict + CUI citations given evidence.

Hybrid scoring formula:
    R = rule_weight * rule_score + umls_weight * umls_score

If UMLS/judge fails, gracefully degrades to pure rule-based scoring.

Designed for verl's async Reward Loop:
    https://verl.readthedocs.io/en/latest/preparation/reward_function.html

Usage in verl config:
    custom_reward_function:
        path: scripts/self_play/agentic_reward
        name: compute_score
"""

import asyncio
import json
import logging
import os
import re
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import aiohttp

from scripts.self_play.reward_function import (
    compute_score as rule_compute_score,
    REWARD_EXACT,
    REWARD_MISS,
    REWARD_PARTIAL,
    FORMAT_BONUS,
)
from scripts.self_play.utils import (
    parse_assessor_answer,
    strip_thinking,
    split_sentences,
    diff_sentences,
    number_sentences,
    parse_numbered_sentences,
)
from scripts.self_play.judge_prompts import (
    get_extraction_system_prompt,
    get_extraction_user_template,
    get_adjudication_system_prompt,
    get_adjudication_user_template,
    get_model_params,
    format_evidence_for_prompt,
)
from scripts.self_play.umls_async import (
    UMLSEvidence,
    gather_evidence_batch,
    clear_cache as umls_clear_cache,
)

logger = logging.getLogger(__name__)

# =============================================================================
# Configuration (from environment)
# =============================================================================

# vLLM judge server endpoint (same model as actors, separate port)
JUDGE_URL = os.getenv("JUDGE_VLLM_URL", "http://localhost:8001/v1/chat/completions")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "Qwen/Qwen3-4B")

# Hybrid scoring weights (must sum to 1.0)
RULE_WEIGHT = float(os.getenv("RULE_WEIGHT", "0.6"))
UMLS_WEIGHT = float(os.getenv("UMLS_WEIGHT", "0.4"))

# Max entities per extraction to avoid UMLS API overload
MAX_ENTITIES = int(os.getenv("MAX_ENTITIES_PER_SENTENCE", "10"))

# Timeouts
LLM_TIMEOUT = float(os.getenv("JUDGE_LLM_TIMEOUT", "30"))
TOTAL_TIMEOUT = float(os.getenv("JUDGE_TOTAL_TIMEOUT", "60"))

# Logging
LOG_DIR = Path(__file__).parent.parent.parent / "results" / "self_play" / "judge_traces"
LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_FILE = LOG_DIR / f"judge_trace_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

# Module-level aiohttp session (lazy init)
_session: Optional[aiohttp.ClientSession] = None


async def _get_session() -> aiohttp.ClientSession:
    """Lazy-init a module-level aiohttp session."""
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession()
    return _session


# =============================================================================
# Step 1: LLM Entity Extraction
# =============================================================================

async def _extract_entities(sentence: str) -> List[Dict[str, str]]:
    """Call the judge LLM to extract medical entities from a sentence.

    Returns:
        List of {"name": str, "type": str} dicts.
    """
    session = await _get_session()
    params = get_model_params("extraction")

    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": get_extraction_system_prompt()},
            {"role": "user", "content": get_extraction_user_template().format(sentence=sentence)},
        ],
        "temperature": params.get("temperature", 0.1),
        "max_tokens": params.get("max_tokens", 256),
        "top_p": params.get("top_p", 0.95),
    }

    try:
        async with session.post(
            JUDGE_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=LLM_TIMEOUT),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"Judge LLM extraction returned {resp.status}")
                return []
            data = await resp.json()
            content = data["choices"][0]["message"]["content"].strip()

            # Strip thinking tags if present
            _, content = strip_thinking(content)

            # Strip markdown fences
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

            entities = json.loads(content)
            if not isinstance(entities, list):
                return []
            return entities[:MAX_ENTITIES]
    except json.JSONDecodeError:
        logger.debug(f"Entity extraction returned invalid JSON: {content[:200]}")
        return []
    except Exception as e:
        logger.debug(f"Entity extraction failed: {e}")
        return []


# =============================================================================
# Step 2: UMLS/RxNorm Evidence Retrieval (delegated to umls_async.py)
# =============================================================================

async def _retrieve_evidence(entities: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    """Query UMLS/RxNorm for each extracted entity.

    Returns:
        List of evidence dicts (from UMLSEvidence.to_dict()).
    """
    if not entities:
        return []

    session = await _get_session()
    try:
        evidence_objs = await gather_evidence_batch(session, entities)
        return [ev.to_dict() for ev in evidence_objs]
    except Exception as e:
        logger.debug(f"Evidence retrieval failed: {e}")
        return []


# =============================================================================
# Step 3: LLM Adjudication
# =============================================================================

async def _adjudicate(
    original_sentence: str,
    modified_sentence: str,
    assessor_prediction: str,
    evidence: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Call the judge LLM to produce a verdict with CUI citations.

    Returns:
        Dict with keys: verdict, score, reasoning, cuis_cited.
        On failure returns {"verdict": "ABSTAIN", "score": 0.0, ...}.
    """
    session = await _get_session()
    params = get_model_params("adjudication")
    evidence_str = format_evidence_for_prompt(evidence)

    user_content = get_adjudication_user_template().format(
        original_sentence=original_sentence,
        modified_sentence=modified_sentence,
        assessor_prediction=assessor_prediction,
        evidence_json=evidence_str,
    )

    payload = {
        "model": JUDGE_MODEL,
        "messages": [
            {"role": "system", "content": get_adjudication_system_prompt()},
            {"role": "user", "content": user_content},
        ],
        "temperature": params.get("temperature", 0.1),
        "max_tokens": params.get("max_tokens", 512),
        "top_p": params.get("top_p", 0.95),
    }

    default_result = {
        "verdict": "ABSTAIN",
        "score": 0.0,
        "reasoning": "Judge failed to produce a verdict.",
        "cuis_cited": [],
    }

    try:
        async with session.post(
            JUDGE_URL,
            json=payload,
            timeout=aiohttp.ClientTimeout(total=LLM_TIMEOUT),
        ) as resp:
            if resp.status != 200:
                logger.warning(f"Judge LLM adjudication returned {resp.status}")
                return default_result
            data = await resp.json()
            content = data["choices"][0]["message"]["content"].strip()

            # Strip thinking tags
            _, content = strip_thinking(content)

            # Strip markdown fences
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)

            verdict = json.loads(content)
            if not isinstance(verdict, dict):
                return default_result

            # Validate required fields
            v = verdict.get("verdict", "ABSTAIN").upper()
            if v not in ("PASS", "FAIL"):
                v = "ABSTAIN"

            score = float(verdict.get("score", 0.0))
            score = max(0.0, min(1.0, score))

            cuis = verdict.get("cuis_cited", [])
            if not isinstance(cuis, list):
                cuis = []

            # If no CUIs cited, force score to 0 (per prompt rules)
            if not cuis:
                score = 0.0

            return {
                "verdict": v,
                "score": score,
                "reasoning": verdict.get("reasoning", ""),
                "cuis_cited": cuis,
            }
    except json.JSONDecodeError:
        logger.debug(f"Adjudication returned invalid JSON: {content[:200]}")
        return default_result
    except Exception as e:
        logger.debug(f"Adjudication failed: {e}")
        return default_result


# =============================================================================
# Identify the changed sentence
# =============================================================================

def _identify_changed_sentences(extra_info: dict) -> Tuple[str, str]:
    """Extract the original and modified sentences from extra_info.

    Uses extra_info fields: correct_note, incorrect_note, error_sentence,
    error_sentence_id, mode.

    Returns:
        (original_sentence, modified_sentence) — empty strings if not found.
    """
    mode = extra_info.get("mode", "")
    original_sentence = ""
    modified_sentence = ""

    if mode == "benign":
        # For benign notes, both are the same — no modification
        return "", ""

    # Try to get sentences from the notes
    correct_note = extra_info.get("correct_note", "")
    incorrect_note = extra_info.get("incorrect_note", "")
    error_sid = extra_info.get("error_sentence_id")
    error_text = extra_info.get("error_sentence", "")

    if error_sid is not None and correct_note and incorrect_note:
        try:
            sid = int(error_sid)
            orig_sents = parse_numbered_sentences(correct_note)
            mod_sents = parse_numbered_sentences(incorrect_note)

            # Also try raw split if numbered parsing fails
            if not orig_sents:
                raw_orig = split_sentences(correct_note)
                orig_sents = {i + 1: s for i, s in enumerate(raw_orig)}
            if not mod_sents:
                raw_mod = split_sentences(incorrect_note)
                mod_sents = {i + 1: s for i, s in enumerate(raw_mod)}

            original_sentence = orig_sents.get(sid, "")
            modified_sentence = mod_sents.get(sid, "")
        except (ValueError, TypeError):
            pass

    # Fallback: use error_sentence field
    if not modified_sentence and error_text:
        modified_sentence = error_text

    return original_sentence, modified_sentence


# =============================================================================
# Full judge pipeline (async)
# =============================================================================

async def _judge_pipeline(
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
) -> Tuple[float, Dict[str, Any]]:
    """Run the full 3-step judge pipeline.

    Returns:
        (umls_score, trace_dict) where umls_score ∈ [0, 1].
    """
    trace = {
        "step1_entities": [],
        "step2_evidence": [],
        "step3_verdict": {},
        "umls_score": 0.0,
        "skipped": False,
        "skip_reason": "",
    }

    mode = extra_info.get("mode", "")

    # For benign notes, the judge has less to verify — skip the full pipeline
    # and just check if the assessor correctly said CORRECT
    if mode == "benign":
        label, _ = parse_assessor_answer(solution_str)
        if label == "CORRECT":
            trace["umls_score"] = 1.0
            trace["skipped"] = True
            trace["skip_reason"] = "benign_correct"
        else:
            trace["umls_score"] = 0.0
            trace["skipped"] = True
            trace["skip_reason"] = "benign_false_positive"
        return trace["umls_score"], trace

    # Identify the changed sentence pair
    original_sent, modified_sent = _identify_changed_sentences(extra_info)

    if not original_sent and not modified_sent:
        trace["skipped"] = True
        trace["skip_reason"] = "no_sentence_pair"
        return 0.0, trace

    # Use the sentence that was actually changed for entity extraction
    target_sentence = modified_sent or original_sent

    # Step 1: Extract entities
    entities = await _extract_entities(target_sentence)
    trace["step1_entities"] = entities

    if not entities:
        # Extraction failed — fall back to extracting from original too
        if original_sent and original_sent != target_sentence:
            entities = await _extract_entities(original_sent)
            trace["step1_entities"] = entities

    # Step 2: Retrieve UMLS evidence
    evidence = await _retrieve_evidence(entities)
    trace["step2_evidence"] = evidence

    # Step 3: Adjudicate
    label, pred_sid = parse_assessor_answer(solution_str)
    assessor_prediction = f"{label}" + (f" (sentence {pred_sid})" if pred_sid else "")

    verdict = await _adjudicate(
        original_sentence=original_sent,
        modified_sentence=modified_sent,
        assessor_prediction=assessor_prediction,
        evidence=evidence,
    )
    trace["step3_verdict"] = verdict

    # Convert verdict to a score
    if verdict["verdict"] == "PASS":
        umls_score = verdict["score"]
    elif verdict["verdict"] == "FAIL":
        umls_score = 1.0 - verdict["score"]
    else:
        # ABSTAIN — no opinion, score 0
        umls_score = 0.0

    trace["umls_score"] = umls_score
    return umls_score, trace


# =============================================================================
# Log trace to JSONL
# =============================================================================

def _log_trace(
    data_source: str,
    ground_truth: str,
    rule_score: float,
    umls_score: float,
    final_score: float,
    trace: Dict[str, Any],
    extra_info: dict,
):
    """Append a judge trace entry to the JSONL log."""
    entry = {
        "timestamp": datetime.now().isoformat(),
        "data_source": str(data_source),
        "ground_truth": ground_truth,
        "mode": extra_info.get("mode", ""),
        "note_id": extra_info.get("note_id", ""),
        "rule_score": rule_score,
        "umls_score": umls_score,
        "final_score": final_score,
        "rule_weight": RULE_WEIGHT,
        "umls_weight": UMLS_WEIGHT,
        **trace,
    }
    try:
        with open(_LOG_FILE, "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")
    except Exception:
        pass


# =============================================================================
# Public API: compute_score (verl-compatible)
# =============================================================================

def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Hybrid reward: rule-based + Agentic UMLS Judge.

    Drop-in replacement for reward_function.compute_score.
    Called by verl's RewardManager after each rollout.

    Scoring formula:
        R = RULE_WEIGHT * rule_score + UMLS_WEIGHT * umls_score

    If the async judge pipeline fails, falls back to pure rule-based score.

    Args:
        data_source (str): Dataset identifier.
        solution_str (str): Model's generated response.
        ground_truth (str): "CORRECT" or str(sentence_id).
        extra_info (dict): Contains correct_note, incorrect_note,
            error_sentence, error_sentence_id, mode, note_id, etc.

    Returns:
        float: Hybrid reward score.
    """
    # Step A: Always compute the rule-based score (fast, synchronous)
    rule_score = rule_compute_score(data_source, solution_str, ground_truth, extra_info)

    # If UMLS weight is 0, short-circuit
    if UMLS_WEIGHT <= 0:
        return rule_score

    extra_info = extra_info or {}

    # Step B: Run the async judge pipeline
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # We're inside an existing event loop (verl's async reward)
            # Create a future and run it
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(asyncio.run, _judge_pipeline(solution_str, ground_truth, extra_info))
                umls_score, trace = future.result(timeout=TOTAL_TIMEOUT)
        else:
            umls_score, trace = loop.run_until_complete(
                _judge_pipeline(solution_str, ground_truth, extra_info)
            )
    except RuntimeError:
        # No event loop — create one
        try:
            umls_score, trace = asyncio.run(
                _judge_pipeline(solution_str, ground_truth, extra_info)
            )
        except Exception as e:
            logger.warning(f"Judge pipeline failed: {e}")
            umls_score, trace = 0.0, {"skipped": True, "skip_reason": f"error: {e}"}
    except Exception as e:
        logger.warning(f"Judge pipeline failed: {e}")
        umls_score, trace = 0.0, {"skipped": True, "skip_reason": f"error: {e}"}

    # Step C: Compute hybrid score
    final_score = RULE_WEIGHT * rule_score + UMLS_WEIGHT * umls_score

    # Step D: Log the trace
    _log_trace(
        data_source=data_source,
        ground_truth=ground_truth,
        rule_score=rule_score,
        umls_score=umls_score,
        final_score=final_score,
        trace=trace,
        extra_info=extra_info,
    )

    return final_score


# =============================================================================
# Async compute_score (for verl async reward mode)
# =============================================================================

async def async_compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """Async version of compute_score for verl's async Reward Loop.

    Use this when verl supports native async reward functions.
    """
    rule_score = rule_compute_score(data_source, solution_str, ground_truth, extra_info)

    if UMLS_WEIGHT <= 0:
        return rule_score

    extra_info = extra_info or {}

    try:
        umls_score, trace = await asyncio.wait_for(
            _judge_pipeline(solution_str, ground_truth, extra_info),
            timeout=TOTAL_TIMEOUT,
        )
    except Exception as e:
        logger.warning(f"Async judge pipeline failed: {e}")
        umls_score = 0.0
        trace = {"skipped": True, "skip_reason": f"error: {e}"}

    final_score = RULE_WEIGHT * rule_score + UMLS_WEIGHT * umls_score

    _log_trace(
        data_source=data_source,
        ground_truth=ground_truth,
        rule_score=rule_score,
        umls_score=umls_score,
        final_score=final_score,
        trace=trace,
        extra_info=extra_info,
    )

    return final_score


# =============================================================================
# Cleanup
# =============================================================================

async def cleanup():
    """Close the aiohttp session. Call at training end."""
    global _session
    if _session and not _session.closed:
        await _session.close()
        _session = None
    umls_clear_cache()
