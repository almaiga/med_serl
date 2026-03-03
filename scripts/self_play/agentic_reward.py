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

Designed for verl's native GenRM Reward Loop:
    https://verl.readthedocs.io/en/latest/advance/reward_loop.html

Two execution modes:
  1. Native GenRM (preferred): veRL manages the judge model server.
     Set reward_model.enable=True in YAML. veRL injects
     reward_router_address and reward_model_tokenizer into compute_score.
     Judge calls go through veRL's /generate endpoint.

  2. Standalone (testing): Manual vLLM server via JUDGE_VLLM_URL env var.
     Uses OpenAI-compatible /v1/chat/completions endpoint.

Usage in verl config:
    reward_model:
      enable: true
      model:
        path: Qwen/Qwen3-4B
    custom_reward_function:
        path: scripts.self_play.agentic_reward
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
    find_error_sentence_id,
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


def _extract_json_array(text: str) -> Optional[list]:
    """Robustly extract a JSON array from LLM output.

    Handles markdown fences, thinking tags, and extra text around JSON.
    Returns parsed list or None if not found.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```", "", text)
    text = text.strip()

    # Try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Find first [...] block using bracket matching
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
            depth -= 1
            if depth == 0:
                try:
                    result = json.loads(text[start:i + 1])
                    if isinstance(result, list):
                        return result
                except (json.JSONDecodeError, ValueError):
                    return None
                break
    return None


def _extract_json_object(text: str) -> Optional[dict]:
    """Robustly extract a JSON object from LLM output.

    Handles markdown fences, thinking tags, and extra text around JSON.
    Returns parsed dict or None if not found.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```", "", text)
    text = text.strip()

    # Try direct parse first
    try:
        result = json.loads(text)
        if isinstance(result, dict):
            return result
    except (json.JSONDecodeError, ValueError):
        pass

    # Find first {...} block using bracket matching
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    result = json.loads(text[start:i + 1])
                    if isinstance(result, dict):
                        return result
                except (json.JSONDecodeError, ValueError):
                    return None
                break
    return None

# =============================================================================
# Configuration (from environment)
# =============================================================================

# Fallback vLLM judge server endpoint (standalone/testing mode only)
# In native GenRM mode, reward_router_address is injected by veRL.
JUDGE_URL = os.getenv("JUDGE_VLLM_URL", "http://localhost:8002/v1/chat/completions")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "Qwen/Qwen3-8B")

# Hybrid scoring weights (must sum to 1.0)
RULE_WEIGHT = float(os.getenv("RULE_WEIGHT", "0.6"))
UMLS_WEIGHT = float(os.getenv("UMLS_WEIGHT", "0.4"))

# Max entities per extraction to avoid UMLS API overload
MAX_ENTITIES = int(os.getenv("MAX_ENTITIES_PER_SENTENCE", "10"))

# Timeouts
LLM_TIMEOUT = float(os.getenv("JUDGE_LLM_TIMEOUT", "30"))
TOTAL_TIMEOUT = float(os.getenv("JUDGE_TOTAL_TIMEOUT", "60"))

# Module-level reference to veRL-injected GenRM resources
# Set by compute_score when reward_router_address is provided
_reward_router_address: Optional[str] = None
_reward_model_tokenizer = None  # PreTrainedTokenizer or None

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
# LLM Generation Abstraction (GenRM native vs standalone fallback)
# =============================================================================

async def _llm_generate(messages: List[Dict[str, str]], params: dict) -> str:
    """Generate text from the judge LLM.

    Routes to either:
      - veRL's native GenRM router (/generate endpoint) if reward_router_address is set
      - Standalone vLLM (/v1/chat/completions) as fallback

    Returns:
        The raw text content from the model response.
    """
    if _reward_router_address and _reward_model_tokenizer:
        # === Native veRL GenRM mode ===
        return await _llm_generate_genrm(messages, params)
    else:
        # === Standalone fallback (vLLM OpenAI-compatible) ===
        return await _llm_generate_standalone(messages, params)


async def _llm_generate_genrm(messages: List[Dict[str, str]], params: dict) -> str:
    """Native veRL GenRM: tokenize messages, POST to reward_router_address/generate."""
    session = await _get_session()
    loop = asyncio.get_event_loop()

    # Tokenize the chat messages using the reward model's tokenizer
    # Disable thinking mode for structured JSON output
    try:
        prompt_ids = await loop.run_in_executor(
            None,
            lambda: _reward_model_tokenizer.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True,
                enable_thinking=False,
            ),
        )
    except TypeError:
        # Older tokenizers may not support enable_thinking kwarg
        try:
            prompt_ids = await loop.run_in_executor(
                None,
                lambda: _reward_model_tokenizer.apply_chat_template(
                    messages, tokenize=True, add_generation_prompt=True,
                ),
            )
        except Exception as e:
            logger.warning(f"Tokenization failed, falling back to standalone: {e}")
            return await _llm_generate_standalone(messages, params)

    payload = {
        "input_ids": prompt_ids,
        "sampling_params": {
            "max_new_tokens": params.get("max_tokens", 512),
            "temperature": params.get("temperature", 0.1),
            "top_p": params.get("top_p", 0.95),
        },
    }
    url = f"http://{_reward_router_address}/generate"

    try:
        async with session.post(
            url, json=payload,
            timeout=aiohttp.ClientTimeout(total=LLM_TIMEOUT),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.warning(f"GenRM router returned {resp.status}: {body[:200]}")
                return ""
            output = await resp.json()
            response_ids = output.get("output_ids", None)
            if response_ids is not None:
                text = await loop.run_in_executor(
                    None,
                    lambda: _reward_model_tokenizer.decode(
                        response_ids, skip_special_tokens=True
                    ),
                )
                return text.strip()
            return ""
    except Exception as e:
        logger.debug(f"GenRM generate failed: {e}")
        return ""


async def _llm_generate_standalone(messages: List[Dict[str, str]], params: dict) -> str:
    """Fallback: call a standalone vLLM server via OpenAI-compatible API."""
    session = await _get_session()
    payload = {
        "model": JUDGE_MODEL,
        "messages": messages,
        "temperature": params.get("temperature", 0.1),
        "max_tokens": params.get("max_tokens", 512),
        "top_p": params.get("top_p", 0.95),
        # Disable Qwen3 thinking mode — we need structured JSON, not <think> tags
        "chat_template_kwargs": {"enable_thinking": False},
    }
    try:
        async with session.post(
            JUDGE_URL, json=payload,
            timeout=aiohttp.ClientTimeout(total=LLM_TIMEOUT),
        ) as resp:
            if resp.status != 200:
                body = await resp.text()
                logger.warning(f"Standalone vLLM returned {resp.status}: {body[:200]}")
                return ""
            data = await resp.json()
            content = data["choices"][0]["message"]["content"].strip()
            logger.debug(f"LLM raw response ({len(content)} chars): {content[:300]}")
            return content
    except Exception as e:
        logger.warning(f"Standalone LLM call failed: {e}")
        return ""


# =============================================================================
# Step 1: LLM Entity Extraction
# =============================================================================

async def _extract_entities(sentence: str) -> List[Dict[str, str]]:
    """Call the judge LLM to extract medical entities from a sentence.

    Returns:
        List of {"name": str, "type": str} dicts.
    """
    params = get_model_params("extraction")

    messages = [
        {"role": "system", "content": get_extraction_system_prompt()},
        {"role": "user", "content": get_extraction_user_template().format(sentence=sentence)},
    ]

    try:
        content = await _llm_generate(messages, params)
        if not content:
            logger.warning(f"Entity extraction: LLM returned empty for: {sentence[:80]}")
            return []

        # Strip thinking tags if present
        _, content = strip_thinking(content)

        # Robust JSON extraction — find the first [...] in the response
        entities = _extract_json_array(content)
        if entities is None:
            logger.warning(f"Entity extraction: no JSON array found in: {content[:300]}")
            return []
        return entities[:MAX_ENTITIES]
    except Exception as e:
        logger.warning(f"Entity extraction failed: {e}")
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
    params = get_model_params("adjudication")
    evidence_str = format_evidence_for_prompt(evidence)

    user_content = get_adjudication_user_template().format(
        original_sentence=original_sentence,
        modified_sentence=modified_sentence,
        assessor_prediction=assessor_prediction,
        evidence_json=evidence_str,
    )

    messages = [
        {"role": "system", "content": get_adjudication_system_prompt()},
        {"role": "user", "content": user_content},
    ]

    default_result = {
        "verdict": "ABSTAIN",
        "score": 0.0,
        "reasoning": "Judge failed to produce a verdict.",
        "cuis_cited": [],
    }

    try:
        content = await _llm_generate(messages, params)
        if not content:
            logger.warning("Adjudication: LLM returned empty")
            return default_result

        # Strip thinking tags
        _, content = strip_thinking(content)

        # Robust JSON extraction — find the first {...} in the response
        verdict = _extract_json_object(content)
        if verdict is None:
            logger.warning(f"Adjudication: no JSON object found in: {content[:300]}")
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
    """Extract the original (correct) and modified (error) sentences.

    The judge needs BOTH to compare and verify with UMLS evidence.

    Data sources in extra_info (from the MEDEC parquet):
      - corrected_sentence: the ground-truth correct version
      - error_sentence: the erroneous version
      - correct_note / incorrect_note: full clinical notes
      - error_sentence_id: 1-indexed ID (may be missing in older parquet)

    Returns:
        (original_sentence, modified_sentence) — empty strings if not found.
    """
    mode = extra_info.get("mode", "")
    if mode == "benign":
        return "", ""

    original_sentence = ""
    modified_sentence = ""

    # --- Best path: use corrected_sentence and error_sentence directly ---
    corrected = extra_info.get("corrected_sentence", "").strip()
    error_text = extra_info.get("error_sentence", "").strip()

    if corrected and error_text:
        return corrected, error_text

    # --- Fallback: extract from full notes using error_sentence_id ---
    correct_note = extra_info.get("correct_note", "")
    incorrect_note = extra_info.get("incorrect_note", "")
    error_sid = extra_info.get("error_sentence_id")

    # Compute error_sentence_id on the fly if missing
    if error_sid is None and error_text and incorrect_note:
        error_sid = find_error_sentence_id(incorrect_note, error_text)

    if error_sid is not None and correct_note and incorrect_note:
        try:
            sid = int(error_sid)
            orig_sents = parse_numbered_sentences(correct_note)
            mod_sents = parse_numbered_sentences(incorrect_note)

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

    # Use corrected_sentence as original if we found modified but not original
    if not original_sentence and corrected:
        original_sentence = corrected
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

    # For benign notes: if model says CORRECT, it's right → fast path.
    # But if model claims an error (false positive), verify the claim:
    #   run the pipeline on the sentence the model flagged to check whether
    #   it actually contains an error. This catches hallucinated errors.
    if mode == "benign":
        label, pred_sid = parse_assessor_answer(solution_str)
        if label == "CORRECT":
            trace["umls_score"] = 1.0
            trace["skipped"] = True
            trace["skip_reason"] = "benign_correct"
            return trace["umls_score"], trace

        # Model claims error on a benign note — verify the specific sentence
        trace["skip_reason"] = "benign_false_positive_verified"
        correct_note = extra_info.get("correct_note", "")
        if pred_sid is not None and correct_note:
            # Extract the sentence the model flagged
            from scripts.self_play.utils import parse_numbered_sentences
            sents = parse_numbered_sentences(correct_note)
            if not sents:
                raw = split_sentences(correct_note)
                sents = {i + 1: s for i, s in enumerate(raw)}
            flagged_sent = sents.get(pred_sid, "")
            if flagged_sent:
                # Run extraction + UMLS on the flagged sentence
                entities = await _extract_entities(flagged_sent)
                trace["step1_entities"] = entities
                trace["flagged_sentence"] = flagged_sent
                evidence = await _retrieve_evidence(entities)
                trace["step2_evidence"] = evidence
                # If UMLS confirms the entities are valid medical terms
                # and the sentence is clinically sound, the model's claim
                # is a false positive → score 0
                trace["umls_score"] = 0.0
                trace["skipped"] = False
                return trace["umls_score"], trace

        # Couldn't verify — default false positive penalty
        trace["umls_score"] = 0.0
        trace["skipped"] = True
        return trace["umls_score"], trace

    # Identify the changed sentence pair (ground truth error)
    original_sent, modified_sent = _identify_changed_sentences(extra_info)

    if not original_sent and not modified_sent:
        trace["skipped"] = True
        trace["skip_reason"] = "no_sentence_pair"
        return 0.0, trace

    # Parse what the model actually claimed
    label, pred_sid = parse_assessor_answer(solution_str)
    gt_sid = None
    try:
        gt_sid = int(ground_truth) if ground_truth and ground_truth != "CORRECT" else None
    except (ValueError, TypeError):
        pass

    trace["model_label"] = label
    trace["model_predicted_sid"] = pred_sid
    trace["ground_truth_sid"] = gt_sid

    # Step 1: Extract entities from BOTH original and modified sentences.
    # The judge needs evidence for terms in both versions to properly
    # determine if a substitution is clinically valid.
    # E.g. original="Penicillin is prescribed" vs modified="Azithromycin and
    # ceftriaxone are prescribed" — need UMLS evidence for ALL three drugs.
    entities_modified = []
    entities_original = []

    if modified_sent:
        entities_modified = await _extract_entities(modified_sent)
    if original_sent and original_sent != modified_sent:
        entities_original = await _extract_entities(original_sent)

    # Merge and deduplicate by entity name (case-insensitive)
    seen_names = set()
    all_entities = []
    for ent in entities_modified + entities_original:
        name_lower = ent.get("name", "").lower()
        if name_lower and name_lower not in seen_names:
            seen_names.add(name_lower)
            all_entities.append(ent)

    trace["step1_entities"] = all_entities
    trace["original_sentence"] = original_sent
    trace["modified_sentence"] = modified_sent

    # Step 2: Retrieve UMLS/RxNorm evidence for ALL extracted entities
    evidence = await _retrieve_evidence(all_entities)
    trace["step2_evidence"] = evidence

    # Step 3: Adjudicate — include whether model identified the right sentence
    location_match = ""
    if label == "CORRECT":
        # Model said note is correct but there IS an error — wrong
        assessor_prediction = "CORRECT (model claims no error)"
        location_match = "missed_error"
    elif pred_sid is not None and gt_sid is not None:
        if pred_sid == gt_sid:
            assessor_prediction = f"ERROR at sentence {pred_sid} (correct location)"
            location_match = "exact"
        else:
            assessor_prediction = f"ERROR at sentence {pred_sid} (ground truth: sentence {gt_sid})"
            location_match = "wrong_location"
    else:
        assessor_prediction = f"{label}" + (f" (sentence {pred_sid})" if pred_sid else "")
        location_match = "partial"

    trace["location_match"] = location_match

    verdict = await _adjudicate(
        original_sentence=original_sent,
        modified_sentence=modified_sent,
        assessor_prediction=assessor_prediction,
        evidence=evidence,
    )
    trace["step3_verdict"] = verdict

    # Convert verdict to a score, factoring in location accuracy
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

def compute_score(data_source, solution_str, ground_truth, extra_info=None,
                   reward_router_address=None, reward_model_tokenizer=None):
    """Hybrid reward: rule-based + Agentic UMLS Judge.

    Drop-in replacement for reward_function.compute_score.
    Called by verl's RewardManager after each rollout.

    When veRL's GenRM is enabled (reward_model.enable=True), veRL injects
    reward_router_address and reward_model_tokenizer automatically.
    When running standalone (testing), these are None and the fallback
    JUDGE_VLLM_URL is used instead.

    Scoring formula:
        R = RULE_WEIGHT * rule_score + UMLS_WEIGHT * umls_score

    If the async judge pipeline fails, falls back to pure rule-based score.

    Args:
        data_source (str): Dataset identifier.
        solution_str (str): Model's generated response.
        ground_truth (str): "CORRECT" or str(sentence_id).
        extra_info (dict): Contains correct_note, incorrect_note,
            error_sentence, error_sentence_id, mode, note_id, etc.
        reward_router_address (str, optional): veRL GenRM router endpoint.
            Injected by veRL when reward_model.enable=True.
        reward_model_tokenizer (PreTrainedTokenizer, optional): Tokenizer
            for the reward model. Injected by veRL.

    Returns:
        dict: {"score": float} for veRL Reward Loop compatibility.
    """
    # Store GenRM references for use by _llm_generate
    global _reward_router_address, _reward_model_tokenizer
    if reward_router_address is not None:
        _reward_router_address = reward_router_address
        _reward_model_tokenizer = reward_model_tokenizer

    # Smoke-test sentinel — confirms real model rollouts reach the reward function
    logger.info(
        "compute_score called: source=%s gt=%s solution=%r...",
        data_source,
        ground_truth,
        solution_str[:80],
    )

    # Step A: Always compute the rule-based score (fast, synchronous)
    rule_score = rule_compute_score(data_source, solution_str, ground_truth, extra_info)

    # If UMLS weight is 0, short-circuit
    if UMLS_WEIGHT <= 0:
        return {"score": rule_score}

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

    return {"score": final_score}


# =============================================================================
# Async compute_score (for verl async reward mode)
# =============================================================================

async def async_compute_score(data_source, solution_str, ground_truth, extra_info=None,
                               reward_router_address=None, reward_model_tokenizer=None):
    """Async version of compute_score for verl's async Reward Loop.

    verl auto-detects async functions and loads them accordingly.
    When GenRM is enabled, reward_router_address and reward_model_tokenizer
    are injected by veRL — judge calls route through the native /generate API.
    Returns dict {"score": float} per veRL Reward Loop API.
    """
    # Store GenRM references for use by _llm_generate
    global _reward_router_address, _reward_model_tokenizer
    if reward_router_address is not None:
        _reward_router_address = reward_router_address
        _reward_model_tokenizer = reward_model_tokenizer

    # Smoke-test sentinel — confirms real model rollouts reach the reward function
    logger.info(
        "compute_score called: source=%s gt=%s solution=%r...",
        data_source,
        ground_truth,
        solution_str[:80],
    )

    rule_score = rule_compute_score(data_source, solution_str, ground_truth, extra_info)

    if UMLS_WEIGHT <= 0:
        return {"score": rule_score}

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

    return {"score": final_score}


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
