"""Simple Qwen judge reward for MedSeRL self-play.

This is the lightweight replacement for the older UMLS-based judge path.

It supports:
1. Multi-turn injector -> assessor self-play using MedicalGameInteraction
2. Single-turn separated assessor examples

The score is:
    final_score = base_rule_score + SIMPLE_JUDGE_WEIGHT * judge_score

Where:
- base_rule_score comes from the existing rule-based reward
- judge_score is signed in [-1, +1] from a Qwen judge prompt
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from scripts.self_play.reward_function import (
    compute_score as rule_compute_score,
    interaction_reward_passthrough,
)
from scripts.self_play.utils import number_sentences

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "prompts" / "simple_judge_prompts.json"
_PROMPT_CACHE: Optional[dict] = None

JUDGE_URL = os.getenv("JUDGE_VLLM_URL", "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "Qwen/Qwen3-8B")
JUDGE_WEIGHT = float(os.getenv("SIMPLE_JUDGE_WEIGHT", "0.3"))
JUDGE_TIMEOUT = float(os.getenv("SIMPLE_JUDGE_TIMEOUT", "20"))
JUDGE_DISABLED = os.getenv("DISABLE_SIMPLE_JUDGE", "0") == "1"


def _load_prompt_config() -> dict:
    global _PROMPT_CACHE
    if _PROMPT_CACHE is None:
        with open(_PROMPT_PATH, "r") as f:
            _PROMPT_CACHE = json.load(f)
    return _PROMPT_CACHE


def _extract_json_object(text: str) -> Optional[dict]:
    text = text.strip()
    if not text:
        return None
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else None
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

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
                    parsed = json.loads(text[start:i + 1])
                    return parsed if isinstance(parsed, dict) else None
                except (json.JSONDecodeError, TypeError, ValueError):
                    return None
    return None


def _has_interaction_result(extra_info: dict) -> bool:
    return extra_info.get("phase") == "game_complete" or bool(extra_info.get("phases_separated"))


def _is_assessor_evaluation(extra_info: dict) -> bool:
    if _has_interaction_result(extra_info):
        return True
    return extra_info.get("role", "assessor") == "assessor"


def _resolve_numbered_note(extra_info: dict) -> str:
    note = extra_info.get("modified_sentences") or extra_info.get("sentences")
    if note:
        return str(note)
    raw_note = extra_info.get("incorrect_note") or extra_info.get("correct_note") or ""
    return number_sentences(raw_note) if raw_note else ""


def _resolve_assessor_output(solution_str: str, extra_info: dict) -> str:
    assessor_output = extra_info.get("assessor_output")
    if assessor_output:
        return str(assessor_output)

    label = extra_info.get("assessor_label")
    pred_sid = extra_info.get("assessor_pred_sid")
    if label == "CORRECT":
        return "CORRECT"
    if label == "ERROR" and pred_sid is not None:
        return str(pred_sid)
    return solution_str or ""


def _base_score(data_source, solution_str, ground_truth, extra_info: dict) -> float:
    if _has_interaction_result(extra_info):
        return interaction_reward_passthrough(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
    return rule_compute_score(data_source, solution_str, ground_truth, extra_info)


def _sync_post_json(url: str, payload: dict) -> dict:
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=JUDGE_TIMEOUT) as resp:
        body = resp.read().decode("utf-8")
    return json.loads(body)


async def _post_json(url: str, payload: dict) -> dict:
    try:
        import aiohttp  # type: ignore
    except ImportError:
        return await asyncio.to_thread(_sync_post_json, url, payload)

    timeout = aiohttp.ClientTimeout(total=JUDGE_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def _judge_with_llm(
    solution_str: str,
    ground_truth: str,
    extra_info: dict,
    reward_router_address: Optional[str] = None,
) -> float:
    prompt_cfg = _load_prompt_config()
    numbered_note = _resolve_numbered_note(extra_info)
    assessor_output = _resolve_assessor_output(solution_str, extra_info)

    if not numbered_note or not assessor_output or not ground_truth:
        return 0.0

    user_prompt = prompt_cfg["user_template"].format(
        mode=extra_info.get("mode", "unknown"),
        note_id=extra_info.get("note_id", ""),
        numbered_note=numbered_note[:6000],
        ground_truth=str(ground_truth),
        assessor_output=assessor_output[:1500],
    )
    messages = [
        {"role": "system", "content": prompt_cfg["system_prompt"]},
        {"role": "user", "content": user_prompt},
    ]

    payload = {"messages": messages, **prompt_cfg.get("sampling_params", {})}
    if reward_router_address:
        target_url = reward_router_address
    elif JUDGE_URL:
        target_url = JUDGE_URL
        payload.setdefault("model", JUDGE_MODEL)
    else:
        return 0.0

    try:
        result = await _post_json(target_url, payload)
    except (URLError, OSError, asyncio.TimeoutError, ValueError) as exc:
        logger.warning("Simple judge request failed: %s", exc)
        return 0.0
    except Exception as exc:  # pragma: no cover - safety net
        logger.warning("Simple judge unexpected failure: %s", exc)
        return 0.0

    try:
        content = result["choices"][0]["message"]["content"].strip()
    except Exception:
        return 0.0

    verdict = _extract_json_object(content)
    if not verdict:
        return 0.0

    label = str(verdict.get("verdict", "ABSTAIN")).upper()
    try:
        score = float(verdict.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))

    if label == "PASS":
        return score
    if label == "FAIL":
        return -score
    return 0.0


async def async_compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    reward_router_address=None,
    reward_model_tokenizer=None,
):
    del reward_model_tokenizer

    extra_info = extra_info or {}
    base_score = _base_score(data_source, solution_str, ground_truth, extra_info)

    if JUDGE_DISABLED or JUDGE_WEIGHT <= 0 or not _is_assessor_evaluation(extra_info):
        return {"score": base_score}

    if not reward_router_address and not JUDGE_URL:
        return {"score": base_score}

    judge_score = await _judge_with_llm(
        solution_str=solution_str or "",
        ground_truth=str(ground_truth) if ground_truth else "",
        extra_info=extra_info,
        reward_router_address=reward_router_address,
    )
    return {"score": base_score + JUDGE_WEIGHT * judge_score}


def compute_score(
    data_source,
    solution_str,
    ground_truth,
    extra_info=None,
    reward_router_address=None,
    reward_model_tokenizer=None,
):
    coro = async_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        reward_router_address=reward_router_address,
        reward_model_tokenizer=reward_model_tokenizer,
    )

    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result(timeout=JUDGE_TIMEOUT + 5)
