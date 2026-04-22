"""Shared simple-judge client used by rollout and reward wrappers."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from scripts.self_play.utils import strip_thinking

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "prompts" / "simple_judge_prompts.json"
JUDGE_URL = os.getenv("JUDGE_VLLM_URL", "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "Qwen/Qwen3-8B")
JUDGE_TIMEOUT = float(os.getenv("SIMPLE_JUDGE_TIMEOUT", "20"))

_PROMPT_CACHE: Optional[dict] = None


def load_prompt_config() -> dict:
    global _PROMPT_CACHE
    if _PROMPT_CACHE is None:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            _PROMPT_CACHE = json.load(f)
    return _PROMPT_CACHE


def extract_json_object(text: str) -> Optional[dict]:
    text = (text or "").strip()
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
    for idx in range(start, len(text)):
        if text[idx] == "{":
            depth += 1
        elif text[idx] == "}":
            depth -= 1
            if depth == 0:
                try:
                    parsed = json.loads(text[start : idx + 1])
                except (json.JSONDecodeError, TypeError, ValueError):
                    return None
                return parsed if isinstance(parsed, dict) else None
    return None


def build_judge_messages(
    *,
    note_id: str,
    error_type: str,
    changed_sid: Optional[int],
    original_sentence: str,
    modified_sentence: str,
) -> list[dict]:
    prompt_cfg = load_prompt_config()
    user_prompt = prompt_cfg["user_template"].format(
        note_id=note_id,
        error_type=error_type,
        changed_sid=changed_sid,
        original_sentence=original_sentence[:2000],
        modified_sentence=modified_sentence[:2000],
    )
    messages = [{"role": "system", "content": prompt_cfg["system_prompt"]}]
    for ex in prompt_cfg.get("few_shot_examples", []):
        messages.append(
            {
                "role": "user",
                "content": prompt_cfg["user_template"].format(
                    note_id=ex.get("note_id", "example"),
                    error_type=ex.get("error_type", ""),
                    changed_sid=ex.get("changed_sid", "?"),
                    original_sentence=ex["original_sentence"],
                    modified_sentence=ex["modified_sentence"],
                ),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "verdict": ex["verdict"],
                        "score": ex["score"],
                        "reason": ex["reason"],
                    }
                ),
            }
        )
    messages.append({"role": "user", "content": user_prompt})
    return messages


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


async def post_json(url: str, payload: dict) -> dict:
    try:
        import aiohttp  # type: ignore
    except ImportError:
        return await asyncio.to_thread(_sync_post_json, url, payload)

    timeout = aiohttp.ClientTimeout(total=JUDGE_TIMEOUT)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.post(url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


async def judge_sentence_pair(
    *,
    note_id: str,
    error_type: str,
    changed_sid: Optional[int],
    original_sentence: str,
    modified_sentence: str,
    judge_url: Optional[str] = None,
    judge_model: Optional[str] = None,
    reward_router_address: Optional[str] = None,
) -> dict:
    """Run the simple sentence-pair judge and return its raw verdict."""
    if not original_sentence and not modified_sentence:
        return {
            "verdict": "ABSTAIN",
            "judge_score": 0.0,
            "reason": "missing_input",
            "judge_output": "",
        }

    prompt_cfg = load_prompt_config()
    payload = {
        "messages": build_judge_messages(
            note_id=note_id,
            error_type=error_type,
            changed_sid=changed_sid,
            original_sentence=original_sentence,
            modified_sentence=modified_sentence,
        ),
        **prompt_cfg.get("sampling_params", {}),
    }
    payload["chat_template_kwargs"] = {"enable_thinking": False}

    target_url = reward_router_address or judge_url or JUDGE_URL
    if not target_url:
        return {
            "verdict": "ABSTAIN",
            "judge_score": 0.0,
            "reason": "no_judge_url",
            "judge_output": "",
        }

    if not reward_router_address:
        payload.setdefault("model", judge_model or JUDGE_MODEL)

    try:
        result = await post_json(target_url, payload)
    except (URLError, OSError, asyncio.TimeoutError, ValueError) as exc:
        logger.warning("Simple judge request failed: %s", exc)
        return {
            "verdict": "ABSTAIN",
            "judge_score": 0.0,
            "reason": f"request_failed:{exc}",
            "judge_output": "",
        }
    except Exception as exc:  # pragma: no cover
        logger.warning("Simple judge unexpected failure: %s", exc)
        return {
            "verdict": "ABSTAIN",
            "judge_score": 0.0,
            "reason": f"unexpected_error:{exc}",
            "judge_output": "",
        }

    try:
        content = result["choices"][0]["message"]["content"].strip()
    except Exception:
        return {
            "verdict": "ABSTAIN",
            "judge_score": 0.0,
            "reason": "bad_response_shape",
            "judge_output": "",
        }

    _, stripped = strip_thinking(content)
    verdict = extract_json_object(stripped) or extract_json_object(content)
    if not verdict:
        return {
            "verdict": "ABSTAIN",
            "judge_score": 0.0,
            "reason": "no_json_verdict",
            "judge_output": content[:2000],
        }

    label = str(verdict.get("verdict", "ABSTAIN")).upper()
    try:
        score = float(verdict.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))
    return {
        "verdict": label,
        "judge_score": score,
        "reason": str(verdict.get("reason", ""))[:1000],
        "judge_output": content[:2000],
    }
