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

from scripts.self_play.utils import normalized_edit_distance, strip_thinking

logger = logging.getLogger(__name__)

PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "prompts" / "simple_judge_prompts.json"
MEDRECT_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "prompts" / "medrect_judge_prompts.json"
JUDGE_URL = os.getenv("JUDGE_VLLM_URL", "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "Qwen/Qwen3-8B")
# "sentence_pair" = legacy Qwen JSON-verdict judge; "detection" = MedRECT-style
# detector run on the modified note (CORRECT / sentence-number output).
JUDGE_TYPE = os.getenv("JUDGE_TYPE", "sentence_pair")
# Prompt style for the detection judge: "native" (note only) or "hint" (note + edit context).
JUDGE_PROMPT_STYLE = os.getenv("JUDGE_PROMPT_STYLE", "native")
JUDGE_TIMEOUT = float(os.getenv("SIMPLE_JUDGE_TIMEOUT", "60"))
JUDGE_MAX_RETRIES = int(os.getenv("SIMPLE_JUDGE_MAX_RETRIES", "2"))

_PROMPT_CACHE: Optional[dict] = None
_MEDRECT_PROMPT_CACHE: Optional[dict] = None


def load_prompt_config() -> dict:
    global _PROMPT_CACHE
    if _PROMPT_CACHE is None:
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            _PROMPT_CACHE = json.load(f)
    return _PROMPT_CACHE


def load_medrect_prompt_config() -> dict:
    global _MEDRECT_PROMPT_CACHE
    if _MEDRECT_PROMPT_CACHE is None:
        with open(MEDRECT_PROMPT_PATH, "r", encoding="utf-8") as f:
            _MEDRECT_PROMPT_CACHE = json.load(f)
    return _MEDRECT_PROMPT_CACHE


def build_detection_messages(
    *,
    modified_note: str,
    changed_sid: Optional[int],
    original_sentence: str = "",
    modified_sentence: str = "",
    style: Optional[str] = None,
) -> list[dict]:
    """Build messages for a MedRECT-style detection judge over the modified note."""
    cfg = load_medrect_prompt_config()
    style = style or JUDGE_PROMPT_STYLE
    sub = cfg.get(style) or cfg["native"]
    note = (modified_note or "")[:6000]
    if "{original_sentence}" in sub["user_template"]:
        user = sub["user_template"].format(
            changed_sid=changed_sid,
            original_sentence=(original_sentence or "")[:2000],
            modified_sentence=(modified_sentence or "")[:2000],
            modified_note=note,
        )
    else:
        user = sub["user_template"].format(modified_note=note)
    return [
        {"role": "system", "content": sub["system_prompt"]},
        {"role": "user", "content": user},
    ]


def parse_detection_verdict(content: str, changed_sid: Optional[int]) -> dict:
    """Map a MedRECT detection output to a SAME/CHANGED/ABSTAIN judge verdict.

    CORRECT                        -> SAME  (the edit introduced no detectable error)
    flags the changed sentence     -> CHANGED
    flags a different sentence / no parse -> ABSTAIN (neutral, ambiguous localization)
    """
    from scripts.self_play.utils import parse_assessor_answer

    label, detected_sid = parse_assessor_answer(content or "")
    if label == "CORRECT":
        return {
            "verdict": "SAME",
            "status": "ok",
            "judge_score": 1.0,
            "reason": "detector reports no error",
            "detected_sid": None,
        }
    if label == "ERROR":
        if (
            detected_sid is not None
            and changed_sid is not None
            and int(detected_sid) == int(changed_sid)
        ):
            return {
                "verdict": "CHANGED",
                "status": "ok",
                "judge_score": 1.0,
                "reason": f"detector flagged the changed sentence ({detected_sid})",
                "detected_sid": detected_sid,
            }
        return {
            "verdict": "ABSTAIN",
            "status": "detected_other_sid",
            "judge_score": 0.5,
            "reason": f"detector flagged sentence {detected_sid}, not the changed sentence {changed_sid}",
            "detected_sid": detected_sid,
        }
    return {
        "verdict": "ABSTAIN",
        "status": "no_detection_parse",
        "judge_score": 0.0,
        "reason": "detector output unparseable",
        "detected_sid": None,
    }


def extract_json_object(text: str) -> Optional[dict]:
    text = (text or "").strip()
    if not text:
        return None
    # Strip markdown code fences (```json ... ``` or ``` ... ```) that Qwen3
    # sometimes wraps around its JSON output.
    import re as _re
    fence_match = _re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence_match:
        text = fence_match.group(1).strip()
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
    norm_edit: Optional[float] = None,
    modified_note: Optional[str] = None,
) -> list[dict]:
    prompt_cfg = load_prompt_config()
    if norm_edit is None:
        norm_edit = normalized_edit_distance(original_sentence, modified_sentence)

    # Use the contextual template (with full note) for live calls when available.
    # Few-shot examples always use the sentence-only template — they don't carry
    # full notes and still teach the SAME/CHANGED/ABSTAIN taxonomy correctly.
    if modified_note and "user_template_with_context" in prompt_cfg:
        live_template = prompt_cfg["user_template_with_context"]
        user_prompt = live_template.format(
            note_id=note_id,
            error_type=error_type,
            changed_sid=changed_sid,
            original_sentence=original_sentence[:2000],
            modified_sentence=modified_sentence[:2000],
            modified_note=modified_note[:3000],
            norm_edit=f"{norm_edit:.2f}",
        )
    else:
        user_prompt = prompt_cfg["user_template"].format(
            note_id=note_id,
            error_type=error_type,
            changed_sid=changed_sid,
            original_sentence=original_sentence[:2000],
            modified_sentence=modified_sentence[:2000],
            norm_edit=f"{norm_edit:.2f}",
        )

    messages = [{"role": "system", "content": prompt_cfg["system_prompt"]}]
    for ex in prompt_cfg.get("few_shot_examples", []):
        ex_norm_edit = normalized_edit_distance(
            ex["original_sentence"], ex["modified_sentence"]
        )
        messages.append(
            {
                "role": "user",
                "content": prompt_cfg["user_template"].format(
                    note_id=ex.get("note_id", "example"),
                    error_type=ex.get("error_type", ""),
                    changed_sid=ex.get("changed_sid", "?"),
                    original_sentence=ex["original_sentence"],
                    modified_sentence=ex["modified_sentence"],
                    norm_edit=f"{ex_norm_edit:.2f}",
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
    norm_edit: Optional[float] = None,
    modified_note: Optional[str] = None,
) -> dict:
    """Run the simple sentence-pair judge and return its raw verdict."""
    if not original_sentence and not modified_sentence:
        return {
            "verdict": "ABSTAIN",
            "status": "missing_input",
            "judge_score": 0.0,
            "reason": "missing_input",
            "judge_output": "",
        }

    if norm_edit is None:
        norm_edit = normalized_edit_distance(original_sentence, modified_sentence)

    is_detection = JUDGE_TYPE == "detection"
    if is_detection:
        medrect_cfg = load_medrect_prompt_config()
        payload = {
            "messages": build_detection_messages(
                modified_note=modified_note or "",
                changed_sid=changed_sid,
                original_sentence=original_sentence,
                modified_sentence=modified_sentence,
            ),
            **medrect_cfg.get("sampling_params", {}),
        }
        # MedRECT-32B is Qwen3-based. Thinking must stay ON: with it OFF the
        # judge is blind to plausible clinical errors (0/6 on medium and 0/6
        # on subtle errors in the 48-case synthetic test; 44/48 with thinking).
        # The v6 run trained against the thinking-OFF judge and that is why
        # self-play plateaued. parse_assessor_answer strips <think>...</think>,
        # and max_tokens in medrect_judge_prompts.json is sized (2048) so the
        # model finishes reasoning AND emits the verdict within budget.
        payload["chat_template_kwargs"] = {"enable_thinking": True}
    else:
        prompt_cfg = load_prompt_config()
        payload = {
            "messages": build_judge_messages(
                note_id=note_id,
                error_type=error_type,
                changed_sid=changed_sid,
                original_sentence=original_sentence,
                modified_sentence=modified_sentence,
                norm_edit=norm_edit,
                modified_note=modified_note,
            ),
            **prompt_cfg.get("sampling_params", {}),
        }
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    target_url = reward_router_address or judge_url or JUDGE_URL
    if not target_url:
        return {
            "verdict": "ABSTAIN",
            "status": "no_judge_url",
            "judge_score": 0.0,
            "reason": "no_judge_url",
            "judge_output": "",
        }

    if not reward_router_address:
        payload.setdefault("model", judge_model or JUDGE_MODEL)

    _transient_exc = (URLError, OSError, asyncio.TimeoutError, ValueError)
    try:
        import aiohttp as _aiohttp
        _transient_exc = _transient_exc + (_aiohttp.ClientError,)
    except ImportError:
        pass

    last_exc: Optional[Exception] = None
    for attempt in range(1 + JUDGE_MAX_RETRIES):
        if attempt > 0:
            await asyncio.sleep(2 ** (attempt - 1))  # 1s, 2s, ...
        try:
            result = await post_json(target_url, payload)
            last_exc = None
            break
        except _transient_exc as exc:
            last_exc = exc
            logger.warning(
                "Simple judge request failed (attempt %d/%d) [%s] url=%s timeout=%ss: %s",
                attempt + 1,
                1 + JUDGE_MAX_RETRIES,
                type(exc).__name__,
                target_url,
                JUDGE_TIMEOUT,
                exc,
            )
        except Exception as exc:  # pragma: no cover
            logger.warning("Simple judge unexpected failure [%s] url=%s: %s", type(exc).__name__, target_url, exc)
            return {
                "verdict": "ABSTAIN",
                "status": "unexpected_error",
                "judge_score": 0.0,
                "reason": f"unexpected_error:{exc}",
                "judge_output": "",
            }

    if last_exc is not None:
        return {
            "verdict": "ABSTAIN",
            "status": "request_failed",
            "judge_score": 0.0,
            "reason": f"request_failed:{last_exc}",
            "judge_output": "",
        }

    try:
        content = result["choices"][0]["message"]["content"].strip()
    except Exception:
        return {
            "verdict": "ABSTAIN",
            "status": "bad_response_shape",
            "judge_score": 0.0,
            "reason": "bad_response_shape",
            "judge_output": "",
        }

    if is_detection:
        result_dict = parse_detection_verdict(content, changed_sid)
        result_dict["norm_edit"] = float(norm_edit)
        result_dict["judge_output"] = content[:2000]
        return result_dict

    _, stripped = strip_thinking(content)
    verdict = extract_json_object(stripped) or extract_json_object(content)
    if not verdict:
        return {
            "verdict": "ABSTAIN",
            "status": "no_json_verdict",
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
    if label in {"SAME", "CHANGED"}:
        # Reject CHANGED verdicts the judge isn't confident about — these are
        # typically garbage outputs (meta-text, complete rewrites) where the
        # judge detects difference but assigns near-zero quality score.
        status = "low_confidence_changed" if label == "CHANGED" and score < 0.3 else "ok"
    else:
        status = "semantic_abstain"
    return {
        "verdict": label,
        "status": status,
        "judge_score": score,
        "norm_edit": float(norm_edit),
        "reason": str(verdict.get("reason", ""))[:1000],
        "judge_output": content[:2000],
    }
