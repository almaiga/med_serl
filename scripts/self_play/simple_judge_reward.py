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
from datetime import datetime
from typing import Any, Dict, Optional
from urllib.error import URLError
from urllib.request import Request, urlopen

from scripts.self_play.reward_function import (
    compute_score as rule_compute_score,
    interaction_reward_passthrough,
    _resolve_assessor_ground_truth,
)
from scripts.self_play.utils import (
    number_sentences,
    parse_injector_compact,
    parse_numbered_sentences,
    strip_thinking,
)

logger = logging.getLogger(__name__)

_PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "configs" / "prompts" / "simple_judge_prompts.json"
_PROMPT_CACHE: Optional[dict] = None

JUDGE_URL = os.getenv("JUDGE_VLLM_URL", "")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "Qwen/Qwen3-8B")
JUDGE_WEIGHT = float(os.getenv("SIMPLE_JUDGE_WEIGHT", "0.3"))
JUDGE_TIMEOUT = float(os.getenv("SIMPLE_JUDGE_TIMEOUT", "20"))
JUDGE_DISABLED = os.getenv("DISABLE_SIMPLE_JUDGE", "0") == "1"
_JUDGE_TRACE_FILE: Optional[Path] = None


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


def _is_injector_evaluation(extra_info: dict) -> bool:
    return extra_info.get("role") == "injector" and not _has_interaction_result(extra_info)


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


def _resolve_injector_sentence_pair(solution_str: str, extra_info: dict) -> tuple[Optional[int], str, str]:
    sid, modified_sentence = parse_injector_compact(solution_str or "")
    if sid is None or modified_sentence is None:
        return None, "", ""

    original_sentences = parse_numbered_sentences(str(extra_info.get("sentences", "")))
    original_sentence = original_sentences.get(sid, "")
    return sid, original_sentence, modified_sentence


def _resolve_ground_truth_for_judge(ground_truth: Any, extra_info: dict) -> tuple[str, str]:
    raw = str(ground_truth) if ground_truth else ""
    if _is_assessor_evaluation(extra_info):
        return raw, _resolve_assessor_ground_truth(raw, extra_info)
    return raw, raw


def _judge_trace_file() -> Path:
    global _JUDGE_TRACE_FILE
    if _JUDGE_TRACE_FILE is not None:
        return _JUDGE_TRACE_FILE

    log_dir = Path(os.environ.get("MEDSERL_GAME_LOG", "")).parent if os.environ.get("MEDSERL_GAME_LOG") else (
        Path(__file__).resolve().parent.parent.parent / "results" / "self_play" / "interactions"
    )
    log_dir.mkdir(parents=True, exist_ok=True)
    _JUDGE_TRACE_FILE = log_dir / f"judge_interactions_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    return _JUDGE_TRACE_FILE


def _append_judge_trace(entry: dict) -> None:
    try:
        with open(_judge_trace_file(), "a") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as exc:  # pragma: no cover - logging must never break rewarding
        logger.warning("Failed to write judge trace: %s", exc)


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
    extra_info: dict,
    expected_relation: str,
    reward_router_address: Optional[str] = None,
) -> dict:
    prompt_cfg = _load_prompt_config()
    changed_sid, original_sentence, modified_sentence = _resolve_injector_sentence_pair(
        solution_str, extra_info
    )

    if not original_sentence and not modified_sentence:
        return {
            "signed_score": 0.0,
            "verdict": "ABSTAIN",
            "judge_score": 0.0,
            "reason": "missing_input",
            "judge_output": "",
            "original_sentence": original_sentence,
            "modified_sentence": modified_sentence,
            "changed_sid": changed_sid,
        }

    user_prompt = prompt_cfg["user_template"].format(
        note_id=extra_info.get("note_id", ""),
        error_type=extra_info.get("error_type", ""),
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

    payload = {
        "messages": messages,
        **prompt_cfg.get("sampling_params", {}),
    }
    payload["chat_template_kwargs"] = {"enable_thinking": False}
    if reward_router_address:
        target_url = reward_router_address
    elif JUDGE_URL:
        target_url = JUDGE_URL
        payload.setdefault("model", JUDGE_MODEL)
    else:
        return {
            "signed_score": 0.0,
            "verdict": "ABSTAIN",
            "judge_score": 0.0,
            "reason": "no_judge_url",
            "judge_output": "",
            "original_sentence": original_sentence,
            "modified_sentence": modified_sentence,
            "changed_sid": changed_sid,
        }

    try:
        result = await _post_json(target_url, payload)
    except (URLError, OSError, asyncio.TimeoutError, ValueError) as exc:
        logger.warning("Simple judge request failed: %s", exc)
        return {
            "signed_score": 0.0,
            "verdict": "ABSTAIN",
            "judge_score": 0.0,
            "reason": f"request_failed:{exc}",
            "judge_output": "",
            "original_sentence": original_sentence,
            "modified_sentence": modified_sentence,
            "changed_sid": changed_sid,
        }
    except Exception as exc:  # pragma: no cover - safety net
        logger.warning("Simple judge unexpected failure: %s", exc)
        return {
            "signed_score": 0.0,
            "verdict": "ABSTAIN",
            "judge_score": 0.0,
            "reason": f"unexpected_error:{exc}",
            "judge_output": "",
            "original_sentence": original_sentence,
            "modified_sentence": modified_sentence,
            "changed_sid": changed_sid,
        }

    try:
        content = result["choices"][0]["message"]["content"].strip()
    except Exception:
        return {
            "signed_score": 0.0,
            "verdict": "ABSTAIN",
            "judge_score": 0.0,
            "reason": "bad_response_shape",
            "judge_output": "",
            "original_sentence": original_sentence,
            "modified_sentence": modified_sentence,
            "changed_sid": changed_sid,
        }

    _, stripped = strip_thinking(content)
    verdict = _extract_json_object(stripped) or _extract_json_object(content)
    if not verdict:
        return {
            "signed_score": 0.0,
            "verdict": "ABSTAIN",
            "judge_score": 0.0,
            "reason": "no_json_verdict",
            "judge_output": content[:2000],
            "original_sentence": original_sentence,
            "modified_sentence": modified_sentence,
            "changed_sid": changed_sid,
        }

    label = str(verdict.get("verdict", "ABSTAIN")).upper()
    try:
        score = float(verdict.get("score", 0.0))
    except (TypeError, ValueError):
        score = 0.0
    score = max(0.0, min(1.0, score))

    if label == expected_relation:
        signed_score = score
    elif label in {"SAME", "CHANGED"}:
        signed_score = -score
    else:
        signed_score = 0.0

    return {
        "signed_score": signed_score,
        "verdict": label,
        "judge_score": score,
        "reason": str(verdict.get("reason", ""))[:1000],
        "judge_output": content[:2000],
        "original_sentence": original_sentence,
        "modified_sentence": modified_sentence,
        "changed_sid": changed_sid,
    }


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
    ground_truth_raw, ground_truth_resolved = _resolve_ground_truth_for_judge(
        ground_truth, extra_info
    )
    base_score = _base_score(data_source, solution_str, ground_truth_resolved, extra_info)

    if JUDGE_DISABLED or JUDGE_WEIGHT <= 0 or not _is_injector_evaluation(extra_info):
        return {"score": base_score}

    if not reward_router_address and not JUDGE_URL:
        return {"score": base_score}

    expected_relation = "SAME" if extra_info.get("mode") == "benign" else "CHANGED"

    judge_result = await _judge_with_llm(
        solution_str=solution_str or "",
        extra_info=extra_info,
        expected_relation=expected_relation,
        reward_router_address=reward_router_address,
    )
    weighted_judge_score = JUDGE_WEIGHT * float(judge_result["signed_score"])
    final_score = base_score + weighted_judge_score

    _append_judge_trace(
        {
            "timestamp": datetime.now().isoformat(),
            "data_source": str(data_source),
            "note_id": extra_info.get("note_id", ""),
            "role": extra_info.get("role", "assessor"),
            "mode": extra_info.get("mode", "unknown"),
            "ground_truth_raw": ground_truth_raw,
            "ground_truth_resolved": ground_truth_resolved,
            "expected_relation": expected_relation,
            "judge_verdict": judge_result["verdict"],
            "judge_score": float(judge_result["judge_score"]),
            "judge_signed_score": float(judge_result["signed_score"]),
            "weighted_judge_score": weighted_judge_score,
            "base_score": float(base_score),
            "final_score": float(final_score),
            "judge_reason": judge_result["reason"],
            "judge_output": judge_result["judge_output"],
            "changed_sid": judge_result.get("changed_sid"),
            "original_sentence": judge_result.get("original_sentence", ""),
            "modified_sentence": judge_result.get("modified_sentence", ""),
        }
    )

    return {"score": final_score}


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
