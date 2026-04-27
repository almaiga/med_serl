"""Custom vLLM-backed agent loop for batched MedSeRL self-play."""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

try:
    from omegaconf import OmegaConf
except ImportError:  # pragma: no cover - local fallback
    OmegaConf = None  # type: ignore

try:
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopMetrics,
        AgentLoopOutput,
    )
except ImportError:  # pragma: no cover - local unit tests without verl
    class AgentLoopBase:  # type: ignore
        _class_initialized = False

        def __init__(self, trainer_config=None, server_manager=None, tokenizer=None, processor=None, **kwargs):
            self.config = getattr(trainer_config, "config", None)
            self.server_manager = server_manager
            self.tokenizer = tokenizer
            self.processor = processor

        @classmethod
        def init_class(cls, *args, **kwargs):
            cls._class_initialized = True

    class AgentLoopMetrics:  # type: ignore
        def __init__(self, generate_sequences: float = 0.0, tool_calls: float = 0.0):
            self.generate_sequences = generate_sequences
            self.tool_calls = tool_calls

    class AgentLoopOutput:  # type: ignore
        def __init__(
            self,
            prompt_ids,
            response_ids,
            response_mask,
            response_logprobs=None,
            multi_modal_data=None,
            reward_score=None,
            num_turns=0,
            metrics=None,
            extra_fields=None,
        ):
            self.prompt_ids = prompt_ids
            self.response_ids = response_ids
            self.response_mask = response_mask
            self.response_logprobs = response_logprobs
            self.multi_modal_data = multi_modal_data
            self.reward_score = reward_score
            self.num_turns = num_turns
            self.metrics = metrics
            self.extra_fields = extra_fields or {}

from scripts.self_play.game_reward import (
    FORMAT_BONUS,
    REWARD_MISS,
    compute_assessor_game_reward,
    compute_injector_game_reward,
    derive_assessor_ground_truth,
)
from scripts.self_play.judge_client import judge_sentence_pair
from scripts.self_play.utils import (
    parse_injector_compact,
    parse_numbered_sentences,
    reconstruct_note,
    strip_thinking,
)

_FALLBACK_LOG_DIR = Path(__file__).resolve().parents[3] / "results" / "self_play" / "interactions"
_LOG_LOCK = threading.Lock()
_LOG_FILE: Optional[Path] = None


def _get_log_file() -> Path:
    global _LOG_FILE
    if _LOG_FILE is not None:
        return _LOG_FILE
    env_log = os.environ.get("MEDSERL_GAME_LOG")
    if env_log:
        p = Path(env_log)
        p.parent.mkdir(parents=True, exist_ok=True)
        _LOG_FILE = p
        return _LOG_FILE
    _FALLBACK_LOG_DIR.mkdir(parents=True, exist_ok=True)
    _LOG_FILE = _FALLBACK_LOG_DIR / f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    return _LOG_FILE


def _append_game_log(entry: dict[str, Any]) -> None:
    try:
        with _LOG_LOCK:
            with open(_get_log_file(), "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception:
        pass


class _AttrDict(dict):
    """Fallback config object supporting both .get() and attribute access."""

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(name) from exc


class _DummyDataConfig:
    """Fallback data_config for direct unit-test instantiation."""

    def __init__(self):
        self.config = {}


def _to_agent_loop_config(value: Any) -> Any:
    if hasattr(value, "get") and hasattr(value, "actor_rollout_ref"):
        return value
    if OmegaConf is not None:
        try:
            return OmegaConf.create(value)
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        value = vars(value)
    if isinstance(value, dict):
        return _AttrDict({k: _to_agent_loop_config(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return type(value)(_to_agent_loop_config(v) for v in value)
    return value


class MedSerlSelfPlayAgentLoop(AgentLoopBase):
    """Two-phase injector -> judge -> assessor agent loop for vLLM async rollout."""

    _class_initialized = False
    detection_prompts: dict[str, Any]
    injector_max_new_tokens: int
    assessor_max_new_tokens: int
    assessor_temperature: float
    assessor_top_p: Optional[float]
    assessor_top_k: Optional[int]
    assessor_repetition_penalty: Optional[float]
    judge_url: str
    judge_model: str

    def __init__(self, trainer_config=None, server_manager=None, tokenizer=None, processor=None, **kwargs):
        normalized_config = _to_agent_loop_config(getattr(trainer_config, "config", trainer_config))
        normalized_trainer_config = trainer_config
        if trainer_config is not None and hasattr(trainer_config, "config"):
            trainer_config.config = normalized_config
        try:
            super().__init__(
                trainer_config=normalized_trainer_config,
                server_manager=server_manager,
                tokenizer=tokenizer,
                processor=processor,
                **kwargs,
            )
        except TypeError as exc:
            if "dataset_cls" not in str(exc) and "data_config" not in str(exc):
                raise
            dataset_cls = kwargs.pop("dataset_cls", None)
            data_config = kwargs.pop("data_config", None)
            if data_config is None:
                data_config = _DummyDataConfig()
            super().__init__(
                trainer_config=normalized_trainer_config,
                server_manager=server_manager,
                tokenizer=tokenizer,
                processor=processor,
                dataset_cls=dataset_cls,
                data_config=data_config,
                **kwargs,
            )
        cls = type(self)
        if not getattr(cls, "_class_initialized", False):
            cls.init_class(
                normalized_config,
                tokenizer,
                processor,
                **kwargs,
            )

    @classmethod
    def init_class(
        cls,
        config,
        tokenizer,
        processor,
        *,
        detection_prompts_path: str = "configs/prompts/detection_localization_prompts.json",
        injector_max_new_tokens: int = 512,
        assessor_max_new_tokens: int = 256,
        assessor_temperature: float = 1.0,
        assessor_top_p: Optional[float] = None,
        assessor_top_k: Optional[int] = None,
        assessor_repetition_penalty: Optional[float] = None,
        judge_url: str = "",
        judge_model: str = "",
        **kwargs,
    ):
        del tokenizer, processor, kwargs
        if getattr(cls, "_class_initialized", False):
            return
        with open(detection_prompts_path, "r", encoding="utf-8") as f:
            cls.detection_prompts = json.load(f)
        cls.injector_max_new_tokens = int(injector_max_new_tokens)
        cls.assessor_max_new_tokens = int(
            os.environ.get("MEDSERL_ASSESSOR_MAX_NEW_TOKENS", assessor_max_new_tokens)
        )
        cls.assessor_temperature = float(assessor_temperature)
        cls.assessor_top_p = float(assessor_top_p) if assessor_top_p is not None else None
        cls.assessor_top_k = int(assessor_top_k) if assessor_top_k is not None else None
        cls.assessor_repetition_penalty = (
            float(assessor_repetition_penalty) if assessor_repetition_penalty is not None else None
        )
        cls.judge_url = judge_url or os.environ.get("JUDGE_VLLM_URL", "")
        cls.judge_model = judge_model or os.environ.get("JUDGE_MODEL", "Qwen/Qwen3-8B")
        cls._class_initialized = True

    def _coerce_messages(self, prompt: Any) -> list[dict[str, str]]:
        if isinstance(prompt, list):
            return [dict(item) for item in prompt]
        raise TypeError(f"Expected prompt as list[dict], got {type(prompt)!r}")

    def _tokenize_messages(self, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
        token_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
        if isinstance(token_ids, dict):
            token_ids = token_ids.get("input_ids", token_ids)
        elif hasattr(token_ids, "data") and isinstance(getattr(token_ids, "data"), dict):
            token_ids = token_ids.data.get("input_ids", token_ids)
        if hasattr(token_ids, "tolist"):
            token_ids = token_ids.tolist()
        if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
            token_ids = token_ids[0]
        if not isinstance(token_ids, list):
            raise TypeError(f"Expected token ids as list[int], got {type(token_ids)!r}")
        return list(token_ids)

    def _token_delta(
        self,
        previous_messages: list[dict[str, str]],
        current_messages: list[dict[str, str]],
    ) -> list[int]:
        prev_ids = self._tokenize_messages(previous_messages, add_generation_prompt=False)
        curr_ids = self._tokenize_messages(current_messages, add_generation_prompt=True)
        prefix_len = 0
        max_prefix = min(len(prev_ids), len(curr_ids))
        while prefix_len < max_prefix and prev_ids[prefix_len] == curr_ids[prefix_len]:
            prefix_len += 1
        return curr_ids[prefix_len:]

    async def _generate(
        self,
        request_id: str,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
    ):
        return await self.server_manager.generate(
            request_id=request_id,
            prompt_ids=prompt_ids,
            sampling_params=sampling_params,
        )

    def _decode(self, token_ids: list[int]) -> str:
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)

    def _force_think_mode(self, messages: list[dict[str, str]]) -> list[dict[str, str]]:
        updated = [dict(item) for item in messages]
        for item in updated:
            content = str(item.get("content", ""))
            if "/no_think" in content:
                item["content"] = content.replace("/no_think", "/think")

        for item in reversed(updated):
            if item.get("role") == "user":
                content = str(item.get("content", "")).strip()
                if "/think" not in content:
                    item["content"] = f"{content}\n/think"
                break
        return updated

    def _construct_assessor_prompt(self, modified_sentences: str) -> str:
        system_prompt = self.detection_prompts.get("system_prompt", "")
        user_template = self.detection_prompts.get("user_template", "{sentences}")
        user_content = user_template.format(sentences=modified_sentences)

        override = (
            "/think\n"
            "IGNORE ALL PREVIOUS TASK INSTRUCTIONS.\n"
            "The injector step is finished.\n"
            "You are now the ASSESSOR.\n"
            "Do not edit or rewrite the note.\n"
            "Use the thinking block for reasoning if needed.\n"
            "After reasoning, return exactly one final output token:\n"
            "- CORRECT\n"
            "- or the sentence number containing the single medical error\n"
            "Do not provide any extra final text beyond that answer."
        )

        if system_prompt:
            return f"{override}\n\n{system_prompt}\n\n{user_content}"
        return f"{override}\n\n{user_content}"

    def _sampling_with_max_tokens(
        self,
        sampling_params: dict[str, Any],
        *,
        max_new_tokens: int,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None,
    ) -> dict[str, Any]:
        updated = dict(sampling_params)
        updated["max_tokens"] = int(max_new_tokens)
        if temperature is not None:
            updated["temperature"] = float(temperature)
        if top_p is not None:
            updated["top_p"] = float(top_p)
        if top_k is not None:
            updated["top_k"] = int(top_k)
        if repetition_penalty is not None:
            updated["repetition_penalty"] = float(repetition_penalty)
        return updated

    def _single_turn_output(
        self,
        *,
        prompt_ids: list[int],
        injector_ids: list[int],
        injector_log_probs: Optional[list[float]],
        injector_reward: float,
        extra_fields: dict[str, Any],
        num_turns: int,
        total_time: float,
    ) -> AgentLoopOutput:
        token_level_scores = [0.0] * len(injector_ids)
        if injector_ids:
            token_level_scores[-1] = float(injector_reward)
        extra_fields = dict(extra_fields)
        extra_fields["generated_token_scores"] = token_level_scores
        extra_fields["turn_reward_spans"] = [
            {
                "role": "injector",
                "start": 0,
                "end": max(len(injector_ids) - 1, 0),
                "reward": float(injector_reward),
            }
        ]
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=injector_ids,
            response_mask=[1] * len(injector_ids),
            response_logprobs=injector_log_probs,
            reward_score=0.0,
            num_turns=num_turns,
            metrics=AgentLoopMetrics(generate_sequences=total_time, tool_calls=0.0),
            extra_fields=extra_fields,
        )

    @staticmethod
    def _assign_injector_reward(
        *,
        raw_injector_reward: float,
        assessor_reward: float,
        assessor_format_bonus: float,
        mode: str,
        game_valid: bool,
        gamma: float = 1.0,
    ) -> tuple[float, float, str]:
        """Assign reward to the injector span for adversarial self-play.

        The later assessor token reward is part of the same trajectory. We set
        the injector-span reward so the injector return after that future reward
        equals the desired game value.

        The task component is adversarial for every requested mode, including
        benign edits. Role-local format/compliance bonuses remain local: the
        injector keeps its own format bonus inside `raw_injector_reward`, while
        the assessor's format bonus is cancelled out of the adversarial term.
        """
        raw = float(raw_injector_reward)
        assessor_total = float(assessor_reward)
        assessor_task = assessor_total - float(assessor_format_bonus)
        mode = str(mode or "")

        if not game_valid:
            target_return = raw
            assigned_reward = target_return - gamma * assessor_total
            return assigned_reward, target_return, "uncoupled_game_invalid"

        if mode in {"benign", "error_injection"}:
            target_return = raw - assessor_task
            assigned_reward = target_return - gamma * assessor_total
            return assigned_reward, target_return, f"adversarial_{mode}"

        target_return = raw
        assigned_reward = target_return - gamma * assessor_total
        return assigned_reward, target_return, "uncoupled_unknown_mode"

    @staticmethod
    def _assessor_format_bonus(reward_result: Any) -> float:
        """Return the role-local assessor format bonus included in its reward."""
        if not getattr(reward_result, "has_valid_format", False):
            return 0.0
        if getattr(reward_result, "ground_truth", None) is None:
            return 0.0
        if getattr(reward_result, "outcome", "") == "partial_match" and getattr(reward_result, "pred_sid", None) is None:
            return 0.0
        return FORMAT_BONUS

    async def run(
        self,
        sampling_params: dict[str, Any],
        *,
        prompt,
        extra_info=None,
        reward_model=None,
        interaction_kwargs=None,
        **kwargs,
    ) -> AgentLoopOutput:
        del reward_model, kwargs
        t0 = time.perf_counter()

        prompt_messages = self._force_think_mode(self._coerce_messages(prompt))
        extra_info = dict(extra_info or {})
        interaction_kwargs = dict(interaction_kwargs or extra_info.get("interaction_kwargs") or {})

        request_id = str(uuid4())
        prompt_ids = self._tokenize_messages(prompt_messages, add_generation_prompt=True)

        injector_sampling = self._sampling_with_max_tokens(
            sampling_params,
            max_new_tokens=self.injector_max_new_tokens,
        )
        injector_output = await self._generate(request_id, prompt_ids, injector_sampling)
        injector_ids = list(injector_output.token_ids)
        injector_log_probs = (
            list(injector_output.log_probs) if getattr(injector_output, "log_probs", None) is not None else None
        )
        injector_text = self._decode(injector_ids)

        changed_sid, modified_sentence = parse_injector_compact(injector_text or "")
        original_sentences = parse_numbered_sentences(str(extra_info.get("sentences", "")))
        original_sentence = original_sentences.get(changed_sid or -1, "")
        parse_success = bool(changed_sid is not None and modified_sentence and original_sentence)

        mode = str(extra_info.get("mode", interaction_kwargs.get("mode", "error_injection")))
        note_id = str(extra_info.get("note_id", interaction_kwargs.get("note_id", "")))
        error_type = str(extra_info.get("error_type", interaction_kwargs.get("error_type", "")))

        if parse_success:
            modified_note = reconstruct_note(str(extra_info.get("sentences", "")), int(changed_sid), str(modified_sentence))
            judge_result = await judge_sentence_pair(
                note_id=note_id,
                error_type=error_type,
                changed_sid=changed_sid,
                original_sentence=original_sentence,
                modified_sentence=str(modified_sentence),
                judge_url=self.judge_url,
                judge_model=self.judge_model,
            )
        else:
            modified_note = str(extra_info.get("sentences", ""))
            judge_result = {
                "verdict": "ABSTAIN",
                "status": "parse_failure",
                "judge_score": 0.0,
                "reason": "parse_failure",
                "judge_output": "",
            }

        injector_reward, injector_outcome, game_valid = compute_injector_game_reward(
            mode,
            judge_result["verdict"],
            parse_success=parse_success,
            judge_status=judge_result.get("status"),
        )
        if not parse_success:
            assessor_ground_truth_override = "CORRECT"
        elif judge_result.get("status") in {"request_failed", "unexpected_error", "bad_response_shape", "no_json_verdict", "no_judge_url"}:
            assessor_ground_truth_override = None
        elif judge_result.get("status") == "semantic_abstain":
            assessor_ground_truth_override = None
        else:
            assessor_ground_truth_override = None

        base_metadata = {
            "phase": "game_complete",
            "requested_mode": mode,
            "mode": mode,
            "note_id": note_id,
            "error_type": error_type,
            "changed_sid": changed_sid,
            "original_sentences": str(extra_info.get("sentences", "")),
            "original_sentence": original_sentence,
            "modified_sentence": str(modified_sentence or ""),
            "modified_sentences": modified_note,
            "injector_output": injector_text[:4000],
            "judge_truth": judge_result["verdict"],
            "judge_verdict": judge_result["verdict"],
            "judge_status": judge_result.get("status", ""),
            "judge_score": float(judge_result["judge_score"]),
            "judge_reason": judge_result["reason"],
            "judge_output": judge_result["judge_output"],
            "injector_reward": float(injector_reward),
            "injector_outcome": injector_outcome,
            "game_valid": bool(game_valid),
            "injector_format_valid": bool(parse_success),
            "role": "injector",
        }

        assessor_prompt = self._construct_assessor_prompt(modified_note)
        conversation_before_assessor = prompt_messages + [{"role": "assistant", "content": injector_text}]
        conversation_with_assessor = conversation_before_assessor + [{"role": "user", "content": assessor_prompt}]
        assessor_prompt_ids = self._token_delta(conversation_before_assessor, conversation_with_assessor)

        turn2_prompt_ids = prompt_ids + injector_ids + assessor_prompt_ids
        assessor_sampling = self._sampling_with_max_tokens(
            sampling_params,
            max_new_tokens=self.assessor_max_new_tokens,
            temperature=self.assessor_temperature,
            top_p=self.assessor_top_p,
            top_k=self.assessor_top_k,
            repetition_penalty=self.assessor_repetition_penalty,
        )
        assessor_output = await self._generate(request_id, turn2_prompt_ids, assessor_sampling)
        assessor_ids = list(assessor_output.token_ids)
        assessor_log_probs = (
            list(assessor_output.log_probs) if getattr(assessor_output, "log_probs", None) is not None else None
        )
        assessor_text = self._decode(assessor_ids)
        assessor_cap_hit = len(assessor_ids) >= int(self.assessor_max_new_tokens)

        assessor_reward = compute_assessor_game_reward(
            assessor_text,
            judge_verdict=judge_result["verdict"],
            changed_sid=changed_sid,
            ground_truth_override=assessor_ground_truth_override,
        )
        assessor_format_bonus = self._assessor_format_bonus(assessor_reward)

        response_ids = injector_ids + assessor_prompt_ids + assessor_ids
        response_mask = (
            [1] * len(injector_ids)
            + [0] * len(assessor_prompt_ids)
            + [1] * len(assessor_ids)
        )
        response_logprobs = None
        if injector_log_probs is not None or assessor_log_probs is not None:
            response_logprobs = (
                list(injector_log_probs or [0.0] * len(injector_ids))
                + [0.0] * len(assessor_prompt_ids)
                + list(assessor_log_probs or [0.0] * len(assessor_ids))
            )

        gamma = 1.0  # approximate PPO discount factor
        assigned_injector_reward, injector_target_return, coupling_mode = self._assign_injector_reward(
            raw_injector_reward=float(injector_reward),
            assessor_reward=float(assessor_reward.reward),
            assessor_format_bonus=float(assessor_format_bonus),
            mode=mode,
            game_valid=bool(game_valid),
            gamma=gamma,
        )

        token_level_scores = [0.0] * len(response_ids)
        if injector_ids:
            token_level_scores[len(injector_ids) - 1] = assigned_injector_reward
        if assessor_ids:
            token_level_scores[len(response_ids) - 1] = float(assessor_reward.reward)

        ground_truth = assessor_ground_truth_override or derive_assessor_ground_truth(judge_result["verdict"], changed_sid)
        extra_fields = {
            **base_metadata,
            "assessor_ground_truth": ground_truth,
            "ground_truth": ground_truth,
            "assessor_output": assessor_text[:4000],
            "assessor_label": assessor_reward.label,
            "assessor_pred_sid": assessor_reward.pred_sid,
            "assessor_reward": float(assessor_reward.reward),
            "assessor_format_bonus": float(assessor_format_bonus),
            "assessor_task_reward": float(assessor_reward.reward) - float(assessor_format_bonus),
            "assessor_outcome": assessor_reward.outcome,
            "assessor_format_valid": bool(assessor_reward.has_valid_format),
            "assessor_cap_hit": bool(assessor_cap_hit),
            "injector_token_count": len(injector_ids),
            "assessor_prompt_token_count": len(assessor_prompt_ids),
            "assessor_token_count": len(assessor_ids),
            "injector_max_new_tokens": int(self.injector_max_new_tokens),
            "assessor_max_new_tokens": int(self.assessor_max_new_tokens),
            "injector_assigned_reward": float(assigned_injector_reward),
            "injector_target_return": float(injector_target_return),
            "injector_coupling_mode": coupling_mode,
            "generated_token_scores": token_level_scores,
            "turn_reward_spans": [
                {
                    "role": "injector",
                    "start": 0,
                    "end": max(len(injector_ids) - 1, 0),
                    "reward": float(assigned_injector_reward),
                    "target_return": float(injector_target_return),
                    "coupling_mode": coupling_mode,
                    "raw_reward": float(injector_reward),
                },
                {
                    "role": "assessor",
                    "start": len(injector_ids) + len(assessor_prompt_ids),
                    "end": max(len(response_ids) - 1, len(injector_ids) + len(assessor_prompt_ids)),
                    "reward": float(assessor_reward.reward),
                },
            ],
        }

        total_time = time.perf_counter() - t0
        _append_game_log(
            {
                "timestamp": datetime.now().isoformat(),
                **extra_fields,
            }
        )

        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            reward_score=0.0,
            num_turns=4,
            metrics=AgentLoopMetrics(generate_sequences=total_time, tool_calls=0.0),
            extra_fields=extra_fields,
        )
