"""Medical Error Detection + Localization Game Interaction for verl multi-turn RL training.

This interaction orchestrates the two-turn self-play game:
1. Turn 1 (Injector): Model receives numbered sentences and outputs "N. <modified sentence>"
2. Turn 2 (Assessor): Model receives the modified numbered note and outputs CORRECT or a sentence number

The Assessor sees the full modified note (CoT is stripped from Injector output).

Per-turn sampling:
- Injector: temperature=0.6, top_p=0.95 (Qwen3 recommendation for thinking mode)
- Assessor: temperature=0.3, top_p=0.95 (lower temperature for precise classification)

Rewards are 3-tier:
- Exact match (CORRECT↔CORRECT, or same sentence number): +1.0
- Detection only (error detected but wrong sentence number): +0.3
- Miss (wrong or unparseable): -1.0
- Format bonus: +0.2 for valid output format
"""

import json
import logging
import re
import threading
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from uuid import uuid4

from scripts.self_play.utils import (
    number_sentences,
    parse_assessor_answer,
    reconstruct_note,
    strip_thinking,
)
from scripts.self_play.cot_parser import (
    parse_injector_output,
    extract_note_for_assessor,
)

logger = logging.getLogger(__name__)

# Import verl's BaseInteraction
try:
    from verl.interactions.base import BaseInteraction
except ImportError:
    class BaseInteraction:
        def __init__(self, config: dict):
            self.config = config
            self.name = config.get("name", "interaction_agent")

# Reward constants
REWARD_EXACT = 1.0      # Exact match (correct label + sentence)
REWARD_PARTIAL = 0.3    # Detected error but wrong sentence number
REWARD_MISS = -1.0      # Missed or wrong classification
FORMAT_BONUS = 0.2      # Bonus for parseable output format

# Per-turn sampling parameters
# Injector: moderate creativity for generating plausible modifications
INJECTOR_TEMPERATURE = 0.6   # Qwen3 official recommendation for thinking mode
INJECTOR_TOP_P = 0.95        # Qwen3 official recommendation
# Assessor: low temperature for precise, deterministic classification
ASSESSOR_TEMPERATURE = 0.3
ASSESSOR_TOP_P = 0.95
ASSESSOR_MAX_NEW_TOKENS = 256


# JSONL logging — written by the interaction (source of truth for rewards)
# Log path is resolved lazily at first write so that MEDSERL_GAME_LOG set via
# Ray runtime_env.env_vars is always picked up (env vars from runtime_env arrive
# before the worker executes tasks, but after the module may have been imported).
import os as _os
_FALLBACK_LOG_DIR = Path(__file__).parent.parent.parent.parent / "results" / "self_play" / "interactions"
_log_lock = threading.Lock()
_resolved_log_file: "Path | None" = None  # resolved on first write


def _get_log_file() -> Path:
    """Resolve the log file path once, lazily, so runtime_env env vars are seen."""
    global _resolved_log_file
    if _resolved_log_file is not None:
        return _resolved_log_file
    env_log = _os.environ.get("MEDSERL_GAME_LOG")
    if env_log:
        p = Path(env_log)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            _resolved_log_file = p
            print(f"[MedicalGameInteraction] game log → {p} (from MEDSERL_GAME_LOG)", flush=True)
            return _resolved_log_file
        except Exception as e:
            print(f"[MedicalGameInteraction] WARNING: cannot create {p.parent}: {e} — using fallback", flush=True)
    _FALLBACK_LOG_DIR.mkdir(parents=True, exist_ok=True)
    _resolved_log_file = _FALLBACK_LOG_DIR / f"game_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"[MedicalGameInteraction] game log → {_resolved_log_file} (fallback)", flush=True)
    return _resolved_log_file


def _write_log(entry: dict) -> None:
    try:
        log_file = _get_log_file()
        with _log_lock:
            with open(log_file, "a") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[MedicalGameInteraction] _write_log FAILED: {e}", flush=True)
        logger.warning("_write_log failed: %s", e)


class MedicalGameInteraction(BaseInteraction):
    """Two-turn medical error detection + localization game interaction.
    
    Phase 1: Injector modifies a clinical note (benign or error injection)
             Output: "N. <modified sentence>"
             Sampling: temperature=0.6, top_p=0.95
    Phase 2: Assessor classifies the modified note
             Output: "CORRECT" or "<sentence_number>"
             Sampling: temperature=0.3, top_p=0.95
    
    Rewards are 3-tier:
    - Exact match: +1.0 + format_bonus
    - Partial (detected error, wrong sentence): +0.3 + format_bonus
    - Miss: -1.0 (no format bonus for invalid format)
    """
    
    def __init__(self, config: Dict[str, Any]):
        super().__init__(config)
        self.name: str = config.get("name", "medical_game")
        self._instance_dict = {}
        
        # Per-turn sampling (overridable via config)
        self.injector_temperature = config.get("injector_temperature", INJECTOR_TEMPERATURE)
        self.injector_top_p = config.get("injector_top_p", INJECTOR_TOP_P)
        self.assessor_temperature = config.get("assessor_temperature", ASSESSOR_TEMPERATURE)
        self.assessor_top_p = config.get("assessor_top_p", ASSESSOR_TOP_P)
        self.assessor_repetition_penalty = config.get("assessor_repetition_penalty", 1.1)
        self.assessor_max_new_tokens = int(
            _os.environ.get(
                "MEDSERL_ASSESSOR_MAX_NEW_TOKENS",
                config.get("assessor_max_new_tokens", ASSESSOR_MAX_NEW_TOKENS),
            )
        )
        
        # Load detection+localization prompts
        self.detection_prompts = self._load_prompts(
            config.get("detection_prompts_path", 
                      "configs/prompts/detection_localization_prompts.json")
        )
        
    def _load_prompts(self, path: str) -> dict:
        """Load prompts from JSON config file."""
        with open(path, 'r') as f:
            return json.load(f)
    
    async def start_interaction(
        self, 
        instance_id: Optional[str] = None,
        ground_truth: Optional[str] = None,
        mode: Optional[str] = None,
        note_data: Optional[dict] = None,
        sentences: Optional[str] = None,
        correct_note: Optional[str] = None,
        note_id: Optional[str] = None,
        error_type: Optional[str] = None,
        error_sentence_id: Optional[int] = None,
        **kwargs
    ) -> str:
        """Initialize interaction session.
        
        Args:
            instance_id: Unique session ID (auto-generated if None)
            ground_truth: "CORRECT" or str(sentence_id)
            mode: "benign" or "error_injection"
            sentences: Pre-numbered sentences string
            correct_note: The original correct note text (fallback if sentences missing)
            error_sentence_id: 1-indexed sentence number with the error
        """
        if instance_id is None:
            instance_id = str(uuid4())
        
        # Build sentences if not provided
        if not sentences and correct_note:
            sentences = number_sentences(correct_note)
        
        if note_data is None:
            note_data = {
                "correct_note": correct_note or "",
                "note_id": note_id or "",
                "error_type": error_type or "",
                "error_sentence_id": error_sentence_id,
            }
        
        self._instance_dict[instance_id] = {
            "ground_truth": ground_truth or kwargs.get("ground_truth"),
            "mode": mode or kwargs.get("mode"),
            "note_data": note_data,
            "sentences": sentences or "",
            "injector_output": None,
            "modified_sentences": None,
            "changed_sid": None,
            "assessor_output": None,
            "turn": 0,
        }
        
        return instance_id
    
    async def generate_response(
        self, 
        instance_id: str, 
        messages: List[Dict[str, Any]], 
        **kwargs
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        """Process model response and generate next prompt or terminate.
        
        Returns (done, next_prompt_or_feedback, reward, info).
        ``info`` may contain ``sampling_params`` dict to override temperature/top_p
        for the next generation turn.
        """
        instance = self._instance_dict[instance_id]
        current_turn = instance["turn"]
        instance["turn"] += 1

        if current_turn == 0:
            return await self._process_injector_turn(instance_id, messages)
        elif current_turn == 1:
            return await self._process_assessor_turn(instance_id, messages)
        else:
            return True, "Error: Invalid turn number", 0.0, {}
    
    async def _process_injector_turn(
        self,
        instance_id: str,
        messages: List[Dict[str, Any]]
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        """Process Phase 1 (Injector) output and prepare Phase 2 (Assessor) prompt.

        Parses compact "N. <modified sentence>" output, reconstructs the full
        note, and passes it to the Assessor with proper system prompt.

        The injector receives a per-turn proxy reward (MEDEC ground-truth based)
        applied to its tokens immediately, matching the compute_score(role="injector")
        logic in reward_function.py:
          - Valid format, benign mode          : +1.2  (REWARD_EXACT + FORMAT_BONUS)
          - Valid format, correct sentence      : +1.2
          - Valid format, wrong sentence        : +0.5  (REWARD_PARTIAL + FORMAT_BONUS)
          - Invalid format / early termination  : -1.0  (REWARD_MISS)

        Returns sampling_params for the assessor turn (temperature=0.3).
        """
        instance = self._instance_dict[instance_id]

        # Extract Injector's response (last assistant message)
        injector_output = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                injector_output = msg.get("content", "")
                break

        instance["injector_output"] = injector_output

        # Parse compact output: "N. <modified sentence>"
        parsed = parse_injector_output(injector_output)

        if parsed.parse_success:
            instance["changed_sid"] = parsed.changed_sentence_id
            modified = reconstruct_note(
                instance["sentences"], parsed.changed_sentence_id, parsed.modified_text
            )
            instance["modified_sentences"] = modified
            logger.debug(
                "Injector parsed: sid=%d, modified=%r",
                parsed.changed_sentence_id, (parsed.modified_text or "")[:80],
            )
            # MEDEC-proxy reward: reward injector for correct sentence placement
            injector_reward = self._compute_injector_proxy_reward(
                parsed.changed_sentence_id, instance
            )
        else:
            # Fallback: pass original sentences (injector failed to modify)
            instance["modified_sentences"] = extract_note_for_assessor(
                injector_output, instance["sentences"]
            )
            if not instance["modified_sentences"]:
                return True, "Invalid output format. Expected 'N. <modified sentence>'.", REWARD_MISS, {}
            logger.warning("Injector parse failed — using fallback note for assessor.")
            injector_reward = REWARD_MISS  # penalise invalid format

        instance["injector_reward"] = injector_reward

        # Construct Assessor prompt with system context + modified note
        assessor_prompt = self._construct_assessor_prompt(instance["modified_sentences"])

        # Pass per-turn sampling params for the assessor turn
        info = {
            "phase": "injector_complete",
            "injector_reward": injector_reward,
            "sampling_params": {
                "temperature": self.assessor_temperature,
                "top_p": self.assessor_top_p,
                "repetition_penalty": self.assessor_repetition_penalty,
                "max_new_tokens": self.assessor_max_new_tokens,
            },
        }

        # Return the injector's per-turn reward so verl applies it to injector tokens.
        return False, assessor_prompt, injector_reward, info

    def _compute_injector_proxy_reward(self, changed_sid: int, instance: dict) -> float:
        """MEDEC-proxy reward for the injector's sentence placement quality.

        Mirrors the compute_score(role="injector") logic in reward_function.py
        so both training paths (chained-data and multi-turn) assign consistent
        injector rewards.
        """
        mode = instance.get("mode", "error_injection")
        if mode == "benign":
            # Any valid benign edit is rewarded identically
            return REWARD_EXACT + FORMAT_BONUS  # +1.2

        note_data = instance.get("note_data", {})
        expected_sid = note_data.get("error_sentence_id")
        if expected_sid is not None and changed_sid == int(expected_sid):
            return REWARD_EXACT + FORMAT_BONUS   # +1.2 — correct sentence
        elif expected_sid is not None:
            return REWARD_PARTIAL + FORMAT_BONUS # +0.5 — valid format, wrong sentence
        else:
            return REWARD_PARTIAL + FORMAT_BONUS # +0.5 — no GT available, reward valid format
    
    async def _process_assessor_turn(
        self,
        instance_id: str,
        messages: List[Dict[str, Any]]
    ) -> Tuple[bool, str, float, Dict[str, Any]]:
        """Process Phase 2 (Assessor) output and compute 3-tier reward."""
        instance = self._instance_dict[instance_id]
        
        # Extract Assessor's response
        assessor_output = ""
        for msg in reversed(messages):
            if msg.get("role") == "assistant":
                assessor_output = msg.get("content", "")
                break
        
        instance["assessor_output"] = assessor_output
        
        # Parse Assessor's answer
        label, pred_sid = parse_assessor_answer(assessor_output)
        ground_truth = instance["ground_truth"]
        
        # Check format compliance
        has_valid_format = label != "UNKNOWN"
        format_bonus = FORMAT_BONUS if has_valid_format else 0.0
        
        # 3-tier reward
        assessor_reward, feedback = self._compute_reward(
            label, pred_sid, ground_truth, format_bonus
        )

        # Determine outcome label for logging
        gt_is_correct = (ground_truth == "CORRECT")
        if label == "UNKNOWN":
            outcome = "invalid_format"
        elif gt_is_correct and label == "CORRECT":
            outcome = "exact_match"
        elif not gt_is_correct and label == "ERROR":
            pred_str = str(pred_sid) if pred_sid else ""
            outcome = "exact_match" if pred_str == ground_truth else "partial_match"
        else:
            outcome = "miss"

        # Zero-sum adversarial signal:
        #   error mode:  injector "wins" when assessor misses → injector_retroactive = +R
        #                injector "loses" when assessor detects → injector_retroactive = -R
        #   benign mode: cooperative — both get the same direction signal
        mode = instance.get("mode", "error_injection")
        if mode == "error_injection":
            injector_retroactive = -assessor_reward  # zero-sum
        else:
            injector_retroactive = assessor_reward   # cooperative on benign notes

        note_data = instance.get("note_data", {})
        assessor_public_answer = self._extract_public_answer(assessor_output)
        _write_log({
            "timestamp": datetime.now().isoformat(),
            "data_source": "medec_selfplay",
            "ground_truth": ground_truth,
            "assessor_label": label,
            "assessor_pred_sid": pred_sid,
            "assessor_public_answer": assessor_public_answer,
            "outcome": outcome,
            "reward": float(assessor_reward),
            "has_valid_format": has_valid_format,
            "mode": mode,
            "note_id": note_data.get("note_id", ""),
            "error_type": note_data.get("error_type", ""),
            # Per-turn rewards for analysis / future retroactive assignment
            "injector_reward": float(instance.get("injector_reward", 0.0)),
            "injector_retroactive_reward": float(injector_retroactive),
            # Per-turn responses (correctly split — no parsing needed)
            "injector_response": (instance.get("injector_output") or "")[:4000],
            "modified_sentences": (instance.get("modified_sentences") or "")[:4000],
            "assessor_response": assessor_output[:4000],
            "assessor_actually_ran": True,
            "phases_separated": True,
        })

        return True, feedback, assessor_reward, {
            "phase": "game_complete",
            "assessor_label": label,
            "assessor_pred_sid": pred_sid,
            "ground_truth": ground_truth,
            "has_valid_format": has_valid_format,
            "mode": mode,
            "note_id": note_data.get("note_id", ""),
            "modified_sentences": instance.get("modified_sentences", ""),
            "injector_output": (instance.get("injector_output") or "")[:4000],
            "assessor_output": assessor_output[:4000],
            "assessor_public_answer": assessor_public_answer,
            "phases_separated": True,
            # Zero-sum injector penalty — available for retroactive reward assignment
            # if the verl tool_agent_loop is extended to support per-turn retroactive rewards.
            "injector_retroactive_reward": float(injector_retroactive),
        }
    
    def _compute_reward(
        self,
        label: str,
        pred_sid: Optional[int],
        ground_truth: str,
        format_bonus: float,
    ) -> Tuple[float, str]:
        """Compute 3-tier reward for the Assessor.
        
        Returns (reward, feedback_message).
        """
        gt_is_correct = (ground_truth == "CORRECT")

        # Case 1: Note is correct and assessor says CORRECT → exact match
        if gt_is_correct and label == "CORRECT":
            return REWARD_EXACT + format_bonus, "Correct! The note has no errors."

        # Case 2: Note is correct but assessor says error → false positive
        if gt_is_correct and label != "CORRECT":
            return REWARD_MISS + format_bonus, f"Incorrect. The note was CORRECT, but you flagged sentence {pred_sid}."

        # Case 3: Note has error, assessor says CORRECT → missed error
        if not gt_is_correct and label == "CORRECT":
            return REWARD_MISS + format_bonus, f"Incorrect. There was an error at sentence {ground_truth}, but you said CORRECT."

        # Case 4: Note has error, assessor detected error
        if not gt_is_correct and label == "ERROR":
            pred_str = str(pred_sid) if pred_sid else "?"
            if pred_str == ground_truth:
                # Exact sentence match
                return REWARD_EXACT + format_bonus, f"Correct! Error at sentence {ground_truth}."
            elif pred_sid is not None:
                # Detected error but wrong sentence number → partial credit
                return REWARD_PARTIAL + format_bonus, f"Partially correct. Error at sentence {ground_truth}, you said {pred_str}."
            else:
                # Detected error but couldn't determine sentence → partial (lower)
                return REWARD_PARTIAL, f"Detected error but no sentence number. Error was at sentence {ground_truth}."

        # Fallback: UNKNOWN or unparseable
        return REWARD_MISS, f"Invalid response format. Expected 'CORRECT' or a sentence number."
    
    def _construct_assessor_prompt(self, modified_sentences: str) -> str:
        """Construct the Assessor's classification prompt.
        
        The original injector system prompt is still present earlier in the
        multi-turn conversation, so this prompt must explicitly override it and
        force a role switch.
        """
        system_prompt = self.detection_prompts.get("system_prompt", "")
        user_template = self.detection_prompts.get("user_template", "{sentences}")
        user_content = user_template.format(sentences=modified_sentences)

        override = (
            "/no_think\n"
            "IGNORE ALL PREVIOUS TASK INSTRUCTIONS.\n"
            "The injector step is finished.\n"
            "You are now the ASSESSOR.\n"
            "Do not edit or rewrite the note.\n"
            "Return exactly one token of output:\n"
            "- CORRECT\n"
            "- or the sentence number containing the single medical error\n"
            "Do not provide analysis, explanations, or any extra text."
        )

        if system_prompt:
            return f"{override}\n\n{system_prompt}\n\n{user_content}"
        return f"{override}\n\n{user_content}"

    def _extract_public_answer(self, text: str) -> str:
        """Return the visible assessor answer without hidden thinking."""
        _, content = strip_thinking(text)
        content = content.strip()
        if not content:
            return ""
        lines = [line.strip() for line in content.splitlines() if line.strip()]
        if not lines:
            return ""
        return lines[0][:200]
    
    async def calculate_score(
        self, 
        instance_id: str, 
        **kwargs
    ) -> float:
        """Calculate final score for the interaction."""
        instance = self._instance_dict.get(instance_id, {})
        
        if instance.get("turn", 1) != 2:
            return 0.0
        
        assessor_output = instance.get("assessor_output", "")
        label, pred_sid = parse_assessor_answer(assessor_output)
        ground_truth = instance.get("ground_truth", "")
        
        has_valid_format = label != "UNKNOWN"
        format_bonus = FORMAT_BONUS if has_valid_format else 0.0
        
        reward, _ = self._compute_reward(label, pred_sid, ground_truth, format_bonus)
        return reward
    
    async def finalize_interaction(
        self, 
        instance_id: str, 
        **kwargs
    ) -> None:
        """Clean up resources for this interaction session."""
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
