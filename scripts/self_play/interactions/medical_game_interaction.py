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
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from uuid import uuid4

from scripts.self_play.utils import (
    number_sentences,
    parse_assessor_answer,
    reconstruct_note,
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
        else:
            # Fallback: pass original sentences (injector failed to modify)
            instance["modified_sentences"] = extract_note_for_assessor(
                injector_output, instance["sentences"]
            )
            if not instance["modified_sentences"]:
                return True, "Invalid output format. Expected 'N. <modified sentence>'.", -1.0, {}
            logger.warning("Injector parse failed — using fallback note for assessor.")
        
        # Construct Assessor prompt with system context + modified note
        assessor_prompt = self._construct_assessor_prompt(instance["modified_sentences"])
        
        # Pass per-turn sampling params for the assessor turn
        info = {
            "phase": "injector_complete",
            "sampling_params": {
                "temperature": self.assessor_temperature,
                "top_p": self.assessor_top_p,
            },
        }
        
        return False, assessor_prompt, 0.0, info
    
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
        
        return True, feedback, assessor_reward, {
            "phase": "game_complete",
            "assessor_label": label,
            "assessor_pred_sid": pred_sid,
            "ground_truth": ground_truth,
            "has_valid_format": has_valid_format,
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
        
        Includes the system-level instructions (output CORRECT or sentence number)
        prepended to the modified note so the model knows to switch from Injector
        to Assessor role.  verl's multi-turn system inserts this as the next
        ``user`` turn in the conversation.
        """
        system_prompt = self.detection_prompts.get("system_prompt", "")
        user_template = self.detection_prompts.get("user_template", "{sentences}")
        user_content = user_template.format(sentences=modified_sentences)
        
        # Combine system instructions with the note in a single user message.
        # The system_prompt already ends with /no_think for Qwen3.
        if system_prompt:
            return f"{system_prompt}\n\n{user_content}"
        return user_content
    
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
