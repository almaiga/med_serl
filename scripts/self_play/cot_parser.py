"""Chain-of-Thought parsing utilities for self-play game.

Handles extraction of public responses (stripping hidden CoT) and
parsing structured outputs from Injector and Assessor roles.

Injector output format (v4):
    <think>...</think>  (optional)
    N. <modified sentence text>

Assessor output format:
    <think>...</think>  (optional)
    CORRECT  or  <sentence_number>
"""

import re
import json
from typing import Optional
from dataclasses import dataclass

from scripts.self_play.utils import (
    strip_thinking,
    parse_assessor_answer,
    parse_injector_compact,
    reconstruct_note,
)


@dataclass
class InjectorOutput:
    """Parsed output from the Injector role."""
    raw_response: str
    thinking: Optional[str]       # Hidden CoT (not shown to Assessor)
    changed_sentence_id: Optional[int]   # 1-indexed sentence that was modified
    modified_text: Optional[str]         # New text for that sentence
    parse_success: bool


@dataclass
class AssessorOutput:
    """Parsed output from the Assessor role."""
    raw_response: str
    thinking: Optional[str]
    label: str                    # "CORRECT", "ERROR", or "UNKNOWN"
    sentence_id: Optional[int]   # 1-indexed sentence number if ERROR, else None
    parse_success: bool


def extract_thinking(text: str) -> tuple[Optional[str], str]:
    """Extract <think>...</think> content and return (thinking, rest).
    
    Returns:
        Tuple of (thinking_content, text_without_thinking)
    """
    thinking, rest = strip_thinking(text)
    return (thinking or None), rest


def extract_public_response(full_response: str) -> str:
    """Strip hidden CoT and return only public response for opponent.
    
    This is the key function for hidden CoT - the Assessor only sees
    the public portion of the Injector's output.
    """
    _, public = strip_thinking(full_response)
    return public


def parse_injector_output(response: str) -> InjectorOutput:
    """Parse Injector's compact output.
    
    Expected format (v4):
        <think>...</think>  (optional reasoning)
        N. <modified sentence text>
    
    Example:
        <think>I'll change sentence 3 to introduce a dosage error...</think>
        3. Patient was prescribed 500mg of amoxicillin twice daily.
    """
    thinking, _ = strip_thinking(response)
    sid, modified = parse_injector_compact(response)
    
    return InjectorOutput(
        raw_response=response,
        thinking=thinking or None,
        changed_sentence_id=sid,
        modified_text=modified,
        parse_success=sid is not None and modified is not None,
    )


def extract_note_for_assessor(
    injector_response: str,
    original_sentences: str,
) -> str:
    """Build the full modified note for the Assessor from injector output.
    
    The Assessor sees the complete numbered note with the injector's change
    applied. All CoT reasoning is stripped.
    
    Args:
        injector_response: Raw injector model output.
        original_sentences: The original numbered sentences string.
    
    Returns:
        Full numbered note with the injector's modification applied.
        Returns original_sentences unchanged if parsing fails.
    """
    parsed = parse_injector_output(injector_response)
    
    if not parsed.parse_success:
        return original_sentences
    
    return reconstruct_note(
        original_sentences, parsed.changed_sentence_id, parsed.modified_text
    )


def parse_assessor_output(response: str) -> AssessorOutput:
    """Parse Assessor's output: CORRECT or a sentence number.
    
    Expected format:
        <think>...</think>  (optional)
        CORRECT
        or
        7
    """
    thinking, _ = strip_thinking(response)
    label, sentence_id = parse_assessor_answer(response)
    
    return AssessorOutput(
        raw_response=response,
        thinking=thinking or None,
        label=label,
        sentence_id=sentence_id,
        parse_success=label != "UNKNOWN",
    )


def validate_injector_output(
    parsed: InjectorOutput,
    total_sentences: int,
) -> bool:
    """Check if injector output is valid.
    
    Returns True if:
    - Parse succeeded
    - Sentence ID is within range [1, total_sentences]
    - Modified text is non-empty
    """
    if not parsed.parse_success:
        return False
    if parsed.changed_sentence_id < 1 or parsed.changed_sentence_id > total_sentences:
        return False
    if not parsed.modified_text or len(parsed.modified_text.strip()) < 5:
        return False
    return True
