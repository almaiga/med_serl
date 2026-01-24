"""Chain-of-Thought parsing utilities for self-play game.

Handles extraction of public responses (stripping hidden CoT) and
parsing structured outputs from Injector and Assessor roles.
"""

import re
import json
from typing import Optional
from dataclasses import dataclass


@dataclass
class InjectorOutput:
    """Parsed output from the Injector role."""
    raw_response: str
    thinking: Optional[str]  # Hidden CoT (not shown to Assessor)
    generated_note: Optional[str]
    final_answer: Optional[str]  # "CORRECT" or "INCORRECT"
    changes_made: Optional[dict]
    parse_success: bool


@dataclass
class AssessorOutput:
    """Parsed output from the Assessor role."""
    raw_response: str
    thinking: Optional[str]
    final_answer: Optional[str]  # "CORRECT" or "INCORRECT"
    explanation: Optional[str]
    parse_success: bool


def extract_thinking(text: str) -> tuple[Optional[str], str]:
    """Extract <think>...</think> content and return (thinking, rest).
    
    Returns:
        Tuple of (thinking_content, text_without_thinking)
    """
    pattern = r'<think>(.*?)</think>'
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    
    if match:
        thinking = match.group(1).strip()
        rest = re.sub(pattern, '', text, flags=re.DOTALL | re.IGNORECASE).strip()
        return thinking, rest
    
    return None, text


def extract_public_response(full_response: str) -> str:
    """Strip hidden CoT and return only public response for opponent.
    
    This is the key function for hidden CoT - the Assessor only sees
    the public portion of the Injector's output.
    """
    _, public = extract_thinking(full_response)
    return public


def parse_injector_output(response: str) -> InjectorOutput:
    """Parse Injector's structured output.
    
    Expected format:
        <think>...</think>  (optional)
        
        generated_note:
        [the clinical note]
        
        final_answer: "CORRECT" or "INCORRECT"
        
        changes_made:
        {"original_sentence": "...", "modified_sentence": "...", ...}
    """
    thinking, rest = extract_thinking(response)
    
    # Extract generated_note - try multiple patterns for robustness
    generated_note = None
    note_patterns = [
        r'generated_note:\s*\n(.*?)(?=\n\s*final_answer:|\n\s*changes_made:|$)',
        r'generated_note:\s*(.*?)(?=final_answer:|changes_made:|$)',
    ]
    
    for pattern in note_patterns:
        note_match = re.search(pattern, rest, re.DOTALL | re.IGNORECASE)
        if note_match:
            generated_note = note_match.group(1).strip()
            if generated_note:
                break
    
    # Extract final_answer
    final_answer = None
    answer_pattern = r'final_answer:\s*["\']?(CORRECT|INCORRECT)["\']?'
    answer_match = re.search(answer_pattern, rest, re.IGNORECASE)
    if answer_match:
        final_answer = answer_match.group(1).upper()
    
    # Extract changes_made JSON
    changes_made = None
    changes_pattern = r'changes_made:\s*\n?\s*(\{.*?\})'
    changes_match = re.search(changes_pattern, rest, re.DOTALL)
    if changes_match:
        try:
            changes_made = json.loads(changes_match.group(1))
        except json.JSONDecodeError:
            # Try to fix common JSON issues
            json_str = changes_match.group(1)
            json_str = re.sub(r'(\w+):', r'"\1":', json_str)  # Add quotes to keys
            try:
                changes_made = json.loads(json_str)
            except json.JSONDecodeError:
                pass
    
    parse_success = generated_note is not None and final_answer is not None
    
    return InjectorOutput(
        raw_response=response,
        thinking=thinking,
        generated_note=generated_note,
        final_answer=final_answer,
        changes_made=changes_made,
        parse_success=parse_success,
    )


def extract_note_for_assessor(injector_response: str) -> str:
    """Extract ONLY the generated note for Assessor - hiding all other information.
    
    This is the key function for the Hidden CoT design (SeRL paper).
    The Assessor should see ONLY the clinical note text, with:
    - NO <think>...</think> reasoning
    - NO final_answer declaration
    - NO changes_made metadata
    
    Returns:
        The sanitized clinical note text only
    """
    parsed = parse_injector_output(injector_response)
    
    if not parsed.generated_note:
        return ""
    
    note = parsed.generated_note
    
    # Sanitize: remove any leaked information that might hint at the answer
    # Remove any stray final_answer or CORRECT/INCORRECT keywords
    note = re.sub(r'final_answer:\s*["\']?(?:CORRECT|INCORRECT)["\']?', '', note, flags=re.IGNORECASE)
    note = re.sub(r'changes_made:\s*\{.*?\}', '', note, flags=re.DOTALL | re.IGNORECASE)
    note = re.sub(r'\n\s*"?(CORRECT|INCORRECT)"?\s*$', '', note, flags=re.IGNORECASE)
    
    return note.strip()


def parse_assessor_output(response: str) -> AssessorOutput:
    """Parse Assessor's structured output.
    
    Expected format:
        <think>...</think>  (optional)
        
        final_answer: "CORRECT" or "INCORRECT"
        Explanation: [one sentence]
    """
    thinking, rest = extract_thinking(response)
    
    # Extract final_answer
    final_answer = None
    answer_pattern = r'final_answer:\s*["\']?(CORRECT|INCORRECT)["\']?'
    answer_match = re.search(answer_pattern, rest, re.IGNORECASE)
    if answer_match:
        final_answer = answer_match.group(1).upper()
    
    # Extract explanation
    explanation = None
    explanation_pattern = r'[Ee]xplanation:\s*(.+?)(?:\n|$)'
    explanation_match = re.search(explanation_pattern, rest)
    if explanation_match:
        explanation = explanation_match.group(1).strip()
    
    parse_success = final_answer is not None
    
    return AssessorOutput(
        raw_response=response,
        thinking=thinking,
        final_answer=final_answer,
        explanation=explanation,
        parse_success=parse_success,
    )


def validate_injector_note(
    original_note: str,
    generated_note: str,
    max_edit_distance_ratio: float = 0.3,
) -> bool:
    """Check if generated note is a valid minimal edit of original.
    
    Returns True if the edit is within acceptable bounds.
    Used for sanity checking, not for reward computation in v1.
    """
    if not generated_note or not original_note:
        return False
    
    # Simple length-based check
    len_ratio = len(generated_note) / len(original_note)
    if len_ratio < 0.7 or len_ratio > 1.3:
        return False
    
    return True
