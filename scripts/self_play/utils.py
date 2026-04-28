"""Shared utilities for self-play pipeline.

Provides canonical functions for:
- Numbering clinical note sentences (1-indexed)
- Parsing assessor output (CORRECT or sentence number)
- Finding error sentence IDs by text matching
- Parsing injector compact output (N. <modified sentence>)
- Reconstructing a full note after injector modification
"""

import re
from difflib import SequenceMatcher
from typing import Optional, Tuple


# ─────────────────────────────────────────────────────────────────────────────
# Sentence numbering
# ─────────────────────────────────────────────────────────────────────────────

def split_sentences(note: str) -> list[str]:
    """Split a clinical note into sentences.

    Handles common clinical note patterns:
    - Period-delimited sentences
    - Already-numbered lines ("0 First sentence.\\n1 Second.")
    - One sentence per line
    """
    note = note.strip()
    if not note:
        return []

    lines = note.split("\n")

    # Check if already numbered (MEDEC 0-indexed format: "0 text")
    numbered = []
    for line in lines:
        m = re.match(r"^\d+\s+(.+)$", line.strip())
        if m:
            numbered.append(m.group(1).strip())
    if numbered and len(numbered) == len([l for l in lines if l.strip()]):
        return numbered

    # Check if multi-line (one sentence per line, each ≥ 20 chars)
    non_empty = [l.strip() for l in lines if l.strip()]
    if len(non_empty) > 1 and all(len(l) >= 20 for l in non_empty):
        return non_empty

    # Fallback: split on sentence-ending punctuation followed by space
    # Handles abbreviations like "Dr." / "vs." by requiring next char to be uppercase
    raw = " ".join(non_empty)
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])', raw)
    return [p.strip() for p in parts if p.strip()]


def number_sentences(note: str) -> str:
    """Convert a clinical note into 1-indexed numbered sentences.

    "First sentence. Second sentence." → "1. First sentence.\\n2. Second sentence."
    """
    sentences = split_sentences(note)
    return "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))


def sentences_to_1indexed(raw: str) -> str:
    """Convert MEDEC 0-indexed sentences to 1-indexed.

    "0 First sentence.\\n1 Second." → "1. First sentence.\\n2. Second."
    Reused from medrect/inference_detection.py for compatibility.
    """
    lines = []
    for line in raw.strip().split("\n"):
        m = re.match(r"^(\d+)\s+(.+)$", line.strip())
        lines.append(f"{int(m.group(1)) + 1}. {m.group(2)}" if m else line)
    return "\n".join(lines)


def parse_numbered_sentences(text: str) -> dict[int, str]:
    """Parse numbered sentences back into a dict {sentence_id: text}.

    "1. First sentence.\\n2. Second." → {1: "First sentence.", 2: "Second."}
    """
    result = {}
    for line in text.strip().split("\n"):
        m = re.match(r"^(\d+)\.\s+(.+)$", line.strip())
        if m:
            result[int(m.group(1))] = m.group(2).strip()
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Assessor output parsing
# ─────────────────────────────────────────────────────────────────────────────

def strip_thinking(content: str) -> Tuple[str, str]:
    """Strip <think>...</think> from content.

    Returns (thinking_text, answer_text).
    """
    m = re.search(r"<think>(.*?)</think>\s*", content, re.DOTALL)
    if m:
        return m.group(1).strip(), content[m.end():].strip()
    return "", content


def parse_assessor_answer(text: str) -> Tuple[str, Optional[int]]:
    """Canonical parser for assessor output.

    Returns:
        ("CORRECT", None)  — note is correct
        ("ERROR", sentence_id)  — error at sentence_id (1-indexed)
        ("UNKNOWN", None)  — could not parse
    """
    _, content = strip_thinking(text)

    # Use first non-empty line; strip known prefixes
    answer = ""
    for line in content.split("\n"):
        line = line.strip()
        if line:
            answer = re.sub(
                r"^(answer|label|output|result|final_answer)\s*[:=]\s*",
                "", line, flags=re.IGNORECASE,
            )
            break

    # Explicit CORRECT (but not INCORRECT)
    if re.search(r"\bcorrect\b", answer, re.IGNORECASE) and not re.search(
        r"\bincorrect\b", answer, re.IGNORECASE
    ):
        return "CORRECT", None

    # Sentence number
    m = re.search(r"\b(\d+)\b", answer)
    if m:
        return "ERROR", int(m.group(1))

    # Fallback: error-ish keywords
    if re.search(r"error|incorrect|mistake|wrong", answer, re.IGNORECASE):
        return "ERROR", None

    return "UNKNOWN", None


# ─────────────────────────────────────────────────────────────────────────────
# Injector output parsing (compact format: "N. <modified sentence>")
# ─────────────────────────────────────────────────────────────────────────────

_INJECTOR_META_PHRASES = (
    "identify the target",
    "identify target",
    "review each sentence",
    "the best target",
    "select the sentence",
    "choose the sentence",
    "pick the sentence",
    "find the sentence",
    "step-by-step",
    "step by step",
)


def parse_injector_compact(text: str) -> Tuple[Optional[int], Optional[str]]:
    """Parse injector compact output: 'N. <modified sentence text>'.

    Strips <think>...</think> first, then extracts the sentence line.
    Rejects lines that are meta-reasoning rather than an actual clinical
    sentence (e.g. "3. Identify the target sentence for modification.").

    Returns:
        (sentence_id, modified_text) or (None, None) on failure.
    """
    _, content = strip_thinking(text)

    # Search all lines for pattern "N. text"
    for line in content.strip().split("\n"):
        line = line.strip()
        m = re.match(r"^(\d+)\.\s+(.+)$", line)
        if m:
            candidate = m.group(2).strip()
            low = candidate.lower()
            # Reject lines that are injector meta-reasoning leaked into output
            if any(phrase in low for phrase in _INJECTOR_META_PHRASES):
                continue
            return int(m.group(1)), candidate

    return None, None


def reconstruct_note(
    original_sentences: str,
    changed_sid: int,
    modified_text: str,
) -> str:
    """Reconstruct the full numbered note after injector modification.

    Args:
        original_sentences: The original numbered sentences string
            ("1. First.\\n2. Second.\\n3. Third.")
        changed_sid: 1-indexed sentence number that was changed
        modified_text: The new text for that sentence

    Returns:
        Full numbered note with the specified sentence replaced.
    """
    sentences = parse_numbered_sentences(original_sentences)
    if changed_sid not in sentences:
        # If sentence ID is out of range, return original
        return original_sentences
    sentences[changed_sid] = modified_text
    return "\n".join(f"{sid}. {txt}" for sid, txt in sorted(sentences.items()))


# ─────────────────────────────────────────────────────────────────────────────
# Error sentence ID detection
# ─────────────────────────────────────────────────────────────────────────────

def find_error_sentence_id(
    note: str,
    error_sentence: str,
    threshold: float = 0.6,
) -> Optional[int]:
    """Find which 1-indexed sentence in `note` matches `error_sentence`.

    Uses fuzzy matching (SequenceMatcher) to handle minor punctuation or
    whitespace differences between the stored error_sentence and the
    actual sentence in the note.

    Args:
        note: The clinical note text (raw or already numbered).
        error_sentence: The text of the error sentence to locate.
        threshold: Minimum similarity ratio to accept a match.

    Returns:
        1-indexed sentence ID, or None if no match above threshold.
    """
    if not error_sentence or not note:
        return None

    error_sentence = error_sentence.strip()
    sentences = split_sentences(note)

    best_id, best_ratio = None, 0.0
    for i, sent in enumerate(sentences):
        # Try exact substring first
        if error_sentence in sent or sent in error_sentence:
            return i + 1

        ratio = SequenceMatcher(None, sent.lower(), error_sentence.lower()).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_id = i + 1

    return best_id if best_ratio >= threshold else None


def diff_sentences(
    original_sentences: str,
    modified_sentences: str,
) -> Optional[int]:
    """Find which sentence changed between original and modified notes.

    Both inputs should be numbered-sentence strings.
    Returns the 1-indexed sentence ID that differs, or None.
    """
    orig = parse_numbered_sentences(original_sentences)
    modi = parse_numbered_sentences(modified_sentences)

    changed = []
    for sid in orig:
        if sid in modi and orig[sid] != modi[sid]:
            changed.append(sid)

    # Also check for sentences only in modified (shouldn't happen, but be safe)
    for sid in modi:
        if sid not in orig:
            changed.append(sid)

    return changed[0] if len(changed) == 1 else None
