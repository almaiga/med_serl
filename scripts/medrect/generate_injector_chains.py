#!/usr/bin/env python3
"""
Generate DeepSeek-R1 injector reasoning chains for SFT training.

Uses DeepSeek-R1-0528 to produce injector-perspective reasoning chains for both
error injection and benign change scenarios. Implements MedRECT's 3-step pipeline
(generate → clean meta-references → filter correct-only) inline.

Output is the medrect JSONL format compatible with train_medrect_lora.py:
    {sample_id, system_prompt, user_prompt, label, reasoning, language,
     error_type, error_flag, error_sentence_id, role, scenario,
     changed_sid, modified_text, reconstructed_note}

Usage:
    # Test with 3 error injection samples
    python scripts/medrect/generate_injector_chains.py \\
        --scenario injector_error --limit 3

    # Generate all benign chains
    python scripts/medrect/generate_injector_chains.py \\
        --scenario injector_benign

    # Generate both error + benign
    python scripts/medrect/generate_injector_chains.py \\
        --scenario all

    # Resume from checkpoint
    python scripts/medrect/generate_injector_chains.py \\
        --scenario injector_error --resume
"""

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = lambda: None  # noop if python-dotenv not installed

try:
    from openai import AsyncOpenAI, RateLimitError, APIError
except ImportError:
    AsyncOpenAI = None  # deferred check in DeepSeekR1Client.__init__
    RateLimitError = Exception
    APIError = Exception

# Add project root to path for shared utils
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.self_play.utils import (
    reconstruct_note,
    parse_numbered_sentences,
)

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Paths
DEFAULT_ERROR_DATA = (
    PROJECT_ROOT / "data_processed" / "medec_v2" / "ms_train.jsonl"
)
DEFAULT_BENIGN_DATA = (
    PROJECT_ROOT / "data_processed" / "benign_changes" / "benign_v2.jsonl"
)
DEFAULT_PROMPT_CONFIG = (
    PROJECT_ROOT / "configs" / "prompts" / "sft" / "injector_reasoning_prompts.json"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_processed" / "medrect"
DEFAULT_MEDEC_SPLIT_DIR = PROJECT_ROOT / "data_processed" / "medec_v2"


# =============================================================================
# DATA NORMALIZATION
# =============================================================================

@dataclass
class NormalizedRecord:
    """Unified representation for both error and benign records."""
    note_id: str
    scenario: str  # "error" or "benign"
    input_note: str  # correct_note / original_note
    output_note: str  # incorrect_note / modified_note
    input_sentences: str  # numbered version of input_note
    target_sentence_original: Optional[str]  # original sentence (benign only)
    target_sentence_modified: str  # error_sentence / modified_sentence
    changed_sid: Optional[int]  # sentence number of the change
    error_type: Optional[str]  # MEDEC error type (error only)
    change_type: Optional[str]  # pseudo_factual, etc (benign only)
    original_term: Optional[str]
    replacement_term: Optional[str]


def build_sentences_lookup(split_dir: Path) -> Dict[str, Dict]:
    """Build {note_id: {sentences, error_sentence_id, corrected_sentence}} from medec_v2 ERROR rows."""
    lookup: Dict[str, Dict] = {}
    for fname in ("ms_train.jsonl", "ms_val.jsonl", "uw_val.jsonl"):
        path = split_dir / fname
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                if r.get("error_flag") == 1:
                    lookup[r["note_id"]] = {
                        "sentences": r.get("sentences", ""),
                        "correct_sentences": r.get("correct_sentences", ""),
                        "error_sentence_id": r.get("error_sentence_id"),
                        "corrected_sentence": r.get("corrected_sentence", ""),
                    }
    logger.info("Sentences lookup: %d medec_v2 ERROR rows loaded", len(lookup))
    return lookup


def get_correct_sentences(medec_row: Dict) -> str:
    """Return correct note's numbered sentences.

    Prefers the pre-built correct_sentences field from build_medec_splits.py;
    falls back to substitution when not available (legacy data).
    """
    stored = medec_row.get("correct_sentences", "")
    if stored:
        return stored
    # Fallback: substitute error sentence in the incorrect-note sentences
    sentences = medec_row.get("sentences", "")
    sid = medec_row.get("error_sentence_id")
    corrected = medec_row.get("corrected_sentence", "")
    if not sid or not corrected:
        return sentences
    lines = sentences.split("\n")
    result = []
    for line in lines:
        m = re.match(r"^(\d+)\.\s+(.+)$", line.strip())
        if m and int(m.group(1)) == sid:
            result.append(f"{sid}. {corrected}")
        else:
            result.append(line)
    return "\n".join(result)


def find_sentence_in_numbered(
    numbered: str, target: str, original_term: str = ""
) -> Optional[int]:
    """Return 1-indexed sentence number for target text in a pre-numbered string.

    Tries in order:
    1. Exact match
    2. Normalized-whitespace match (handles CSV whitespace differences)
    3. Substring match (handles "Mr. <NAME/>" prefix differences)
    4. original_term search (handles misaligned multi-line original_sentence fields)
    """
    target = (target or "").strip()
    if not target:
        return None

    lines_parsed = []
    for line in numbered.split("\n"):
        m = re.match(r"^(\d+)\.\s+(.+)$", line.strip())
        if m:
            lines_parsed.append((int(m.group(1)), m.group(2).strip()))

    # 1. Exact match
    for sid, content in lines_parsed:
        if content == target:
            return sid

    # 2. Normalized whitespace match
    target_norm = re.sub(r"\s+", " ", target)
    for sid, content in lines_parsed:
        if re.sub(r"\s+", " ", content) == target_norm:
            return sid

    # 3. Substring match — handles prefix differences and partial sentences.
    # Use the first non-empty line of target in case it's multi-line.
    target_first = re.sub(r"\s+", " ", target.split("\n")[0].strip())
    if len(target_first) >= 10:  # avoid false positives on very short strings
        for sid, content in lines_parsed:
            content_norm = re.sub(r"\s+", " ", content)
            if target_first in content_norm or content_norm in target_norm:
                return sid

    # 4. original_term search — most reliable when original_sentence is misaligned
    term = (original_term or "").strip()
    if term and len(term) >= 4:
        for sid, content in lines_parsed:
            if term in content:
                return sid

    return None


def normalize_error_record(
    pair: Dict, sentences_lookup: Dict[str, Dict]
) -> Optional[NormalizedRecord]:
    """Normalize a medec_v2 ms_train/sft_split ERROR row for injector generation."""
    note_id = pair.get("note_id", "unknown")
    # Support both medec_v2 field names and legacy sft_train.jsonl names
    corrected_text = pair.get("corrected_text") or pair.get("correct_note", "")
    incorrect_note = pair.get("text") or pair.get("incorrect_note", "")
    error_sentence = pair.get("error_sentence", "")

    if not corrected_text or not error_sentence:
        return None

    medec_row = sentences_lookup.get(note_id)
    if medec_row is None:
        logger.warning("No sentences lookup entry for error record %s", note_id)
        return None

    input_sentences = get_correct_sentences(medec_row)
    changed_sid = medec_row.get("error_sentence_id")
    if changed_sid is None:
        return None

    return NormalizedRecord(
        note_id=note_id,
        scenario="error",
        input_note=corrected_text,
        output_note=incorrect_note,
        input_sentences=input_sentences,
        target_sentence_original=None,
        target_sentence_modified=error_sentence,
        changed_sid=changed_sid,
        error_type=pair.get("error_type"),
        change_type=None,
        original_term=None,
        replacement_term=None,
    )


def normalize_benign_record(
    record: Dict, sentences_lookup: Dict[str, Dict]
) -> Optional[NormalizedRecord]:
    """Normalize a benign_v2.jsonl record (changed_sid and sentences pre-computed)."""
    note_id = record.get("note_id", "unknown")
    original_note = record.get("original_note", "")
    modified_note = record.get("modified_note", "")
    original_sentence = record.get("original_sentence", "")
    modified_sentence = record.get("modified_sentence", "")
    sentences = record.get("sentences", "")
    changed_sid = record.get("changed_sid")

    if not original_note or not modified_note or not modified_sentence:
        return None
    if not sentences or changed_sid is None:
        logger.warning("Missing sentences/changed_sid for benign %s — re-run build_benign_splits.py", note_id)
        return None

    return NormalizedRecord(
        note_id=note_id,
        scenario="benign",
        input_note=original_note,
        output_note=modified_note,
        input_sentences=sentences,
        target_sentence_original=original_sentence,
        target_sentence_modified=modified_sentence,
        changed_sid=changed_sid,
        error_type=None,
        change_type=record.get("change_type", "pseudo_factual"),
        original_term=record.get("original_term"),
        replacement_term=record.get("replacement_term"),
    )


# =============================================================================
# META-REFERENCE CLEANING (MedRECT ReasoningCleaner port)
# =============================================================================

class ReasoningCleaner:
    """Clean reasoning text by removing meta-reference sentences.

    Ported from MedRECT clean_reasoning_en.py.
    """

    def __init__(self, config: Dict):
        cleaning = config.get("meta_reference_cleaning", {})
        self.meta_keywords = cleaning.get("keywords", [])
        regex_strs = cleaning.get("regex_patterns", [])

        # Compile keyword patterns
        self.meta_patterns = [
            re.compile(rf".*{re.escape(kw)}.*", re.IGNORECASE)
            for kw in self.meta_keywords
        ]

        # Compile specific regex patterns
        self.specific_patterns = [
            re.compile(p, re.IGNORECASE) for p in regex_strs
        ]

        self.all_patterns = self.meta_patterns + self.specific_patterns
        logger.info(
            f"ReasoningCleaner loaded: {len(self.meta_keywords)} keywords, "
            f"{len(regex_strs)} regex patterns"
        )

    def find_sentences_to_remove(self, text: str) -> List[Tuple[int, int]]:
        """Find sentences containing meta-references.

        Splits on both period boundaries AND newlines so we catch
        meta-ref lines that don't end with a period.
        """
        if not text:
            return []

        sentences_to_remove = []
        # Split into segments by newline first, then by period within each
        # This captures both "sentence." units and standalone lines
        segment_pattern = re.compile(r"[^\n]+")

        for line_match in segment_pattern.finditer(text):
            line = line_match.group()
            line_start = line_match.start()

            # Try to split the line further by periods
            sentence_pattern = re.compile(r"[^.]*\.")
            sub_matches = list(sentence_pattern.finditer(line))

            if sub_matches:
                # Check each period-delimited sentence
                for match in sub_matches:
                    sentence = match.group().strip()
                    if not sentence:
                        continue
                    for pattern in self.all_patterns:
                        if pattern.search(sentence):
                            sentences_to_remove.append(
                                (line_start + match.start(), line_start + match.end())
                            )
                            logger.debug(f"Pattern match: {pattern.pattern[:50]}...")
                            break

                # Also check the trailing part after the last period
                last_end = sub_matches[-1].end()
                if last_end < len(line):
                    trailing = line[last_end:].strip()
                    if trailing:
                        for pattern in self.all_patterns:
                            if pattern.search(trailing):
                                sentences_to_remove.append(
                                    (line_start + last_end, line_start + len(line))
                                )
                                logger.debug(f"Pattern match: {pattern.pattern[:50]}...")
                                break
            else:
                # No period — check the whole line
                stripped = line.strip()
                if stripped:
                    for pattern in self.all_patterns:
                        if pattern.search(stripped):
                            sentences_to_remove.append(
                                (line_start, line_start + len(line))
                            )
                            logger.debug(f"Pattern match: {pattern.pattern[:50]}...")
                            break

        return sentences_to_remove

    def clean(self, text: str) -> Tuple[str, int]:
        """Clean reasoning text. Returns (cleaned_text, removed_count)."""
        if not text:
            return text, 0

        to_remove = self.find_sentences_to_remove(text)
        if not to_remove:
            return text, 0

        # Remove from end to start to preserve positions
        to_remove.sort(key=lambda x: x[0], reverse=True)
        cleaned = text
        for start, end in to_remove:
            cleaned = cleaned[:start] + cleaned[end:]

        return cleaned, len(to_remove)


# =============================================================================
# DEEPSEEK-R1 CLIENT
# =============================================================================

class DeepSeekR1Client:
    """Async client for an OpenAI-compatible reasoning endpoint.

    Returns reasoning_content (chain-of-thought) separately from content
    (final answer). Model, base_url and API key env var are configurable so
    we can target the exact reasoning model (e.g. deepseek-r1-0528 via a
    third-party host) rather than whatever DeepSeek's first-party alias
    currently points to.

    Defaults are read from environment:
        REASONING_MODEL     (default "deepseek-reasoner")
        REASONING_BASE_URL  (default "https://api.deepseek.com")
        REASONING_API_KEY_ENV (default "DEEPSEEK_API_KEY")
    and can be overridden via constructor args.
    """

    def __init__(
        self,
        max_concurrent: int = 50,
        max_retries: int = 3,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key_env: Optional[str] = None,
        reasoning_effort: Optional[str] = None,
    ):
        self.reasoning_effort = reasoning_effort
        self.model = model or os.getenv("REASONING_MODEL", "deepseek-reasoner")
        base_url = base_url or os.getenv("REASONING_BASE_URL", "https://api.deepseek.com")
        api_key_env = api_key_env or os.getenv("REASONING_API_KEY_ENV", "DEEPSEEK_API_KEY")

        api_key = os.getenv(api_key_env)
        if not api_key:
            raise ValueError(
                f"{api_key_env} not found. Set it in .env or environment."
            )
        if AsyncOpenAI is None:
            raise ImportError(
                "openai package required: pip install openai"
            )

        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.max_concurrent = max_concurrent
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrent)
        logger.info("Reasoning model=%s  base_url=%s  key_env=%s",
                    self.model, base_url, api_key_env)

        # Stats
        self.total_calls = 0
        self.successful_calls = 0
        self.failed_calls = 0

        logger.info(f"DeepSeekR1Client initialized (concurrency={max_concurrent})")

    async def generate(
        self, system_prompt: str, user_prompt: str, sample_id: str
    ) -> Optional[Dict[str, Any]]:
        """Call DeepSeek-R1 and return {reasoning, content, usage}.

        Returns None on failure after retries.
        """
        async with self.semaphore:
            self.total_calls += 1

            for attempt in range(self.max_retries):
                try:
                    create_kwargs = {
                        "model": self.model,
                        "messages": [
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                        ],
                    }
                    if self.reasoning_effort:
                        create_kwargs["reasoning_effort"] = self.reasoning_effort
                    response = await self.client.chat.completions.create(
                        **create_kwargs
                    )

                    choice = response.choices[0]
                    msg = choice.message

                    # Extract reasoning_content (R1-specific field)
                    reasoning = ""
                    if hasattr(msg, "reasoning_content") and msg.reasoning_content:
                        reasoning = msg.reasoning_content
                    elif hasattr(msg, "reasoning") and msg.reasoning:
                        reasoning = msg.reasoning

                    content = msg.content or ""

                    usage = {}
                    if response.usage:
                        usage = {
                            "prompt_tokens": response.usage.prompt_tokens,
                            "completion_tokens": response.usage.completion_tokens,
                            "total_tokens": response.usage.total_tokens,
                        }

                    self.successful_calls += 1
                    return {
                        "reasoning": reasoning,
                        "content": content,
                        "usage": usage,
                    }

                except RateLimitError:
                    wait = (2 ** attempt) * 2
                    logger.warning(f"Rate limit for {sample_id}, retry in {wait}s")
                    await asyncio.sleep(wait)

                except APIError as e:
                    wait = 2 ** attempt
                    logger.warning(
                        f"API error for {sample_id}: {e}, retry {attempt + 1}"
                    )
                    await asyncio.sleep(wait)

                except Exception as e:
                    logger.error(f"Unexpected error for {sample_id}: {e}")
                    if attempt == self.max_retries - 1:
                        self.failed_calls += 1
                        return None
                    await asyncio.sleep(2 ** attempt)

            self.failed_calls += 1
            return None

    def stats(self) -> Dict:
        return {
            "total": self.total_calls,
            "success": self.successful_calls,
            "failed": self.failed_calls,
        }


# =============================================================================
# PROMPT BUILDING
# =============================================================================

def build_error_prompt(
    record: NormalizedRecord, config: Dict
) -> Tuple[str, str]:
    """Build system + user prompt for error injection scenario."""
    cfg = config["injector_error"]
    system_prompt = cfg["system_prompt"]

    cheat_info = cfg["cheat_info_template"].format(
        error_sentence_id=record.changed_sid or "unknown",
    )
    error_hint = cfg["error_hint_template"].format(
        error_sentence=record.target_sentence_modified,
        error_type=record.error_type or "clinical error",
    )

    user_prompt = cfg["user_template"].format(
        cheat_info=cheat_info,
        task_description=cfg["task_description"],
        step4_instruction=cfg["step4_instruction"],
        error_hint=error_hint,
        sentences=record.input_sentences,
    )

    return system_prompt, user_prompt


def build_benign_prompt(
    record: NormalizedRecord, config: Dict
) -> Tuple[str, str]:
    """Build system + user prompt for benign change scenario."""
    cfg = config["injector_benign"]
    system_prompt = cfg["system_prompt"]

    cheat_info = cfg["cheat_info_template"].format(
        change_type=record.change_type or "pseudo_factual",
    )
    change_hint = cfg["change_hint_template"].format(
        change_type=record.change_type or "pseudo_factual",
        original_term=record.original_term or "",
        replacement_term=record.replacement_term or "",
    )

    user_prompt = cfg["user_template"].format(
        cheat_info=cheat_info,
        task_description=cfg["task_description"],
        step4_instruction=cfg["step4_instruction"],
        change_hint=change_hint,
        sentences=record.input_sentences,
    )

    return system_prompt, user_prompt


# =============================================================================
# VALIDATION (correct-only extraction)
# =============================================================================

def validate_error_chain(
    record: NormalizedRecord,
    r1_content: str,
) -> Tuple[bool, Optional[int], Optional[str], str]:
    """Validate that R1 targeted the correct sentence for error injection.

    Returns (is_correct, parsed_sid, parsed_text, reason).
    """
    parsed_sid, parsed_text = _parse_compact_output(r1_content)

    if parsed_sid is None:
        return False, None, None, "Could not parse N. <sentence> from output"

    if record.changed_sid is not None and parsed_sid != record.changed_sid:
        return (
            False,
            parsed_sid,
            parsed_text,
            f"Wrong sentence: expected {record.changed_sid}, got {parsed_sid}",
        )

    return True, parsed_sid, parsed_text, "ok"


def validate_benign_chain(
    record: NormalizedRecord,
    r1_content: str,
) -> Tuple[bool, Optional[int], Optional[str], str]:
    """Validate that R1 modified the correct sentence for benign change.

    Returns (is_correct, parsed_sid, parsed_text, reason).
    """
    parsed_sid, parsed_text = _parse_compact_output(r1_content)

    if parsed_sid is None:
        return False, None, None, "Could not parse N. <sentence> from output"

    if record.changed_sid is not None and parsed_sid != record.changed_sid:
        return (
            False,
            parsed_sid,
            parsed_text,
            f"Wrong sentence: expected {record.changed_sid}, got {parsed_sid}",
        )

    return True, parsed_sid, parsed_text, "ok"


def _parse_compact_output(text: str) -> Tuple[Optional[int], Optional[str]]:
    """Parse 'N. <modified sentence>' from R1's final content."""
    if not text:
        return None, None

    for line in text.strip().split("\n"):
        line = line.strip().strip("`")
        m = re.match(r"^(\d+)\.\s+(.+)$", line)
        if m:
            return int(m.group(1)), m.group(2).strip()

    return None, None


# =============================================================================
# OUTPUT FORMATTING (medrect JSONL)
# =============================================================================

def format_medrect_record(
    record: NormalizedRecord,
    system_prompt: str,
    user_prompt: str,
    reasoning: str,
    parsed_sid: int,
    parsed_text: str,
    reconstructed: str,
) -> Dict[str, Any]:
    """Format a validated chain into medrect JSONL format.

    Compatible with train_medrect_lora.py format_for_sft().
    """
    sample_id = f"{record.note_id}_injector_{record.scenario}"

    # Label is the compact injector output: "N. <modified sentence>"
    label = f"{parsed_sid}. {parsed_text}"

    return {
        "sample_id": sample_id,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "label": label,
        "reasoning": reasoning,
        "language": "en",
        "error_type": record.error_type,
        "error_flag": 1 if record.scenario == "error" else 0,
        "error_sentence_id": parsed_sid,
        # Extra fields for traceability
        "role": "injector",
        "scenario": record.scenario,
        "change_type": record.change_type,
        "changed_sid": parsed_sid,
        "modified_text": parsed_text,
        "reconstructed_note": reconstructed,
    }


# =============================================================================
# CHECKPOINT
# =============================================================================

class Checkpoint:
    """Track processed sample_ids for resume support."""

    def __init__(self, path: Optional[Path] = None):
        self.path = path
        self.processed: set = set()
        if path and path.exists() and path.stat().st_size > 0:
            with open(path, "r") as f:
                data = json.load(f)
                self.processed = set(data.get("processed", []))
            logger.info(f"Resumed checkpoint: {len(self.processed)} already done")

    def is_done(self, sample_id: str) -> bool:
        return sample_id in self.processed

    def mark_done(self, sample_id: str):
        self.processed.add(sample_id)

    def save(self):
        if not self.path:
            return
        with open(self.path, "w") as f:
            json.dump({"processed": sorted(self.processed)}, f)


# =============================================================================
# MAIN GENERATION PIPELINE
# =============================================================================

async def process_record(
    record: NormalizedRecord,
    client: DeepSeekR1Client,
    cleaner: ReasoningCleaner,
    config: Dict,
) -> Optional[Dict[str, Any]]:
    """Generate, clean, validate, and format a single record."""

    sample_id = f"{record.note_id}_injector_{record.scenario}"

    # 1. Build prompt
    if record.scenario == "error":
        system_prompt, user_prompt = build_error_prompt(record, config)
    else:
        system_prompt, user_prompt = build_benign_prompt(record, config)

    # 2. Generate with DeepSeek-R1
    result = await client.generate(system_prompt, user_prompt, sample_id)
    if result is None:
        logger.warning(f"SKIP {sample_id}: API failure")
        return None

    reasoning_raw = result["reasoning"]
    content = result["content"]

    # 3. Clean meta-references from reasoning
    reasoning_cleaned, removed_count = cleaner.clean(reasoning_raw)
    if removed_count > 0:
        logger.debug(f"{sample_id}: cleaned {removed_count} meta-ref sentences")

    # 4. Validate (correct-only extraction)
    if record.scenario == "error":
        is_correct, parsed_sid, parsed_text, reason = validate_error_chain(
            record, content
        )
    else:
        is_correct, parsed_sid, parsed_text, reason = validate_benign_chain(
            record, content
        )

    if not is_correct:
        logger.info(f"REJECT {sample_id}: {reason}")
        return None

    # 5. Reconstruct the modified note
    reconstructed = reconstruct_note(
        record.input_sentences, parsed_sid, parsed_text
    )

    # 6. Format as medrect JSONL record
    # Use prompts from error_injection_prompts_v4.json (what the 8B model
    # actually sees at inference/self-play), NOT the R1 cheat prompts
    clean_user_prompt = _build_clean_user_prompt(record, config)
    clean_system_prompt = _get_inference_system_prompt(record)

    output = format_medrect_record(
        record=record,
        system_prompt=clean_system_prompt,
        user_prompt=clean_user_prompt,
        reasoning=reasoning_cleaned,
        parsed_sid=parsed_sid,
        parsed_text=parsed_text,
        reconstructed=reconstructed,
    )
    output["_meta"] = {
        "meta_refs_removed": removed_count,
        "reasoning_cleaned": removed_count > 0,
        "original_reasoning_length": len(reasoning_raw),
        "cleaned_reasoning_length": len(reasoning_cleaned),
        "reasoning_tokens": result["usage"].get("completion_tokens", 0),
        "prompt_tokens": result["usage"].get("prompt_tokens", 0),
        "total_tokens": result["usage"].get("total_tokens", 0),
        "raw_content": content,
    }

    return output


def _get_inference_system_prompt(record: NormalizedRecord) -> str:
    """Get the system prompt the 8B model sees at inference/self-play.

    Must match error_injection_prompts_v4.json exactly so training and
    inference system prompts are identical.
    """
    v4_path = PROJECT_ROOT / "configs" / "prompts" / "error_injection_prompts_v4.json"
    with open(v4_path) as f:
        v4 = json.load(f)

    if record.scenario == "error":
        return v4["system_prompt_incorrect"]
    else:
        return v4["system_prompt_correct"]


def _build_clean_user_prompt(record: NormalizedRecord, config: Dict) -> str:
    """Build the training user prompt WITHOUT cheat info.

    IMPORTANT: Must exactly match the prompts in error_injection_prompts_v4.json
    that the 8B model sees at inference/self-play. Any mismatch between training
    and inference prompts degrades 8B model performance.
    """
    # Load the self-play prompts to ensure exact match
    v4_path = PROJECT_ROOT / "configs" / "prompts" / "error_injection_prompts_v4.json"
    with open(v4_path) as f:
        v4 = json.load(f)

    if record.scenario == "error":
        # Match injector_incorrect_template exactly
        # Use generic "clinical" as prompt_intent (not the specific error_type)
        template = v4["injector_incorrect_template"]
        return template.format(
            prompt_intent="clinical",
            sentences=record.input_sentences,
        )
    else:
        # Match injector_correct_template exactly
        change_type = record.change_type or "pseudo_factual"
        change_types = v4.get("benign_change_types", {})
        change_desc = change_types.get(
            change_type,
            f"Make a {change_type} change that preserves clinical meaning.",
        )
        template = v4["injector_correct_template"]
        return template.format(
            change_type=change_type,
            change_type_description=change_desc,
            sentences=record.input_sentences,
        )


async def run_pipeline(args: argparse.Namespace) -> None:
    """Run the full generation pipeline."""

    # Load config
    config_path = Path(args.prompt_config)
    with open(config_path) as f:
        config = json.load(f)
    logger.info(f"Loaded prompt config: {config_path}")

    # Build sentence lookup from medec_v2 split files
    sentences_lookup = build_sentences_lookup(Path(args.medec_split_dir))

    # Init components
    client = DeepSeekR1Client(
        max_concurrent=args.concurrency,
        max_retries=args.max_retries,
        model=args.model,
        base_url=args.base_url,
        api_key_env=args.api_key_env,
        reasoning_effort=args.reasoning_effort,
    )
    cleaner = ReasoningCleaner(config)

    # Determine output paths
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    scenarios = []
    if args.scenario in ("injector_error", "all"):
        scenarios.append("error")
    if args.scenario in ("injector_benign", "all"):
        scenarios.append("benign")

    for scenario in scenarios:
        logger.info(f"\n{'='*60}")
        logger.info(f"Scenario: injector_{scenario}")
        logger.info(f"{'='*60}")

        # Load and normalize data
        if scenario == "error":
            data_path = Path(args.error_data)
            records = _load_and_normalize_error(
                data_path, args.limit, sentences_lookup
            )
            out_file = output_dir / f"injector_error_chains_{timestamp}.jsonl"
        else:
            data_path = Path(args.benign_data)
            records = _load_and_normalize_benign(
                data_path, args.limit, sentences_lookup
            )
            out_file = output_dir / f"injector_benign_chains_{timestamp}.jsonl"

        logger.info(f"Loaded {len(records)} records from {data_path}")

        # Checkpoint
        ckpt_path = output_dir / f"checkpoint_injector_{scenario}.json"
        ckpt = Checkpoint(ckpt_path) if args.resume else Checkpoint()

        # Process records
        stats = {"total": 0, "generated": 0, "rejected": 0, "skipped": 0, "api_fail": 0}
        # Collect records to process (skip checkpointed ones)
        to_process = []
        for record in records:
            sample_id = f"{record.note_id}_injector_{record.scenario}"
            if ckpt.is_done(sample_id):
                stats["skipped"] += 1
                continue
            to_process.append(record)

        stats["total"] = len(to_process)
        logger.info(
            f"Processing {len(to_process)} records "
            f"(skipped {stats['skipped']} from checkpoint), "
            f"concurrency={args.concurrency}"
        )

        # Lock for thread-safe file writes and stats updates
        write_lock = asyncio.Lock()
        progress_count = 0

        async def _process_and_write(idx: int, record: NormalizedRecord, fout):
            nonlocal progress_count
            result = await process_record(record, client, cleaner, config)

            async with write_lock:
                progress_count += 1
                if result is not None:
                    fout.write(json.dumps(result, ensure_ascii=False) + "\n")
                    fout.flush()
                    stats["generated"] += 1
                    ckpt.mark_done(f"{record.note_id}_injector_{record.scenario}")
                else:
                    stats["rejected"] += 1

                # Progress logging every 10 completions or at the end
                if progress_count % 10 == 0 or progress_count == len(to_process):
                    logger.info(
                        f"  [{progress_count}/{len(to_process)}] "
                        f"gen={stats['generated']} rej={stats['rejected']} | "
                        f"API: {client.stats()}"
                    )

                # Checkpoint save periodically
                if args.resume and progress_count % args.checkpoint_every == 0:
                    ckpt.save()

        with open(out_file, "a" if args.resume else "w") as fout:
            # Launch all tasks — semaphore in DeepSeekR1Client bounds concurrency
            tasks = [
                _process_and_write(i, rec, fout)
                for i, rec in enumerate(to_process)
            ]
            await asyncio.gather(*tasks)

        # Final checkpoint
        if args.resume:
            ckpt.save()

        # Summary (MedRECT-style statistics with error_flag breakdown)
        logger.info(f"\n{'='*60}")
        logger.info(f"RESULTS: injector_{scenario}")
        logger.info(f"  Total processed: {stats['total']}")
        logger.info(f"  Generated (valid): {stats['generated']}")
        logger.info(f"  Rejected (wrong answer): {stats['rejected']}")
        logger.info(f"  Skipped (checkpoint): {stats['skipped']}")
        pass_rate = (
            stats["generated"] / max(stats["generated"] + stats["rejected"], 1) * 100
        )
        logger.info(f"  Pass rate: {pass_rate:.1f}%")
        logger.info(f"  Output: {out_file}")
        logger.info(f"  API stats: {client.stats()}")
        logger.info(f"{'='*60}\n")

        # Save detailed stats (aligned with MedRECT extract_correct_samples format)
        stats_path = output_dir / f"stats_injector_{scenario}_{timestamp}.json"
        with open(stats_path, "w") as f:
            json.dump(
                {
                    "scenario": f"injector_{scenario}",
                    "total_samples": len(records),
                    "processed": stats["total"],
                    "correct_samples_count": stats["generated"],
                    "rejected_samples_count": stats["rejected"],
                    "skipped_checkpoint": stats["skipped"],
                    "accuracy": pass_rate / 100.0,
                    "pass_rate_pct": pass_rate,
                    "error_flag": 1 if scenario == "error" else 0,
                    "api_stats": client.stats(),
                    "output_file": str(out_file),
                    "timestamp": timestamp,
                },
                f, indent=2,
            )


def _load_and_normalize_error(
    path: Path, limit: Optional[int], sentences_lookup: Dict[str, Dict]
) -> List[NormalizedRecord]:
    """Load medec_v2 ms_train/sft_split JSONL and normalize ERROR rows."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            pair = json.loads(line)
            # Skip CORRECT rows — injector only processes error pairs
            if pair.get("error_flag", 1) == 0:
                continue
            rec = normalize_error_record(pair, sentences_lookup)
            if rec and rec.changed_sid is not None:
                records.append(rec)
            if limit and len(records) >= limit:
                break
    return records


def _load_and_normalize_benign(
    path: Path, limit: Optional[int], sentences_lookup: Dict[str, Dict]
) -> List[NormalizedRecord]:
    """Load benign_v2.jsonl (pre-computed changed_sid) and normalize."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            rec = normalize_benign_record(raw, sentences_lookup)
            if rec and rec.changed_sid is not None:
                records.append(rec)
            if limit and len(records) >= limit:
                break
    return records


# =============================================================================
# CLI
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate DeepSeek-R1 injector reasoning chains (medrect format)."
    )
    parser.add_argument(
        "--scenario",
        choices=["injector_error", "injector_benign", "all"],
        default="all",
        help="Which scenario(s) to generate (default: all)",
    )
    parser.add_argument(
        "--medec-split-dir",
        default=str(DEFAULT_MEDEC_SPLIT_DIR),
        help="Path to data_processed/medec_v2/ directory (for sentence lookups)",
    )
    parser.add_argument(
        "--error-data",
        default=str(DEFAULT_ERROR_DATA),
        help="Path to medec_v2/ms_train.jsonl or sft_split.jsonl (MEDEC error pairs)",
    )
    parser.add_argument(
        "--benign-data",
        default=str(DEFAULT_BENIGN_DATA),
        help="Path to benign_v2.jsonl (pre-computed changed_sid + sentences)",
    )
    parser.add_argument(
        "--prompt-config",
        default=str(DEFAULT_PROMPT_CONFIG),
        help="Path to injector_reasoning_prompts.json",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Output directory for generated chains",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of records to process (for testing)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=50,
        help="Max concurrent API calls (default: 50). DeepSeek has no rate limit.",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Max retries per API call (default: 3)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (skip already-processed samples)",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=25,
        help="Save checkpoint every N records (default: 25)",
    )
    parser.add_argument(
        "--model", default=None,
        help="Reasoning model id (default: env REASONING_MODEL or deepseek-reasoner)",
    )
    parser.add_argument(
        "--base-url", default=None,
        help="OpenAI-compatible base URL (default: env REASONING_BASE_URL or DeepSeek)",
    )
    parser.add_argument(
        "--api-key-env", default=None,
        help="Env var name holding the API key (default: env REASONING_API_KEY_ENV or DEEPSEEK_API_KEY)",
    )
    parser.add_argument(
        "--reasoning-effort", default=None,
        help="DeepSeek thinking-mode effort: high | max (optional)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    asyncio.run(run_pipeline(args))


if __name__ == "__main__":
    main()
