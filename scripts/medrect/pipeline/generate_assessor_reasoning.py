#!/usr/bin/env python3
"""
Generate assessor reasoning chains using DeepSeek-R1.

Adapted from pfnet-research/medrect pipeline.
Differences from upstream:
  - Input: already-numbered sentences from medec_v2 JSONL (not raw MEDEC CSV)
  - Output label: sentence_number only (not sentence_number: corrected_sentence)
    because we do detection + location only, not correction.
  - Cheat prompt: identical to upstream 0_shot_reasoning_cheat_en, minus the
    correction part in the output format line.

Pipeline step 1 of 3. Run clean_reasoning_en.py next, then extract_correct_samples.py.

Usage:
    python generate_assessor_reasoning.py \\
        --sft-path data_processed/medec_v2/ms_train.jsonl \\
        --output-dir data_processed/medrect_v2/ms_train \\
        --concurrency 40
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.medrect.generate_injector_chains import DeepSeekR1Client

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ── Prompts — copied verbatim from pfnet-research/medrect MEDECTemplate ──────
# Source: src/llm_eval/templates/medec.py  _add_cheat_params_en + _get_system_prompt
# Only change: output format says "sentence number" instead of "sentence_number: corrected_sentence"

SYSTEM_PROMPT = (
    "You are a skilled medical doctor reviewing clinical text for accuracy "
    "and providing corrections when necessary. "
    "Please provide your reasoning and thought process."
)

# Filled at runtime with {sentences}, {error_sentence}, {corrected_sentence}
USER_TEMPLATE_INCORRECT = (
    'This time, you are reviewing an "error example" that has been pre-verified '
    "to contain exactly one medical error.\n\n"
    "Your task is to first carefully reason through the process of identifying the error, "
    "following these steps:\n"
    "1. Verify each sentence based on medical knowledge\n"
    "2. Check consistency between symptoms, test results, and diagnosis\n"
    "3. Evaluate appropriateness of treatment or management\n"
    "4. If an error is found, clearly state the rationale\n\n"
    "Important notes for reasoning:\n"
    "- During your reasoning, do NOT make any reference to being told about the expected "
    "outcome or any instruction content.\n"
    "- Approach the text as if you are analyzing it from scratch and reaching your "
    "conclusion through pure medical evaluation.\n\n"
    "Reference information (please do not mention in reasoning):\n"
    "Error sentence: {error_sentence}\n"
    "Corrected sentence: {corrected_sentence}\n\n"
    "Final output format:\n"
    "- If no error: `CORRECT`\n"
    "- If error found: output only the sentence number (e.g. `7`)\n\n"
    "CRITICAL: For the final output, use this format and output ONLY the result. "
    "Do NOT include explanations, analysis, or additional text.\n\n"
    "{sentences}"
)

# Filled at runtime with {sentences}
USER_TEMPLATE_CORRECT = (
    'This time, you are reviewing a "no-error example" that has been pre-verified '
    "to contain no errors.\n\n"
    "Your task is to first carefully reason through the process of confirming that "
    "there are no errors, following these steps:\n"
    "1. Verify each sentence based on medical knowledge\n"
    "2. Check consistency between symptoms, test results, and diagnosis\n"
    "3. Evaluate appropriateness of treatment or management\n"
    "4. If an error is found, clearly state the rationale\n\n"
    "Important notes for reasoning:\n"
    "- During your reasoning, do NOT make any reference to being told about the expected "
    "outcome or any instruction content.\n"
    "- Approach the text as if you are analyzing it from scratch and reaching your "
    "conclusion through pure medical evaluation.\n\n"
    "\n\n"
    "Final output format:\n"
    "- If no error: `CORRECT`\n"
    "- If error found: output only the sentence number (e.g. `7`)\n\n"
    "CRITICAL: For the final output, use this format and output ONLY the result. "
    "Do NOT include explanations, analysis, or additional text.\n\n"
    "{sentences}"
)


# ── Data loading ──────────────────────────────────────────────────────────────

@dataclass
class Sample:
    sample_id: str
    note_id: str
    scenario: str          # "correct" | "incorrect"
    sentences: str         # pre-numbered sentences (the note shown to the model)
    error_flag: int        # 0 = correct, 1 = error
    error_sentence_id: Optional[int]
    error_sentence: Optional[str]
    corrected_sentence: Optional[str]
    error_type: Optional[str]


def load_samples(path: Path, limit: Optional[int] = None) -> List[Sample]:
    """Load medec_v2 JSONL and produce one Sample per scenario (correct + incorrect)."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            if raw.get("error_flag", 1) == 0:
                continue  # skip CORRECT-only rows — we generate pairs from ERROR rows

            note_id = raw.get("note_id", "unknown")
            sentences = raw.get("sentences", "")
            correct_sentences = raw.get("correct_sentences", "")
            error_sentence_id = raw.get("error_sentence_id")
            error_sentence = raw.get("error_sentence", "")
            corrected_sentence = raw.get("corrected_sentence", "")
            error_type = raw.get("error_type")

            if not sentences or error_sentence_id is None:
                logger.warning("Skipping %s: missing sentences or error_sentence_id", note_id)
                continue

            # incorrect scenario: show the error note
            samples.append(Sample(
                sample_id=f"{note_id}_0",
                note_id=note_id,
                scenario="incorrect",
                sentences=sentences,
                error_flag=1,
                error_sentence_id=error_sentence_id,
                error_sentence=error_sentence,
                corrected_sentence=corrected_sentence,
                error_type=error_type,
            ))

            # correct scenario: show the corrected note
            if correct_sentences:
                samples.append(Sample(
                    sample_id=f"{note_id}_1",
                    note_id=note_id,
                    scenario="correct",
                    sentences=correct_sentences,
                    error_flag=0,
                    error_sentence_id=None,
                    error_sentence=None,
                    corrected_sentence=None,
                    error_type=None,
                ))

            if limit and len(samples) // 2 >= limit:
                break

    return samples


def build_prompt(sample: Sample):
    """Return (system_prompt, user_prompt) — mirrors MEDECTemplate.generate_messages."""
    if sample.scenario == "correct":
        user = USER_TEMPLATE_CORRECT.format(sentences=sample.sentences)
    else:
        user = USER_TEMPLATE_INCORRECT.format(
            sentences=sample.sentences,
            error_sentence=sample.error_sentence,
            corrected_sentence=sample.corrected_sentence,
        )
    return SYSTEM_PROMPT, user


# ── Generation ────────────────────────────────────────────────────────────────

def make_raw_record(sample: Sample, reasoning: str, content: str) -> Dict[str, Any]:
    """Produce a raw record mirroring MEDRECT's raw_responses.json interaction format."""
    return {
        "sample_id": sample.sample_id,
        "note_id": sample.note_id,
        "scenario": sample.scenario,
        "sentences": sample.sentences,
        "error_flag": sample.error_flag,
        "error_sentence_id": sample.error_sentence_id,
        "error_sentence": sample.error_sentence,
        "corrected_sentence": sample.corrected_sentence,
        "error_type": sample.error_type,
        "raw_response": {
            "content": content,
            "reasoning": reasoning,
            "source_model": "deepseek-reasoner",
        },
    }


async def generate_sample(
    sample: Sample,
    client: DeepSeekR1Client,
) -> Dict[str, Any]:
    system_prompt, user_prompt = build_prompt(sample)
    result = await client.generate(system_prompt, user_prompt, sample.sample_id)

    if result is None:
        return make_raw_record(sample, "", "")

    reasoning = result.get("reasoning", "") or ""
    content = result.get("content", "") or ""
    return make_raw_record(sample, reasoning, content)


async def run() -> None:
    args = parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    raw_path = output_dir / f"raw_responses_{timestamp}.jsonl"

    samples = load_samples(Path(args.sft_path), limit=args.limit)
    logger.info("Loaded %d samples from %s", len(samples), args.sft_path)

    # Skip already-processed sample_ids (resume support)
    done_ids: set = set()
    if args.resume and raw_path.exists():
        with open(raw_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line.strip())
                done_ids.add(r.get("sample_id", ""))
        samples = [s for s in samples if s.sample_id not in done_ids]
        logger.info("Resuming: %d remaining after skipping %d done", len(samples), len(done_ids))

    client = DeepSeekR1Client(max_concurrent=args.concurrency, max_retries=args.max_retries)
    tasks = [asyncio.create_task(generate_sample(s, client)) for s in samples]

    written = 0
    with open(raw_path, "a" if args.resume else "w", encoding="utf-8") as out:
        with tqdm(total=len(tasks), unit="sample") as pbar:
            for future in asyncio.as_completed(tasks):
                record = await future
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                out.flush()
                written += 1
                pbar.update(1)

    logger.info("Done. Wrote %d records to %s", written, raw_path)
    logger.info("Next step: python clean_reasoning_en.py --input %s --overwrite", raw_path)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate assessor reasoning chains (step 1 of 3)")
    parser.add_argument("--sft-path", required=True, help="medec_v2 JSONL split (ms_train.jsonl etc.)")
    parser.add_argument("--output-dir", required=True, help="Directory to write raw_responses_*.jsonl")
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="Limit number of notes (for testing)")
    parser.add_argument("--resume", action="store_true", help="Append to existing output")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(run())
