#!/usr/bin/env python3
"""
Test suite for the adapted MEDRECT assessor pipeline.

Checks that our pipeline matches pfnet-research/medrect exactly, with the
only allowed differences being:
  1. Label format: sentence_number only (not sentence_number: corrected_sentence)
  2. Input: already-numbered sentences from medec_v2 JSONL (not raw CSV)

Run:
    python scripts/medrect/pipeline/test_pipeline.py
"""

import json
import re
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.medrect.pipeline.generate_assessor_reasoning import (
    SYSTEM_PROMPT,
    USER_TEMPLATE_CORRECT,
    USER_TEMPLATE_INCORRECT,
    Sample,
    build_prompt,
    load_samples,
)
from scripts.medrect.pipeline.clean_reasoning_en import ReasoningCleaner
from scripts.medrect.pipeline.extract_correct_samples import (
    extract_correct_samples,
    format_for_sft,
    parse_prediction,
)

PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
_failures = []


def check(name: str, condition: bool, detail: str = "") -> None:
    if condition:
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name}" + (f": {detail}" if detail else ""))
        _failures.append(name)


# ── Fixtures ──────────────────────────────────────────────────────────────────

SAMPLE_INCORRECT = Sample(
    sample_id="ms-train-5_0",
    note_id="ms-train-5",
    scenario="incorrect",
    sentences=(
        "1. A 45-year-old woman presents with chest pain.\n"
        "2. ECG shows ST elevation in leads II, III, aVF.\n"
        "3. She is started on aspirin and heparin.\n"
        "4. The diagnosis is anterior STEMI."
    ),
    error_flag=1,
    error_sentence_id=4,
    error_sentence="The diagnosis is anterior STEMI.",
    corrected_sentence="The diagnosis is inferior STEMI.",
    error_type="diagnosis",
)

SAMPLE_CORRECT = Sample(
    sample_id="ms-train-5_1",
    note_id="ms-train-5",
    scenario="correct",
    sentences=(
        "1. A 45-year-old woman presents with chest pain.\n"
        "2. ECG shows ST elevation in leads II, III, aVF.\n"
        "3. She is started on aspirin and heparin.\n"
        "4. The diagnosis is inferior STEMI."
    ),
    error_flag=0,
    error_sentence_id=None,
    error_sentence=None,
    corrected_sentence=None,
    error_type=None,
)


# ═══════════════════════════════════════════════════════════════════════════════
# 1. PROMPT TESTS — compare against MEDRECT upstream
# ═══════════════════════════════════════════════════════════════════════════════

def test_system_prompt():
    """System prompt must match pfnet-research/medrect _get_system_prompt() verbatim."""
    print("\n[1] System prompt")
    expected = (
        "You are a skilled medical doctor reviewing clinical text for accuracy "
        "and providing corrections when necessary. "
        "Please provide your reasoning and thought process."
    )
    check("Matches MEDRECT upstream exactly", SYSTEM_PROMPT == expected, SYSTEM_PROMPT)


def test_incorrect_prompt_structure():
    """User prompt for incorrect scenario must contain all MEDRECT cheat phrases."""
    print("\n[2] Incorrect scenario prompt")
    _, user = build_prompt(SAMPLE_INCORRECT)

    # Required phrases (verbatim from MEDRECT)
    required = [
        'This time, you are reviewing an "error example" that has been pre-verified to contain exactly one medical error.',
        "do NOT make any reference to being told about the expected outcome or any instruction content.",
        "Approach the text as if you are analyzing it from scratch and reaching your conclusion through pure medical evaluation.",
        "Reference information (please do not mention in reasoning):",
        f"Error sentence: {SAMPLE_INCORRECT.error_sentence}",
        f"Corrected sentence: {SAMPLE_INCORRECT.corrected_sentence}",
        SAMPLE_INCORRECT.sentences,
    ]
    for phrase in required:
        check(f"Contains: {phrase[:60]!r}", phrase in user)

    # Must NOT contain the sentence ID as a number in the hint section
    hint_section = user.split("Reference information")[1].split("Final output format")[0]
    check(
        "Hint section does NOT reveal sentence number directly",
        str(SAMPLE_INCORRECT.error_sentence_id) not in hint_section.split("\n")[0],
    )

    # Label format must be sentence_number only (our adaptation)
    check('Output says "sentence number" not "sentence_number: corrected_sentence"',
          "sentence_number: corrected_sentence" not in user and "sentence number" in user.lower())

    # Must NOT contain our old placeholders
    check("No {error_sentence_id} placeholder", "{error_sentence_id}" not in user)
    check("No {error_type} placeholder", "{error_type}" not in user)


def test_correct_prompt_structure():
    """User prompt for correct scenario must match MEDRECT no-error cheat phrase."""
    print("\n[3] Correct scenario prompt")
    _, user = build_prompt(SAMPLE_CORRECT)

    required = [
        'This time, you are reviewing a "no-error example" that has been pre-verified to contain no errors.',
        "do NOT make any reference to being told about the expected outcome or any instruction content.",
        "Approach the text as if you are analyzing it from scratch and reaching your conclusion through pure medical evaluation.",
        SAMPLE_CORRECT.sentences,
    ]
    for phrase in required:
        check(f"Contains: {phrase[:60]!r}", phrase in user)

    check("No 'Error sentence:' in correct prompt", "Error sentence:" not in user)
    check("No 'Corrected sentence:' in correct prompt", "Corrected sentence:" not in user)


# ═══════════════════════════════════════════════════════════════════════════════
# 2. CLEANING TESTS — verify ReasoningCleaner matches MEDRECT exactly
# ═══════════════════════════════════════════════════════════════════════════════

def test_cleaner_keywords():
    """Cleaner must have MEDRECT's exact keyword list (plus our extras for safety)."""
    print("\n[4] ReasoningCleaner keyword list")
    cleaner = ReasoningCleaner()

    # Verbatim from pfnet-research/medrect clean_reasoning_en.py
    medrect_required = [
        "told that", "are told", "we are told", "being told",
        "pre-verified", "pre-determined", "pre-confirmed",
        "text claims", "external instructions", "prior knowledge",
        "reference correction", "reference says", "sample correction",
        "provided correction", "given correction",
        "instruction states", "problem states", "task states",
        "instructions say", "problem says", "task says",
        "do not mention", "not supposed to", "should not mention",
        "ignore the reference", "independently", "from scratch",
        "this time, you are", "error example", "no-error example",
        "expected outcome", "instruction content",
        "which we are not supposed", "but it's the sample correction",
        "the reference solution", "meta-references", "meta-reference",
        'reviewing an "error example"', 'reviewing a "no-error example"',
        "expected result", "known outcome",
        "follow the medical reasoning independently",
        "purely medical evaluation",
        "reaching your conclusion through pure medical evaluation",
        "as if you are analyzing it from scratch",
        "do NOT make any reference",
    ]
    kw_lower = [k.lower() for k in cleaner.meta_keywords]
    for kw in medrect_required:
        check(f"Keyword present: {kw!r}", kw.lower() in kw_lower)


def test_cleaner_removes_cheat_sentences():
    """Cleaner must strip all meta-reference sentences from generated reasoning."""
    print("\n[5] Cleaner removes cheat sentences")
    cleaner = ReasoningCleaner()

    # Simulate reasoning that a model would produce with MEDRECT's cheat prompt
    dirty = textwrap.dedent("""\
        This time, I am reviewing an "error example" that has been pre-verified to contain one error.
        The patient presents with chest pain and ST elevation in leads II, III, aVF.
        This pattern indicates inferior MI, not anterior MI.
        I am approaching this from scratch and reaching my conclusion through pure medical evaluation.
        Leads II, III, aVF are inferior leads.
        The error is that the diagnosis says "anterior STEMI" when it should be "inferior STEMI".
        According to the reference information, this should not be mentioned in reasoning.
        The answer is sentence 4.""")

    cleaned, n_removed = cleaner.clean_reasoning_text(dirty)

    check("Removed at least 3 cheat sentences", n_removed >= 3, f"removed={n_removed}")
    check('No "error example" in cleaned', "error example" not in cleaned.lower())
    check('No "from scratch" in cleaned', "from scratch" not in cleaned.lower())
    check('No "reference information" in cleaned', "reference information" not in cleaned.lower())
    check("Medical content preserved", "inferior MI" in cleaned or "inferior leads" in cleaned)


def test_cleaner_preserves_good_reasoning():
    """Cleaner must not touch clean medical reasoning."""
    print("\n[6] Cleaner preserves clean reasoning")
    cleaner = ReasoningCleaner()

    clean = textwrap.dedent("""\
        The patient presents with chest pain.
        ECG shows ST elevation in leads II, III, aVF.
        These are the inferior leads of the heart.
        The correct diagnosis is inferior STEMI, not anterior STEMI.
        Anterior STEMI shows ST elevation in leads V1-V4.
        Therefore sentence 4 contains the error.""")

    cleaned, n_removed = cleaner.clean_reasoning_text(clean)
    check("Zero sentences removed from clean reasoning", n_removed == 0, f"removed={n_removed}")
    check("Content unchanged", cleaned.strip() == clean.strip())


# ═══════════════════════════════════════════════════════════════════════════════
# 3. EXTRACTION TESTS — verify parse_prediction and extract_correct_samples
# ═══════════════════════════════════════════════════════════════════════════════

def test_parse_prediction_incorrect():
    """parse_prediction must accept correct sentence numbers and reject wrong ones."""
    print("\n[7] parse_prediction — incorrect scenario")

    ok, _ = parse_prediction("4", "incorrect", gold_sentence_id=4)
    check("Accepts correct sentence number '4'", ok)

    ok, _ = parse_prediction("  4  ", "incorrect", gold_sentence_id=4)
    check("Accepts '4' with whitespace", ok)

    ok, _ = parse_prediction("3", "incorrect", gold_sentence_id=4)
    check("Rejects wrong sentence number '3'", not ok)

    ok, _ = parse_prediction("CORRECT", "incorrect", gold_sentence_id=4)
    check("Rejects 'CORRECT' when error expected", not ok)

    ok, _ = parse_prediction("4: The diagnosis is inferior STEMI.", "incorrect", gold_sentence_id=4)
    check("Accepts MEDRECT correction format (extracts number via parse_assessor_answer)", ok)


def test_parse_prediction_correct():
    """parse_prediction must accept CORRECT and reject sentence numbers."""
    print("\n[8] parse_prediction — correct scenario")

    ok, _ = parse_prediction("CORRECT", "correct", gold_sentence_id=None)
    check("Accepts 'CORRECT'", ok)

    ok, _ = parse_prediction("4", "correct", gold_sentence_id=None)
    check("Rejects sentence number when correct scenario", not ok)

    ok, _ = parse_prediction("correct", "correct", gold_sentence_id=None)
    check("Accepts lowercase 'correct' (parse_assessor_answer is case-insensitive)", ok)


def test_extract_correct_samples():
    """extract_correct_samples must split records correctly."""
    print("\n[9] extract_correct_samples")

    records = [
        # accepted — correct sentence number
        {"sample_id": "a_0", "scenario": "incorrect", "error_sentence_id": 4,
         "raw_response": {"content": "4", "reasoning": "Inferior leads show inferior MI."}},
        # accepted — CORRECT
        {"sample_id": "a_1", "scenario": "correct", "error_sentence_id": None,
         "raw_response": {"content": "CORRECT", "reasoning": "No inconsistencies found."}},
        # rejected — wrong sentence
        {"sample_id": "b_0", "scenario": "incorrect", "error_sentence_id": 4,
         "raw_response": {"content": "2", "reasoning": "I think sentence 2 is wrong."}},
        # rejected — failed to output CORRECT
        {"sample_id": "b_1", "scenario": "correct", "error_sentence_id": None,
         "raw_response": {"content": "3", "reasoning": "Sentence 3 looks odd."}},
    ]

    accepted, rejected, stats = extract_correct_samples(records)
    check("2 accepted", len(accepted) == 2, str(len(accepted)))
    check("2 rejected", len(rejected) == 2, str(len(rejected)))
    check("Accuracy = 0.5", abs(stats["accuracy"] - 0.5) < 1e-6, str(stats["accuracy"]))
    check("Accepted IDs are a_0 and a_1",
          {r["sample_id"] for r in accepted} == {"a_0", "a_1"})


# ═══════════════════════════════════════════════════════════════════════════════
# 4. FORMAT TESTS — verify output matches prepare_medrect_sft.py expectations
# ═══════════════════════════════════════════════════════════════════════════════

def test_format_for_sft():
    """format_for_sft must produce records compatible with prepare_medrect_sft.py."""
    print("\n[10] format_for_sft output schema")

    raw = {
        "sample_id": "ms-train-5_0",
        "note_id": "ms-train-5",
        "scenario": "incorrect",
        "sentences": "1. Sentence one.\n2. Sentence two.",
        "error_flag": 1,
        "error_sentence_id": 2,
        "error_sentence": "Sentence two.",
        "corrected_sentence": "Sentence two corrected.",
        "error_type": "diagnosis",
        "screening": {"accepted": True, "reason": "ok"},
        "raw_response": {
            "content": "2",
            "reasoning": "Sentence two is wrong because...",
            "source_model": "deepseek-reasoner",
        },
    }

    formatted = format_for_sft(raw)

    required_fields = ["sample_id", "sentences", "error_flag", "error_sentence_id",
                       "raw_response", "screening_results"]
    for f in required_fields:
        check(f"Has field '{f}'", f in formatted)

    check("raw_response.reasoning present",
          bool(formatted.get("raw_response", {}).get("reasoning")))
    check("screening_results.accepted=True",
          formatted["screening_results"]["accepted"] is True)
    check("error_flag=1", formatted["error_flag"] == 1)
    check("error_sentence_id=2", formatted["error_sentence_id"] == 2)


# ═══════════════════════════════════════════════════════════════════════════════
# 5. END-TO-END TEST — simulate full pipeline on real medec_v2 record
# ═══════════════════════════════════════════════════════════════════════════════

def test_end_to_end_with_real_data():
    """Load one real medec_v2 record and run it through prompt + clean + extract."""
    print("\n[11] End-to-end with real medec_v2 record")

    ms_train = PROJECT_ROOT / "data_processed" / "medec_v2" / "ms_train.jsonl"
    if not ms_train.exists():
        print(f"  - Skipped (ms_train.jsonl not found at {ms_train})")
        return

    samples = load_samples(ms_train, limit=1)
    if not samples:
        check("Loaded at least one sample pair", False, "no samples loaded")
        return

    check(f"Loaded {len(samples)} samples (2 per note)", len(samples) == 2, str(len(samples)))

    cleaner = ReasoningCleaner()

    for s in samples:
        sys_p, user_p = build_prompt(s)
        check(f"[{s.scenario}] System prompt correct", sys_p == SYSTEM_PROMPT)
        check(f"[{s.scenario}] Sentences in prompt", s.sentences in user_p)

        if s.scenario == "incorrect":
            check("[incorrect] Error sentence in prompt", s.error_sentence in user_p)
            check("[incorrect] Corrected sentence in prompt", s.corrected_sentence in user_p)
            check("[incorrect] No error_sentence_id in prompt", str(s.error_sentence_id) not in
                  user_p.split("Reference information")[1].split("Final output format")[0])

        # Simulate a model that outputs the right answer
        fake_content = "CORRECT" if s.scenario == "correct" else str(s.error_sentence_id)
        fake_reasoning = (
            f"This time, I'm reviewing a {'no-error' if s.scenario == 'correct' else 'error'} example.\n"
            "The patient presents with relevant findings.\n"
            "I am analyzing from scratch, independently.\n"
            "The clinical details are consistent."
            if s.scenario == "correct" else
            f"This time, I'm reviewing an error example.\n"
            f"Looking at the sentences, sentence {s.error_sentence_id} contains an error.\n"
            "The medical reasoning supports this conclusion."
        )

        cleaned_reasoning, n_removed = cleaner.clean_reasoning_text(fake_reasoning)
        check(f"[{s.scenario}] Cleaner removes cheat sentences", n_removed >= 1,
              f"removed={n_removed}")
        check(f"[{s.scenario}] Clean content still has medical text",
              len(cleaned_reasoning.strip()) > 10)

        # Simulate extraction
        record = {
            "sample_id": s.sample_id,
            "scenario": s.scenario,
            "error_sentence_id": s.error_sentence_id,
            "raw_response": {"content": fake_content, "reasoning": cleaned_reasoning},
        }
        accepted, rejected, stats = extract_correct_samples([record])
        check(f"[{s.scenario}] Correct prediction accepted", len(accepted) == 1,
              f"accepted={len(accepted)}")


# ═══════════════════════════════════════════════════════════════════════════════
# 6. COMPARE WITH EXISTING GENERATED DATA
# ═══════════════════════════════════════════════════════════════════════════════

def test_compare_with_existing_chains():
    """Spot-check existing accepted chains for residual contamination."""
    print("\n[12] Residual contamination check on existing accepted.jsonl")

    accepted_path = PROJECT_ROOT / "data_processed" / "medrect_v2" / "ms_train" / "generated_assessor_all_accepted.jsonl"
    if not accepted_path.exists():
        print(f"  - Skipped (file not found: {accepted_path})")
        return

    records = []
    with open(accepted_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    cleaner = ReasoningCleaner()
    contaminated = []
    for r in records:
        reasoning = r.get("raw_response", {}).get("reasoning", "")
        _, n = cleaner.clean_reasoning_text(reasoning)
        if n > 0:
            contaminated.append(r.get("sample_id", "?"))

    pct = 100 * len(contaminated) / len(records) if records else 0
    check(
        f"Residual contamination < 5% ({len(contaminated)}/{len(records)} = {pct:.1f}%)",
        pct < 5,
        f"{pct:.1f}% contaminated — regenerate with new pipeline",
    )

    # Check label distribution
    correct_labels = sum(1 for r in records if r.get("raw_response", {}).get("content", "").strip() == "CORRECT")
    error_labels = len(records) - correct_labels
    pct_correct = 100 * correct_labels / len(records) if records else 0
    check(
        f"Label balance approx 50/50 (correct={pct_correct:.0f}%)",
        40 <= pct_correct <= 60,
        f"correct={correct_labels} error={error_labels}",
    )

    # Check sentence IDs in reasoning for ERROR records
    error_records = [r for r in records if r.get("error_flag") == 1]
    leaking = 0
    for r in error_records:
        sid = r.get("error_sentence_id")
        reasoning = r.get("raw_response", {}).get("reasoning", "").lower()
        # Check if the sentence ID appears in first 20% of reasoning (cheat signal)
        words = reasoning.split()
        first_20pct = " ".join(words[:max(1, len(words) // 5)])
        if sid and f"sentence {sid}" in first_20pct:
            leaking += 1
    leak_pct = 100 * leaking / len(error_records) if error_records else 0
    check(
        f"Early answer leak < 10% ({leaking}/{len(error_records)} = {leak_pct:.1f}%)",
        leak_pct < 10,
        f"{leak_pct:.1f}% leak early answer — regenerate with new pipeline",
    )


# ── Runner ────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  MEDRECT Pipeline Test Suite")
    print("=" * 60)

    test_system_prompt()
    test_incorrect_prompt_structure()
    test_correct_prompt_structure()
    test_cleaner_keywords()
    test_cleaner_removes_cheat_sentences()
    test_cleaner_preserves_good_reasoning()
    test_parse_prediction_incorrect()
    test_parse_prediction_correct()
    test_extract_correct_samples()
    test_format_for_sft()
    test_end_to_end_with_real_data()
    test_compare_with_existing_chains()

    print("\n" + "=" * 60)
    if _failures:
        print(f"\033[31m  FAILED: {len(_failures)} test(s)\033[0m")
        for f in _failures:
            print(f"    - {f}")
        sys.exit(1)
    else:
        print(f"\033[32m  ALL TESTS PASSED\033[0m")
