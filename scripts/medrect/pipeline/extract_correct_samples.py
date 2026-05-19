#!/usr/bin/env python3
"""
Extract correctly-predicted samples from cleaned raw_responses JSONL.

Adapted from pfnet-research/medrect/scripts/training_data/extract_correct_samples.py
Differences from upstream:
  - Input is JSONL (one record per line) not nested JSON
  - Gold answer for "incorrect" scenario is just the sentence number (e.g. "7")
    not "sentence_number: corrected_sentence" — because we do detection+location only
  - Gold answer for "correct" scenario is "CORRECT"

Pipeline step 3 of 3. Run after clean_reasoning_en.py.

Usage:
    python extract_correct_samples.py \\
        --input data_processed/medrect_v2/ms_train/raw_responses_*.jsonl \\
        --output-dir data_processed/medrect_v2/ms_train
"""

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.self_play.utils import parse_assessor_answer

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ── Prediction parsing ────────────────────────────────────────────────────────

def parse_prediction(content: str, scenario: str, gold_sentence_id: Optional[int]) -> Tuple[bool, str]:
    """
    Check whether the model's prediction is correct.

    Returns (is_correct, reason_string).
    Mirrors pfnet-research/medrect extract_correct_samples extract_samples_from_predictions logic.
    """
    content = (content or "").strip()

    if scenario == "correct":
        # Gold: CORRECT
        if content.upper() == "CORRECT":
            return True, "ok"
        return False, f"expected CORRECT, got {content!r}"

    # scenario == "incorrect"
    # Gold: the sentence number as a string (e.g. "7")
    label, sid = parse_assessor_answer(content)
    if label == "ERROR" and sid == gold_sentence_id:
        return True, "ok"
    return False, f"expected sentence {gold_sentence_id}, got {content!r}"


# ── Loading ───────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ── Extraction ────────────────────────────────────────────────────────────────

def extract_correct_samples(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """
    Split records into accepted (correct prediction) and rejected.

    Returns (accepted, rejected, stats).
    Mirrors pfnet-research/medrect extract_samples_from_predictions.
    """
    accepted = []
    rejected = []

    n_correct_scenario = 0
    n_incorrect_scenario = 0
    n_correct_prediction = 0
    n_incorrect_prediction = 0

    for record in records:
        scenario = record.get("scenario", "incorrect")
        content = record.get("raw_response", {}).get("content", "")
        gold_sid = record.get("error_sentence_id")

        if scenario == "correct":
            n_correct_scenario += 1
        else:
            n_incorrect_scenario += 1

        is_correct, reason = parse_prediction(content, scenario, gold_sid)

        # Attach screening result (mirrors MEDRECT metadata)
        record = record.copy()
        record["screening"] = {"accepted": is_correct, "reason": reason}

        if is_correct:
            n_correct_prediction += 1
            accepted.append(record)
        else:
            n_incorrect_prediction += 1
            rejected.append(record)

    total = len(records)
    stats = {
        "total": total,
        "accepted": len(accepted),
        "rejected": len(rejected),
        "accuracy": len(accepted) / total if total else 0.0,
        "correct_scenario": n_correct_scenario,
        "incorrect_scenario": n_incorrect_scenario,
    }
    return accepted, rejected, stats


# ── Output formatting ─────────────────────────────────────────────────────────

def format_for_sft(record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a raw record to the format expected by prepare_medrect_sft.py.

    Mirrors MEDRECT's jmedec_dataset output format used as input to the SFT formatter.
    Fields match what generate_missing_assessor_chains.py used to produce.
    """
    raw_response = record.get("raw_response", {})
    return {
        "sample_id": record.get("sample_id"),
        "sentences": record.get("sentences", ""),
        "error_flag": record.get("error_flag", 0),
        "error_type": record.get("error_type"),
        "error_sentence_id": record.get("error_sentence_id"),
        "error_sentence": record.get("error_sentence"),
        "corrected_sentence": record.get("corrected_sentence"),
        "metadata": {
            "note_id": record.get("note_id"),
            "scenario": record.get("scenario"),
            "source": "pipeline_v2",
        },
        "raw_response": {
            "content": raw_response.get("content", ""),
            "reasoning": raw_response.get("reasoning", ""),
            "source_model": raw_response.get("source_model", "deepseek-reasoner"),
        },
        "screening_results": {
            "accepted": record.get("screening", {}).get("accepted", False),
            "reason": record.get("screening", {}).get("reason", ""),
        },
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract correct samples from cleaned JSONL (step 3 of 3)")
    parser.add_argument("--input", nargs="+", required=True,
                        help="One or more cleaned raw_responses JSONL files")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to write accepted.jsonl, rejected.jsonl, stats.json")
    parser.add_argument("--accepted-name", default="generated_assessor_all_accepted.jsonl")
    parser.add_argument("--rejected-name", default="generated_assessor_all_rejected.jsonl")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load all input files
    all_records = []
    for pattern in args.input:
        for path in sorted(Path(".").glob(pattern)):
            recs = load_jsonl(path)
            logger.info("Loaded %d records from %s", len(recs), path)
            all_records.extend(recs)

    if not all_records:
        logger.error("No records loaded. Check --input paths.")
        sys.exit(1)

    logger.info("Total records: %d", len(all_records))

    accepted, rejected, stats = extract_correct_samples(all_records)

    logger.info(
        "Accepted: %d / %d  (%.1f%%)",
        stats["accepted"], stats["total"], 100 * stats["accuracy"],
    )

    # Write accepted (formatted for prepare_medrect_sft.py)
    accepted_path = output_dir / args.accepted_name
    with open(accepted_path, "w", encoding="utf-8") as f:
        for r in accepted:
            f.write(json.dumps(format_for_sft(r), ensure_ascii=False) + "\n")
    logger.info("Accepted → %s", accepted_path)

    # Write rejected
    rejected_path = output_dir / args.rejected_name
    with open(rejected_path, "w", encoding="utf-8") as f:
        for r in rejected:
            f.write(json.dumps(format_for_sft(r), ensure_ascii=False) + "\n")
    logger.info("Rejected → %s", rejected_path)

    # Write stats
    stats_path = output_dir / "generated_assessor_all_summary.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info("Stats → %s", stats_path)

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
