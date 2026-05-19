#!/usr/bin/env python3
"""
Build clean MEDEC split files from raw CSVs.

Reads all MEDEC CSV files and writes JSONL split files with sentence numbering
taken directly from the MEDEC `Sentences` field (via sentences_to_1indexed),
which is the format used by all evaluation scripts.

Outputs → data_processed/medec_v2/:
  ms_train.jsonl   — MS training set (2189 rows: ERROR + CORRECT)
  ms_val.jsonl     — MS validation set (574 rows)
  uw_val.jsonl     — UW validation set (160 rows)
  rl_split.jsonl   — Subset of ms_train ERROR rows whose note_id is in rl_train.jsonl
  sft_split.jsonl  — Remaining ms_train ERROR rows (not in rl_split)
  rl_split_ids.json — Set of note_ids in the rl split (for downstream scripts)

Record schema (all splits):
  note_id          str   e.g. "ms-train-0"
  split            str   "ms_train" | "ms_val" | "uw_val"
  error_flag       int   0 or 1
  sentences        str   1-indexed numbered sentences from MEDEC CSV
  error_sentence_id int|None  1-indexed (CSV value + 1), None for CORRECT rows
  error_type       str|None
  error_sentence   str|None
  corrected_sentence str|None
  text             str   raw note text (incorrect for ERROR, correct for CORRECT)
  corrected_text   str|None  corrected note text for ERROR rows, None for CORRECT

Usage:
  python scripts/medrect/build_medec_splits.py
  python scripts/medrect/build_medec_splits.py --output-dir data_processed/medec_v2
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.self_play.utils import sentences_to_1indexed

CSV_FILES = {
    "ms_train": PROJECT_ROOT / "data_raw/MEDEC/MEDEC-MS/MEDEC-Full-TrainingSet-with-ErrorType.csv",
    "ms_val":   PROJECT_ROOT / "data_raw/MEDEC/MEDEC-MS/MEDEC-MS-ValidationSet-with-GroundTruth-and-ErrorType.csv",
    "uw_val":   PROJECT_ROOT / "data_raw/MEDEC/MEDEC-UW/MEDEC-UW-ValidationSet-with-GroundTruth-and-ErrorType.csv",
}

DEFAULT_RL_IDS_SOURCE = PROJECT_ROOT / "data_processed/medec_paired/train_val_split/rl_train.jsonl"
DEFAULT_OUTPUT_DIR    = PROJECT_ROOT / "data_processed/medec_v2"


def derive_correct_sentences(
    sentences: str,
    error_sentence_id: int | None,
    corrected_sentence: str | None,
) -> str:
    """Fallback: substitute the error sentence with corrected_sentence in numbered string."""
    if not error_sentence_id or not corrected_sentence:
        return sentences
    lines = sentences.split("\n")
    result = []
    for line in lines:
        m = re.match(r"^(\d+)\.\s+(.+)$", line.strip())
        if m and int(m.group(1)) == error_sentence_id:
            result.append(f"{error_sentence_id}. {corrected_sentence.strip()}")
        else:
            result.append(line)
    return "\n".join(result)


def load_rl_note_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    if not path.exists():
        print(f"  Warning: rl_train path not found: {path}", file=sys.stderr)
        return ids
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                ids.add(json.loads(line).get("note_id", ""))
    return ids


def read_csv_split(path: Path, split_name: str) -> list[dict]:
    records = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            flag_raw = row.get("Error Flag", "").strip()
            if not flag_raw:
                continue
            error_flag = int(flag_raw)
            is_error = error_flag == 1

            sentences_raw = row.get("Sentences", "")
            sentences = sentences_to_1indexed(sentences_raw) if sentences_raw.strip() else ""

            error_sid_raw = row.get("Error Sentence ID", "").strip()
            if is_error and error_sid_raw and error_sid_raw not in ("NA", "-1"):
                error_sentence_id = int(error_sid_raw) + 1  # 0-indexed → 1-indexed
            else:
                error_sentence_id = None

            corrected_text_raw = row.get("Corrected Text", "").strip()
            corrected_text = corrected_text_raw if is_error and corrected_text_raw and corrected_text_raw != "NA" else None
            corrected_sentence = row.get("Corrected Sentence", "").strip() or None if is_error else None

            # correct_sentences: error note's sentences with the error sentence substituted.
            # MEDEC CORRECT rows are different text variants, not direct pairs — lookup
            # matching via Corrected Text does not work. Substitution is the only correct approach.
            correct_sentences = derive_correct_sentences(
                sentences, error_sentence_id, corrected_sentence
            ) if is_error else None

            records.append({
                "note_id":            row["Text ID"].strip(),
                "split":              split_name,
                "error_flag":         error_flag,
                "sentences":          sentences,
                "correct_sentences":  correct_sentences,
                "error_sentence_id":  error_sentence_id,
                "error_type":         row.get("Error Type", "").strip() or None if is_error else None,
                "error_sentence":     row.get("Error Sentence", "").strip() or None if is_error else None,
                "corrected_sentence": corrected_sentence,
                "text":               row.get("Text", "").strip(),
                "corrected_text":     corrected_text,
            })
    return records


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"  Written: {path} ({len(records)} records)")


def print_stats(records: list[dict], name: str) -> None:
    total   = len(records)
    n_error = sum(1 for r in records if r["error_flag"] == 1)
    n_correct = total - n_error
    n_no_sid = sum(1 for r in records if r["error_flag"] == 1 and r["error_sentence_id"] is None)
    print(f"  {name}: {total} rows  (ERROR={n_error}, CORRECT={n_correct}"
          + (f", ERROR_missing_sid={n_no_sid}" if n_no_sid else "") + ")")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build clean MEDEC split files from raw CSVs.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--rl-ids", default=str(DEFAULT_RL_IDS_SOURCE),
                        help="Path to rl_train.jsonl for tagging the rl split")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rl_ids = load_rl_note_ids(Path(args.rl_ids))
    print(f"Loaded {len(rl_ids)} rl_train note IDs from {args.rl_ids}")

    all_splits: dict[str, list[dict]] = {}

    print("\nReading MEDEC CSVs...")
    for split_name, csv_path in CSV_FILES.items():
        if not csv_path.exists():
            print(f"  Skipping (not found): {csv_path}", file=sys.stderr)
            continue
        records = read_csv_split(csv_path, split_name)
        all_splits[split_name] = records
        print_stats(records, split_name)
        write_jsonl(output_dir / f"{split_name}.jsonl", records)

    # Build rl_split and sft_split from ms_train ERROR rows
    ms_train = all_splits.get("ms_train", [])
    ms_train_errors = [r for r in ms_train if r["error_flag"] == 1]
    rl_split  = [r for r in ms_train_errors if r["note_id"] in rl_ids]
    sft_split = [r for r in ms_train_errors if r["note_id"] not in rl_ids]

    print(f"\nBuilding rl/sft splits from ms_train ERROR rows ({len(ms_train_errors)} total):")
    print_stats(rl_split,  "rl_split")
    print_stats(sft_split, "sft_split")
    write_jsonl(output_dir / "rl_split.jsonl",  rl_split)
    write_jsonl(output_dir / "sft_split.jsonl", sft_split)

    # Write rl note_id set for downstream scripts
    rl_ids_path = output_dir / "rl_split_ids.json"
    with open(rl_ids_path, "w") as f:
        json.dump(sorted(rl_ids), f, indent=2)
    print(f"  Written: {rl_ids_path} ({len(rl_ids)} IDs)")

    print("\nDone.")


if __name__ == "__main__":
    main()
