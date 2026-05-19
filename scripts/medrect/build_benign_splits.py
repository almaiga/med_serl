#!/usr/bin/env python3
"""
Enrich benign_train_clean.jsonl with MEDEC-aligned sentence numbering.

For each benign record, finds changed_sid by exact-matching original_sentence
against the MEDEC CSV pre-split (correct_sentences from medec_v2). Records
where the match fails are dropped to keep data quality high.

Output → data_processed/benign_changes/benign_v2.jsonl:
  All fields from benign_train_clean.jsonl, plus:
    sentences   str   MEDEC-aligned numbered sentences (correct version of note)
    changed_sid int   1-indexed position of the changed sentence in sentences

Usage:
  python scripts/medrect/build_benign_splits.py
  python scripts/medrect/build_benign_splits.py --output data_processed/benign_changes/benign_v2.jsonl
"""

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_BENIGN_INPUT = PROJECT_ROOT / "data_processed/benign_changes/benign_train_clean.jsonl"
DEFAULT_MEDEC_DIR    = PROJECT_ROOT / "data_processed/medec_v2"
DEFAULT_OUTPUT       = PROJECT_ROOT / "data_processed/benign_changes/benign_v2.jsonl"

MEDEC_SPLITS = ["ms_train", "ms_val", "uw_val"]


def build_lookup(medec_dir: Path) -> dict:
    """Load {note_id: {sentences, correct_sentences}} for all ERROR rows."""
    lookup = {}
    for split in MEDEC_SPLITS:
        path = medec_dir / f"{split}.jsonl"
        if not path.exists():
            print(f"  Warning: not found: {path}", file=sys.stderr)
            continue
        with open(path) as f:
            for line in f:
                r = json.loads(line)
                if r.get("error_flag") == 1:
                    lookup[r["note_id"]] = {
                        "sentences":         r.get("sentences", ""),
                        "correct_sentences": r.get("correct_sentences", ""),
                    }
    return lookup


def find_changed_sid(correct_sentences: str, original_sentence: str) -> int | None:
    """Exact match of original_sentence against MEDEC pre-split numbered sentences."""
    target = (original_sentence or "").strip()
    if not target:
        return None
    for line in correct_sentences.split("\n"):
        m = re.match(r"^(\d+)\.\s+(.+)$", line.strip())
        if m and m.group(2).strip() == target:
            return int(m.group(1))
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Enrich benign data with MEDEC sentence numbering.")
    parser.add_argument("--input",      default=str(DEFAULT_BENIGN_INPUT))
    parser.add_argument("--medec-dir",  default=str(DEFAULT_MEDEC_DIR))
    parser.add_argument("--output",     default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    medec_dir = Path(args.medec_dir)
    print(f"Loading medec_v2 lookup from {medec_dir}...")
    lookup = build_lookup(medec_dir)
    print(f"  Loaded {len(lookup)} ERROR rows")

    input_path = Path(args.input)
    with open(input_path) as f:
        records = [json.loads(l) for l in f if l.strip()]
    print(f"Input records: {len(records)}")

    kept = dropped = 0
    drop_reasons: dict[str, int] = {}
    output_records = []

    for rec in records:
        note_id = rec.get("note_id", "")
        original_sentence = rec.get("original_sentence", "").strip()

        medec_row = lookup.get(note_id)
        if medec_row is None:
            dropped += 1
            drop_reasons["no_medec_entry"] = drop_reasons.get("no_medec_entry", 0) + 1
            continue

        correct_sentences = medec_row["correct_sentences"]
        if not correct_sentences:
            dropped += 1
            drop_reasons["no_correct_sentences"] = drop_reasons.get("no_correct_sentences", 0) + 1
            continue

        changed_sid = find_changed_sid(correct_sentences, original_sentence)
        if changed_sid is None:
            dropped += 1
            drop_reasons["sentence_not_found"] = drop_reasons.get("sentence_not_found", 0) + 1
            continue

        output_records.append({
            **rec,
            "sentences":   correct_sentences,
            "changed_sid": changed_sid,
        })
        kept += 1

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in output_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nKept:    {kept}")
    print(f"Dropped: {dropped}  {drop_reasons}")
    print(f"Written: {output_path}")


if __name__ == "__main__":
    main()
