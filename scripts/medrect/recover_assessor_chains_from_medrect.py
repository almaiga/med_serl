#!/usr/bin/env python3
"""
Recover MedRECT assessor data by exact text matching against local SFT pairs.

This script matches local pair data from:
  data_processed/medec_paired/train_val_split/sft_train.jsonl

against a cloned MEDRECT repo:
  medrect/data/medec/medec-train-and-val.json
  medrect/data/medrect/medrect-en-train.json

Outputs:
  - exact raw MEDRECT matches (coverage check)
  - exact accepted-chain matches from medrect-en-train
  - summary with unmatched IDs
"""

import argparse
import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SFT = PROJECT_ROOT / "data_processed" / "medec_paired" / "train_val_split" / "sft_train.jsonl"
DEFAULT_MEDRECT_REPO = PROJECT_ROOT / "medrect"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_processed" / "medrect"


def normalize(text: str | None) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def pair_key(incorrect_note: str, correct_note: str) -> Tuple[str, str]:
    return normalize(incorrect_note), normalize(correct_note)


def medrect_key(record: Dict[str, Any]) -> Tuple[str, str]:
    return (
        normalize(record.get("metadata", {}).get("original_text")),
        normalize(record.get("corrected_text")),
    )


def build_index(records: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    index: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for record in records:
        key = medrect_key(record)
        if key[0] and key[1]:
            index[key] = record
    return index


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover MedRECT assessor chains by exact text match.")
    parser.add_argument("--sft-path", default=str(DEFAULT_SFT), help="Local SFT pair JSONL")
    parser.add_argument("--medrect-repo", default=str(DEFAULT_MEDRECT_REPO), help="Path to cloned MEDRECT repo")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sft_path = Path(args.sft_path)
    medrect_repo = Path(args.medrect_repo)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_path = medrect_repo / "data" / "medec" / "medec-train-and-val.json"
    train_path = medrect_repo / "data" / "medrect" / "medrect-en-train.json"

    raw_records = read_json(raw_path)
    train_records = read_json(train_path)
    raw_index = build_index(raw_records)
    train_index = build_index(train_records)

    matched_raw: List[Dict[str, Any]] = []
    matched_train: List[Dict[str, Any]] = []
    missing_from_raw: List[str] = []
    missing_from_train: List[str] = []
    present_in_raw_but_missing_train: List[str] = []

    total = 0
    for pair in read_jsonl(sft_path):
        total += 1
        key = pair_key(pair.get("incorrect_note", ""), pair.get("correct_note", ""))
        note_id = pair.get("note_id", "unknown")

        raw_hit = raw_index.get(key)
        if raw_hit is None:
            missing_from_raw.append(note_id)
        else:
            row = dict(raw_hit)
            row["_match"] = {
                "local_note_id": note_id,
                "source": "medec-train-and-val.json",
            }
            matched_raw.append(row)

        train_hit = train_index.get(key)
        if train_hit is None:
            missing_from_train.append(note_id)
            if raw_hit is not None:
                present_in_raw_but_missing_train.append(note_id)
        else:
            row = dict(train_hit)
            row["_match"] = {
                "local_note_id": note_id,
                "source": "medrect-en-train.json",
            }
            matched_train.append(row)

    raw_out = output_dir / "recovered_medrect_raw_matches.jsonl"
    train_out = output_dir / "recovered_medrect_assessor_chains.jsonl"
    summary_out = output_dir / "recovered_medrect_match_summary.json"

    write_jsonl(raw_out, matched_raw)
    write_jsonl(train_out, matched_train)

    summary = {
        "sft_pairs_total": total,
        "matched_in_medec_train_and_val": len(matched_raw),
        "matched_in_medrect_en_train": len(matched_train),
        "missing_from_medec_train_and_val": len(missing_from_raw),
        "missing_from_medrect_en_train": len(missing_from_train),
        "present_in_raw_but_missing_train": len(present_in_raw_but_missing_train),
        "missing_from_medec_train_and_val_ids": missing_from_raw,
        "missing_from_medrect_en_train_ids": missing_from_train,
        "present_in_raw_but_missing_train_ids": present_in_raw_but_missing_train,
        "raw_output": str(raw_out),
        "train_output": str(train_out),
    }
    with open(summary_out, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
