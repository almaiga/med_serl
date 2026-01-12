#!/usr/bin/env python3
"""
Convert paired RL notes into OpenRLHF prompt/label JSONL.
"""

import argparse
import json
import os
from typing import Dict, List, Optional


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare OpenRLHF RL prompts from paired MEDEC data.")
    parser.add_argument(
        "--input",
        default="data_processed/medec_paired/train_val_split/rl_train.jsonl",
        help="Input paired RL JSONL (correct_note + incorrect_note).",
    )
    parser.add_argument(
        "--output",
        default="data_processed/medec_paired/train_val_split/rl_train_openrlhf.jsonl",
        help="Output JSONL with input/label fields for OpenRLHF.",
    )
    parser.add_argument(
        "--max-notes",
        type=int,
        default=None,
        help="Limit number of paired notes to convert (for quick tests).",
    )
    return parser.parse_args()


def build_rows(record: Dict) -> List[Dict]:
    note_id = record.get("note_id")
    error_type = (record.get("error_type") or "").strip()
    incorrect_label = f"Error: {error_type}" if error_type else "Error"
    rows = []
    correct_note = record.get("correct_note")
    incorrect_note = record.get("incorrect_note")
    if correct_note:
        rows.append(
            {
                "note_id": f"{note_id}_correct" if note_id else None,
                "input": correct_note,
                "label": "Clean",
                "error_type": None,
            }
        )
    if incorrect_note:
        rows.append(
            {
                "note_id": f"{note_id}_incorrect" if note_id else None,
                "input": incorrect_note,
                "label": incorrect_label,
                "error_type": error_type or None,
            }
        )
    return rows


def convert(input_path: str, output_path: str, max_notes: Optional[int]) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    total_records = 0
    total_rows = 0

    with open(input_path, "r", encoding="utf-8") as in_handle, open(
        output_path, "w", encoding="utf-8"
    ) as out_handle:
        for line in in_handle:
            if max_notes is not None and total_records >= max_notes:
                break
            record = json.loads(line)
            rows = build_rows(record)
            for row in rows:
                out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            total_records += 1
            total_rows += len(rows)

    print(f"Wrote {total_rows} rows from {total_records} paired notes to {output_path}")


def main() -> None:
    args = parse_args()
    convert(args.input, args.output, args.max_notes)


if __name__ == "__main__":
    main()
