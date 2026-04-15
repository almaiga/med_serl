#!/usr/bin/env python3
"""Inspect paired assessor/injector SFT examples for role entanglement.

This is a read-only diagnostic script. It answers a narrow question:

  "For the same base note, how similar are the assessor and injector
   training examples?"

It pairs:
  - assessor rows from a MedRECT-style assessor SFT JSONL
  - injector-error rows from an injector chain JSONL

For each matched base sample ID it reports:
  - whether the note body is identical
  - whether both roles target the same sentence
  - prompt/label/reasoning snippets side by side
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional


def load_jsonl(path: Path) -> Iterable[dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def base_sample_id(sample_id: str) -> str:
    sample_id = sample_id or ""
    for suffix in ("_injector_error", "_injector_benign", "_assessor_error", "_assessor_benign"):
        if sample_id.endswith(suffix):
            return sample_id[: -len(suffix)]
    return sample_id


def extract_note_block(text: str) -> str:
    text = text or ""
    lines = text.splitlines()
    start = None
    for i, line in enumerate(lines):
        if re.match(r"^\s*1\.\s+", line):
            start = i
            break
    if start is None:
        return text.strip()

    end = len(lines)
    for i in range(start + 1, len(lines)):
        line = lines[i].strip()
        if line.startswith("Respond with EXACTLY"):
            end = i
            break
    return "\n".join(lines[start:end]).strip()


def parse_injector_sid(label: str) -> Optional[int]:
    m = re.match(r"^\s*(\d+)\.", label or "")
    return int(m.group(1)) if m else None


def split_numbered_sentences(note: str) -> list[str]:
    out = []
    for line in note.splitlines():
        line = line.strip()
        if re.match(r"^\d+\.\s+", line):
            out.append(line)
    return out


def differing_sentence_ids(note_a: str, note_b: str) -> list[int]:
    a = split_numbered_sentences(note_a)
    b = split_numbered_sentences(note_b)
    diff = []
    for i, (la, lb) in enumerate(zip(a, b), start=1):
        if la != lb:
            diff.append(i)
    if len(a) != len(b):
        longer = max(len(a), len(b))
        diff.extend(range(min(len(a), len(b)) + 1, longer + 1))
    return diff


def short(text: str, limit: int = 220) -> str:
    text = (text or "").strip().replace("\n", " ")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def load_assessor_rows(path: Path) -> Dict[str, dict]:
    rows = {}
    for row in load_jsonl(path):
        sid = base_sample_id(row.get("sample_id", ""))
        if sid:
            rows[sid] = row
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare paired assessor/injector training rows.")
    parser.add_argument(
        "--assessor",
        default="data_processed/medrect/assessor_sft_recovered_plus_uw.jsonl",
        help="Assessor SFT JSONL",
    )
    parser.add_argument(
        "--injector-error",
        default="data_processed/medrect/injector_error_chains_20260310_135156.jsonl",
        help="Injector error-chain JSONL",
    )
    parser.add_argument(
        "--show",
        type=int,
        default=3,
        help="Number of matched examples to print",
    )
    args = parser.parse_args()

    assessor_path = Path(args.assessor)
    injector_path = Path(args.injector_error)

    assessor_rows = load_assessor_rows(assessor_path)

    matches = []
    target_counter = Counter()
    note_equal = 0
    target_equal = 0
    single_sentence_diff = 0
    diff_matches_target = 0

    for inj in load_jsonl(injector_path):
        sid = base_sample_id(inj.get("sample_id", ""))
        ass = assessor_rows.get(sid)
        if not ass:
            continue

        ass_note = extract_note_block(ass.get("user_prompt", ""))
        inj_note = extract_note_block(inj.get("user_prompt", ""))
        same_note = ass_note == inj_note
        note_equal += int(same_note)
        diff_ids = differing_sentence_ids(ass_note, inj_note)

        ass_target = ass.get("label")
        inj_target = parse_injector_sid(inj.get("label", ""))
        same_target = ass_target.isdigit() and inj_target is not None and int(ass_target) == inj_target
        target_equal += int(same_target)
        if len(diff_ids) == 1:
            single_sentence_diff += 1
            if ass_target.isdigit() and int(ass_target) == diff_ids[0]:
                diff_matches_target += 1

        target_counter[(same_note, same_target)] += 1
        matches.append(
            {
                "base_id": sid,
                "assessor": ass,
                "injector": inj,
                "same_note": same_note,
                "same_target": same_target,
                "diff_ids": diff_ids,
            }
        )

    total = len(matches)
    print(f"Assessor rows loaded : {len(assessor_rows)}")
    print(f"Matched injector rows: {total}")
    if total == 0:
        return

    print(f"Same note body       : {note_equal}/{total} ({100*note_equal/total:.1f}%)")
    print(f"Same target sentence : {target_equal}/{total} ({100*target_equal/total:.1f}%)")
    print(f"One sentence differs : {single_sentence_diff}/{total} ({100*single_sentence_diff/total:.1f}%)")
    print(f"Diff == assessor sid : {diff_matches_target}/{total} ({100*diff_matches_target/total:.1f}%)")
    print("Pair breakdown       :")
    for key, count in sorted(target_counter.items()):
        same_note, same_target = key
        print(f"  same_note={same_note:<5} same_target={same_target:<5} n={count}")

    print("\nExamples")
    shown = 0
    for item in matches:
        if shown >= args.show:
            break
        ass = item["assessor"]
        inj = item["injector"]
        print("=" * 80)
        print(f"Base ID          : {item['base_id']}")
        print(f"Same note        : {item['same_note']}")
        print(f"Same target      : {item['same_target']}")
        print(f"Differing sids   : {item['diff_ids'][:8]}")
        print(f"Assessor label   : {ass.get('label')}")
        print(f"Injector label   : {inj.get('label')}")
        print(f"Assessor system  : {short(ass.get('system_prompt', ''))}")
        print(f"Injector system  : {short(inj.get('system_prompt', ''))}")
        print(f"Assessor reason  : {short(ass.get('reasoning', ''))}")
        print(f"Injector reason  : {short(inj.get('reasoning', ''))}")
        print(f"Assessor note L1 : {extract_note_block(ass.get('user_prompt', '')).splitlines()[0]}")
        print(f"Injector note L1 : {extract_note_block(inj.get('user_prompt', '')).splitlines()[0]}")
        shown += 1


if __name__ == "__main__":
    main()
