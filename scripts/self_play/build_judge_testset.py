#!/usr/bin/env python3
"""Build a judge benchmark set from HELD-OUT MEDEC test data.

The MedRECT-32B judge candidate was trained on MEDEC ms-train + ms-val, so the
judge must be benchmarked on ms-test (and uw-test) to avoid leakage. This script
turns the real ms-test errors into benchmark cases in the exact schema consumed
by benchmark_simple_judge.py (`user_prompt`, `changed_sid`, `error_sentence_id`,
`modified_text`, `label`).

Error cases (expected CHANGED):
  - original note  = the corrected (clean) note  -> original_sentence = corrected_sentence
  - modified note  = the note with the real error -> MedRECT should flag error_sentence_id
  - These are genuine, subtle, held-out medical errors: the exact binding
    constraint (the Qwen judge missed ~27% of injected errors).

Benign cases: per the agreed plan, start with the existing training benign
chains (pass that file as --benign-file to the benchmark). Optionally build a
clean held-out precision check from the test correct notes with --with-benign,
which feeds genuinely-correct ms-test notes unchanged (identity edit, expected
SAME) to measure each judge's false-positive rate on unseen clean notes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.self_play.utils import parse_numbered_sentences, reconstruct_note  # noqa: E402

DEFAULT_TEST_JSON = PROJECT_ROOT / "medrect" / "data" / "medec" / "medec-test.json"
DEFAULT_OUT_DIR = PROJECT_ROOT / "data_processed" / "judge_bench"


def build_error_cases(records: list[dict]) -> list[dict]:
    cases = []
    for r in records:
        if r.get("error_flag") != 1:
            continue
        esid = r.get("error_sentence_id")
        err_sentence = (r.get("error_sentence") or "").strip()
        corr_sentence = (r.get("corrected_sentence") or "").strip()
        incorrect_note = r.get("sentences") or ""
        if esid is None or not err_sentence or not corr_sentence or not incorrect_note:
            continue
        esid = int(esid)
        # Numbered CLEAN note: replace the error sentence with its correction.
        correct_note = reconstruct_note(incorrect_note, esid, corr_sentence)
        # Sanity: the clean note must actually contain the corrected sentence at esid.
        if parse_numbered_sentences(correct_note).get(esid, "").strip() != corr_sentence:
            continue
        cases.append(
            {
                "sample_id": r.get("sample_id", ""),
                "user_prompt": correct_note,
                "changed_sid": esid,
                "error_sentence_id": esid,
                "modified_text": err_sentence,
                "error_type": r.get("error_type", ""),
                "label": f"{esid}. {err_sentence}",
            }
        )
    return cases


def build_benign_identity_cases(records: list[dict]) -> list[dict]:
    """Held-out precision check: feed genuinely-correct test notes unchanged.

    Identity 'edit' (modified == original) is trivially meaning-preserving, so the
    expected verdict is SAME. For the detection judge this measures hallucinated
    errors on unseen clean notes; for the pair judge it is a sanity floor.
    """
    cases = []
    for r in records:
        if r.get("error_flag") != 0:
            continue
        note = r.get("sentences") or ""
        sents = parse_numbered_sentences(note)
        if not sents:
            continue
        # Pick a deterministic mid-note content sentence (avoid sentence 1 = demographics).
        sids = sorted(sents)
        sid = sids[len(sids) // 2]
        sentence = sents[sid].strip()
        if not sentence:
            continue
        cases.append(
            {
                "sample_id": r.get("sample_id", ""),
                "user_prompt": note,
                "changed_sid": sid,
                "error_sentence_id": None,
                "modified_text": sentence,  # identity edit
                "error_type": "benign_identity",
                "label": f"{sid}. {sentence}",
            }
        )
    return cases


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--test-json", type=Path, default=DEFAULT_TEST_JSON)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument(
        "--with-benign",
        action="store_true",
        help="Also build a held-out benign (identity) file from test correct notes.",
    )
    args = ap.parse_args()

    records = json.load(open(args.test_json, encoding="utf-8"))
    error_cases = build_error_cases(records)
    err_path = args.out_dir / "test_error_chains.jsonl"
    write_jsonl(err_path, error_cases)
    print(f"Wrote {len(error_cases)} error cases -> {err_path}")

    if args.with_benign:
        benign_cases = build_benign_identity_cases(records)
        ben_path = args.out_dir / "test_benign_identity.jsonl"
        write_jsonl(ben_path, benign_cases)
        print(f"Wrote {len(benign_cases)} benign-identity cases -> {ben_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
