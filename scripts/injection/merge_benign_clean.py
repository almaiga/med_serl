"""
Assemble benign_train_clean.jsonl from verified PASSes + fixed entries.

Merges:
  1. PASS entries from verified_*.jsonl
  2. Entries with fix_change_kept=YES from fixed_benign_*.jsonl
     (updates the original record with the fixed replacement_term/sentences)
  3. Optionally prepends an existing clean file (--append-from) to preserve prior records

Usage:
    python scripts/injection/merge_benign_clean.py
    python scripts/injection/merge_benign_clean.py \
        --verified data_processed/benign_changes/verified_20260218_150217.jsonl \
        --fixed data_processed/benign_changes/fixed_benign_20260218_150814.jsonl \
        --append-from data_processed/benign_changes/benign_train_clean.jsonl
"""

import json
import argparse
from pathlib import Path
from collections import Counter

DATA_DIR = Path("data_processed/benign_changes")
OUTPUT = DATA_DIR / "benign_train_clean.jsonl"


def find_latest(prefix: str) -> Path:
    """Find latest file matching prefix_TIMESTAMP.jsonl."""
    matches = sorted(DATA_DIR.glob(f"{prefix}_*.jsonl"), reverse=True)
    if not matches:
        raise FileNotFoundError(f"No {prefix}_*.jsonl files in {DATA_DIR}")
    return matches[0]


def load_jsonl(path: Path):
    records = []
    with open(path) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def main(verified_path=None, fixed_path=None, append_from=None):
    # Auto-detect latest files if not specified
    verified_path = Path(verified_path) if verified_path else find_latest("verified")
    fixed_path = Path(fixed_path) if fixed_path else find_latest("fixed_benign")

    print(f"Verified file: {verified_path}")
    print(f"Fixed file:    {fixed_path}")

    # 1. Collect PASSes from verified
    verified = load_jsonl(verified_path)
    passes = [r for r in verified if r.get("llm_verdict") == "PASS"]
    print(f"\nVerified total: {len(verified)}")
    print(f"  PASS: {len(passes)}")
    print(f"  FAIL: {sum(1 for r in verified if r.get('llm_verdict') == 'FAIL')}")

    # 2. Collect fixed entries where change was kept
    fixed = load_jsonl(fixed_path)
    kept = [r for r in fixed if r.get("fix_change_kept") == "YES"]
    reverted = [r for r in fixed if r.get("fix_change_kept") == "NO"]
    print(f"\nFixed total: {len(fixed)}")
    print(f"  change_kept=YES: {len(kept)}")
    print(f"  change_kept=NO:  {len(reverted)}")

    # 3. Clean fields - remove verification/fix metadata, keep core benign fields
    core_fields = [
        "note_id", "change_type", "original_note", "modified_note",
        "original_sentence", "modified_sentence", "original_term",
        "replacement_term", "sentence_position", "is_benign",
        "change_made", "verification_source", "verified",
        "error_type", "corrected_sentence"
    ]

    clean = []

    # Add PASSes (strip llm_verdict/llm_reason)
    for r in passes:
        entry = {k: r[k] for k in core_fields if k in r}
        entry["source"] = "verified_pass"
        clean.append(entry)

    # Add fixed entries - apply the fix
    for r in kept:
        entry = {k: r[k] for k in core_fields if k in r}
        # If the fix changed the replacement term, update it
        if r.get("fix_replacement_term"):
            entry["replacement_term"] = r["fix_replacement_term"]
        if r.get("fix_modified_sentence"):
            entry["modified_sentence"] = r["fix_modified_sentence"]
        if r.get("fix_modified_note"):
            entry["modified_note"] = r["fix_modified_note"]
        entry["source"] = "fixed_kept"
        clean.append(entry)

    # 3b. Prepend existing clean records if --append-from specified
    if append_from:
        append_path = Path(append_from)
        existing = load_jsonl(append_path)
        print(f"\nAppending from: {append_path} ({len(existing)} existing records)")
        clean = existing + clean

    # 4. Deduplicate by note_id + change_type (shouldn't happen, but safety)
    seen = set()
    deduped = []
    for entry in clean:
        key = (entry.get("note_id", ""), entry.get("change_type", ""))
        if key not in seen:
            seen.add(key)
            deduped.append(entry)

    # 5. Write output
    with open(OUTPUT, "w") as f:
        for entry in deduped:
            f.write(json.dumps(entry) + "\n")

    # Stats
    by_type = Counter(e.get("change_type", "unknown") for e in deduped)
    by_source = Counter(e.get("source", "unknown") for e in deduped)

    print(f"\n{'='*50}")
    print(f"CLEAN DATASET: {len(deduped)} records")
    print(f"Output: {OUTPUT}")
    print(f"{'='*50}")
    print(f"\nBy source:")
    for src, cnt in sorted(by_source.items()):
        print(f"  {src:20s} {cnt}")
    print(f"\nBy change type:")
    for ct, cnt in sorted(by_type.items()):
        print(f"  {ct:30s} {cnt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge verified+fixed benign changes into clean dataset")
    parser.add_argument("--verified", default=None, help="Path to verified_*.jsonl (auto-detects latest)")
    parser.add_argument("--fixed", default=None, help="Path to fixed_benign_*.jsonl (auto-detects latest)")
    parser.add_argument("--append-from", default=None, dest="append_from",
                        help="Existing clean JSONL to preserve (new records are appended, deduped by note_id+change_type)")
    args = parser.parse_args()
    main(args.verified, args.fixed, args.append_from)
