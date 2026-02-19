"""
Use GPT-4o-mini to fix failed benign changes.
Reads failed_checks_*.jsonl, asks LLM to correct the modified sentence,
then re-verifies and saves the fixed entries.

Usage:
    python scripts/injection/fix_failed_benign.py \
        --failures data_processed/benign_changes/failed_checks_20260218_142325.jsonl
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm

OUTPUT_DIR = Path("data_processed/benign_changes")

SYSTEM_PROMPT = """You are a clinical editor. A benign change was applied to a medical note sentence but the result is grammatically wrong or uses a bad synonym.

Your task: Fix the modified sentence so that:
1. The replacement term is still used (or a better synonym if it's truly wrong)
2. The sentence is grammatically correct and reads naturally
3. The clinical meaning is preserved

Respond with EXACTLY:
fixed_sentence: <the corrected sentence>
change_kept: YES or NO (did you keep the replacement term, or revert to original?)
note: one sentence explaining what you fixed"""

USER_TEMPLATE = """Change type: {change_type}
Original term: "{original_term}"
Replacement term: "{replacement_term}"

ORIGINAL sentence: "{original_sentence}"
BAD modified sentence: "{modified_sentence}"
Problem: {llm_reason}

Fix the modified sentence:"""


def fix_one(client, record):
    user_prompt = USER_TEMPLATE.format(
        change_type=record.get("change_type", ""),
        original_term=record.get("original_term", ""),
        replacement_term=record.get("replacement_term", ""),
        original_sentence=record.get("original_sentence", ""),
        modified_sentence=record.get("modified_sentence", ""),
        llm_reason=record.get("llm_reason", ""),
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
            max_tokens=150,
        )
        reply = response.choices[0].message.content.strip()

        fixed_sentence = None
        change_kept = "UNKNOWN"
        note = reply

        for line in reply.splitlines():
            if line.lower().startswith("fixed_sentence:"):
                fixed_sentence = line.split(":", 1)[1].strip()
            elif line.lower().startswith("change_kept:"):
                change_kept = "YES" if "YES" in line.upper() else "NO"
            elif line.lower().startswith("note:"):
                note = line.split(":", 1)[1].strip()

        if fixed_sentence:
            # Update modified_sentence and modified_note with the fix
            fixed_record = dict(record)
            old_modified = record.get("modified_sentence", "")
            fixed_record["modified_sentence"] = fixed_sentence
            # Also patch it in the modified_note if present
            if old_modified and record.get("modified_note"):
                fixed_record["modified_note"] = record["modified_note"].replace(
                    old_modified, fixed_sentence, 1
                )
            fixed_record["fix_applied"] = True
            fixed_record["fix_change_kept"] = change_kept
            fixed_record["fix_note"] = note
            fixed_record["llm_verdict"] = "FIXED"
            return fixed_record
        else:
            return {**record, "fix_applied": False, "fix_note": "Could not parse fixed_sentence", "llm_verdict": "FIX_FAILED"}

    except Exception as e:
        return {**record, "fix_applied": False, "fix_note": str(e), "llm_verdict": "ERROR"}


def main(failures_file):
    client = OpenAI()

    records = []
    with open(failures_file) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    # Skip empty logical_restatement entries (nothing to fix)
    fixable = [r for r in records if r.get("original_sentence", "").strip() not in ("", "None")]
    skipped = len(records) - len(fixable)
    print(f"Loaded {len(records)} failures ({skipped} skipped - empty entries)")
    print(f"Fixing {len(fixable)} entries...")

    results = []
    for r in tqdm(fixable, desc="Fixing"):
        results.append(fix_one(client, r))

    counts = {"FIXED": 0, "FIX_FAILED": 0, "ERROR": 0}
    change_kept = 0
    for r in results:
        counts[r.get("llm_verdict", "ERROR")] += 1
        if r.get("fix_change_kept") == "YES":
            change_kept += 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_fixed  = OUTPUT_DIR / f"fixed_benign_{timestamp}.jsonl"
    out_failed = OUTPUT_DIR / f"still_failed_{timestamp}.jsonl"

    with open(out_fixed, "w") as f_fix, open(out_failed, "w") as f_fail:
        for r in results:
            if r.get("fix_applied"):
                f_fix.write(json.dumps(r) + "\n")
            else:
                f_fail.write(json.dumps(r) + "\n")

    total = len(results)
    print(f"\n{'='*50}")
    print(f"FIXED:      {counts['FIXED']}/{total} (replacement kept: {change_kept})")
    print(f"FIX_FAILED: {counts['FIX_FAILED']}/{total}")
    print(f"ERROR:      {counts['ERROR']}/{total}")
    print(f"{'='*50}")
    print(f"Fixed entries: {out_fixed}")
    print(f"Still failed:  {out_failed}")

    # Show samples
    fixed = [r for r in results if r.get("fix_applied")]
    if fixed:
        print(f"\nSample fixes:")
        for r in fixed[:5]:
            print(f"  [{r.get('change_type')}] {r.get('original_term')} → {r.get('replacement_term')}")
            print(f"  Before: {r.get('original_sentence', '')[:100]}")
            print(f"  After:  {r.get('modified_sentence', '')[:100]}")
            print(f"  Note:   {r.get('fix_note', '')}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix failed benign changes using GPT-4o")
    parser.add_argument("--failures", required=True, help="Path to failed_checks_*.jsonl")
    args = parser.parse_args()
    main(args.failures)
