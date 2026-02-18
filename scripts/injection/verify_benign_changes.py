"""
LLM sanity check for benign changes in benign_train.jsonl.
Verifies each modified note still makes clinical sense.

Usage:
    python scripts/injection/verify_benign_changes.py --limit 20
    python scripts/injection/verify_benign_changes.py --all
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
from openai import OpenAI
from tqdm import tqdm

INPUT_FILE = Path("data_processed/benign_changes/benign_train.jsonl")
OUTPUT_DIR = Path("data_processed/benign_changes")

SYSTEM_PROMPT = """You are a clinical reviewer. A benign change was applied to a medical note sentence.
Check if the modified sentence still makes clinical sense and preserves the original meaning.

Respond with EXACTLY two lines:
verdict: PASS
reason: one sentence

PASS = change is safe, sentence reads naturally and preserves clinical meaning
FAIL = change causes confusion, alters clinical meaning, or reads unnaturally"""

USER_TEMPLATE = """Change type: {change_type}
Original term: "{original_term}"
Replacement term: "{replacement_term}"

ORIGINAL: "{original_sentence}"
MODIFIED: "{modified_sentence}"

Is this clinically safe?"""


def verify_one(client, record):
    user_prompt = USER_TEMPLATE.format(
        change_type=record.get("change_type", ""),
        original_term=record.get("original_term", ""),
        replacement_term=record.get("replacement_term", ""),
        original_sentence=record.get("original_sentence", ""),
        modified_sentence=record.get("modified_sentence", ""),
    )
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,
            max_tokens=80,
        )
        reply = response.choices[0].message.content.strip()

        verdict = "UNKNOWN"
        reason = reply
        for line in reply.splitlines():
            if line.lower().startswith("verdict:"):
                verdict = "PASS" if "PASS" in line.upper() else "FAIL"
            elif line.lower().startswith("reason:"):
                reason = line.split(":", 1)[1].strip()

        return {**record, "llm_verdict": verdict, "llm_reason": reason}

    except Exception as e:
        return {**record, "llm_verdict": "ERROR", "llm_reason": str(e)}


def main(limit=None):
    client = OpenAI()

    records = []
    with open(INPUT_FILE) as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    if limit:
        records = records[:limit]

    print(f"Verifying {len(records)} benign changes...")

    results = []
    for r in tqdm(records, desc="Verifying"):
        results.append(verify_one(client, r))

    counts = {"PASS": 0, "FAIL": 0, "ERROR": 0, "UNKNOWN": 0}
    for r in results:
        counts[r.get("llm_verdict", "UNKNOWN")] += 1

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_all   = OUTPUT_DIR / f"verified_{timestamp}.jsonl"
    out_fails = OUTPUT_DIR / f"failed_checks_{timestamp}.jsonl"

    with open(out_all, "w") as f_all, open(out_fails, "w") as f_fail:
        for r in results:
            f_all.write(json.dumps(r) + "\n")
            if r.get("llm_verdict") in ("FAIL", "ERROR"):
                f_fail.write(json.dumps(r) + "\n")

    total = len(results)
    print(f"\n{'='*50}")
    print(f"PASS:    {counts['PASS']}/{total} ({100*counts['PASS']/total:.1f}%)")
    print(f"FAIL:    {counts['FAIL']}/{total} ({100*counts['FAIL']/total:.1f}%)")
    print(f"ERROR:   {counts['ERROR']}/{total}")
    print(f"UNKNOWN: {counts['UNKNOWN']}/{total}")
    print(f"{'='*50}")
    print(f"Results:  {out_all}")
    print(f"Failures: {out_fails}")

    # Show sample failures
    fails = [r for r in results if r.get("llm_verdict") == "FAIL"]
    if fails:
        print(f"\nSample failures ({len(fails)} total):")
        for r in fails[:5]:
            print(f"  [{r.get('change_type')}] \"{r.get('original_term')}\" → \"{r.get('replacement_term')}\"")
            print(f"  Reason: {r.get('llm_reason')}\n")

    # Breakdown by change type
    print("Breakdown by change type:")
    by_type = {}
    for r in results:
        ct = r.get("change_type", "unknown")
        if ct not in by_type:
            by_type[ct] = {"PASS": 0, "FAIL": 0, "ERROR": 0}
        verdict = r.get("llm_verdict", "ERROR")
        by_type[ct][verdict] = by_type[ct].get(verdict, 0) + 1
    for ct, c in sorted(by_type.items()):
        total_ct = sum(c.values())
        pass_pct = 100 * c["PASS"] / total_ct if total_ct > 0 else 0
        print(f"  {ct:30s} PASS: {c['PASS']:4d}/{total_ct} ({pass_pct:.1f}%)  FAIL: {c['FAIL']:3d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify benign changes using GPT-4o-mini")
    parser.add_argument("--limit", type=int, default=None, help="Number of records to verify (default: 20)")
    parser.add_argument("--all", action="store_true", help="Process all records")
    args = parser.parse_args()
    main(limit=None if args.all else (args.limit or 20))
