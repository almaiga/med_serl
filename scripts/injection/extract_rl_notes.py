#!/usr/bin/env python3
"""Generate structured medical entity extractions for rl_train.jsonl notes.

Mirrors the format of data_processed/parsed_medical_note/extractions.jsonl,
which was built for sft_train.jsonl. Output is appended to a separate file
so process_all_benign_changes.py can consume it via --data-path.

Usage
-----
    cd /workspace/med_serl
    python scripts/injection/extract_rl_notes.py                   # fresh run (concurrency=40)
    python scripts/injection/extract_rl_notes.py --resume          # skip already done
    python scripts/injection/extract_rl_notes.py --limit 10        # quick test
    python scripts/injection/extract_rl_notes.py --concurrency 60  # more parallel calls
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from openai import AsyncOpenAI
from tqdm.asyncio import tqdm as atqdm

EXTRACTION_SCHEMA = """{
  "demographics": {
    "age": <int or null>,
    "gender": <"M"/"F"/"Other" or null>,
    "race_ethnicity": <string or null>,
    "occupation": <string or null>,
    "bmi_data": {"height": null, "weight": null, "bmi": null}
  },
  "history": {
    "chief_complaint": <string or null>,
    "history_of_present_illness": <string or null>,
    "past_medical_history": [<string>, ...] or null,
    "past_surgical_history": <string or null>,
    "social_history": <string or null>,
    "family_history": <string or null>,
    "medications_prior_to_admission": <string or null>,
    "allergies": <string or null>
  },
  "examination": {
    "vital_signs": {
      "temperature": <string or null>,
      "heart_rate": <string or null>,
      "blood_pressure": <string or null>,
      "respiratory_rate": <string or null>,
      "oxygen_saturation": <string or null>
    },
    "physical_exam_findings": <string or null>
  },
  "clinical_data": {
    "lab_results": <string or null>,
    "imaging_results": <string or null>,
    "microbiology": <string or null>
  },
  "course_and_outcome": {
    "hospital_course_summary": <string or null>,
    "procedures_performed": <string or null>,
    "diagnosis_primary": <string or null>,
    "plan_and_treatment": <string or null>
  }
}"""

SYSTEM_PROMPT = (
    "You are a clinical NLP system. Extract structured medical information from "
    "clinical notes. Return ONLY valid JSON matching the schema exactly. "
    "Use null for missing fields. Do not add extra keys."
)

USER_TEMPLATE = (
    "Extract structured information from this clinical note.\n\n"
    "SCHEMA:\n{schema}\n\n"
    "CLINICAL NOTE:\n{note}\n\n"
    "Return ONLY the JSON object, no explanation."
)


def load_done(output_path: Path) -> set:
    done = set()
    if output_path.exists():
        with output_path.open() as f:
            for line in f:
                try:
                    done.add(json.loads(line)["note_id"])
                except Exception:
                    pass
    return done


async def extract_one(
    client: AsyncOpenAI,
    rec: dict,
    model: str,
    max_retries: int,
    semaphore: asyncio.Semaphore,
    out_f,
    lock: asyncio.Lock,
    counters: dict,
    pbar,
) -> None:
    prompt = USER_TEMPLATE.format(schema=EXTRACTION_SCHEMA, note=rec["correct_note"][:6000])
    extraction = None
    error_msg = None

    async with semaphore:
        for attempt in range(max_retries):
            try:
                resp = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=1024,
                    response_format={"type": "json_object"},
                )
                extraction = json.loads(resp.choices[0].message.content)
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(2 ** attempt)
                else:
                    error_msg = str(e)

    row = {
        "note_id":            rec["note_id"],
        "error_type":         rec.get("error_type", ""),
        "corrected_sentence": rec.get("corrected_sentence", ""),
        "correct_note":       rec["correct_note"],
        "extraction":         json.dumps(extraction) if extraction else "{}",
        "extraction_error":   error_msg,
        "timestamp":          datetime.utcnow().isoformat(),
    }

    async with lock:
        out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        out_f.flush()
        if error_msg:
            counters["err"] += 1
        else:
            counters["ok"] += 1

    pbar.update(1)


async def run(args):
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        sys.exit("ERROR: OPENAI_API_KEY not set.")
    client = AsyncOpenAI(api_key=api_key)

    src  = Path(args.input)
    dest = Path(args.output)
    dest.parent.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    if args.limit:
        records = records[: args.limit]

    done    = load_done(dest) if args.resume else set()
    pending = [r for r in records if r["note_id"] not in done]

    print(f"Input       : {src}  ({len(records)} records)")
    print(f"Output      : {dest}")
    print(f"Model       : {args.model}")
    print(f"Concurrency : {args.concurrency}")
    print(f"Done        : {len(done)}  |  Pending: {len(pending)}")
    print()

    if not pending:
        print("Nothing to do.")
        return

    semaphore = asyncio.Semaphore(args.concurrency)
    lock      = asyncio.Lock()
    counters  = {"ok": 0, "err": 0}

    with dest.open("a") as out_f:
        with atqdm(total=len(pending), desc="Extracting") as pbar:
            tasks = [
                extract_one(client, rec, args.model, args.max_retries,
                            semaphore, out_f, lock, counters, pbar)
                for rec in pending
            ]
            await asyncio.gather(*tasks)

    print(f"\nDone. OK={counters['ok']}  Errors={counters['err']}")
    print(f"Output: {dest}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       default="data_processed/medec_paired/train_val_split/rl_train.jsonl")
    p.add_argument("--output",      default="data_processed/parsed_medical_note/rl_extractions.jsonl")
    p.add_argument("--model",       default="gpt-4o-mini")
    p.add_argument("--concurrency", type=int, default=40)
    p.add_argument("--limit",       type=int, default=None)
    p.add_argument("--resume",      action="store_true")
    p.add_argument("--max-retries", type=int, default=3)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
