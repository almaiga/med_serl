#!/usr/bin/env python3
"""Direct-prompt a judge/detector model on test data. Model + prompt + data.

It sends each test note to the running server using the given prompt, prints the
raw model output, and scores it against the ground truth in the data. Nothing
else. Swap --prompt to test a different prompt; swap --data to test other notes.

The served model is auto-detected from /v1/models, so you never pass a name.

Data format (medec-test.json): each row has
  sentences (numbered note), error_flag (1/0), error_sentence_id, error_sentence.

Examples:
  # MedRECT detection prompt on held-out ms-test:
  python3 scripts/self_play/run_detection_prompt.py \
    --prompt configs/prompts/detection_localization_prompts.json --n 100

  # Try a different prompt file:
  python3 scripts/self_play/run_detection_prompt.py \
    --prompt configs/prompts/my_other_prompt.json --n 100
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.self_play.utils import parse_assessor_answer  # noqa: E402

DEFAULT_PROMPT = PROJECT_ROOT / "configs" / "prompts" / "detection_localization_prompts.json"
DEFAULT_DATA = PROJECT_ROOT / "medrect" / "data" / "medec" / "medec-test.json"


def detect_model(base_url: str, timeout: float) -> str:
    with urllib.request.urlopen(base_url.rstrip("/") + "/v1/models", timeout=timeout) as r:
        return json.loads(r.read())["data"][0]["id"]


def ask(base_url: str, payload: dict, timeout: float) -> str:
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["choices"][0]["message"]["content"].strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-url", default="http://127.0.0.1:8002")
    ap.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--n", type=int, default=100, help="cases to run (balanced if possible)")
    ap.add_argument("--show", type=int, default=10, help="raw outputs to print")
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args()

    cfg = json.load(open(args.prompt, encoding="utf-8"))
    system_prompt = cfg["system_prompt"]
    user_template = cfg["user_template"]
    rows = json.load(open(args.data, encoding="utf-8"))

    # Balanced sample: half error, half correct (up to --n).
    errors = [r for r in rows if r.get("error_flag") == 1]
    corrects = [r for r in rows if r.get("error_flag") == 0]
    half = max(1, args.n // 2)
    sample = errors[:half] + corrects[:half]

    model = detect_model(args.base_url, args.timeout)
    print(f"Server : {args.base_url}")
    print(f"Model  : {model}  (auto-detected)")
    print(f"Prompt : {args.prompt}")
    print(f"Data   : {args.data}  ({len(sample)} cases: "
          f"{min(half, len(errors))} error / {min(half, len(corrects))} correct)\n")

    # tallies
    err_flagged = err_total = 0      # recall: errors correctly flagged as error
    err_localized = 0                # of flagged errors, correct sentence id
    cor_correct = cor_total = 0      # specificity: correct notes said CORRECT
    failures = 0
    shown = 0

    for r in sample:
        user = user_template.format(sentences=r["sentences"])
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
            "temperature": 0.0,
            "max_tokens": args.max_tokens,
        }
        try:
            raw = ask(args.base_url, payload, args.timeout)
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"[{r.get('sample_id')}] REQUEST FAILED: {exc}")
            continue

        label, sid = parse_assessor_answer(raw)
        gt_error = r.get("error_flag") == 1
        gt_sid = r.get("error_sentence_id")

        if gt_error:
            err_total += 1
            if label == "ERROR":
                err_flagged += 1
                if sid is not None and gt_sid is not None and int(sid) == int(gt_sid):
                    err_localized += 1
        else:
            cor_total += 1
            if label == "CORRECT":
                cor_correct += 1

        if shown < args.show:
            shown += 1
            gt = f"ERROR@{gt_sid}" if gt_error else "CORRECT"
            pred = f"{label}@{sid}" if sid is not None else label
            print(f"--- {r.get('sample_id')}  gt={gt}  pred={pred}  RAW={raw[:60]!r}")

    print("\n" + "=" * 56)
    if err_total:
        print(f"error recall (flagged as error) : {err_flagged}/{err_total} = "
              f"{err_flagged / err_total:.1%}")
        print(f"  of those, correct sentence    : {err_localized}/{err_flagged} = "
              f"{(err_localized / err_flagged) if err_flagged else 0:.1%}")
    if cor_total:
        print(f"correct-note specificity        : {cor_correct}/{cor_total} = "
              f"{cor_correct / cor_total:.1%}")
    if failures:
        print(f"request failures                : {failures}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
