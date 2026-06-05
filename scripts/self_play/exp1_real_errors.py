#!/usr/bin/env python3
"""Exp 1: run the chosen judge on the real held-out MEDEC ms-test errors.

Exp 2 used hand-crafted analog swaps on CORRECT test notes to test bucket-A
discrimination. Exp 1 uses the 311 real MEDEC ms-test records with error_flag=1
— these are the genuine subtle errors written by clinicians, the held-out
distribution that MedRECT-32B has not seen during fine-tuning.

For each error record we hand the judge:
  - the original (correct) sentence text  (probe.original_sentence)
  - the modified (error) sentence text    (probe.modified_sentence)
  - the modified note (the incorrect note)  (probe.modified_note)
  - the changed sentence id                (probe.changed_sid)

Expected verdict is CHANGED. We measure:
  - recall  = fraction the judge correctly rules CHANGED (the binding constraint)
  - ABSTAIN = the judge flagged a different sentence (off-target localisation)
  - SAME    = the judge missed the error  (the v5 failure mode, was 27 % for Qwen3-8B)

Default config is the Exp 2 winner: medrect_hint_v2 on MedRECT-32B.

Examples:
  python3 scripts/self_play/exp1_real_errors.py
  python3 scripts/self_play/exp1_real_errors.py --style medrect_hint  --model pfnet/Preferred-MedRECT-32B
  python3 scripts/self_play/exp1_real_errors.py --style qwen_pair     --model Qwen/Qwen3-8B
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.self_play.utils import reconstruct_note  # noqa: E402
from scripts.self_play.exp2_run_probes import (  # noqa: E402
    PROMPT_BUILDERS, PARSERS, STYLE_DEFAULTS, model_tag, render,
)

DEFAULT_DATA = PROJECT_ROOT / "data_processed" / "judge_bench" / "medec_test.json"


def build_probes_from_real_errors(rows: list[dict]) -> list[dict]:
    """Convert each error_flag=1 record into a probe-style dict for the judge."""
    out = []
    for r in rows:
        if r.get("error_flag") != 1:
            continue
        esid = r.get("error_sentence_id")
        err = (r.get("error_sentence") or "").strip()
        corr = (r.get("corrected_sentence") or "").strip()
        if esid is None or not err or not corr:
            continue
        modified_note = r["sentences"]  # note with the error in place
        original_note = reconstruct_note(modified_note, int(esid), corr)
        out.append({
            "probe_id": r.get("sample_id", ""),
            "bucket": "REAL",
            "bucket_name": "medec_real_error",
            "note_id": r.get("sample_id", ""),
            "changed_sid": int(esid),
            "original_sentence": corr,
            "modified_sentence": err,
            "modified_note": modified_note,
            "original_note": original_note,
            "expected": "CHANGED",
            "rule": r.get("error_type", "real"),
            "error_type": r.get("error_type", ""),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--style", choices=list(STYLE_DEFAULTS), default="medrect_hint_v2")
    ap.add_argument("--model", default=None,
                    help="HF id or path. Defaults to the style's canonical model.")
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--n", type=int, default=0,
                    help="cap on number of errors to evaluate (0 = all)")
    ap.add_argument("--show", type=int, default=10,
                    help="how many MISSED errors to print at the end")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--thinking", action="store_true",
                    help="enable thinking; default OFF for a judge")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    model = args.model or STYLE_DEFAULTS[args.style]
    out_path = args.out or (
        PROJECT_ROOT / "results" / "exp1" /
        f"{args.style}__{model_tag(model)}.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = json.load(open(args.data, encoding="utf-8"))
    probes = build_probes_from_real_errors(rows)
    if args.n > 0:
        probes = probes[:args.n]

    print(f"Style    : {args.style}")
    print(f"Model    : {model}")
    print(f"Data     : {args.data}")
    print(f"Probes   : {len(probes)} held-out MEDEC ms-test errors")
    print(f"Thinking : {args.thinking}\n")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    llm = LLM(
        model=model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, seed=0,
    )

    builder = PROMPT_BUILDERS[args.style]
    parser = PARSERS[args.style]

    prompts = [render(tokenizer, builder(p), args.thinking) for p in probes]
    print(f"Running inference on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling)

    verdict_counts: dict[str, int] = defaultdict(int)
    by_error_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])  # caught, total
    results = []
    missed_examples: list[dict] = []

    for probe, out in zip(probes, outputs):
        raw = out.outputs[0].text.strip()
        verdict, reason = parser(raw, probe)
        verdict_counts[verdict] += 1
        is_caught = verdict == "CHANGED"
        et = probe["error_type"] or "unknown"
        by_error_type[et][1] += 1
        if is_caught:
            by_error_type[et][0] += 1
        row = {
            "probe_id": probe["probe_id"],
            "error_type": et,
            "expected": "CHANGED",
            "verdict": verdict,
            "caught_error": is_caught,
            "original_sentence": probe["original_sentence"],
            "modified_sentence": probe["modified_sentence"],
            "changed_sid": probe["changed_sid"],
            "raw": raw[:400],
            "reason": reason,
        }
        results.append(row)
        if not is_caught and len(missed_examples) < args.show:
            missed_examples.append(row)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(probes)
    n_caught = verdict_counts.get("CHANGED", 0)
    n_same = verdict_counts.get("SAME", 0)
    n_abstain = verdict_counts.get("ABSTAIN", 0)

    print("\n" + "=" * 72)
    print(f"Results: {out_path}\n")
    print(f"  total real held-out errors        : {n}")
    print(f"  judge ruled CHANGED  (caught)     : {n_caught:>3d} / {n}   "
          f"= {n_caught/n:.1%}   <- recall on real errors")
    print(f"  judge ruled SAME     (missed)     : {n_same:>3d} / {n}   "
          f"= {n_same/n:.1%}   <- v5 failure mode (Qwen3-8B: ~27 %)")
    print(f"  judge ruled ABSTAIN  (off-target) : {n_abstain:>3d} / {n}   "
          f"= {n_abstain/n:.1%}   <- flagged a different sentence")
    print()
    print("  by error type:")
    for et, (c, t) in sorted(by_error_type.items(),
                              key=lambda x: -x[1][1]):
        print(f"    {et:24s}  {c:>3d}/{t:<3d} = {c/t:.0%}")

    if missed_examples:
        print()
        print(f"--- {len(missed_examples)} missed examples ---")
        for r in missed_examples:
            print(f"  [{r['verdict']:7s}] {r['probe_id']}  type={r['error_type']}")
            print(f"      orig: {r['original_sentence'][:90]!r}")
            print(f"      err : {r['modified_sentence'][:90]!r}")
            print(f"      raw : {r['raw'][:120]!r}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
