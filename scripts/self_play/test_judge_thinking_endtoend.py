#!/usr/bin/env python3
"""End-to-end judge smoke test — is thinking REALLY on in the production path?

Runs the discriminating probe cases through judge_client's ACTUAL functions
(build_detection_messages + parse_detection_verdict + the real prompt config)
with the fixed config (enable_thinking=True, max_tokens=2048). For each case it
reports:
  - thinking_present : did the model emit a closed <think>...</think> block?
  - tokens / truncated: did it hit max_tokens before finishing? (the v6 failure)
  - verdict + correct : did it produce the right SAME/CHANGED and parse cleanly?

Headline verdict:
  GO   — thinking active, no truncation, verdicts correct (incl. the medium/
         subtle errors the thinking-OFF judge missed). Safe to train.
  STOP — thinking not detected, or truncation, or verdicts wrong.

Runs in-process on one GPU (fine on the A100). ~1-2 min.

Usage:
    JUDGE_PROMPT_STYLE=hint_v2 python3 scripts/self_play/test_judge_thinking_endtoend.py
"""

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.self_play.judge_client import (  # noqa: E402
    build_detection_messages, parse_detection_verdict,
    load_medrect_prompt_config,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe",
                    default="data_processed/synthetic_test/thinking_probe_8.jsonl")
    ap.add_argument("--model", default="pfnet/Preferred-MedRECT-32B")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--style", default=os.environ.get("JUDGE_PROMPT_STYLE", "hint_v2"))
    args = ap.parse_args()

    cfg = load_medrect_prompt_config()
    max_tokens = int(cfg.get("sampling_params", {}).get("max_tokens", 2048))
    print(f"Model        : {args.model}")
    print(f"Prompt style : {args.style}")
    print(f"max_tokens   : {max_tokens}  (from medrect_judge_prompts.json)")
    print(f"Config asserts thinking-on via chat_template_kwargs in judge_client\n")

    cases = [json.loads(l) for l in open(args.probe) if l.strip()]
    print(f"Loaded {len(cases)} probe cases\n")

    # Build the exact messages judge_client would send
    built = []
    for c in cases:
        msgs = build_detection_messages(
            modified_note=c.get("modified_sentences", ""),
            changed_sid=c.get("changed_sid"),
            original_sentence=c.get("original_sentence", ""),
            modified_sentence=c.get("modified_sentence", ""),
            style=args.style,
        )
        built.append((c, msgs))

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    prompts = []
    for _c, msgs in built:
        # enable_thinking=True mirrors chat_template_kwargs in judge_client
        prompts.append(tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=True))

    llm = LLM(model=args.model, dtype="bfloat16",
              tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=8192, trust_remote_code=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    outs = llm.generate(prompts, sampling_params=sampling)

    print("=" * 100)
    print(f"{'note_id':<14} {'expect':>8} {'verdict':>8} {'think?':>7} "
          f"{'tokens':>7} {'trunc?':>7} {'parse':>16}  ok")
    print("=" * 100)

    n_ok = n_think = n_trunc = 0
    for (c, _m), o in zip(built, outs):
        gen = o.outputs[0]
        text = gen.text
        ntok = len(gen.token_ids)
        thinking_present = "</think>" in text
        truncated = (ntok >= max_tokens) and not thinking_present
        r = parse_detection_verdict(text, c.get("changed_sid"))
        verdict = r["verdict"]
        # map expected CHANGED/CORRECT to judge verdicts CHANGED/SAME
        want = "CHANGED" if c["expected_verdict"] == "CHANGED" else "SAME"
        ok = (verdict == want)
        n_ok += ok
        n_think += thinking_present
        n_trunc += truncated
        print(f"{c['note_id']:<14} {want:>8} {verdict:>8} "
              f"{('yes' if thinking_present else 'NO'):>7} {ntok:>7} "
              f"{('YES' if truncated else '-'):>7} {r['status']:>16}  "
              f"{'OK' if ok else 'XX'}")

    n = len(cases)
    print()
    print(f"thinking present : {n_think}/{n}")
    print(f"truncated        : {n_trunc}/{n}")
    print(f"verdicts correct : {n_ok}/{n}")
    print()

    if n_think == n and n_trunc == 0 and n_ok >= n - 1:
        print("  GO — thinking is ON in the production path, no truncation, "
              "verdicts correct (incl. the medium/subtle errors the "
              "thinking-OFF judge missed). Judge is ready for training.")
        sys.exit(0)
    elif n_think < n:
        print("  STOP — thinking NOT detected on some cases. The judge is not "
              "reasoning; check enable_thinking in judge_client.py.")
        sys.exit(1)
    elif n_trunc > 0:
        print("  STOP — truncation detected. max_tokens too small; raise it "
              "in medrect_judge_prompts.json.")
        sys.exit(1)
    else:
        print("  REVIEW — thinking on and no truncation, but some verdicts "
              "wrong. Inspect the failing cases before training.")
        sys.exit(2)


if __name__ == "__main__":
    main()
