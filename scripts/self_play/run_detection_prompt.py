#!/usr/bin/env python3
"""Direct-prompt a model on test data. Model + prompt + data. No server.

Loads the chosen model in-process with vLLM, sends each test note using the
given prompt, prints the raw output, and scores it against the ground truth.
Swap --model to pick the judge (MedRECT-32B or Qwen3-8B); swap --prompt to test
a different prompt; swap --data for other notes.

Data format (medec-test.json): each row has
  sentences (numbered note), error_flag (1/0), error_sentence_id.

Examples:
  # MedRECT-32B, number-only detection prompt:
  python3 scripts/self_play/run_detection_prompt.py \
    --model pfnet/Preferred-MedRECT-32B \
    --prompt configs/prompts/detection_localization_prompts.json --n 200

  # MedRECT-32B, correction prompt:
  python3 scripts/self_play/run_detection_prompt.py \
    --model pfnet/Preferred-MedRECT-32B \
    --prompt configs/prompts/detection_localization_correction_prompts.json --n 200

  # Qwen3-8B, same prompt:
  python3 scripts/self_play/run_detection_prompt.py \
    --model Qwen/Qwen3-8B \
    --prompt configs/prompts/detection_localization_prompts.json --n 200
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.self_play.utils import parse_assessor_answer, strip_thinking  # noqa: E402

DEFAULT_PROMPT = PROJECT_ROOT / "configs" / "prompts" / "detection_localization_prompts.json"
DEFAULT_DATA = PROJECT_ROOT / "data_processed" / "judge_bench" / "medec_test.json"


def render_prompt(tokenizer, messages, thinking: bool) -> str:
    """Apply chat template; tolerate models whose template lacks enable_thinking."""
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="pfnet/Preferred-MedRECT-32B",
                    help="HF model id or path (e.g. Qwen/Qwen3-8B).")
    ap.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--n", type=int, default=200, help="cases to run (balanced if possible)")
    ap.add_argument("--show", type=int, default=10, help="raw outputs to print")
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--thinking", action="store_true", help="enable model thinking (default off)")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
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

    print(f"Model  : {args.model}")
    print(f"Prompt : {args.prompt}")
    print(f"Data   : {args.data}  ({len(sample)} cases: "
          f"{min(half, len(errors))} error / {min(half, len(corrects))} correct)")
    print(f"Thinking: {args.thinking}\n")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    sampling = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens)

    prompts = []
    for r in sample:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_template.format(sentences=r["sentences"])},
        ]
        prompts.append(render_prompt(tokenizer, messages, args.thinking))

    print(f"Running inference on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling)

    err_flagged = err_total = 0
    err_localized = 0
    cor_correct = cor_total = 0
    shown = 0

    for r, out in zip(sample, outputs):
        raw = out.outputs[0].text.strip()
        _, content = strip_thinking(raw)
        label, sid = parse_assessor_answer(content or raw)
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
            print(f"--- {r.get('sample_id')}  gt={gt}  pred={pred}  RAW={raw[:80]!r}")

    print("\n" + "=" * 56)
    print(f"Model  : {args.model}")
    print(f"Prompt : {args.prompt.name}")
    if err_total:
        print(f"error recall (flagged as error) : {err_flagged}/{err_total} = "
              f"{err_flagged / err_total:.1%}")
        print(f"  of those, correct sentence    : {err_localized}/{err_flagged} = "
              f"{(err_localized / err_flagged) if err_flagged else 0:.1%}")
    if cor_total:
        print(f"correct-note specificity        : {cor_correct}/{cor_total} = "
              f"{cor_correct / cor_total:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
