#!/usr/bin/env python3
"""Run ONE judge configuration on the Exp 2 adversarial probe set.

Single GPU. Loads the chosen model in-process with vLLM and evaluates per
bucket. Outputs:
  - results JSONL (one row per probe, raw output + parsed verdict + correct?)
  - per-bucket accuracy summary to stdout.

Three styles, each with its own model + prompt + parser:

  --style qwen_pair        Qwen-style sentence-pair JSON judge
                           (simple_judge_prompts.json with few-shot, output JSON
                           {verdict, score, reason}).  Default --model: Qwen/Qwen3-8B.

  --style medrect_native   MedRECT-style detector on the MODIFIED note only.
                           Verdict: CORRECT -> SAME; flags changed_sid -> CHANGED;
                           flags any other sid -> ABSTAIN (ambiguous).
                           Default --model: pfnet/Preferred-MedRECT-32B.

  --style medrect_hint     MedRECT-style detector on the modified note + a focus
                           block stating which sentence was edited and what the
                           original was.  This leverages MedRECT's medical
                           knowledge on a strictly easier task than its native
                           training task: we hand it the candidate edit instead
                           of asking it to find one cold.  Same parser as native.
                           Default --model: pfnet/Preferred-MedRECT-32B.

Examples:
  python3 scripts/self_play/exp2_run_probes.py --style medrect_native --n 30
  python3 scripts/self_play/exp2_run_probes.py --style medrect_hint   --n 30
  python3 scripts/self_play/exp2_run_probes.py --style qwen_pair      --n 30
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.self_play.utils import (  # noqa: E402
    normalized_edit_distance,
    parse_assessor_answer,
    strip_thinking,
)

DEFAULT_PROBES = PROJECT_ROOT / "data_processed" / "judge_bench" / "exp2_probes.jsonl"
QWEN_PROMPT_PATH = PROJECT_ROOT / "configs" / "prompts" / "simple_judge_prompts.json"
MEDRECT_PROMPT_PATH = PROJECT_ROOT / "configs" / "prompts" / "detection_localization_prompts.json"

STYLE_DEFAULTS = {
    "qwen_pair":      "Qwen/Qwen3-8B",
    "medrect_native": "pfnet/Preferred-MedRECT-32B",
    "medrect_hint":   "pfnet/Preferred-MedRECT-32B",
}


# ─────────────────────────────────────────────────────────────────────────────
# Prompt builders
# ─────────────────────────────────────────────────────────────────────────────

def build_qwen_pair_messages(probe: dict) -> list[dict]:
    """Sentence-pair JSON judge with the project's existing few-shot."""
    cfg = json.load(open(QWEN_PROMPT_PATH, encoding="utf-8"))
    messages: list[dict] = [{"role": "system", "content": cfg["system_prompt"]}]
    for ex in cfg.get("few_shot_examples", []):
        norm = normalized_edit_distance(ex["original_sentence"], ex["modified_sentence"])
        messages.append({
            "role": "user",
            "content": cfg["user_template"].format(
                note_id=ex.get("note_id", "example"),
                error_type=ex.get("error_type", ""),
                changed_sid=ex.get("changed_sid", "?"),
                original_sentence=ex["original_sentence"],
                modified_sentence=ex["modified_sentence"],
                norm_edit=f"{norm:.2f}",
            ),
        })
        messages.append({
            "role": "assistant",
            "content": json.dumps({
                "verdict": ex["verdict"],
                "score": ex["score"],
                "reason": ex["reason"],
            }),
        })
    norm = normalized_edit_distance(probe["original_sentence"], probe["modified_sentence"])
    user_template = cfg.get("user_template_with_context", cfg["user_template"])
    user_kwargs = dict(
        note_id=probe["note_id"],
        error_type=probe["bucket_name"],
        changed_sid=probe["changed_sid"],
        original_sentence=probe["original_sentence"][:2000],
        modified_sentence=probe["modified_sentence"][:2000],
        norm_edit=f"{norm:.2f}",
    )
    if "{modified_note}" in user_template:
        user_kwargs["modified_note"] = probe["modified_note"][:3000]
    messages.append({"role": "user", "content": user_template.format(**user_kwargs)})
    return messages


def build_medrect_native_messages(probe: dict) -> list[dict]:
    cfg = json.load(open(MEDRECT_PROMPT_PATH, encoding="utf-8"))
    return [
        {"role": "system", "content": cfg["system_prompt"]},
        {"role": "user", "content": cfg["user_template"].format(sentences=probe["modified_note"])},
    ]


def build_medrect_hint_messages(probe: dict) -> list[dict]:
    """Note-only detection + a focus block that names the candidate edit.

    MedRECT was fine-tuned for ``find the single error in a numbered note''.
    Here we give it strictly more information: which sentence was edited and
    what the prior text was.  The output format stays the same (CORRECT or a
    sentence number), so the model stays in distribution while answering an
    easier question than its training task.
    """
    cfg = json.load(open(MEDRECT_PROMPT_PATH, encoding="utf-8"))
    user = (
        f"A reviewer edited sentence {probe['changed_sid']} of the note below.\n"
        f"Original sentence {probe['changed_sid']}: {probe['original_sentence']}\n"
        f"Edited sentence {probe['changed_sid']}: {probe['modified_sentence']}\n\n"
        f"Full note (with the edit applied):\n{probe['modified_note']}"
    )
    return [
        {"role": "system", "content": cfg["system_prompt"]},
        {"role": "user", "content": user},
    ]


PROMPT_BUILDERS = {
    "qwen_pair":      build_qwen_pair_messages,
    "medrect_native": build_medrect_native_messages,
    "medrect_hint":   build_medrect_hint_messages,
}


# ─────────────────────────────────────────────────────────────────────────────
# Parsers
# ─────────────────────────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict | None:
    text = (text or "").strip()
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    obj = json.loads(text[start:i + 1])
                    return obj if isinstance(obj, dict) else None
                except (ValueError, TypeError):
                    return None
    return None


def parse_qwen_pair(raw: str, probe: dict) -> tuple[str, str]:
    """Return (verdict, reason)."""
    _, content = strip_thinking(raw)
    obj = _extract_json(content) or _extract_json(raw)
    if not obj:
        return "ABSTAIN", "no_json"
    verdict = str(obj.get("verdict", "ABSTAIN")).upper()
    if verdict not in {"SAME", "CHANGED", "ABSTAIN"}:
        return "ABSTAIN", f"bad_verdict:{verdict}"
    return verdict, str(obj.get("reason", ""))[:200]


def parse_medrect(raw: str, probe: dict) -> tuple[str, str]:
    """Return (verdict, reason).  Common parser for native and hint styles."""
    answer = raw.rsplit("</think>", 1)[1].strip() if "</think>" in raw else raw.strip()
    label, sid = parse_assessor_answer(answer)
    if label == "UNKNOWN":
        # fallback: scan lines for bare number or CORRECT
        for line in reversed([l.strip() for l in answer.splitlines() if l.strip()]):
            if re.fullmatch(r"\d+", line):
                label, sid = "ERROR", int(line)
                break
            if re.fullmatch(r"CORRECT", line, flags=re.IGNORECASE):
                label, sid = "CORRECT", None
                break
    if label == "CORRECT":
        return "SAME", "detector_says_correct"
    if label == "ERROR":
        if sid is not None and int(sid) == int(probe["changed_sid"]):
            return "CHANGED", f"flagged_sid_{sid}"
        return "ABSTAIN", f"flagged_other_sid_{sid}"
    return "ABSTAIN", f"unparseable:{answer[:60]!r}"


PARSERS = {
    "qwen_pair":      parse_qwen_pair,
    "medrect_native": parse_medrect,
    "medrect_hint":   parse_medrect,
}


# ─────────────────────────────────────────────────────────────────────────────
# Driver
# ─────────────────────────────────────────────────────────────────────────────

def render(tokenizer, messages, thinking: bool) -> str:
    kwargs = dict(tokenize=False, add_generation_prompt=True)
    try:
        return tokenizer.apply_chat_template(messages, enable_thinking=thinking, **kwargs)
    except TypeError:
        return tokenizer.apply_chat_template(messages, **kwargs)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--style", choices=list(STYLE_DEFAULTS), required=True)
    ap.add_argument("--model", default=None,
                    help="HF id or path. Defaults to the style's canonical model.")
    ap.add_argument("--probes", type=Path, default=DEFAULT_PROBES)
    ap.add_argument("--n", type=int, default=30,
                    help="probes per bucket to evaluate (capped by available)")
    ap.add_argument("--show", type=int, default=4,
                    help="raw outputs to print per bucket")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--thinking", action="store_true",
                    help="enable model thinking. Default OFF: a judge should answer directly. "
                         "MedRECT-32B is Qwen3-based and will burn tokens on a <think> block "
                         "if thinking is on, leaving no budget for the actual verdict.")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--out", type=Path, default=None,
                    help="results JSONL path (default: results/exp2/<style>.jsonl)")
    args = ap.parse_args()

    model = args.model or STYLE_DEFAULTS[args.style]
    out_path = args.out or (PROJECT_ROOT / "results" / "exp2" / f"{args.style}.jsonl")
    thinking = not args.no_thinking

    probes_all = [json.loads(l) for l in open(args.probes, encoding="utf-8") if l.strip()]
    probes_by_bucket: dict[str, list[dict]] = defaultdict(list)
    for p in probes_all:
        probes_by_bucket[p["bucket"]].append(p)
    probes: list[dict] = []
    for bucket in sorted(probes_by_bucket):
        probes.extend(probes_by_bucket[bucket][:args.n])

    print(f"Style    : {args.style}")
    print(f"Model    : {model}")
    print(f"Probes   : {args.probes}  ({len(probes)} cases, {args.n}/bucket)")
    print(f"Thinking : {thinking}\n")

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
    sampling = SamplingParams(temperature=0.0, top_p=1.0,
                              max_tokens=args.max_tokens, seed=0)

    builder = PROMPT_BUILDERS[args.style]
    parser = PARSERS[args.style]

    prompts = [render(tokenizer, builder(p), thinking) for p in probes]
    print(f"Running inference on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    correct_by_bucket: dict[str, int] = defaultdict(int)
    total_by_bucket: dict[str, int] = defaultdict(int)
    verdict_by_bucket: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    raw_examples: dict[str, list[dict]] = defaultdict(list)

    for probe, out in zip(probes, outputs):
        raw = out.outputs[0].text.strip()
        verdict, reason = parser(raw, probe)
        ok = verdict == probe["expected"]
        bucket = probe["bucket"]
        total_by_bucket[bucket] += 1
        if ok:
            correct_by_bucket[bucket] += 1
        verdict_by_bucket[bucket][verdict] += 1
        row = {
            "probe_id": probe["probe_id"],
            "bucket": bucket,
            "bucket_name": probe["bucket_name"],
            "expected": probe["expected"],
            "verdict": verdict,
            "correct": ok,
            "rule": probe["rule"],
            "original_sentence": probe["original_sentence"],
            "modified_sentence": probe["modified_sentence"],
            "raw": raw[:600],
            "reason": reason,
        }
        results.append(row)
        if len(raw_examples[bucket]) < args.show:
            raw_examples[bucket].append(row)

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Report
    print("\n" + "=" * 64)
    print(f"Results: {out_path}\n")
    overall_ok = sum(correct_by_bucket.values())
    overall_n = sum(total_by_bucket.values())
    bucket_names = {p["bucket"]: p["bucket_name"] for p in probes}
    for bucket in sorted(total_by_bucket):
        n = total_by_bucket[bucket]
        ok = correct_by_bucket[bucket]
        verdicts = dict(verdict_by_bucket[bucket])
        name = bucket_names.get(bucket, "?")
        print(f"  bucket {bucket} ({name:18s})  {ok:>2d}/{n:<2d} = {ok/n:.0%}   verdicts={verdicts}")
    print(f"\n  OVERALL                       {overall_ok:>2d}/{overall_n:<2d} = {overall_ok/overall_n:.0%}\n")

    # Show a few raw outputs per bucket so failure modes are visible.
    for bucket in sorted(raw_examples):
        print(f"--- bucket {bucket} samples ({bucket_names[bucket]}) ---")
        for r in raw_examples[bucket]:
            tag = "OK" if r["correct"] else "X "
            print(f"  [{tag}] expected={r['expected']:7s} got={r['verdict']:7s} rule={r['rule']}")
            print(f"        orig: {r['original_sentence'][:90]!r}")
            print(f"        mod : {r['modified_sentence'][:90]!r}")
            print(f"        raw : {r['raw'][:120]!r}\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
