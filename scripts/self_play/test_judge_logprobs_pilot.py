#!/usr/bin/env python3
"""Phase 1 pilot — does the judge's confidence correlate with correctness?

Runs the MedRECT-32B judge IN-PROCESS via vLLM (no external HTTP server) on
the 10 hand-audited v6_run2 games from 2026-07-13, extracting `logprobs` so
we can measure per-token confidence. Prints a table and a heuristic GO /
NO-GO verdict on whether paper 1 (LLM-as-a-Verifier — continuous scoring
via expected logits) is worth implementing.

Single VM, single Python process. No JUDGE_VLLM_URL, no HTTP.

Two-phase execution to save server time on retries:

  1. --dry-run          Loads dataset + tokenizer only. Verifies fields,
                        builds all 10 prompts, prints preview. ~1 min, no GPU.
                        Run this FIRST to catch schema / prompt issues cheaply.

  2. (default)          Everything from --dry-run, THEN loads MedRECT-32B into
                        vLLM (~5 min cold, ~1 min warm) and runs inference.

Usage:
    python3 scripts/self_play/test_judge_logprobs_pilot.py --dry-run
    python3 scripts/self_play/test_judge_logprobs_pilot.py
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

# ── The 10 audit cases from 2026-07-13's manual review ──────────────────────
AUDIT_CASES = [
    ("ms-val-364-error",   "error_injection", "AMBIGUOUS"),
    ("ms-train-880-error", "error_injection", "INJECTOR FAILURE"),
    ("ms-val-563-error",   "error_injection", "INJECTOR FAILURE"),
    ("ms-train-718-error", "error_injection", "INJECTOR FAILURE"),
    ("ms-train-363-error", "error_injection", "REAL JUDGE FAILURE"),
    ("ms-val-121-benign",  "benign",          "INJECTOR FAILURE"),
    ("uw-val-2-benign",    "benign",          "INJECTOR FAILURE"),
    ("uw-val-156-benign",  "benign",          "INJECTOR FAILURE"),
    ("ms-train-1716-benign", "benign",        "JUDGE OVER-FLAG"),
    ("ms-train-945-benign", "benign",         "INJECTOR OVER-EDIT"),
]

# Prompts match configs/prompts/medrect_judge_prompts.json exactly.
SYS_PROMPT_BASE = (
    "You are a medical expert reviewing clinical text for accuracy.\n"
    "The text contains either no error or exactly one medical error.\n"
    "Identify any medical error related to treatment, diagnosis, management,"
    " or causation.\n\n"
    "Output Format:\n"
    "- If no error: 'CORRECT'\n"
    "- If error found: output only the sentence number (e.g. '7')\n\n"
    "CRITICAL: Output ONLY the result. Do NOT include explanations, analysis,"
    " or additional text."
)

HINT_V2_CARVEOUT = (
    "\n\nIMPORTANT: surface-only edits that preserve clinical meaning are NOT"
    " errors. Output CORRECT if the edit is one of:\n"
    "- a medical abbreviation expansion or contraction (e.g. MI <->"
    " myocardial infarction, T2DM <-> type 2 diabetes mellitus)\n"
    "- a brand-name <-> generic-name drug substitution (e.g. Lipitor <->"
    " atorvastatin, Glucophage <-> metformin)\n"
    "- an exact temporal-unit conversion (e.g. 1 week <-> 7 days, 1 year <->"
    " 12 months)\n"
    "- any other stylistic rewording with identical clinical meaning.\n"
    "Only output a sentence number if the edit changes the clinical meaning."
)

USER_HINT_V2 = (
    "A reviewer edited sentence {changed_sid} of the note below.\n"
    "Original sentence {changed_sid}: {original_sentence}\n"
    "Edited sentence {changed_sid}: {modified_sentence}\n\n"
    "Full note (with the edit applied):\n{modified_note}"
)


# ── Robust dataset loading ──────────────────────────────────────────────────

def load_all_games(dataset_id: str, split_hint: str) -> List[Dict]:
    """Fetch the interactions dataset from HF, coping with unknown split names.

    Our upload put files under `v6_run2/*.jsonl`, so the auto-detected split
    might be 'train', 'v6_run2', or something else. Try the hint first, then
    fall back to whatever exists.
    """
    from datasets import load_dataset

    try:
        ds = load_dataset(dataset_id, split=split_hint)
        print(f"  loaded split '{split_hint}': {len(ds)} rows")
    except Exception as first_err:
        try:
            ds_dict = load_dataset(dataset_id)
        except Exception:
            raise first_err
        splits = list(ds_dict.keys())
        if not splits:
            raise first_err
        chosen = splits[0]
        ds = ds_dict[chosen]
        print(f"  split '{split_hint}' unavailable; using '{chosen}':"
              f" {len(ds)} rows")

    print(f"  fields: {sorted(ds.column_names)}")
    return list(ds)


def find_game(games: List[Dict], note_id: str, mode: str) -> Optional[Dict]:
    for g in games:
        if (
            g.get("note_id") == note_id
            and g.get("mode") == mode
            and g.get("phase") == "game_complete"
        ):
            return g
    return None


def build_messages(game: Dict, prompt_style: str) -> Optional[Dict]:
    modified_note = (
        game.get("modified_sentences")
        or game.get("modified_note")
        or ""
    )
    if not modified_note:
        return None

    original_sentence = game.get("original_sentence") or ""
    modified_sentence = game.get("modified_sentence") or ""
    changed_sid = (
        game.get("changed_sid")
        or game.get("error_sentence_id")
        or ""
    )

    if (
        prompt_style == "hint_v2"
        and original_sentence
        and modified_sentence
        and str(changed_sid).strip()
    ):
        sys_p = SYS_PROMPT_BASE + HINT_V2_CARVEOUT
        usr_p = USER_HINT_V2.format(
            changed_sid=changed_sid,
            original_sentence=original_sentence,
            modified_sentence=modified_sentence,
            modified_note=modified_note,
        )
        return dict(messages=[
            {"role": "system", "content": sys_p},
            {"role": "user",   "content": usr_p},
        ], used_hint_v2=True)

    return dict(messages=[
        {"role": "system", "content": SYS_PROMPT_BASE},
        {"role": "user",   "content": str(modified_note)},
    ], used_hint_v2=False)


def apply_chat_template_no_thinking(tokenizer, messages) -> str:
    """Render messages as a prompt string with enable_thinking=False.

    MedRECT-32B is Qwen3-based and its chat template defaults to
    enable_thinking=True, which would emit <think>...</think> tokens and
    corrupt our first-token logprob analysis. Force it off. Falls back
    gracefully if the tokenizer doesn't accept enable_thinking.
    """
    try:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        # Older tokenizers don't support enable_thinking kwarg — try without.
        # This is a real risk with older Qwen; warn loudly.
        print("  WARN: tokenizer does not accept enable_thinking=False."
              " If MedRECT emits <think> tokens the logprobs will be wrong.")
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )


def summarize_first_token(logprobs_seq, tokenizer) -> Dict:
    """First non-whitespace, non-<think> token + softmax over its top-K."""
    if not logprobs_seq:
        return dict(first_token=None, p_top=None, top5=[])

    skip_texts = {"", "<think>", "\n", "\n\n"}

    for step in logprobs_seq:
        # step is Dict[int, Logprob]  (token_id -> Logprob object)
        candidates = []
        for tok_id, lp in step.items():
            tok = getattr(lp, "decoded_token", None)
            if tok is None:
                # Fall back to decoding via tokenizer
                try:
                    tok = tokenizer.decode([tok_id])
                except Exception:
                    tok = ""
            candidates.append((tok, float(lp.logprob)))
        candidates.sort(key=lambda x: -x[1])
        if not candidates:
            continue

        top_tok = candidates[0][0]
        if top_tok.strip() and top_tok.strip() not in skip_texts:
            probs = {t: math.exp(lp) for t, lp in candidates}
            total = sum(probs.values())
            probs = {t: p / total for t, p in probs.items()}
            top5 = sorted(probs.items(), key=lambda x: -x[1])[:5]
            return dict(first_token=top_tok, p_top=top5[0][1], top5=top5)

    return dict(first_token=None, p_top=None, top5=[])


# ── Main ────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="results/judge_logprobs_pilot.json")
    ap.add_argument("--dataset", default="Abdine/medserl-v6-run2-interactions")
    ap.add_argument("--split", default="train",
                    help="split hint; script falls back to whatever's available")
    ap.add_argument("--model", default="pfnet/Preferred-MedRECT-32B")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--top-logprobs", type=int, default=20)
    ap.add_argument("--prompt-style", default="hint_v2",
                    choices=["hint_v2", "native"])
    ap.add_argument("--dry-run", action="store_true",
                    help="Load dataset + tokenizer only, skip the 32B model."
                         " ~1 min, no GPU. Run this first.")
    args = ap.parse_args()

    print(f"Model         : {args.model}")
    print(f"TP size       : {args.tensor_parallel_size}")
    print(f"Prompt style  : {args.prompt_style}")
    print(f"Dataset       : {args.dataset}")
    if args.dry_run:
        print("Mode          : DRY RUN (no model load, no inference)")
    print()

    # ── Load dataset first (cheap; catches schema issues before GPU work) ──
    print(f"[1/3] Loading dataset {args.dataset}...")
    games = load_all_games(args.dataset, args.split)

    prompts_built: List[Dict] = []
    for note_id, mode, audit in AUDIT_CASES:
        game = find_game(games, note_id, mode)
        if game is None:
            print(f"  MISS  {note_id} ({mode}): no matching game_complete row")
            continue
        p = build_messages(game, args.prompt_style)
        if p is None:
            print(f"  SKIP  {note_id}: modified_sentences missing / empty")
            continue
        prompts_built.append(dict(
            note_id=note_id, mode=mode, audit=audit,
            original_verdict=game.get("judge_verdict"),
            messages=p["messages"],
            used_hint_v2=p["used_hint_v2"],
        ))

    if not prompts_built:
        sys.exit("No prompts built — check dataset access, split name, and note_ids.")

    hv2_ok = sum(1 for p in prompts_built if p["used_hint_v2"])
    print(f"  built {len(prompts_built)} prompts "
          f"({hv2_ok} using hint_v2, {len(prompts_built) - hv2_ok} falling back to native)")
    if hv2_ok == 0 and args.prompt_style == "hint_v2":
        print("  WARN: hint_v2 requested but no games have original_sentence /"
              " modified_sentence / changed_sid fields. All prompts use"
              " native. Signal is still valid but slightly off-distribution"
              " vs the actual training prompts.")
    print()

    # ── Load tokenizer (fast) and build rendered prompt strings ────────────
    print(f"[2/3] Loading tokenizer {args.model} (fast; ~15s)...")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    for p in prompts_built:
        p["prompt_str"] = apply_chat_template_no_thinking(tokenizer, p["messages"])

    # Preview one prompt so user can eyeball what the judge will see
    print("  sample rendered prompt (first 400 chars):")
    print("  " + "-" * 70)
    preview = prompts_built[0]["prompt_str"][:400].replace("\n", "\n  ")
    print(f"  {preview}...")
    print("  " + "-" * 70)
    print()

    if args.dry_run:
        print("DRY RUN complete. Re-run without --dry-run to load the 32B model"
              " and get logprobs.")
        return

    # ── Load MedRECT-32B in-process ─────────────────────────────────────────
    print(f"[3/3] Loading {args.model} into vLLM (~5 min cold, ~1 min warm)...")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=0.0,
        max_tokens=32,   # room for MedRECT to emit "CORRECT" or a sid + eos
        logprobs=args.top_logprobs,
    )
    print()

    print(f"Running judge on {len(prompts_built)} prompts (batched)...")
    prompt_strs = [p["prompt_str"] for p in prompts_built]
    outputs = llm.generate(prompt_strs, sampling_params=sampling)

    results: List[Dict] = []
    for meta, out in zip(prompts_built, outputs):
        gen = out.outputs[0]
        text = gen.text.strip()
        summary = summarize_first_token(gen.logprobs, tokenizer)
        results.append(dict(
            note_id=meta["note_id"],
            mode=meta["mode"],
            audit=meta["audit"],
            original_verdict=meta["original_verdict"],
            used_hint_v2=meta["used_hint_v2"],
            replayed_text=text,
            first_token=summary["first_token"],
            p_top=summary["p_top"],
            top5=summary["top5"],
        ))

    # ── Table ───────────────────────────────────────────────────────────────
    print()
    print("=" * 108)
    print(
        f"{'note_id':<26} {'audit':<24} {'orig':>6} {'said':>10} "
        f"{'p_top':>6}  top-3 alts"
    )
    print("=" * 108)
    for r in results:
        top3 = ", ".join(
            f"{repr(tok)[:12]}={p:.2f}" for tok, p in r["top5"][:3]
        )
        p_top = f"{r['p_top']:.2f}" if r["p_top"] is not None else "n/a"
        said = (r["replayed_text"] or "?")[:10]
        orig = (r["original_verdict"] or "?")[:6]
        print(
            f"{r['note_id']:<26} {r['audit']:<24} {orig:>6} {said:>10} "
            f"{p_top:>6}  {top3}"
        )

    # ── Heuristic GO/NO-GO ──────────────────────────────────────────────────
    difficult_labels = {"REAL JUDGE FAILURE", "JUDGE OVER-FLAG", "AMBIGUOUS"}
    difficult = [r for r in results if r["audit"] in difficult_labels
                 and r["p_top"] is not None]
    easy = [r for r in results if r["audit"] not in difficult_labels
            and r["p_top"] is not None]

    print()
    if not difficult or not easy:
        print("  (not enough labeled cases for a signal test — check MISS/SKIP lines)")
    else:
        avg_diff = sum(r["p_top"] for r in difficult) / len(difficult)
        avg_easy = sum(r["p_top"] for r in easy) / len(easy)
        gap = avg_easy - avg_diff
        print(f"Avg p_top on DIFFICULT cases (n={len(difficult)}): {avg_diff:.3f}")
        print(f"Avg p_top on EASY cases       (n={len(easy)}): {avg_easy:.3f}")
        print(f"Signal gap: {gap:+.3f}")
        print()
        if gap > 0.15:
            print("  GO — clear confidence gap. Paper 1 (continuous scoring) "
                  "is likely to soften the reward cliff meaningfully. Proceed "
                  "to Phase 2 (925-sample MEDEC calibration).")
        elif gap > 0.05:
            print("  WEAK — marginal gap. Worth running Phase 2 on 925 MEDEC "
                  "samples to quantify before committing engineering time.")
        else:
            print("  NO-GO — judge is uniformly confident regardless of "
                  "correctness. Paper 1 would give confident wrong rewards. "
                  "Skip; focus on paper 3 (co-evolving judge) or bigger "
                  "data / base model.")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        def _clean(r):
            r2 = dict(r); r2["top5"] = [[t, p] for t, p in r["top5"]]
            return r2
        json.dump(dict(
            model=args.model,
            prompt_style=args.prompt_style,
            n_cases=len(results),
            cases=[_clean(r) for r in results],
        ), f, indent=2)
    print(f"\nFull results written to: {out}")


if __name__ == "__main__":
    main()
