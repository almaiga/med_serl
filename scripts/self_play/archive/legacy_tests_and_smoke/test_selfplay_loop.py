#!/usr/bin/env python3
"""Quick self-play simulation — NO verl, NO RL, just prompting.

Tests the full 2-turn loop with Qwen3-4B:
  Turn 1 (Injector): model receives numbered note → outputs "N. <modified sentence>"
  Turn 2 (Assessor): model receives modified note → outputs CORRECT or sentence number

Usage:
  # Baseline: error mode, thinking OFF (default), 10 examples
  python scripts/self_play/test_selfplay_loop.py --mode error --n 10

  # Guided injection — model is shown the exact target sentence to modify
  python scripts/self_play/test_selfplay_loop.py --mode error --n 10 --guided

  # Mode comparison: run error + benign side-by-side
  python scripts/self_play/test_selfplay_loop.py --mode both --n 10

  # Temperature sweep — compare injector temps 0.3 / 0.6 / 0.9 on same examples
  python scripts/self_play/test_selfplay_loop.py --mode error --n 10 \
      --sweep-injector-temps 0.3,0.6,0.9

  # Enable Qwen3 thinking mode (use on GPU where it's fast)
  python scripts/self_play/test_selfplay_loop.py --mode error --n 5 --think

  # Custom model / budget
  python scripts/self_play/test_selfplay_loop.py --model Qwen/Qwen3-4B \
      --injector-temp 0.6 --assessor-temp 0.3 --max-tokens-injector 512

Requires: pip install transformers torch
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

# ── Setup PYTHONPATH ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.self_play.utils import (
    number_sentences,
    split_sentences,
    parse_assessor_answer,
    parse_injector_compact,
    reconstruct_note,
    strip_thinking,
    find_error_sentence_id,
)

# ─── Reward constants (same as training) ────────────────────────────────────
REWARD_EXACT = 1.0
REWARD_PARTIAL = 0.3
REWARD_MISS = -1.0
FORMAT_BONUS = 0.2


# ─── Prompt builders ────────────────────────────────────────────────────────

def load_prompts():
    inj = json.loads((PROJECT_ROOT / "configs/prompts/error_injection_prompts_v4.json").read_text())
    det = json.loads((PROJECT_ROOT / "configs/prompts/detection_localization_prompts.json").read_text())
    return inj, det


def build_injector_messages(pair, inj_prompts, mode="error", guided=False):
    """Build chat messages for the Injector turn.

    guided=True: pre-select the target sentence and show it explicitly.
      Error mode: uses corrected_sentence to locate the sentence id.
      Benign mode: picks a random sentence.
    This removes sentence-selection reasoning from the model, making the
    task purely about modifying the given text.
    """
    sentences = number_sentences(pair["correct_note"])
    sents_list = split_sentences(pair["correct_note"])

    if mode == "benign":
        system = inj_prompts["system_prompt_correct"]
        change_types = inj_prompts.get("benign_change_types", {})
        ct = random.choice(list(change_types.keys())) if change_types else "pseudo_factual"
        ct_desc = change_types.get(ct, "Replace a medical term with an equivalent synonym.")
        if guided:
            sid = random.randint(1, len(sents_list))
            orig = sents_list[sid - 1]
            user = (
                f"Make a {ct} change to this sentence:\n"
                f"{ct_desc}\n\n"
                f"Sentence {sid}: {orig}\n\n"
                f"Output only:\n{sid}. <modified sentence>"
            )
        else:
            template = inj_prompts["injector_correct_template"]
            user = template.format(
                sentences=sentences, change_type=ct,
                change_type_description=ct_desc,
            )
    else:  # error mode
        system = inj_prompts["system_prompt_incorrect"]
        error_type = pair.get("error_type", "clinical error")
        if guided:
            # Locate the correct (pre-error) sentence in the note
            corrected = pair.get("corrected_sentence", "")
            sid = find_error_sentence_id(pair["correct_note"], corrected)
            if sid and 1 <= sid <= len(sents_list):
                orig = sents_list[sid - 1]
            else:
                # fallback: random sentence
                sid = random.randint(1, len(sents_list))
                orig = sents_list[sid - 1]
            user = (
                f"Inject a {error_type} error into this sentence by changing 1-3 words:\n\n"
                f"Sentence {sid}: {orig}\n\n"
                f"Output only:\n{sid}. <modified sentence>"
            )
        else:
            template = inj_prompts["injector_incorrect_template"]
            user = template.format(sentences=sentences, prompt_intent=error_type)

    return sentences, [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def build_assessor_messages(modified_note, det_prompts):
    """Build chat messages for the Assessor turn."""
    system = det_prompts["system_prompt"]
    user = det_prompts.get("user_template", "{sentences}").format(sentences=modified_note)
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


# ─── Model wrapper ──────────────────────────────────────────────────────────

class QwenModel:
    """Thin wrapper around HuggingFace Qwen3 for chat generation."""

    def __init__(self, model_path, device="auto"):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        print(f"Loading {model_path} ...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        # MPS (Apple Silicon) doesn't handle float16/bfloat16 sampling well
        # — produces NaN logits → RuntimeError in multinomial.
        # Force float32 on MPS; use auto (bfloat16) elsewhere.
        if device == "auto" and torch.backends.mps.is_available():
            dtype = torch.float32
            device_map = "mps"
        else:
            dtype = "auto"
            device_map = device

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )
        self.model.eval()
        print(f"Model loaded on {self.model.device} (dtype={self.model.dtype})")

    def generate(self, messages, temperature=0.6, top_p=0.95,
                 max_new_tokens=2048, enable_thinking=True):
        """Generate a single response from chat messages.

        Args:
            enable_thinking: If True (default), Qwen3 generates <think>...</think>
                reasoning before the answer. If False, thinking is suppressed
                via the chat template and the model outputs the answer directly.
        """
        # Qwen3 chat template accepts enable_thinking to control <think> mode
        template_kwargs = dict(
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
        text = self.tokenizer.apply_chat_template(messages, **template_kwargs)
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        n_prompt_tokens = inputs["input_ids"].shape[1]

        kwargs = dict(
            max_new_tokens=max_new_tokens,
            top_p=top_p,
            do_sample=temperature > 0,
        )
        if temperature > 0:
            kwargs["temperature"] = temperature

        output_ids = self.model.generate(**inputs, **kwargs)
        # Decode only the new tokens
        new_ids = output_ids[0, n_prompt_tokens:]
        n_gen_tokens = len(new_ids)
        response = self.tokenizer.decode(new_ids, skip_special_tokens=True)
        return response, n_gen_tokens


# ─── Reward logic ────────────────────────────────────────────────────────────

def compute_reward(label, pred_sid, ground_truth):
    """3-tier reward identical to training."""
    has_valid = label != "UNKNOWN"
    fb = FORMAT_BONUS if has_valid else 0.0
    gt_correct = (ground_truth == "CORRECT")

    if gt_correct and label == "CORRECT":
        return REWARD_EXACT + fb, "exact_match"
    if gt_correct and label != "CORRECT":
        return REWARD_MISS + fb, "false_positive"
    if not gt_correct and label == "CORRECT":
        return REWARD_MISS + fb, "missed_error"
    if not gt_correct and label == "ERROR":
        ps = str(pred_sid) if pred_sid else ""
        if ps == ground_truth:
            return REWARD_EXACT + fb, "exact_match"
        elif pred_sid is not None:
            return REWARD_PARTIAL + fb, "partial_match"
        else:
            return REWARD_PARTIAL, "partial_no_sid"
    return REWARD_MISS, "invalid_format"


# ─── Main loop ───────────────────────────────────────────────────────────────

def run_selfplay_test(args):
    inj_prompts, det_prompts = load_prompts()

    # Load data
    data_path = PROJECT_ROOT / args.data
    pairs = [json.loads(l) for l in data_path.read_text().splitlines() if l.strip()]
    print(f"Loaded {len(pairs)} pairs from {data_path.name}")

    # Sample once — reused across all sweep temps
    if args.n < len(pairs):
        pairs = random.sample(pairs, args.n)
    else:
        pairs = pairs[:args.n]

    # Load model
    model = QwenModel(args.model, device=args.device)

    # Determine modes and temperature sweep
    modes = ["error", "benign"] if args.mode == "both" else [args.mode]
    sweep_temps = (
        [float(t.strip()) for t in args.sweep_injector_temps.split(",")]
        if args.sweep_injector_temps
        else [args.injector_temp]
    )
    if len(sweep_temps) > 1:
        print(f"Temperature sweep: injector temps = {sweep_temps}")

    results = []
    for inj_temp in sweep_temps:
        if len(sweep_temps) > 1:
            print(f"\n{'#'*70}")
            print(f"# Injector temp = {inj_temp}")
            print(f"{'#'*70}")
        for i, pair in enumerate(pairs):
          for mode in modes:
            gt = "CORRECT" if mode == "benign" else str(
                find_error_sentence_id(
                    pair["correct_note"], pair.get("corrected_sentence", "")
                ) or "?"
            )

            print(f"\n{'='*70}")
            guided_tag = " guided" if args.guided else ""
            print(f"Example {i+1}/{len(pairs)} | mode={mode}{guided_tag} | "
                  f"note_id={pair.get('note_id','?')} | GT={gt}")
            print(f"{'='*70}")

            # ── Turn 1: Injector ──
            original_sentences, inj_msgs = build_injector_messages(
                pair, inj_prompts, mode, guided=args.guided,
            )
            t0 = time.time()
            inj_response, inj_tokens = model.generate(
                inj_msgs,
                temperature=inj_temp,
                top_p=0.95,
                max_new_tokens=args.max_tokens_injector,
                enable_thinking=args.enable_thinking,
            )
            inj_time = time.time() - t0

            # Parse injector output
            sid, mod_text = parse_injector_compact(inj_response)
            inj_thinking, inj_public = strip_thinking(inj_response)
            inj_think_tokens = len(inj_thinking.split()) if inj_thinking else 0

            # Detect truncated thinking
            inj_truncated = ("<think>" in inj_response
                             and "</think>" not in inj_response)

            think_tag = "THINK" if args.enable_thinking else "no-think"
            print(f"\n--- Injector (T={inj_temp}, "
                  f"{think_tag}, {inj_tokens}tok, {inj_time:.1f}s) ---")
            if inj_truncated:
                print(f"  ⚠ TRUNCATED THINKING — model used all "
                      f"{inj_tokens} tokens on <think> without closing")
            elif inj_thinking:
                print(f"  Thinking: ~{inj_think_tokens} words")
            print(f"Answer: {inj_public.strip()[:300]}")
            print(f"Parsed: sid={sid}, text="
                  f"{mod_text[:120] if mod_text else 'PARSE FAIL'}")

            if sid and mod_text:
                modified_note = reconstruct_note(
                    original_sentences, sid, mod_text,
                )
            else:
                print("  WARNING: Injector parse failed — "
                      "passing original note to assessor")
                modified_note = original_sentences

            # ── Turn 2: Assessor ──
            assess_msgs = build_assessor_messages(modified_note, det_prompts)
            t0 = time.time()
            assess_response, assess_tokens = model.generate(
                assess_msgs,
                temperature=args.assessor_temp,
                top_p=0.95,
                max_new_tokens=args.max_tokens_assessor,
                enable_thinking=args.enable_thinking,
            )
            assess_time = time.time() - t0

            label, pred_sid = parse_assessor_answer(assess_response)
            assess_thinking, assess_public = strip_thinking(assess_response)
            assess_think_tokens = len(assess_thinking.split()) if assess_thinking else 0

            # Detect truncated thinking
            assess_truncated = ("<think>" in assess_response
                                and "</think>" not in assess_response)

            print(f"\n--- Assessor (T={args.assessor_temp}, "
                  f"{think_tag}, {assess_tokens}tok, {assess_time:.1f}s) ---")
            if assess_truncated:
                print(f"  ⚠ TRUNCATED THINKING — model used all "
                      f"{assess_tokens} tokens on <think> without closing")
            elif assess_thinking:
                print(f"  Thinking: ~{assess_think_tokens} words")
            print(f"Answer: {assess_public.strip()[:300]}")
            print(f"Parsed: label={label}, pred_sid={pred_sid}")

            # ── Reward ──
            reward, outcome = compute_reward(label, pred_sid, gt)
            sym = {"exact_match": "✓", "partial_match": "~",
                   "partial_no_sid": "~"}.get(outcome, "✗")

            print(f"\n  {sym} outcome={outcome}  reward={reward:+.1f}  "
                  f"(GT={gt}, pred={label} {pred_sid})")

            results.append({
                "note_id": pair.get("note_id", ""),
                "mode": mode,
                "guided": args.guided,
                "injector_temp": inj_temp,
                "ground_truth": gt,
                "injector_sid": sid,
                "injector_text": (mod_text or "")[:200],
                "injector_parse_ok": sid is not None,
                "injector_truncated": inj_truncated,
                "injector_tokens": inj_tokens,
                "assessor_label": label,
                "assessor_pred_sid": pred_sid,
                "assessor_truncated": assess_truncated,
                "assessor_tokens": assess_tokens,
                "outcome": outcome,
                "reward": reward,
                "injector_time": round(inj_time, 1),
                "assessor_time": round(assess_time, 1),
                "injector_response": inj_public.strip()[:500],
                "assessor_response": assess_public.strip()[:500],
                "enable_thinking": args.enable_thinking,
            })

    # ── Summary ──────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    total = len(results)
    if total == 0:
        print("No results.")
        return

    inj_ok = sum(1 for r in results if r["injector_parse_ok"])
    inj_trunc = sum(1 for r in results if r["injector_truncated"])
    assess_trunc = sum(1 for r in results if r["assessor_truncated"])
    exact = sum(1 for r in results if r["outcome"] == "exact_match")
    partial = sum(1 for r in results if "partial" in r["outcome"])
    miss = total - exact - partial
    avg_reward = sum(r["reward"] for r in results) / total
    avg_inj_tok = sum(r["injector_tokens"] for r in results) / total
    avg_assess_tok = sum(r["assessor_tokens"] for r in results) / total

    guided_flag = results[0].get("guided", False)
    print(f"Thinking        : {'ON' if results[0].get('enable_thinking') else 'OFF'}")
    print(f"Guided mode     : {'YES' if guided_flag else 'NO'}")
    print(f"Total examples  : {total}")
    print(f"Injector parsed : {inj_ok}/{total} ({inj_ok/total:.0%})")
    print(f"Inj truncated   : {inj_trunc}/{total} (thinking exceeded budget)")
    print(f"Assess truncated: {assess_trunc}/{total}")
    print(f"Avg tokens      : injector={avg_inj_tok:.0f}, assessor={avg_assess_tok:.0f}")
    print(f"Exact match     : {exact}/{total} ({exact/total:.0%})")
    print(f"Partial match   : {partial}/{total} ({partial/total:.0%})")
    print(f"Miss/invalid    : {miss}/{total} ({miss/total:.0%})")
    print(f"Avg reward      : {avg_reward:+.3f}")

    # By mode
    modes_seen = sorted(set(r["mode"] for r in results))
    for m in modes_seen:
        mr = [r for r in results if r["mode"] == m]
        if mr:
            avg_r = sum(r["reward"] for r in mr) / len(mr)
            ex = sum(1 for r in mr if r["outcome"] == "exact_match")
            iok = sum(1 for r in mr if r["injector_parse_ok"])
            print(f"  [{m:6s}] n={len(mr)}  "
                  f"injector_ok={iok}  exact={ex}  "
                  f"avg_reward={avg_r:+.3f}")

    # Temperature sweep comparison table
    temps_seen = sorted(set(r["injector_temp"] for r in results))
    if len(temps_seen) > 1:
        print(f"\n{'─'*60}")
        print(f"{'Temp':>6}  {'Parsed':>7}  {'Exact':>6}  {'Partial':>8}  {'Reward':>7}  {'Tok':>5}")
        print(f"{'─'*60}")
        for t in temps_seen:
            tr = [r for r in results if r["injector_temp"] == t]
            n = len(tr)
            p_ok  = sum(1 for r in tr if r["injector_parse_ok"])
            p_ex  = sum(1 for r in tr if r["outcome"] == "exact_match")
            p_par = sum(1 for r in tr if "partial" in r["outcome"])
            avg_r = sum(r["reward"] for r in tr) / n
            avg_t = sum(r["injector_tokens"] for r in tr) / n
            print(f"  {t:>4}  {p_ok:>3}/{n:<3}  {p_ex:>3}/{n:<3}  "
                  f"{p_par:>4}/{n:<3}  {avg_r:>+6.3f}  {avg_t:>5.0f}")
        print(f"{'─'*60}")

    # Save results
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = (PROJECT_ROOT / "results" / "self_play"
                / f"test_loop_{ts}.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nResults saved: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Quick self-play simulation (no verl, no RL)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", default="Qwen/Qwen3-4B",
        help="HF model path (default: Qwen/Qwen3-4B)",
    )
    parser.add_argument(
        "--device", default="auto",
        help="Device map: auto / cuda / cpu",
    )
    parser.add_argument(
        "--mode", choices=["error", "benign", "both"], default="error",
        help="Which mode to test (default: error)",
    )
    parser.add_argument(
        "--n", type=int, default=5,
        help="Number of note pairs to test (default: 5)",
    )
    parser.add_argument(
        "--data",
        default="data_processed/medec_paired/train_val_split/rl_train.jsonl",
        help="Input JSONL path (relative to project root)",
    )
    parser.add_argument(
        "--injector-temp", type=float, default=0.6,
        help="Injector temperature (default: 0.6, Qwen3 recommended)",
    )
    parser.add_argument(
        "--assessor-temp", type=float, default=0.3,
        help="Assessor temperature (default: 0.3, precise classification)",
    )
    parser.add_argument(
        "--max-tokens-injector", type=int, default=2048,
        help="Max new tokens for injector turn (default: 2048)",
    )
    parser.add_argument(
        "--max-tokens-assessor", type=int, default=1024,
        help="Max new tokens for assessor turn (default: 1024)",
    )
    parser.add_argument(
        "--guided", action="store_true", default=False,
        help=(
            "Guided injection: show the target sentence explicitly. "
            "Error mode uses corrected_sentence; benign uses a random sentence. "
            "Reduces model confusion when parsing long notes."
        ),
    )
    parser.add_argument(
        "--sweep-injector-temps", default=None,
        metavar="T1,T2,...",
        help=(
            "Comma-separated injector temps to sweep, e.g. '0.3,0.6,0.9'. "
            "Runs the same examples at each temp for direct comparison. "
            "Overrides --injector-temp."
        ),
    )
    parser.add_argument(
        "--think", action="store_true", default=False,
        help="Enable Qwen3 thinking mode (default: OFF — use on GPU where it's fast)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)",
    )

    args = parser.parse_args()
    args.enable_thinking = args.think
    random.seed(args.seed)
    run_selfplay_test(args)


if __name__ == "__main__":
    main()
