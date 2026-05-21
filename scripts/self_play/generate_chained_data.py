"""Generate MAGIC-style chained injector->assessor training data.

The assessor sees the injector model's ACTUAL output (not MEDEC ground truth).

Steps:
  1. Load MEDEC pairs
  2. Build injector prompts (benign + error modes)
  3. Run offline vllm inference -> get injector model outputs
  4. Parse each output with parse_injector_compact -> (sentence_id, modified_text)
  5. Reconstruct the note with injector's modification applied
  6. Build assessor prompts from those reconstructed notes
  7. Assessor ground truth = what the injector ACTUALLY did:
       benign mode  -> "CORRECT"
       error mode   -> str(sentence_id injector used), or fallback to MEDEC sid
  8. (--zero-sum) Run second assessor inference pass; set injector reward to
     -assessor_reward (error) / +assessor_reward (benign) for adversarial coupling.
  9. Save combined injector + assessor rows to parquet

Usage:
    python3 scripts/self_play/generate_chained_data.py \\
        --model  Qwen/Qwen3-4B \\
        --input  data_processed/medec_paired/train_val_split/rl_train.jsonl \\
        --output data_processed/self_play/train_chained.parquet \\
        --max-pairs 20 [--zero-sum]
"""

import argparse
import json
import random
import sys
from pathlib import Path

# ── make scripts/ importable ────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.self_play.utils import (  # noqa: E402
    number_sentences,
    split_sentences,
    parse_injector_compact,
    find_error_sentence_id,
    parse_assessor_answer,
)


# ─── helpers ────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_json(path: str) -> dict:
    with open(path) as fh:
        return json.load(fh)


def reconstruct_note(
    original_note: str, sentence_id: int, modified_text: str
) -> str:
    """Replace sentence <sentence_id> (1-indexed) with modified_text."""
    sentences = split_sentences(original_note)
    idx = sentence_id - 1
    if 0 <= idx < len(sentences):
        sentences[idx] = modified_text
    return " ".join(sentences)


def build_numbered(note: str) -> str:
    return number_sentences(note)


# ─── prompt builders ────────────────────────────────────────────────────────

def build_injector_prompts(pairs, injection_prompts, max_pairs=None):
    """Return list of (pair_idx, mode, messages, meta) for all injector examples."""
    items = []
    change_types = injection_prompts.get("benign_change_types", {})

    for idx, pair in enumerate(pairs):
        if max_pairs and idx >= max_pairs:
            break
        if not pair.get("correct_note") or not pair.get("incorrect_note"):
            continue

        # benign mode
        sentences_correct = number_sentences(pair["correct_note"])
        change_type = (
            random.choice(list(change_types.keys()))
            if change_types else "pseudo_factual"
        )
        change_desc = change_types.get(
            change_type,
            "Replace a medical term with an equivalent synonym.",
        )
        benign_user = injection_prompts["injector_correct_template"].format(
            sentences=sentences_correct,
            change_type=change_type,
            change_type_description=change_desc,
        )
        items.append((
            idx, "benign",
            [
                {
                    "role": "system",
                    "content": injection_prompts["system_prompt_correct"],
                },
                {"role": "user", "content": benign_user},
            ],
            {
                "note": pair["correct_note"],
                "sentences": sentences_correct,
                "medec_sid": None,
                "error_type": "",
                "change_type": change_type,
            }
        ))

        # error_injection mode
        sentences_incorrect = number_sentences(pair["incorrect_note"])
        error_type = pair.get("error_type", "clinical error")
        error_user = injection_prompts["injector_incorrect_template"].format(
            sentences=sentences_incorrect,
            prompt_intent=error_type,
        )
        medec_sid = find_error_sentence_id(
            pair["incorrect_note"], pair.get("error_sentence", "")
        )
        items.append((
            idx, "error_injection",
            [
                {
                    "role": "system",
                    "content": injection_prompts["system_prompt_incorrect"],
                },
                {"role": "user", "content": error_user},
            ],
            {
                "note": pair["incorrect_note"],
                "sentences": sentences_incorrect,
                "medec_sid": medec_sid,
                "error_type": error_type,
                "change_type": None,
            }
        ))

    return items


# ─── inference ──────────────────────────────────────────────────────────────

def _make_llm_and_tokenizer(
    model_path: str,
    *,
    gpu_memory_utilization: float = 0.45,
    max_model_len: int = 4096,
):
    """Load vllm LLM + tokenizer. Returns (llm, tokenizer, apply_fn)."""
    from vllm import LLM
    from transformers import AutoTokenizer

    print(f"[generate_chained] Loading model {model_path} with vllm ...")
    llm = LLM(
        model=model_path,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=True,
        enable_prefix_caching=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def apply_no_think(msgs):
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )

    return llm, tokenizer, apply_no_think


def run_vllm_inference(
    model_path: str,
    chat_inputs: list,
    *,
    llm=None,
    apply_fn=None,
    max_tokens: int = 3072,
    temperature: float = 0.7,
    top_p: float = 0.9,
    gpu_memory_utilization: float = 0.45,
    max_model_len: int = 4096,
) -> list:
    """Run offline vllm batch inference. Returns one response string per input.

    If *llm* and *apply_fn* are already-loaded instances from
    _make_llm_and_tokenizer, they are reused — avoids double-loading for
    the zero-sum second inference pass.
    """
    from vllm import SamplingParams

    if llm is None or apply_fn is None:
        llm, _, apply_fn = _make_llm_and_tokenizer(
            model_path,
            gpu_memory_utilization=gpu_memory_utilization,
            max_model_len=max_model_len,
        )

    sampling = SamplingParams(
        temperature=temperature, top_p=top_p, max_tokens=max_tokens
    )
    prompts = [apply_fn(msgs) for msgs in chat_inputs]
    outputs = llm.generate(prompts, sampling)
    return [o.outputs[0].text.strip() for o in outputs]


# ─── zero-sum reward helper ──────────────────────────────────────────────────

def _score_assessor_output(raw_output: str, ground_truth: str) -> float:
    """3-tier assessor reward — mirrors _compute_assessor_reward in reward_function.py."""
    REWARD_EXACT = 1.0
    REWARD_PARTIAL = 0.5
    REWARD_MISS = -1.5
    FORMAT_BONUS = 0.2

    label, pred_sid = parse_assessor_answer(raw_output)
    has_fmt = label != "UNKNOWN"
    fmt_bonus = FORMAT_BONUS if has_fmt else 0.0
    gt_correct = ground_truth == "CORRECT"

    if label == "UNKNOWN":
        return REWARD_MISS
    if gt_correct and label == "CORRECT":
        return REWARD_EXACT + fmt_bonus
    if gt_correct and label != "CORRECT":
        return REWARD_MISS + fmt_bonus
    if not gt_correct and label == "CORRECT":
        return REWARD_MISS + fmt_bonus
    # error detected
    pred_str = str(pred_sid) if pred_sid is not None else ""
    if pred_str == ground_truth:
        return REWARD_EXACT + fmt_bonus
    if pred_sid is not None:
        return REWARD_PARTIAL + fmt_bonus
    return REWARD_PARTIAL


# ─── assemble parquet rows ───────────────────────────────────────────────────

def make_injector_row(pair, idx, mode, messages, meta, ground_truth, data_source):
    note_id = pair.get("note_id", f"selfplay-{idx}")
    return {
        "data_source": data_source,
        "prompt": messages,
        "ability": "medical_error_detection",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "role": "injector",
            "note_id": f"{note_id}-{mode}",
            "correct_note": pair["correct_note"],
            "incorrect_note": pair.get("incorrect_note", ""),
            "sentences": meta["sentences"],
            "error_type": meta["error_type"],
            "error_sentence": pair.get("error_sentence", ""),
            "error_sentence_id": meta["medec_sid"],
            "corrected_sentence": pair.get("corrected_sentence", ""),
            "mode": mode,
            # zero_sum_reward backfilled after second pass (None = use MEDEC proxy)
            "zero_sum_reward": None,
        },
    }


def make_assessor_row(
    pair, idx, mode, detection_prompts,
    reconstructed_note, ground_truth, data_source,
):
    system_prompt = detection_prompts["system_prompt"]
    user_template = detection_prompts["user_template"]
    sentences = number_sentences(reconstructed_note)
    user_prompt = user_template.format(sentences=sentences)
    note_id = pair.get("note_id", f"selfplay-{idx}")

    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "ability": "medical_error_detection",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "role": "assessor",
            "note_id": f"{note_id}-assessor-chained-{mode}",
            "correct_note": pair["correct_note"],
            "incorrect_note": reconstructed_note,
            "sentences": sentences,
            "error_type": pair.get("error_type", ""),
            "error_sentence": "",
            "error_sentence_id": (
                None if ground_truth == "CORRECT" else int(ground_truth)
            ),
            "corrected_sentence": "",
            "mode": mode,
            "chained": True,  # assessor saw model-generated injector output
        },
    }


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate MAGIC-style chained injector->assessor training data"
    )
    parser.add_argument(
        "--model", required=True,
        help="Model path or HF id (used as injector)",
    )
    parser.add_argument(
        "--input", type=Path,
        default=Path(
            "data_processed/medec_paired/train_val_split/rl_train.jsonl"
        ),
    )
    parser.add_argument(
        "--output", type=Path,
        default=Path("data_processed/self_play/train_chained.parquet"),
    )
    parser.add_argument(
        "--injection-prompts",
        default="configs/prompts/error_injection_prompts_v4.json",
    )
    parser.add_argument(
        "--detection-prompts",
        default="configs/prompts/detection_localization_prompts.json",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=0,
        help="Maximum number of note pairs to use. Set to 0 for all pairs.",
    )
    parser.add_argument("--data-source", default="medec_chained")
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.45,
        help="vLLM GPU memory utilization for chained datagen.",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="vLLM max model length for chained datagen.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Injector generation max tokens for chained datagen.",
    )
    parser.add_argument(
        "--zero-sum", action="store_true",
        help=(
            "Run a second assessor inference pass in Phase A. "
            "Injector reward = -assessor_reward (error) or "
            "+assessor_reward (benign), stored in extra_info['zero_sum_reward']. "
            "compute_score uses it instead of the MEDEC proxy when present."
        ),
    )
    args = parser.parse_args()

    injection_prompts = load_json(args.injection_prompts)
    detection_prompts = load_json(args.detection_prompts)
    pairs = load_jsonl(args.input)
    max_pairs = args.max_pairs if args.max_pairs and args.max_pairs > 0 else None
    max_pairs_display = max_pairs if max_pairs is not None else "ALL"
    print(
        f"[generate_chained] Loaded {len(pairs)} pairs,"
        f" capping at {max_pairs_display}"
    )
    print(
        f"[generate_chained] vLLM gpu_mem={args.gpu_memory_utilization}"
        f" max_model_len={args.max_model_len}"
        f" max_tokens={args.max_tokens}"
    )

    # 1. Build all injector prompts
    injector_items = build_injector_prompts(pairs, injection_prompts, max_pairs)
    n_benign = sum(1 for _, m, *_ in injector_items if m == "benign")
    n_error = sum(1 for _, m, *_ in injector_items if m == "error_injection")
    print(
        f"[generate_chained] {len(injector_items)} injector prompts"
        f" ({n_benign} benign, {n_error} error)"
    )

    # 2. Run injector inference (load model once; reuse for zero-sum pass)
    chat_inputs = [msgs for _, _, msgs, _ in injector_items]
    if args.zero_sum:
        llm, _, apply_fn = _make_llm_and_tokenizer(
            args.model,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
        outputs = run_vllm_inference(
            args.model,
            chat_inputs,
            llm=llm,
            apply_fn=apply_fn,
            max_tokens=args.max_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
    else:
        llm, apply_fn = None, None
        outputs = run_vllm_inference(
            args.model,
            chat_inputs,
            max_tokens=args.max_tokens,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )

    # 3. Parse outputs and build rows
    rows = []
    parse_ok = 0
    parse_fail = 0
    # (inj_row_idx, mode, assessor_gt, assessor_prompt_msgs)
    zero_sum_queue = []

    selected_pairs = pairs[:max_pairs] if max_pairs is not None else pairs
    pairs_by_idx = {i: p for i, p in enumerate(selected_pairs)}

    for (pair_idx, mode, messages, meta), raw_output in zip(injector_items, outputs):
        pair = pairs_by_idx[pair_idx]
        print(f"  [{mode}] injector output: {raw_output[:80]!r}")

        # ── injector ground truth (MEDEC-based) ──
        if mode == "benign":
            injector_gt = "CORRECT"
        else:
            injector_gt = (
                str(meta["medec_sid"]) if meta["medec_sid"] else "INCORRECT"
            )

        inj_row = make_injector_row(
            pair, pair_idx, mode, messages, meta,
            injector_gt, args.data_source,
        )
        rows.append(inj_row)
        inj_row_idx = len(rows) - 1  # index to backfill with zero_sum_reward

        # ── parse injector output to build chained assessor ──
        sentence_id, modified_text = parse_injector_compact(raw_output)

        if sentence_id is not None and modified_text:
            parse_ok += 1
            reconstructed = reconstruct_note(
                meta["note"], sentence_id, modified_text
            )
            assessor_gt = "CORRECT" if mode == "benign" else str(sentence_id)
        else:
            parse_fail += 1
            print("    WARNING: could not parse injector output — MEDEC fallback")
            reconstructed = meta["note"]
            assessor_gt = "CORRECT" if mode == "benign" else injector_gt

        asm_row = make_assessor_row(
            pair, pair_idx, mode, detection_prompts,
            reconstructed, assessor_gt, args.data_source,
        )
        rows.append(asm_row)

        if args.zero_sum:
            zero_sum_queue.append(
                (inj_row_idx, mode, assessor_gt, asm_row["prompt"])
            )

    print(
        f"\n[generate_chained] Injector parse: {parse_ok} OK,"
        f" {parse_fail} fallback"
    )
    n_inj = sum(1 for r in rows if r["extra_info"]["role"] == "injector")
    n_asm = sum(1 for r in rows if r["extra_info"]["role"] == "assessor")
    print(
        f"[generate_chained] Total rows: {len(rows)}"
        f" (injector: {n_inj}, assessor: {n_asm})"
    )

    # 4. Zero-sum pass: run assessor inference and backfill injector rewards
    if args.zero_sum and zero_sum_queue:
        print(
            f"\n[generate_chained] --zero-sum: running assessor inference"
            f" on {len(zero_sum_queue)} prompts ..."
        )
        asm_prompts = [prompt_msgs for _, _, _, prompt_msgs in zero_sum_queue]
        asm_outputs = run_vllm_inference(
            args.model, asm_prompts,
            llm=llm, apply_fn=apply_fn,
            # Assessor needs thinking enabled; lower temperature for precision
            temperature=0.3, top_p=0.95, max_tokens=1024,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len,
        )
        zs_exact = zs_partial = zs_miss = 0
        for (inj_idx, mode, assessor_gt, _), asm_out in zip(
            zero_sum_queue, asm_outputs
        ):
            asm_reward = _score_assessor_output(asm_out, assessor_gt)
            # Zero-sum: error mode is adversarial, benign is cooperative
            if mode == "error_injection":
                inj_zs_reward = -asm_reward
            else:
                inj_zs_reward = asm_reward
            rows[inj_idx]["extra_info"]["zero_sum_reward"] = inj_zs_reward
            if asm_reward >= 1.0:
                zs_exact += 1
            elif asm_reward > 0:
                zs_partial += 1
            else:
                zs_miss += 1
            print(
                f"    [{mode}] assessor={asm_out[:40]!r}"
                f" -> asm_reward={asm_reward:+.2f}"
                f"  inj_zs={inj_zs_reward:+.2f}"
            )
        print(
            f"[generate_chained] Zero-sum assessor pass:"
            f" exact={zs_exact} partial={zs_partial} miss={zs_miss}"
        )

    # 5. Save parquet
    from datasets import Dataset  # noqa: E402
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(rows)
    ds.to_parquet(str(args.output))
    print(f"[generate_chained] Saved -> {args.output}")

    # Quick verification
    loaded = Dataset.from_parquet(str(args.output))
    print(
        f"[generate_chained] Verified: {len(loaded)} rows,"
        f" prompt type={type(loaded[0]['prompt'])}"
    )


if __name__ == "__main__":
    main()
