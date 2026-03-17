"""Generate MAGIC-style chained injector→assessor training data.

The assessor sees the injector model's ACTUAL output (not MEDEC ground truth).

Steps:
  1. Load MEDEC pairs
  2. Build injector prompts (benign + error modes)
  3. Run offline vllm inference → get injector model outputs
  4. Parse each output with parse_injector_compact → (sentence_id, modified_text)
  5. Reconstruct the note with injector's modification applied
  6. Build assessor prompts from those reconstructed notes
  7. Assessor ground truth = what the injector ACTUALLY did:
       benign mode  → "CORRECT"
       error mode   → str(sentence_id injector used), or fallback to MEDEC sid
  8. Save combined injector + assessor rows to parquet

Usage:
    python3 scripts/self_play/generate_chained_data.py \\
        --model  Qwen/Qwen3-4B \\
        --input  data_processed/medec_paired/train_val_split/rl_train.jsonl \\
        --output data_processed/self_play/train_chained.parquet \\
        --max-pairs 20
"""

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Optional

# ── make scripts/ importable ────────────────────────────────────────────────
_ROOT = Path(__file__).parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from scripts.self_play.utils import (
    number_sentences,
    split_sentences,
    parse_injector_compact,
    find_error_sentence_id,
)


# ─── helpers ────────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def load_json(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def reconstruct_note(original_note: str, sentence_id: int, modified_text: str) -> str:
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
        change_type = random.choice(list(change_types.keys())) if change_types else "pseudo_factual"
        change_desc = change_types.get(change_type, "Replace a medical term with an equivalent synonym.")
        benign_user = injection_prompts["injector_correct_template"].format(
            sentences=sentences_correct,
            change_type=change_type,
            change_type_description=change_desc,
        )
        items.append((
            idx, "benign",
            [
                {"role": "system", "content": injection_prompts["system_prompt_correct"]},
                {"role": "user",   "content": benign_user},
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
        medec_sid = find_error_sentence_id(pair["incorrect_note"], pair.get("error_sentence", ""))
        items.append((
            idx, "error_injection",
            [
                {"role": "system", "content": injection_prompts["system_prompt_incorrect"]},
                {"role": "user",   "content": error_user},
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

def run_vllm_inference(model_path: str, chat_inputs: list[list[dict]]) -> list[str]:
    """Run offline vllm batch inference. Returns one response string per input."""
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print(f"[generate_chained] Loading model {model_path} with vllm …")
    llm = LLM(
        model=model_path,
        max_model_len=4096,
        gpu_memory_utilization=0.70,
        enforce_eager=True,
        enable_prefix_caching=False,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    sampling = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=3072,  # match run_grpo_smoke.sh max_response_length
    )

    # Disable thinking for Qwen3 — Phase A only needs "N. sentence", not CoT.
    # enable_thinking=False adds /no_think token so the model skips <think> blocks.
    def _apply(msgs):
        try:
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            # Older tokenizer that doesn't support enable_thinking
            return tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )

    prompts = [_apply(msgs) for msgs in chat_inputs]
    outputs = llm.generate(prompts, sampling)
    return [o.outputs[0].text.strip() for o in outputs]


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
        },
    }


def make_assessor_row(pair, idx, mode, detection_prompts,
                      reconstructed_note, ground_truth, data_source):
    system_prompt = detection_prompts["system_prompt"]
    user_template  = detection_prompts["user_template"]
    sentences = number_sentences(reconstructed_note)
    user_prompt = user_template.format(sentences=sentences)
    note_id = pair.get("note_id", f"selfplay-{idx}")

    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
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
            "error_sentence_id": None if ground_truth == "CORRECT" else int(ground_truth),
            "corrected_sentence": "",
            "mode": mode,
            "chained": True,  # mark: assessor saw model-generated injector output
        },
    }


# ─── main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate MAGIC-style chained injector→assessor training data"
    )
    parser.add_argument("--model", required=True, help="Model path or HF id (used as injector)")
    parser.add_argument("--input", type=Path,
                        default=Path("data_processed/medec_paired/train_val_split/rl_train.jsonl"))
    parser.add_argument("--output", type=Path,
                        default=Path("data_processed/self_play/train_chained.parquet"))
    parser.add_argument("--injection-prompts",
                        default="configs/prompts/error_injection_prompts_v4.json")
    parser.add_argument("--detection-prompts",
                        default="configs/prompts/detection_localization_prompts.json")
    parser.add_argument("--max-pairs", type=int, default=20)
    parser.add_argument("--data-source", default="medec_chained")
    args = parser.parse_args()

    injection_prompts = load_json(args.injection_prompts)
    detection_prompts = load_json(args.detection_prompts)
    pairs = load_jsonl(args.input)
    print(f"[generate_chained] Loaded {len(pairs)} pairs, capping at {args.max_pairs}")

    # 1. Build all injector prompts
    injector_items = build_injector_prompts(pairs, injection_prompts, args.max_pairs)
    print(f"[generate_chained] {len(injector_items)} injector prompts "
          f"({sum(1 for _,m,*_ in injector_items if m=='benign')} benign, "
          f"{sum(1 for _,m,*_ in injector_items if m=='error_injection')} error)")

    # 2. Run model inference on all injector prompts
    chat_inputs = [msgs for _, _, msgs, _ in injector_items]
    outputs = run_vllm_inference(args.model, chat_inputs)

    # 3. Parse outputs and build rows
    rows = []
    parse_ok = 0
    parse_fail = 0

    pairs_by_idx = {i: p for i, p in enumerate(pairs[:args.max_pairs])}

    for (pair_idx, mode, messages, meta), raw_output in zip(injector_items, outputs):
        pair = pairs_by_idx[pair_idx]
        print(f"  [{mode}] injector output: {raw_output[:80]!r}")

        # ── injector ground truth (MEDEC-based, same as run_grpo_smoke.sh) ──
        if mode == "benign":
            injector_gt = "CORRECT"
        else:
            injector_gt = str(meta["medec_sid"]) if meta["medec_sid"] else "INCORRECT"

        rows.append(make_injector_row(pair, pair_idx, mode, messages, meta,
                                      injector_gt, args.data_source))

        # ── parse injector output to build chained assessor ──
        sentence_id, modified_text = parse_injector_compact(raw_output)

        if sentence_id is not None and modified_text:
            parse_ok += 1
            reconstructed = reconstruct_note(meta["note"], sentence_id, modified_text)
            # Assessor ground truth = what injector ACTUALLY did
            assessor_gt = "CORRECT" if mode == "benign" else str(sentence_id)
        else:
            # Fallback: use the original MEDEC note and ground truth
            parse_fail += 1
            print(f"    WARNING: could not parse injector output — using MEDEC fallback")
            reconstructed = meta["note"]
            assessor_gt = "CORRECT" if mode == "benign" else injector_gt

        rows.append(make_assessor_row(pair, pair_idx, mode, detection_prompts,
                                       reconstructed, assessor_gt, args.data_source))

    print(f"\n[generate_chained] Injector parse: {parse_ok} OK, {parse_fail} fallback")
    print(f"[generate_chained] Total rows: {len(rows)} "
          f"(injector: {sum(1 for r in rows if r['extra_info']['role']=='injector')}, "
          f"assessor: {sum(1 for r in rows if r['extra_info']['role']=='assessor')})")

    # 4. Save parquet
    from datasets import Dataset
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ds = Dataset.from_list(rows)
    ds.to_parquet(str(args.output))
    print(f"[generate_chained] Saved → {args.output}")

    # Quick verification
    loaded = Dataset.from_parquet(str(args.output))
    print(f"[generate_chained] Verified: {len(loaded)} rows, "
          f"prompt type={type(loaded[0]['prompt'])}")


if __name__ == "__main__":
    main()
