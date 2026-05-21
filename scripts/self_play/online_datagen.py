"""Online data regeneration for MedSeRL self-play.

Each training epoch, the injector has improved. This script re-runs the
current model in INJECTOR mode on the MEDEC note pairs and creates fresh
assessor examples from those live injections.

"Chasing a moving target" requires this feedback loop:
  Epoch N model injects → Epoch N+1 assessor trains on those injections
  → assessor gets better → injector must inject more subtly → repeat

Usage (after each epoch):
    python3 scripts/self_play/online_datagen.py \\
        --model-path outputs/self_play/<run>/actor \\
        --medec-jsonl data_processed/medec_paired/train_val_split/rl_train.jsonl \\
        --output data_processed/self_play/train.parquet \\
        --max-pairs 100

The output parquet is in the same format as preprocess_medec.py so it is a
drop-in replacement for the static train.parquet. The next training run
uses it via SKIP_DATAGEN=1.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

from datasets import Dataset


# ---------------------------------------------------------------------------
# Helpers shared with preprocess_medec.py
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list:
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]


def _load_prompts(path: str) -> dict:
    with open(path) as f:
        return json.load(f)


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Build injector prompts (same logic as preprocess_medec.py)
# ---------------------------------------------------------------------------

def _build_injector_prompt(pair: dict, injection_prompts: dict) -> list:
    """Return a chat message list for the injector role (error mode)."""
    import random
    from scripts.self_play.utils import number_sentences

    # Prefer the canonical pre-numbered sentences (consistent with SFT/eval/live loop).
    sentences = pair.get("correct_sentences") or number_sentences(pair["correct_note"])
    system_prompt = injection_prompts["system_prompt_incorrect"]
    user_template = injection_prompts["injector_error_template"]

    error_types = injection_prompts.get("error_types", [])
    error_type = random.choice(error_types) if error_types else "treatment"

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_template.format(
            sentences=sentences,
            prompt_intent=error_type,
        )},
    ]


# ---------------------------------------------------------------------------
# Parse injector output → (sentence_id, modified_text)
# ---------------------------------------------------------------------------

def _parse_injector_output(text: str) -> tuple:
    """Extract (sentence_id: int, modified_text: str) from injector response.

    Expects compact format: '<N>. <modified sentence text>'
    Strips leading <think>...</think> blocks first.
    Returns (None, None) on parse failure.
    """
    import re
    # Strip thinking block
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Match "N. <text>"
    m = re.match(r"^(\d+)\.\s+(.+)", text, re.DOTALL)
    if m:
        return int(m.group(1)), m.group(2).strip()
    return None, None


# ---------------------------------------------------------------------------
# Apply injection to note
# ---------------------------------------------------------------------------

def _reconstruct_note(original_note: str, sentence_id: int, new_text: str) -> str:
    """Replace sentence <sentence_id> in the note with new_text."""
    from scripts.self_play.utils import number_sentences, split_sentences

    sentences = split_sentences(original_note)
    if 1 <= sentence_id <= len(sentences):
        sentences[sentence_id - 1] = new_text
    return " ".join(sentences)


# ---------------------------------------------------------------------------
# vLLM batch inference
# ---------------------------------------------------------------------------

def _generate_injections(
    model_path: str,
    prompts: list,
    *,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.95,
    gpu_memory_utilization: float = 0.5,
    tensor_parallel_size: int = 1,
) -> list:
    """Run batch generation with vLLM. Returns list of output strings."""
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("ERROR: vllm not installed. Run: pip install vllm", file=sys.stderr)
        sys.exit(1)

    print(f"  Loading model from {model_path} …")
    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=8192,
        dtype="bfloat16",
    )

    sampling = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )

    # vLLM needs tokenized chat inputs; apply chat template
    tokenizer = llm.get_tokenizer()
    formatted = []
    for msgs in prompts:
        try:
            text = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            # Fallback: join user message
            text = msgs[-1]["content"]
        formatted.append(text)

    print(f"  Generating {len(formatted)} injections …")
    outputs = llm.generate(formatted, sampling)
    return [o.outputs[0].text for o in outputs]


# ---------------------------------------------------------------------------
# Build assessor example from live injection
# ---------------------------------------------------------------------------

def _build_assessor_example(
    pair: dict,
    idx: int,
    injected_note: str,
    injected_sentence_id: int,
    detection_prompts: dict,
) -> dict:
    """Create an assessor training example from a live model injection.

    `injected_note` is already a canonical numbered string (built via
    reconstruct_note on the pair's canonical `sentences`), so use it as-is.
    """
    numbered = injected_note
    system_prompt = detection_prompts["system_prompt"]
    user_template = detection_prompts["user_template"]
    ground_truth = str(injected_sentence_id)
    note_id = pair.get("note_id", f"online-{idx}")

    return {
        "data_source": "medec_selfplay_online",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_template.format(sentences=numbered)},
        ],
        "ability": "medical_error_detection",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "role": "assessor",
            "mode": "error_injection",
            "note_id": f"{note_id}-online-assessor",
            "correct_note": pair["correct_note"],
            "injected_note": injected_note,
            "sentences": numbered,
            "error_sentence_id": injected_sentence_id,
            "source": "online_datagen",
        },
        "interaction_kwargs": {},
    }


# ---------------------------------------------------------------------------
# Fallback static assessor examples (when injection fails to parse)
# ---------------------------------------------------------------------------

def _build_static_assessor_example(
    pair: dict,
    idx: int,
    detection_prompts: dict,
    mode: str = "error",
) -> dict:
    """Fallback: use MEDEC ground-truth injection when model output is unparseable."""
    from scripts.self_play.utils import number_sentences, find_error_sentence_id

    if mode == "benign":
        # Canonical clean note sentences.
        sentences = pair.get("correct_sentences") or number_sentences(pair["correct_note"])
        ground_truth = "CORRECT"
        error_sentence_id = None
    else:
        # Canonical error note sentences + canonical error_sentence_id.
        note_text = pair.get("incorrect_note", pair["correct_note"])
        sentences = pair.get("sentences") or number_sentences(note_text)
        canonical_esid = pair.get("error_sentence_id")
        error_sentence_id = (
            canonical_esid if canonical_esid is not None
            else find_error_sentence_id(note_text, pair.get("error_sentence", ""))
        )
        ground_truth = str(error_sentence_id) if error_sentence_id else "INCORRECT"

    system_prompt = detection_prompts["system_prompt"]
    user_template = detection_prompts["user_template"]
    note_id = pair.get("note_id", f"static-{idx}")

    return {
        "data_source": "medec_selfplay",
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_template.format(sentences=sentences)},
        ],
        "ability": "medical_error_detection",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "role": "assessor",
            "mode": mode if mode == "benign" else "error_injection",
            "note_id": f"{note_id}-static-{mode}",
            "correct_note": pair["correct_note"],
            "sentences": sentences,
            "error_sentence_id": error_sentence_id,
            "source": "static_fallback",
        },
        "interaction_kwargs": {},
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    root = _project_root()

    parser = argparse.ArgumentParser(
        description="Online data regeneration: run current model as injector, "
                    "build fresh assessor examples from live injections."
    )
    parser.add_argument(
        "--model-path", required=True,
        help="HuggingFace model dir (actor checkpoint or base model). "
             "verl saves actor checkpoints under outputs/<run>/actor/."
    )
    parser.add_argument(
        "--medec-jsonl", required=True,
        help="MEDEC JSONL pairs (rl_train.jsonl or rl_val.jsonl)."
    )
    parser.add_argument(
        "--output", required=True,
        help="Output parquet path (replaces static train.parquet)."
    )
    parser.add_argument(
        "--injection-prompts",
        default=str(root / "configs/prompts/error_injection_prompts_v4.json"),
    )
    parser.add_argument(
        "--detection-prompts",
        default=str(root / "configs/prompts/detection_localization_prompts.json"),
    )
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=512,
                        help="Max tokens for injector generation.")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument(
        "--benign-fraction", type=float, default=0.3,
        help="Fraction of assessor examples to fill with static benign examples "
             "(model-injected examples are all error mode)."
    )
    args = parser.parse_args()

    sys.path.insert(0, str(root))

    pairs = _load_jsonl(Path(args.medec_jsonl))
    if args.max_pairs < len(pairs):
        pairs = pairs[:args.max_pairs]
    print(f"Loaded {len(pairs)} pairs from {args.medec_jsonl}")

    injection_prompts = _load_prompts(args.injection_prompts)
    detection_prompts = _load_prompts(args.detection_prompts)

    # Build injector prompts for all pairs
    injector_prompts = [_build_injector_prompt(p, injection_prompts) for p in pairs]

    # Run batch inference with current model
    raw_outputs = _generate_injections(
        args.model_path,
        injector_prompts,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    # Build training examples
    examples = []
    parse_ok = 0
    parse_fail = 0

    for idx, (pair, raw) in enumerate(zip(pairs, raw_outputs)):
        sid, modified_text = _parse_injector_output(raw)
        if sid is not None and modified_text:
            # Live model injection → assessor example.
            # Reconstruct on the canonical numbered sentences (consistent with the
            # live agent loop, SFT and eval) instead of re-segmenting the raw note.
            from scripts.self_play.utils import number_sentences, reconstruct_note
            canonical = pair.get("correct_sentences") or number_sentences(pair["correct_note"])
            injected_note = reconstruct_note(canonical, sid, modified_text)
            ex = _build_assessor_example(
                pair, idx, injected_note, sid, detection_prompts
            )
            examples.append(ex)
            parse_ok += 1
        else:
            # Unparseable → fall back to MEDEC static error example
            ex = _build_static_assessor_example(pair, idx, detection_prompts, "error")
            examples.append(ex)
            parse_fail += 1

    # Fill in benign examples (static — the assessor also needs non-error cases)
    n_benign = max(1, int(len(examples) * args.benign_fraction))
    import random
    benign_pairs = random.sample(pairs, min(n_benign, len(pairs)))
    for idx, pair in enumerate(benign_pairs):
        ex = _build_static_assessor_example(pair, idx, detection_prompts, "benign")
        examples.append(ex)

    random.shuffle(examples)

    print(f"\nOnline data generation complete:")
    print(f"  Live model injections (parsed OK) : {parse_ok}")
    print(f"  Static fallback (parse failed)    : {parse_fail}")
    print(f"  Benign assessor examples          : {n_benign}")
    print(f"  Total examples                    : {len(examples)}")

    # Save as parquet (same format as preprocess_medec.py)
    rows = [
        {
            "data_source": ex["data_source"],
            "prompt": ex["prompt"],
            "ability": ex["ability"],
            "reward_model": ex["reward_model"],
            "extra_info": ex["extra_info"],
            "interaction_kwargs": ex["interaction_kwargs"],
        }
        for ex in examples
    ]
    ds = Dataset.from_list(rows)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    ds.to_parquet(args.output)
    print(f"  Saved → {args.output}")


if __name__ == "__main__":
    main()
