"""Preprocess MEDEC data for self-play training with verl.

Converts JSONL data to verl-compatible format.
For each note pair, generates up to 4 examples (controlled by --roles flag):
  injector role (2 per pair):
    1. Benign mode (ground_truth="CORRECT") - model makes a surface edit
    2. Error mode (ground_truth=str(sentence_id)) - model injects an error
  assessor role (2 per pair):
    3. Benign assessor (ground_truth="CORRECT") - model detects no error
    4. Error assessor (ground_truth=str(sentence_id)) - model localizes error

Notes are pre-numbered (1-indexed) so the assessor can output a sentence number.
Error sentence IDs are computed by matching error_sentence text against the note.

Prompts are loaded from JSON config files at preprocessing time.

CRITICAL: verl's RLHFDataset expects 'prompt' as a native list of message dicts,
NOT a JSON-encoded string. We use HuggingFace datasets for proper serialization.
"""

import json
import random
import argparse
from pathlib import Path
from typing import Any

from scripts.self_play.utils import (
    number_sentences,
    find_error_sentence_id,
)


def load_jsonl(filepath: Path) -> list[dict]:
    """Load JSONL file."""
    data = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))
    return data


def load_prompts(prompt_file: str) -> dict:
    """Load prompts from JSON config file."""
    with open(prompt_file, 'r') as f:
        return json.load(f)


def create_benign_example(
    pair: dict,
    injection_prompts: dict,
    idx: int,
    data_source: str = "medec_selfplay",
) -> dict:
    """Create a benign modification example (ground_truth = CORRECT).
    
    Injector receives pre-numbered sentences and makes surface edits.
    A random benign change type is selected from the prompt config.
    """
    system_prompt = injection_prompts["system_prompt_correct"]
    user_template = injection_prompts["injector_correct_template"]

    # Pre-number the sentences
    sentences = number_sentences(pair["correct_note"])

    # Pick a random benign change type
    change_types = injection_prompts.get("benign_change_types", {})
    change_type = random.choice(list(change_types.keys())) if change_types else "pseudo_factual"
    change_desc = change_types.get(change_type, "Replace a medical term with an equivalent synonym.")

    user_prompt = user_template.format(
        sentences=sentences,
        change_type=change_type,
        change_type_description=change_desc,
    )
    
    note_id = pair.get("note_id", f"selfplay-{idx}")
    
    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "ability": "medical_error_detection",
        "reward_model": {
            "style": "rule",
            "ground_truth": "CORRECT"
        },
        "extra_info": {
            "role": "injector",
            "note_id": f"{note_id}-benign",
            "correct_note": pair["correct_note"],
            "sentences": sentences,
            "incorrect_note": "",
            "error_type": "",
            "error_sentence": "",
            "error_sentence_id": None,
            "corrected_sentence": "",
            "change_type": change_type,
            "mode": "benign",
        },
        "interaction_kwargs": {
            "name": "medical_game",
            "ground_truth": "CORRECT",
            "mode": "benign",
            "note_id": f"{note_id}-benign",
            "sentences": sentences,
        }
    }


def create_error_example(
    pair: dict,
    injection_prompts: dict,
    idx: int,
    data_source: str = "medec_selfplay",
) -> dict:
    """Create an error injection example (ground_truth = sentence number).
    
    Injector receives pre-numbered sentences and error_type to guide injection.
    Ground truth is the 1-indexed sentence number containing the error.
    """
    system_prompt = injection_prompts["system_prompt_incorrect"]
    user_template = injection_prompts["injector_incorrect_template"]
    error_type = pair.get("error_type", "clinical error")

    # Pre-number the sentences (use incorrect_note so error_sentence can be located)
    sentences = number_sentences(pair["incorrect_note"])

    # Find error sentence ID by matching error_sentence text
    error_sentence_text = pair.get("error_sentence", "")
    error_sentence_id = find_error_sentence_id(pair["incorrect_note"], error_sentence_text)
    
    # Ground truth is the sentence number (as string), not "INCORRECT"
    ground_truth = str(error_sentence_id) if error_sentence_id else "INCORRECT"

    user_prompt = user_template.format(
        sentences=sentences,
        prompt_intent=error_type,
    )
    
    note_id = pair.get("note_id", f"selfplay-{idx}")
    
    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        "ability": "medical_error_detection",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "role": "injector",
            "note_id": f"{note_id}-error",
            "correct_note": pair["correct_note"],
            "incorrect_note": pair["incorrect_note"],
            "sentences": sentences,
            "error_type": pair.get("error_type", ""),
            "error_sentence": pair.get("error_sentence", ""),
            "error_sentence_id": error_sentence_id,
            "corrected_sentence": pair.get("corrected_sentence", ""),
            "mode": "error_injection",
        },
        "interaction_kwargs": {
            "name": "medical_game",
            "ground_truth": ground_truth,
            "mode": "error_injection",
            "note_id": f"{note_id}-error",
            "sentences": sentences,
            "error_type": error_type,
            "error_sentence_id": error_sentence_id,
        }
    }


def create_assessor_example(
    pair: dict,
    detection_prompts: dict,
    idx: int,
    mode: str,
    data_source: str = "medec_selfplay",
) -> dict:
    """Create an assessor-role example using ground-truth MEDEC notes.

    Args:
        mode: "benign" → correct_note, ground_truth="CORRECT"
              "error"  → incorrect_note, ground_truth=str(sentence_id)
    """
    system_prompt = detection_prompts["system_prompt"]
    user_template = detection_prompts["user_template"]

    if mode == "benign":
        note_text = pair["correct_note"]
        sentences = number_sentences(note_text)
        ground_truth = "CORRECT"
        error_sentence_id = None
        note_suffix = "benign"
    else:
        note_text = pair["incorrect_note"]
        sentences = number_sentences(note_text)
        error_sentence_text = pair.get("error_sentence", "")
        error_sentence_id = find_error_sentence_id(note_text, error_sentence_text)
        ground_truth = str(error_sentence_id) if error_sentence_id else "INCORRECT"
        note_suffix = "error"

    user_prompt = user_template.format(sentences=sentences)
    note_id = pair.get("note_id", f"selfplay-{idx}")

    return {
        "data_source": data_source,
        "prompt": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "ability": "medical_error_detection",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": {
            "role": "assessor",
            "note_id": f"{note_id}-assessor-{note_suffix}",
            "correct_note": pair["correct_note"],
            "incorrect_note": pair.get("incorrect_note", ""),
            "sentences": sentences,
            "error_type": pair.get("error_type", ""),
            "error_sentence": pair.get("error_sentence", ""),
            "error_sentence_id": error_sentence_id,
            "corrected_sentence": pair.get("corrected_sentence", ""),
            "mode": mode,
        },
        "interaction_kwargs": {},
    }


def convert_to_parquet(
    input_path: Path,
    output_path: Path,
    injection_prompts_path: str = "configs/prompts/error_injection_prompts_v4.json",
    detection_prompts_path: str = "configs/prompts/detection_localization_prompts.json",
    data_source: str = "medec_selfplay",
    max_pairs: int = None,
    roles: str = "mixed",
) -> None:
    """Convert JSONL to Parquet format for verl.

    Args:
        roles: Which role examples to generate:
            "injector" — 2 per pair (benign + error injector prompts)
            "assessor" — 2 per pair (benign + error detection prompts)
            "mixed"    — 4 per pair (all of the above)  [default]
        max_pairs: If set, limit to this many pairs.

    IMPORTANT: verl expects 'prompt' as a native list of dicts, NOT a JSON string.
    See: https://github.com/volcengine/verl/blob/main/verl/utils/dataset/rl_dataset.py
    """
    pairs = load_jsonl(input_path)
    print(f"Loaded {len(pairs)} pairs from {input_path}")

    if max_pairs is not None and max_pairs < len(pairs):
        pairs = pairs[:max_pairs]
        examples_per_pair = {"injector": 2, "assessor": 2, "mixed": 4}[roles]
        print(f"Limited to {len(pairs)} pairs (will generate ~{len(pairs) * examples_per_pair} examples)")

    injection_prompts = load_prompts(injection_prompts_path)
    print(f"Loaded injection prompts from {injection_prompts_path}")

    detection_prompts = None
    if roles in ("assessor", "mixed"):
        detection_prompts = load_prompts(detection_prompts_path)
        print(f"Loaded detection prompts from {detection_prompts_path}")

    verl_examples = []
    missing_sid_count = 0
    for idx, pair in enumerate(pairs):
        if not pair.get("correct_note") or not pair.get("incorrect_note"):
            print(f"Warning: Skipping pair {idx} - missing required fields")
            continue

        if roles in ("injector", "mixed"):
            benign_ex = create_benign_example(pair, injection_prompts, idx, data_source)
            verl_examples.append(benign_ex)
            error_ex = create_error_example(pair, injection_prompts, idx, data_source)
            verl_examples.append(error_ex)
            if error_ex["extra_info"]["error_sentence_id"] is None:
                missing_sid_count += 1

        if roles in ("assessor", "mixed"):
            assessor_benign = create_assessor_example(pair, detection_prompts, idx, "benign", data_source)
            verl_examples.append(assessor_benign)
            assessor_error = create_assessor_example(pair, detection_prompts, idx, "error", data_source)
            verl_examples.append(assessor_error)
            if assessor_error["extra_info"]["error_sentence_id"] is None:
                missing_sid_count += 1

    role_counts = {r: sum(1 for e in verl_examples if e["extra_info"].get("role") == r)
                   for r in ("injector", "assessor")}
    print(f"Generated {len(verl_examples)} examples "
          f"(injector: {role_counts['injector']}, assessor: {role_counts['assessor']})")
    if missing_sid_count:
        print(f"Warning: {missing_sid_count} error examples could not resolve error_sentence_id")
    
    # Use HuggingFace datasets for proper serialization
    from datasets import Dataset
    
    rows = []
    for ex in verl_examples:
        rows.append({
            "data_source": ex["data_source"],
            "prompt": ex["prompt"],
            "ability": ex["ability"],
            "reward_model": ex["reward_model"],
            "extra_info": ex["extra_info"],
            "interaction_kwargs": ex["interaction_kwargs"],
        })
    
    hf_dataset = Dataset.from_list(rows)
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    hf_dataset.to_parquet(str(output_path))
    print(f"Saved {len(rows)} examples to {output_path}")
    
    # Print sample for verification
    if verl_examples:
        for sample in verl_examples[:4]:
            role = sample["extra_info"].get("role", "?")
            mode = sample["extra_info"].get("mode", "?")
            gt = sample["reward_model"]["ground_truth"]
            sid = sample["extra_info"].get("error_sentence_id")
            user_content = sample["prompt"][1]["content"]
            print(f"\n--- role={role} mode={mode} gt={gt} sid={sid} ---")
            print(f"prompt preview: {user_content[:300]}...")
    
    # Verify the saved file
    print("\n--- Verifying saved parquet ---")
    loaded = Dataset.from_parquet(str(output_path))
    print(f"Loaded {len(loaded)} examples")
    first_row = loaded[0]
    print(f"prompt type: {type(first_row['prompt'])}")
    print(f"prompt[0]: {first_row['prompt'][0] if isinstance(first_row['prompt'], list) else 'NOT A LIST!'}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MEDEC data for self-play training"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data_processed/medec_paired/train_val_split/rl_train.jsonl"),
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data_processed/self_play/train.parquet"),
        help="Output Parquet file path",
    )
    parser.add_argument(
        "--injection-prompts",
        type=str,
        default="configs/prompts/error_injection_prompts_v4.json",
        help="Path to injection prompts JSON",
    )
    parser.add_argument(
        "--detection-prompts",
        type=str,
        default="configs/prompts/detection_localization_prompts.json",
        help="Path to detection/assessor prompts JSON",
    )
    parser.add_argument(
        "--data-source",
        type=str,
        default="medec_selfplay",
        help="Data source identifier",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Max number of pairs to process",
    )
    parser.add_argument(
        "--roles",
        type=str,
        default="mixed",
        choices=["injector", "assessor", "mixed"],
        help="Which role examples to generate: injector (2/pair), assessor (2/pair), mixed (4/pair)",
    )

    args = parser.parse_args()
    convert_to_parquet(
        args.input,
        args.output,
        args.injection_prompts,
        args.detection_prompts,
        args.data_source,
        args.max_pairs,
        args.roles,
    )


if __name__ == "__main__":
    main()
