#!/usr/bin/env python3
"""
Safe SFT Data Generation for MedSeRL

Strategy:
- Use MEDEC errors AS-IS (doctor-verified, no new error generation)
- Only generate CORRECT paraphrases with GPT-4o
- Output 50/50 CORRECT/INCORRECT for model calibration
- Optional: Add reasoning to explain WHY errors are wrong

Usage:
    python scripts/generate_sft_data_safe.py \
        --input-jsonl data_processed/medec_paired/train_val_split/sft_train.jsonl \
        --output-dir data/sft_safe_medec \
        --prompt-file configs/prompts/gpt4o_medec_safe_augmentation.json \
        --num-pairs 500 \
        --add-reasoning \
        --openai-api-key $OPENAI_API_KEY
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import time

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("Warning: openai package not installed. Install with: pip install openai")


@dataclass
class MedecPair:
    """MEDEC note pair with verified error."""
    note_id: str
    correct_note: str
    incorrect_note: str
    error_type: str
    error_sentence: Optional[str] = None
    corrected_sentence: Optional[str] = None


def load_medec_pairs(jsonl_path: str, max_pairs: Optional[int] = None) -> List[MedecPair]:
    """Load MEDEC paired notes from JSONL."""
    pairs = []

    with open(jsonl_path, 'r') as f:
        for line in f:
            record = json.loads(line.strip())

            pair = MedecPair(
                note_id=record.get("note_id", record.get("text_id", "unknown")),
                correct_note=record["correct_note"],
                incorrect_note=record["incorrect_note"],
                error_type=record.get("error_type", "unknown"),
                error_sentence=record.get("error_sentence"),
                corrected_sentence=record.get("corrected_sentence"),
            )
            pairs.append(pair)

            if max_pairs and len(pairs) >= max_pairs:
                break

    print(f"Loaded {len(pairs)} MEDEC pairs from {jsonl_path}")
    return pairs


def load_prompts(prompt_file: str) -> Dict:
    """Load prompt configuration."""
    with open(prompt_file, 'r') as f:
        return json.load(f)


def paraphrase_correct_note(
    client: OpenAI,
    correct_note: str,
    prompts: Dict,
    temperature: float = 0.7,
) -> Tuple[str, Dict]:
    """
    Generate CORRECT paraphrase using GPT-4o.

    Returns:
        (paraphrased_note, metadata)
    """
    system_prompt = prompts["correct_paraphrase_system"]
    user_prompt = prompts["correct_paraphrase_user"].format(note=correct_note)

    # Generation config
    gen_config = prompts.get("generation_params", {}).get("correct_paraphrase", {})
    temp = gen_config.get("temperature", temperature)
    top_p = gen_config.get("top_p", 0.9)
    max_tokens = gen_config.get("max_tokens", 2048)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temp,
            top_p=top_p,
            max_tokens=max_tokens,
        )

        paraphrased = response.choices[0].message.content.strip()

        metadata = {
            "model": "gpt-4o",
            "temperature": temp,
            "tokens_used": response.usage.total_tokens,
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }

        return paraphrased, metadata

    except Exception as e:
        print(f"Error generating paraphrase: {e}")
        return None, {"error": str(e)}


def generate_correct_reasoning(
    client: OpenAI,
    original_note: str,
    paraphrased_note: str,
    prompts: Dict,
) -> Optional[Dict]:
    """
    Generate reasoning explaining why the paraphrase is SAFE.

    Teaches model what changes preserve medical accuracy.
    """
    if not prompts.get("correct_reasoning_system"):
        return None

    system_prompt = prompts["correct_reasoning_system"]
    user_prompt = prompts["correct_reasoning_user"].format(
        original_note=original_note,
        paraphrased_note=paraphrased_note,
    )

    gen_config = prompts.get("generation_params", {}).get("reasoning", {})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=gen_config.get("temperature", 0.3),
            top_p=gen_config.get("top_p", 0.9),
            max_tokens=gen_config.get("max_tokens", 1024),
            response_format={"type": "json_object"},
        )

        reasoning = json.loads(response.choices[0].message.content)
        return reasoning

    except Exception as e:
        print(f"Error generating correct reasoning: {e}")
        return None


def generate_error_reasoning(
    client: OpenAI,
    pair: MedecPair,
    prompts: Dict,
) -> Optional[Dict]:
    """
    Generate INJECTOR reasoning explaining HOW and WHY to make this error.

    This teaches the model the injection strategy (plausible but wrong).
    """
    if not prompts.get("reasoning_system"):
        return None

    system_prompt = prompts["reasoning_system"]
    user_prompt = prompts["reasoning_user"].format(
        correct_note=pair.correct_note,
        incorrect_note=pair.incorrect_note,
        error_sentence=pair.error_sentence or "N/A",
        corrected_sentence=pair.corrected_sentence or "N/A",
        error_type=pair.error_type,
    )

    gen_config = prompts.get("generation_params", {}).get("reasoning", {})
    temp = gen_config.get("temperature", 0.3)
    top_p = gen_config.get("top_p", 0.9)
    max_tokens = gen_config.get("max_tokens", 1024)

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temp,
            top_p=top_p,
            max_tokens=max_tokens,
            response_format={"type": "json_object"},
        )

        reasoning = json.loads(response.choices[0].message.content)
        return reasoning

    except Exception as e:
        print(f"Error generating reasoning: {e}")
        return None


def generate_safe_sft_data(
    pairs: List[MedecPair],
    client: OpenAI,
    prompts: Dict,
    output_dir: str,
    add_reasoning: bool = True,
    batch_size: int = 10,
    save_every: int = 5,
) -> Dict:
    """
    Generate safe SFT training data with batch processing and incremental saves.

    For each MEDEC pair:
    1. CORRECT: Generate paraphrase with GPT-4o
    2. INCORRECT: Use MEDEC incorrect_note AS-IS (doctor-verified)
    3. Optional: Add reasoning explaining the error

    Features:
    - Batch processing (configurable batch size)
    - Incremental saves every N pairs (prevents data loss)
    - Full path resolution (can run from anywhere)
    - Progress tracking with stats

    Output: 50% CORRECT, 50% INCORRECT
    """
    # Resolve full paths
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    correct_output = os.path.join(output_dir, "sft_correct.jsonl")
    incorrect_output = os.path.join(output_dir, "sft_incorrect.jsonl")
    combined_output = os.path.join(output_dir, "sft_combined_safe.jsonl")
    checkpoint_file = os.path.join(output_dir, "checkpoint.json")

    # Load checkpoint if exists (resume from interruption)
    start_idx = 0
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file, 'r') as f:
            checkpoint = json.load(f)
            start_idx = checkpoint.get("last_completed_idx", 0) + 1
            print(f"Resuming from pair {start_idx} (checkpoint found)")

    stats = {
        "total_pairs": len(pairs),
        "correct_generated": 0,
        "incorrect_used": 0,
        "correct_failed": 0,
        "correct_reasoning_added": 0,
        "correct_reasoning_failed": 0,
        "incorrect_reasoning_added": 0,
        "incorrect_reasoning_failed": 0,
        "total_tokens_used": 0,
        "batches_completed": 0,
    }

    # Open files in append mode for incremental writes
    mode = 'a' if start_idx > 0 else 'w'
    with open(correct_output, mode) as f_correct, \
         open(incorrect_output, mode) as f_incorrect, \
         open(combined_output, mode) as f_combined:

        # Process pairs with batch saving
        for idx, pair in enumerate(tqdm(pairs[start_idx:], desc="Generating SFT data", initial=start_idx, total=len(pairs))):
            actual_idx = start_idx + idx

            try:
                # ============================================================
                # CORRECT: Generate paraphrase with GPT-4o
                # ============================================================
                paraphrased, meta = paraphrase_correct_note(
                    client, pair.correct_note, prompts
                )

                if paraphrased:
                    # Generate reasoning for CORRECT (why paraphrase is safe)
                    correct_reasoning = None
                    if add_reasoning:
                        correct_reasoning = generate_correct_reasoning(
                            client, pair.correct_note, paraphrased, prompts
                        )
                        if correct_reasoning:
                            stats["correct_reasoning_added"] += 1
                        else:
                            stats["correct_reasoning_failed"] += 1

                    correct_example = {
                        "note": paraphrased,
                        "label": "CORRECT",
                        "original_note_id": pair.note_id,
                        "source": "medec_correct_paraphrased",
                        "metadata": {
                            "generation_type": "correct_paraphrase",
                            "gpt4o_tokens": meta.get("tokens_used", 0),
                        },
                        "reasoning": correct_reasoning,
                    }

                    f_correct.write(json.dumps(correct_example) + "\n")
                    f_combined.write(json.dumps(correct_example) + "\n")

                    stats["correct_generated"] += 1
                    stats["total_tokens_used"] += meta.get("tokens_used", 0)
                else:
                    stats["correct_failed"] += 1
                    print(f"Failed to generate CORRECT paraphrase for {pair.note_id}")

                # ============================================================
                # INCORRECT: Use MEDEC AS-IS (doctor-verified)
                # ============================================================
                incorrect_example = {
                    "note": pair.incorrect_note,  # AS-IS from MEDEC!
                    "label": "INCORRECT",
                    "original_note_id": pair.note_id,
                    "error_type": pair.error_type,
                    "error_location": pair.error_sentence,
                    "correction": pair.corrected_sentence,
                    "source": "medec_incorrect_verified",
                    "metadata": {
                        "generation_type": "medec_verified_error",
                    },
                    "reasoning": None,  # Will add below if requested
                }

                # Optional: Add reasoning (injector strategy - plausible but wrong)
                if add_reasoning:
                    reasoning = generate_error_reasoning(client, pair, prompts)
                    if reasoning:
                        incorrect_example["reasoning"] = reasoning
                        stats["incorrect_reasoning_added"] += 1
                    else:
                        stats["incorrect_reasoning_failed"] += 1

                f_incorrect.write(json.dumps(incorrect_example) + "\n")
                f_combined.write(json.dumps(incorrect_example) + "\n")

                stats["incorrect_used"] += 1

                # Incremental save checkpoint every N pairs
                if (actual_idx + 1) % save_every == 0:
                    # Flush file buffers
                    f_correct.flush()
                    f_incorrect.flush()
                    f_combined.flush()

                    # Save checkpoint
                    checkpoint = {
                        "last_completed_idx": actual_idx,
                        "stats": stats,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    }
                    with open(checkpoint_file, 'w') as f_ckpt:
                        json.dump(checkpoint, f_ckpt, indent=2)

                    print(f"\n✓ Checkpoint saved at pair {actual_idx + 1}/{len(pairs)}")
                    stats["batches_completed"] += 1

                # Rate limiting (GPT-4o: ~10k RPM, so sleep briefly)
                time.sleep(0.1)

            except Exception as e:
                print(f"\n❌ Error processing pair {actual_idx} ({pair.note_id}): {e}")
                print(f"Saving checkpoint and continuing...")

                # Save checkpoint on error
                checkpoint = {
                    "last_completed_idx": actual_idx - 1,  # Last successful
                    "stats": stats,
                    "error": str(e),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                with open(checkpoint_file, 'w') as f_ckpt:
                    json.dump(checkpoint, f_ckpt, indent=2)

                # Continue with next pair
                continue

    # Final flush and checkpoint cleanup
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print("\n✓ Checkpoint file removed (generation complete)")

    # Save statistics
    stats_file = os.path.join(output_dir, "generation_stats_safe.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "="*60)
    print("Safe SFT Data Generation Complete")
    print("="*60)
    print(f"Total pairs processed: {stats['total_pairs']}")
    print(f"CORRECT paraphrases generated: {stats['correct_generated']}")
    print(f"INCORRECT notes used (AS-IS): {stats['incorrect_used']}")
    print(f"CORRECT generation failures: {stats['correct_failed']}")
    if add_reasoning:
        print(f"CORRECT reasoning added: {stats['correct_reasoning_added']}")
        print(f"CORRECT reasoning failures: {stats['correct_reasoning_failed']}")
        print(f"INCORRECT reasoning added: {stats['incorrect_reasoning_added']}")
        print(f"INCORRECT reasoning failures: {stats['incorrect_reasoning_failed']}")
    print(f"Total GPT-4o tokens used: {stats['total_tokens_used']:,}")
    print(f"Batches saved: {stats['batches_completed']}")
    print(f"\nOutput files (full paths):")
    print(f"  - CORRECT only: {correct_output}")
    print(f"  - INCORRECT only: {incorrect_output}")
    print(f"  - Combined (50/50): {combined_output}")
    print(f"  - Statistics: {stats_file}")
    print("="*60)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate safe SFT training data from MEDEC pairs"
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        required=True,
        help="Path to MEDEC paired JSONL (e.g., sft_train.jsonl)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for generated data",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        required=True,
        help="Path to prompt configuration JSON",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=None,
        help="Number of pairs to process (default: all)",
    )
    parser.add_argument(
        "--add-reasoning",
        action="store_true",
        help="Add clinical reasoning to INCORRECT examples",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing (default: 10)",
    )
    parser.add_argument(
        "--save-every",
        type=int,
        default=5,
        help="Save checkpoint every N pairs (default: 5)",
    )

    args = parser.parse_args()

    # Check OpenAI
    if not HAS_OPENAI:
        print("Error: openai package not installed")
        print("Install with: pip install openai")
        return 1

    # API key
    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key required")
        print("Provide via --openai-api-key or OPENAI_API_KEY env var")
        return 1

    client = OpenAI(api_key=api_key)

    # Resolve full paths
    input_jsonl = os.path.abspath(args.input_jsonl)
    output_dir = os.path.abspath(args.output_dir)
    prompt_file = os.path.abspath(args.prompt_file)

    print(f"Input file: {input_jsonl}")
    print(f"Output directory: {output_dir}")
    print(f"Prompt file: {prompt_file}")

    # Load data
    print("\nLoading MEDEC pairs...")
    pairs = load_medec_pairs(input_jsonl, max_pairs=args.num_pairs)

    print("Loading prompts...")
    prompts = load_prompts(prompt_file)

    # Generate
    print("\nGenerating safe SFT data...")
    print(f"Strategy: Use MEDEC errors AS-IS, generate CORRECT paraphrases")
    print(f"Output: 50/50 CORRECT/INCORRECT split")
    if args.add_reasoning:
        print(f"Adding clinical reasoning to INCORRECT examples")
    print()

    stats = generate_safe_sft_data(
        pairs=pairs,
        client=client,
        prompts=prompts,
        output_dir=output_dir,
        add_reasoning=args.add_reasoning,
        batch_size=args.batch_size,
        save_every=args.save_every,
    )

    return 0


if __name__ == "__main__":
    exit(main())
