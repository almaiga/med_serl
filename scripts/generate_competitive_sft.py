#!/usr/bin/env python3
"""
Competitive SFT Data Generation for Self-Play Training

Strategy:
- CORRECT: Heavily paraphrase notes (prevent memorization)
- INCORRECT: Paraphrase + apply error VARIATION based on MEDEC pattern
- Goal: Keep self-play competitive, prevent degeneracy

Key Insight:
By paraphrasing BOTH correct and incorrect notes, plus varying the
specific error, we force the model to learn error PATTERNS rather than
memorizing specific note text.
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
    print("Warning: openai package not installed")


@dataclass
class MedecPair:
    """MEDEC note pair with verified error pattern."""
    note_id: str
    correct_note: str
    incorrect_note: str
    error_type: str
    error_sentence: Optional[str] = None
    corrected_sentence: Optional[str] = None


def load_medec_pairs(jsonl_path: str, max_pairs: Optional[int] = None) -> List[MedecPair]:
    """Load MEDEC paired notes."""
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


def generate_correct_paraphrase(
    client: OpenAI,
    correct_note: str,
    prompts: Dict,
) -> Tuple[str, Dict]:
    """
    Generate heavily paraphrased CORRECT note.

    Goal: 30-50% word changes to prevent memorization.
    """
    system_prompt = prompts["correct_paraphrase_system"]
    user_prompt = prompts["correct_paraphrase_user"].format(note=correct_note)

    gen_config = prompts.get("generation_config", {}).get("correct_paraphrase", {})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=gen_config.get("temperature", 0.8),
            top_p=gen_config.get("top_p", 0.95),
            max_tokens=gen_config.get("max_tokens", 2048),
            presence_penalty=gen_config.get("presence_penalty", 0.4),
            frequency_penalty=gen_config.get("frequency_penalty", 0.4),
        )

        paraphrased = response.choices[0].message.content.strip()

        metadata = {
            "model": "gpt-4o",
            "tokens_used": response.usage.total_tokens,
        }

        return paraphrased, metadata

    except Exception as e:
        print(f"Error generating correct paraphrase: {e}")
        return None, {"error": str(e)}


def generate_competitive_error(
    client: OpenAI,
    pair: MedecPair,
    prompts: Dict,
) -> Tuple[Optional[str], Optional[Dict], Dict]:
    """
    Generate competitive error by:
    1. Paraphrasing the correct note
    2. Applying error VARIATION based on MEDEC pattern

    Returns:
        (paraphrased_note_with_error, error_metadata, generation_metadata)
    """
    system_prompt = prompts["injector_system_competitive"]

    # Extract error pattern description from MEDEC
    error_pattern_description = infer_error_pattern(pair)

    user_prompt = prompts["injector_user_competitive"].format(
        correct_note=pair.correct_note,
        error_type=pair.error_type,
        error_location=pair.error_sentence or "N/A",
        correction=pair.corrected_sentence or "N/A",
        error_pattern_description=error_pattern_description,
    )

    gen_config = prompts.get("generation_config", {}).get("competitive_error_injection", {})

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=gen_config.get("temperature", 0.9),
            top_p=gen_config.get("top_p", 0.95),
            max_tokens=gen_config.get("max_tokens", 2048),
            presence_penalty=gen_config.get("presence_penalty", 0.5),
            frequency_penalty=gen_config.get("frequency_penalty", 0.5),
            response_format={"type": "json_object"},
        )

        result = json.loads(response.choices[0].message.content)

        paraphrased_note_with_error = result.get("paraphrased_note_with_error")
        error_metadata = result.get("error_applied")

        generation_metadata = {
            "model": "gpt-4o",
            "tokens_used": response.usage.total_tokens,
        }

        return paraphrased_note_with_error, error_metadata, generation_metadata

    except Exception as e:
        print(f"Error generating competitive error: {e}")
        return None, None, {"error": str(e)}


def infer_error_pattern(pair: MedecPair) -> str:
    """
    Infer error pattern description from MEDEC pair.

    This helps GPT-4o understand what type of variation to apply.
    """
    error_type = pair.error_type

    # Simple heuristic based on error type
    patterns = {
        "diagnosis": "Similar clinical presentations requiring differentiation",
        "medication": "Same drug class or similar indication",
        "treatment": "Similar therapeutic approaches",
        "management": "Alternative management strategies",
        "laboratory": "Similar measurement error pattern (transposition, decimal, unit confusion)",
    }

    return patterns.get(error_type, "Similar error pattern")


def generate_competitive_sft_data(
    pairs: List[MedecPair],
    client: OpenAI,
    prompts: Dict,
    output_dir: str,
) -> Dict:
    """
    Generate competitive SFT data for self-play training.

    For each MEDEC pair:
    1. CORRECT: Heavily paraphrase correct_note
    2. INCORRECT: Paraphrase + apply error variation

    Output: 50% CORRECT (paraphrased), 50% INCORRECT (paraphrased + error)
    """
    os.makedirs(output_dir, exist_ok=True)

    correct_output = os.path.join(output_dir, "sft_correct_competitive.jsonl")
    incorrect_output = os.path.join(output_dir, "sft_incorrect_competitive.jsonl")
    combined_output = os.path.join(output_dir, "sft_combined_competitive.jsonl")

    stats = {
        "total_pairs": len(pairs),
        "correct_generated": 0,
        "incorrect_generated": 0,
        "correct_failed": 0,
        "incorrect_failed": 0,
        "total_tokens_used": 0,
    }

    with open(correct_output, 'w') as f_correct, \
         open(incorrect_output, 'w') as f_incorrect, \
         open(combined_output, 'w') as f_combined:

        for pair in tqdm(pairs, desc="Generating competitive SFT data"):

            # ============================================================
            # CORRECT: Heavy paraphrase (prevent memorization)
            # ============================================================
            paraphrased_correct, meta_correct = generate_correct_paraphrase(
                client, pair.correct_note, prompts
            )

            if paraphrased_correct:
                correct_example = {
                    "note": paraphrased_correct,
                    "label": "CORRECT",
                    "original_note_id": pair.note_id,
                    "source": "competitive_correct_paraphrased",
                    "metadata": {
                        "generation_type": "competitive_correct_paraphrase",
                        "gpt4o_tokens": meta_correct.get("tokens_used", 0),
                        "paraphrase_strategy": "30-50% word changes to prevent memorization",
                    },
                }

                f_correct.write(json.dumps(correct_example) + "\n")
                f_combined.write(json.dumps(correct_example) + "\n")

                stats["correct_generated"] += 1
                stats["total_tokens_used"] += meta_correct.get("tokens_used", 0)
            else:
                stats["correct_failed"] += 1
                print(f"Failed CORRECT for {pair.note_id}")

            # ============================================================
            # INCORRECT: Paraphrase + error variation (competitive)
            # ============================================================
            paraphrased_with_error, error_meta, gen_meta = generate_competitive_error(
                client, pair, prompts
            )

            if paraphrased_with_error and error_meta:
                incorrect_example = {
                    "note": paraphrased_with_error,
                    "label": "INCORRECT",
                    "original_note_id": pair.note_id,
                    "source": "competitive_error_variation",
                    "metadata": {
                        "generation_type": "competitive_error_injection",
                        "gpt4o_tokens": gen_meta.get("tokens_used", 0),
                        "medec_error_type": pair.error_type,
                        "medec_error_location": pair.error_sentence,
                        "medec_correction": pair.corrected_sentence,
                        "error_variation_applied": error_meta,
                    },
                }

                f_incorrect.write(json.dumps(incorrect_example) + "\n")
                f_combined.write(json.dumps(incorrect_example) + "\n")

                stats["incorrect_generated"] += 1
                stats["total_tokens_used"] += gen_meta.get("tokens_used", 0)
            else:
                stats["incorrect_failed"] += 1
                print(f"Failed INCORRECT for {pair.note_id}")

            # Rate limiting
            time.sleep(0.1)

    # Save statistics
    stats_file = os.path.join(output_dir, "generation_stats_competitive.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print("\n" + "="*60)
    print("Competitive SFT Data Generation Complete")
    print("="*60)
    print(f"Total pairs processed: {stats['total_pairs']}")
    print(f"CORRECT paraphrases: {stats['correct_generated']}")
    print(f"INCORRECT with error variations: {stats['incorrect_generated']}")
    print(f"Failures: {stats['correct_failed']} CORRECT, {stats['incorrect_failed']} INCORRECT")
    print(f"Total GPT-4o tokens: {stats['total_tokens_used']:,}")
    print(f"\nOutput files:")
    print(f"  - CORRECT: {correct_output}")
    print(f"  - INCORRECT: {incorrect_output}")
    print(f"  - Combined (50/50): {combined_output}")
    print(f"  - Statistics: {stats_file}")
    print("="*60)

    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate competitive SFT data for self-play training"
    )
    parser.add_argument(
        "--input-jsonl",
        type=str,
        required=True,
        help="Path to MEDEC paired JSONL",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory",
    )
    parser.add_argument(
        "--prompt-file",
        type=str,
        required=True,
        help="Path to competitive prompt configuration",
    )
    parser.add_argument(
        "--num-pairs",
        type=int,
        default=None,
        help="Number of pairs to process",
    )
    parser.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="OpenAI API key",
    )

    args = parser.parse_args()

    # Check dependencies
    if not HAS_OPENAI:
        print("Error: openai package not installed")
        return 1

    # API key
    api_key = args.openai_api_key or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key required")
        return 1

    client = OpenAI(api_key=api_key)

    # Load data and prompts
    print("Loading MEDEC pairs...")
    pairs = load_medec_pairs(args.input_jsonl, max_pairs=args.num_pairs)

    print("Loading competitive prompts...")
    prompts = load_prompts(args.prompt_file)

    # Generate
    print("\nGenerating competitive SFT data...")
    print("Strategy: Heavy paraphrasing + error variations")
    print("Goal: Prevent memorization, keep self-play competitive\n")

    stats = generate_competitive_sft_data(
        pairs=pairs,
        client=client,
        prompts=prompts,
        output_dir=args.output_dir,
    )

    return 0


if __name__ == "__main__":
    exit(main())
