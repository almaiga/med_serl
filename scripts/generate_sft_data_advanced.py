#!/usr/bin/env python3
"""
Advanced SFT data generation using GPT-4o's full capabilities.

Features:
- Complex paraphrasing (30-50% word changes)
- Diverse error scenarios (15+ error types)
- Optional complexity enhancement
- Multi-model verification (GPT-4o + Claude)
- Rich metadata for analysis

Usage:
    python scripts/generate_sft_data_advanced.py \
        --input-jsonl data/medec_train.jsonl \
        --output-dir data/sft_training_advanced \
        --num-correct 1000 \
        --num-incorrect 1000 \
        --enhance-complexity \
        --openai-api-key $OPENAI_API_KEY \
        --anthropic-api-key $ANTHROPIC_API_KEY
"""

import argparse
import json
import os
import random
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional
from tqdm import tqdm

try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


@dataclass
class GenerationResult:
    """Result from generation attempt."""
    success: bool
    generated_note: Optional[str]
    metadata: Dict
    verification: Optional[Dict]
    error: Optional[str]


def load_prompts(prompt_file: str) -> Dict:
    """Load GPT-4o advanced prompts."""
    with open(prompt_file, 'r') as f:
        return json.load(f)


def call_gpt4o_json(
    user_prompt: str,
    system_prompt: str,
    client: OpenAI,
    temperature: float = 0.7,
    max_tokens: int = 3000,
) -> Dict:
    """
    Call GPT-4o with JSON mode for structured output.
    """
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        response_format={"type": "json_object"}
    )

    return json.loads(response.choices[0].message.content)


def call_gpt4o_text(
    user_prompt: str,
    system_prompt: str,
    client: OpenAI,
    temperature: float = 0.7,
    max_tokens: int = 3000,
) -> str:
    """Call GPT-4o for text output."""
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )

    return response.choices[0].message.content


def enhance_complexity(
    note: str,
    prompts: Dict,
    client: OpenAI,
) -> str:
    """Enhance note with additional clinical complexity."""
    system_prompt = prompts["complexity_enhancement_system"]
    user_prompt = prompts["complexity_enhancement_user_template"].format(note=note)

    enhanced = call_gpt4o_text(user_prompt, system_prompt, client, temperature=0.6)
    return enhanced.strip()


def generate_correct_paraphrase(
    original_note: str,
    prompts: Dict,
    client: OpenAI,
    enhance: bool = False,
) -> GenerationResult:
    """
    Generate ambitious paraphrase with GPT-4o.

    Returns GenerationResult with:
    - generated_note: Paraphrased text
    - metadata: Empty for correct (no error info)
    """
    try:
        # Optional: Enhance complexity first
        if enhance:
            original_note = enhance_complexity(original_note, prompts, client)

        system_prompt = prompts["correct_paraphrase_system"]
        user_prompt = prompts["correct_paraphrase_user_template"].format(note=original_note)

        generated_note = call_gpt4o_text(
            user_prompt,
            system_prompt,
            client,
            temperature=0.8,  # Higher temp for diversity
            max_tokens=3000,
        )

        return GenerationResult(
            success=True,
            generated_note=generated_note.strip(),
            metadata={
                "generation_type": "correct_paraphrase",
                "enhanced": enhance,
            },
            verification=None,
            error=None,
        )

    except Exception as e:
        return GenerationResult(
            success=False,
            generated_note=None,
            metadata={},
            verification=None,
            error=str(e),
        )


def generate_incorrect_with_error(
    original_note: str,
    error_category: str,
    prompts: Dict,
    client: OpenAI,
    enhance: bool = False,
) -> GenerationResult:
    """
    Generate note with sophisticated error injection.

    Args:
        original_note: Original clinical note
        error_category: One of the error categories (random, medication, laboratory, etc.)
        prompts: Loaded prompts dictionary
        client: OpenAI client
        enhance: Whether to enhance complexity first

    Returns GenerationResult with:
    - generated_note: Note with error
    - metadata: Rich error metadata (category, severity, impact, etc.)
    """
    try:
        # Optional: Enhance complexity first
        if enhance:
            original_note = enhance_complexity(original_note, prompts, client)

        # Get error guidance
        error_guidance = prompts["error_type_guidance"].get(error_category, "Choose any realistic error.")

        system_prompt = prompts["incorrect_injection_system"]
        user_prompt = prompts["incorrect_injection_user_template"].format(
            note=original_note,
            error_guidance=error_guidance,
        )

        # Generate with JSON mode for structured output
        response = call_gpt4o_json(
            user_prompt,
            system_prompt,
            client,
            temperature=0.9,  # High temp for diverse errors
            max_tokens=3000,
        )

        return GenerationResult(
            success=True,
            generated_note=response.get("generated_note", ""),
            metadata={
                "generation_type": "incorrect_injection",
                "error_category": error_category,
                "enhanced": enhance,
                **response.get("error_metadata", {}),
            },
            verification=None,
            error=None,
        )

    except Exception as e:
        return GenerationResult(
            success=False,
            generated_note=None,
            metadata={"error_category": error_category},
            verification=None,
            error=str(e),
        )


def verify_with_gpt4o(
    original: str,
    generated: str,
    prompts: Dict,
    client: OpenAI,
) -> Dict:
    """Verify semantic equivalence using GPT-4o."""
    system_prompt = prompts["verification_system"]
    user_prompt = prompts["verification_user_template"].format(
        original_note=original,
        generated_note=generated,
    )

    try:
        result = call_gpt4o_json(
            user_prompt,
            system_prompt,
            client,
            temperature=0.1,  # Low temp for consistency
        )
        return result
    except Exception as e:
        return {
            "classification": "ERROR",
            "confidence": 0.0,
            "reasoning": f"Verification failed: {e}",
            "differences_found": [],
        }


def verify_with_claude(
    original: str,
    generated: str,
    client: anthropic.Anthropic,
) -> Dict:
    """Verify semantic equivalence using Claude."""
    if not HAS_ANTHROPIC or client is None:
        return None

    try:
        prompt = f"""Compare these two clinical notes and determine if they are semantically equivalent (same clinical meaning) or if one contains an error.

Original note:
{original}

Generated note:
{generated}

Respond with JSON:
{{"classification": "EQUIVALENT or ERROR_DETECTED", "confidence": 0.0-1.0, "reasoning": "brief explanation"}}"""

        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            temperature=0.1,
            messages=[{"role": "user", "content": prompt}]
        )

        # Parse JSON from response
        text = response.content[0].text
        # Find JSON object in response
        import re
        json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
        else:
            return {"classification": "ERROR", "confidence": 0.0, "reasoning": "Failed to parse"}

    except Exception as e:
        return {"classification": "ERROR", "confidence": 0.0, "reasoning": f"Claude error: {e}"}


def multi_model_verification(
    original: str,
    generated: str,
    prompts: Dict,
    gpt4o_client: OpenAI,
    claude_client: Optional[anthropic.Anthropic],
) -> Dict:
    """
    Verify with multiple models and compute consensus.

    Returns:
    {
        "gpt4o_verdict": Dict,
        "claude_verdict": Dict,
        "consensus": "EQUIVALENT" | "ERROR_DETECTED" | "UNCERTAIN",
        "confidence": float,
    }
    """
    # GPT-4o verification
    gpt4o_verdict = verify_with_gpt4o(original, generated, prompts, gpt4o_client)

    # Claude verification (optional)
    claude_verdict = None
    if claude_client:
        claude_verdict = verify_with_claude(original, generated, claude_client)

    # Compute consensus
    verdicts = [gpt4o_verdict]
    if claude_verdict and claude_verdict.get("classification") != "ERROR":
        verdicts.append(claude_verdict)

    # Count votes
    equivalent_votes = sum(
        1 for v in verdicts
        if v.get("classification") == "EQUIVALENT"
    )

    error_votes = sum(
        1 for v in verdicts
        if v.get("classification") == "ERROR_DETECTED"
    )

    if equivalent_votes > error_votes:
        consensus = "EQUIVALENT"
        confidence = equivalent_votes / len(verdicts)
    elif error_votes > equivalent_votes:
        consensus = "ERROR_DETECTED"
        confidence = error_votes / len(verdicts)
    else:
        consensus = "UNCERTAIN"
        confidence = 0.5

    return {
        "gpt4o_verdict": gpt4o_verdict,
        "claude_verdict": claude_verdict,
        "consensus": consensus,
        "confidence": confidence,
    }


def generate_dataset(args):
    """Main generation function."""

    # Load prompts
    prompts = load_prompts(args.prompt_file)

    # Initialize clients
    if not HAS_OPENAI:
        raise RuntimeError("OpenAI package required: pip install openai")

    gpt4o_client = OpenAI(api_key=args.openai_api_key)
    claude_client = None
    if HAS_ANTHROPIC and args.anthropic_api_key:
        claude_client = anthropic.Anthropic(api_key=args.anthropic_api_key)

    print(f"[INFO] GPT-4o client initialized")
    if claude_client:
        print(f"[INFO] Claude client initialized for verification")

    # Load input data
    with open(args.input_jsonl, 'r') as f:
        notes = [json.loads(line) for line in f]

    print(f"[INFO] Loaded {len(notes)} notes")

    # Prepare output
    os.makedirs(args.output_dir, exist_ok=True)

    # Statistics
    stats = {
        "correct": {
            "attempted": 0,
            "generated": 0,
            "verified": 0,
            "enhanced": 0,
        },
        "incorrect": {
            "attempted": 0,
            "generated": 0,
            "verified": 0,
            "enhanced": 0,
            "by_category": {},
        },
        "verification_failures": {},
    }

    # Error categories for diverse generation
    error_categories = [
        "random",
        "medication",
        "laboratory",
        "diagnostic",
        "temporal",
        "treatment",
        "contextual",
    ]

    # Generate CORRECT examples
    print(f"\n[1/2] Generating {args.num_correct} CORRECT paraphrases...")
    correct_examples = []

    for i, note_data in enumerate(tqdm(notes[:args.num_correct * 2])):
        if len(correct_examples) >= args.num_correct:
            break

        original_note = note_data.get("text") or note_data.get("note") or note_data.get("correct_note", "")
        if not original_note or len(original_note) < 50:
            continue

        stats["correct"]["attempted"] += 1

        # Randomly enhance some examples
        enhance = args.enhance_complexity and random.random() < 0.3

        # Generate
        result = generate_correct_paraphrase(
            original_note,
            prompts,
            gpt4o_client,
            enhance=enhance,
        )

        if not result.success:
            continue

        stats["correct"]["generated"] += 1
        if enhance:
            stats["correct"]["enhanced"] += 1

        # Verify
        verification = multi_model_verification(
            original_note,
            result.generated_note,
            prompts,
            gpt4o_client,
            claude_client,
        )

        if verification["consensus"] == "EQUIVALENT" and verification["confidence"] >= 0.5:
            correct_examples.append({
                "original_note": original_note,
                "generated_note": result.generated_note,
                "label": "CORRECT",
                "metadata": result.metadata,
                "verification": verification,
            })
            stats["correct"]["verified"] += 1
        else:
            reason = verification["gpt4o_verdict"].get("reasoning", "unknown")
            stats["verification_failures"][reason] = stats["verification_failures"].get(reason, 0) + 1

    # Generate INCORRECT examples
    print(f"\n[2/2] Generating {args.num_incorrect} INCORRECT examples...")
    incorrect_examples = []

    for i, note_data in enumerate(tqdm(notes[:args.num_incorrect * 2])):
        if len(incorrect_examples) >= args.num_incorrect:
            break

        original_note = note_data.get("text") or note_data.get("note") or note_data.get("correct_note", "")
        if not original_note or len(original_note) < 50:
            continue

        stats["incorrect"]["attempted"] += 1

        # Choose error category (cycle through for diversity)
        error_category = error_categories[i % len(error_categories)]
        stats["incorrect"]["by_category"][error_category] = stats["incorrect"]["by_category"].get(error_category, 0) + 1

        # Randomly enhance some examples
        enhance = args.enhance_complexity and random.random() < 0.3

        # Generate
        result = generate_incorrect_with_error(
            original_note,
            error_category,
            prompts,
            gpt4o_client,
            enhance=enhance,
        )

        if not result.success:
            continue

        stats["incorrect"]["generated"] += 1
        if enhance:
            stats["incorrect"]["enhanced"] += 1

        # For incorrect, just check that it's different from original
        if result.generated_note and result.generated_note != original_note:
            incorrect_examples.append({
                "original_note": original_note,
                "generated_note": result.generated_note,
                "label": "INCORRECT",
                "metadata": result.metadata,
            })
            stats["incorrect"]["verified"] += 1

    # Write outputs
    correct_output = os.path.join(args.output_dir, "sft_correct_advanced.jsonl")
    incorrect_output = os.path.join(args.output_dir, "sft_incorrect_advanced.jsonl")
    combined_output = os.path.join(args.output_dir, "sft_combined_advanced.jsonl")
    stats_output = os.path.join(args.output_dir, "generation_stats_advanced.json")

    print(f"\n[INFO] Writing outputs...")

    with open(correct_output, 'w') as f:
        for ex in correct_examples:
            f.write(json.dumps(ex) + "\n")

    with open(incorrect_output, 'w') as f:
        for ex in incorrect_examples:
            f.write(json.dumps(ex) + "\n")

    # Combined and shuffled
    all_examples = correct_examples + incorrect_examples
    random.shuffle(all_examples)

    with open(combined_output, 'w') as f:
        for ex in all_examples:
            f.write(json.dumps(ex) + "\n")

    with open(stats_output, 'w') as f:
        json.dump(stats, f, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print("ADVANCED GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"CORRECT: {len(correct_examples)} verified ({stats['correct']['enhanced']} enhanced)")
    print(f"INCORRECT: {len(incorrect_examples)} verified ({stats['incorrect']['enhanced']} enhanced)")
    print(f"Total: {len(all_examples)} examples")
    print(f"\nError category distribution:")
    for cat, count in sorted(stats["incorrect"]["by_category"].items()):
        print(f"  {cat}: {count}")
    print(f"\nOutput files:")
    print(f"  {correct_output}")
    print(f"  {incorrect_output}")
    print(f"  {combined_output}")
    print(f"  {stats_output}")


def main():
    parser = argparse.ArgumentParser(description="Advanced SFT data generation with GPT-4o")
    parser.add_argument("--input-jsonl", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--prompt-file", default="configs/prompts/gpt4o_sft_generation_prompts.json")
    parser.add_argument("--num-correct", type=int, default=1000)
    parser.add_argument("--num-incorrect", type=int, default=1000)
    parser.add_argument("--enhance-complexity", action="store_true", help="Randomly enhance 30%% of notes with additional complexity")
    parser.add_argument("--openai-api-key", required=True)
    parser.add_argument("--anthropic-api-key", help="Optional Claude API key for multi-model verification")

    args = parser.parse_args()

    generate_dataset(args)


if __name__ == "__main__":
    main()
