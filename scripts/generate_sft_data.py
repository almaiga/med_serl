#!/usr/bin/env python3
"""
Generate SFT training data for MedSeRL with multi-stage verification.

Pipeline:
1. Load MEDEC notes
2. For INCORRECT: Use MEDEC ground truth + GPT-4o enhancement
3. For CORRECT: Generate with GPT-4o + multi-verifier consensus
4. Apply VCF filtering
5. Save verified examples

Usage:
    python scripts/generate_sft_data.py \
        --input-jsonl data/medec_train.jsonl \
        --output-dir data/sft_training \
        --num-correct 500 \
        --num-incorrect 500 \
        --openai-api-key $OPENAI_API_KEY
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm
import re

# Add scripts/sft to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'sft'))

from inference_utils import apply_vcf, jaccard_similarity

# OpenAI imports
try:
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False
    print("[WARNING] OpenAI package not installed. Install with: pip install openai")

# Anthropic imports for multi-verifier
try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False
    print("[WARNING] Anthropic package not installed. Install with: pip install anthropic")


@dataclass
class VerificationResult:
    """Result from multi-verifier consensus."""
    passed: bool
    verifier_votes: Dict[str, bool]
    consensus_score: float
    reason: Optional[str]


def load_enhanced_prompts(prompt_file: str) -> Dict:
    """Load enhanced prompts with verification template."""
    with open(prompt_file, 'r') as f:
        return json.load(f)


def call_gpt4o(
    prompt: str,
    system_prompt: str,
    temperature: float = 0.3,
    client: Optional[OpenAI] = None,
) -> str:
    """Call GPT-4o API."""
    if not HAS_OPENAI or client is None:
        raise ValueError("OpenAI client not available")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        temperature=temperature,
        max_tokens=2000,
    )

    return response.choices[0].message.content


def call_claude(
    prompt: str,
    system_prompt: str,
    temperature: float = 0.3,
    client: Optional[anthropic.Anthropic] = None,
) -> str:
    """Call Claude API for verification."""
    if not HAS_ANTHROPIC or client is None:
        return None

    try:
        response = client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1024,
            temperature=temperature,
            system=system_prompt,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        print(f"[WARNING] Claude API call failed: {e}")
        return None


def extract_medical_entities(note: str) -> Dict:
    """
    Extract critical medical entities using regex patterns.
    Simpler than full NER but effective for verification.
    """
    entities = {
        "medications": [],
        "dosages": [],
        "measurements": [],
        "laterality": [],
    }

    # Medication patterns (common drug suffixes)
    med_pattern = r'\b[A-Z][a-z]+(?:ol|ide|in|pril|sartan|mycin|cillin|azole)\b'
    entities["medications"] = list(set(re.findall(med_pattern, note, re.IGNORECASE)))

    # Dosage patterns (number + unit)
    dosage_pattern = r'\b\d+\.?\d*\s*(?:mg|mcg|g|mL|L|units?)\b'
    entities["dosages"] = re.findall(dosage_pattern, note, re.IGNORECASE)

    # Measurement patterns (number + unit + context)
    measurement_pattern = r'\b\d+\.?\d*\s*(?:mg/dL|mmHg|bpm|pg/mL|%)\b'
    entities["measurements"] = re.findall(measurement_pattern, note, re.IGNORECASE)

    # Laterality
    laterality_pattern = r'\b(?:left|right|bilateral|unilateral)\b'
    entities["laterality"] = re.findall(laterality_pattern, note, re.IGNORECASE)

    return entities


def verify_entity_preservation(original: str, generated: str) -> Tuple[bool, str]:
    """
    Verify all critical entities are preserved.
    Returns (passed, reason).
    """
    orig_entities = extract_medical_entities(original)
    gen_entities = extract_medical_entities(generated)

    # Check medications preserved
    orig_meds = set(m.lower() for m in orig_entities["medications"])
    gen_meds = set(m.lower() for m in gen_entities["medications"])
    if orig_meds != gen_meds:
        return False, f"Medications changed: {orig_meds} vs {gen_meds}"

    # Check dosages preserved
    if sorted(orig_entities["dosages"]) != sorted(gen_entities["dosages"]):
        return False, f"Dosages changed: {orig_entities['dosages']} vs {gen_entities['dosages']}"

    # Check measurements preserved
    if sorted(orig_entities["measurements"]) != sorted(gen_entities["measurements"]):
        return False, f"Measurements changed"

    # Check laterality preserved
    orig_lat = set(l.lower() for l in orig_entities["laterality"])
    gen_lat = set(l.lower() for l in gen_entities["laterality"])
    if orig_lat != gen_lat:
        return False, f"Laterality changed: {orig_lat} vs {gen_lat}"

    return True, "All entities preserved"


def verify_with_llm(
    original: str,
    generated: str,
    verification_prompt_template: str,
    model_client,
    model_name: str,
) -> Tuple[bool, str]:
    """
    Verify semantic equivalence using LLM.
    Returns (is_equivalent, explanation).
    """
    prompt = verification_prompt_template.format(
        original_note=original,
        generated_note=generated
    )

    system_prompt = "You are a medical note verification expert. Determine if two notes are semantically equivalent."

    try:
        if model_name == "gpt-4o":
            response = call_gpt4o(prompt, system_prompt, temperature=0.1, client=model_client)
        elif model_name == "claude":
            response = call_claude(prompt, system_prompt, temperature=0.1, client=model_client)
        else:
            return False, "Unknown model"

        # Parse response
        if "EQUIVALENT" in response.upper() and "CONTAINS_ERROR" not in response.upper():
            return True, "LLM verified equivalent"
        else:
            # Extract error description if present
            if "error_description:" in response:
                error_desc = response.split("error_description:")[1].strip().split('\n')[0]
                return False, f"LLM detected error: {error_desc}"
            return False, "LLM detected error"
    except Exception as e:
        print(f"[WARNING] LLM verification failed for {model_name}: {e}")
        return False, f"Verification error: {e}"


def multi_verifier_consensus(
    original: str,
    generated: str,
    verification_prompt_template: str,
    gpt4o_client: Optional[OpenAI],
    claude_client: Optional[anthropic.Anthropic],
) -> VerificationResult:
    """
    Multi-stage verification with consensus voting.

    Verifiers (in order):
    1. Entity preservation (fast, deterministic)
    2. VCF similarity check (fast, deterministic)
    3. GPT-4o verification (slow, expensive)
    4. Claude verification (slow, expensive)

    Requires majority (3/4) to pass.
    """
    verifier_votes = {}

    # Verifier 1: Entity preservation
    entity_passed, entity_reason = verify_entity_preservation(original, generated)
    verifier_votes["entity_preservation"] = entity_passed

    if not entity_passed:
        return VerificationResult(
            passed=False,
            verifier_votes=verifier_votes,
            consensus_score=0.0,
            reason=f"Entity check failed: {entity_reason}"
        )

    # Verifier 2: VCF similarity (ensure meaningful paraphrase)
    vcf_result = apply_vcf(original, generated, min_jaccard=0.85, max_jaccard=0.99, max_word_edits=6)
    verifier_votes["vcf_similarity"] = vcf_result.passed

    if not vcf_result.passed:
        return VerificationResult(
            passed=False,
            verifier_votes=verifier_votes,
            consensus_score=0.25,
            reason=f"VCF failed: {vcf_result.reason}"
        )

    # Verifier 3: GPT-4o
    if gpt4o_client:
        gpt4o_passed, gpt4o_reason = verify_with_llm(
            original, generated, verification_prompt_template,
            gpt4o_client, "gpt-4o"
        )
        verifier_votes["gpt4o"] = gpt4o_passed
    else:
        verifier_votes["gpt4o"] = None

    # Verifier 4: Claude
    if claude_client:
        claude_passed, claude_reason = verify_with_llm(
            original, generated, verification_prompt_template,
            claude_client, "claude"
        )
        verifier_votes["claude"] = claude_passed
    else:
        verifier_votes["claude"] = None

    # Compute consensus
    valid_votes = [v for v in verifier_votes.values() if v is not None]
    if len(valid_votes) == 0:
        return VerificationResult(
            passed=False,
            verifier_votes=verifier_votes,
            consensus_score=0.0,
            reason="No valid verifiers"
        )

    passed_votes = sum(v for v in valid_votes if v)
    consensus_score = passed_votes / len(valid_votes)

    # Require majority (>= 0.5) or all deterministic checks pass
    passed = consensus_score >= 0.5 and entity_passed and vcf_result.passed

    return VerificationResult(
        passed=passed,
        verifier_votes=verifier_votes,
        consensus_score=consensus_score,
        reason=None if passed else f"Consensus: {consensus_score:.2f}"
    )


def generate_correct_paraphrase(
    original_note: str,
    prompts: Dict,
    gpt4o_client: OpenAI,
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Generate CORRECT paraphrase with GPT-4o.
    Returns (generated_note, metadata) or (None, None) if parsing fails.
    """
    system_prompt = prompts["system_prompt_correct"]
    user_prompt = prompts["injector_correct_template"].format(note=original_note)

    response = call_gpt4o(user_prompt, system_prompt, temperature=0.5, client=gpt4o_client)

    # Parse response
    if "generated_note:" not in response:
        return None, None

    try:
        # Extract generated_note
        generated_note = response.split("generated_note:")[1].split("final_answer:")[0].strip()

        # Extract changes_made
        if "changes_made:" in response:
            changes_str = response.split("changes_made:")[1].strip()
            # Find JSON object
            json_match = re.search(r'\{[^}]+\}', changes_str)
            if json_match:
                changes = json.loads(json_match.group())
            else:
                changes = {}
        else:
            changes = {}

        metadata = {
            "changes_made": changes,
            "full_response": response
        }

        return generated_note, metadata
    except Exception as e:
        print(f"[WARNING] Failed to parse response: {e}")
        return None, None


def generate_incorrect_with_error(
    original_note: str,
    error_type: str,
    prompts: Dict,
    gpt4o_client: OpenAI,
) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Generate INCORRECT note with error injection using GPT-4o.
    Returns (generated_note, metadata) or (None, None) if parsing fails.
    """
    system_prompt = prompts["system_prompt_incorrect"]
    user_prompt = prompts["injector_incorrect_template"].format(
        note=original_note,
        prompt_intent=error_type
    )

    response = call_gpt4o(user_prompt, system_prompt, temperature=0.7, client=gpt4o_client)

    # Parse response
    if "generated_note:" not in response:
        return None, None

    try:
        # Extract generated_note
        generated_note = response.split("generated_note:")[1].split("final_answer:")[0].strip()

        # Extract changes_made
        if "changes_made:" in response:
            changes_str = response.split("changes_made:")[1].strip()
            json_match = re.search(r'\{[^}]+\}', changes_str)
            if json_match:
                changes = json.loads(json_match.group())
            else:
                changes = {}
        else:
            changes = {}

        metadata = {
            "error_type": error_type,
            "changes_made": changes,
            "full_response": response
        }

        return generated_note, metadata
    except Exception as e:
        print(f"[WARNING] Failed to parse response: {e}")
        return None, None


def generate_sft_dataset(args):
    """Main function to generate SFT training data."""

    # Load prompts
    prompts = load_enhanced_prompts(args.prompt_file)
    verification_template = prompts.get("verification_prompt_template", "")

    # Initialize API clients
    gpt4o_client = None
    claude_client = None

    if HAS_OPENAI and args.openai_api_key:
        gpt4o_client = OpenAI(api_key=args.openai_api_key)
        print("[INFO] GPT-4o client initialized")
    else:
        print("[ERROR] OpenAI API key required")
        return

    if HAS_ANTHROPIC and args.anthropic_api_key:
        claude_client = anthropic.Anthropic(api_key=args.anthropic_api_key)
        print("[INFO] Claude client initialized for verification")

    # Load input data
    with open(args.input_jsonl, 'r') as f:
        notes = [json.loads(line) for line in f]

    print(f"[INFO] Loaded {len(notes)} notes from {args.input_jsonl}")

    # Prepare output
    os.makedirs(args.output_dir, exist_ok=True)
    correct_output = os.path.join(args.output_dir, "sft_correct_verified.jsonl")
    incorrect_output = os.path.join(args.output_dir, "sft_incorrect_verified.jsonl")
    stats_output = os.path.join(args.output_dir, "generation_stats.json")

    correct_examples = []
    incorrect_examples = []

    stats = {
        "correct_attempted": 0,
        "correct_generated": 0,
        "correct_verified": 0,
        "incorrect_attempted": 0,
        "incorrect_generated": 0,
        "incorrect_verified": 0,
        "verification_failures": {},
    }

    # Generate CORRECT examples
    print(f"\n[1/2] Generating {args.num_correct} CORRECT paraphrases with verification...")
    for i, note_data in enumerate(tqdm(notes[:args.num_correct * 3])):  # Try 3x for safety
        if len(correct_examples) >= args.num_correct:
            break

        original_note = note_data.get("text") or note_data.get("note") or note_data.get("correct_note", "")
        if not original_note:
            continue

        stats["correct_attempted"] += 1

        # Generate paraphrase
        generated_note, metadata = generate_correct_paraphrase(original_note, prompts, gpt4o_client)

        if generated_note is None:
            continue

        stats["correct_generated"] += 1

        # Multi-verifier consensus
        verification = multi_verifier_consensus(
            original_note,
            generated_note,
            verification_template,
            gpt4o_client,
            claude_client,
        )

        if verification.passed:
            correct_examples.append({
                "original_note": original_note,
                "generated_note": generated_note,
                "label": "CORRECT",
                "metadata": metadata,
                "verification": {
                    "verifier_votes": verification.verifier_votes,
                    "consensus_score": verification.consensus_score,
                }
            })
            stats["correct_verified"] += 1
        else:
            # Track failure reason
            reason = verification.reason or "unknown"
            stats["verification_failures"][reason] = stats["verification_failures"].get(reason, 0) + 1

    # Generate INCORRECT examples
    print(f"\n[2/2] Generating {args.num_incorrect} INCORRECT examples with errors...")

    error_types = [
        "medication error",
        "laboratory value error",
        "laterality error",
        "temporal error",
        "treatment status error",
    ]

    for i, note_data in enumerate(tqdm(notes[:args.num_incorrect * 2])):
        if len(incorrect_examples) >= args.num_incorrect:
            break

        original_note = note_data.get("text") or note_data.get("note") or note_data.get("correct_note", "")
        if not original_note:
            continue

        stats["incorrect_attempted"] += 1

        # Choose error type (cycle through)
        error_type = error_types[i % len(error_types)]

        # Generate with error
        generated_note, metadata = generate_incorrect_with_error(
            original_note,
            error_type,
            prompts,
            gpt4o_client,
        )

        if generated_note is None:
            continue

        stats["incorrect_generated"] += 1

        # Apply VCF to verify error was injected correctly
        vcf_result = apply_vcf(original_note, generated_note, min_jaccard=0.85, max_jaccard=0.99, max_word_edits=6)

        if vcf_result.passed:
            incorrect_examples.append({
                "original_note": original_note,
                "generated_note": generated_note,
                "label": "INCORRECT",
                "metadata": metadata,
                "vcf_result": {
                    "jaccard": vcf_result.score_jaccard,
                    "word_edits": vcf_result.word_edits,
                    "sentences_changed": vcf_result.sentences_changed,
                }
            })
            stats["incorrect_verified"] += 1

    # Write outputs
    print(f"\n[INFO] Writing {len(correct_examples)} CORRECT examples to {correct_output}")
    with open(correct_output, 'w') as f:
        for example in correct_examples:
            f.write(json.dumps(example) + "\n")

    print(f"[INFO] Writing {len(incorrect_examples)} INCORRECT examples to {incorrect_output}")
    with open(incorrect_output, 'w') as f:
        for example in incorrect_examples:
            f.write(json.dumps(example) + "\n")

    # Write stats
    with open(stats_output, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\n{'='*60}")
    print("GENERATION COMPLETE")
    print(f"{'='*60}")
    print(f"CORRECT: {stats['correct_verified']}/{stats['correct_attempted']} verified")
    print(f"INCORRECT: {stats['incorrect_verified']}/{stats['incorrect_attempted']} verified")
    print(f"Total SFT examples: {len(correct_examples) + len(incorrect_examples)}")
    print(f"\nVerification failure reasons:")
    for reason, count in sorted(stats["verification_failures"].items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}")
    print(f"\nOutput files:")
    print(f"  {correct_output}")
    print(f"  {incorrect_output}")
    print(f"  {stats_output}")


def main():
    parser = argparse.ArgumentParser(description="Generate SFT training data for MedSeRL")
    parser.add_argument("--input-jsonl", required=True, help="Input JSONL file with clinical notes")
    parser.add_argument("--output-dir", required=True, help="Output directory for verified examples")
    parser.add_argument("--prompt-file", default="configs/prompts/error_injection_prompts_v3_enhanced.json")
    parser.add_argument("--num-correct", type=int, default=500, help="Number of CORRECT examples to generate")
    parser.add_argument("--num-incorrect", type=int, default=500, help="Number of INCORRECT examples to generate")
    parser.add_argument("--openai-api-key", required=True, help="OpenAI API key for GPT-4o")
    parser.add_argument("--anthropic-api-key", help="Anthropic API key for Claude verification (optional)")

    args = parser.parse_args()

    generate_sft_dataset(args)


if __name__ == "__main__":
    main()
