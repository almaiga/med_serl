#!/usr/bin/env python3
"""
Quick inference utility for a Qwen3-4B LoRA adapter.

Example:
  python scripts/sft/quick_infer_qwen3_4b_lora.py \
    --model-name Qwen/Qwen3-4B \
    --adapter-dir outputs/qwen3-4b-lora \
    --mode assessor \
    --input-note "A 55-year-old man ... Diagnosis: stable angina."
"""

import argparse
import json
import random
import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPTS = {
    "assessor": (
        "You are a meticulous clinical note assessor in a self-play loop. Your job"
        " is to analyze the note for clinical correctness, detect errors when they"
        " exist, and provide a clear final answer with a Yes/No error decision.\n\n"
        "CRITICAL: Your response MUST end with EXACTLY this format on the last line:\n"
        'final_answer: "CORRECT"\n'
        "OR\n"
        'final_answer: "INCORRECT"\n\n'
        "Do not add any text after the final_answer line."
    ),
    "injector": (
        "You are an error injector in a self-play loop. Follow the prompt intent to"
        " transform the input note into a new note, either correct or with a subtle"
        " error, and provide a clear final answer.\n\n"
        "CRITICAL: Your response MUST end with EXACTLY this format on the last line:\n"
        'final_answer: "CORRECT"\n'
        "OR\n"
        'final_answer: "INCORRECT"\n\n'
        "Do not add any text after the final_answer line."
    ),
}


def build_messages(mode: str, note: str, prompt_intent: str) -> List[Dict[str, str]]:
    if mode == "assessor":
        user_content = (
            "Role: assessor\n"
            "Task: analyze the clinical note for errors and classify it as CORRECT or INCORRECT.\n\n"
            f"Clinical note:\n{note}\n\n"
            "Provide your reasoning in a <think> block, then output:\n"
            'final_answer: "CORRECT" or "INCORRECT"\n'
        )
        system_prompt = SYSTEM_PROMPTS["assessor"]
    else:
        user_content = (
            "Role: error injector\n"
            "Task: follow the prompt intent and transform the input note into a new note.\n"
            f'prompt_intent: "{prompt_intent}"\n\n'
            f"input_note:\n{note}\n\n"
            "Provide your reasoning in a <think> block, then output:\n"
            "generated_note:\n... \n"
            'final_answer: "CORRECT" or "INCORRECT"\n'
        )
        system_prompt = SYSTEM_PROMPTS["injector"]

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick inference for Qwen3-4B LoRA.")
    parser.add_argument("--model-name", required=True, help="Base model name.")
    parser.add_argument("--adapter-dir", required=True, help="LoRA adapter directory.")
    parser.add_argument(
        "--jsonl-file",
        default=None,
        help="Optional JSONL file (rl_train.jsonl) for sampling examples.",
    )
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--scenarios",
        default="assessor_correct,assessor_incorrect,injector_correct,injector_incorrect",
        help="Comma-separated scenarios to run.",
    )
    parser.add_argument("--input-note", default=None, help="Fallback single input note.")
    parser.add_argument(
        "--prompt-intent",
        default="Create a realistic note with no clinical errors.",
        help="Injector intent.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--min-similarity", type=float, default=0.80, help="Minimum similarity threshold.")
    parser.add_argument("--max-similarity", type=float, default=0.99, help="Maximum similarity threshold.")
    return parser.parse_args()


def extract_final_answer(text: str) -> Optional[str]:
    """Extract final answer with multiple fallback patterns."""
    # Primary pattern: exact format
    match = re.search(r'final_answer:\s*"(CORRECT|INCORRECT)"', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback 1: without quotes
    match = re.search(r'final_answer:\s*(CORRECT|INCORRECT)', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    # Fallback 2: look for Error Detected field (from model's old format)
    if re.search(r'Error Detected:\s*Yes', text, re.IGNORECASE):
        return "INCORRECT"
    elif re.search(r'Error Detected:\s*No', text, re.IGNORECASE):
        return "CORRECT"
    
    # Fallback 3: look at assessment line
    match = re.search(r'Assessment:\s*(CORRECT|INCORRECT)', text, re.IGNORECASE)
    if match:
        return match.group(1).upper()
    
    return None


def extract_generated_note(text: str) -> Optional[str]:
    """Extract the generated_note section from injector output."""
    # Split by "generated_note:" first to isolate the section
    parts = re.split(r'generated_note:\s*\n', text, flags=re.IGNORECASE)
    
    if len(parts) < 2:
        return None
    
    # Take everything after "generated_note:"
    after_label = parts[1]
    
    # Now extract until "final_answer:" (stop marker)
    match = re.search(r'^(.*?)\s*final_answer:', after_label, re.DOTALL | re.IGNORECASE)
    
    if match:
        generated = match.group(1).strip()
    else:
        # If no final_answer found, take everything
        generated = after_label.strip()
    
    # Clean up any remaining artifacts
    # Remove trailing </think> tags that might have leaked through
    generated = re.sub(r'</think>\s*$', '', generated).strip()
    
    return generated if generated else None


def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity ratio between two texts using SequenceMatcher."""
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()


def check_similarity_validity(original: str, generated: str, min_sim: float, max_sim: float) -> Tuple[bool, float]:
    """
    Check if generated note has valid similarity (between min and max thresholds).
    Returns (is_valid, similarity_score).
    """
    similarity = calculate_similarity(original, generated)
    is_valid = min_sim <= similarity < max_sim
    return is_valid, similarity


def load_records(jsonl_file: str) -> List[Dict]:
    records = []
    with open(jsonl_file, "r", encoding="utf-8") as handle:
        for line in handle:
            records.append(json.loads(line))
    return records


def scenario_samples(records: List[Dict], scenario: str, num_samples: int) -> List[Dict]:
    if not records:
        return []
    if scenario == "assessor_correct":
        pool = [r for r in records if r.get("correct_note")]
    elif scenario == "assessor_incorrect":
        pool = [r for r in records if r.get("incorrect_note")]
    elif scenario == "injector_correct":
        pool = [r for r in records if r.get("correct_note")]
    else:
        pool = [r for r in records if r.get("correct_note") and r.get("incorrect_note")]
    if not pool:
        return []
    return pool[:num_samples]


def print_recap_table(results: Dict) -> None:
    """Print a comprehensive recap table of all results."""
    print("\n" + "=" * 100)
    print("FINAL RECAP - MODEL PERFORMANCE SUMMARY")
    print("=" * 100)
    
    # Assessor Task Results
    print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│ ASSESSOR TASK - Error Detection Performance                                 │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    print("│ Scenario              │ Samples │ Correct │ Accuracy │ Notes               │")
    print("├───────────────────────┼─────────┼─────────┼──────────┼─────────────────────┤")
    
    for scenario in ["assessor_correct", "assessor_incorrect"]:
        if scenario in results:
            data = results[scenario]
            scenario_name = "Correct Notes" if scenario == "assessor_correct" else "Incorrect Notes"
            accuracy = f"{data['accuracy']:.1f}%" if data['total'] > 0 else "N/A"
            note = "Should detect NO errors" if scenario == "assessor_correct" else "Should detect errors"
            print(f"│ {scenario_name:<21} │ {data['total']:>7} │ {data['correct']:>7} │ {accuracy:>8} │ {note:<19} │")
    
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    
    # Calculate overall assessor accuracy
    total_assessor = sum(results.get(s, {}).get('total', 0) for s in ["assessor_correct", "assessor_incorrect"])
    correct_assessor = sum(results.get(s, {}).get('correct', 0) for s in ["assessor_correct", "assessor_incorrect"])
    if total_assessor > 0:
        overall_assessor_acc = correct_assessor / total_assessor * 100
        print(f"\n  Overall Assessor Accuracy: {correct_assessor}/{total_assessor} ({overall_assessor_acc:.1f}%)")
    
    # Injector Task Results
    print("\n┌─────────────────────────────────────────────────────────────────────────────────────────────────┐")
    print("│ INJECTOR TASK - Note Generation Performance                                                     │")
    print("├─────────────────────────────────────────────────────────────────────────────────────────────────┤")
    print("│ Scenario              │ Samples │ Correct │ Accuracy │ Avg Sim │ Valid Sim │ Sim Range         │")
    print("├───────────────────────┼─────────┼─────────┼──────────┼─────────┼───────────┼───────────────────┤")
    
    for scenario in ["injector_correct", "injector_incorrect"]:
        if scenario in results:
            data = results[scenario]
            scenario_name = "Generate Correct" if scenario == "injector_correct" else "Inject Errors"
            accuracy = f"{data['accuracy']:.1f}%" if data['total'] > 0 else "N/A"
            
            if data['similarity_scores']:
                avg_sim = sum(data['similarity_scores']) / len(data['similarity_scores'])
                sim_valid_rate = data['similarity_valid_count'] / len(data['similarity_scores']) * 100
                min_sim = min(data['similarity_scores'])
                max_sim = max(data['similarity_scores'])
                sim_range = f"{min_sim:.0%}-{max_sim:.0%}"
                print(f"│ {scenario_name:<21} │ {data['total']:>7} │ {data['correct']:>7} │ {accuracy:>8} │ {avg_sim:>6.1%} │ {sim_valid_rate:>7.1f}% │ {sim_range:>17} │")
            else:
                print(f"│ {scenario_name:<21} │ {data['total']:>7} │ {data['correct']:>7} │ {accuracy:>8} │ {'N/A':>7} │ {'N/A':>9} │ {'N/A':>17} │")
    
    print("└─────────────────────────────────────────────────────────────────────────────────────────────────┘")
    
    # Calculate overall injector accuracy
    total_injector = sum(results.get(s, {}).get('total', 0) for s in ["injector_correct", "injector_incorrect"])
    correct_injector = sum(results.get(s, {}).get('correct', 0) for s in ["injector_correct", "injector_incorrect"])
    if total_injector > 0:
        overall_injector_acc = correct_injector / total_injector * 100
        print(f"\n  Overall Injector Accuracy: {correct_injector}/{total_injector} ({overall_injector_acc:.1f}%)")
    
    # Overall similarity metrics
    all_sim_scores = []
    all_sim_valid = 0
    for scenario in ["injector_correct", "injector_incorrect"]:
        if scenario in results and results[scenario]['similarity_scores']:
            all_sim_scores.extend(results[scenario]['similarity_scores'])
            all_sim_valid += results[scenario]['similarity_valid_count']
    
    if all_sim_scores:
        overall_avg_sim = sum(all_sim_scores) / len(all_sim_scores)
        overall_sim_valid_rate = all_sim_valid / len(all_sim_scores) * 100
        print(f"  Overall Average Similarity: {overall_avg_sim:.2%}")
        print(f"  Overall Valid Similarity Rate: {all_sim_valid}/{len(all_sim_scores)} ({overall_sim_valid_rate:.1f}%)")
    
    # Key Insights
    print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
    print("│ KEY INSIGHTS                                                                 │")
    print("├─────────────────────────────────────────────────────────────────────────────┤")
    
    # Assessor insights
    if "assessor_correct" in results and "assessor_incorrect" in results:
        correct_acc = results["assessor_correct"]["accuracy"]
        incorrect_acc = results["assessor_incorrect"]["accuracy"]
        
        if correct_acc >= 80 and incorrect_acc >= 80:
            print("│ ✓ Assessor: STRONG - Good at detecting both correct and incorrect notes   │")
        elif correct_acc >= 80:
            print("│ ⚠ Assessor: Over-sensitive - Good with correct notes, struggles with errors│")
        elif incorrect_acc >= 80:
            print("│ ⚠ Assessor: Under-sensitive - Good with errors, flags correct notes       │")
        else:
            print("│ ✗ Assessor: WEAK - Struggles with both correct and incorrect notes        │")
    
    # Injector insights
    if "injector_correct" in results and "injector_incorrect" in results:
        correct_data = results["injector_correct"]
        incorrect_data = results["injector_incorrect"]
        
        correct_acc = correct_data["accuracy"]
        incorrect_acc = incorrect_data["accuracy"]
        
        if correct_data['similarity_scores'] and incorrect_data['similarity_scores']:
            correct_sim = sum(correct_data['similarity_scores']) / len(correct_data['similarity_scores'])
            incorrect_sim = sum(incorrect_data['similarity_scores']) / len(incorrect_data['similarity_scores'])
            
            if correct_acc >= 80 and incorrect_acc >= 80:
                print("│ ✓ Injector: STRONG - Generates both correct and incorrect notes well     │")
            elif correct_acc >= 80:
                print("│ ⚠ Injector: Partial - Good with correct notes, struggles with errors     │")
            elif incorrect_acc >= 80:
                print("│ ⚠ Injector: Partial - Good with errors, struggles with correct notes     │")
            else:
                print("│ ✗ Injector: WEAK - Struggles with both note types                         │")
            
            if correct_sim > 0.95:
                print("│ ⚠ Similarity: Too high for correct notes (mostly copying)                 │")
            elif correct_sim < 0.80:
                print("│ ⚠ Similarity: Too low for correct notes (changing too much)               │")
            else:
                print("│ ✓ Similarity: Good range for correct notes (80-95%)                       │")
    
    print("└─────────────────────────────────────────────────────────────────────────────┘")
    
    print("\n" + "=" * 100 + "\n")


def main() -> None:
    args = parse_args()
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    random.seed(args.seed)

    records = []
    if args.jsonl_file:
        records = load_records(args.jsonl_file)
        random.shuffle(records)

    if not records and not args.input_note:
        raise SystemExit("Provide --jsonl-file or --input-note.")

    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto",
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, args.adapter_dir)
    model.eval()

    # Store results for final recap
    all_results = {}

    for scenario in scenarios:
        samples = scenario_samples(records, scenario, args.num_samples)
        if not samples and args.input_note:
            samples = [{"correct_note": args.input_note, "incorrect_note": args.input_note}]

        print(f"\n=== Scenario: {scenario} (n={len(samples)}) ===")
        correct = 0
        total = 0
        
        # For injector scenarios, track similarity metrics
        is_injector = scenario.startswith("injector")
        similarity_valid_count = 0
        similarity_scores = []

        for idx, record in enumerate(samples, start=1):
            if scenario == "assessor_correct":
                note = record.get("correct_note")
                expected = "CORRECT"
                messages = build_messages("assessor", note, args.prompt_intent)
            elif scenario == "assessor_incorrect":
                note = record.get("incorrect_note")
                expected = "INCORRECT"
                messages = build_messages("assessor", note, args.prompt_intent)
            elif scenario == "injector_correct":
                note = record.get("correct_note")
                expected = "CORRECT"
                messages = build_messages("injector", note, args.prompt_intent)
            else:
                note = record.get("correct_note")
                expected = "INCORRECT"
                error_type = (record.get("error_type") or "").strip()
                if error_type:
                    intent = f"Introduce a {error_type} error while keeping the note realistic."
                else:
                    intent = "Introduce a subtle clinical error while keeping the note realistic."
                messages = build_messages("injector", note, intent)

            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    pad_token_id=tokenizer.eos_token_id,
                )

            # Decode ONLY the generated tokens (not the input prompt)
            input_length = inputs['input_ids'].shape[1]
            generated_tokens = outputs[0][input_length:]
            generated = tokenizer.decode(generated_tokens, skip_special_tokens=True)
            predicted = extract_final_answer(generated)
            is_correct = predicted == expected
            total += 1
            correct += int(is_correct)

            print(f"\n--- Example {idx} ---")
            print(f"expected: {expected} | predicted: {predicted or 'MISSING'} | match: {is_correct}")
            
            # For injector scenarios, check similarity
            if is_injector:
                generated_note = extract_generated_note(generated)
                if generated_note:
                    is_valid_sim, sim_score = check_similarity_validity(
                        note, generated_note, args.min_similarity, args.max_similarity
                    )
                    similarity_scores.append(sim_score)
                    similarity_valid_count += int(is_valid_sim)
                    
                    print(f"similarity: {sim_score:.2%} | valid: {is_valid_sim} (target: {args.min_similarity:.0%}-{args.max_similarity:.0%})")
                else:
                    print("similarity: N/A (could not extract generated_note)")
            
            print(generated)

        # Store results for this scenario
        scenario_results = {
            'total': total,
            'correct': correct,
            'accuracy': (correct / total * 100) if total > 0 else 0,
        }
        
        if is_injector:
            scenario_results['similarity_scores'] = similarity_scores
            scenario_results['similarity_valid_count'] = similarity_valid_count
        
        all_results[scenario] = scenario_results

        if total:
            acc = correct / total * 100
            print(f"\nScenario accuracy: {correct}/{total} ({acc:.1f}%)")
            
            # Print similarity metrics for injector scenarios
            if is_injector and similarity_scores:
                avg_sim = sum(similarity_scores) / len(similarity_scores)
                sim_valid_rate = similarity_valid_count / len(similarity_scores) * 100
                print(f"Similarity metrics:")
                print(f"  - Average similarity: {avg_sim:.2%}")
                print(f"  - Valid similarity rate: {similarity_valid_count}/{len(similarity_scores)} ({sim_valid_rate:.1f}%)")
                print(f"  - Similarity range: {min(similarity_scores):.2%} - {max(similarity_scores):.2%}")

    # Print final recap table
    print_recap_table(all_results)


if __name__ == "__main__":
    main()
