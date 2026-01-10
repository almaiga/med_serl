#!/usr/bin/env python3
"""
Quick inference utility for a Qwen3-4B LoRA adapter.

Example:
  python scripts/sft/quick_infer_qwen3_4b_lora.py \
    --model-name Qwen/Qwen3-4B \
    --adapter-dir outputs/qwen3-4b-lora \
    --jsonl-file data_processed/medec_paired/train_val_split/rl_train.jsonl \
    --num-samples 3
"""

import argparse
import json
import os
import random
import re
from datetime import datetime
from typing import Dict, List, Optional

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer


SYSTEM_PROMPTS = {
    "injector": (
        "You are a clinical note transformer in a self-play loop.\n\n"
        "You MUST strictly follow the prompt intent.\n\n"
        "If asked to create a CORRECT note:\n"
        "- You MUST preserve ALL clinical facts from the input note\n"
        "- Including diagnosis, lab values, findings, and clinical interpretation\n"
        "- You are ONLY allowed to change surface form (wording, order, phrasing)\n"
        "- You are STRICTLY FORBIDDEN from reinterpreting, normalizing, or changing meaning\n\n"
        "If asked to create an INCORRECT note:\n"
        "- You MUST introduce exactly ONE subtle clinical error\n"
        "- All other clinical facts MUST remain unchanged\n\n"
        "CRITICAL: Your response MUST end with EXACTLY this format on the last line:\n"
        'final_answer: "CORRECT"\n'
        "OR\n"
        'final_answer: "INCORRECT"\n\n'
        "Do not add any text after the final_answer line."
    ),
}


def build_messages(
    mode: str,
    note: str,
    prompt_intent: str,
    assessor_prompts: Optional[Dict[str, str]] = None,
) -> List[Dict[str, str]]:
    if mode == "assessor":
        if assessor_prompts:
            system_prompt = assessor_prompts["system_prompt"]
            user_content = assessor_prompts["user_template"].format(note=note)
        else:
            system_prompt = (
                "You are a meticulous clinical note assessor in a self-play loop. Your job"
                " is to analyze the note for clinical correctness, detect errors when they"
                " exist, and provide a clear final answer with a Yes/No error decision.\n\n"
                "CRITICAL: Your response MUST end with EXACTLY this format on the last line:\n"
                "final_answer: \"CORRECT\"\n"
                "OR\n"
                "final_answer: \"INCORRECT\"\n\n"
                "Do not add any text after the final_answer line."
            )
            user_content = (
                "Role: assessor\n"
                "Task: analyze the clinical note for errors and classify it as CORRECT or INCORRECT.\n\n"
                f"Clinical note:\n{note}\n\n"
                "Provide your reasoning in a <think> block, then output:\n"
                'final_answer: "CORRECT" or "INCORRECT"\n'
            )

    else:
        is_correct = "no clinical errors" in prompt_intent.lower()

        if is_correct:
            user_content = (
                "Role: injector\n"
                "Task: Create a CLINICALLY IDENTICAL variant of the input note.\n\n"
                "IMPORTANT:\n"
                "- You must preserve ALL clinical facts exactly.\n"
                "- You are allowed to make ONLY MINIMAL surface edits.\n"
                "- You should change as LITTLE as possible.\n"
                "- You MAY leave the note unchanged if no safe edit is obvious.\n"
                "- You MUST NOT change diagnoses, clinical conclusions, or interpretations.\n\n"
                "Allowed edits:\n"
                "- punctuation\n"
                "- wording (synonyms with identical meaning)\n"
                "- sentence order\n\n"
                "Forbidden edits:\n"
                "- changing diagnosis terms\n"
                "- changing medical classifications\n"
                "- changing clinical conclusions\n\n"
                f"input_note:\n{note}\n\n"
                "In your <think> block, briefly describe the surface-level change.\n\n"
                "generated_note:\n[the updated note]\n\n"
                'final_answer: "CORRECT"\n'
            )
        else:
            user_content = (
                "Role: injector\n"
                "Task: Create a variant of the input note with EXACTLY ONE subtle clinical error.\n\n"
                "Rules:\n"
                "- Introduce exactly ONE error\n"
                "- All other clinical facts must remain unchanged\n"
                "- The error must be clinically plausible and local\n\n"
                f"input_note:\n{note}\n\n"
                "In your <think> block, briefly describe the injected error.\n\n"
                "generated_note:\n[the updated note]\n\n"
                'final_answer: "INCORRECT"\n'
            )

        system_prompt = SYSTEM_PROMPTS["injector"]

    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]


def load_assessor_prompts(prompt_file: Optional[str]) -> Optional[Dict[str, str]]:
    if not prompt_file:
        return None
    with open(prompt_file, "r", encoding="utf-8") as handle:
        prompts = json.load(handle)
    if "system_prompt" not in prompts or "user_template" not in prompts:
        raise ValueError("Prompt file must include system_prompt and user_template.")
    return prompts


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
    parser.add_argument(
        "--assessor-prompt-file",
        default=None,
        help="Optional JSON prompt file for assessor (system_prompt + user_template).",
    )
    parser.add_argument(
        "--output-dir",
        default="results/inference/quick_test",
        help="Directory to write JSONL results.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional run name (defaults to timestamp).",
    )
    parser.add_argument("--input-note", default=None, help="Fallback single input note.")
    parser.add_argument(
        "--prompt-intent",
        default="Create a clinically identical variant of the input note.",
        help="Injector intent.",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size for generation.")
    parser.add_argument(
        "--force-answer",
        action="store_true",
        help="If Answer/CORRECT/INCORRECT is missing, run a short follow-up decode.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="HF embedding model for cosine similarity.",
    )
    parser.add_argument(
        "--disable-embedding",
        action="store_true",
        help="Disable embedding cosine similarity scoring.",
    )
    return parser.parse_args()


def extract_final_answer(text: str) -> Optional[str]:
    """Extract final answer with multiple fallback patterns."""
    # Use findall to get ALL matches, then take the last one
    matches = re.findall(r'final_answer:\s*"(CORRECT|INCORRECT)"', text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()  # Return the LAST match
    
    # Fallback 1: without quotes (also get last match)
    matches = re.findall(r'final_answer:\s*(CORRECT|INCORRECT)', text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    
    # Fallback 2: look for Error Detected field (from model's old format)
    if re.search(r'Error Detected:\s*Yes', text, re.IGNORECASE):
        return "INCORRECT"
    elif re.search(r'Error Detected:\s*No', text, re.IGNORECASE):
        return "CORRECT"

    # Fallback 3: baseline prompt format "Answer: CORRECT/INCORRECT"
    matches = re.findall(r'Answer:\s*(CORRECT|INCORRECT)', text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    
    # Fallback 4: look at assessment line (get last match)
    matches = re.findall(r'Assessment:\s*(CORRECT|INCORRECT)', text, re.IGNORECASE)
    if matches:
        return matches[-1].upper()
    
    return None


def force_answer_from_prompt(
    prompt: str,
    tokenizer,
    model,
    device: str,
    max_new_tokens: int = 12,
) -> Optional[str]:
    """Run a short follow-up decode to elicit the Answer line."""
    followup = prompt.rstrip() + "\nAnswer:"
    inputs = tokenizer(followup, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    input_length = inputs["input_ids"].shape[1]
    generated_tokens = outputs[0][input_length:]
    generated = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    return extract_final_answer(generated)




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


def tokenize_for_jaccard(text: str) -> List[str]:
    return re.findall(r"[a-zA-Z0-9]+", text.lower())


def jaccard_similarity(text1: str, text2: str) -> float:
    set1 = set(tokenize_for_jaccard(text1))
    set2 = set(tokenize_for_jaccard(text2))
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    return len(set1 & set2) / len(set1 | set2)


def load_embedding_model(model_name: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()
    return tokenizer, model


def embed_texts(tokenizer, model, device: str, texts: List[str]) -> torch.Tensor:
    with torch.no_grad():
        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        outputs = model(**inputs)
        hidden = outputs.last_hidden_state
        mask = inputs["attention_mask"].unsqueeze(-1)
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
        pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
        return pooled.detach().cpu()


def cosine_similarity(emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
    return float((emb_a * emb_b).sum().item())


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
    sample_count = min(num_samples, len(pool))
    return random.sample(pool, sample_count)


def write_jsonl(path: str, rows: List[Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_summary(path: str, summary: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)


def summarize_results(rows: List[Dict]) -> Dict:
    summary: Dict[str, Dict] = {}
    for row in rows:
        scenario = row["scenario"]
        stats = summary.setdefault(
            scenario,
            {
                "total": 0,
                "correct": 0,
                "similarity_jaccard": [],
                "similarity_embedding": [],
            },
        )
        stats["total"] += 1
        stats["correct"] += int(bool(row.get("match")))
        if row.get("score_jaccard") is not None:
            stats["similarity_jaccard"].append(row["score_jaccard"])
        if row.get("score_embedding_cosine") is not None:
            stats["similarity_embedding"].append(row["score_embedding_cosine"])

    def agg_sim(values: List[float]) -> Dict:
        if not values:
            return {"avg": None, "min": None, "max": None}
        return {
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
        }

    assessor_correct = summary.get("assessor_correct", {"total": 0, "correct": 0})
    assessor_incorrect = summary.get("assessor_incorrect", {"total": 0, "correct": 0})
    injector_correct = summary.get("injector_correct", {"total": 0, "correct": 0})
    injector_incorrect = summary.get("injector_incorrect", {"total": 0, "correct": 0})

    total_assessor = assessor_correct["total"] + assessor_incorrect["total"]
    correct_assessor = assessor_correct["correct"] + assessor_incorrect["correct"]
    total_injector = injector_correct["total"] + injector_incorrect["total"]
    correct_injector = injector_correct["correct"] + injector_incorrect["correct"]

    return {
        "assessor": {
            "correct_notes": {
                "samples": assessor_correct["total"],
                "correct": assessor_correct["correct"],
                "accuracy": (assessor_correct["correct"] / assessor_correct["total"]) if assessor_correct["total"] else 0.0,
            },
            "incorrect_notes": {
                "samples": assessor_incorrect["total"],
                "correct": assessor_incorrect["correct"],
                "accuracy": (assessor_incorrect["correct"] / assessor_incorrect["total"]) if assessor_incorrect["total"] else 0.0,
            },
            "overall": {
                "samples": total_assessor,
                "correct": correct_assessor,
                "accuracy": (correct_assessor / total_assessor) if total_assessor else 0.0,
            },
        },
        "injector": {
            "generate_correct": {
                "samples": injector_correct["total"],
                "correct": injector_correct["correct"],
                "accuracy": (injector_correct["correct"] / injector_correct["total"]) if injector_correct["total"] else 0.0,
                "similarity_jaccard": agg_sim(summary.get("injector_correct", {}).get("similarity_jaccard", [])),
                "similarity_embedding": agg_sim(summary.get("injector_correct", {}).get("similarity_embedding", [])),
            },
            "inject_errors": {
                "samples": injector_incorrect["total"],
                "correct": injector_incorrect["correct"],
                "accuracy": (injector_incorrect["correct"] / injector_incorrect["total"]) if injector_incorrect["total"] else 0.0,
                "similarity_jaccard": agg_sim(summary.get("injector_incorrect", {}).get("similarity_jaccard", [])),
                "similarity_embedding": agg_sim(summary.get("injector_incorrect", {}).get("similarity_embedding", [])),
            },
            "overall": {
                "samples": total_injector,
                "correct": correct_injector,
                "accuracy": (correct_injector / total_injector) if total_injector else 0.0,
                "similarity_jaccard": agg_sim(
                    (summary.get("injector_correct", {}).get("similarity_jaccard", []) +
                     summary.get("injector_incorrect", {}).get("similarity_jaccard", []))
                ),
                "similarity_embedding": agg_sim(
                    (summary.get("injector_correct", {}).get("similarity_embedding", []) +
                     summary.get("injector_incorrect", {}).get("similarity_embedding", []))
                ),
            },
        },
    }


def main() -> None:
    args = parse_args()
    scenarios = [s.strip() for s in args.scenarios.split(",") if s.strip()]
    random.seed(args.seed)
    run_name = args.run_name or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(args.output_dir, f"quick_test_{run_name}.jsonl")
    summary_path = os.path.join(args.output_dir, f"quick_test_{run_name}_summary.json")
    os.makedirs(args.output_dir, exist_ok=True)

    records = []
    if args.jsonl_file:
        records = load_records(args.jsonl_file)
        random.shuffle(records)

    if not records and not args.input_note:
        raise SystemExit("Provide --jsonl-file or --input-note.")

    assessor_prompts = load_assessor_prompts(args.assessor_prompt_file)

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

    embedder = None
    embedder_tokenizer = None
    embed_device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.disable_embedding:
        embedder_tokenizer, embedder = load_embedding_model(args.embedding_model, embed_device)

    results_rows: List[Dict] = []
    scenario_stats: Dict[str, Dict[str, int]] = {}
    scenario_samples_map: Dict[str, List[Dict]] = {}

    total_examples = 0
    for scenario in scenarios:
        samples = scenario_samples(records, scenario, args.num_samples)
        if not samples and args.input_note:
            samples = [{"correct_note": args.input_note, "incorrect_note": args.input_note}]
        scenario_samples_map[scenario] = samples
        total_examples += len(samples)

    with open(output_path, "a", encoding="utf-8") as out_handle:
        with tqdm(total=total_examples, desc="Inference", unit="ex") as pbar:
            for scenario in scenarios:
                samples = scenario_samples_map[scenario]
                correct = 0
                total = 0

                prompts: List[str] = []
                metas: List[Dict] = []
                for record in samples:
                    if scenario == "assessor_correct":
                        note = record.get("correct_note")
                        expected = "CORRECT"
                        prompt_intent = "Assess note correctness."
                        messages = build_messages("assessor", note, prompt_intent, assessor_prompts)
                    elif scenario == "assessor_incorrect":
                        note = record.get("incorrect_note")
                        expected = "INCORRECT"
                        prompt_intent = "Assess note correctness."
                        messages = build_messages("assessor", note, prompt_intent, assessor_prompts)
                    elif scenario == "injector_correct":
                        note = record.get("correct_note")
                        expected = "CORRECT"
                        prompt_intent = args.prompt_intent
                        messages = build_messages("injector", note, prompt_intent, assessor_prompts)
                    else:
                        note = record.get("correct_note")
                        expected = "INCORRECT"
                        error_type = (record.get("error_type") or "").strip()
                        if error_type:
                            prompt_intent = f"Introduce a {error_type} error while keeping the note realistic."
                        else:
                            prompt_intent = "Introduce a subtle clinical error while keeping the note realistic."
                        messages = build_messages("injector", note, prompt_intent, assessor_prompts)

                    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    prompts.append(prompt)
                    metas.append(
                        {
                            "record": record,
                            "note": note,
                            "expected": expected,
                            "prompt_intent": prompt_intent,
                            "is_injector": scenario.startswith("injector"),
                            "prompt": prompt,
                        }
                    )

                for start in range(0, len(prompts), args.batch_size):
                    batch_prompts = prompts[start:start + args.batch_size]
                    batch_metas = metas[start:start + args.batch_size]
                    inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(model.device)
                    input_lengths = inputs["attention_mask"].sum(1).tolist()

                    with torch.no_grad():
                        outputs = model.generate(
                            **inputs,
                            max_new_tokens=args.max_new_tokens,
                            do_sample=True,
                            temperature=args.temperature,
                            top_p=args.top_p,
                            pad_token_id=tokenizer.eos_token_id,
                        )

                    for idx, meta in enumerate(batch_metas):
                        generated_tokens = outputs[idx][input_lengths[idx]:]
                        generated = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        predicted = extract_final_answer(generated)
                        if predicted is None:
                            full_text = tokenizer.decode(outputs[idx], skip_special_tokens=True)
                            predicted = extract_final_answer(full_text)
                        if predicted is None and args.force_answer:
                            predicted = force_answer_from_prompt(
                                meta["prompt"],
                                tokenizer,
                                model,
                                model.device,
                            )
                        is_correct = predicted == meta["expected"]
                        total += 1
                        correct += int(is_correct)

                        generated_note = None
                        score_jaccard = None
                        score_embedding = None

                        if meta["is_injector"]:
                            generated_note = extract_generated_note(generated)

                        if generated_note:
                            score_jaccard = jaccard_similarity(meta["note"], generated_note)
                            if embedder is not None and embedder_tokenizer is not None:
                                embeddings = embed_texts(
                                    embedder_tokenizer,
                                        embedder,
                                        embed_device,
                                        [meta["note"], generated_note],
                                    )
                                    score_embedding = cosine_similarity(embeddings[0], embeddings[1])

                        row = {
                            "run_name": run_name,
                            "scenario": scenario,
                            "task_group": "injector" if meta["is_injector"] else "assessor",
                            "note_id": meta["record"].get("note_id"),
                            "error_type": meta["record"].get("error_type"),
                            "prompt_intent": meta["prompt_intent"],
                            "original_note": meta["note"],
                            "generated_note": generated_note,
                            "expected": meta["expected"],
                            "predicted": predicted,
                            "match": is_correct,
                            "score_jaccard": score_jaccard,
                            "score_embedding_cosine": score_embedding,
                            "raw_output": generated,
                        }
                        results_rows.append(row)
                        out_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        out_handle.flush()
                        pbar.update(1)

                scenario_stats[scenario] = {"total": total, "correct": correct}

    write_summary(summary_path, summarize_results(results_rows))
    print(f"Wrote {len(results_rows)} rows to {output_path}")
    print(f"Wrote summary to {summary_path}")
    for scenario, stats in scenario_stats.items():
        total = stats["total"]
        correct = stats["correct"]
        acc = (correct / total * 100) if total else 0.0
        print(f"{scenario}: {correct}/{total} ({acc:.1f}%)")


if __name__ == "__main__":
    main()
