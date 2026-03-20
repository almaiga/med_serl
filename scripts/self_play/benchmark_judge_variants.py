#!/usr/bin/env python3
"""Benchmark the judge in isolation on sentence-pair adjudication.

This benchmark evaluates sentence-pair adjudication, which matches the current
agentic judge design:
  - MEDEC error pairs: modified sentence should be judged as FAIL
  - benign-change pairs: modified sentence should be judged as PASS

Variants:
  1. plain: direct pair adjudication from original vs modified sentence
  2. rag: the existing extraction -> UMLS/RxNorm retrieval -> adjudication judge

The RAG path reuses the prompts already defined in
`configs/prompts/agentic_judge_prompts.json`.

Usage:
    python3 scripts/self_play/benchmark_judge_variants.py \
        --judge qwen3-4b=Qwen/Qwen3-4B

    # Compare multiple judge models in one run
    python3 scripts/self_play/benchmark_judge_variants.py \
        --judge qwen3-4b=Qwen/Qwen3-4B \
        --judge qwen3-8b=Qwen/Qwen3-8B

    # Dry-run to validate datasets and variant expansion only
    python3 scripts/self_play/benchmark_judge_variants.py --dry-run
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.self_play.judge_prompts import (
    format_evidence_for_prompt,
    get_adjudication_system_prompt,
    get_adjudication_user_template,
    get_extraction_system_prompt,
    get_extraction_user_template,
    get_model_params,
)
from scripts.self_play.umls_async import gather_evidence_batch

DEFAULT_JUDGE_SPECS = ["qwen3-4b=Qwen/Qwen3-4B"]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "self_play" / "judge_benchmark"

JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
JSON_ARR_RE = re.compile(r"\[.*\]", re.DOTALL)
_LOCAL_RUNNERS: dict[str, "LocalJudgeRunner"] = {}


@dataclass
class Example:
    dataset_name: str
    note_id: str
    task_type: str
    subtype: str
    gold_verdict: str
    original_note: str
    original_sentence: str
    modified_sentence: str


@dataclass(frozen=True)
class Variant:
    label: str
    model: str
    mode: str

    @property
    def name(self) -> str:
        return f"{self.label}_{self.mode}"


def default_error_datasets() -> list[Path]:
    return sorted((PROJECT_ROOT / "data_processed" / "medec_paired").glob("**/medec_pairs_*.jsonl"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark the judge only on paired sentence adjudication."
    )
    parser.add_argument(
        "--judge",
        action="append",
        default=None,
        help="Repeatable label=model spec, e.g. --judge qwen3-4b=Qwen/Qwen3-4B",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=("plain", "rag", "note_plain", "note_rag", "auto_plain", "auto_rag"),
        default=None,
        help="Repeatable benchmark mode.",
    )
    parser.add_argument(
        "--error-dataset",
        action="append",
        default=None,
        help="Repeatable path to MEDEC paired dataset.",
    )
    parser.add_argument(
        "--benign-dataset",
        action="append",
        default=None,
        help="Repeatable path to benign-change dataset.",
    )
    parser.add_argument("--max-error", type=int, default=100, help="Max sampled error examples.")
    parser.add_argument("--max-benign", type=int, default=100, help="Max sampled benign examples.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--max-entities", type=int, default=12)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--prefix", default=None)
    parser.add_argument("--dry-run", action="store_true", help="Only load data and write summary.")
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_error_examples(path: Path) -> list[Example]:
    rows = load_jsonl(path)
    dataset_name = str(path.relative_to(PROJECT_ROOT))
    examples = []
    for row in rows:
        incorrect_note = str(row.get("incorrect_note", ""))
        corrected_sentence = str(row.get("corrected_sentence", ""))
        error_sentence = str(row.get("error_sentence", ""))
        if not incorrect_note or not corrected_sentence or not error_sentence:
            continue
        examples.append(
            Example(
                dataset_name=dataset_name,
                note_id=str(row.get("note_id", "")),
                task_type="error",
                subtype=str(row.get("error_type", "unknown")),
                gold_verdict="FAIL",
                original_note=str(row.get("correct_note", "")),
                original_sentence=corrected_sentence,
                modified_sentence=error_sentence,
            )
        )
    return examples


def load_benign_examples(path: Path) -> list[Example]:
    rows = load_jsonl(path)
    dataset_name = str(path.relative_to(PROJECT_ROOT))
    examples = []
    for row in rows:
        if row.get("change_made") is False:
            continue
        if "verified" in row and not row.get("verified"):
            continue
        original_note = str(row.get("original_note", ""))
        modified_note = str(row.get("modified_note", ""))
        original_sentence = str(row.get("original_sentence", ""))
        modified_sentence = str(row.get("modified_sentence", ""))
        if not original_note or not modified_note or not original_sentence or not modified_sentence:
            continue
        examples.append(
            Example(
                dataset_name=dataset_name,
                note_id=str(row.get("note_id", "")),
                task_type="benign",
                subtype=str(row.get("change_type", "unknown")),
                gold_verdict="PASS",
                original_note=original_note,
                original_sentence=original_sentence,
                modified_sentence=modified_sentence,
            )
        )
    return examples


def sample_examples(examples: list[Example], max_n: int, seed: int) -> list[Example]:
    if len(examples) <= max_n:
        return examples
    rng = random.Random(seed)
    sampled = list(examples)
    rng.shuffle(sampled)
    return sampled[:max_n]


def parse_variants(args: argparse.Namespace) -> list[Variant]:
    judge_specs = args.judge or DEFAULT_JUDGE_SPECS
    modes = args.mode or ["plain", "rag", "note_plain", "note_rag", "auto_plain", "auto_rag"]
    variants = []
    for spec in judge_specs:
        if "=" not in spec:
            raise SystemExit(f"Invalid --judge spec '{spec}'. Expected label=model.")
        label, model = spec.split("=", 1)
        label = label.strip()
        model = model.strip()
        if not label or not model:
            raise SystemExit(f"Invalid --judge spec '{spec}'. Expected label=model.")
        for mode in modes:
            variants.append(Variant(label=label, model=model, mode=mode))
    return variants


def robust_json_object(text: str) -> Optional[dict[str, Any]]:
    text = text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```", "", text)
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    match = JSON_OBJ_RE.search(text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        return None


def robust_json_array(text: str) -> Optional[list[Any]]:
    text = text.strip()
    text = re.sub(r"```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```", "", text)
    if not text:
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return parsed
    except json.JSONDecodeError:
        pass

    match = JSON_ARR_RE.search(text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, list) else None
    except json.JSONDecodeError:
        return None


def normalize_verdict(value: Any) -> str:
    if value is None:
        return "ABSTAIN"
    text = str(value).strip().upper()
    if text in {"PASS", "CORRECT", "BENIGN", "NO_ERROR"}:
        return "PASS"
    if text in {"FAIL", "ERROR", "INCORRECT"}:
        return "FAIL"
    return "ABSTAIN"


def normalize_score(value: Any) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, min(1.0, score))


def metric_div(num: int, denom: int) -> Optional[float]:
    return num / denom if denom else None


def pair_only_adjudication_context() -> str:
    return (
        "PAIR-ONLY BENCHMARK. There is no assessor prediction in this evaluation. "
        "Judge only whether the MODIFIED sentence preserves the clinical meaning "
        "of the ORIGINAL sentence."
    )


HIGH_RISK_SUBTYPES = {
    "diagnosis",
    "management",
    "treatment",
    "pharmacotherapy",
    "causalorganism",
    "contraindication",
    "prognosis",
    "disposition",
}

CONTEXT_SENSITIVE_RE = re.compile(
    r"\b("
    r"he|she|his|her|their|patient|this|that|these|those|it|former|latter|"
    r"after|before|during|despite|because|due to|therefore|however|"
    r"history|allergy|pregnan|renal|hepatic|comorbid|family history"
    r")\b",
    re.IGNORECASE,
)


class LocalJudgeRunner:
    def __init__(self, model_path: str):
        from scripts.inference_error_detection import detect_model_type, load_model_and_tokenizer

        self.model_path = model_path
        self.model_type = detect_model_type(model_path)
        self.model, self.tokenizer = load_model_and_tokenizer(model_path, self.model_type)
        self.lock = threading.Lock()

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        import torch

        with self.lock:
            prompt_kwargs = {"tokenize": False, "add_generation_prompt": True}
            try:
                prompt_kwargs["enable_thinking"] = False
                prompt = self.tokenizer.apply_chat_template(messages, **prompt_kwargs)
            except TypeError:
                prompt_kwargs.pop("enable_thinking", None)
                prompt = self.tokenizer.apply_chat_template(messages, **prompt_kwargs)

            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "do_sample": temperature > 0,
                "temperature": temperature if temperature > 0 else None,
                "top_p": top_p if temperature > 0 else None,
                "pad_token_id": self.tokenizer.pad_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            gen_kwargs = {k: v for k, v in gen_kwargs.items() if v is not None}

            with torch.no_grad():
                outputs = self.model.generate(**inputs, **gen_kwargs)

            output_ids = outputs[0, inputs.input_ids.shape[1]:]
            return self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()


def get_local_runner(model_path: str) -> LocalJudgeRunner:
    runner = _LOCAL_RUNNERS.get(model_path)
    if runner is None:
        print(f"Loading local judge model: {model_path}")
        runner = LocalJudgeRunner(model_path)
        _LOCAL_RUNNERS[model_path] = runner
    return runner


async def llm_generate(
    judge_model: str,
    messages: list[dict[str, str]],
    *,
    max_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.95,
) -> str:
    runner = get_local_runner(judge_model)
    return await asyncio.to_thread(
        runner.generate,
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )


async def extract_entities_for_sentence(
    judge_model: str,
    sentence: str,
) -> list[dict[str, str]]:
    if not sentence.strip():
        return []
    params = get_model_params("extraction")
    messages = [
        {"role": "system", "content": get_extraction_system_prompt()},
        {"role": "user", "content": get_extraction_user_template().format(sentence=sentence)},
    ]
    raw = await llm_generate(
        judge_model=judge_model,
        messages=messages,
        max_tokens=int(params.get("max_tokens", 768)),
        temperature=float(params.get("temperature", 0.1)),
        top_p=float(params.get("top_p", 0.95)),
    )
    parsed = robust_json_array(raw) or []
    entities = []
    for item in parsed:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name", "")).strip()
        etype = str(item.get("type", "")).strip()
        if name:
            entities.append({"name": name, "type": etype})
    return entities


def dedupe_entities(entities: list[dict[str, str]], max_entities: int) -> list[dict[str, str]]:
    deduped = []
    seen = set()
    for entity in entities:
        key = (entity.get("name", "").lower(), entity.get("type", "").lower())
        if not key[0] or key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
        if len(deduped) >= max_entities:
            break
    return deduped


def build_plain_messages(example: Example) -> list[dict[str, str]]:
    system_prompt = (
        "/no_think You are a clinical note adjudication judge. "
        "Compare the ORIGINAL sentence and the MODIFIED sentence. "
        "PASS means the modification is clinically equivalent or benign. "
        "FAIL means the modification introduces a clinically meaningful medical error. "
        "Output ONLY JSON with keys verdict, score, reasoning. "
        'Use verdict values "PASS" or "FAIL".'
    )
    user_prompt = (
        "=== ORIGINAL SENTENCE ===\n"
        f"{example.original_sentence}\n\n"
        "=== MODIFIED SENTENCE ===\n"
        f"{example.modified_sentence}\n\n"
        "Return JSON only:\n"
        '{"verdict":"PASS|FAIL","score":0.0,"reasoning":"brief reason"}'
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_rag_messages(example: Example, evidence_text: str) -> list[dict[str, str]]:
    user_prompt = get_adjudication_user_template().format(
        original_sentence=example.original_sentence,
        modified_sentence=example.modified_sentence,
        assessor_prediction=pair_only_adjudication_context(),
        evidence_json=evidence_text,
    )
    return [
        {"role": "system", "content": get_adjudication_system_prompt()},
        {"role": "user", "content": user_prompt},
    ]


def build_note_plain_messages(example: Example) -> list[dict[str, str]]:
    system_prompt = (
        "/no_think You are a clinical note adjudication judge. "
        "You are given the ORIGINAL NOTE for patient context, the TARGET ORIGINAL SENTENCE, "
        "and a PROPOSED MODIFIED SENTENCE. "
        "Decide whether replacing the target sentence with the modified sentence preserves "
        "the clinical meaning of the note for this patient. "
        "PASS means clinically equivalent or benign in context. "
        "FAIL means the change introduces a clinically meaningful medical error in context. "
        "Output ONLY JSON with keys verdict, score, reasoning. "
        'Use verdict values "PASS" or "FAIL".'
    )
    user_prompt = (
        "=== ORIGINAL NOTE ===\n"
        f"{example.original_note}\n\n"
        "=== TARGET ORIGINAL SENTENCE ===\n"
        f"{example.original_sentence}\n\n"
        "=== PROPOSED MODIFIED SENTENCE ===\n"
        f"{example.modified_sentence}\n\n"
        "Return JSON only:\n"
        '{"verdict":"PASS|FAIL","score":0.0,"reasoning":"brief reason"}'
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def build_note_rag_messages(example: Example, evidence_text: str) -> list[dict[str, str]]:
    system_prompt = (
        "/no_think You are a clinical note adjudication judge with UMLS evidence. "
        "You are given the ORIGINAL NOTE for patient context, the TARGET ORIGINAL SENTENCE, "
        "and a PROPOSED MODIFIED SENTENCE. "
        "Decide whether replacing the target sentence with the modified sentence preserves "
        "the clinical meaning of the note for this patient. "
        "PASS means clinically equivalent or benign in context. "
        "FAIL means the change introduces a clinically meaningful medical error in context. "
        "If you cite no CUIs, score must be 0.0. "
        "Output ONLY JSON with keys verdict, score, reasoning, cuis_cited."
    )
    user_prompt = (
        "=== ORIGINAL NOTE ===\n"
        f"{example.original_note}\n\n"
        "=== TARGET ORIGINAL SENTENCE ===\n"
        f"{example.original_sentence}\n\n"
        "=== PROPOSED MODIFIED SENTENCE ===\n"
        f"{example.modified_sentence}\n\n"
        "=== UMLS EVIDENCE ===\n"
        f"{evidence_text}\n\n"
        "Return JSON only:\n"
        '{"verdict":"PASS|FAIL","score":0.0,"reasoning":"brief reason","cuis_cited":["C..."]}'
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def parse_plain_prediction(raw: str) -> dict[str, Any]:
    parsed = robust_json_object(raw) or {}
    return {
        "raw": raw,
        "verdict": normalize_verdict(parsed.get("verdict")),
        "score": normalize_score(parsed.get("score")),
        "reasoning": str(parsed.get("reasoning", "")),
        "entities": [],
        "evidence": [],
        "cuis_cited": [],
    }


def parse_rag_prediction(raw: str, entities: list[dict[str, str]], evidence: list[dict[str, Any]]) -> dict[str, Any]:
    parsed = robust_json_object(raw) or {}
    cuis = parsed.get("cuis_cited", [])
    if not isinstance(cuis, list):
        cuis = []
    score = normalize_score(parsed.get("score"))
    if not cuis:
        score = 0.0
    return {
        "raw": raw,
        "verdict": normalize_verdict(parsed.get("verdict")),
        "score": score,
        "reasoning": str(parsed.get("reasoning", "")),
        "entities": entities,
        "evidence": evidence,
        "cuis_cited": cuis,
    }


async def evaluate_plain(
    example: Example,
    variant: Variant,
) -> dict[str, Any]:
    raw = await llm_generate(
        judge_model=variant.model,
        messages=build_plain_messages(example),
        max_tokens=256,
        temperature=0.1,
        top_p=0.95,
    )
    return parse_plain_prediction(raw)


async def evaluate_rag(
    example: Example,
    variant: Variant,
    max_entities: int,
) -> dict[str, Any]:
    entities_original = await extract_entities_for_sentence(
        judge_model=variant.model,
        sentence=example.original_sentence,
    )
    entities_modified = await extract_entities_for_sentence(
        judge_model=variant.model,
        sentence=example.modified_sentence,
    )
    entities = dedupe_entities(entities_original + entities_modified, max_entities=max_entities)
    import aiohttp
    async with aiohttp.ClientSession() as session:
        evidence_objects = await gather_evidence_batch(session, entities)
    evidence = [asdict(item) for item in evidence_objects]
    evidence_text = format_evidence_for_prompt(evidence)
    params = get_model_params("adjudication")
    raw = await llm_generate(
        judge_model=variant.model,
        messages=build_rag_messages(example, evidence_text),
        max_tokens=int(params.get("max_tokens", 512)),
        temperature=float(params.get("temperature", 0.1)),
        top_p=float(params.get("top_p", 0.95)),
    )
    return parse_rag_prediction(raw, entities, evidence)


async def evaluate_note_plain(
    example: Example,
    variant: Variant,
) -> dict[str, Any]:
    raw = await llm_generate(
        judge_model=variant.model,
        messages=build_note_plain_messages(example),
        max_tokens=384,
        temperature=0.1,
        top_p=0.95,
    )
    pred = parse_plain_prediction(raw)
    pred["context_mode"] = "note"
    return pred


async def evaluate_note_rag(
    example: Example,
    variant: Variant,
    max_entities: int,
) -> dict[str, Any]:
    entities_original = await extract_entities_for_sentence(
        judge_model=variant.model,
        sentence=example.original_sentence,
    )
    entities_modified = await extract_entities_for_sentence(
        judge_model=variant.model,
        sentence=example.modified_sentence,
    )
    entities = dedupe_entities(entities_original + entities_modified, max_entities=max_entities)
    import aiohttp
    async with aiohttp.ClientSession() as session:
        evidence_objects = await gather_evidence_batch(session, entities)
    evidence = [asdict(item) for item in evidence_objects]
    evidence_text = format_evidence_for_prompt(evidence)
    raw = await llm_generate(
        judge_model=variant.model,
        messages=build_note_rag_messages(example, evidence_text),
        max_tokens=640,
        temperature=0.1,
        top_p=0.95,
    )
    pred = parse_rag_prediction(raw, entities, evidence)
    pred["context_mode"] = "note"
    return pred


def is_context_sensitive(example: Example) -> bool:
    if example.task_type == "error" and example.subtype.lower() in HIGH_RISK_SUBTYPES:
        return True
    text = f"{example.original_sentence} {example.modified_sentence}"
    return bool(CONTEXT_SENSITIVE_RE.search(text))


def should_escalate(example: Example, prediction: dict[str, Any]) -> bool:
    if prediction.get("verdict") == "ABSTAIN":
        return True
    if prediction.get("score", 0.0) < 0.80:
        return True
    if is_context_sensitive(example):
        return True
    if prediction.get("evidence") and not prediction.get("cuis_cited"):
        return True
    return False


async def evaluate_auto_plain(
    example: Example,
    variant: Variant,
) -> dict[str, Any]:
    pair_pred = await evaluate_plain(example=example, variant=variant)
    pair_pred["context_mode"] = "pair"
    pair_pred["escalated"] = False
    if should_escalate(example, pair_pred):
        note_pred = await evaluate_note_plain(example=example, variant=variant)
        note_pred["escalated"] = True
        note_pred["fallback_prediction"] = pair_pred
        return note_pred
    return pair_pred


async def evaluate_auto_rag(
    example: Example,
    variant: Variant,
    max_entities: int,
) -> dict[str, Any]:
    pair_pred = await evaluate_rag(example=example, variant=variant, max_entities=max_entities)
    pair_pred["context_mode"] = "pair"
    pair_pred["escalated"] = False
    if should_escalate(example, pair_pred):
        note_pred = await evaluate_note_rag(example=example, variant=variant, max_entities=max_entities)
        note_pred["escalated"] = True
        note_pred["fallback_prediction"] = pair_pred
        return note_pred
    return pair_pred


def score_prediction(example: Example, prediction: dict[str, Any]) -> dict[str, Any]:
    verdict_correct = prediction.get("verdict") == example.gold_verdict
    evidence_hit = bool(prediction.get("evidence"))
    cited = bool(prediction.get("cuis_cited"))
    escalated = bool(prediction.get("escalated"))
    used_note_context = prediction.get("context_mode") == "note"
    return {
        "verdict_correct": verdict_correct,
        "evidence_hit": evidence_hit,
        "cui_cited": cited,
        "escalated": escalated,
        "used_note_context": used_note_context,
    }


def aggregate_metrics(records: list[dict[str, Any]], variant_name: str) -> dict[str, Any]:
    total = len(records)
    total_errors = sum(1 for row in records if row["task_type"] == "error")
    total_benign = sum(1 for row in records if row["task_type"] == "benign")

    verdict_correct = sum(1 for row in records if row["variants"][variant_name]["verdict_correct"])
    benign_correct = sum(
        1
        for row in records
        if row["task_type"] == "benign" and row["variants"][variant_name]["verdict_correct"]
    )
    error_correct = sum(
        1
        for row in records
        if row["task_type"] == "error" and row["variants"][variant_name]["verdict_correct"]
    )
    tp = sum(
        1
        for row in records
        if row["task_type"] == "error"
        and row["variants"][variant_name]["prediction"]["verdict"] == "FAIL"
    )
    fp = sum(
        1
        for row in records
        if row["task_type"] == "benign"
        and row["variants"][variant_name]["prediction"]["verdict"] == "FAIL"
    )
    fn = sum(
        1
        for row in records
        if row["task_type"] == "error"
        and row["variants"][variant_name]["prediction"]["verdict"] != "FAIL"
    )
    evidence_hits = sum(1 for row in records if row["variants"][variant_name]["evidence_hit"])
    cui_citations = sum(1 for row in records if row["variants"][variant_name]["cui_cited"])
    escalations = sum(1 for row in records if row["variants"][variant_name]["escalated"])
    note_context_uses = sum(1 for row in records if row["variants"][variant_name]["used_note_context"])
    avg_score = metric_div(
        sum(row["variants"][variant_name]["prediction"]["score"] for row in records),
        total,
    )

    precision = metric_div(tp, tp + fp)
    recall = metric_div(tp, tp + fn)
    f1 = None
    if precision is not None and recall is not None and precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)

    return {
        "n": total,
        "accuracy": metric_div(verdict_correct, total),
        "benign_accuracy": metric_div(benign_correct, total_benign),
        "error_accuracy": metric_div(error_correct, total_errors),
        "error_precision": precision,
        "error_recall": recall,
        "error_f1": f1,
        "avg_score": avg_score,
        "evidence_hit_rate": metric_div(evidence_hits, total),
        "cui_citation_rate": metric_div(cui_citations, total),
        "escalation_rate": metric_div(escalations, total),
        "note_context_rate": metric_div(note_context_uses, total),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def make_summary(records: list[dict[str, Any]], args: argparse.Namespace, variants: list[Variant]) -> dict[str, Any]:
    by_dataset: dict[str, dict[str, Any]] = {}
    for dataset_name in sorted({row["dataset_name"] for row in records}):
        ds_records = [row for row in records if row["dataset_name"] == dataset_name]
        by_dataset[dataset_name] = {
            variant.name: aggregate_metrics(ds_records, variant.name) for variant in variants
        }

    return {
        "created_at": datetime.now().isoformat(),
        "variants": [
            {"name": variant.name, "label": variant.label, "mode": variant.mode, "model": variant.model}
            for variant in variants
        ],
        "max_error": args.max_error,
        "max_benign": args.max_benign,
        "overall": {variant.name: aggregate_metrics(records, variant.name) for variant in variants},
        "by_dataset": by_dataset,
    }


def fmt_metric(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value:.3f}"


def print_dataset_summary(
    errors: list[Example],
    benign: list[Example],
    variants: list[Variant],
) -> None:
    print(f"Loaded {len(errors)} error examples and {len(benign)} benign examples.")
    print("Variants:")
    for variant in variants:
        print(f"  - {variant.name}: model={variant.model} mode={variant.mode}")
    if errors:
        print("  Error datasets:")
        counts = defaultdict(int)
        for ex in errors:
            counts[ex.dataset_name] += 1
        for name, count in sorted(counts.items()):
            print(f"    {name}: {count}")
    if benign:
        print("  Benign datasets:")
        counts = defaultdict(int)
        for ex in benign:
            counts[ex.dataset_name] += 1
        for name, count in sorted(counts.items()):
            print(f"    {name}: {count}")


async def run_benchmark(
    args: argparse.Namespace,
    examples: list[Example],
    variants: list[Variant],
) -> list[dict[str, Any]]:
    semaphore = asyncio.Semaphore(args.max_concurrency)
    async def process_one(example: Example) -> dict[str, Any]:
        async with semaphore:
            variant_results = {}
            for variant in variants:
                if variant.mode == "plain":
                    prediction = await evaluate_plain(
                        example=example,
                        variant=variant,
                    )
                elif variant.mode == "rag":
                    prediction = await evaluate_rag(
                        example=example,
                        variant=variant,
                        max_entities=args.max_entities,
                    )
                elif variant.mode == "note_plain":
                    prediction = await evaluate_note_plain(
                        example=example,
                        variant=variant,
                    )
                elif variant.mode == "note_rag":
                    prediction = await evaluate_note_rag(
                        example=example,
                        variant=variant,
                        max_entities=args.max_entities,
                    )
                elif variant.mode == "auto_plain":
                    prediction = await evaluate_auto_plain(
                        example=example,
                        variant=variant,
                    )
                elif variant.mode == "auto_rag":
                    prediction = await evaluate_auto_rag(
                        example=example,
                        variant=variant,
                        max_entities=args.max_entities,
                    )
                else:
                    raise ValueError(f"Unsupported mode: {variant.mode}")
                variant_results[variant.name] = {
                    "prediction": prediction,
                    **score_prediction(example, prediction),
                }

            return {
                "task": "judge_pair_verdict",
                "dataset_name": example.dataset_name,
                "note_id": example.note_id,
                "task_type": example.task_type,
                "subtype": example.subtype,
                "gold_verdict": example.gold_verdict,
                "original_note": example.original_note,
                "original_sentence": example.original_sentence,
                "modified_sentence": example.modified_sentence,
                "variants": variant_results,
            }

    tasks = [asyncio.create_task(process_one(example)) for example in examples]
    results: list[dict[str, Any]] = []
    with tqdm(total=len(tasks), desc="Benchmarking judge", unit="example") as pbar:
        for task in asyncio.as_completed(tasks):
            results.append(await task)
            pbar.update(1)
    return results


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def main() -> None:
    args = parse_args()
    variants = parse_variants(args)

    error_paths = [Path(p) for p in (args.error_dataset or default_error_datasets())]
    benign_paths = [Path(p) for p in (args.benign_dataset or [
        PROJECT_ROOT / "data_processed" / "benign_changes" / "benign_train_clean.jsonl"
    ])]

    if not error_paths:
        raise SystemExit("No error datasets found. Pass --error-dataset explicitly.")

    errors: list[Example] = []
    for path in error_paths:
        errors.extend(load_error_examples(path))
    benign: list[Example] = []
    for path in benign_paths:
        benign.extend(load_benign_examples(path))

    errors = sample_examples(errors, args.max_error, args.seed)
    benign = sample_examples(benign, args.max_benign, args.seed + 1)
    examples = errors + benign

    print_dataset_summary(errors, benign, variants)

    output_dir = args.output_dir
    if output_dir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = DEFAULT_OUTPUT_DIR / f"judge_ab_{stamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = args.prefix or "judge_ab"

    if args.dry_run:
        dry_summary = {
            "created_at": datetime.now().isoformat(),
            "dry_run": True,
            "n_examples": len(examples),
            "n_error": len(errors),
            "n_benign": len(benign),
            "variants": [
                {"name": variant.name, "label": variant.label, "mode": variant.mode, "model": variant.model}
                for variant in variants
            ],
            "datasets": sorted({ex.dataset_name for ex in examples}),
        }
        summary_path = output_dir / f"{prefix}_summary.json"
        summary_path.write_text(json.dumps(dry_summary, indent=2), encoding="utf-8")
        print(f"Dry-run summary written to {summary_path}")
        return

    results = asyncio.run(run_benchmark(args, examples, variants))
    summary = make_summary(results, args, variants)

    results_path = output_dir / f"{prefix}_results.jsonl"
    summary_path = output_dir / f"{prefix}_summary.json"
    write_jsonl(results_path, results)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nOverall metrics")
    for variant in variants:
        metrics = summary["overall"][variant.name]
        print(
            f"  {variant.name}: "
            f"acc={fmt_metric(metrics['accuracy'])} "
            f"benign={fmt_metric(metrics['benign_accuracy'])} "
            f"error={fmt_metric(metrics['error_accuracy'])} "
            f"f1={fmt_metric(metrics['error_f1'])} "
            f"cui={fmt_metric(metrics['cui_citation_rate'])}"
        )
    print(f"Results: {results_path}")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
