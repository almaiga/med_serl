#!/usr/bin/env python3
"""Benchmark plain pair-judge vs the existing UMLS-backed judge pipeline.

This benchmark evaluates sentence-pair adjudication, which matches the current
agentic judge design:
  - MEDEC error pairs: modified sentence should be judged as FAIL
  - benign-change pairs: modified sentence should be judged as PASS

Variants:
  1. plain: direct pair adjudication from original vs modified sentence
  2. rag: existing extraction -> UMLS/RxNorm retrieval -> adjudication pipeline

The RAG path reuses the prompts already defined in
`configs/prompts/agentic_judge_prompts.json`.

Usage:
    export JUDGE_VLLM_URL=http://<host>:8002/v1/chat/completions
    python3 scripts/self_play/benchmark_judge_variants.py

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
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import aiohttp

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
from scripts.self_play.utils import find_error_sentence_id


DEFAULT_JUDGE_URL = "http://localhost:8002/v1/chat/completions"
DEFAULT_JUDGE_SPECS = ["qwen3-4b=Qwen/Qwen3-4B"]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "self_play" / "judge_benchmark"

JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)
JSON_ARR_RE = re.compile(r"\[.*\]", re.DOTALL)


@dataclass
class Example:
    dataset_name: str
    note_id: str
    task_type: str
    subtype: str
    gold_verdict: str
    gold_sentence_id: Optional[int]
    gold_sentence_text: str
    original_note: str
    modified_note: str
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
        description="Benchmark plain pair-judge vs UMLS-backed adjudication."
    )
    parser.add_argument("--judge-url", default=os.getenv("JUDGE_VLLM_URL", DEFAULT_JUDGE_URL))
    parser.add_argument(
        "--judge",
        action="append",
        default=None,
        help="Repeatable label=model spec, e.g. --judge qwen3-4b=Qwen/Qwen3-4B",
    )
    parser.add_argument(
        "--mode",
        action="append",
        choices=("plain", "rag"),
        default=None,
        help="Repeatable benchmark mode. Defaults to both plain and rag.",
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
                gold_sentence_id=find_error_sentence_id(incorrect_note, error_sentence),
                gold_sentence_text=error_sentence,
                original_note=str(row.get("correct_note", "")),
                modified_note=incorrect_note,
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
                gold_sentence_id=find_error_sentence_id(modified_note, modified_sentence),
                gold_sentence_text=modified_sentence,
                original_note=original_note,
                modified_note=modified_note,
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
    modes = args.mode or ["plain", "rag"]
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


def adjudication_assessor_prediction() -> str:
    return "CORRECT (assume the modified sentence is clinically correct)"


async def llm_generate(
    session: aiohttp.ClientSession,
    judge_url: str,
    judge_model: str,
    messages: list[dict[str, str]],
    timeout: float,
    *,
    max_tokens: int = 512,
    temperature: float = 0.1,
    top_p: float = 0.95,
) -> str:
    payload = {
        "model": judge_model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": top_p,
    }
    async with session.post(
        judge_url,
        json=payload,
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as response:
        response.raise_for_status()
        data = await response.json()
        return data["choices"][0]["message"]["content"].strip()


async def extract_entities_for_sentence(
    session: aiohttp.ClientSession,
    judge_url: str,
    judge_model: str,
    sentence: str,
    timeout: float,
) -> list[dict[str, str]]:
    if not sentence.strip():
        return []
    params = get_model_params("extraction")
    messages = [
        {"role": "system", "content": get_extraction_system_prompt()},
        {"role": "user", "content": get_extraction_user_template().format(sentence=sentence)},
    ]
    raw = await llm_generate(
        session=session,
        judge_url=judge_url,
        judge_model=judge_model,
        messages=messages,
        timeout=timeout,
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
        assessor_prediction=adjudication_assessor_prediction(),
        evidence_json=evidence_text,
    )
    return [
        {"role": "system", "content": get_adjudication_system_prompt()},
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
    session: aiohttp.ClientSession,
    example: Example,
    variant: Variant,
    judge_url: str,
    timeout: float,
) -> dict[str, Any]:
    raw = await llm_generate(
        session=session,
        judge_url=judge_url,
        judge_model=variant.model,
        messages=build_plain_messages(example),
        timeout=timeout,
        max_tokens=256,
        temperature=0.1,
        top_p=0.95,
    )
    return parse_plain_prediction(raw)


async def evaluate_rag(
    session: aiohttp.ClientSession,
    example: Example,
    variant: Variant,
    judge_url: str,
    timeout: float,
    max_entities: int,
) -> dict[str, Any]:
    entities_original = await extract_entities_for_sentence(
        session=session,
        judge_url=judge_url,
        judge_model=variant.model,
        sentence=example.original_sentence,
        timeout=timeout,
    )
    entities_modified = await extract_entities_for_sentence(
        session=session,
        judge_url=judge_url,
        judge_model=variant.model,
        sentence=example.modified_sentence,
        timeout=timeout,
    )
    entities = dedupe_entities(entities_original + entities_modified, max_entities=max_entities)
    evidence_objects = await gather_evidence_batch(session, entities)
    evidence = [asdict(item) for item in evidence_objects]
    evidence_text = format_evidence_for_prompt(evidence)
    params = get_model_params("adjudication")
    raw = await llm_generate(
        session=session,
        judge_url=judge_url,
        judge_model=variant.model,
        messages=build_rag_messages(example, evidence_text),
        timeout=timeout,
        max_tokens=int(params.get("max_tokens", 512)),
        temperature=float(params.get("temperature", 0.1)),
        top_p=float(params.get("top_p", 0.95)),
    )
    return parse_rag_prediction(raw, entities, evidence)


def score_prediction(example: Example, prediction: dict[str, Any]) -> dict[str, Any]:
    verdict_correct = prediction.get("verdict") == example.gold_verdict
    evidence_hit = bool(prediction.get("evidence"))
    cited = bool(prediction.get("cuis_cited"))
    return {
        "verdict_correct": verdict_correct,
        "evidence_hit": evidence_hit,
        "cui_cited": cited,
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
        "judge_url": args.judge_url,
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


def print_dataset_summary(errors: list[Example], benign: list[Example], variants: list[Variant]) -> None:
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
    session_timeout = aiohttp.ClientTimeout(total=args.timeout)

    async with aiohttp.ClientSession(timeout=session_timeout) as session:
        async def process_one(example: Example) -> dict[str, Any]:
            async with semaphore:
                variant_results = {}
                for variant in variants:
                    if variant.mode == "plain":
                        prediction = await evaluate_plain(
                            session=session,
                            example=example,
                            variant=variant,
                            judge_url=args.judge_url,
                            timeout=args.timeout,
                        )
                    else:
                        prediction = await evaluate_rag(
                            session=session,
                            example=example,
                            variant=variant,
                            judge_url=args.judge_url,
                            timeout=args.timeout,
                            max_entities=args.max_entities,
                        )
                    variant_results[variant.name] = {
                        "prediction": prediction,
                        **score_prediction(example, prediction),
                    }

                return {
                    "dataset_name": example.dataset_name,
                    "note_id": example.note_id,
                    "task_type": example.task_type,
                    "subtype": example.subtype,
                    "gold_verdict": example.gold_verdict,
                    "gold_sentence_id": example.gold_sentence_id,
                    "gold_sentence_text": example.gold_sentence_text,
                    "variants": variant_results,
                }

        return await asyncio.gather(*(process_one(example) for example in examples))


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
            "judge_url": args.judge_url,
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

    if not args.judge_url:
        raise SystemExit("JUDGE_VLLM_URL / --judge-url is required for full benchmark.")

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
