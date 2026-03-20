#!/usr/bin/env python3
"""
Generate missing MedRECT-style assessor chains for unmatched local SFT pairs.

Pipeline:
1. Select local SFT pairs that were not recovered from medrect-en-train
2. Generate reasoning with DeepSeek
3. Clean meta-references
4. Accept or reject based on exact gold answer
5. Write accepted/rejected raw MedRECT-style JSONL

Accepted outputs can be converted with prepare_medrect_sft.py.
"""

import argparse
import asyncio
import json
import logging
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.medrect.generate_injector_chains import DeepSeekR1Client, ReasoningCleaner
from scripts.self_play.utils import find_error_sentence_id, number_sentences, parse_assessor_answer


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_SFT_PATH = PROJECT_ROOT / "data_processed" / "medec_paired" / "train_val_split" / "sft_train.jsonl"
DEFAULT_MATCH_SUMMARY = PROJECT_ROOT / "data_processed" / "medrect" / "recovered_medrect_match_summary.json"
DEFAULT_PROMPT_CONFIG = PROJECT_ROOT / "configs" / "prompts" / "sft" / "medrect_assessor_reasoning_prompts.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data_processed" / "medrect"


@dataclass
class PairRecord:
    note_id: str
    correct_note: str
    incorrect_note: str
    error_type: Optional[str]
    error_sentence: str
    corrected_sentence: str
    error_sentence_id: int


@dataclass
class Task:
    pair: PairRecord
    scenario: str  # correct or incorrect

    @property
    def sample_id(self) -> str:
        return f"{self.pair.note_id}_{1 if self.scenario == 'correct' else 0}"

    @property
    def note(self) -> str:
        return self.pair.correct_note if self.scenario == "correct" else self.pair.incorrect_note

    @property
    def numbered(self) -> str:
        return number_sentences(self.note)


def normalize(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_pairs(path: Path, selected_ids: set[str], limit: Optional[int]) -> List[PairRecord]:
    records: List[PairRecord] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            note_id = raw.get("note_id", "")
            if selected_ids and note_id not in selected_ids:
                continue

            incorrect_note = raw.get("incorrect_note", "")
            error_sentence = raw.get("error_sentence", "")
            sid = find_error_sentence_id(incorrect_note, error_sentence)
            if sid is None:
                logger.warning("Skipping %s: cannot locate error sentence", note_id)
                continue

            records.append(
                PairRecord(
                    note_id=note_id,
                    correct_note=raw.get("correct_note", ""),
                    incorrect_note=incorrect_note,
                    error_type=raw.get("error_type"),
                    error_sentence=error_sentence,
                    corrected_sentence=raw.get("corrected_sentence", ""),
                    error_sentence_id=sid,
                )
            )
            if limit and len(records) >= limit:
                break
    return records


def build_selected_ids(summary_path: Path, scope: str) -> set[str]:
    summary = load_json(summary_path)
    if scope == "uw_only":
        return {
            note_id
            for note_id in summary["missing_from_medec_train_and_val_ids"]
            if note_id.startswith("uw-")
        }
    if scope == "missing_train":
        return set(summary["missing_from_medrect_en_train_ids"])
    if scope == "raw_missing":
        return set(summary["missing_from_medec_train_and_val_ids"])
    return set()


def build_prompt(task: Task, config: Dict[str, Any]) -> tuple[str, str]:
    if task.scenario == "correct":
        cfg = config["assessor_correct"]
        user = cfg["user_template"].format(
            sentences=task.numbered,
        )
        return cfg["system_prompt"], user

    cfg = config["assessor_incorrect"]
    user = cfg["user_template"].format(
        sentences=task.numbered,
        error_sentence_id=task.pair.error_sentence_id,
        error_sentence=task.pair.error_sentence,
        corrected_sentence=task.pair.corrected_sentence,
        error_type=task.pair.error_type or "medical error",
    )
    return cfg["system_prompt"], user


def validate(task: Task, content: str) -> tuple[bool, str, str]:
    label, sid = parse_assessor_answer(content)
    if task.scenario == "correct":
        if label == "CORRECT":
            return True, "CORRECT", "ok"
        return False, "", f"expected CORRECT, got {content!r}"

    if label == "ERROR" and sid == task.pair.error_sentence_id:
        return True, str(task.pair.error_sentence_id), "ok"
    return False, "", f"expected sentence {task.pair.error_sentence_id}, got {content!r}"


def make_raw_record(task: Task, reasoning: str, content: str) -> Dict[str, Any]:
    is_error = task.scenario == "incorrect"
    return {
        "sample_id": task.sample_id,
        "sentences": task.numbered,
        "error_flag": 1 if is_error else 0,
        "error_type": task.pair.error_type if is_error else None,
        "error_sentence_id": task.pair.error_sentence_id if is_error else None,
        "error_sentence": task.pair.error_sentence if is_error else None,
        "corrected_sentence": task.pair.corrected_sentence if is_error else None,
        "corrected_text": task.pair.correct_note if is_error else None,
        "metadata": {
            "local_note_id": task.pair.note_id,
            "original_text": task.note,
            "source": "generated_missing_assessor",
            "scenario": task.scenario,
        },
        "raw_response": {
            "content": content,
            "reasoning": reasoning,
            "source_model": "deepseek-reasoner",
        },
    }


async def process_task(
    task: Task,
    client: DeepSeekR1Client,
    cleaner: ReasoningCleaner,
    config: Dict[str, Any],
) -> tuple[str, Dict[str, Any]]:
    system_prompt, user_prompt = build_prompt(task, config)
    result = await client.generate(system_prompt, user_prompt, task.sample_id)
    if result is None:
        return "rejected", {
            "sample_id": task.sample_id,
            "note_id": task.pair.note_id,
            "scenario": task.scenario,
            "reason": "api_failure",
        }

    cleaned_reasoning, removed = cleaner.clean(result.get("reasoning", ""))
    content = result.get("content", "") or ""
    ok, _, reason = validate(task, content)

    row = make_raw_record(task, cleaned_reasoning, content)
    row["screening_results"] = {
        "accepted": ok,
        "reason": reason,
        "meta_refs_removed": removed,
        "prompt_tokens": result.get("usage", {}).get("prompt_tokens", 0),
        "completion_tokens": result.get("usage", {}).get("completion_tokens", 0),
        "total_tokens": result.get("usage", {}).get("total_tokens", 0),
    }
    return ("accepted" if ok else "rejected"), row


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate missing MedRECT assessor chains.")
    parser.add_argument("--sft-path", default=str(DEFAULT_SFT_PATH))
    parser.add_argument("--match-summary", default=str(DEFAULT_MATCH_SUMMARY))
    parser.add_argument(
        "--scope",
        choices=["uw_only", "missing_train", "raw_missing", "all"],
        default="uw_only",
        help="Which local notes to generate for",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prompt-config", default=str(DEFAULT_PROMPT_CONFIG))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


async def run() -> None:
    args = parse_args()
    config = load_json(Path(args.prompt_config))
    cleaner = ReasoningCleaner(config)
    client = DeepSeekR1Client(max_concurrent=args.concurrency, max_retries=args.max_retries)

    selected_ids = build_selected_ids(Path(args.match_summary), args.scope)
    pairs = load_pairs(Path(args.sft_path), selected_ids, args.limit)
    tasks = [Task(pair=pair, scenario=scenario) for pair in pairs for scenario in ("correct", "incorrect")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / f"generated_assessor_{args.scope}_accepted.jsonl"
    rejected_path = output_dir / f"generated_assessor_{args.scope}_rejected.jsonl"
    summary_path = output_dir / f"generated_assessor_{args.scope}_summary.json"

    accepted = 0
    rejected = 0
    accepted_rows: List[Dict[str, Any]] = []
    rejected_rows: List[Dict[str, Any]] = []

    async_tasks = [asyncio.create_task(process_task(task, client, cleaner, config)) for task in tasks]
    with tqdm(total=len(async_tasks), desc=f"Generating {args.scope} assessor", unit="sample") as pbar:
        for future in asyncio.as_completed(async_tasks):
            status, row = await future
            if status == "accepted":
                accepted += 1
                accepted_rows.append(row)
            else:
                rejected += 1
                rejected_rows.append(row)
            pbar.update(1)
            pbar.set_postfix(accepted=accepted, rejected=rejected)

    with open(accepted_path, "w", encoding="utf-8") as handle:
        for row in accepted_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(rejected_path, "w", encoding="utf-8") as handle:
        for row in rejected_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "scope": args.scope,
        "pairs_selected": len(pairs),
        "tasks_total": len(tasks),
        "accepted": accepted,
        "rejected": rejected,
        "accepted_output": str(accepted_path),
        "rejected_output": str(rejected_path),
        "api_stats": client.stats(),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(run())
