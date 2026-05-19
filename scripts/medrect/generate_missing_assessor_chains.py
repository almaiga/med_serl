#!/usr/bin/env python3
"""
Generate MedRECT-style assessor chains for local SFT pairs.

Pipeline:
1. Select local SFT pairs based on the requested scope
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
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.medrect.generate_injector_chains import DeepSeekR1Client, ReasoningCleaner
from scripts.self_play.utils import parse_assessor_answer, sentences_to_1indexed


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


DEFAULT_SFT_PATH = PROJECT_ROOT / "data_processed" / "medec_v2" / "ms_train.jsonl"
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
    sentences: str       # pre-numbered via sentences_to_1indexed() on MEDEC CSV Sentences field
    correct_sentences: str  # pre-numbered sentences for the correct note


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
        return self.pair.correct_sentences if self.scenario == "correct" else self.pair.sentences


def normalize(text: str) -> str:
    text = (text or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    text = re.sub(r"\s+", " ", text)
    return text


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def durable_write_jsonl(handle, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def durable_write_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def load_existing_sample_ids(*paths: Path) -> set[str]:
    processed: set[str] = set()
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            continue
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning("Skipping unreadable JSONL line in %s", path)
                    continue
                sample_id = row.get("sample_id")
                if sample_id:
                    processed.add(sample_id)
    return processed


def load_pairs(path: Path, selected_ids: set[str], limit: Optional[int]) -> List[PairRecord]:
    """Load pairs from a medec_v2 split file (has pre-computed sentences & error_sentence_id).

    Falls back gracefully if the new fields are absent (legacy sft_train.jsonl format),
    but logs a warning since sentence numbering will not be aligned with evaluation.
    """
    records: List[PairRecord] = []
    n_legacy = 0
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            note_id = raw.get("note_id", "")
            if selected_ids and note_id not in selected_ids:
                continue

            # Skip CORRECT-only rows — we only generate pairs for ERROR notes
            # (the correct scenario uses corrected_text as the input note)
            error_flag = raw.get("error_flag", 1)
            if error_flag == 0:
                continue

            # Prefer pre-computed sentence numbering from medec_v2 split files
            sentences = raw.get("sentences", "")
            sid = raw.get("error_sentence_id")

            # Correct-scenario note: use corrected_text (reconstructed, clean text)
            corrected_text = raw.get("corrected_text") or raw.get("correct_note", "")
            # Number the correct note using sentences_to_1indexed if it has MEDEC format,
            # otherwise fall back to splitting the clean corrected text line-by-line.
            correct_sentences_raw = raw.get("correct_sentences", "")
            if correct_sentences_raw:
                correct_sentences = correct_sentences_raw
            else:
                # corrected_text is clean reconstructed prose — number it properly
                correct_sentences = sentences_to_1indexed(corrected_text) if "\n" in corrected_text and re.match(r"^\d+\s", corrected_text) else _number_clean_text(corrected_text)

            if not sentences or sid is None:
                n_legacy += 1
                logger.warning(
                    "Skipping %s: missing 'sentences' or 'error_sentence_id' fields. "
                    "Use medec_v2 split files from build_medec_splits.py.", note_id
                )
                continue

            incorrect_note = raw.get("incorrect_note") or raw.get("text", "")
            error_sentence = raw.get("error_sentence", "")
            corrected_sentence = raw.get("corrected_sentence", "")

            records.append(
                PairRecord(
                    note_id=note_id,
                    correct_note=corrected_text,
                    incorrect_note=incorrect_note,
                    error_type=raw.get("error_type"),
                    error_sentence=error_sentence,
                    corrected_sentence=corrected_sentence,
                    error_sentence_id=sid,
                    sentences=sentences,
                    correct_sentences=correct_sentences,
                )
            )
            if limit and len(records) >= limit:
                break

    if n_legacy:
        logger.warning("%d records skipped due to missing medec_v2 fields.", n_legacy)
    return records


def _number_clean_text(text: str) -> str:
    """Number a clean prose note (not MEDEC 0-indexed format) using sentences_to_1indexed logic."""
    import re as _re
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    if len(lines) > 1:
        # Already one sentence per line
        return "\n".join(f"{i+1}. {l}" for i, l in enumerate(lines))
    # Single block — split on punctuation + uppercase
    parts = _re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return "\n".join(f"{i+1}. {p.strip()}" for i, p in enumerate(parts) if p.strip())


def build_selected_ids(summary_path: Optional[Path], scope: str) -> set[str]:
    if scope == "all":
        return set()

    if summary_path is None:
        raise ValueError(f"--match-summary is required for scope={scope}")

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
    parser.add_argument(
        "--match-summary",
        default=str(DEFAULT_MATCH_SUMMARY),
        help="Required for scopes that depend on recovered-match subsets; ignored for --scope all",
    )
    parser.add_argument(
        "--scope",
        choices=["uw_only", "missing_train", "raw_missing", "all"],
        default="uw_only",
        help="Which local notes to generate for",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--prompt-config", default=str(DEFAULT_PROMPT_CONFIG))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--concurrency", type=int, default=40)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Append to existing accepted/rejected outputs and skip already written sample_ids",
    )
    return parser.parse_args()


async def run() -> None:
    args = parse_args()
    config = load_json(Path(args.prompt_config))
    cleaner = ReasoningCleaner(config)
    client = DeepSeekR1Client(max_concurrent=args.concurrency, max_retries=args.max_retries)

    match_summary = Path(args.match_summary) if args.match_summary else None
    selected_ids = build_selected_ids(match_summary, args.scope)
    pairs = load_pairs(Path(args.sft_path), selected_ids, args.limit)
    tasks = [Task(pair=pair, scenario=scenario) for pair in pairs for scenario in ("correct", "incorrect")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accepted_path = output_dir / f"generated_assessor_{args.scope}_accepted.jsonl"
    rejected_path = output_dir / f"generated_assessor_{args.scope}_rejected.jsonl"
    summary_path = output_dir / f"generated_assessor_{args.scope}_summary.json"
    processed_sample_ids = load_existing_sample_ids(accepted_path, rejected_path) if args.resume else set()
    if processed_sample_ids:
        logger.info("Resuming from existing outputs: %d sample_ids already written", len(processed_sample_ids))
        tasks = [task for task in tasks if task.sample_id not in processed_sample_ids]

    accepted = 0
    rejected = 0
    skipped = len(processed_sample_ids)
    started_at = now_iso()

    def build_summary(status: str) -> Dict[str, Any]:
        return {
            "scope": args.scope,
            "status": status,
            "started_at": started_at,
            "updated_at": now_iso(),
            "pairs_selected": len(pairs),
            "tasks_total": len(tasks),
            "completed": skipped + accepted + rejected,
            "skipped_existing": skipped,
            "accepted": accepted,
            "rejected": rejected,
            "accepted_output": str(accepted_path),
            "rejected_output": str(rejected_path),
            "api_stats": client.stats(),
        }

    durable_write_json(summary_path, build_summary("running"))

    async_tasks = [asyncio.create_task(process_task(task, client, cleaner, config)) for task in tasks]
    with open(accepted_path, "a" if args.resume else "w", encoding="utf-8") as accepted_handle, open(
        rejected_path, "a" if args.resume else "w", encoding="utf-8"
    ) as rejected_handle:
        with tqdm(total=len(async_tasks), desc=f"Generating {args.scope} assessor", unit="sample") as pbar:
            for future in asyncio.as_completed(async_tasks):
                status, row = await future
                if status == "accepted":
                    accepted += 1
                    durable_write_jsonl(accepted_handle, row)
                else:
                    rejected += 1
                    durable_write_jsonl(rejected_handle, row)
                durable_write_json(summary_path, build_summary("running"))
                pbar.update(1)
                pbar.set_postfix(accepted=accepted, rejected=rejected)

    summary = build_summary("completed")
    durable_write_json(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(run())
