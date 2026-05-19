#!/usr/bin/env python3
"""
Clean English reasoning responses by removing meta-references.

Copied verbatim from pfnet-research/medrect/scripts/training_data/clean_reasoning_en.py
Only change: replaced `from loguru import logger` with standard logging.

Usage:
    python clean_reasoning_en.py --input raw_responses.jsonl --output cleaned_raw_responses.jsonl
    python clean_reasoning_en.py --input_dir data_processed/medrect_v2/ms_train --recursive
"""

import json
import re
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class ReasoningCleaner:
    """Clean reasoning text by removing meta-references.

    Copied verbatim from pfnet-research/medrect clean_reasoning_en.py.
    """

    def __init__(self):
        # Keywords that indicate meta-references
        self.meta_keywords = [
            "told that",
            "are told",
            "we are told",
            "being told",
            "pre-verified",
            "pre-determined",
            "pre-confirmed",
            "text claims",
            "external instructions",
            "prior knowledge",
            "prior informationreference information",
            "reference correction",
            "reference says",
            "sample correction",
            "provided correction",
            "given correction",
            "instruction states",
            "problem states",
            "task states",
            "instructions say",
            "problem says",
            "task says",
            "do not mention",
            "not supposed to",
            "should not mention",
            "ignore the reference",
            "independently",
            "from scratch",
            "this time, you are",
            "error example",
            "no-error example",
            "expected outcome",
            "instruction content",
            "which we are not supposed",
            "but it's the sample correction",
            "the reference solution",
            "meta-references",
            "meta-reference",
            'reviewing an "error example"',
            'reviewing a "no-error example"',
            "expected result",
            "known outcome",
            "follow the medical reasoning independently",
            "purely medical evaluation",
            "reaching your conclusion through pure medical evaluation",
            "as if you are analyzing it from scratch",
            "do NOT make any reference",
        ]

        self.meta_patterns = [
            re.compile(rf".*{re.escape(keyword)}.*", re.IGNORECASE)
            for keyword in self.meta_keywords
        ]

        self.specific_patterns = [
            re.compile(r".*This time.*reviewing.*error.*example.*", re.IGNORECASE),
            re.compile(r".*do not mention.*reasoning.*", re.IGNORECASE),
            re.compile(r".*reference.*not supposed.*", re.IGNORECASE),
            re.compile(r".*approach.*as if.*analyzing.*scratch.*", re.IGNORECASE),
            re.compile(r".*pure medical evaluation.*", re.IGNORECASE),
            re.compile(r".*independently.*reference.*", re.IGNORECASE),
            re.compile(r".*medical reasoning.*independently.*", re.IGNORECASE),
            re.compile(r".*we must.*ignore.*reference.*", re.IGNORECASE),
            re.compile(r".*problem.*states.*do not.*", re.IGNORECASE),
            re.compile(r".*instruction.*do not.*reference.*", re.IGNORECASE),
        ]

        self.all_patterns = self.meta_patterns + self.specific_patterns

    def find_sentences_to_remove(self, text: str) -> List[tuple]:
        if not text:
            return []

        sentences_to_remove = []
        sentence_pattern = re.compile(r"[^.]*\.")

        for match in sentence_pattern.finditer(text):
            sentence = match.group()
            sentence_stripped = sentence.strip()

            if not sentence_stripped:
                continue

            should_remove = False
            for pattern in self.all_patterns:
                if pattern.search(sentence_stripped):
                    should_remove = True
                    break

            if should_remove:
                sentences_to_remove.append((match.start(), match.end()))

        return sentences_to_remove

    def clean_reasoning_text(self, text: str) -> tuple:
        if not text:
            return text, 0

        sentences_to_remove = self.find_sentences_to_remove(text)

        if not sentences_to_remove:
            return text, 0

        sentences_to_remove.sort(key=lambda x: x[0], reverse=True)

        cleaned_text = text
        removed_count = 0

        for start_pos, end_pos in sentences_to_remove:
            cleaned_text = cleaned_text[:start_pos] + cleaned_text[end_pos:]
            removed_count += 1

        return cleaned_text, removed_count

    def clean_record(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Clean a single JSONL record. Works on our format: raw_response.reasoning."""
        cleaned = record.copy()

        raw = cleaned.get("raw_response")
        if not isinstance(raw, dict):
            return cleaned

        raw = raw.copy()
        reasoning = raw.get("reasoning", "")
        if reasoning:
            cleaned_reasoning, removed = self.clean_reasoning_text(reasoning)
            raw["reasoning"] = cleaned_reasoning
            raw.setdefault("metadata", {})
            raw["metadata"]["reasoning_cleaned"] = True
            raw["metadata"]["removed_sentences"] = removed

        cleaned["raw_response"] = raw
        return cleaned


def clean_jsonl_file(input_path: Path, output_path: Path, overwrite: bool = False) -> None:
    """Clean a JSONL file (one JSON record per line)."""
    if overwrite:
        output_path = input_path

    cleaner = ReasoningCleaner()
    records = []
    total_removed = 0

    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            cleaned = cleaner.clean_record(record)
            removed = cleaned.get("raw_response", {}).get("metadata", {}).get("removed_sentences", 0)
            total_removed += removed
            records.append(cleaned)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    logger.info("Cleaned %s → %s  (%d records, %d sentences removed)",
                input_path, output_path, len(records), total_removed)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    parser = argparse.ArgumentParser(description="Clean English reasoning (meta-reference removal)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input", type=str, help="Single input JSONL file")
    group.add_argument("--input_dir", type=str, help="Directory to search for *.jsonl files")
    parser.add_argument("--output", type=str, help="Output path (single-file mode only)")
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite input files in place")
    args = parser.parse_args()

    if args.input:
        input_path = Path(args.input)
        output_path = Path(args.output) if args.output else input_path.parent / f"cleaned_{input_path.name}"
        clean_jsonl_file(input_path, output_path, overwrite=args.overwrite)
    else:
        input_dir = Path(args.input_dir)
        pattern = "**/*.jsonl" if args.recursive else "*.jsonl"
        files = list(input_dir.glob(pattern))
        logger.info("Found %d JSONL files", len(files))
        for f in files:
            out = f if args.overwrite else f.parent / f"cleaned_{f.name}"
            clean_jsonl_file(f, out, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
