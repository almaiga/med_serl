#!/usr/bin/env python3
"""
Prepare MEDRECT data for detection-only SFT.

Reads MEDRECT JSON files (medrect-en-train.json, medrect-ja-train.json)
and converts them to SFT-ready JSONL with prompts loaded from config.

Output format per record:
{
    "sample_id": "ms-train-0",
    "system_prompt": "<loaded from config>",
    "user_prompt": "<sentences filled into template>",
    "label": "CORRECT" or "7",
    "reasoning": null or "<thinking trace if present>",
    "language": "en",
    "error_type": "causalOrganism" or null,
    "error_flag": 0 or 1,
    "error_sentence_id": null or 7
}

Usage:
    # Standard conversion
    python scripts/medrect/prepare_medrect_sft.py \\
        --input data/medrect/medrect-en-train.json \\
        --output data_processed/medrect/medrect_sft_train.jsonl

    # Generated assessor chains from local SFT pairs
    python scripts/medrect/prepare_medrect_sft.py \\
        --input data_processed/medrect/generated_assessor_all_accepted.jsonl \\
        --output data_processed/medrect/generated_assessor_all_sft.jsonl \\
        --language en

    # Bilingual (en + ja)
    python scripts/medrect/prepare_medrect_sft.py \\
        --input data/medrect/medrect-en-train.json data/medrect/medrect-ja-train.json \\
        --output data_processed/medrect/medrect_sft_train.jsonl

    # Dry run (print 5 samples, don't write)
    python scripts/medrect/prepare_medrect_sft.py \\
        --input data/medrect/medrect-en-train.json --dry-run

    # With reasoning field (if your data has precomputed traces)
    python scripts/medrect/prepare_medrect_sft.py \\
        --input data/medrect/medrect-en-train.json \\
        --reasoning-field thinking
"""

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROMPT_CONFIG = PROJECT_ROOT / "configs" / "prompts" / "sft" / "medrect_detection_prompts.json"
DEFAULT_OUTPUT = PROJECT_ROOT / "data_processed" / "medrect" / "medrect_sft_train.jsonl"


def load_prompt_config(config_path: Path) -> Dict[str, Any]:
    """Load prompt configuration from JSON file."""
    with open(config_path, "r") as f:
        config = json.load(f)
    
    required_keys = ["system_prompt", "user_template"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Prompt config missing required key: '{key}' in {config_path}")
    
    return config


def detect_language(filepath: str) -> str:
    """Detect language from filename convention (medrect-en-*.json, medrect-ja-*.json)."""
    name = Path(filepath).name.lower()
    if "-ja-" in name or "_ja_" in name or name.startswith("ja"):
        return "ja"
    elif "-en-" in name or "_en_" in name or name.startswith("en"):
        return "en"
    else:
        return "unknown"


def load_medrect_json(filepath: str) -> List[Dict]:
    """Load a MEDRECT JSON or JSONL file."""
    path = Path(filepath)
    with open(path, "r", encoding="utf-8") as f:
        first_nonempty = ""
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            if line.strip():
                first_nonempty = line.strip()
                f.seek(pos)
                break

        if not first_nonempty:
            return []

        if first_nonempty.startswith("["):
            data = json.load(f)
            if not isinstance(data, list):
                raise ValueError(
                    f"Expected a JSON array in {filepath}, got {type(data).__name__}"
                )
            return data

        records = []
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
        return records


def convert_record(
    record: Dict,
    prompt_config: Dict[str, Any],
    language: str,
    reasoning_field: Optional[str] = None,
) -> Dict[str, Any]:
    """Convert a single MEDRECT record to SFT format.
    
    Args:
        record: Raw MEDRECT record with fields like sample_id, sentences, error_flag, etc.
        prompt_config: Loaded prompt configuration.
        language: Language code (en, ja).
        reasoning_field: Optional field name containing reasoning traces.
    
    Returns:
        SFT-ready record.
    """
    sentences = record.get("sentences", "")
    error_flag = record.get("error_flag", 0)
    error_sentence_id = record.get("error_sentence_id")
    error_type = record.get("error_type")
    sample_id = record.get("sample_id", "unknown")
    
    # Build prompts from config (not hardcoded)
    system_prompt = prompt_config["system_prompt"]
    user_template = prompt_config["user_template"]
    user_prompt = user_template.format(sentences=sentences)
    
    # Detection-only label: CORRECT or sentence number
    if error_flag == 0 or error_flag is False:
        label = "CORRECT"
    else:
        if error_sentence_id is not None:
            label = str(error_sentence_id)
        else:
            # Fallback: we know there's an error but don't know which sentence
            label = "INCORRECT"
    
    # Extract reasoning if available
    reasoning = None
    if reasoning_field and reasoning_field in record:
        reasoning = record[reasoning_field]
    else:
        # Try common field names (top-level)
        for field in ["reasoning", "thinking", "chain_of_thought", "rationale"]:
            if field in record and record[field]:
                reasoning = record[field]
                break
        # Check nested raw_response.reasoning (e.g. DeepSeek-R1 generated chains)
        if reasoning is None:
            raw_resp = record.get("raw_response")
            if isinstance(raw_resp, dict):
                reasoning = raw_resp.get("reasoning") or raw_resp.get("thinking")
    
    return {
        "sample_id": sample_id,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "label": label,
        "reasoning": reasoning,
        "language": language,
        "error_type": error_type,
        "error_flag": int(error_flag),
        "error_sentence_id": error_sentence_id,
    }


def print_stats(records: List[Dict]) -> None:
    """Print dataset statistics."""
    total = len(records)
    
    # Label distribution
    labels = Counter(r["label"] for r in records)
    correct_count = labels.get("CORRECT", 0)
    error_count = total - correct_count
    
    # Language distribution
    langs = Counter(r["language"] for r in records)
    
    # Error type distribution
    error_types = Counter(r["error_type"] for r in records if r["error_type"])
    
    # Reasoning availability
    has_reasoning = sum(1 for r in records if r["reasoning"])
    
    print(f"\n{'='*60}")
    print(f"  MEDRECT SFT Dataset Statistics")
    print(f"{'='*60}")
    print(f"  Total examples:      {total}")
    print(f"  CORRECT (no error):  {correct_count} ({100*correct_count/total:.1f}%)")
    print(f"  Error detected:      {error_count} ({100*error_count/total:.1f}%)")
    print(f"  Has reasoning:       {has_reasoning} ({100*has_reasoning/total:.1f}%)")
    print(f"\n  By language:")
    for lang, count in sorted(langs.items()):
        print(f"    {lang}: {count} ({100*count/total:.1f}%)")
    if error_types:
        print(f"\n  By error type:")
        for etype, count in sorted(error_types.items(), key=lambda x: -x[1]):
            print(f"    {etype}: {count}")
    print(f"{'='*60}\n")


def print_samples(records: List[Dict], n: int = 5) -> None:
    """Print formatted sample records for inspection."""
    print(f"\n{'='*60}")
    print(f"  Sample Records (showing {min(n, len(records))})")
    print(f"{'='*60}")
    
    for i, rec in enumerate(records[:n]):
        print(f"\n--- Sample {i+1} [{rec['sample_id']}] ---")
        print(f"Language: {rec['language']}")
        print(f"Label: {rec['label']}")
        print(f"Error type: {rec['error_type']}")
        print(f"\n[SYSTEM]")
        print(rec["system_prompt"][:200])
        print(f"\n[USER]")
        # Show first 300 chars of user prompt
        user_text = rec["user_prompt"]
        if len(user_text) > 300:
            print(user_text[:300] + "...")
        else:
            print(user_text)
        print(f"\n[EXPECTED RESPONSE]")
        if rec["reasoning"]:
            print(f"<think>\n{rec['reasoning'][:200]}...\n</think>\n\n{rec['label']}")
        else:
            print(f"<think>\n\n</think>\n\n{rec['label']}")
        print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare MEDRECT data for detection-only SFT training."
    )
    parser.add_argument(
        "--input", nargs="+", required=True,
        help="One or more MEDRECT JSON files (e.g. medrect-en-train.json medrect-ja-train.json)"
    )
    parser.add_argument(
        "--output", type=str, default=str(DEFAULT_OUTPUT),
        help=f"Output JSONL path (default: {DEFAULT_OUTPUT})"
    )
    parser.add_argument(
        "--prompt-config", type=str, default=str(DEFAULT_PROMPT_CONFIG),
        help=f"Prompt config JSON (default: {DEFAULT_PROMPT_CONFIG})"
    )
    parser.add_argument(
        "--language",
        choices=["en", "ja", "unknown"],
        default=None,
        help="Override language for all input files instead of inferring from filename"
    )
    parser.add_argument(
        "--reasoning-field", type=str, default=None,
        help="Field name in input data that contains reasoning traces (e.g. 'thinking')"
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit number of records per input file (for testing)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print samples and stats without writing output"
    )
    parser.add_argument(
        "--show-samples", type=int, default=5,
        help="Number of samples to display (default: 5)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Load prompt config
    prompt_config = load_prompt_config(Path(args.prompt_config))
    print(f"Loaded prompt config: {args.prompt_config}")
    print(f"  version: {prompt_config.get('version', 'N/A')}")
    
    # Process each input file
    all_records = []
    for input_path in args.input:
        language = args.language or detect_language(input_path)
        print(f"\nLoading: {input_path} (language={language})")
        
        raw_data = load_medrect_json(input_path)
        if args.limit:
            raw_data = raw_data[:args.limit]
        
        print(f"  Raw records: {len(raw_data)}")
        
        for record in raw_data:
            converted = convert_record(
                record, prompt_config, language, args.reasoning_field
            )
            all_records.append(converted)
    
    print(f"\nTotal converted: {len(all_records)} records")
    
    # Stats
    print_stats(all_records)
    
    # Samples
    print_samples(all_records, n=args.show_samples)
    
    # Write output
    if args.dry_run:
        print("DRY RUN — no output file written.")
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            for rec in all_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        
        print(f"Written: {output_path} ({len(all_records)} records)")


if __name__ == "__main__":
    main()
