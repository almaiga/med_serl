#!/usr/bin/env python3
"""Merge a LoRA adapter checkpoint into a full causal LM checkpoint.

This is useful before using a PEFT-trained MedReCT adapter with vLLM-based
scripts, which expect a standard model directory rather than an adapter-only
save.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter-dir", required=True, help="Path to LoRA adapter checkpoint directory")
    parser.add_argument("--output-dir", required=True, help="Directory to save merged full model")
    parser.add_argument(
        "--base-model",
        default=None,
        help="Optional base model override. Defaults to base_model_name_or_path from adapter_config.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_dir = Path(args.adapter_dir)
    output_dir = Path(args.output_dir)

    if not adapter_dir.exists():
        raise FileNotFoundError(f"Adapter directory not found: {adapter_dir}")

    peft_config = PeftConfig.from_pretrained(str(adapter_dir))
    base_model = args.base_model or peft_config.base_model_name_or_path
    if not base_model:
        raise ValueError("Could not determine base model. Pass --base-model explicitly.")

    print(f"Merging adapter: {adapter_dir}")
    print(f"Base model: {base_model}")
    print(f"Output dir: {output_dir}")

    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model = model.merge_and_unload()

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(output_dir), safe_serialization=True)

    try:
        tokenizer = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    except (AttributeError, TypeError):
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        chat_template_path = adapter_dir / "chat_template.jinja"
        if chat_template_path.exists():
            tokenizer.chat_template = chat_template_path.read_text()

    tokenizer.save_pretrained(str(output_dir))

    # Carry over adapter training metadata when present.
    config_json = adapter_dir / "training_config.json"
    if config_json.exists():
        metadata = json.loads(config_json.read_text())
        (output_dir / "merged_from_training_config.json").write_text(
            json.dumps(metadata, indent=2, default=str)
        )

    print("Merged model saved successfully.")


if __name__ == "__main__":
    main()
