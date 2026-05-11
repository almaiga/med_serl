#!/usr/bin/env python3
"""
Run MEDRECT SFT data-scaling experiments — two-phase structure.

Phase 1: Train all N shards sequentially, keeping all LoRA adapters on disk.
Phase 2: Load vLLM base model ONCE, hot-swap each LoRA adapter, evaluate all
         shards without restarting the engine.

Example:
    python scripts/medrect/run_sft_scaling_experiment.py \
        --train-file data_processed/medrect/generated_assessor_all_sft.jsonl \
        --model-name Qwen/Qwen3-4B \
        --output-root outputs/local_training/medrect_sft_scaling \
        --results-root results/medrect_sft_scaling
"""

import argparse
import csv
import json
import math
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
# Make scripts/ importable (needed for inference_detection_vllm and inference_error_detection)
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "medrect"))

DEFAULT_TRAIN_FILE = PROJECT_ROOT / "data_processed" / "medrect" / "generated_assessor_all_sft.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "local_training" / "medrect_sft_scaling"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "medrect_sft_scaling"
DEFAULT_PROMPT_CONFIG = PROJECT_ROOT / "configs" / "prompts" / "detection_localization_prompts.json"


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train MEDRECT SFT adapters for N cumulative data fractions, then evaluate all with a single vLLM session."
    )
    # Data / paths
    parser.add_argument("--train-file", default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B")
    parser.add_argument("--base-model-path", default=None,
                        help="Local path to base model weights (overrides --model-name for vLLM)")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--fractions", type=int, default=10,
                        help="Number of cumulative data fractions to train (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-splits", action="store_true")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip shards whose adapter / eval output already exists")

    # Eval config
    parser.add_argument("--dataset", default="all", choices=["ms", "uw", "all"])
    parser.add_argument("--prompt-config", default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--thinking-budget", type=int, default=4096)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--num-eval-runs", type=int, default=1,
                        help="Inference runs per shard to average out sampling variance")
    parser.add_argument("--eval-base-model", action="store_true",
                        help="Evaluate the base model (no LoRA) before training shards")

    # Training config — mirrors run_assessor_only_sft.sh defaults exactly
    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument("--eval-split", type=float, default=0.05,
                        help="Fraction of training data held out for validation / early stopping (default: 0.05)")
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    # eval-steps is computed per-shard adaptively; this sets the minimum
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=128)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="all-linear")
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--debug-samples", type=int, default=0)
    parser.add_argument("--dataloader-num-workers", type=int, default=4)
    parser.add_argument("--no-bf16", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="medrect-sft-scaling")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


# ── Data helpers ──────────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"No records found in {path}")
    return records


def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_cumulative_splits(
    records: List[Dict], split_dir: Path, fractions: int, seed: int, overwrite: bool
) -> List[Dict]:
    if fractions < 1:
        raise ValueError("--fractions must be >= 1")
    split_dir.mkdir(parents=True, exist_ok=True)
    shuffled = list(records)
    random.Random(seed).shuffle(shuffled)
    total = len(shuffled)
    manifest = []
    for idx in range(1, fractions + 1):
        fraction = idx / fractions
        count = math.ceil(total * fraction)
        path = split_dir / f"train_frac_{idx:02d}_of_{fractions}.jsonl"
        if overwrite or not path.exists():
            write_jsonl(path, shuffled[:count])
        manifest.append(dict(
            index=idx, fractions=fractions, fraction=fraction,
            train_count=count, train_file=str(path),
        ))
    manifest_path = split_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(dict(seed=seed, total_records=total, splits=manifest), f, indent=2)
    return manifest


# ── Subprocess helpers ────────────────────────────────────────────────────────

def launcher(nproc_per_node: Optional[int]) -> List[str]:
    if nproc_per_node and nproc_per_node > 1:
        return ["torchrun", "--standalone", f"--nproc_per_node={nproc_per_node}"]
    return [sys.executable]


def run_command(cmd: List[str], log_path: Path, dry_run: bool) -> None:
    print(" ".join(cmd))
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as log:
        process = subprocess.Popen(
            cmd, cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        rc = process.wait()
    if rc != 0:
        raise subprocess.CalledProcessError(rc, cmd)


# ── Metrics helpers ───────────────────────────────────────────────────────────

def read_metrics(summary_path: Path) -> Dict:
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    m = summary["metrics"]
    det = m["detection"]
    sent = m["sentence_extraction"]
    return dict(
        summary_file=str(summary_path),
        test_samples=m["total_samples"],
        detection_accuracy=det["accuracy"],
        detection_precision=det["precision"],
        detection_recall=det["recall"],
        detection_f1=det["f1"],
        sentence_accuracy=sent["accuracy"],
        sentence_exact_matches=sent["exact_matches"],
        sentence_total_errors=sent["total_errors"],
    )


def average_metrics(all_metrics: List[Dict]) -> Dict:
    if len(all_metrics) == 1:
        return all_metrics[0]
    keys = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float))]
    averaged: Dict = {"summary_file": all_metrics[0]["summary_file"], "num_eval_runs": len(all_metrics)}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        averaged[k] = sum(vals) / len(vals)
        averaged[f"{k}_std"] = (sum((v - averaged[k]) ** 2 for v in vals) / len(vals)) ** 0.5
    return averaged


def maybe_write_plot(csv_path: Path, png_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping PNG plot.")
        return
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return
    x = [int(r["train_count"]) for r in rows if r.get("train_count", "0").isdigit()]
    y = [float(r["detection_accuracy"]) for r in rows if r.get("train_count", "0").isdigit()]
    plt.figure(figsize=(7, 4.5))
    plt.plot(x, y, marker="o", label="Detection accuracy")
    plt.xlabel("SFT training examples")
    plt.ylabel("Accuracy on full test set")
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    png_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(png_path, dpi=200)
    plt.close()


# ── Phase 1: Train all shards ─────────────────────────────────────────────────

def run_phase1_training(args: argparse.Namespace, splits: List[Dict],
                        output_root: Path, log_dir: Path) -> None:
    print(f"\n{'='*60}")
    print(f"PHASE 1 — Training {args.fractions} shards (all adapters kept on disk)")
    print(f"{'='*60}")

    for split in splits:
        idx = split["index"]
        train_count = split["train_count"]
        adapter_dir = output_root / f"adapter_frac_{idx:02d}_of_{args.fractions}"
        train_done = adapter_dir / "adapter_config.json"

        print(f"\n{'─'*50}")
        print(f"Shard {idx}/{args.fractions}  ({train_count} samples)  →  {adapter_dir.name}")
        print(f"{'─'*50}")

        if args.skip_existing and train_done.exists():
            print("  Already trained — skipping.")
            continue

        # Adaptive eval_steps so that eval fires at least twice per epoch
        # even for small shards with few optimizer steps.
        effective_train = math.ceil(train_count * (1.0 - args.eval_split)) if args.eval_split > 0 else train_count
        steps_per_epoch = max(1, math.ceil(effective_train / (args.batch_size * args.grad_accum)))
        adaptive_eval_steps = max(5, steps_per_epoch // 2)
        adaptive_save_steps = max(adaptive_eval_steps, args.save_steps)
        print(f"  steps/epoch={steps_per_epoch}  eval_steps={adaptive_eval_steps}  save_steps={adaptive_save_steps}")

        train_cmd = launcher(args.nproc_per_node) + [
            "scripts/medrect/train_medrect_lora.py",
            "--train-file",           split["train_file"],
            "--model-name",           args.model_name,
            "--output-dir",           str(adapter_dir),
            "--eval-split",           str(args.eval_split),
            "--max-seq-length",       str(args.max_seq_length),
            "--per-device-train-batch-size", str(args.batch_size),
            "--gradient-accumulation-steps", str(args.grad_accum),
            "--learning-rate",        str(args.learning_rate),
            "--num-train-epochs",     str(args.num_train_epochs),
            "--warmup-ratio",         str(args.warmup_ratio),
            "--weight-decay",         str(args.weight_decay),
            "--logging-steps",        str(args.logging_steps),
            "--save-steps",           str(adaptive_save_steps),
            "--eval-steps",           str(adaptive_eval_steps),
            "--lora-r",               str(args.lora_r),
            "--lora-alpha",           str(args.lora_alpha),
            "--lora-dropout",         str(args.lora_dropout),
            "--lora-target-modules",  args.lora_target_modules,
            "--debug-samples",        str(args.debug_samples),
            "--dataloader-num-workers", str(args.dataloader_num_workers),
            "--seed",                 str(args.seed),
        ]
        if args.no_bf16:
            train_cmd.append("--no-bf16")
        else:
            train_cmd.append("--bf16")
        if args.no_early_stopping:
            train_cmd.append("--no-early-stopping")
        if args.wandb:
            train_cmd.extend(["--wandb", "--wandb-project", args.wandb_project])

        run_command(train_cmd, log_dir / f"train_frac_{idx:02d}.log", args.dry_run)

    print(f"\n{'='*60}")
    print("PHASE 1 complete — all adapters saved.")
    print(f"{'='*60}")


# ── Phase 2: Evaluate all adapters with a single vLLM session ────────────────

def run_phase2_eval(
    args: argparse.Namespace,
    splits: List[Dict],
    output_root: Path,
    results_root: Path,
) -> List[Dict]:
    print(f"\n{'='*60}")
    print("PHASE 2 — Single vLLM session evaluating all adapters")
    print(f"{'='*60}")

    # Delayed imports so dry-run / training-only uses don't need vLLM
    import pandas as pd
    from tqdm import tqdm
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    # Import shared helpers from the standalone inference script
    from inference_detection_vllm import (  # type: ignore[import]
        load_prompt_config,
        sentences_to_1indexed,
        split_thinking,
        parse_output_candidates,
        compute_metrics,
        print_metrics,
    )
    from inference_error_detection import load_test_data  # type: ignore[import]

    base_model = args.base_model_path or args.model_name
    use_thinking = not args.no_thinking
    temperature = args.temperature if args.temperature is not None else (0.6 if use_thinking else 0.7)
    top_p = args.top_p if args.top_p is not None else (0.95 if use_thinking else 0.8)
    max_tokens = (args.thinking_budget + 256) if use_thinking else args.max_new_tokens
    mode_slug = "thinking" if use_thinking else "no_thinking"
    prompt_cfg_path = Path(args.prompt_config) if args.prompt_config else DEFAULT_PROMPT_CONFIG

    # ── Load tokenizer, prompt config, test data — once ──────────────────────
    print(f"\nLoading tokenizer from: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    prompt_config = load_prompt_config(prompt_cfg_path)
    test_df = load_test_data(args.dataset)
    if args.max_eval_samples:
        test_df = test_df.head(args.max_eval_samples)

    system_prompt = prompt_config["system_prompt"]
    user_template = prompt_config["user_template"]

    # ── Build prompts once — identical for every adapter ─────────────────────
    print(f"Building prompts for {len(test_df)} test samples…")
    prompts: List[str] = []
    meta: List[Dict] = []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="Prompts"):
        error_flag = int(row["Error Flag"])
        sentences = sentences_to_1indexed(str(row.get("Sentences", row.get("Text", ""))))
        raw_sid = row.get("Error Sentence ID")
        if error_flag == 1 and pd.notna(raw_sid):
            gt_sid, gt_label = int(raw_sid) + 1, str(int(raw_sid) + 1)
        else:
            gt_sid, gt_label = None, "CORRECT"
        meta.append(dict(
            text_id=str(row.get("Text ID", "")),
            dataset=str(row.get("dataset", "")),
            error_flag=error_flag,
            gt_label=gt_label,
            gt_sid=gt_sid,
            error_type=str(row.get("Error Type", "")) if pd.notna(row.get("Error Type")) else "",
            sentences=sentences,
        ))
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_template.format(sentences=sentences)},
        ]
        prompt_kwargs: Dict = dict(tokenize=False, add_generation_prompt=True)
        if use_thinking:
            prompt_kwargs["enable_thinking"] = True
        prompts.append(tokenizer.apply_chat_template(messages, **prompt_kwargs))

    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        top_k=args.top_k,
        min_p=args.min_p,
        presence_penalty=args.presence_penalty,
        max_tokens=max_tokens,
    )

    # ── Start vLLM engine ONCE with LoRA support ──────────────────────────────
    print(f"\nStarting vLLM engine (base: {base_model}, enable_lora=True)…")
    llm = LLM(
        model=base_model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=8192,
        trust_remote_code=True,
        enable_lora=True,
        max_lora_rank=64,
    )

    # ── Inner helpers ─────────────────────────────────────────────────────────

    def _infer_and_save(lora_request: Optional[object], eval_dir: Path, label: str) -> Dict:
        outputs = llm.generate(prompts, sampling_params, lora_request=lora_request)
        results = []
        for i, (m, out) in enumerate(zip(meta, outputs)):
            full_text = out.outputs[0].text.strip()
            thinking, content = split_thinking(full_text)
            if not content:
                content = full_text
            pred_type, pred_sid = parse_output_candidates(content, full_text, thinking)
            pred_label = (
                "CORRECT" if pred_type == "CORRECT"
                else (str(pred_sid) if pred_sid is not None else "UNKNOWN")
            )
            detection_correct = (
                (m["gt_label"] == "CORRECT" and pred_label == "CORRECT") or
                (m["gt_label"] != "CORRECT" and pred_label not in ("CORRECT", "UNKNOWN"))
            )
            localization_correct = (
                m["gt_sid"] is not None and pred_sid is not None and m["gt_sid"] == pred_sid
            )
            if i == 0:
                print(f"\nDEBUG first sample | GT={m['gt_label']}  Pred={pred_label}")
                print(f"  {content[:200]}")
            results.append(dict(
                text_id=m["text_id"], dataset=m["dataset"], error_type=m["error_type"],
                gt_label=m["gt_label"], gt_sid=m["gt_sid"],
                pred_label=pred_label, pred_sid=pred_sid,
                detection_correct=detection_correct, localization_correct=localization_correct,
                thinking=thinking, raw_output=content,
            ))

        metrics = compute_metrics(results)
        print_metrics(metrics)

        eval_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        gen_slug = (
            f"{mode_slug}_tb{args.thinking_budget}_n{len(results)}" if use_thinking
            else f"{mode_slug}_max{args.max_new_tokens}_n{len(results)}"
        )
        results_file = eval_dir / f"{args.dataset}_{gen_slug}_{ts}.jsonl"
        summary_file = eval_dir / f"{args.dataset}_{gen_slug}_{ts}_summary.json"
        with open(results_file, "w") as f:
            for r in results:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        with open(summary_file, "w") as f:
            json.dump(dict(
                model_path=label, backend="vllm", dataset=args.dataset, timestamp=ts,
                prompt_config=str(prompt_cfg_path),
                generation=dict(
                    mode=mode_slug, thinking=use_thinking,
                    temperature=temperature, top_p=top_p,
                    top_k=args.top_k, min_p=args.min_p,
                    presence_penalty=args.presence_penalty,
                    max_tokens=max_tokens, thinking_budget=args.thinking_budget,
                    tensor_parallel_size=args.tensor_parallel_size,
                ),
                metrics=metrics,
            ), f, indent=2)
        print(f"  → {summary_file}")
        return read_metrics(summary_file)

    def _eval_adapter(
        adapter_path: Optional[Path],
        eval_dir: Path,
        label: str,
        lora_uid: int,
    ) -> Optional[Dict]:
        """Run num_eval_runs passes and return averaged metrics."""
        done_marker = eval_dir / "DONE"
        if args.skip_existing and done_marker.exists():
            print(f"  Skipping existing eval: {eval_dir.name}")
            summaries = sorted(eval_dir.glob(f"{args.dataset}_*_summary.json"))
            if summaries:
                return average_metrics([read_metrics(s) for s in summaries])
            return None

        lora_request = (
            LoRARequest(f"adapter_{lora_uid}", lora_uid, str(adapter_path))
            if adapter_path is not None else None
        )
        all_metrics = []
        for run_idx in range(args.num_eval_runs):
            print(f"\n{'─'*50}")
            print(f"  Eval: {label}  (run {run_idx + 1}/{args.num_eval_runs})")
            print(f"{'─'*50}")
            m = _infer_and_save(lora_request, eval_dir, label)
            all_metrics.append(m)

        if not args.dry_run:
            done_marker.write_text("done\n", encoding="utf-8")

        return average_metrics(all_metrics)

    # ── Run evals ─────────────────────────────────────────────────────────────
    all_rows: List[Dict] = []

    # Optional: evaluate base model (no LoRA) — lora_request=None works with enable_lora=True
    if args.eval_base_model:
        print(f"\n{'='*50}\nBase model evaluation (no LoRA): {base_model}\n{'='*50}")
        base_eval_dir = results_root / "eval_base_model"
        base_metrics = _eval_adapter(None, base_eval_dir, base_model, lora_uid=0)
        if base_metrics and not args.dry_run:
            row: Dict = dict(index=0, fraction=0.0, fractions=args.fractions,
                             train_count=0, train_file="", label="base_model")
            row.update(base_metrics)
            all_rows.append(row)

    # Evaluate each trained adapter
    for split in splits:
        idx = split["index"]
        adapter_dir = output_root / f"adapter_frac_{idx:02d}_of_{args.fractions}"
        eval_dir = results_root / f"eval_frac_{idx:02d}_of_{args.fractions}"

        print(f"\n{'='*50}")
        print(f"Evaluating shard {idx}/{args.fractions}  ({split['train_count']} training samples)")
        print(f"  Adapter: {adapter_dir}")
        print(f"{'='*50}")

        if not (adapter_dir / "adapter_config.json").exists():
            print(f"  WARNING: adapter_config.json not found — skipping shard {idx}.")
            continue

        metrics = _eval_adapter(adapter_dir, eval_dir, str(adapter_dir), lora_uid=idx)
        row = dict(split)
        if metrics and not args.dry_run:
            row.update(metrics)
        all_rows.append(row)

    print(f"\n{'='*60}")
    print("PHASE 2 complete — all evaluations done.")
    print(f"{'='*60}")
    return all_rows


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    experiment_name = args.experiment_name or datetime.now().strftime("sft_scaling_%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) / experiment_name
    results_root = Path(args.results_root) / experiment_name
    split_dir = output_root / "splits"
    log_dir = output_root / "logs"

    print(f"\n{'='*60}")
    print(f"MedSeRL SFT Scaling Experiment")
    print(f"  Experiment : {experiment_name}")
    print(f"  Train file : {args.train_file}")
    print(f"  Model      : {args.model_name}")
    print(f"  Fractions  : {args.fractions}")
    print(f"  Eval split : {args.eval_split}")
    print(f"  Grad accum : {args.grad_accum}  (eff. batch = {args.batch_size * args.grad_accum})")
    print(f"  Eval runs  : {args.num_eval_runs}")
    print(f"{'='*60}")

    records = load_jsonl(Path(args.train_file))
    splits = make_cumulative_splits(
        records, split_dir=split_dir, fractions=args.fractions,
        seed=args.seed, overwrite=args.overwrite_splits,
    )

    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: train all shards ─────────────────────────────────────────────
    run_phase1_training(args, splits, output_root, log_dir)

    # ── Phase 2: evaluate all with single vLLM session ───────────────────────
    rows = run_phase2_eval(args, splits, output_root, results_root)

    # ── Save summary CSV + JSON + plot ────────────────────────────────────────
    if rows and not args.dry_run:
        results_root.mkdir(parents=True, exist_ok=True)
        csv_path = results_root / "accuracy_vs_sft_quantity.csv"
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        summary_path = results_root / "accuracy_vs_sft_quantity.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(dict(
                experiment_name=experiment_name,
                train_file=args.train_file,
                model_name=args.model_name,
                dataset=args.dataset,
                seed=args.seed,
                rows=rows,
            ), f, indent=2)

        maybe_write_plot(csv_path, results_root / "accuracy_vs_sft_quantity.png")
        print(f"\nSummary written to: {summary_path}")


if __name__ == "__main__":
    main()
