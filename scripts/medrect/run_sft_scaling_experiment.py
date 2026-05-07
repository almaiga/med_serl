#!/usr/bin/env python3
"""
Run MEDRECT SFT data-scaling experiments.

The script shuffles a prepared SFT JSONL once, splits it into 10 deterministic
parts, trains cumulative subsets (10%, 20%, ..., 100%), evaluates each adapter
on the full MEDEC test set, and writes plot-ready metrics.

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
from typing import Dict, List, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TRAIN_FILE = PROJECT_ROOT / "data_processed" / "medrect" / "generated_assessor_all_sft.jsonl"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs" / "local_training" / "medrect_sft_scaling"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "results" / "medrect_sft_scaling"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/evaluate MEDRECT SFT adapters for 10 cumulative data fractions."
    )
    parser.add_argument("--train-file", default=str(DEFAULT_TRAIN_FILE))
    parser.add_argument("--model-name", default="Qwen/Qwen3-4B")
    parser.add_argument("--base-model-path", default=None)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--results-root", default=str(DEFAULT_RESULTS_ROOT))
    parser.add_argument("--experiment-name", default=None)
    parser.add_argument("--fractions", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite-splits", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")

    parser.add_argument("--dataset", default="all", choices=["ms", "uw", "all"])
    parser.add_argument("--prompt-config", default=None)
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--min-p", type=float, default=0.0)
    parser.add_argument("--presence-penalty", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--thinking-budget", type=int, default=4096)
    parser.add_argument("--no-thinking", action="store_true")
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Number of GPUs for vLLM tensor parallelism during eval")
    parser.add_argument("--num-eval-runs", type=int, default=1,
                        help="Number of inference runs per shard to average (reduces sampling variance)")
    parser.add_argument("--eval-base-model", action="store_true",
                        help="Evaluate the base model (no LoRA) before training shards as a 0-shot baseline")

    parser.add_argument("--nproc-per-node", type=int, default=None)
    parser.add_argument("--eval-split", type=float, default=0.0)
    parser.add_argument("--max-seq-length", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-train-epochs", type=int, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=50)
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


def make_cumulative_splits(records: List[Dict], split_dir: Path, fractions: int, seed: int, overwrite: bool) -> List[Dict]:
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
        manifest.append(
            {
                "index": idx,
                "fractions": fractions,
                "fraction": fraction,
                "train_count": count,
                "train_file": str(path),
            }
        )

    manifest_path = split_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(
            {"seed": seed, "total_records": total, "splits": manifest},
            f,
            indent=2,
        )
    return manifest


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
            cmd,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, cmd)


def latest_summary(output_dir: Path, dataset: str) -> Path:
    summaries = sorted(output_dir.glob(f"{dataset}_*_summary.json"))
    if not summaries:
        raise FileNotFoundError(f"No summary JSON found in {output_dir}")
    return summaries[-1]


def read_metrics(summary_path: Path) -> Dict:
    with open(summary_path, "r", encoding="utf-8") as f:
        summary = json.load(f)
    metrics = summary["metrics"]
    detection = metrics["detection"]
    sentence = metrics["sentence_extraction"]
    return {
        "summary_file": str(summary_path),
        "test_samples": metrics["total_samples"],
        "detection_accuracy": detection["accuracy"],
        "detection_precision": detection["precision"],
        "detection_recall": detection["recall"],
        "detection_f1": detection["f1"],
        "sentence_accuracy": sentence["accuracy"],
        "sentence_exact_matches": sentence["exact_matches"],
        "sentence_total_errors": sentence["total_errors"],
    }


def average_metrics(all_metrics: List[Dict]) -> Dict:
    if len(all_metrics) == 1:
        return all_metrics[0]
    keys = [k for k in all_metrics[0] if isinstance(all_metrics[0][k], (int, float))]
    averaged = {"summary_file": all_metrics[0]["summary_file"], "num_eval_runs": len(all_metrics)}
    for k in keys:
        vals = [m[k] for m in all_metrics]
        averaged[k] = sum(vals) / len(vals)
        averaged[f"{k}_std"] = (sum((v - averaged[k]) ** 2 for v in vals) / len(vals)) ** 0.5
    return averaged


def maybe_write_plot(csv_path: Path, png_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping PNG plot.")
        return

    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    if not rows:
        return

    x = [int(r["train_count"]) for r in rows]
    y = [float(r["detection_accuracy"]) for r in rows]
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


def main() -> None:
    args = parse_args()
    experiment_name = args.experiment_name or datetime.now().strftime("sft_scaling_%Y%m%d_%H%M%S")
    output_root = Path(args.output_root) / experiment_name
    results_root = Path(args.results_root) / experiment_name
    split_dir = output_root / "splits"
    log_dir = output_root / "logs"

    records = load_jsonl(Path(args.train_file))
    splits = make_cumulative_splits(
        records,
        split_dir=split_dir,
        fractions=args.fractions,
        seed=args.seed,
        overwrite=args.overwrite_splits,
    )

    def build_eval_cmd(model_path: str, eval_dir: Path, base_model_path: Optional[str] = None) -> List[str]:
        cmd = [
            sys.executable,
            "scripts/medrect/inference_detection_vllm.py",
            "--model_path", model_path,
            "--dataset", args.dataset,
            "--top_k", str(args.top_k),
            "--min_p", str(args.min_p),
            "--presence_penalty", str(args.presence_penalty),
            "--max_new_tokens", str(args.max_new_tokens),
            "--thinking_budget", str(args.thinking_budget),
            "--tensor_parallel_size", str(args.tensor_parallel_size),
            "--output_dir", str(eval_dir),
        ]
        if args.prompt_config:
            cmd.extend(["--prompt_config", args.prompt_config])
        if base_model_path:
            cmd.extend(["--base_model_path", base_model_path])
        if args.temperature is not None:
            cmd.extend(["--temperature", str(args.temperature)])
        if args.top_p is not None:
            cmd.extend(["--top_p", str(args.top_p)])
        if args.no_thinking:
            cmd.append("--no_thinking")
        if args.max_eval_samples:
            cmd.extend(["--max_samples", str(args.max_eval_samples)])
        return cmd

    def run_eval_runs(eval_cmd: List[str], eval_dir: Path, log_prefix: str) -> Dict:
        for run_idx in range(args.num_eval_runs):
            log_suffix = f"_run{run_idx + 1}" if args.num_eval_runs > 1 else ""
            run_command(eval_cmd, log_dir / f"{log_prefix}{log_suffix}.log", args.dry_run)
        summaries = sorted(eval_dir.glob(f"{args.dataset}_*_summary.json"))
        all_metrics = [read_metrics(s) for s in summaries]
        return average_metrics(all_metrics)

    rows = []

    # ── Optional base model evaluation (0-shot baseline) ─────────────────────
    if args.eval_base_model:
        base_model_name = args.base_model_path or args.model_name
        base_eval_dir = results_root / "eval_base_model"
        base_done = base_eval_dir / "DONE"
        if args.skip_existing and base_done.exists():
            print(f"Skipping existing base model evaluation: {base_eval_dir}")
        else:
            print(f"\n{'='*50}\nEvaluating base model: {base_model_name}\n{'='*50}")
            eval_cmd = build_eval_cmd(base_model_name, base_eval_dir)
            run_eval_runs(eval_cmd, base_eval_dir, "eval_base_model")
            if not args.dry_run:
                base_done.write_text("done\n", encoding="utf-8")
        if not args.dry_run:
            base_row = {"index": 0, "fraction": 0.0, "fractions": args.fractions, "train_count": 0,
                        "train_file": "", "label": "base_model"}
            summaries = sorted(base_eval_dir.glob(f"{args.dataset}_*_summary.json"))
            base_row.update(average_metrics([read_metrics(s) for s in summaries]))
            rows.append(base_row)

    for split in splits:
        idx = split["index"]
        adapter_dir = output_root / f"adapter_frac_{idx:02d}_of_{args.fractions}"
        eval_dir = results_root / f"eval_frac_{idx:02d}_of_{args.fractions}"
        train_done = adapter_dir / "adapter_config.json"
        eval_done = eval_dir / "DONE"

        if args.skip_existing and train_done.exists():
            print(f"Skipping existing training output: {adapter_dir}")
        else:
            train_cmd = launcher(args.nproc_per_node) + [
                "scripts/medrect/train_medrect_lora.py",
                "--train-file",
                split["train_file"],
                "--model-name",
                args.model_name,
                "--output-dir",
                str(adapter_dir),
                "--eval-split",
                str(args.eval_split),
                "--max-seq-length",
                str(args.max_seq_length),
                "--per-device-train-batch-size",
                str(args.batch_size),
                "--gradient-accumulation-steps",
                str(args.grad_accum),
                "--learning-rate",
                str(args.learning_rate),
                "--num-train-epochs",
                str(args.num_train_epochs),
                "--warmup-ratio",
                str(args.warmup_ratio),
                "--weight-decay",
                str(args.weight_decay),
                "--logging-steps",
                str(args.logging_steps),
                "--save-steps",
                str(args.save_steps),
                "--eval-steps",
                str(args.eval_steps),
                "--lora-r",
                str(args.lora_r),
                "--lora-alpha",
                str(args.lora_alpha),
                "--lora-dropout",
                str(args.lora_dropout),
                "--lora-target-modules",
                args.lora_target_modules,
                "--debug-samples",
                str(args.debug_samples),
                "--dataloader-num-workers",
                str(args.dataloader_num_workers),
                "--seed",
                str(args.seed),
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

        if args.skip_existing and eval_done.exists():
            print(f"Skipping existing evaluation output: {eval_dir}")
        else:
            base_for_eval = args.base_model_path or args.model_name
            eval_cmd = build_eval_cmd(str(adapter_dir), eval_dir, base_model_path=base_for_eval)
            run_eval_runs(eval_cmd, eval_dir, f"eval_frac_{idx:02d}")
            if not args.dry_run:
                eval_done.write_text("done\n", encoding="utf-8")

        row = dict(split)
        if not args.dry_run:
            summaries = sorted(eval_dir.glob(f"{args.dataset}_*_summary.json"))
            all_metrics = [read_metrics(s) for s in summaries]
            row.update(average_metrics(all_metrics))
        rows.append(row)

        csv_path = results_root / "accuracy_vs_sft_quantity.csv"
        results_root.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=sorted(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    summary_path = results_root / "accuracy_vs_sft_quantity.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment_name": experiment_name,
                "train_file": args.train_file,
                "model_name": args.model_name,
                "dataset": args.dataset,
                "seed": args.seed,
                "rows": rows,
            },
            f,
            indent=2,
        )

    if not args.dry_run:
        maybe_write_plot(
            results_root / "accuracy_vs_sft_quantity.csv",
            results_root / "accuracy_vs_sft_quantity.png",
        )
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    main()
