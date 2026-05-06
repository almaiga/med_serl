#!/usr/bin/env python3
"""Split a training JSONL into 8 cumulative subsets for a data-scaling experiment.

Usage
-----
    python scripts/scaling/split_training_data.py \
        --input data_processed/medrect/generated_assessor_all_sft.jsonl \
        --output-dir data_processed/scaling/ \
        --n-shards 8 --seed 42
"""

import argparse
import json
import math
import random
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input",       required=True, help="Source JSONL file")
    p.add_argument("--output-dir",  default="data_processed/scaling")
    p.add_argument("--n-shards",    type=int, default=8)
    p.add_argument("--seed",        type=int, default=42)
    args = p.parse_args()

    src = Path(args.input)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    records = [json.loads(l) for l in src.read_text().splitlines() if l.strip()]
    rng = random.Random(args.seed)
    rng.shuffle(records)
    N = len(records)

    print(f"Source : {src}  ({N} samples)")
    print(f"Shards : {args.n_shards}  |  seed={args.seed}")
    print(f"Output : {out}/")
    print()
    print(f"{'Shard':<8} {'Samples':>8} {'Fraction':>10}  File")
    print("-" * 55)

    for k in range(1, args.n_shards + 1):
        size = math.floor(N * k / args.n_shards)
        subset = records[:size]
        dest = out / f"scale_{k}of{args.n_shards}.jsonl"
        with dest.open("w") as f:
            for row in subset:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"{k}/{args.n_shards:<5} {size:>8}   {size/N:>8.1%}   {dest}")

    print()
    print("Done.")


if __name__ == "__main__":
    main()
