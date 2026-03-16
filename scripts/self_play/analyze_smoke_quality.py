#!/usr/bin/env python3
"""Step 3: Response Quality Analysis for the training-only smoke test.

Usage:
    python3 scripts/self_play/analyze_smoke_quality.py \
        --project-root <PROJECT_ROOT> \
        --output-dir   <OUTPUT_DIR> \
        --smoke-log    <SMOKE_LOG>
"""
import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--project-root",        required=True)
    p.add_argument("--output-dir",          required=True)
    p.add_argument("--smoke-log",           required=True)
    p.add_argument("--max-response-length", type=int, default=6144,
                   help="Token budget for the full response (all turns combined)")
    return p.parse_args()


def strip_thinking(text):
    m = re.search(r"<think>(.*?)</think>\s*", text, re.DOTALL)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return "", text


def tokens(text):
    return int(len(text) / 1.4)  # ~1.4 chars/token for medical text


def find_source(project_root: Path, output_dir: Path, smoke_log: Path):
    trace_dir = project_root / "results/self_play/interactions"
    game_log  = project_root / output_dir / "game_interactions.jsonl"

    if game_log.exists() and game_log.stat().st_size > 0:
        return game_log, "game (interaction, run-specific)"

    if trace_dir.exists():
        game_files = sorted(trace_dir.glob("game_*.jsonl"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
        all_files  = sorted(trace_dir.glob("*.jsonl"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
    else:
        game_files, all_files = [], []

    if game_files:
        return game_files[0], "game (interaction)"
    if all_files:
        latest    = all_files[0].stat().st_mtime
        run_files = [f for f in all_files if latest - f.stat().st_mtime < 300]
        return run_files[-1], "interactions (oldest-in-run fallback)"

    return None, None


def load_records(src: Path):
    records = []
    for line in src.read_text().splitlines():
        if line.startswith("{") and '"timestamp"' in line:
            try:
                records.append(json.loads(line))
            except Exception:
                pass
    return records


def main():
    args = parse_args()
    root = Path(args.project_root)
    outdir = Path(args.output_dir)
    slog = Path(args.smoke_log)
    budget = args.max_response_length

    src, label = find_source(root, outdir, slog)

    if src is None:
        print("  No JSONL found — checking smoke log for inline records ...")
        src = slog
    else:
        try:
            rel = src.relative_to(root)
        except ValueError:
            rel = src.name
        print(f"  Reading ({label}): {rel}")

    records = load_records(src)

    if not records:
        print("  No JSONL records found — model may not have produced any rewards yet")
        sys.exit(0)

    print(f"  Total records : {len(records)}")

    # Per-mode breakdown
    by_mode = defaultdict(list)
    for r in records:
        by_mode[r.get("mode", "?")].append(r)
    for mode, recs in sorted(by_mode.items()):
        print(f"  Mode '{mode}': {len(recs)} records")

    # Multi-turn health check
    assessor_ran   = sum(1 for r in records if r.get("assessor_actually_ran", False))
    phases_sep     = sum(1 for r in records if r.get("phases_separated", False))
    outcomes_found = sum(1 for r in records if r.get("outcome") not in (None, "invalid_format"))
    print(f"  Assessor ran (split)  : {assessor_ran}/{len(records)}")
    print(f"  Phases separated      : {phases_sep}/{len(records)}")
    print(f"  Valid reward outcomes : {outcomes_found}/{len(records)}")
    if assessor_ran == 0 and outcomes_found == 0:
        print("  WARNING: No assessor responses AND no valid outcomes — multi-turn may be broken")
    elif assessor_ran == 0:
        print("  NOTE: Turn split not found in solution_str (verl strips chat tokens).")
        print("        Rewards computed correctly from full response. Use model_response_full for debug.")

    # Analyze full response
    inj_stats = []
    for r in records:
        resp = r.get("injector_response", "") or r.get("model_response_full", "")
        if not resp:
            continue
        has_open  = "<think>" in resp
        has_close = "</think>" in resp
        truncated   = has_open and not has_close
        think_count = resp.count("</think>")
        thinking_part, answer = strip_thinking(resp)
        inj_stats.append({
            "total_tok":    tokens(resp),
            "think_tok":    tokens(thinking_part),
            "answer_tok":   tokens(answer),
            "truncated":    truncated,
            "think_count":  think_count,
            "answer_empty": len(answer.strip()) == 0,
        })

    # Analyze assessor responses if the split worked
    assess_stats = []
    for r in records:
        resp = r.get("assessor_response", "")
        if not resp:
            continue
        thinking, answer = strip_thinking(resp)
        assess_stats.append({
            "total_tok":  tokens(resp),
            "think_tok":  tokens(thinking),
            "answer_tok": tokens(answer),
            "has_think":  "<think>" in resp,
            "answer":     answer.strip(),
        })

    if inj_stats:
        n         = len(inj_stats)
        avg_tot   = sum(s["total_tok"]  for s in inj_stats) / n
        avg_think = sum(s["think_tok"]  for s in inj_stats) / n
        avg_ans   = sum(s["answer_tok"] for s in inj_stats) / n
        max_tot   = max(s["total_tok"]  for s in inj_stats)
        truncated = sum(1 for s in inj_stats if s["truncated"])
        over_bud  = sum(1 for s in inj_stats if s["total_tok"] > budget)
        two_turns = sum(1 for s in inj_stats if s["think_count"] >= 2)

        print(f"""
  ── Full response (injector + assessor, or assessor-only) ──
  Samples           : {n}
  Tokens  avg/max   : {avg_tot:,.0f} / {max_tot:,}  (budget: {budget})
    Thinking avg    : {avg_think:,.0f}
    Answer   avg    : {avg_ans:,.0f}
  Over budget       : {over_bud}/{n}
  Inj truncated (no </think>) : {truncated}/{n}  {'WARNING ' if truncated > n//4 else 'OK'}
  Two </think> blocks (2 turns visible) : {two_turns}/{n}""")

    if assess_stats:
        n               = len(assess_stats)
        avg_tot         = sum(s["total_tok"]  for s in assess_stats) / n
        avg_think       = sum(s["think_tok"]  for s in assess_stats) / n
        avg_ans         = sum(s["answer_tok"] for s in assess_stats) / n
        max_tot         = max(s["total_tok"]  for s in assess_stats)
        correct_answers = sum(1 for s in assess_stats if re.search(r'\bCORRECT\b', s["answer"], re.IGNORECASE))
        numeric_answers = sum(1 for s in assess_stats if re.search(r'^\s*\d+\s*$', s["answer"]))
        empty_answers   = sum(1 for s in assess_stats if not s["answer"].strip())

        print(f"""
  ── Assessor (Phase 2 split succeeded) ────────────────
  Samples           : {n}
  Tokens  avg/max   : {avg_tot:,.0f} / {max_tot:,}
    Thinking avg    : {avg_think:,.0f}
    Answer   avg    : {avg_ans:,.0f}
  Answer patterns:
    'CORRECT'       : {correct_answers}/{n}
    Sentence nums   : {numeric_answers}/{n}
    Empty           : {empty_answers}/{n}""")

    # Outcome breakdown
    outcome_counts = Counter(r.get("outcome", "unknown") for r in records)
    rewards    = [r.get("reward", 0.0) for r in records]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    invalid    = sum(1 for r in records if not r.get("has_valid_format", True))

    print(f"\n  ── Reward outcomes ──────────────────────────────────")
    for oc, cnt in sorted(outcome_counts.items(), key=lambda x: -x[1]):
        print(f"  {oc:20s}: {cnt}")
    print(f"  Avg reward        : {avg_reward:.3f}")
    print(f"  Invalid format    : {invalid}/{len(records)}")

    # Training metrics from smoke log
    try:
        grep_out = subprocess.run(
            ["grep", "-oP", r"critic/score/mean:[0-9e.+\-]+", str(slog)],
            capture_output=True, text=True
        )
        scores = [float(x.split(":")[1]) for x in grep_out.stdout.strip().split("\n") if x]
        if scores:
            print(f"\n  Training critic/score/mean : {scores[0]:.4f} (step 1) → {scores[-1]:.4f} (last step)")
    except Exception:
        pass

    # Sample records
    sample_recs = [r for r in records if r.get("injector_response") or r.get("assessor_response")][:3]
    if sample_recs:
        print(f"\n  ── Sample game records ──────────────────────────────")
        for i, r in enumerate(sample_recs):
            gt  = r.get("ground_truth", "?")
            oc  = r.get("outcome", "?")
            rwd = r.get("reward", 0.0)
            inj = (r.get("injector_response") or "")[:150].replace("\n", " ↵ ")
            asm = (r.get("assessor_response") or "")[:80].replace("\n", " ↵ ")
            print(f"  [{i+1}] gt={gt!r:8} outcome={oc:15} reward={rwd:+.2f}")
            if inj:
                print(f"       INJ: {inj!r}")
            if asm:
                print(f"       ASM: {asm!r}")
    print()


if __name__ == "__main__":
    main()
