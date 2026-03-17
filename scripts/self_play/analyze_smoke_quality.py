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
    p.add_argument("--project-root", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--smoke-log", required=True)
    p.add_argument(
        "--max-response-length",
        type=int,
        default=6144,
        help="Token budget for the full response (all turns combined)",
    )
    return p.parse_args()


def strip_thinking(text):
    m = re.search(r"<think>(.*?)</think>\s*", text, re.DOTALL)
    if m:
        return m.group(1).strip(), text[m.end():].strip()
    return "", text


def tokens(text):
    return int(len(text) / 1.4)  # ~1.4 chars/token for medical text


def find_sources(project_root: Path, output_dir: Path, smoke_log: Path):
    """Return (list_of_paths, label).

    Priority:
      1. game_interactions.jsonl in the run output_dir (multi-turn)
      2. Any *.jsonl in the run output_dir
      3. game_*.jsonl in results/self_play/interactions/ after run start
      4. interactions_*.jsonl after run start (GRPO-separated)
    """
    # Determine when this run started (smoke_log created ~ run start)
    import time as _time
    if smoke_log.exists():
        run_start = smoke_log.stat().st_mtime
    else:
        run_start = _time.time() - 3600  # fall back to 1h ago

    # 1. Run-specific game log (MEDSERL_GAME_LOG path)
    game_log = output_dir / "game_interactions.jsonl"
    if not game_log.is_absolute():
        game_log = project_root / game_log
    if game_log.exists() and game_log.stat().st_size > 0:
        return [game_log], "game (interaction, run-specific)"

    # 2. Any jsonl files written inside the output_dir itself
    out_abs = (
        output_dir if output_dir.is_absolute()
        else project_root / output_dir
    )
    if out_abs.exists():
        out_files = sorted(
            out_abs.glob("*.jsonl"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if out_files:
            return out_files, f"output_dir jsonl ({len(out_files)} files)"

    # Shared interaction log directory
    trace_dir = project_root / "results/self_play/interactions"
    if not trace_dir.exists():
        return [], None

    all_trace = sorted(
        trace_dir.glob("*.jsonl"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    # 3. game_*.jsonl files created after this run started
    game_files = [
        f for f in all_trace
        if f.name.startswith("game_") and f.stat().st_mtime >= run_start
    ]
    if game_files:
        return [game_files[0]], "game (interaction, this run)"

    # 4. interactions_*.jsonl created after this run started (GRPO-separated)
    run_files = [
        f for f in all_trace
        if f.stat().st_mtime >= run_start
    ]
    if run_files:
        label = f"interactions ({len(run_files)} worker files, this run)"
        return run_files, label

    print(
        "  WARNING: No interaction files found for this run."
        " Showing most recent file (may be from a previous run):"
    )
    if all_trace:
        return [all_trace[0]], f"STALE: {all_trace[0].name}"
    return [], None


def load_records(sources):
    """Load JSONL records from a list of paths."""
    if isinstance(sources, Path):
        sources = [sources]
    records = []
    for src in sources:
        for line in src.read_text().splitlines():
            if line.startswith("{") and '"timestamp"' in line:
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    return records


def pct(n, d):
    return f"{n / d * 100:.0f}%" if d else "N/A"


def avg_chars_str(chars):
    if not chars:
        return "N/A"
    a = sum(chars) / len(chars)
    return f"{a:.0f} chars ({a / 4:.0f} tokens)"


def _print_assessor(asm_recs):
    n = len(asm_recs)
    asm_benign = [r for r in asm_recs if r.get("mode") == "benign"]
    asm_error = [r for r in asm_recs if r.get("mode") == "error"]
    asm_oc = Counter(r.get("outcome", "?") for r in asm_recs)
    rewards = [r.get("reward", 0.0) for r in asm_recs]
    chars = [r["response_chars"] for r in asm_recs if r.get("response_chars")]
    trunc = sum(1 for r in asm_recs if r.get("is_truncated"))
    think = sum(1 for r in asm_recs if r.get("has_think_tag"))

    nb = len(asm_benign)
    ne = len(asm_error)
    bneg_correct = sum(
        1 for r in asm_benign if r.get("outcome") == "exact_match"
    )
    bneg_fp = sum(
        1 for r in asm_benign
        if r.get("outcome") in ("miss", "partial_match")
    )
    bneg_inv = sum(
        1 for r in asm_benign if r.get("outcome") == "invalid_format"
    )
    err_exact = sum(1 for r in asm_error if r.get("outcome") == "exact_match")
    err_partial = sum(
        1 for r in asm_error if r.get("outcome") == "partial_match"
    )
    err_miss = sum(1 for r in asm_error if r.get("outcome") == "miss")
    err_inv = sum(
        1 for r in asm_error if r.get("outcome") == "invalid_format"
    )

    avg_rwd = sum(rewards) / len(rewards)
    print(f"\n  ── Assessor ({n} records) " + "─" * 33)
    print(f"  Avg reward   : {avg_rwd:+.3f}")
    print(f"  Truncated    : {trunc}/{n}   Has <think>: {think}/{n}")
    print(f"  Avg response : {avg_chars_str(chars)}")

    print(f"\n  Benign notes (gt=CORRECT) — {nb} samples:")
    print(f"    True negative  (said CORRECT)   :"
          f" {bneg_correct:3d}/{nb}  {pct(bneg_correct, nb)}")
    print(f"    False positive (flagged error)   :"
          f" {bneg_fp:3d}/{nb}  {pct(bneg_fp, nb)}")
    print(f"    Invalid format                   :"
          f" {bneg_inv:3d}/{nb}  {pct(bneg_inv, nb)}")

    print(f"\n  Error notes (gt=sentence_id) — {ne} samples:")
    print(f"    Exact  (correct sentence)        :"
          f" {err_exact:3d}/{ne}  {pct(err_exact, ne)}")
    print(f"    Partial (detected, wrong sent)   :"
          f" {err_partial:3d}/{ne}  {pct(err_partial, ne)}")
    print(f"    Miss   (said CORRECT on error)   :"
          f" {err_miss:3d}/{ne}  {pct(err_miss, ne)}")
    print(f"    Invalid format                   :"
          f" {err_inv:3d}/{ne}  {pct(err_inv, ne)}")

    print(
        f"\n  Overall: exact={asm_oc['exact_match']}"
        f"  partial={asm_oc['partial_match']}"
        f"  miss={asm_oc['miss']}"
        f"  invalid={asm_oc['invalid_format']}"
    )
    pred_labels = Counter(r.get("pred_label", "?") for r in asm_recs)
    print(f"  Pred labels  : {dict(pred_labels)}")


def _print_multiturn_game(records):
    """Per-mode analysis for multi-turn game log records."""
    n = len(records)
    rewards = [r.get("reward", 0.0) for r in records]
    avg_rwd = sum(rewards) / len(rewards) if rewards else 0.0

    by_mode = defaultdict(list)
    for r in records:
        by_mode[r.get("mode", "?")].append(r)

    benign_recs = by_mode.get("benign", [])
    error_recs = by_mode.get("error_injection", [])
    nb = len(benign_recs)
    ne = len(error_recs)

    print(f"\n  ── Multi-turn game ({n} records) " + "─" * 26)
    print(f"  Avg reward   : {avg_rwd:+.3f}")
    print(f"  Benign games : {nb}   Error injection games : {ne}")

    if nb:
        bneg_tn = sum(
            1 for r in benign_recs if r.get("outcome") == "exact_match"
        )
        bneg_fp = sum(
            1 for r in benign_recs
            if r.get("outcome") in ("miss", "partial_match")
        )
        bneg_inv = sum(
            1 for r in benign_recs if r.get("outcome") == "invalid_format"
        )
        bneg_rwd = (
            sum(r.get("reward", 0.0) for r in benign_recs) / nb
        )
        print(
            f"\n  Benign games (injector benign edit;"
            f" assessor should say CORRECT) — {nb} samples:"
        )
        print(f"    Avg reward                       : {bneg_rwd:+.3f}")
        print(
            f"    True negative  (said CORRECT)    :"
            f" {bneg_tn:3d}/{nb}  {pct(bneg_tn, nb)}"
        )
        print(
            f"    False positive (flagged error)    :"
            f" {bneg_fp:3d}/{nb}  {pct(bneg_fp, nb)}"
        )
        print(
            f"    Invalid format                    :"
            f" {bneg_inv:3d}/{nb}  {pct(bneg_inv, nb)}"
        )

    if ne:
        err_exact = sum(
            1 for r in error_recs if r.get("outcome") == "exact_match"
        )
        err_partial = sum(
            1 for r in error_recs if r.get("outcome") == "partial_match"
        )
        err_miss = sum(
            1 for r in error_recs if r.get("outcome") == "miss"
        )
        err_inv = sum(
            1 for r in error_recs if r.get("outcome") == "invalid_format"
        )
        err_rwd = (
            sum(r.get("reward", 0.0) for r in error_recs) / ne
        )
        print(
            f"\n  Error injection games"
            f" (assessor sees injected note) — {ne} samples:"
        )
        print(f"    Avg reward                       : {err_rwd:+.3f}")
        print(
            f"    Exact  (correct sentence)        :"
            f" {err_exact:3d}/{ne}  {pct(err_exact, ne)}"
        )
        print(
            f"    Partial (detected, wrong sent)   :"
            f" {err_partial:3d}/{ne}  {pct(err_partial, ne)}"
        )
        print(
            f"    Miss   (said CORRECT on error)   :"
            f" {err_miss:3d}/{ne}  {pct(err_miss, ne)}"
        )
        print(
            f"    Invalid format                   :"
            f" {err_inv:3d}/{ne}  {pct(err_inv, ne)}"
        )

    labels = Counter(r.get("assessor_label", "?") for r in records)
    print(f"\n  Assessor labels: {dict(labels)}")


def _print_injector(inj_recs):
    n = len(inj_recs)
    inj_benign = [r for r in inj_recs if r.get("mode") == "benign"]
    inj_error = [r for r in inj_recs if r.get("mode") == "error_injection"]
    inj_oc = Counter(r.get("outcome", "?") for r in inj_recs)
    rewards = [r.get("reward", 0.0) for r in inj_recs]
    chars = [r["response_chars"] for r in inj_recs if r.get("response_chars")]
    trunc = sum(1 for r in inj_recs if r.get("is_truncated"))
    think = sum(1 for r in inj_recs if r.get("has_think_tag"))

    nb = len(inj_benign)
    ne = len(inj_error)
    b_exact = sum(1 for r in inj_benign if r.get("outcome") == "exact_match")
    b_inv = sum(
        1 for r in inj_benign if r.get("outcome") == "invalid_format"
    )
    e_exact = sum(1 for r in inj_error if r.get("outcome") == "exact_match")
    e_partial = sum(
        1 for r in inj_error if r.get("outcome") == "partial_match"
    )
    e_inv = sum(
        1 for r in inj_error if r.get("outcome") == "invalid_format"
    )

    avg_rwd = sum(rewards) / len(rewards)
    print(f"\n  ── Injector ({n} records) " + "─" * 33)
    print(f"  Avg reward   : {avg_rwd:+.3f}")
    print(f"  Truncated    : {trunc}/{n}   Has <think>: {think}/{n}")
    print(f"  Avg response : {avg_chars_str(chars)}")

    print(f"\n  Benign edits — {nb} samples:")
    print(f"    Valid format (rewarded)          :"
          f" {b_exact:3d}/{nb}  {pct(b_exact, nb)}")
    print(f"    Invalid format                   :"
          f" {b_inv:3d}/{nb}  {pct(b_inv, nb)}")

    print(f"\n  Error injection — {ne} samples:")
    print(f"    Correct sentence (gt match)      :"
          f" {e_exact:3d}/{ne}  {pct(e_exact, ne)}")
    print(f"    Valid but wrong sentence         :"
          f" {e_partial:3d}/{ne}  {pct(e_partial, ne)}")
    print(f"    Invalid format                   :"
          f" {e_inv:3d}/{ne}  {pct(e_inv, ne)}")

    print(
        f"\n  Overall: exact={inj_oc['exact_match']}"
        f"  partial={inj_oc['partial_match']}"
        f"  miss={inj_oc['miss']}"
        f"  invalid={inj_oc['invalid_format']}"
    )


def main():
    args = parse_args()
    root = Path(args.project_root)
    outdir = Path(args.output_dir)
    slog = Path(args.smoke_log)
    budget = args.max_response_length

    sources, label = find_sources(root, outdir, slog)

    if not sources:
        print("  No JSONL found — checking smoke log for inline records ...")
        sources = [slog]
    else:
        for src in sources:
            try:
                rel = src.relative_to(root)
            except ValueError:
                rel = src.name
            print(f"  Reading ({label}): {rel}")

    records = load_records(sources)

    if not records:
        print(
            "  No JSONL records found"
            " — model may not have produced rewards"
        )
        sys.exit(0)

    print(f"  Total records : {len(records)}")

    by_mode = defaultdict(list)
    for r in records:
        by_mode[r.get("mode", "?")].append(r)
    for mode, recs in sorted(by_mode.items()):
        print(f"  Mode '{mode}': {len(recs)} records")

    # Detect run type: GRPO-separated (role field) vs multi-turn
    roles_present = {r.get("role") for r in records if r.get("role")}
    is_grpo_sep = bool(roles_present & {"injector", "assessor"})
    outcomes_found = sum(
        1 for r in records
        if r.get("outcome") not in (None, "invalid_format")
    )

    if is_grpo_sep:
        by_role = defaultdict(list)
        for r in records:
            by_role[r.get("role", "unknown")].append(r)

        n_inj = len(by_role.get("injector", []))
        n_asm = len(by_role.get("assessor", []))
        n_tot = len(records)

        print(
            "\n  ── GRPO Separated (single-turn, mixed roles) "
            + "─" * 14
        )
        print(
            f"  Total records  : {n_tot}"
            f"  (injector={n_inj}, assessor={n_asm})"
        )
        print(f"  Valid outcomes : {outcomes_found}/{n_tot}")
        if n_inj != n_asm:
            print(
                f"  NOTE: {n_inj}/{n_asm} injector/assessor split"
                f" (expected ~equal)."
            )
            print(
                "        Small batches + random sampling cause variance;"
                " rebalances over full training."
            )

        if by_role.get("assessor"):
            _print_assessor(by_role["assessor"])
        if by_role.get("injector"):
            _print_injector(by_role["injector"])

    else:
        assessor_ran = sum(
            1 for r in records if r.get("assessor_actually_ran", False)
        )
        phases_sep = sum(
            1 for r in records if r.get("phases_separated", False)
        )
        print(f"\n  Total records          : {len(records)}")
        print(f"  Assessor ran (turn 2)  : {assessor_ran}/{len(records)}")
        print(f"  Valid reward outcomes   : {outcomes_found}/{len(records)}")
        if assessor_ran == 0 and outcomes_found == 0:
            print(
                "  WARNING: No assessor responses AND no valid outcomes"
                " — check game_interactions.jsonl"
            )
        else:
            _print_multiturn_game(records)

    # Multi-turn response analysis (only meaningful for game log records)
    inj_stats = []
    for r in records:
        resp = (
            r.get("injector_response", "")
            or r.get("model_response_full", "")
        )
        if not resp:
            continue
        has_open = "<think>" in resp
        has_close = "</think>" in resp
        truncated = has_open and not has_close
        think_count = resp.count("</think>")
        thinking_part, answer = strip_thinking(resp)
        inj_stats.append({
            "total_tok": tokens(resp),
            "think_tok": tokens(thinking_part),
            "answer_tok": tokens(answer),
            "truncated": truncated,
            "think_count": think_count,
            "answer_empty": len(answer.strip()) == 0,
        })

    assess_stats = []
    for r in records:
        resp = r.get("assessor_response", "")
        if not resp:
            continue
        thinking, answer = strip_thinking(resp)
        assess_stats.append({
            "total_tok": tokens(resp),
            "think_tok": tokens(thinking),
            "answer_tok": tokens(answer),
            "has_think": "<think>" in resp,
            "answer": answer.strip(),
        })

    if inj_stats:
        n = len(inj_stats)
        avg_tot = sum(s["total_tok"] for s in inj_stats) / n
        avg_think = sum(s["think_tok"] for s in inj_stats) / n
        avg_ans = sum(s["answer_tok"] for s in inj_stats) / n
        max_tot = max(s["total_tok"] for s in inj_stats)
        truncated = sum(1 for s in inj_stats if s["truncated"])
        over_bud = sum(1 for s in inj_stats if s["total_tok"] > budget)
        two_turns = sum(1 for s in inj_stats if s["think_count"] >= 2)
        warn = "WARNING" if truncated > n // 4 else "OK"
        print(
            "\n  ── Full response (injector+assessor combined) "
            + "─" * 13
        )
        print(f"  Samples          : {n}")
        print(f"  Tokens avg/max   : {avg_tot:,.0f} / {max_tot:,}"
              f"  (budget: {budget})")
        print(f"    Thinking avg   : {avg_think:,.0f}")
        print(f"    Answer avg     : {avg_ans:,.0f}")
        print(f"  Over budget      : {over_bud}/{n}")
        print(f"  Truncated        : {truncated}/{n}  {warn}")
        print(f"  Two-turn visible : {two_turns}/{n}")

    if assess_stats:
        n = len(assess_stats)
        avg_tot = sum(s["total_tok"] for s in assess_stats) / n
        avg_think = sum(s["think_tok"] for s in assess_stats) / n
        avg_ans = sum(s["answer_tok"] for s in assess_stats) / n
        max_tot = max(s["total_tok"] for s in assess_stats)
        correct_answers = sum(
            1 for s in assess_stats
            if re.search(r"\bCORRECT\b", s["answer"], re.IGNORECASE)
        )
        numeric_answers = sum(
            1 for s in assess_stats
            if re.search(r"^\s*\d+\s*$", s["answer"])
        )
        empty_answers = sum(1 for s in assess_stats if not s["answer"].strip())
        print("\n  ── Assessor (Phase 2 split succeeded) " + "─" * 21)
        print(f"  Samples          : {n}")
        print(f"  Tokens avg/max   : {avg_tot:,.0f} / {max_tot:,}")
        print(f"    Thinking avg   : {avg_think:,.0f}")
        print(f"    Answer avg     : {avg_ans:,.0f}")
        print(f"  'CORRECT'        : {correct_answers}/{n}")
        print(f"  Sentence nums    : {numeric_answers}/{n}")
        print(f"  Empty            : {empty_answers}/{n}")

    # Overall reward breakdown
    outcome_counts = Counter(r.get("outcome", "unknown") for r in records)
    rewards = [r.get("reward", 0.0) for r in records]
    avg_reward = sum(rewards) / len(rewards) if rewards else 0.0
    invalid = sum(1 for r in records if not r.get("has_valid_format", True))

    print("\n  ── Reward outcomes " + "─" * 40)
    for oc, cnt in sorted(outcome_counts.items(), key=lambda x: -x[1]):
        print(f"  {oc:20s}: {cnt}")
    print(f"  Avg reward          : {avg_reward:.3f}")
    print(f"  Invalid format      : {invalid}/{len(records)}")

    # Training metrics from smoke log
    try:
        grep_out = subprocess.run(
            ["grep", "-oP", r"critic/score/mean:[0-9e.+\-]+", str(slog)],
            capture_output=True,
            text=True,
        )
        scores = [
            float(x.split(":")[1])
            for x in grep_out.stdout.strip().split("\n")
            if x
        ]
        if scores:
            print(
                f"\n  Training critic/score/mean :"
                f" {scores[0]:.4f} (step 1)"
                f" → {scores[-1]:.4f} (last step)"
            )
    except Exception:
        pass

    # Sample records
    if is_grpo_sep:
        sample_recs = []
        for role in ("assessor", "injector"):
            role_recs = [r for r in records if r.get("role") == role]
            if role_recs:
                sample_recs.append(role_recs[0])
    else:
        sample_recs = [
            r for r in records
            if r.get("injector_response") or r.get("assessor_response")
        ][:3]

    if sample_recs:
        print("\n  ── Sample records " + "─" * 40)
        for i, r in enumerate(sample_recs):
            gt = r.get("ground_truth", "?")
            oc = r.get("outcome", "?")
            rwd = r.get("reward", 0.0)
            role = r.get("role", "")
            role_tag = f" role={role}" if role else ""
            print(
                f"  [{i + 1}]{role_tag}"
                f" gt={gt!r:8} outcome={oc:15} reward={rwd:+.2f}"
            )
            resp = (
                r.get("injector_response")
                or r.get("model_response")
                or ""
            )[:150].replace("\n", " ↵ ")
            asm = (r.get("assessor_response") or "")[:80].replace(
                "\n", " ↵ "
            )
            if resp:
                print(f"       resp: {resp!r}")
            if asm:
                print(f"       ASM : {asm!r}")
    print()


if __name__ == "__main__":
    main()
