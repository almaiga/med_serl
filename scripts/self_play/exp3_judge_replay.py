#!/usr/bin/env python3
"""Exp 3: replay v5 game-log injector edits through the new judge.

Each v5 game log records the injector's edit (the original sentence, the edited
sentence, the changed sentence id, the resulting modified note) and the old
(Qwen3-8B) judge's verdict on it. We run the new judge (MedRECT-32B + hint_v2)
on the same edits and produce a confusion matrix vs the old verdict, separately
for error_injection and benign games.

The two cells that determine the launch verdict (per failure analysis §5):

  old=SAME -> new=CHANGED on error_injection   the 489-penalty cell, RESCUED
  old=CHANGED -> new=SAME on error_injection   regression, should be ~zero

Plus the benign confusion gives an upper bound on the mirror reward hack on
the actual injector distribution (vs Bucket C's 23 % on hand-crafted probes).

Schema is autodetected — the script prints the first record's keys at startup
so you can see exactly what it picked up. Pass --field-aliases-from-env if you
need to override the defaults.

Cost: ~5-10 min on A100. Output: confusion matrices + a results JSONL of
every record with the old + new verdict for manual spot-check.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.self_play.exp2_run_probes import (  # noqa: E402
    PROMPT_BUILDERS, PARSERS, STYLE_DEFAULTS, model_tag, render,
)

DEFAULT_LOGS = PROJECT_ROOT / "scripts" / "self_play" / "logs"

# Common field-name variants we try, in priority order. The first that exists
# (and is not None) wins. Override at the CLI with --field <logical>=<name>.
FIELD_ALIASES: dict[str, list[str]] = {
    "mode":              ["requested_mode", "mode", "game_mode"],
    "original_sentence": ["original_sentence", "correct_sentence",
                          "orig_sentence", "corr_sentence", "source_sentence"],
    "modified_sentence": ["modified_sentence", "error_sentence",
                          "edited_sentence", "injected_sentence",
                          "injector_modified_sentence"],
    "modified_note":     ["modified_note", "incorrect_note", "edited_note",
                          "note", "sentences", "injected_note",
                          "injector_modified_note"],
    "changed_sid":       ["changed_sid", "error_sentence_id", "sid",
                          "sentence_id", "injected_sid", "injector_sid"],
    "old_verdict":       ["judge_verdict", "verdict", "old_judge_verdict"],
    "injector_outcome":  ["injector_outcome", "inj_outcome"],
    "judge_status":      ["judge_status", "status"],
    "sample_id":         ["sample_id", "note_id", "game_id", "id"],
}


def get_first(record: dict, names: list[str]):
    for n in names:
        if n in record and record[n] not in (None, ""):
            return record[n]
    return None


def extract_probe(record: dict, idx: int) -> dict | None:
    mode = get_first(record, FIELD_ALIASES["mode"])
    if mode not in ("error_injection", "benign"):
        return None
    # Only replay records where the injector produced a clean (parseable) edit
    inj_outcome = get_first(record, FIELD_ALIASES["injector_outcome"])
    if inj_outcome is not None and inj_outcome != "exact_match":
        return None
    sid = get_first(record, FIELD_ALIASES["changed_sid"])
    orig = get_first(record, FIELD_ALIASES["original_sentence"])
    mod = get_first(record, FIELD_ALIASES["modified_sentence"])
    note = get_first(record, FIELD_ALIASES["modified_note"])
    if sid is None or not orig or not mod or not note:
        return None
    try:
        sid_int = int(sid)
    except (TypeError, ValueError):
        return None
    return {
        "probe_id": str(get_first(record, FIELD_ALIASES["sample_id"]) or f"rec_{idx}"),
        "bucket": "REPLAY",
        "bucket_name": f"v5_{mode}",
        "note_id": str(get_first(record, FIELD_ALIASES["sample_id"]) or ""),
        "changed_sid": sid_int,
        "original_sentence": str(orig),
        "modified_sentence": str(mod),
        "modified_note": str(note),
        "original_note": str(note),
        "expected": "CHANGED" if mode == "error_injection" else "SAME",
        "rule": f"v5_replay_{mode}",
        "error_type": mode,
        "old_verdict": str(get_first(record, FIELD_ALIASES["old_verdict"]) or "?"),
        "mode": mode,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--logs-dir", type=Path, default=DEFAULT_LOGS,
                    help="directory containing v5 game-log JSONL files")
    ap.add_argument("--logs-glob", default="game_*.jsonl",
                    help="glob pattern for log files within logs-dir")
    ap.add_argument("--style", choices=list(STYLE_DEFAULTS),
                    default="medrect_hint_v2")
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=0,
                    help="cap on number of probes (0 = all)")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--max-model-len", type=int, default=8192)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--field", action="append", default=[],
                    metavar="LOGICAL=ACTUAL",
                    help="override a field alias, e.g. "
                         "--field modified_sentence=inj_out_text")
    args = ap.parse_args()

    # Apply CLI field overrides (prepend so they take priority).
    for override in args.field:
        if "=" not in override:
            print(f"bad --field {override!r}, expected LOGICAL=ACTUAL")
            return 2
        logical, actual = override.split("=", 1)
        if logical not in FIELD_ALIASES:
            print(f"unknown logical field {logical!r}; "
                  f"known: {list(FIELD_ALIASES)}")
            return 2
        FIELD_ALIASES[logical] = [actual] + FIELD_ALIASES[logical]

    model = args.model or STYLE_DEFAULTS[args.style]
    out_path = args.out or (
        PROJECT_ROOT / "results" / "exp3" /
        f"{args.style}__{model_tag(model)}__v5replay.jsonl"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    log_files = sorted(args.logs_dir.glob(args.logs_glob))
    print(f"logs dir : {args.logs_dir}")
    print(f"glob     : {args.logs_glob}")
    print(f"files    : {len(log_files)}")
    if not log_files:
        print(f"\nNo files matched. Try a different --logs-dir / --logs-glob.")
        return 1

    # Show the first record's schema so the user can sanity-check the aliases
    first_record = None
    for lf in log_files:
        with open(lf, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        first_record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    break
        if first_record:
            break
    if first_record is None:
        print("no parseable records found in any log file")
        return 1
    print(f"\nFirst record has {len(first_record)} keys:")
    for k in sorted(first_record.keys()):
        v = first_record[k]
        sample = str(v).replace("\n", " ")[:70]
        print(f"  {k:28s}  {sample}")
    print("\nField resolution:")
    for logical in ["mode", "original_sentence", "modified_sentence",
                    "modified_note", "changed_sid", "old_verdict",
                    "injector_outcome", "sample_id"]:
        chosen = None
        for n in FIELD_ALIASES[logical]:
            if n in first_record:
                chosen = n
                break
        marker = chosen or "<MISSING>"
        print(f"  {logical:20s} -> {marker}")
    print()

    # Build probes
    probes: list[dict] = []
    counts = defaultdict(int)
    total = 0
    for log_file in log_files:
        with open(log_file, encoding="utf-8") as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                total += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    counts["bad_json"] += 1
                    continue
                mode = get_first(record, FIELD_ALIASES["mode"])
                if mode not in ("error_injection", "benign"):
                    counts["non_game_mode"] += 1
                    continue
                inj = get_first(record, FIELD_ALIASES["injector_outcome"])
                if inj is not None and inj != "exact_match":
                    counts["bad_injector_outcome"] += 1
                    continue
                p = extract_probe(record, total)
                if p is None:
                    counts["missing_fields"] += 1
                    continue
                probes.append(p)
                counts["extracted"] += 1

    print(f"Total records seen     : {total}")
    for k, v in counts.items():
        print(f"  {k:24s} : {v}")
    if args.n > 0:
        probes = probes[:args.n]
        print(f"  capped to              : {len(probes)}")
    if not probes:
        print("\nno probes extracted; the field aliases above failed to match. "
              "use --field LOGICAL=ACTUAL to override.")
        return 1

    by_mode = defaultdict(int)
    for p in probes:
        by_mode[p["mode"]] += 1
    print(f"\nBy mode: {dict(by_mode)}\n")

    print(f"Style    : {args.style}")
    print(f"Model    : {model}")
    print(f"Thinking : {args.thinking}\n")

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    llm = LLM(
        model=model,
        dtype="bfloat16",
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        trust_remote_code=True,
    )
    sampling = SamplingParams(
        temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, seed=0,
    )

    builder = PROMPT_BUILDERS[args.style]
    parser = PARSERS[args.style]

    prompts = [render(tokenizer, builder(p), args.thinking) for p in probes]
    print(f"Running inference on {len(prompts)} prompts...")
    outputs = llm.generate(prompts, sampling)

    confusion: dict[str, dict[str, dict[str, int]]] = {
        "error_injection": defaultdict(lambda: defaultdict(int)),
        "benign":          defaultdict(lambda: defaultdict(int)),
    }
    results = []
    for probe, out in zip(probes, outputs):
        raw = out.outputs[0].text.strip()
        new_verdict, reason = parser(raw, probe)
        old = probe["old_verdict"]
        mode = probe["mode"]
        confusion[mode][old][new_verdict] += 1
        results.append({
            "probe_id": probe["probe_id"],
            "mode": mode,
            "old_verdict": old,
            "new_verdict": new_verdict,
            "expected": probe["expected"],
            "agree_with_expected_new": new_verdict == probe["expected"],
            "agree_with_expected_old": old == probe["expected"],
            "changed_sid": probe["changed_sid"],
            "original_sentence": probe["original_sentence"][:300],
            "modified_sentence": probe["modified_sentence"][:300],
            "raw": raw[:300],
            "reason": reason,
        })

    with open(out_path, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # --- Confusion matrices ------------------------------------------------
    verdicts = ["SAME", "CHANGED", "ABSTAIN"]
    print("\n" + "=" * 72)
    print("Old judge (Qwen3-8B v5)  ->  New judge (MedRECT-32B + hint_v2)")
    print("=" * 72)
    for mode in ("error_injection", "benign"):
        expected = "CHANGED" if mode == "error_injection" else "SAME"
        n_mode = sum(sum(v.values()) for v in confusion[mode].values())
        print(f"\nMode = {mode}   (expected = {expected})   n = {n_mode}")
        print(f"  {'old\\new':<10s}  "
              + "  ".join(f"{v:>10s}" for v in verdicts)
              + f"  {'total':>6s}")
        for old in verdicts + ["?"]:
            row = confusion[mode].get(old, {})
            row_n = sum(row.values())
            if row_n == 0:
                continue
            cells = []
            for v in verdicts:
                c = row.get(v, 0)
                pct = c / row_n
                cells.append(f"{c:>4d} ({pct:>3.0%})")
            print(f"  {old:<10s}  " + "  ".join(cells) + f"  {row_n:>6d}")

    # --- Key cells --------------------------------------------------------
    err = confusion["error_injection"]
    ben = confusion["benign"]
    err_n = sum(sum(v.values()) for v in err.values()) or 1
    ben_n = sum(sum(v.values()) for v in ben.values()) or 1

    rescued = err["SAME"]["CHANGED"]
    regressed = err["CHANGED"]["SAME"]
    new_overflag = ben["SAME"]["CHANGED"]
    de_overflag = ben["CHANGED"]["SAME"]

    print("\n" + "=" * 72)
    print("Key cells (launch decision)")
    print("=" * 72)
    print(f"  RESCUED   old=SAME->new=CHANGED  on error_injection:  "
          f"{rescued:>4d}/{err_n} = {rescued/err_n:.1%}")
    print(f"            -> the 489-penalty cell from §5; want this large")
    print(f"  REGRESS   old=CHANGED->new=SAME  on error_injection:  "
          f"{regressed:>4d}/{err_n} = {regressed/err_n:.1%}")
    print(f"            -> should be ~0; non-trivial means new bias introduced")
    print(f"  NEW OVER  old=SAME->new=CHANGED  on benign:           "
          f"{new_overflag:>4d}/{ben_n} = {new_overflag/ben_n:.1%}")
    print(f"            -> mirror reward hack on real injector benigns")
    print(f"            -> Exp 2 Bucket C upper bound was 23 %; "
          f"want similar or lower")
    print(f"  FIXED OF  old=CHANGED->new=SAME  on benign:           "
          f"{de_overflag:>4d}/{ben_n} = {de_overflag/ben_n:.1%}")
    print(f"            -> net fix on the old judge's 12 % over-flag")

    print(f"\nResults JSONL: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
