#!/usr/bin/env python3
"""Diagnose injector parse_failures: did the injector finish thinking or run out?

For each parse_failure, shows total length, whether <think>/</think> are present,
and the LAST chars — so we can tell if the model (a) ran out of tokens mid-think
(no </think>) or (b) finished thinking but produced the edit in a format the
parser missed (</think> present + text after).
"""
import glob
import json
import sys


def main() -> None:
    log = sys.argv[1] if len(sys.argv) > 1 else \
        sorted(glob.glob("results/self_play/interactions/game_*.jsonl"))[-1]
    rows = [json.loads(l) for l in open(log) if l.strip()]
    fails = [r for r in rows if r.get("injector_outcome") == "parse_failure"]
    print(f"{len(fails)} injector parse_failures in {log}\n")

    n_open = n_close = 0
    for r in fails:
        out = (r.get("injector_output") or r.get("injector_raw")
               or r.get("raw_injector") or "")
        has_open = "<think>" in out
        has_close = "</think>" in out
        n_open += has_open
        n_close += has_close

    print(f"of {len(fails)} failures:")
    print(f"  have <think>  : {n_open}")
    print(f"  have </think> : {n_close}   "
          f"(if << n_open, they ran OUT OF TOKENS mid-thinking)")
    print()

    for r in fails[:5]:
        out = (r.get("injector_output") or r.get("injector_raw")
               or r.get("raw_injector") or "")
        print("=" * 72)
        print(f"note: {r.get('note_id')}")
        print(f"len_chars: {len(out)}   "
              f"<think>: {'<think>' in out}   </think>: {'</think>' in out}")
        print("--- LAST 450 chars ---")
        print(out[-450:] if out else "(empty)")
        print()


if __name__ == "__main__":
    main()
