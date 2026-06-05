#!/usr/bin/env python3
"""Build an adversarial probe set for Exp 2 of the judge-bottleneck study.

Five buckets, with deterministic ground truth (no model in the loop for labeling):

  A. pharm_analog       drug X -> same-class analog Y      -> CHANGED
  B. dose_route         dose x10 / PO<->IV / freq swap     -> CHANGED
  C. synonym            equivalent abbreviation / exact temporal conversion / generic<->brand
                                                            -> SAME
  D. surface_opposite   left<->right, with<->without,
                        positive<->negative, etc.           -> CHANGED
  E. nonsense           truncated sentence (<40% length)    -> CHANGED (must be filtered before reward)

Source notes: error_flag=0 records in data_processed/judge_bench/medec_test.json
(genuinely correct held-out ms-test notes, unseen by MedRECT-32B). Each probe
modifies exactly ONE sentence per the bucket's rule and records:

  - the modified numbered note (what a detection judge sees),
  - the original and modified sentence text (what a pair judge sees),
  - changed_sid, bucket, rule, expected verdict.

This script is purely deterministic and CPU-only.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.self_play.utils import parse_numbered_sentences, reconstruct_note  # noqa: E402

DEFAULT_DATA = PROJECT_ROOT / "data_processed" / "judge_bench" / "medec_test.json"
DEFAULT_OUT = PROJECT_ROOT / "data_processed" / "judge_bench" / "exp2_probes.jsonl"


# ─────────────────────────────────────────────────────────────────────────────
# Rules
# Drugs / synonyms chosen so that the swap is unambiguous from a clinical
# standpoint: bucket A swaps change at least one of indication, mechanism,
# adverse-effect profile or monitoring; bucket C swaps preserve all of those.
# ─────────────────────────────────────────────────────────────────────────────

# (X, Y): same drug class but clinically distinct.  Swap in either direction.
PHARM_ANALOG_PAIRS: list[tuple[str, str]] = [
    ("metoprolol", "atenolol"),            # cardio-selectivity differs in COPD context
    ("losartan", "lisinopril"),            # ARB vs ACEi -> angioedema/cough risk
    ("omeprazole", "pantoprazole"),        # PPI metabolism (CYP2C19) differs
    ("atorvastatin", "rosuvastatin"),      # potency / interactions differ
    ("ceftriaxone", "ceftazidime"),        # antipseudomonal coverage differs
    ("azithromycin", "erythromycin"),      # QT / motility differ
    ("warfarin", "apixaban"),              # monitoring + reversal differ
    ("prednisone", "hydrocortisone"),      # potency (4×) differs
    ("amoxicillin", "ampicillin"),         # oral bioavailability differs
    ("furosemide", "hydrochlorothiazide"), # loop vs thiazide
    ("morphine", "fentanyl"),              # potency / route options differ
    ("regular insulin", "insulin glargine"),  # onset / duration differ
    ("haloperidol", "olanzapine"),         # typical vs atypical
    ("ibuprofen", "naproxen"),             # half-life differs
    ("metformin", "glipizide"),            # different class, hypoglycemia risk
]

# Synonyms that preserve clinical meaning.  Swap in either direction.
SYNONYM_PAIRS: list[tuple[str, str]] = [
    ("type 2 diabetes mellitus", "T2DM"),
    ("type 1 diabetes mellitus", "T1DM"),
    ("myocardial infarction", "MI"),
    ("congestive heart failure", "CHF"),
    ("chronic obstructive pulmonary disease", "COPD"),
    ("deep vein thrombosis", "DVT"),
    ("pulmonary embolism", "PE"),
    ("atrial fibrillation", "AFib"),
    ("acute kidney injury", "AKI"),
    # generic / brand
    ("metformin", "Glucophage"),
    ("atorvastatin", "Lipitor"),
    ("omeprazole", "Prilosec"),
    ("lisinopril", "Zestril"),
    # exact temporal conversions
    ("1 week", "7 days"),
    ("2 weeks", "14 days"),
    ("1 year", "12 months"),
    ("1 day", "24 hours"),
    ("5 days", "120 hours"),
]

# Surface-similar, clinically opposite tokens.
# Each entry is (regex, replacement); we always swap to the partner.
OPPOSITE_SWAPS: list[tuple[str, str]] = [
    (r"\bleft\b", "right"),
    (r"\bright\b", "left"),
    (r"\bwarm\b", "hot"),
    (r"\bhot\b", "warm"),
    (r"\bincreased\b", "decreased"),
    (r"\bdecreased\b", "increased"),
    (r"\belevated\b", "decreased"),
    (r"\bpositive\b", "negative"),
    (r"\bnegative\b", "positive"),
    (r"\bwith\b", "without"),
    (r"\bwithout\b", "with"),
    (r"\babove\b", "below"),
    (r"\bbelow\b", "above"),
    (r"\bnormal\b", "abnormal"),
    (r"\bafebrile\b", "febrile"),
]

# Dose / route / frequency perturbations.  We apply at most one perturbation
# per sentence so the ground truth stays unambiguous.
DOSE_RE = re.compile(r"\b(\d+(?:\.\d+)?)\s*(mg|mcg|g)\b", re.IGNORECASE)
ROUTE_SWAPS: list[tuple[str, str]] = [
    (r"\bPO\b", "IV"),
    (r"\bIV\b", "PO"),
    (r"\boral\b", "intravenous"),
    (r"\bintravenous\b", "oral"),
]
FREQ_SWAPS: list[tuple[str, str]] = [
    (r"\bonce daily\b", "every 6 hours"),
    (r"\btwice daily\b", "every 6 hours"),
    (r"\bdaily\b", "every 6 hours"),
    (r"\bq\s*24\s*h\b", "q6h"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Probe construction
# ─────────────────────────────────────────────────────────────────────────────

def make_probe(
    note: dict,
    sid: int,
    original_sentence: str,
    modified_sentence: str,
    bucket: str,
    bucket_name: str,
    expected: str,
    rule: str,
) -> dict:
    """Assemble one probe record."""
    modified_note = reconstruct_note(note["sentences"], sid, modified_sentence)
    return {
        "probe_id": f"{note['sample_id']}::{bucket}::{rule[:40]}",
        "bucket": bucket,
        "bucket_name": bucket_name,
        "note_id": note["sample_id"],
        "changed_sid": sid,
        "original_sentence": original_sentence,
        "modified_sentence": modified_sentence,
        "modified_note": modified_note,
        "original_note": note["sentences"],
        "expected": expected,
        "rule": rule,
    }


def _first_match_swap(
    text: str,
    pairs: Iterable[tuple[str, str]],
    bidirectional: bool,
) -> tuple[str, str] | None:
    """Apply the first matching pair, count=1.  Returns (new_text, rule) or None."""
    for a, b in pairs:
        candidates = [(a, b), (b, a)] if bidirectional else [(a, b)]
        for src, dst in candidates:
            # We treat `src` as either a raw token (string) or a regex (str).
            if any(ch in src for ch in r".\^$*+?{}[]|()"):
                pat = src
            else:
                pat = rf"\b{re.escape(src)}\b"
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                new_text = re.sub(pat, dst, text, count=1, flags=re.IGNORECASE)
                if new_text != text:
                    return new_text, f"{src} -> {dst}"
    return None


def gen_pharm_analog(notes: list[dict], target_n: int, rng: random.Random) -> list[dict]:
    probes = []
    for note in notes:
        if len(probes) >= target_n:
            break
        sents = parse_numbered_sentences(note["sentences"])
        for sid in sorted(sents):
            text = sents[sid]
            swap = _first_match_swap(text, PHARM_ANALOG_PAIRS, bidirectional=True)
            if swap is not None:
                new_text, rule = swap
                probes.append(make_probe(note, sid, text, new_text, "A", "pharm_analog", "CHANGED", rule))
                break  # one per note
    rng.shuffle(probes)
    return probes[:target_n]


def gen_dose_route(notes: list[dict], target_n: int, rng: random.Random) -> list[dict]:
    probes = []
    for note in notes:
        if len(probes) >= target_n:
            break
        sents = parse_numbered_sentences(note["sentences"])
        made = False
        for sid in sorted(sents):
            if made:
                break
            text = sents[sid]
            # Try dose x10 first.
            m = DOSE_RE.search(text)
            if m:
                old_val = float(m.group(1))
                new_val = old_val * 10
                new_str = f"{int(new_val) if new_val.is_integer() else new_val} {m.group(2)}"
                new_text = text[:m.start()] + new_str + text[m.end():]
                if new_text != text:
                    probes.append(make_probe(
                        note, sid, text, new_text, "B", "dose_route",
                        "CHANGED", f"dose x10: {m.group(0)} -> {new_str}"
                    ))
                    made = True
                    continue
            # Then route.
            swap = _first_match_swap(text, ROUTE_SWAPS, bidirectional=False)
            if swap is not None:
                new_text, rule = swap
                probes.append(make_probe(note, sid, text, new_text, "B", "dose_route", "CHANGED", rule))
                made = True
                continue
            # Then frequency.
            swap = _first_match_swap(text, FREQ_SWAPS, bidirectional=False)
            if swap is not None:
                new_text, rule = swap
                probes.append(make_probe(note, sid, text, new_text, "B", "dose_route", "CHANGED", rule))
                made = True
    rng.shuffle(probes)
    return probes[:target_n]


def gen_synonyms(notes: list[dict], target_n: int, rng: random.Random) -> list[dict]:
    probes = []
    for note in notes:
        if len(probes) >= target_n:
            break
        sents = parse_numbered_sentences(note["sentences"])
        for sid in sorted(sents):
            text = sents[sid]
            swap = _first_match_swap(text, SYNONYM_PAIRS, bidirectional=True)
            if swap is not None:
                new_text, rule = swap
                probes.append(make_probe(note, sid, text, new_text, "C", "synonym", "SAME", rule))
                break
    rng.shuffle(probes)
    return probes[:target_n]


def gen_surface_opposite(notes: list[dict], target_n: int, rng: random.Random) -> list[dict]:
    probes = []
    for note in notes:
        if len(probes) >= target_n:
            break
        sents = parse_numbered_sentences(note["sentences"])
        for sid in sorted(sents):
            text = sents[sid]
            swap = _first_match_swap(text, OPPOSITE_SWAPS, bidirectional=False)
            if swap is not None:
                new_text, rule = swap
                probes.append(make_probe(note, sid, text, new_text, "D", "surface_opposite", "CHANGED", rule))
                break
    rng.shuffle(probes)
    return probes[:target_n]


def gen_nonsense(notes: list[dict], target_n: int, rng: random.Random) -> list[dict]:
    probes = []
    candidates = list(notes)
    rng.shuffle(candidates)
    for note in candidates:
        if len(probes) >= target_n:
            break
        sents = parse_numbered_sentences(note["sentences"])
        sids = [s for s in sorted(sents) if len(sents[s].split()) >= 8]
        if not sids:
            continue
        sid = rng.choice(sids)
        text = sents[sid]
        words = text.split()
        keep = max(1, int(len(words) * 0.25))
        new_text = " ".join(words[:keep])
        if new_text != text:
            probes.append(make_probe(
                note, sid, text, new_text, "E", "nonsense",
                "CHANGED", f"truncated to {keep}/{len(words)} words"
            ))
    return probes[:target_n]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

GENERATORS = {
    "A": ("pharm_analog", gen_pharm_analog),
    "B": ("dose_route", gen_dose_route),
    "C": ("synonym", gen_synonyms),
    "D": ("surface_opposite", gen_surface_opposite),
    "E": ("nonsense", gen_nonsense),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--per-bucket", type=int, default=30,
                    help="target number of probes per bucket")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = json.load(open(args.data, encoding="utf-8"))
    correct_notes = [r for r in rows if r.get("error_flag") == 0]
    print(f"loaded {len(rows)} test rows, {len(correct_notes)} correct (error_flag=0)")

    rng = random.Random(args.seed)
    correct_notes = sorted(correct_notes, key=lambda r: r["sample_id"])
    shuffled = correct_notes.copy()
    rng.shuffle(shuffled)

    all_probes: list[dict] = []
    counts: dict[str, int] = {}
    for bucket, (name, fn) in GENERATORS.items():
        # Use a per-bucket RNG so adding a bucket doesn't reshuffle others.
        bucket_rng = random.Random(args.seed + ord(bucket))
        notes_for_bucket = shuffled.copy()
        bucket_rng.shuffle(notes_for_bucket)
        probes = fn(notes_for_bucket, args.per_bucket, bucket_rng)
        counts[bucket] = len(probes)
        all_probes.extend(probes)
        if probes:
            sample = probes[0]
            print(f"  bucket {bucket} ({name}): {len(probes)} probes  "
                  f"e.g. rule={sample['rule']!r} expected={sample['expected']}")
        else:
            print(f"  bucket {bucket} ({name}): 0 probes")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for p in all_probes:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"\nwrote {len(all_probes)} probes -> {args.out}")
    print("bucket counts:", counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
