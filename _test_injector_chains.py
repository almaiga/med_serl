#!/usr/bin/env python3
"""Quick validation test for generate_injector_chains.py"""
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))

from scripts.medrect.generate_injector_chains import (
    normalize_error_record, normalize_benign_record,
    build_error_prompt, build_benign_prompt,
    ReasoningCleaner, _parse_compact_output, validate_error_chain,
    validate_benign_chain, format_medrect_record, _build_clean_user_prompt,
    reconstruct_note,
)

# Load config
with open('configs/prompts/sft/injector_reasoning_prompts.json') as f:
    config = json.load(f)

errors = []

# --- Test error normalization ---
print("=== ERROR NORMALIZATION ===")
with open('data_processed/medec_paired/train_val_split/sft_train.jsonl') as f:
    pair = json.loads(f.readline())

rec = normalize_error_record(pair)
assert rec is not None, "normalize_error_record returned None"
assert rec.scenario == "error"
assert rec.changed_sid is not None
assert rec.error_type is not None
print(f"  note_id={rec.note_id}, scenario={rec.scenario}, changed_sid={rec.changed_sid}, error_type={rec.error_type}")
print(f"  target_mod={rec.target_sentence_modified[:80]}...")
print("  OK")

# --- Test benign normalization ---
print("\n=== BENIGN NORMALIZATION ===")
with open('data_processed/benign_changes/benign_train_clean.jsonl') as f:
    brec_raw = json.loads(f.readline())

brec = normalize_benign_record(brec_raw)
assert brec is not None, "normalize_benign_record returned None"
assert brec.scenario == "benign"
assert brec.changed_sid is not None
print(f"  note_id={brec.note_id}, scenario={brec.scenario}, changed_sid={brec.changed_sid}, change_type={brec.change_type}")
print("  OK")

# --- Test prompt building ---
print("\n=== PROMPT BUILDING ===")
sys_p, usr_p = build_error_prompt(rec, config)
assert "error injection" in usr_p.lower() or "error" in usr_p.lower()
assert "N. <modified sentence>" in usr_p
print(f"  Error prompt: sys={len(sys_p)} chars, usr={len(usr_p)} chars")

bsys_p, busr_p = build_benign_prompt(brec, config)
assert "benign" in busr_p.lower()
assert "N. <modified sentence>" in busr_p
print(f"  Benign prompt: sys={len(bsys_p)} chars, usr={len(busr_p)} chars")
print("  OK")

# --- Test ReasoningCleaner ---
print("\n=== REASONING CLEANER ===")
cleaner = ReasoningCleaner(config)
test_text = (
    "First I examine the symptoms. "
    "We are told that sentence 5 is the target. "
    "The patient shows tachycardia. "
    "This reference information should not be mentioned. "
    "Blood pressure is 90/60."
)
cleaned, removed = cleaner.clean(test_text)
assert removed >= 2, f"Expected at least 2 removals, got {removed}"
assert "are told" not in cleaned.lower()
assert "reference information" not in cleaned.lower()
assert "tachycardia" in cleaned
print(f"  Removed {removed} meta-ref sentences")
print(f"  Cleaned: {cleaned[:200]}")
print("  OK")

# --- Test parse output ---
print("\n=== PARSE OUTPUT ===")
sid, txt = _parse_compact_output("12. Suspected of Creutzfeldt-Jakob disease.")
assert sid == 12
assert "Creutzfeldt" in txt
print(f"  Parsed: sid={sid}, text={txt[:60]}")

# Test parse with backticks
sid2, txt2 = _parse_compact_output("`5. Changed sentence here.`")
assert sid2 == 5
print(f"  Parsed with backticks: sid={sid2}")
print("  OK")

# --- Test validation ---
print("\n=== VALIDATION ===")
ok, p_sid, p_txt, reason = validate_error_chain(rec, f"{rec.changed_sid}. Some modified text.")
assert ok, f"Expected valid, got: {reason}"
print(f"  Valid error chain: sid={p_sid}, reason={reason}")

bad_ok, _, _, bad_reason = validate_error_chain(rec, "999. Wrong sentence.")
assert not bad_ok
print(f"  Invalid error chain caught: {bad_reason}")
print("  OK")

# --- Test reconstruct_note ---
print("\n=== RECONSTRUCT NOTE ===")
from scripts.self_play.utils import number_sentences
orig_sentences = number_sentences(rec.input_note)
reconstructed = reconstruct_note(orig_sentences, rec.changed_sid, rec.target_sentence_modified)
assert len(reconstructed) > 0
print(f"  Reconstructed note: {len(reconstructed)} chars")
print("  OK")

# --- Test clean user prompt ---
print("\n=== CLEAN USER PROMPT ===")
clean_p = _build_clean_user_prompt(rec, config)
assert "cheat" not in clean_p.lower()
assert "reference information" not in clean_p.lower()
assert "N. <modified sentence>" in clean_p
print(f"  Clean error prompt: {len(clean_p)} chars (no cheat info)")

clean_bp = _build_clean_user_prompt(brec, config)
assert "cheat" not in clean_bp.lower()
assert "reference information" not in clean_bp.lower()
print(f"  Clean benign prompt: {len(clean_bp)} chars (no cheat info)")
print("  OK")

# --- Test medrect output format ---
print("\n=== OUTPUT FORMAT ===")
out = format_medrect_record(
    rec, sys_p, clean_p,
    "test reasoning...",
    rec.changed_sid, rec.target_sentence_modified,
    reconstructed
)
required_keys = {"sample_id", "system_prompt", "user_prompt", "label", "reasoning",
                 "language", "error_type", "error_flag", "error_sentence_id",
                 "role", "scenario", "reconstructed_note"}
missing = required_keys - set(out.keys())
assert not missing, f"Missing keys: {missing}"
assert out["role"] == "injector"
assert out["scenario"] == "error"
assert out["error_flag"] == 1
print(f"  sample_id: {out['sample_id']}")
print(f"  label: {out['label'][:80]}...")
print(f"  role={out['role']}, scenario={out['scenario']}, error_flag={out['error_flag']}")
print(f"  Keys: {sorted(out.keys())}")
print("  OK")

# --- Test batch loading ---
print("\n=== BATCH LOADING ===")
from scripts.medrect.generate_injector_chains import _load_and_normalize_error, _load_and_normalize_benign
from pathlib import Path

error_recs = _load_and_normalize_error(
    Path('data_processed/medec_paired/train_val_split/sft_train.jsonl'), limit=5
)
assert len(error_recs) == 5, f"Expected 5, got {len(error_recs)}"
print(f"  Loaded {len(error_recs)} error records")

benign_recs = _load_and_normalize_benign(
    Path('data_processed/benign_changes/benign_train_clean.jsonl'), limit=5
)
assert len(benign_recs) == 5, f"Expected 5, got {len(benign_recs)}"
print(f"  Loaded {len(benign_recs)} benign records")
print("  OK")

print("\n" + "="*50)
print("ALL TESTS PASSED")
print("="*50)
