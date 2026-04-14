#!/usr/bin/env python3
"""
Medical Error Detection + Localization Inference

Task  : given numbered clinical sentences, output CORRECT or the error sentence number.
Reuses: load_model_and_tokenizer, detect_model_type, load_test_data, Qwen3 token
        constants from scripts/inference_error_detection.py.

Metrics
-------
- Detection  : precision / recall / F1  (error vs no-error)
- Localization: exact sentence-number match among detected errors

Usage
-----
    python scripts/medrect/inference_detection.py \\
        --model_path FreedomIntelligence/HuatuoGPT-o1-7B \\
        --dataset all --batch_size 8 --temperature 0.7 \\
        --thinking_budget 1024 --max_new_tokens 1536
"""

import json
import os
import re
import sys
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import torch
from tqdm import tqdm

# ── resolve project root and import shared helpers ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from inference_error_detection import (
    detect_model_type,
    load_model_and_tokenizer,
    load_test_data,
)
from self_play.utils import parse_assessor_answer

DEFAULT_PROMPT_CONFIG = PROJECT_ROOT / "configs" / "prompts" / "detection_localization_prompts.json"

# ── Constants ────────────────────────────────────────────────────────────────
MODEL_TYPE_QWEN = "qwen"
IM_START_TOKEN = "<|im_start|>"
IM_END_TOKEN = "₃"
THINK_TOKEN = "<|think|>"
THINK_END_TOKEN = "<|end_think|>"

# Qwen3 special token IDs
IM_START_TOKEN_ID = 151644
IM_END_TOKEN_ID = 151645
THINK_TOKEN_ID = 151646
THINK_END_TOKEN_ID = 151668


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_prompt_config(path: Path = DEFAULT_PROMPT_CONFIG) -> Dict:
    with open(path) as f:
        return json.load(f)


def sentences_to_1indexed(raw: str) -> str:
    """Convert MEDEC 0-indexed sentences to 1-indexed.

    "0 First sentence.\\n1 Second." → "1. First sentence.\\n2. Second."
    """
    lines = []
    for line in raw.strip().split("\n"):
        m = re.match(r"^(\d+)\s+(.+)$", line.strip())
        lines.append(f"{int(m.group(1)) + 1}. {m.group(2)}" if m else line)
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Output parsing
# ─────────────────────────────────────────────────────────────────────────────

def split_thinking(content: str) -> Tuple[str, str]:
    """Split model thinking from the final answer.

    Handles explicit think tags emitted in text form.
    Returns (thinking, answer_after_think).
    """
    m = re.search(r"<think>(.*?)</think>\s*", content, re.DOTALL | re.IGNORECASE)
    if m:
        return m.group(1).strip(), content[m.end():].strip()
    return "", content


def parse_output(content: str) -> Tuple[str, Optional[int]]:
    """Return (label, sentence_id).

    label        : "CORRECT" | "ERROR" | "UNKNOWN"
    sentence_id  : int if label is ERROR, else None
    """
    return parse_assessor_answer(content)


def parse_output_candidates(*candidates: str) -> Tuple[str, Optional[int]]:
    """Try parsing multiple candidate texts, from most to least trusted."""
    for text in candidates:
        if not text or not text.strip():
            continue
        label, sid = parse_output(text)
        if label != "UNKNOWN":
            return label, sid

        # Last-line fallback: useful when the answer was serialized after the
        # thinking block but the main split missed it.
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            last = lines[-1]
            if re.fullmatch(r"\d+", last):
                return "ERROR", int(last)
            if re.fullmatch(r"CORRECT", last, flags=re.IGNORECASE):
                return "CORRECT", None

    return "UNKNOWN", None


# ─────────────────────────────────────────────────────────────────────────────
# Inference
# ─────────────────────────────────────────────────────────────────────────────

def run_inference(
    model,
    tokenizer,
    test_df: pd.DataFrame,
    model_type: str,
    prompt_config: Dict,
    *,
    use_thinking: bool = True,
    max_samples: Optional[int] = None,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
    thinking_budget: int = 1024,
    batch_size: int = 8,
) -> List[Dict]:
    if max_samples:
        test_df = test_df.head(max_samples)

    model.eval()
    is_qwen = model_type == MODEL_TYPE_QWEN
    tokenizer.padding_side = "left"

    system_prompt = prompt_config["system_prompt"]
    user_template = prompt_config["user_template"]
    results: List[Dict] = []

    n = len(test_df)
    for batch_start in tqdm(range(0, n, batch_size), desc="Inference"):
        batch = test_df.iloc[batch_start : batch_start + batch_size]
        prompts, meta = [], []

        for _, row in batch.iterrows():
            error_flag = int(row["Error Flag"])
            sentences = sentences_to_1indexed(str(row.get("Sentences", row.get("Text", ""))))

            raw_sid = row.get("Error Sentence ID")
            if error_flag == 1 and pd.notna(raw_sid):
                gt_sid = int(raw_sid) + 1  # 0-indexed → 1-indexed
                gt_label = str(gt_sid)
            else:
                gt_sid, gt_label = None, "CORRECT"

            meta.append(
                dict(
                    text_id=str(row.get("Text ID", "")),
                    dataset=str(row.get("dataset", "")),
                    error_flag=error_flag,
                    gt_label=gt_label,
                    gt_sid=gt_sid,
                    error_type=str(row.get("Error Type", "")) if pd.notna(row.get("Error Type")) else "",
                    sentences=sentences,
                )
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_template.format(sentences=sentences)},
            ]
            prompt_kwargs = dict(tokenize=False, add_generation_prompt=True)
            if is_qwen:
                prompt_kwargs["enable_thinking"] = bool(use_thinking)
            prompts.append(tokenizer.apply_chat_template(messages, **prompt_kwargs))

        inputs = tokenizer(
            prompts, return_tensors="pt", padding=True, truncation=True, max_length=2048
        ).to(model.device)
        input_lens = inputs.attention_mask.sum(dim=1).tolist()

        gen_kwargs = dict(
            max_new_tokens=thinking_budget if (is_qwen and use_thinking) else max_new_tokens,
            temperature=temperature if temperature > 0 else None,
            do_sample=temperature > 0,
            top_p=0.95 if temperature > 0 else None,
            top_k=20 if (is_qwen and use_thinking) else None,
            min_p=0.05 if (is_qwen and use_thinking) else None,
            repetition_penalty=1.1 if (is_qwen and use_thinking) else 1.05,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        with torch.no_grad():
            outputs = model.generate(**inputs, **gen_kwargs)

        # Save original padded input length for position-based slicing
        orig_padded_len = inputs.input_ids.shape[1]

        # ── Two-stage generation for Qwen thinking mode ──────────────────────
        needs_stage2 = [False] * len(prompts)
        stage2_input_len = 0
        input_ids2 = None
        batch_final = []
        if is_qwen and use_thinking:
            for i in range(len(prompts)):
                # NOTE: can't use  != pad_token_id filtering here because
                # pad_token_id == IM_END_TOKEN_ID == 151645, and the prompt
                # contains  consolidated tokens.  Use position-based slicing.
                out_ids = outputs[i, orig_padded_len:].tolist()
                # Right-strip pad/eos tokens
                while out_ids and out_ids[-1] == tokenizer.pad_token_id:
                    out_ids.pop()

                if IM_END_TOKEN_ID not in out_ids:
                    needs_stage2[i] = True
                    if THINK_END_TOKEN_ID not in out_ids:
                        early_stop = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now.\n</think>\n\n"
                        early_ids = tokenizer(early_stop, return_tensors="pt", add_special_tokens=False).input_ids.to(model.device)
                        new_input = torch.cat([outputs[i:i+1], early_ids], dim=-1)
                    else:
                        new_input = outputs[i:i+1]
                    batch_final.append(new_input)
                else:
                    batch_final.append(outputs[i:i+1])

            # Pad and run second-stage generation
            max_len = max(x.size(-1) for x in batch_final)
            padded, masks = [], []
            for x in batch_final:
                pad_len = max_len - x.size(-1)
                if pad_len > 0:
                    pad = torch.full((1, pad_len), tokenizer.pad_token_id, dtype=x.dtype, device=x.device)
                    padded.append(torch.cat([pad, x], dim=-1))
                    masks.append(torch.cat([torch.zeros(1, pad_len, dtype=torch.long, device=x.device),
                                            torch.ones(1, x.size(-1), dtype=torch.long, device=x.device)], dim=-1))
                else:
                    padded.append(x)
                    masks.append(torch.ones_like(x, dtype=torch.long))

            input_ids2 = torch.cat(padded, dim=0)
            attn_mask2 = torch.cat(masks, dim=0)
            stage2_input_len = input_ids2.size(1)
            # Answer stage: only run if at least one sample needs it
            if any(needs_stage2):
                with torch.no_grad():
                    outputs = model.generate(
                        input_ids=input_ids2,
                        attention_mask=attn_mask2,
                        max_new_tokens=16,
                        temperature=temperature if temperature > 0 else None,
                        do_sample=temperature > 0,
                        top_p=0.95 if temperature > 0 else None,
                        top_k=20 if temperature > 0 else None,
                        min_p=0.05 if temperature > 0 else None,
                        repetition_penalty=1.3,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                    )
            else:
                # All samples completed in stage 1 — reuse batch_final as outputs
                outputs = input_ids2

        for i, m in enumerate(meta):
            thinking = ""
            content = ""
            full_text = ""

            if is_qwen and use_thinking and batch_final:
                # ── Extract thinking from batch_final (pre-stage2 data) ──
                bf = batch_final[i].squeeze(0)
                thinking_all = bf[orig_padded_len:].tolist()
                # Right-strip pad/eos (safe — they sit after the real content)
                while thinking_all and thinking_all[-1] == tokenizer.pad_token_id:
                    thinking_all.pop()
                full_text = tokenizer.decode(thinking_all, skip_special_tokens=False).strip()

                # Split at Qwen3 </think> token.
                if THINK_END_TOKEN_ID in thinking_all:
                    idx = thinking_all.index(THINK_END_TOKEN_ID)
                    thinking = tokenizer.decode(thinking_all[:idx], skip_special_tokens=True).strip()
                    after_think = thinking_all[idx + 1 :]
                else:
                    thinking = tokenizer.decode(thinking_all, skip_special_tokens=True).strip()
                    after_think = []

                if needs_stage2[i]:
                    # Answer = only the NEW tokens from stage 2
                    answer_ids = outputs[i, stage2_input_len:].tolist()
                    while answer_ids and answer_ids[-1] == tokenizer.pad_token_id:
                        answer_ids.pop()
                    content = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()
                    if answer_ids:
                        answer_raw = tokenizer.decode(answer_ids, skip_special_tokens=False).strip()
                        full_text = f"{full_text}\n{answer_raw}".strip()
                else:
                    # Answer was already in stage 1, after </think>.
                    content = tokenizer.decode(after_think, skip_special_tokens=True).strip()

                # Clean up injected early-stop message from thinking
                thinking = re.sub(
                    r'\n*Considering the limited time by the user.*$', '',
                    thinking, flags=re.DOTALL
                ).strip()

            else:
                # Non-Qwen or no-thinking: single-stage generation
                out_ids = outputs[i, orig_padded_len:].tolist()
                while out_ids and out_ids[-1] == tokenizer.pad_token_id:
                    out_ids.pop()
                raw = tokenizer.decode(out_ids, skip_special_tokens=False).strip()
                full_text = raw
                thinking, content = split_thinking(raw)
                if not thinking:
                    content = tokenizer.decode(out_ids, skip_special_tokens=True).strip()

            pred_type, pred_sid = parse_output_candidates(content, full_text, thinking)
            pred_label = "CORRECT" if pred_type == "CORRECT" else (str(pred_sid) if pred_sid is not None else "UNKNOWN")

            detection_correct = (m["gt_label"] == "CORRECT" and pred_label == "CORRECT") or (
                m["gt_label"] != "CORRECT" and pred_label not in ("CORRECT", "UNKNOWN")
            )
            localization_correct = (
                m["gt_sid"] is not None and pred_sid is not None and m["gt_sid"] == pred_sid
            )

            results.append(
                dict(
                    text_id=m["text_id"],
                    dataset=m["dataset"],
                    note=m["sentences"],
                    error_type=m["error_type"],
                    gt_label=m["gt_label"],
                    gt_sid=m["gt_sid"],
                    pred_label=pred_label,
                    pred_sid=pred_sid,
                    detection_correct=detection_correct,
                    localization_correct=localization_correct,
                    thinking=thinking,
                    raw_output=content,
                )
            )

            if batch_start == 0 and i == 0:
                print(f"\n{'='*50}\nDEBUG first sample\nGT={m['gt_label']}  Pred={pred_label}\n--- THINKING ---\n{thinking}\n--- ANSWER ---\n{content}\n{'='*50}\n")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_metrics(results: List[Dict]) -> Dict:
    total = len(results)
    error_cases   = [r for r in results if r["gt_label"] != "CORRECT"]
    correct_cases = [r for r in results if r["gt_label"] == "CORRECT"]

    tp = sum(1 for r in error_cases   if r["pred_label"] not in ("CORRECT", "UNKNOWN"))
    fn = sum(1 for r in error_cases   if r["pred_label"] in  ("CORRECT", "UNKNOWN"))
    tn = sum(1 for r in correct_cases if r["pred_label"] == "CORRECT")
    fp = sum(1 for r in correct_cases if r["pred_label"] != "CORRECT")

    prec = tp / (tp + fp) if (tp + fp) else 0
    rec  = tp / (tp + fn) if (tp + fn) else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) else 0

    detected = [r for r in error_cases if r["pred_label"] not in ("CORRECT", "UNKNOWN")]
    sent_ok  = sum(1 for r in error_cases if r["localization_correct"])

    by_type: Dict = {}
    for r in error_cases:
        et = r["error_type"] or "unknown"
        s  = by_type.setdefault(et, {"total": 0, "detected": 0, "sentence_correct": 0})
        s["total"] += 1
        if r["pred_label"] not in ("CORRECT", "UNKNOWN"):
            s["detected"] += 1
        if r["localization_correct"]:
            s["sentence_correct"] += 1

    det_acc = sum(1 for r in results if r["detection_correct"]) / total if total else 0

    return dict(
        total_samples=total,
        detection=dict(
            accuracy=det_acc, precision=prec, recall=rec, f1=f1,
            tp=tp, fp=fp, tn=tn, fn=fn,
        ),
        sentence_extraction=dict(
            total_errors=len(error_cases),
            detected_errors=len(detected),
            exact_matches=sent_ok,
            accuracy=sent_ok / len(error_cases) if error_cases else 0,
        ),
        by_error_type=by_type,
    )


def print_metrics(m: Dict) -> None:
    det, sent = m["detection"], m["sentence_extraction"]
    print(f"\n{'='*50}")
    print("Detection (error vs no-error)")
    print(f"  Accuracy : {det['accuracy']:.3f}")
    print(f"  Precision: {det['precision']:.3f}  Recall: {det['recall']:.3f}  F1: {det['f1']:.3f}")
    print(f"  TP={det['tp']} FP={det['fp']} TN={det['tn']} FN={det['fn']}")
    print(f"\nSentence extraction (exact sentence match)")
    print(f"  Gold error cases : {sent['total_errors']}")
    print(f"  Predicted errors  : {sent['detected_errors']}")
    print(f"  Exact matches     : {sent['exact_matches']}")
    print(f"  Accuracy          : {sent['accuracy']:.3f}")
    print(f"\nBy error type")
    for et, s in sorted(m["by_error_type"].items(), key=lambda x: -x[1]["total"]):
        dr = s["detected"] / s["total"] if s["total"] else 0
        sr = s["sentence_correct"] / s["total"] if s["total"] else 0
        print(f"  {et}: n={s['total']}  det_recall={dr:.2f}  sent_acc={sr:.2f}")
    print(f"{'='*50}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Medical Error Detection + Localization")
    p.add_argument("--model_path",      required=True)
    p.add_argument("--prompt_config",   default=str(DEFAULT_PROMPT_CONFIG))
    p.add_argument("--dataset",         default="all", choices=["ms", "uw", "all"])
    p.add_argument("--max_samples",     type=int, default=None)
    p.add_argument("--batch_size",      type=int, default=8)
    p.add_argument("--temperature",     type=float, default=0.7)
    p.add_argument("--max_new_tokens",  type=int, default=512)
    p.add_argument("--thinking_budget", type=int, default=1024)
    p.add_argument("--no_thinking",     action="store_true", help="Disable thinking mode")
    p.add_argument("--output_dir",      default="results/detection")
    p.add_argument("--base_model_path", default=None,
                   help="Override base-model path for LoRA adapters "
                        "(use when running offline and adapter_config "
                        "points to an HF hub ID)")
    args = p.parse_args()

    model_type = detect_model_type(args.model_path)

    print(f"\n{'='*50}")
    print(f"Model   : {args.model_path}  ({model_type})")
    print(f"Dataset : {args.dataset}  |  Thinking: {not args.no_thinking}")
    print(f"{'='*50}\n")

    model, tokenizer = load_model_and_tokenizer(
        args.model_path, model_type,
        base_model_override=args.base_model_path,
    )
    prompt_config    = load_prompt_config(Path(args.prompt_config))
    test_df          = load_test_data(args.dataset)

    # ── Run inference ────────────────────────────────────────────────────────────
    results = run_inference(
        model,
        tokenizer,
        test_df,
        model_type,
        prompt_config,
        use_thinking=not args.no_thinking,
        max_samples=args.max_samples,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        thinking_budget=args.thinking_budget,
        batch_size=args.batch_size,
    )

    # ── Evaluate and save results ────────────────────────────────────────────────
    df = pd.DataFrame(results)
    metrics = compute_metrics(results)
    print_metrics(metrics)

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    results_file = f"{args.output_dir}/{args.dataset}_{ts}.jsonl"
    with open(results_file, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    summary_file = f"{args.output_dir}/{args.dataset}_{ts}_summary.json"
    with open(summary_file, "w") as f:
        json.dump(
            dict(model_path=args.model_path, dataset=args.dataset,
                 timestamp=ts, prompt_config=args.prompt_config, metrics=metrics),
            f, indent=2,
        )

    print(f"Results : {results_file}")
    print(f"Summary : {summary_file}")


if __name__ == "__main__":
    main()
