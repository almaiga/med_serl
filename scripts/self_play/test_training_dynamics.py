#!/usr/bin/env python3
"""Standalone REINFORCE++ dynamics test — no Ray, no veRL, 1 GPU.

Verifies that:
  1. compute_score produces non-trivial rewards on real parquet data
  2. Log-probs of generated sequences change after each gradient update
  3. Rewards shift across rounds (policy is actually improving or at least moving)

Usage:
    python scripts/self_play/test_training_dynamics.py \
        --model Qwen/Qwen3-4B \
        --parquet data_processed/self_play/train_grpo.parquet \
        --rounds 3 \
        --batch  4 \
        --max-new-tokens 256 \
        --lr 1e-5

What each round does:
    - Generate responses for the batch with the current policy
    - Call compute_score to get rewards
    - Run one REINFORCE++ gradient update (advantage = reward - baseline)
    - Re-compute log-probs of the same (prompt, response) pairs
    - Report Δlp: if non-zero, weights changed
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import pandas as pd

# Keep env clean before importing reward_function
import os
os.environ.setdefault("MEDSERL_GAME_LOG", str(PROJECT_ROOT / "outputs" / "test_dynamics_log.jsonl"))

from scripts.self_play.reward_function import compute_score  # noqa: E402


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _parse_extra_info(val) -> dict:
    if isinstance(val, dict):
        return val
    try:
        return json.loads(str(val))
    except Exception:
        return {}


def load_batch(parquet_path: str, batch_size: int) -> list[dict]:
    """Load up to batch_size samples from the parquet, mixing roles."""
    df = pd.read_parquet(parquet_path)
    # Prefer assessor rows (clearer reward signal for a quick sanity check)
    def _role(row):
        return _parse_extra_info(row.get("extra_info", {})).get("role", "assessor")

    df["_role"] = df.apply(_role, axis=1)
    assessor_df = df[df["_role"] == "assessor"]
    chosen = assessor_df.head(batch_size) if len(assessor_df) >= batch_size else df.head(batch_size)

    samples = []
    for _, row in chosen.iterrows():
        prompt = row.get("prompt") or row.get("messages")
        if isinstance(prompt, str):
            try:
                prompt = json.loads(prompt)
            except Exception:
                continue
        if not isinstance(prompt, list):
            continue

        rm = row.get("reward_model", {})
        if isinstance(rm, dict):
            gt = str(rm.get("ground_truth", ""))
        else:
            gt = str(rm) if rm else ""

        extra_info = _parse_extra_info(row.get("extra_info", {}))
        samples.append({"messages": prompt, "ground_truth": gt, "extra_info": extra_info})

    return samples


# ---------------------------------------------------------------------------
# Log-prob helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_mean_log_prob(model, prompt_ids: torch.Tensor, response_ids: torch.Tensor) -> float:
    """Mean per-token log-prob of response_ids given prompt_ids."""
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    out = model(input_ids=input_ids)
    # logits[t] predicts token t+1
    log_probs = torch.log_softmax(out.logits[:, :-1], dim=-1)  # (1, seq-1, vocab)
    prompt_len = prompt_ids.shape[1]
    resp_len = response_ids.shape[1]
    # Slice to response positions: prompt_len-1 .. prompt_len-1+resp_len
    resp_log_probs = log_probs[0, prompt_len - 1: prompt_len - 1 + resp_len]  # (resp_len, vocab)
    token_lp = resp_log_probs.gather(1, response_ids[0].unsqueeze(1)).squeeze(1)  # (resp_len,)
    return token_lp.mean().item()


def reinforce_step(
    model,
    optimizer,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
    advantage: float,
) -> float:
    """One REINFORCE gradient step. Returns loss value."""
    input_ids = torch.cat([prompt_ids, response_ids], dim=1)
    out = model(input_ids=input_ids)
    log_probs = torch.log_softmax(out.logits[:, :-1], dim=-1)
    prompt_len = prompt_ids.shape[1]
    resp_len = response_ids.shape[1]
    resp_lp = log_probs[0, prompt_len - 1: prompt_len - 1 + resp_len]
    token_lp = resp_lp.gather(1, response_ids[0].unsqueeze(1)).squeeze(1)
    loss = -(advantage * token_lp.mean())
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return loss.item()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Test REINFORCE++ training dynamics (1 GPU, no veRL)")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--parquet", default="data_processed/self_play/train_grpo.parquet")
    ap.add_argument("--rounds", type=int, default=3, help="Number of update rounds")
    ap.add_argument("--batch", type=int, default=4, help="Samples per round")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device       : {device}")
    print(f"Model        : {args.model}")
    print(f"Parquet      : {args.parquet}")
    print(f"Rounds       : {args.rounds}")
    print(f"Batch size   : {args.batch}")
    print(f"Max new tok  : {args.max_new_tokens}")
    print(f"LR           : {args.lr}")
    print()

    # ── Load model ──────────────────────────────────────────────────────────
    print("Loading tokenizer + model …")
    from transformers import AutoTokenizer, AutoModelForCausalLM  # noqa: E402

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    # ── Load data ────────────────────────────────────────────────────────────
    parquet_path = str(PROJECT_ROOT / args.parquet) if not Path(args.parquet).is_absolute() else args.parquet
    print(f"Loading data from {parquet_path} …")
    samples = load_batch(parquet_path, args.batch)
    if not samples:
        print("ERROR: No samples loaded. Check parquet path and format.")
        sys.exit(1)
    print(f"Loaded {len(samples)} samples (role: {samples[0]['extra_info'].get('role','?')})")
    print()

    # ── Pre-tokenize prompts (fixed across all rounds) ───────────────────────
    prompt_ids_list = []
    for s in samples:
        prompt_text = tokenizer.apply_chat_template(
            s["messages"], tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
        prompt_ids_list.append(ids)

    # ── Training rounds ──────────────────────────────────────────────────────
    history = []

    for round_idx in range(args.rounds):
        sep = "=" * 64
        print(f"{sep}\nROUND {round_idx + 1}/{args.rounds}\n{sep}")

        rewards, lp_deltas, losses = [], [], []
        baseline = 0.0  # running mean for REINFORCE++ advantage

        for i, (sample, prompt_ids) in enumerate(zip(samples, prompt_ids_list)):
            # 1. Generate ─────────────────────────────────────────────────────
            with torch.no_grad():
                gen_ids = model.generate(
                    prompt_ids,
                    max_new_tokens=args.max_new_tokens,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            response_ids = gen_ids[:, prompt_ids.shape[1]:]
            response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

            # 2. Reward ───────────────────────────────────────────────────────
            reward = compute_score(
                data_source="test_dynamics",
                solution_str=response_text,
                ground_truth=sample["ground_truth"],
                extra_info=sample["extra_info"],
            )
            rewards.append(reward)

            # 3. Log-prob BEFORE update ───────────────────────────────────────
            lp_before = compute_mean_log_prob(model, prompt_ids, response_ids)

            # 4. REINFORCE++ advantage = reward - baseline ─────────────────────
            advantage = reward - baseline
            baseline = 0.9 * baseline + 0.1 * reward  # exponential moving avg

            # 5. Gradient step ─────────────────────────────────────────────────
            loss_val = reinforce_step(model, optimizer, prompt_ids, response_ids, advantage)
            losses.append(loss_val)

            # 6. Log-prob AFTER update ────────────────────────────────────────
            lp_after = compute_mean_log_prob(model, prompt_ids, response_ids)
            delta_lp = lp_after - lp_before
            lp_deltas.append(delta_lp)

            print(
                f"  [{i+1}/{len(samples)}] "
                f"reward={reward:+.3f}  adv={advantage:+.3f}  loss={loss_val:+.4f}  "
                f"lp_before={lp_before:.4f}  lp_after={lp_after:.4f}  "
                f"Δlp={delta_lp:+.6f}"
            )
            # Show first 80 chars of response (stripped of thinking)
            stripped = response_text.split("</think>")[-1].strip()[:80]
            print(f"           response: {stripped!r}")

        avg_reward = sum(rewards) / len(rewards)
        avg_delta = sum(lp_deltas) / len(lp_deltas)
        policy_moved = any(abs(d) > 1e-7 for d in lp_deltas)

        history.append(
            dict(
                round=round_idx + 1,
                avg_reward=avg_reward,
                rewards=rewards,
                avg_lp_delta=avg_delta,
                policy_moved=policy_moved,
            )
        )

        print(
            f"\n  → avg_reward={avg_reward:+.4f}  "
            f"avg_Δlp={avg_delta:+.8f}  "
            f"policy_moved={'YES' if policy_moved else 'NO'}\n"
        )

    # ── Final verdict ────────────────────────────────────────────────────────
    sep = "=" * 64
    print(f"{sep}\nDYNAMICS SUMMARY\n{sep}")
    reward_series = [h["avg_reward"] for h in history]
    delta_series = [h["avg_lp_delta"] for h in history]

    for h in history:
        print(
            f"  Round {h['round']}: avg_reward={h['avg_reward']:+.4f}  "
            f"avg_Δlp={h['avg_lp_delta']:+.8f}  "
            f"policy_moved={h['policy_moved']}"
        )
    print()

    any_movement = any(h["policy_moved"] for h in history)
    reward_range = max(reward_series) - min(reward_series)

    print(f"  Policy weights changed    : {'✓ YES' if any_movement else '✗ NO  ← weights are frozen or gradient is broken'}")
    print(f"  Reward range across rounds: {reward_range:.4f}")
    print()

    if not any_movement:
        print("  DIAGNOSIS:")
        print("    - Δlp ≈ 0 after every update means weights did not change.")
        print("    - Possible causes:")
        print("      1. Model loaded from flash_attn / quantized ckpt that blocks grad")
        print("      2. All parameters require_grad=False")
        print("      3. Advantage = 0 for all samples (all rewards equal baseline)")
        print()
        # Count frozen parameters
        frozen = sum(1 for p in model.parameters() if not p.requires_grad)
        trainable = sum(1 for p in model.parameters() if p.requires_grad)
        print(f"    - Trainable params: {trainable}, Frozen params: {frozen}")
        if frozen and not trainable:
            print("    - ALL parameters are frozen! Check model loading flags.")
    else:
        print("  Policy is updating correctly.")
        print("  If veRL training still shows no improvement, the issue is in:")
        print("    - veRL's advantage normalization (rewards collapsing to zero mean)")
        print("    - Batch size too small → advantage std too low → tiny gradients")
        print("    - GPU memory offloading race conditions in FSDP2")
    print(sep)


if __name__ == "__main__":
    main()
