# When the Aggression Hides the Judge: A Failure Analysis of Adversarial Self-Play for Medical Error Detection

> **Project:** MedSeRL — self-play RL for the MEDEC medical error detection benchmark.
> **Scope of this note:** why an "improved" training run regressed below its own SFT baseline, and what the failure tells us about adversarial self-play with a frozen judge.

---

## TL;DR

1. We trained a single Qwen3-4B policy to detect medical errors in clinical notes via two-turn adversarial self-play. An early run (here called **r2**) — trained with default-strength KL and a mild symmetric miss penalty — reached **test F1 0.700** and **recall 0.878**, comfortably beating both the SFT init (F1 0.593) and the zero-shot base (F1 0.675).
2. A later run (**v5**) made what looked like a textbook fix: steepen the miss penalty, raise the KL coefficient ≈10×, and align the reward EV. It **regressed to F1 0.498**.
3. The proximate diagnosis we used to justify v5 ("at 50/50 benign/error, always-CORRECT has positive expected value, so steepen the miss") is wrong: r2 ran with exactly those "bad" constants and was the *most aggressive* policy we have, not a conservative one.
4. The real binding constraint is the **judge**. On v5's own game logs (≈4,950 games), of 2,474 injected errors:
   - the judge ruled **CHANGED 1,350** (55%),
   - **SAME 674** (27%, i.e. the judge missed the injected error),
   - **ABSTAIN 450** (18%).
   In the 674 missed-error games, the reward then **penalized the assessor 489 times for correctly catching the error** and **rewarded it 185 times for missing it**.
5. r2 won *despite* this because weak KL and a mild symmetric miss left the assessor aggressive: it flagged a lot, absorbed the false-positive cost, and so still caught real subtle errors on the test set. v5's "fix" stripped that aggression and the latent judge bias surfaced as a recall collapse.
6. **General lesson.** In an adversarial self-play game with a frozen judge, the judge's calibration is the *binding constraint* on the reachable test-time performance. Reward shaping does not fix judge bias; it either *masks* it (by inducing aggression) or *surfaces* it (by inducing conservatism). The only durable fix is the judge.
7. The rest of this document gives the evidence and frames the small experimental program we use to validate a stronger judge (`pfnet/Preferred-MedRECT-32B`) before any further self-play.

---

## 1. The setup

A single Qwen3-4B policy plays both roles in a two-turn game per clinical note:

1. **Injector.** Given the numbered note and a sampled mode ∈ {`benign`, `error_injection`}, it returns a rewritten single sentence at a chosen sentence id. In `benign` mode the edit must be meaning-preserving (synonym, abbreviation, exact temporal conversion). In `error_injection` mode it must introduce a single subtle, realistic clinical error.
2. **Judge (frozen, Qwen3-8B).** Given the original sentence, the modified sentence, and the surrounding note, returns one of `SAME / CHANGED / ABSTAIN` with a confidence score. This is the ground truth the reward is built on.
3. **Assessor.** The same policy, now in detector mode, returns `CORRECT` or a sentence id.

The reward is a three-tier zero-sum-coupled signal: exact (+1.0), partial detection (+0.5), miss (−1.5), with a small format bonus (+0.2). Training is veRL `main_ppo` with `reinforce_plus_plus`, critic-free, with KL anchored to the SFT reference policy.

The judge is **frozen**. This is by design — the assessor must learn to satisfy a fixed external standard — but it makes the judge's mistakes structural, not transient.

---

## 2. The paradox

The two runs we are comparing share the same SFT init, the same data splits, and the same self-play architecture. Held-out evaluation on the MEDEC test set:

| Model | F1 | Recall | Sentence acc. |
|---|---|---|---|
| Base Qwen3-4B (zero-shot) | 0.675 | 0.811 | 0.554 |
| New SFT (R1 chains) | 0.593 | 0.543 | 0.453 |
| **r2** (April, "broken-data" self-play) | **0.700** | **0.878** | **0.739** |
| **v5** (May, "fixed" self-play) | 0.498 | — | — |

Two facts make this a paradox rather than a result:

- r2 outperforms not only its own SFT but also the *zero-shot base* by 25 points of F1. Self-play actually *worked*.
- v5 was deliberately engineered to be an improvement on r2 — re-tuned rewards, stronger KL anchoring, data-alignment fixes — and lost 20 points of F1 against it.

The deltas between r2 and v5, ordered by suspected importance:

| Lever | r2 | v5 | Direction |
|---|---|---|---|
| `algorithm.kl_ctrl.kl_coef` | veRL default (≈ 0.001) | 0.01 | ≈10× anchor strength |
| `REWARD_MISS` | −1.0 | −1.5 | steeper symmetric penalty |
| `REWARD_PARTIAL` | 0.3 | 0.5 | larger detection sub-reward |
| Benign/error mix | 50/50 | 50/50 (intended drop to ~35 % unapplied) | unchanged |
| `rl_train.jsonl` fields | runtime segmentation | pre-joined canonical `sentences`, `error_sentence_id` | both internally consistent (verified) |

The data alignment turned out to be a red herring — current `rl_train.jsonl` round-trips through `parse_numbered_sentences` and the assessor's prompt with `MATCH=True` on every spot check, and the train mix is still 50/50 by the runtime logs. The active levers are the reward retune and the KL change.

---

## 3. The (wrong) initial diagnosis

The story we told ourselves before v5:

> Under the old constants the assessor's *always-say-CORRECT* policy has positive expected value on a 50/50 benign/error mix:
> `EV(always_correct) = 0.5 · (+1.0 + 0.2) + 0.5 · (−1.0 + 0.2) = +0.2`
> Therefore RL slides into "always CORRECT." Fix: steepen `REWARD_MISS` so the always-CORRECT EV goes negative; raise KL so the policy stays close to a well-calibrated SFT.

The math is correct. The diagnosis is not.

The reason we know the diagnosis is wrong is **r2**. r2 was trained with exactly the same `REWARD_EXACT = +1.0`, `REWARD_PARTIAL = +0.3`, `REWARD_MISS = −1.0` constants. If those constants caused conservatism, r2 should have collapsed too. Instead r2 reached recall **0.878** — the most *aggressive* policy in the comparison table. The conservatism we observed in an earlier conservative run cannot be attributed to a property of the reward function that allows a known aggressive run as a solution.

In other words: the +0.2 EV trap is real, but it is not the binding constraint. Something else gates the policy that the reward retune did not address.

---

## 4. The real mechanism: a judge-inverted reward

Take v5's game logs (49,510 turns across roughly 4,950 games) and cross-tabulate the injector's requested mode against the judge's verdict:

| requested_mode \\ judge_verdict | SAME | CHANGED | ABSTAIN | Total |
|---|---|---|---|---|
| benign           | 2,038 (82 %) |   288 (12 %) | 151 (6 %)  | 2,477 |
| error_injection  |   674 (27 %) | 1,350 (55 %) | 450 (18 %) | 2,474 |

The relevant cell is the **674** in the bottom-left: in 27 % of the games where the injector successfully introduced a real medical error, the Qwen3-8B judge rules the edit `SAME`. Combined with the 18 % `ABSTAIN`, **45 % of injected-error games never produce a usable detection signal**.

Now zoom into those 674 missed-error games and ask what the assessor said:

| Assessor said on judge-missed errors | Count | Reward outcome |
|---|---|---|
| `CORRECT` (i.e. matched the judge, which is wrong) | 185 | **+1.2** (rewarded for missing the real error) |
| `ERROR @ id` (i.e. caught the real error correctly) | 489 | **−1.3** (penalized for correctly catching) |

**This is the smoking gun.** On 489 separate occasions in a single run, the policy correctly identified a real injected medical error and was penalized for it because the judge disagreed. Both directions of this cell push the policy toward "stop flagging."

The training-time signal that flagging is unsafe is then visible in the run's trajectory:

```
recall (judge-CHANGED games): 0.73 → 0.55 → 0.69 → 0.74 → 0.82 → 0.80 → 0.76 → 0.69 → 0.61 → 0.68
fp_rate (judge-SAME games):    0.33 → 0.23 → 0.21 → 0.27 → 0.34 → 0.39 → 0.42 → 0.22 → 0.18 → 0.20
```

The false-positive rate falls cleanly from 0.33 to 0.18–0.20 as training progresses — the policy is learning to flag less. Recall on judge-CHANGED games stays roughly flat because the denominator only contains the games where the judge could already recognize the error. **The conservatism is real, but it is induced by the judge × reward interaction, not by the reward EV.**

### Why the game logs *look healthy* while the test result is bad

In-game recall (~0.65–0.82) is computed only over the games the judge ruled `CHANGED`, i.e. the obvious or garbage edits the judge already recognizes. The subtle, realistic edits — the ones an actual reviewer in MEDEC writes — are exactly the ones the judge mislabels as `SAME`, so they are *excluded from the in-game denominator*. The MEDEC test set is essentially the held-out version of that excluded distribution. So:

- The game tells you the policy is fine on the easy half of the edit space.
- The test tells you the policy collapsed on the hard half.
- The numbers move in opposite directions across the training run.

---

## 5. Why the "fix" backfired

`REWARD_MISS` is applied **symmetrically** to false negatives *and* false positives. Steepening it from −1.0 to −1.5 raises the marginal cost of flagging by 50 %. Given a judge that calls a quarter of real injected errors `SAME`, *every correct flag on those cases looks like a false positive to the reward*. The policy therefore learns to flag less, and because subtle real errors *look benign to this judge*, the policy stops flagging them on the test set too. Recall collapses.

The 10× KL anchor compounds the effect. The reference policy v5 anchors to is the new R1-chains SFT, which on test scores recall 0.543 — well below the base model's 0.811. With weak KL, r2 was free to drift toward the base model's flagging frequency. With strong KL, v5 is pinned close to the low-recall SFT and cannot escape it.

Both changes are individually defensible. Both fight the symptom while leaving the binding constraint — the judge — untouched.

---

## 6. The general lesson: aggression hides judge bias

Stated as a principle:

> In an adversarial self-play game with a frozen judge, the judge's calibration on the held-out distribution is the binding constraint on test-time performance. Reward shaping operates on in-game outcomes, but in-game outcomes are conditioned on the judge's ability to recognize the edits. When that conditioning is misaligned with the test distribution, reward shaping moves the policy on a *different* manifold than the one the test measures.

In our case the misalignment is asymmetric: the judge is biased *against* recognizing subtle errors. An aggressive policy (high flagging rate) accidentally compensates by overshooting; a calibrated policy on the in-game signal undershoots on the test signal. r2 stumbled into the aggressive regime because we did not penalize flagging strongly enough; v5 was engineered out of that regime in the name of calibration and immediately collapsed.

Three corollaries that we believe generalize beyond this project:

1. **In-game recall over-states test-time recall whenever the judge under-flags the hard cases.** Watching the in-game number during training will mislead you. The judge's confusion matrix on a held-out probe set is the relevant diagnostic.
2. **Symmetric miss penalties penalize judge-disagreed correct detections at exactly the same rate as judge-agreed false alarms.** This is fine when the judge is well-calibrated and a footgun when it is not.
3. **Strong KL anchoring is only as good as the SFT anchor.** If the SFT has low recall and the judge has a recall bias, KL turns the SFT's bias into a floor.

---

## 7. Implications for the broader self-play literature

The "Self-RedTeam" line of work and adjacent reviewer-game setups have repeatedly bumped into judge-quality issues, usually framed as judge over-refusal (the reviewer rejects safe content). Our failure is the mirror image: judge *under-detection* of subtle harm. The same structural cause underlies both — a frozen judge whose error distribution on training-time edits differs from the test distribution — and we suspect the same diagnostic (probing the judge with a controlled adversarial set before deciding whether reward shaping or judge replacement is the right intervention) is the right first move in either direction.

---

## 8. Where this leaves us: a small experimental program

We do *not* believe further reward tuning is the right next step. The next step is to verify that swapping the judge to `pfnet/Preferred-MedRECT-32B` — a model fine-tuned for exactly the detect-and-localize discrimination the judge has to make — actually fixes the binding constraint, and to characterize its own failure modes before committing to a new training run.

Three experiments, run in order, with explicit decision rules:

### Exp 1 — Standalone discrimination on held-out test

Implementation: `scripts/self_play/run_detection_prompt.py`, in-process vLLM, no server.
Inputs: `(model, prompt, data)`. Compare Qwen3-8B and MedRECT-32B, with the number-only and correction-style detection prompts, on 200 balanced ms-test cases.
Primary metric: **error recall** (sensitivity to real MEDEC errors). Secondary: correct-note specificity, localization accuracy.
**Decision rule:** MedRECT-32B must clear Qwen3-8B by **≥ 10 percentage points of recall** to justify the judge swap.

### Exp 2 — Adversarial probe set

Construct ~30 cases in each of five buckets, derived programmatically from `medec_test.json` so the ground truth is known by construction (no judge in the loop for labeling):

| Bucket | Construction | Expected verdict |
|---|---|---|
| A. Pharmacological analog swap | metoprolol ↔ atenolol, omeprazole ↔ pantoprazole, ceftriaxone ↔ ceftazidime | CHANGED |
| B. Dose / route / timing perturbation | `5 mg` → `50 mg`; `PO` → `IV`; `q6h` → `q24h` | CHANGED |
| C. Meaning-preserving synonyms | `5 days` ↔ `120 hours`; `T2DM` ↔ `type 2 diabetes mellitus` | SAME |
| D. Surface-similar, clinically opposite | `warm water` ↔ `hot water`; `left` ↔ `right`; `ANA+` ↔ `ANA−` | CHANGED |
| E. Degenerate / nonsense | length-truncated; non-medical phrase substituted | filtered before reward |

A and B are the failure modes that produce the 489-penalty cell above; D is the failure mode that produces the 185-reward cell. C is a sanity check that we have not simply created the *opposite* reward hack (a judge that calls everything CHANGED).

**Decision rule:** MedRECT-32B must dominate Qwen3-8B on **A + B + D** *without* over-flagging **C**. If it over-flags C the recall gain will reappear as a false-positive penalty during training.

### Exp 3 — Red-team the chosen judge

After Exp 1 and 2 pick a winner, have the current Qwen3-4B SFT produce 200 edits per mode and label them with the chosen judge. Inspect the top 20 judge-disagreements (judge says SAME on an `error_injection` prompt, or CHANGED on a `benign` prompt). The fraction of episodes that produce judge-disagreements is the upper bound on the reward-hack surface area available to the injector. If that fraction is < 5 % (vs Qwen3-8B's ~27 %), we have controlled the hack.

### Ship decision

- If MedRECT-32B passes Exp 1 + Exp 2 + Exp 3 → run **one** self-play with the new judge, with r2's recipe (weak KL, mild symmetric miss). Compare to r2 on MEDEC test.
  - Beats r2 → ship the new model.
  - Does not beat r2 → ship **r2**; close the project on the finding "judge bottleneck identified and quantified; given the current judge, r2's recipe is the right one."
- If MedRECT-32B fails any of Exp 1 / Exp 2 → ship r2; the judge swap is not the right intervention either.

---

## Appendix A — Reproducible numbers used in this analysis

From the user's own SSH run, 22 game-log files spanning the v5 training run, 4,951 game rows:

```
== requested_mode == benign 2,477 | error_injection 2,474
== judge_verdict  == SAME 2,712  | CHANGED 1,638 | ABSTAIN 601
== judge_status   == ok 4,340 | parse_failure 408 | semantic_abstain 95 | truncation 89 | request_failed 4
== assessor_label == CORRECT 2,706 | ERROR 2,172 | UNKNOWN 73
== assessor_outcome== exact_match 3,222 | miss 1,448 | game_invalid 188 | partial_match 70 | invalid_format 23
== injector_outcome== exact_match 3,388 | wrong_edit_type 962 | parse_failure 413 | judge_semantic_abstain 95 | truncation_filter 89 | judge_unavailable 4
```

Injected errors judged SAME (judge missed real error): **674 / 2,474 = 27 %.**
Of those 674, assessor said CORRECT (rewarded for missing): **185 / 674.**
Therefore assessor said ERROR (penalized for catching): **674 − 185 = 489.**

Per-file trend over the live run (ignoring the tiny April smoke files):

```
file                     n     recall    fp_rate
2026-05-21 103924       24     0.67      0.64
2026-05-21 104607      544     0.73      0.33
2026-05-21 112223      544     0.55      0.23
2026-05-21 115438      544     0.69      0.21
2026-05-21 122332      544     0.74      0.27
2026-05-21 125128      544     0.82      0.34
2026-05-21 131914      160     0.80      0.39
2026-05-21 141344       24     0.80      0.60
2026-05-21 142605      544     0.76      0.42
2026-05-21 145937      544     0.69      0.22
2026-05-21 153107      544     0.61      0.18
2026-05-21 155746      368     0.68      0.20
```

These numbers will not appear in the published paper but are the empirical basis for the qualitative claims in §4 and §5.
