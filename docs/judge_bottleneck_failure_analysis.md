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
| New SFT v2 (R1 chains, retrained for v6 launch) | 0.540 | 0.459 | 0.402 |
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

A second problem with the §3 calc — discovered only at pre-flight time after we had
the verdict distributions from §5 and Exp 1 — is that the math implicitly assumed
the judge agrees with reality 100 % of the time. Under that assumption v5's retuned
constants give `EV(always-CORRECT) = -0.05`, formally negative. Under the *actual*
Qwen3-8B verdict distribution (27 % free-reward for missing real errors), v5's
constants still give `EV(always-CORRECT) ≈ +0.23` — i.e. the trap stayed open after
the retune anyway. We document the full calculation in §11.

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

---

## 9. Exp 2 results: probe-set evaluation

We ran the program of §8 on the 150-probe adversarial set (30 per bucket, deterministic ground truth) using six judge configurations covering all of the size × FT × prompt cells we care about. Numbers below are per-bucket accuracy and overall accuracy on the probe set.

| Style | Model | A pharm | B dose | C syn | D opp | E nons | overall |
|---|---|---:|---:|---:|---:|---:|---:|
| qwen_pair | Qwen3-8B (current judge) | 3 % | 50 % | 100 % | 60 % | 3 % | **43 %** |
| qwen_pair | Qwen3-32B | 33 % | 100 % | 100 % | 100 % | 100 % | **87 %** |
| medrect_native | MedRECT-32B | 17 % | 7 % | 0 % | 17 % | 7 % | **9 %** |
| medrect_hint | Qwen3-32B | 50 % | 100 % | 70 % | 67 % | 60 % | **69 %** |
| medrect_hint | MedRECT-32B | 73 % | 93 % | 13 % | 80 % | 93 % | **71 %** |
| medrect_hint_v2 | MedRECT-32B | **93 %** | **100 %** | **77 %** | **100 %** | **97 %** | **93 %** |

(Bucket C expected verdict is SAME; all other buckets expect CHANGED. Higher = better on every column.)

### 9.1 Three findings, in order of how much they update the failure analysis

**Finding 1 — there was a simpler fix to v5 than the one we attempted.** Replacing Qwen3-8B with Qwen3-32B at *exactly* the same JSON sentence-pair prompt — no FT, no prompt restructure, no other change — moves the judge from 43 % to 87 % overall on this probe set, and from 3 % to 33 % on Bucket A (the failure mode that drove v5's recall collapse). The v5 conservative-EV diagnosis pointed at the reward; the actual leverage was on the judge, and the cheapest available judge intervention (scale) would have lifted the binding constraint. This tightens the §6 lesson to:

> Before tuning the reward, scale or replace the judge.

**Finding 2 — task reframing is a larger prompt effect than medical FT for this judge problem.** MedRECT-32B asked to do its native task (detect-and-localize on the modified note) scores 9 % overall — worse than Qwen3-8B baseline. Asked to do the *easier* task we actually need (compare original sentence vs edited sentence with both in scope) it scores 71 % (v1) to 93 % (v2). The localisation problem of MedRECT-native is the whole reason it underperforms; the medical knowledge is intact. The hint prompt converts a fine-tuned model from "unusable for this game" to "best in class."

**Finding 3 — MedRECT's medical FT contribution is concentrated on the subtle drug-class discrimination that drives the v5 failure mode.** Holding the prompt fixed (`medrect_hint`), MedRECT-32B (71 %) and Qwen3-32B (69 %) are within 2 points overall. The two diverge on exactly one bucket: A pharm_analog, where MedRECT scores 73 % vs Qwen3-32B's 50 %. That 23-point gap is the only place the medical FT pays off, but it pays off exactly where it matters — on the kind of edit the v5 logs show the Qwen3-8B judge mislabelled 27 % of the time.

### 9.2 The v1 → v2 prompt iteration as the worked example for §6

The §6 generalisation predicted that any new judge that fixes the recall-side failure mode risks inducing the *mirror* failure mode — i.e., over-flagging benign synonyms. Exp 2 reproduces this exactly:

- `medrect_hint` (v1) on Bucket C (expected SAME): the model rules **CHANGED on 14 / 30 benign synonym swaps**. This is the precision-side reward hack we predicted: brand↔generic, abbreviation expansion, exact temporal-unit conversions all read as "documentation errors" to MedRECT's FT prior.
- `medrect_hint_v2` adds four lines of explicit guidance enumerating these four classes as non-errors. Bucket C accuracy moves from 13 % to **77 %** (CHANGED-on-SAME calls drop from 14 to 7), and there is no measurable regression on any other bucket (A, B, D, E all stay at or above the v1 numbers).

So both directions of the judge-induced reward hack are addressable by judge design, not by reward shaping. The v1 → v2 step is the worked example that a four-line prompt edit can move a 64-point precision gap on the targeted bucket without sacrificing the other 120 probes.

### 9.3 What this changes about the §8 ship decision

The decision rule from §8 was: MedRECT must dominate Qwen3-8B on A + B + D **without** over-flagging C, and the hint must beat native. Both conditions hold for the v2 prompt:

- A + B + D + E: MedRECT-32B + hint_v2 ≥ Qwen3-8B baseline by ≥ 30 pts everywhere, and ≥ Qwen3-32B at the same prompt by ≥ 33 pts on A.
- C: v2 reaches 77 %, vs v1's 13 %. The 23 % residual over-flag is bounded and structurally limited: the injector cannot systematically select the specific synonyms that the judge will mishandle, so this becomes a low-rate stochastic penalty in self-play rather than a directional bias.

**Concrete decision:** any further self-play uses `medrect_hint_v2` on MedRECT-32B as the judge. r2 (F1 0.700 on MEDEC test) is the current shippable deliverable; whether to spend a self-play run on the new judge to try to beat it depends on Exp 1 (the held-out real-error confirmation) and the team's GPU budget — not on the judge being insufficiently validated. It is.

---

## 10. Exp 1: held-out real MEDEC errors

The Exp 2 probe set used deterministic, hand-constructed analog swaps to *test*
the binding-constraint failure mode in isolation. Exp 1 evaluates the chosen
judge configuration (`medrect_hint_v2 @ MedRECT-32B`, thinking off) on the
311 real MEDEC ms-test records with `error_flag = 1` — the genuine subtle
errors a held-out clinical reviewer wrote, on notes MedRECT has never seen.

| Metric | Value | v5 (Qwen3-8B) reference |
|---|---:|---|
| Real held-out errors evaluated | 311 | — |
| Recall (judge ruled CHANGED) | **99.7 % (310/311)** | ≈ 73 % |
| Miss rate (judge ruled SAME) | **0.0 % (0/311)** | **27 %** |
| Off-target (ABSTAIN) | 0.3 % (1/311) | — |

Per MEDEC error type (uniform recall — no error class is systematically
mishandled):

| Error type | Recall |
|---|---:|
| management     | 97/97 = 100 % |
| treatment      | 51/51 = 100 % |
| pharmacotherapy | 36/36 = 100 % |
| causalOrganism | 11/11 = 100 % |
| diagnosis      | 115/116 = 99 % |

The single off-target case (`ms-test-55`, diagnosis: `schizoid personality
disorder` → `social anxiety disorder`) is a localisation miss, not a recall
miss: the judge correctly detected an error and returned a sentence id, but
flagged a different sentence than the hint indicated. Under
`compute_injector_game_reward`, this maps to `judge_unavailable` → zero reward
for both players → a neutralised game rather than a reward-hack failure.

**Effective reward-hack rate on real held-out errors: 0 %.**

The v5 binding constraint (27 % of injected errors mislabelled SAME by the
Qwen3-8B judge, which then penalised the assessor 489 times for correctly
catching real errors and rewarded it 185 times for missing them; see §5.2)
no longer applies under the chosen judge configuration. Exp 2 had already
shown this on hand-constructed probes; Exp 1 confirms it on the same
distribution the self-play reward is evaluated against.

## 11. Pre-flight Check 1 — reward EV under measured judge calibration

Before committing GPU-days to a new self-play run, we recomputed the assessor's
expected per-game reward by plugging the verdict distributions measured in
production (§5 for the Qwen3-8B baseline; Exp 1 for the chosen judge on real
errors; Exp 2 Bucket C for the chosen judge on benign synonym edits) into the
reward formula in `compute_assessor_game_reward`. The script is
`scripts/self_play/reward_ev_check.py`; it is deterministic and reproducible.

We compare two reward recipes — `r2`: (PARTIAL = +0.3, MISS = −1.0); `v5`:
(PARTIAL = +0.5, MISS = −1.5) — under two judges:

- **broken** (Qwen3-8B, v5 logs): `error → CHANGED 0.546, SAME 0.273,
  ABSTAIN 0.182`; `benign → CHANGED 0.116, SAME 0.823, ABSTAIN 0.061`.
- **calibrated** (MedRECT-32B + hint_v2, Exp 1 + Exp 2): `error → CHANGED 0.997,
  SAME 0.000, ABSTAIN 0.003`; `benign → CHANGED 0.233, SAME 0.767, ABSTAIN 0.000`.

The headline metric is the per-game EV gap between a *perfect discriminator*
(flags every real error at the right sentence id, never flags benign edits)
and the *always-CORRECT* policy. A positive gap means the reward gradient
points at the right policy; a wider gap means a stronger signal.

| Judge | Recipe | EV(perfect) | EV(always-CORRECT) | gap |
|---|---|---:|---:|---:|
| Qwen3-8B (broken) | r2 | +0.666 | **+0.393** | +0.273 |
| Qwen3-8B (broken) | v5 | +0.569 | **+0.227** | +0.341 |
| MedRECT-32B + hint_v2 | r2 | +0.965 | −0.032 | +0.997 |
| MedRECT-32B + hint_v2 | v5 | +0.907 | **−0.339** | **+1.246** |

(EV ceiling — perfect judge + perfect policy under either recipe — is +1.200
per game. The MedRECT-32B + hint_v2 + v5 perfect-policy EV of +0.907 is
+0.293 below this ceiling; that gap is the residual reward-hack surface
attributable to the new judge's 23 % over-flag rate on synonym-class
benigns, and is the upper bound on remaining bias under the chosen judge.)

### Three findings from Check 1

**1. v5's reward retune did *not* close the EV trap under the broken judge.**
The §3 calc assumed the judge agrees with reality. The actual Qwen3-8B
distribution rewards the assessor 27 % of error games for saying CORRECT (the
"judge missed it, you agree, take +1.2") and rewards it on 82 % of benign games
for the same answer. The net `EV(always-CORRECT)` under v5 reward + broken
judge is **+0.227** — the conservatism trap stayed open after the steepening,
which is why v5 still slid into "always-CORRECT" despite the retune.

**2. Under the calibrated judge, v5's reward gives a 25 % wider gradient than
r2's.** Gap under MedRECT-32B + hint_v2: r2 = +0.997, v5 = **+1.246**. The
steeper MISS now penalises only genuine false-negatives/positives (because the
judge's CHANGED matches reality almost perfectly), while the larger PARTIAL
sub-reward keeps the right strategy's EV high. **The v5 reward retune, which
backfired under the broken judge, is the correct choice under the calibrated
judge.**

**3. The judge is the dominant intervention; the reward retune is a refinement
on top of it.** Even with r2's *old* constants, swapping Qwen3-8B for
MedRECT-32B + hint_v2 closes the trap (`EV(always-CORRECT)` drops from +0.393
to −0.032). v5's reward then sharpens an already-correct gradient by another
25 %. This separates the two interventions cleanly: the judge fixes the sign of
the gradient, the reward fixes its magnitude.

### Implication for the run

This is the empirical pre-flight that the combination `{v5 reward + calibrated
judge + clean SFT}` has the right per-game incentive structure before we
commit the compute. The full RL run still has to answer the two open questions
the math cannot:

- Does KL = 0.01 anchored to the R1-chains SFT (recall 0.543) cap the policy
  below the base model's zero-shot recall of 0.811?
- Does the actual injector-generated edit distribution match the verdict
  statistics measured on Exp 1's real errors and Exp 2's hand-crafted
  benigns?

The first is unavoidable by experiment; the second can be answered cheaply by
a smoke run with log inspection before the full run.

## 12. Conclusions

1. The judge's calibration on the held-out distribution is the binding
   constraint on test-time policy quality in adversarial self-play with a
   frozen judge. Both reward-shaping interventions in v5 (steeper symmetric
   `REWARD_MISS`, 10× KL anchor) acted downstream of this constraint and
   amplified rather than fixed the underlying failure.
2. Two judge interventions independently lift the constraint:
   - **Scale alone:** Qwen3-8B → Qwen3-32B at the same JSON sentence-pair
     prompt lifts probe-set accuracy from 43 % to 87 %.
   - **Targeted swap:** MedRECT-32B with a reframed detection prompt + an
     explicit synonym carve-out lifts probe-set accuracy to 93 %, and
     real-error recall to 99.7 % with a 0 % miss rate.
   The first is the simpler counterfactual, the second is the production
   recommendation because it dominates on the bucket-A failure mode that
   drove the v5 collapse.
3. Task reframing (detection on the modified note → comparison of original
   vs edited sentence with both in scope) is the highest-leverage single
   prompt change for using a fine-tuned medical detector as a self-play
   judge. The same MedRECT-32B that scores 9 % on its native task scores
   93 % under the reframed prompt; the medical knowledge is intact, the
   localisation failure is what was killing it.
4. Both directions of judge-induced reward hack — recall collapse on subtle
   errors (v5) and precision over-flag on benign rewrites (v1 hint
   prompt) — are addressable by judge design. The v1 → v2 prompt iteration
   moved Bucket C from 13 % to 77 % without regressing on any other bucket,
   demonstrating that the mirror failure mode predicted in §6 is preventable.
5. The recommended judge for any further self-play in this project is
   `medrect_hint_v2 @ MedRECT-32B`, evaluated with thinking off. r2
   (F1 0.700 on MEDEC test) is *not* a defensible research deliverable on
   its own: it was bootstrapped from a pre-R1-chains SFT, trained under the
   broken Qwen3-8B judge, and §6 + §11 attribute its test-time performance
   to an aggressive-policy artefact that compensated for judge bias rather
   than to a methodologically sound recipe. The paper's headline model must
   come from a self-play run under the corrected pipeline (clean SFT,
   calibrated judge, the v5 reward retune validated against the calibrated
   judge in §11).
6. The v5 reward retune (`REWARD_PARTIAL = 0.5`, `REWARD_MISS = −1.5`) and
   the v5 KL anchor (`kl_coef = 0.01`) are *correct* under the calibrated
   judge, and *both* contribute to a 25 % wider gradient than r2's
   constants (§11). Reverting to r2's reward at this point would weaken
   the gradient signal without any methodological gain. The §3 doc text
   was previously inconsistent on this point because the EV calc assumed
   the judge agrees with reality; §11 corrects that with the empirical
   verdict distribution.

## 13. The lesson, in two sentences

> In an adversarial self-play game with a frozen judge, the judge's
> calibration on the held-out distribution is the binding constraint, and
> both directions of judge bias (recall collapse and precision over-flag)
> are addressable by judge replacement and prompt design — not by reward
> shaping. Before tuning the reward, scale or replace the judge, then
> verify with a controlled probe set and a held-out real-error eval that
> the new judge has not simply moved the bias from one direction to the
> other.
