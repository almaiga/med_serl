# MedSeRL — Progress & Experiment Log

> **Purpose of this file.** A single, honest record of what we have actually run,
> what each run produced, and what we learned — so numbers stop getting mixed up
> and unverifiable claims stop leaking into the paper. Every quantitative claim
> below cites the file it came from. If a number has no source, it does not go in
> this table.
>
> **How to update.** When a run finishes, add a row to the run log (§2) and, if it
> changes a conclusion, edit the finding (§4–§7). Never overwrite a number — mark
> the old one `SUPERSEDED` and say why. Keep the status legend honest.

**Last updated:** 2026-07-15

---

## 0. Status legend

| Tag | Meaning |
|---|---|
| ✅ **VERIFIED** | Number is reproducible from a file in this repo (source cited). |
| ⚠️ **UNVERIFIABLE** | Reported in logs/chat but the model/artifact is lost — cannot be re-run. |
| 🔁 **SUPERSEDED** | Was believed true; a later, better-controlled test overrode it. |
| 🧪 **PLANNED** | Designed and gated locally, not yet run on GPU. |

---

## 1. The system in one paragraph

A single **Qwen3-4B** policy plays both roles in a two-turn game per MEDEC clinical
note. **Injector**: given a numbered note and a mode ∈ {benign, error_injection},
rewrites one sentence (meaning-preserving edit, or a subtle clinical error).
**Judge (frozen)**: rules `SAME / CHANGED / ABSTAIN`; this is the reward ground
truth. **Assessor** (same policy, detector mode): returns `CORRECT` or a sentence
id. Reward is three-tier: exact +1.0, partial +0.5, miss −1.5, format bonus +0.2.
Training is veRL `main_ppo`, `reinforce_plus_plus`, critic-free, KL anchored to the
SFT reference. Test set: MEDEC (MS 597 + UW 328 = 925 samples, ~51% error rate).

---

## 2. Run log (chronological)

| Run | When | Judge | Key levers | Test F1 | Recall | Sent. acc | Status |
|---|---|---|---|---:|---:|---:|---|
| **Base Qwen3-4B** (zero-shot) | — | — | — | 0.675 | 0.811 | 0.554 | ✅ [judge doc §2] |
| **SFT v2** (R1-chains, v6 init) | — | — | mixed assessor+injector SFT | 0.540 | 0.459 | 0.402 | ✅ [judge doc §2] |
| **r2** | Apr | Qwen3-8B | KL≈0.001, MISS −1.0, PARTIAL 0.3 | **0.700** | **0.878** | 0.739 | ⚠️ model lost |
| **v5** | May | Qwen3-8B | KL 0.01, MISS −1.5, PARTIAL 0.5 | 0.498 | — | — | ⚠️ model lost |
| **v6 / step_66** | Jun | MedRECT-32B **thinking OFF**, hint_v2 | KL 0.01, injector budget 1024 | ~0.612 | — | ~0.623 | ⚠️ model lost, plateaued |
| **v7 (clean)** | Jul 15 | MedRECT-32B **thinking ON**, hint_v2 | KL 0.01, injector budget 1536, response_len 8192 | — | — | — | ▶ RUNNING; step33 on HF ✅ |

**v7 progress notes (2026-07-15):** launched on v6's exact code (rollout files unchanged since Apr–May; only judge thinking + injector budget differ). Through step ~33 / ~336 games: judge 100% ok on valid edits, SAME-on-error 12%, assessor exact% 28→53% and reward −0.49→+0.20 across windows — first clean evidence of the assessor learning under a calibrated judge. Injector does NOT compress its thinking (think-tok flat ~700–760, reward holding positive) — long CoT pays for the injector; the "−1.5 teaches compression" hypothesis looks wrong so far. Ops: judge thinking load caused 60s-timeout `request_failed` on ~2% of games (neutralized, not poisoned) → timeout raised to 180s + judge moving to TP=2 at restart. **Checkpoint `Abdine/qwen3-4b-medserl-v7-step33` pushed and independently verified on HF (16.09 GB weights listed from a second machine); run logs in `Abdine/medserl-v7-run-logs`.** Unlike r2/v5/v6, this checkpoint cannot be silently lost.

**Important provenance notes:**
- Base / SFT-v2 / r2 / v5 F1 numbers are from [judge_bottleneck_failure_analysis.md §2](judge_bottleneck_failure_analysis.md). These are the canonical paper numbers.
- **r2, v5, and v6/step_66 model weights are LOST** (never pushed to HF, or pushed as empty/adapter-only — see §8). Their headline numbers **cannot be independently re-verified**. Treat every r2/v5/v6 number as historical, not reproducible.
- The v6 "step_66 F1 0.612 / acc 0.623" figures come from chat logs of the training run, **not** from a file in this repo. Marked ⚠️ accordingly.

---

## 3. The effect of SFT — measured, not assumed

There are **two different SFT experiments** and they must not be conflated:

### 3a. SFT-v2 (the R1-chains mixed SFT used to init v6) — ✅ [judge doc §2]
Init'ing from R1-chains mixed SFT **lowered** headline F1 vs the zero-shot base
(0.675 → 0.540) and collapsed recall (0.811 → 0.459). SFT made the model *less*
aggressive. This is the paradox that started the whole investigation: the base
model's high F1 is driven by high recall, i.e. **over-flagging**, not by better
discrimination.

### 3b. Assessor-only SFT scaling sweep — ✅ [results/medrect_sft_scaling/assessor_scaling_medrect_v1/accuracy_vs_sft_quantity_from_paste.csv]
11 fractions (0 → 2540 examples), 3 seeds each. Full table (F1 / accuracy / recall / sentence-acc):

| examples | acc | F1 | recall | sent-acc |
|---:|---:|---:|---:|---:|
| 0 (base) | 0.587 | **0.668** | **0.808** | 0.573 |
| 254 | 0.586 | 0.575 | 0.545 | 0.406 |
| 508 | 0.597 | 0.599 | 0.587 | 0.454 |
| 1016 | 0.574 | 0.630 | 0.706 | 0.565 |
| 1270 | 0.587 | 0.540 | 0.472 | 0.397 |
| 2540 (full) | 0.601 | 0.616 | 0.622 | 0.521 |

**What this shows (VERIFIED):**
1. **Detection accuracy is flat at ~0.59–0.60 across every SFT quantity**, including 0. More SFT data does *not* move the discriminative ceiling.
2. The base model's F1 advantage (0.668) is a **prevalence / over-flagging artifact**: it comes from recall 0.808 at fixed ~50% prevalence, not from being a better detector. As soon as SFT calms the over-flagging, F1 drops and only claws back at full data.
3. **Sentence-localization accuracy does not improve with SFT** either (0.573 base vs 0.521 full). Localization is the hard part and SFT alone doesn't crack it.

**Honest limits of this claim:** this is *assessor-only* SFT on a specific dataset;
it is not proof that no fine-tuning can help, only that this data at this scale
moves accuracy ~0. The flat accuracy curve is the strongest single piece of
evidence we have that the task is discrimination-bound, not data-bound.

---

## 4. The judge is the binding constraint — ✅ [judge doc §4–§12]

The central finding of the project. In self-play with a **frozen** judge, the
judge's calibration on the live distribution caps reachable test performance.
Reward shaping acts *downstream* of this and cannot fix it.

**Evidence (v5 game logs, 4,951 games — [judge doc Appendix A]):**
- Of 2,474 injected errors, the Qwen3-8B judge ruled **SAME (missed the error) on 674 = 27%**.
- In those 674 missed-error games the reward **penalized the assessor 489 times for correctly catching the error**, and **rewarded it 185 times for missing it**. The reward signal was *inverted* 27% of the time.

**Judge fix, validated on probes — ✅ [judge doc §9]:** on a 150-probe adversarial
set, swapping the judge lifts overall accuracy dramatically:

| Judge config | Probe accuracy |
|---|---:|
| Qwen3-8B (v5 judge) | 43% |
| Qwen3-32B, same prompt | 87% |
| MedRECT-32B, native task prompt | 9% |
| MedRECT-32B + hint_v2 prompt | **93%** |

Two independent findings: (1) **scale alone** (8B→32B) fixes most of it; (2) the
biggest single lever is **task reframing** — MedRECT scores 9% on its native
detect-and-localize task but 93% when asked to *compare original vs edited
sentence*. The medical knowledge was intact; localization was killing it.

---

## 5. Reward hacking — both directions we actually observed

Reward hacking here is not the model gaming a bug; it's the policy exploiting
**judge bias**. We saw both directions:

### 5a. Recall-collapse hack (v5) — ✅ [judge doc §4–§5]
When the judge systematically mislabels subtle errors as SAME, the reward
punishes the assessor for being right. Under strong KL + steep miss penalty the
policy learned the judge's blind spot and **stopped flagging subtle errors** →
recall collapse → F1 0.700 (r2) down to 0.498 (v5). r2 "won" only because weak KL
left it aggressive enough to absorb the false-positive cost and still catch real
errors — **aggression was masking the judge bias**, not solving it.

### 5b. Precision over-flag hack (v1 hint prompt) — ✅ [judge doc §9.2]
The mirror mode. When the judge over-calls benign edits as errors, the injector
is rewarded for meaning-preserving rewrites (brand↔generic, abbreviation
expansion, temporal-unit conversion) that the judge wrongly flags. On Bucket C
(expected SAME), the v1 hint prompt ruled **CHANGED on 14/30 benign swaps**. The
v2 prompt (four lines carving out these classes as non-errors) cut that to 7/30
and moved the bucket from 13%→77% with no regression elsewhere.

**Lesson (VERIFIED):** both directions are fixed by **judge design (prompt +
scale)**, not by reward shaping. "Before tuning the reward, fix the judge."

---

## 6. The thinking-OFF judge discovery (this conversation) — 🔁 updates §4/§10

This is a genuine update to the earlier analysis and must be recorded carefully.

**What the earlier doc concluded:** §10/§12.5 validated `medrect_hint_v2 @
MedRECT-32B` **with thinking OFF** and reported **99.7% recall / 0% miss on 311
real MEDEC errors**. On the strength of that, v6 shipped with the judge running
thinking-off (also because a token-budget incident — the judge overrunning
max_tokens — was "fixed" by turning thinking off and capping max_tokens at 128).

**What we found now:** a new synthetic test built from v6's *observed* failure
modes ([scripts/self_play/generate_synthetic_test_dataset.py], 48 balanced cases)
run through the real judge functions:
- thinking-**ON**: 44/48 ≈ 92% correct; 47/48 emitted a `<think>` block; avg ~509 reasoning tokens ([test_judge_thinking_accuracy.py]).
- thinking-**OFF**: caught **0/6 medium** and **0/6 subtle** errors on the same set.

**Reconciliation (the honest part — not "the old doc was wrong"):**
- The earlier 99.7%/0% validation was on **real MEDEC errors (error_flag=1)** — relatively *clear* clinical errors — and on **hand-constructed analog probes** in fixed buckets. Thinking-off handles those.
- The **live self-play injector** produces a different, harder distribution: subtle *medium-band* errors and near-benign edits sitting right on the SAME/CHANGED boundary. On *that* band, thinking-off is blind. The earlier validation sets did not cover the live distribution's hard middle.
- Therefore thinking-off is a **strong candidate explanation for the v6 plateau**, now fixed for v7. It is *candidate*, not proven, because the v6 model is lost and we cannot re-run the exact comparison. What is proven: on reconstructed subtle cases, thinking-on ≫ thinking-off.

**Caveat on the fix:** the synthetic test is a *reconstruction* of failure modes,
not the raw v6 game logs. The direction (on ≫ off for subtle errors) is robust;
the exact magnitude is specific to the synthetic set.

**Fixes applied for v7:**
- [scripts/self_play/judge_client.py]: detection branch `enable_thinking: True`.
- [configs/prompts/medrect_judge_prompts.json]: `max_tokens 128 → 2048` (room to finish thinking + emit verdict).

---

## 7. No test-time reasoning benefit (memorization probe) — ✅ (with caveat)

Probe: same models evaluated with thinking on vs off on MEDEC.
**Finding:** thinking gives ≈0 lift; accuracy stays flat ~0.60 regardless of
model or thinking mode. Combined with §3b (flat accuracy vs SFT quantity), this
says the task is **discrimination-bound**: neither more SFT data nor test-time
reasoning moves the accuracy ceiling for this 4B policy.

**Honest downgrade:** originally phrased as "memorization." That over-claims —
contamination was never proven. The defensible statement is **"no measurable
test-time reasoning benefit and no SFT-quantity benefit on detection accuracy."**

> Note the distinction from §6: test-time reasoning doesn't help the **4B
> policy/assessor** detect errors, but it *does* help the **32B judge** rule on
> subtle edits. Different model, different role — don't conflate them.

---

## 8. What is lost / unverifiable (so we stop citing ghosts)

| Artifact | Expected location | Reality |
|---|---|---|
| r2 model (F1 0.700) | HF | never pushed / lost |
| v5 model (F1 0.498) | HF | never pushed / lost |
| v6 step_66 (F1 ~0.612) | `Abdine/qwen3-4b-medrect-mixed-v2-step66` | **404 — not pushed** |
| scaling adapters | `Abdine/qwen3-4b-medrect-mixed-r2-adapter` | **empty (only .gitattributes)** |

**Root cause:** old push scripts printed a cosmetic "URLs:" line that was *not*
an upload confirmation. Fixed by [scripts/self_play/hf_push_verified.py] (uploads
then re-lists to confirm required files landed; exits 1 if not) and
[scripts/self_play/checkpoint_watcher.sh] (verified push of every new checkpoint).
**Rule going forward: a push is not done until list-tree confirms the files.**

---

## 9. v7 clean-run configuration (🧪 PLANNED, gated locally)

| Knob | Value | Why |
|---|---|---|
| Actor init | `Abdine/qwen3-4b-medrect-mixed-v2` | clean R1-chains SFT |
| Judge | `pfnet/Preferred-MedRECT-32B`, `hint_v2`, **thinking ON** | §6 fix |
| Judge max_tokens | 2048 | finish thinking + verdict |
| Injector budget | **1536** | §10 crash fix; fits under 8192 |
| Rollout response_length | **8192** | user constraint: larger = too slow |
| KL_COEF | 0.01 | §12.6: correct under calibrated judge |
| Reward | EXACT 1.0 / PARTIAL 0.5 / MISS −1.5 / FORMAT 0.2 | §12.6 |
| Resume | auto | — |
| verl | pinned to pre-2026-06-08 main commit | avoid transfer_queue breakage |

**Pre-flight gates that passed locally:** offline config/reward/data check, HF
auth + verified write, judge thinking test (44/48). See
[RUNBOOK_clean_run.md](RUNBOOK_clean_run.md).

**Bugs found and fixed while gating (this conversation):**
- **Injector parse-failure 27%** — injector (a thinking model) had 1024 tokens for *both* reasoning and the edit; ran out mid-`<think>`. Fixed: budget → 1536.
- **Tensor-size crash** (`Expected 8192 got 8386`) — raising injector to 2048 pushed the multi-turn sequence over response_length 8192. Fixed: 1536 (not 2048), keep 8192.
- **verl `transfer_queue` ModuleNotFoundError** on fresh pods — bleeding-edge main needs an unpublished dep. Fixed: pin verl to a pre-2026-06-08 commit.
- **jq missing** on fresh pods → false "healthy" monitor. Fixed: [game_health.py] (pure stdlib).

---

## 10. Open questions for v7 (what this run must answer)

1. Does a **thinking-ON judge** raise the assessor's ceiling above the v6 plateau, or is the task discrimination-bound regardless (§3b/§7)? This is the whole bet.
2. Does fixing the injector budget (fewer degenerate games) change the reward-signal quality enough to matter?
3. Can we produce a **defensible headline model** — clean SFT + calibrated judge + validated reward — that we can actually push and re-verify (unlike r2/v5/v6)?
4. Watch for the §5b **precision over-flag** hack re-emerging now that the judge is stronger.

---

### One-line summary of where we are

We know *why* v6 plateaued (a candidate: a subtle-error-blind thinking-off judge + a
starved injector budget, both now fixed), we know the task is discrimination-bound
on the 4B policy (flat accuracy vs SFT and vs thinking), and we know the judge —
not the reward — is the binding constraint. v7 is the clean run that tests whether
fixing the judge lifts the ceiling. Every prior headline model is lost and must be
treated as historical.
