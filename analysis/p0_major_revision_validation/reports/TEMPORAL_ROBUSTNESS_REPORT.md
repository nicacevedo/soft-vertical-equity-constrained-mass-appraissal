# TEMPORAL ROBUSTNESS REPORT — strict-date, oracle repeat-parcel, and unseen-parcel designs

Stage 3B. This report covers **four distinct objects** that must never be conflated:

| Object | Status | Evaluation sets | Comparable to the frozen numbers? |
|---|---|---|---|
| **PRIMARY frozen temporal design** | the manuscript's design; unchanged by this stage | frozen rolling-origin blocks | — |
| **D-SNAP** | **strict-date robustness** (A5). Not a replacement design unless Gates G5a→G5b promote it | frozen blocks plus the boundary-date rows moved to the evaluation side | yes, near-identical denominators |
| **D-PURGE** | **ORACLE overlap-removal robustness DIAGNOSTIC** (A15). Explicitly not a deployable or prospectively implementable CCAO split | **bitwise identical to the primary** | yes, same rows, same denominators |
| **D-UNSEEN** | secondary evaluation-**subset** view. **Zero fits** | restricted subsets of each evaluation block | **no** — different denominators, never differenced against the frozen numbers |

**D-PURGE could not be implemented prospectively.** Constructing each training block requires knowing which PINs appear in the *future* evaluation block, which no assessor knows at training time. It is a diagnostic that deliberately *over*-corrects: if the path conclusions survive an oracle removal of all repeat-parcel information, they are not driven by that information. It is **not** a proposed sample-construction rule and **not** a recommendation to CCAO.

**Outcome.** `G5a = ALERT` (one trigger, `T1_beta_log_sign_or_ordering`). `G5b = NOT_CONFIRMED`. `TEMPORAL_STATUS = PASS_PRIMARY_STANDS`.

---

## 1. How the three fitted designs were built

### 1.1 D-SNAP — strict-date separation at all eight boundaries

The positional cut is moved **back** to the first row sharing the boundary date, so every boundary-date row lands on the evaluation side and `max(train_date) < min(eval_date)` strictly.

| boundary | primary train | snap train | rows moved to eval | boundary date | train max | eval min | strict |
|---|---|---|---|---|---|---|---|
| fold_1 | 46,888 | 46,842 | 46 | 2017-02-16 | 2017-02-15 | 2017-02-16 | yes |
| fold_2 | 100,776 | 100,620 | 156 | 2018-04-23 | 2018-04-22 | 2018-04-23 | yes |
| fold_3 | 151,187 | 151,066 | 121 | 2019-06-03 | 2019-06-02 | 2019-06-03 | yes |
| fold_4 | 200,908 | 200,785 | 123 | 2020-08-04 | 2020-08-03 | 2020-08-04 | yes |
| fold_5 | 252,486 | 252,451 | 35 | 2021-08-17 | 2021-08-16 | 2021-08-17 | yes |
| fold_6 | 298,022 | 297,890 | 132 | 2022-08-01 | 2022-07-31 | 2022-08-01 | yes |
| fold_7 | 310,147 | 309,989 | 158 | 2022-11-21 | 2022-11-20 | 2022-11-21 | yes |
| development_heldout | 344,607 | 344,565 | 42 | 2023-11-09 | 2023-11-08 | 2023-11-09 | yes |
| production_2025 | 382,897 | 382,897 | 0 | 2025-01-01 | 2024-12-31 | 2025-01-01 | yes |

All eight development/held-out boundaries are strict after the snap-back; between 35 and 158 rows move per boundary. The **2025 boundary already was strict** — it is year-based — so `production_2025` moves zero rows and every `forward_2025` D-SNAP number is *exactly* the frozen number. That is a design property, not a coincidence, and it is why the 2025 column shows zero deltas throughout.

### 1.2 D-PURGE — oracle removal of paired-evaluation parcel history

From each **training** block, every row whose `meta_pin` appears in the **corresponding evaluation** block is removed. Evaluation blocks are preserved exactly.

| block | primary train | purged train | rows removed | share removed | eval size (unchanged) | eval PINs left in train |
|---|---|---|---|---|---|---|
| fold_1 | 46,888 | 46,383 | 505 | 1.077% | 5,209 | **0** |
| fold_2 | 100,776 | 99,380 | 1,396 | 1.385% | 11,197 | **0** |
| fold_3 | 151,187 | 148,315 | 2,872 | 1.900% | 16,798 | **0** |
| fold_4 | 200,908 | 195,707 | 5,201 | 2.589% | 22,323 | **0** |
| fold_5 | 252,486 | 244,950 | 7,536 | 2.985% | 28,053 | **0** |
| fold_6 | 298,022 | 288,234 | 9,788 | 3.284% | 33,113 | **0** |
| fold_7 | 310,147 | 299,524 | 10,623 | 3.425% | 34,460 | **0** |
| development_pool | 344,607 | 331,416 | 13,191 | 3.828% | 38,290 | **0** |
| production_2016_2024 | 382,897 | 372,097 | 10,800 | 2.821% | 26,641 | **0** |

Training shrinks by 1.1%–3.8%. The purge is complete in every block (zero paired-evaluation PINs survive in training) and evaluation sizes are unchanged — asserted elementwise, not assumed (`mode_overlap` raises `ProtocolViolation` otherwise).

### 1.3 D-UNSEEN — evaluation subset, zero fits

No model is refit. The **frozen** cached predictions are restricted to evaluation rows whose PIN never appears in that model's own training block.

| block | eval rows | never-seen rows | share |
|---|---|---|---|
| fold_1 | 5,209 | 4,713 | 90.5% |
| fold_2 | 11,197 | 9,859 | 88.1% |
| fold_3 | 16,798 | 14,120 | 84.1% |
| fold_4 | 22,323 | 17,629 | 79.0% |
| fold_5 | 28,053 | 21,439 | 76.4% |
| fold_6 | 33,113 | 24,624 | 74.4% |
| fold_7 | 34,460 | 25,340 | 73.5% |
| heldout | 38,290 | 27,126 | 70.8% |
| forward_2025 | 26,641 | 17,744 | 66.6% |

Every mask is re-derived from the frozen rule and its `mask_hash` re-verified before use. **Denominators change**, so D-UNSEEN numbers are reported on their own base and are never differenced against the frozen path.

---

## 2. Validation-block overlap — carried forward from Stage 2, and audited per design

The seven rolling-origin **validation** blocks are **not mutually disjoint**: the validation blocks of **fold 6 and fold 7 overlap**. This is a property of the frozen primary design, not of this stage.

| design | pooled appearances | distinct rows | duplicated appearances | share | max appearances of any row | overlapping pairs |
|---|---|---|---|---|---|---|
| primary | 151,153 | 130,165 | 20,988 | 13.885% | 2 | fold_6&fold_7 |
| dsnap | 151,924 | 130,778 | 21,146 | 13.919% | 2 | fold_6&fold_7 |
| dpurge | 151,153 | 130,165 | 20,988 | 13.885% | 2 | fold_6&fold_7 |

**Interpretation (unchanged from the accepted Stage-2 reading — D1 is not simply 'unaffected').**

- All predictions are genuinely **out-of-training-sample**; no row is ever predicted by a model that trained on it. The overlap is between *validation* blocks, not between train and validation.
- **D1** — the equal-weight seven-fold mean — **remains the frozen primary development coordinate.**
- **However, fold-level results are not independent.** Some observations contribute to more than one validation fold, so the CV mean is a mean over *overlapping* evaluation evidence.
- Fold standard deviations are therefore **descriptive chronological-window variation, not IID standard errors**, and must never be read as such.
- **D2** is the historical duplicate-weighted pooled-OOF coordinate: it explicitly duplicate-weights the overlapping rows.
- **D3** is the row-balanced one-sale-one-vote sensitivity.

**Per-design audit result.** D-PURGE preserves the primary evaluation-set overlap **exactly** — asserted elementwise for all seven folds. D-SNAP's overlap was **measured, not assumed**: moving boundary-date rows to the evaluation side changes the duplicated share only from 13.885% to 13.919%, and the overlapping pair is the same (`fold_6&fold_7`) with the same maximum multiplicity of 2. The non-IID reading above therefore applies identically under all three designs.

---

## 3. Gate G5a — the five material-change criteria on the screening grid

Screening grid: **27 positive ρ + ρ=0** — every 4th index of the frozen 82-point grid, force-including the five display anchors and the four candidate-region endpoints. This grid is **4× coarser** than the frozen path, so a trigger firing here is an **alert, not a verdict**.

| # | criterion | D-SNAP (feeds the gate) | D-PURGE (informational only) |
|---|---|---|---|
| 1 | β_log sign / ordering / monotonicity | **FIRED** | FIRED |
| 2 | loss of the Surrogate dCor rebound | **not fired** | FIRED |
| 3 | loss of the Surrogate Δ_NL rebound | **not fired** | not fired |
| 4 | candidate endpoints moving > ~2× in ρ | **not fired** | not fired |
| 5 | loss of the moderate-ρ accuracy benefit | **not fired** | not fired |

**Only `T1` fired on D-SNAP.** D-PURGE fired `T1` and `T2`, but **D-PURGE does not feed the promotion gate** — it is an oracle diagnostic (§5).

### 3.1 What exactly fired, and why the refinement grid was built the way it was

`T1` is a compound criterion. Two *different* sub-facts fired it, in two disjoint regions of the ρ axis, and each got its own refinement region.

**Region A — Direct/Surrogate β_log ordering flips (small ρ).** Sign of `β_log(Direct) − β_log(Surrogate)` differs between the frozen and D-SNAP paths at four screening ρ on `CV_mean` and four on `heldout`; zero on `forward_2025` (which is identical by construction).

| evaluation | ρ | gap frozen | gap D-SNAP | min\|gap\| |
|---|---|---|---|---|
| CV_mean | 0.00109854 | -0.000724 | +0.000069 | 0.000069 |
| CV_mean | 0.0019307 | +0.000133 | -0.000307 | 0.000133 |
| CV_mean | 0.00339322 | -0.001249 | +0.000981 | 0.000981 |
| CV_mean | 0.00596362 | -0.001133 | +0.000299 | 0.000299 |
| heldout | 0.00109854 | -0.002042 | +0.000863 | 0.000863 |
| heldout | 0.0019307 | +0.001222 | -0.001924 | 0.001222 |
| heldout | 0.0184207 | -0.000465 | +0.001444 | 0.000465 |
| heldout | 0.0323746 | +0.000648 | -0.003321 | 0.000648 |

Every flip happens where the two families are **near-tied**: the largest `min|gap|` over all eight flips is **1.22e−03**, against an overall gap range up to **0.0838** on the same paths. The pre-registered separation tolerance is **τ = 0.002** (§3.2).

**Region B — Surrogate held-out β_log 'all negative' status changed (large ρ).** On the screening grid the frozen Surrogate held-out path reaches **+1.07e−04 at ρ=86.85** — a single point marginally above zero — while the D-SNAP path stays negative, peaking at **−1.54e−04 at ρ=100**. That is a **2.6e−04 movement across zero**.

**Why refinement was required.** Neither sub-fact is in the plan's direct-promotion class (no β_log sign change of the *path*, no ordering reversal holding across the whole grid, no dCor collapse). Both are exactly the kind of turning-point/near-tie artifact a 4×-coarse grid can manufacture. So §J.6 mandates filling in the **skipped original-grid ρ values inside each affected region, extended by one screening interval on each side**.

| region | reason | families | screening span | ρ added | fits |
|---|---|---|---|---|---|
| **A** | Direct/Surrogate beta_log ordering flips at 4 screening rhos | direct, surrogate | [0.00109854, 0.0568987] | 21 | 378 |
| **B** | Surrogate held-out beta_log 'all negative' status changed | surrogate | [15.9986, 100] | 9 | 81 |

Exact ρ added, per region (all are original 82-point grid values the screen skipped — **no new ρ was invented**):

- **Region A** (21 ρ): `0.00126486`, `0.00145635`, `0.00167683`, `0.002223`, `0.00255955`, `0.00294705`, `0.00390694`, `0.00449843`, `0.00517947`, `0.00686649`, `0.00790604`, `0.00910298`, `0.0120679`, `0.013895`, `0.0159986`, `0.0212095`, `0.0244205`, `0.0281177`, `0.0372759`, `0.0429193`, `0.0494171`
- **Region B** (9 ρ): `18.4207`, `21.2095`, `24.4205`, `32.3746`, `37.2759`, `42.9193`, `56.8987`, `65.5129`, `75.4312`

Total refinement: **459 fits** across **45 shards** (459 declared in the frozen config). D-SNAP protocol only — D-PURGE does not feed G5a and was **not** refined.

### 3.2 The materiality rule was fixed before the refinement was read

`TAU_MATCH = 0.002` is **not a new constant**. It is the β_log matching tolerance already frozen in `configs/matched_beta_frozen.json` for the matched-β stage, adopted here verbatim; `mode_g5b` re-reads that file and raises if the two ever disagree. Two configurations closer than τ in β_log are treated as *matched* everywhere else in this P0 pass, so a reversal of an ordering that is tighter than τ is a reversal of a tie, not of a finding.

- **Ordering**: material only if `min(|gap_frozen|, |gap_dsnap|) ≥ τ` — the families are genuinely separated on **both** paths and the ordering genuinely reversed.
- **Sign**: material only if the path maximum that crosses zero exceeds τ in absolute value — otherwise the path merely grazes zero.

---

## 4. Gate G5b — the alert under targeted local refinement

Refined support: **Direct 49 ρ**, **Surrogate 58 ρ** (screening 28 + region A 21 on both families + region B 9 on the Surrogate). The ordering test runs on the **49 ρ common to both families**; the sign test runs on each family's own refined support, so the region-B fits — which exist precisely to test the Surrogate sign alert — actually enter the gate. Duplicate rows dropped on merge: 0.

| evaluation | ρ compared | ordering flips (any) | ordering flips (**material**, τ=0.002) | sign status changed | sign change **material** |
|---|---|---|---|---|---|
| CV_mean | 49 | 10 | **0** | no | **no** |
| heldout | 49 | 13 | **0** | yes | **no** |
| forward_2025 | 49 | 0 | **0** | no | **no** |

Sign-status detail (path maximum of β_log over each family's refined support):

| evaluation | family | frozen all-negative | frozen max | D-SNAP all-negative | D-SNAP max |
|---|---|---|---|---|---|
| CV_mean | Direct | yes | -7.865e-02 | yes | -7.663e-02 |
| CV_mean | Surrogate | yes | -1.861e-02 | yes | -1.830e-02 |
| heldout | Direct | yes | -7.941e-02 | yes | -8.066e-02 |
| heldout | Surrogate | no | +8.384e-04 | yes | -1.541e-04 |
| forward_2025 | Direct | yes | -8.938e-02 | yes | -8.938e-02 |
| forward_2025 | Surrogate | yes | -1.483e-02 | yes | -1.483e-02 |

### 4.1 Per-alert verdict

| region | criterion | screening conclusion | refined evidence | verdict |
|---|---|---|---|---|
| **A** | T1 — Direct/Surrogate β_log ordering | 8 flips on the screening grid (4 `CV_mean`, 4 `heldout`) | the denser grid exposes **more** near-ties, not fewer: 23 flips on the refined support, of which **0 are material** at τ=0.002 | **NOT_CONFIRMED** |
| **B** | T1 — Surrogate held-out β_log sign status | on the screening grid the frozen path reaches +1.07e−04 at ρ=86.85 while D-SNAP stays negative (max −1.54e−04) | the 9 filled-in ρ in [18.4, 75.4] leave the **D-SNAP path negative throughout** (refined max -1.541e-04). The refinement does raise the *frozen* path's maximum to +8.384e-04 at ρ=75.43 — a newly filled point — so the status difference is real but still 2.4× below τ=0.002 | **NOT_CONFIRMED** |

**Gate G5b = `NOT_CONFIRMED`.** no promotion -- the screening alert was a near-tie / near-zero artifact that did not survive local refinement; the primary design stands

`full_dsnap_regeneration_launched_in_this_run = false`. Per the stage authorization, the full 82-point D-SNAP regeneration is NOT launched in this run even under a confirmed promotion; it requires separate authorization.

---

## 5. D-SNAP conclusion — does strict-date separation change any core path conclusion?

**No.**

| evaluation | mean ΔR² | min ΔR² | max ΔR² | mean Δβ_log | max \|Δβ_log\| |
|---|---|---|---|---|---|
| CV_mean | +0.00001 | -0.00090 | +0.00305 | +0.00069 | 0.00284 |
| heldout | -0.00124 | -0.00407 | +0.00225 | +0.00034 | 0.00571 |
| forward_2025 | +0.00000 | +0.00000 | +0.00000 | +0.00000 | 0.00000 |

- The β_log path keeps its **shape, sign and family ordering** wherever the families are separated by more than the matching tolerance; the only ordering changes are near-ties below τ, and they do not survive refinement.
- The **Surrogate dCor rebound survives** on all three evaluations (retained share 1.10, 1.30, 1.00 of the frozen rebound — on `CV_mean` and `heldout` it is in fact **larger** under D-SNAP).
- The **Surrogate Δ_NL rebound survives** (1.00, 1.08, 1.00), with an interior minimum in every case.
- **Candidate-region endpoints do not move materially** (largest factor 1.53× for Direct, 1.00× for Surrogate; the criterion is ~2×). These are screening-grid **proxy** endpoints applied identically to both paths, **not** a reproduction of the published smoothed changepoint estimator.
- The **moderate-ρ accuracy benefit is retained** near ρ≈0.954 on every evaluation.
- `forward_2025` is **numerically identical** to the frozen path, because the 2025 boundary was already strictly date-separated.

**D-SNAP therefore remains reported as strict-date robustness. The primary temporal design stands. No 82-point regeneration is warranted or authorized.**

---

## 6. D-PURGE conclusion — what changes after oracle repeat-parcel removal, and what does not

Reminder: **oracle diagnostic**, evaluation sets bitwise identical to the primary, so every number below is directly comparable on the same rows. Training loses 1.1%–3.8% of its rows.

**What does NOT change (the paper's within-path claims):**

- **Direct/Surrogate ordering** — the T1 flips under D-PURGE are the same near-tie phenomenon as under D-SNAP (3–4 flips, all in the small-ρ near-tied region).
- **The Surrogate Δ_NL rebound survives** (0.94, 0.87, 0.87 retained), interior minimum intact.
- **Candidate-region endpoints** — no material movement (same 1.53× / 1.00× as D-SNAP).
- **The moderate-ρ accuracy benefit** — retained on every evaluation.
- **Held-out and 2025 transfer** — the qualitative pattern is unchanged; the mechanism still moves in the same direction with ρ on both transfer blocks.

**What DOES change:**

- **β_log becomes more negative at the origin.** At ρ=0: -0.01184 (`CV_mean`), -0.02949 (`heldout`), -0.02257 (`forward_2025`). Removing parcel history makes the *uncorrected* baseline measurably **more** regressive — i.e. repeat-parcel information was modestly masking regressivity, which strengthens rather than weakens the paper's motivation.
- **dCor(e, y) rises everywhere** (+0.021 `CV_mean`, +0.052 `heldout`, +0.043 `forward_2025` at ρ=0): residual–price dependence is genuinely higher once parcel history is gone.
- **`T2` fires: the Surrogate dCor rebound is substantially attenuated** — retained share 0.25 (`CV_mean`), 0.13 (`heldout`), 0.13 (`forward_2025`), against a 0.25 retention threshold. The rebound **still exists** with an interior minimum and a positive recovery, but it is compressed from both sides: on held-out the valley floor **rises** (0.2501 → 0.2543, and its location moves from ρ=9.10 to ρ=28.12) while the ρ=100 endpoint **falls** (0.2673 → 0.2566). The mechanism is visible along the whole path — D-PURGE raises dCor at small ρ (+0.052 at ρ=0) and lowers it at large ρ (−0.011 at ρ=100) — so the curve flattens rather than the rebound vanishing. This is the one place where the oracle diagnostic materially changes a reported quantity, and it is reported as such rather than smoothed over.

**Interpretation.** Ordinary shifts in absolute accuracy under D-PURGE are *expected* — parcel history is genuinely informative — and threaten nothing, because every claim in the manuscript is **within-path**. Accuracy deltas are in fact small and mixed (mean ΔR² -0.00091 on `CV_mean`). **Absolute performance deterioration under D-PURGE is not, by itself, evidence against the primary design.**

**D-PURGE is NOT promoted to the primary temporal protocol.** It remains labelled `design_type = oracle_diagnostic` on every row of every table it appears in, and its triggers are recorded with `feeds_g5_gate = False`.

---

## 7. D-UNSEEN conclusion — do the conclusions survive on never-before-seen parcels?

**Yes.** Zero fits were performed; this is a row subset of the frozen cached predictions. **Denominators change** relative to the primary evaluation — between 66.6% and 90.5% of each block survives — so these numbers are reported on their own base and never differenced against the frozen path.

| evaluation | eval rows used | Direct β_log ρ=0 → ρ=100 | Surrogate β_log ρ=0 → ρ=100 | Surrogate attains more correction at |
|---|---|---|---|---|
| CV_mean | — | -0.1393 → -0.0799 | -0.1386 → -0.0195 | 26/27 ρ |
| heldout | 27,126 | -0.1609 → -0.0936 | -0.1590 → -0.0142 | 24/27 ρ |
| forward_2025 | 17,744 | -0.1801 → -0.1129 | -0.1794 → -0.0288 | 24/27 ρ |

- The **headline path ordering persists**: the Surrogate attains substantially more first-order correction than the Direct family at high ρ on every evaluation, on 24–26 of 27 ρ.
- Both **rebounds survive with interior minima**: Surrogate dCor (`CV_mean` 0.2605 at ρ=16 → 0.2668; `heldout` 0.2465 at ρ=28.1 → 0.2541; `forward_2025` 0.2545 at ρ=49.4 → 0.2564) and Surrogate Δ_NL (`CV_mean` 0.0853 at ρ=1.68 → 0.1303; `heldout` 0.0802 at ρ=2.56 → 0.1147; `forward_2025` 0.0739 at ρ=2.95 → 0.0959).
- Baseline regressivity is **stronger** on never-before-seen parcels (`heldout` β_log at ρ=0 is −0.1609 vs −0.1474 on the full block), consistent with the D-PURGE finding that parcel history masks part of the measured regressivity.

---

## 8. Statuses and scope

```
MATCHED_BETA_STATUS = PASS
TEMPORAL_STATUS    = PASS_PRIMARY_STANDS
G5a                = ALERT (T1 only, D-SNAP)
G5b                = NOT_CONFIRMED
```

**Matched-β is untouched by this stage.** No matched-β model was refit and no matched-β artifact was regenerated; the committed tables, figures and report are bit-identical to the checkpoint that produced them. D1 remains the primary development coordinate, D2 the duplicate-weighted pooled-OOF sensitivity and D3 the row-balanced sensitivity.

**Not executed, and not authorized in this pass:** the full 82-point D-SNAP regeneration (J12), any new hyperparameter tuning, any new jurisdiction or model family, P1 PRB inference, P1 VEI significance, Duan smearing, subgroup/township analysis, and any manuscript edit.

---

## 9. Provenance and exact reproduction

- Branch `p0-major-revision-validation`; environment `fairness_env` — Python 3.9.19, LightGBM 4.6.0, numpy 1.26.4, pandas 2.3.1, scikit-learn 1.6.1, scipy 1.13.1, dcor 0.6.
- Every fit in this stage used the frozen 994-tree parameter vector `lgbm_params_sha256 = 8f0f2acd83118de782604b5ca7143acfbd2af3fd186ea9376588f9bcf560585b` under `HISTORICAL (no determinism pins)` settings.
- Screening: 54/54 shards (`robustness_shard__{design}__{block}__{family}.csv`), 2 designs × 9 blocks × 3 families.
- Refinement: 45/45 shards, 459 fits, indexed with per-shard ρ-set and file hashes in `tables/dsnap_refinement_provenance.csv`.
- Split index arrays are stored untracked under `output/` and referenced by SHA-256 in `configs/split_protocol_dsnap.json` / `configs/split_protocol_dpurge.json`; every load re-verifies the hash.
- D-UNSEEN masks are re-derived and re-verified against `configs/unseen_subset_definition.json` on every run.

**Artifacts produced or finalized by this stage**

| artifact | rows |
|---|---|
| `tables/dsnap_refinement.csv` | 1 |
| `tables/dsnap_refinement_provenance.csv` | 45 |
| `tables/robustness_path_dsnap.csv` | 513 |
| `tables/robustness_path_dpurge.csv` | 513 |
| `tables/robustness_unseen_subset.csv` | 495 |
| `tables/robustness_vs_frozen_deltas.csv` | 336 |
| `tables/temporal_validation_overlap_audit.csv` | 6 |
| `tables/temporal_material_change_triggers.csv` | 10 |
| `tables/gate_g5a_outcome.json` | json |
| `tables/gate_g5b_outcome.json` | json |
| `tables/temporal_material_change_triggers_detail.json` | json |
| `tables/temporal_validation_overlap_summary.json` | json |

