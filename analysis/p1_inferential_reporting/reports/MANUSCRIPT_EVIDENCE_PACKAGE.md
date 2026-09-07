# P1 MANUSCRIPT EVIDENCE PACKAGE — inferential reporting

**Status:** FINAL
**Branch:** `p1-inferential-reporting`
**P0 base:** tag `p0-major-revision-final-20260907` = `805c426e1587972a2a07dcaf60220603397c0d3e`
**Scope:** four inferential-reporting tasks. **No manuscript file was edited.**

Every number below is machine-derived in `provenance/p1_headline_numbers.json` and
re-checked by `tests/test_p1_headline_numbers.py`. Artifact hashes are in
`provenance/p1_artifact_hashes.json`.

---

## 0. Verification status

| Check | Result |
|---|---|
| P0 suite **at the immutable tag**, isolated detached worktree | **153 passed, 0 failed, 0 skipped** |
| P0 suite, content-only checkout (no mtimes) | 152 passed, 1 failed — mtime-ordering test, unreproducible by any checkout |
| P0 suite **on the P1 branch** | 152 passed, 1 failed — `test_only_gitignore_modified_outside_p0`, HEAD-relative by design |
| P0 tree vs tag (`git diff`) | empty |
| P0 **final-gate** hash index (Stage-3B / G5b) | **28/28 byte-exact** |
| P0 early-stage indexes (Stage-1, output-artifact) | 48/49 and 118/125 — **the same 8 mismatches the tag itself has**; no new mismatch |
| Protected paths | clean |
| P1 assertion suites | **72 passed, 0 failed** — 64 scientific (12 PRB + 5 smearing-sign + 28 VEI + 19 smearing-apply) + 8 report-consistency |
| Manuscript files touched | **none** |

The tag's certification is **153/153**, reproduced in a separate checkout. See
§5 for exactly what had to be re-attached to that checkout and why, and §6 for
why the P1 branch reports 152/153 without any P0 change.

**On the two early-stage hash indexes.** "Every P0 index is byte-exact" would be a false
claim, and it was never true — not even at the tag. `stage1_frozen_hashes.json` and
`output_artifact_hashes.json` are **early-stage snapshots that P0's own later gates
superseded**: the G2/G3 test modules were edited in those later stages, `POSTFLIGHT.json`,
`slurm_graph.json` and `large_local_artifacts.json` are end-of-run summaries, and
`output_artifact_hashes.json` **lists itself**, a self-reference that can never match.

Verified directly: the mismatch set is **identical at the tag and on the P1 branch** (1 and
7 entries respectively), and all 8 files are **byte-identical between the two trees**. So
the correct immutability criterion — the one the freeze script enforces — is:

1. `git diff` vs the tag over the P0 tree is empty;
2. the **final** gate index (Stage-3B / G5b) is fully byte-exact;
3. **no index acquired a mismatch the tag did not already have**;
4. no index entry went missing;
5. protected paths are clean;
6. the only files changed since the tag are the P1 directory and `.gitignore`.

All six hold. The tag's own blobs are re-hashed via `git show` for criterion 3, so the check
needs no extra checkout.

---

## 1. Task 1 — distance-correlation estimator, documented

Closes the `\todo` at `paper/paper_v17_option1.tex:3204-3206`, which asked whether the
reported estimator is the biased/V-statistic or an unbiased/U-centered variant.

**Answer.** `dcor 0.6`, `dcor.distance_correlation(e, y_true_log, method="auto")`,
`exponent=1`, `compile_mode=AUTO`, `bias_corrected=False`, computed on the **full**
evaluation sample (no subsampling), with `e = y_pred_log - y_true_log` and `y = log P`.
That is the **standard biased / V-statistic (double-centered)** estimator.

Numerically pinned on an `n = 400` probe:

| comparison | absolute difference |
|---|---|
| executed vs hand-rolled double-centered **V**-statistic | `1.33e-15` |
| executed vs `sqrt` U-centered (**unbiased**) statistic | `7.52e-03` |

The executed value is the V-statistic to float tolerance and is **not** the U-centered
statistic. `method="auto"` selects an exact O(n log n) algorithm — an algorithm choice,
not an estimator choice.

**Single-estimator consistency.** Every reported path metric traces to one function,
`utils/motivation_utils.distance_correlation_e_y`. The two other `dcor` call sites in the
repository (`scripts/run_paper_baseline.py`, `quick_test_models.py`) feed no manuscript
table; the third computes a *different quantity* — dCor(ratio, log P), subsampled.
Recorded in `tables/dcor_estimator_audit.csv` / `dcor_estimator_facts.json`.

A benign redundancy is noted and **not** repaired: three P0 scripts compute `dCor_e_y`
twice per cell and overwrite it with the identical value. No effect on any number.

---

## 2. Task 2 — PRB inferential reporting (IAAO 2013, adopted guidance)

480 rows = 48 display entries × 10 evaluation blocks; **440 attained, 40 NOT_ATTAINED**.
95% CIs, classical SE displayed with HC1 beside it.

**Classification is four-state, and a CI that merely crosses a threshold is never treated
as evidence of exceeding it:**

| state | all rows | standards-facing |
|---|---|---|
| `within_pm005` | 186 | 33 |
| `overlaps_pm005` | 40 | 8 |
| `outside_pm005_but_not_pm010` | 193 | 44 |
| `outside_pm010` | 21 | 3 |
| `NOT_ATTAINED` | 40 | 8 |

Rule: *the entire 95% CI must lie outside a band before that band is deemed exceeded.*

**Reconciliation.**

- P1 OLS vs canonical `prb()` on identical observations: max `|Δ| = 1.18e-12` — the P1
  regression reproduces the function that produced the frozen values.
- Frozen artifact reconciliation: **234** comparisons, max `|Δ| = 7.63e-16`.
- D3 pooled-OOF reweighting effect: **`2.72e-03`** — reported as a *sensitivity*, not an
  error. It is the measured consequence of giving each unique sale one vote instead of
  letting the 20,988 overlapping fold-6/fold-7 rows count twice.

**No fold-as-IID aggregation anywhere** — no `mean ± SD/√7` construct exists in the
table, and this is asserted, not merely intended.

---

## 3. Task 3 — VEI inference per the May-2026 Exposure Draft

Implemented literally from the authoritative source, verified by hash:

| field | value |
|---|---|
| SHA256 | `e950e00d0c3684dd067734d401e5278dcf659f471fcfb0d6db65c7584f7c56b8` |
| bytes | 2,013,844 |
| title as printed | **"STANDARD ON RATIO STUDIES / Exposure Draft May 2026"** |
| App. D.2 / App. E | printed page 67 / pages 78–81 |
| committed | **no** — redistribution permission not established |

**Status is disclosed on every row:** proposed guidance, *not* adopted; the 2013 Standard
remains the adopted reference; **not a compliance determination**.

### 3.1 Cell-count adjudication — why 396, and where pooled-OOF went

```
480 rows      = 48 display entries × 10 evaluation blocks
396 evaluated = 44 attained entries ×  9 ED2-applicable blocks
 84 not evaluated = 44 attained pooled_oof + 40 NOT_ATTAINED
```

`pooled_oof` is **not** silently omitted. All 48 of its rows carry
`evaluation_role = not_applicable` and the documented reason, and it is represented as
**`NOT_APPLICABLE_FOR_ED2_INFERENCE`**.

**The adjudication was made against the literal document, not by assumption:**

1. **Every step of App. E is defined on a count of distinct sales.** Step 3's percentile-group
   table is keyed on "Number of Observations"; §E.3's R6 and R7 both take
   *"N is the number of sales ratios in the sample"*. No weight appears anywhere in App. E.
2. **App. D.2 is a rank-based order statistic.** *"Array the ratios in ascending order and
   rank accordingly"*; `j = z√n/2` (+0.5 for even n), rounded up; *"count up and down the
   array from the median"*. The limits are literally array elements at integer ranks, so the
   procedure needs an integer `n` and integer ranks.
3. **The draft does define a weighted interval — for the mean only.** §D.3, immediately
   after D.2, is *"Weighted Mean Confidence Interval"* and builds an **effective sample
   size** from the weights. So the draft is demonstrably capable of specifying weighted
   inference and does so **only** for the mean. The string *"weighted median"* appears
   **nowhere** in the 103-page document, and *"effective sample size"* appears **only**
   inside that weighted-mean section.
4. **The draft never addresses duplicate observations**, repeat sales, or pooled samples.

The D3 row-balanced pooled-OOF sample has 151,153 appearances over 130,165 unique rows
(20,988 rows appearing twice, max multiplicity 2), with `w_ik = 1/m_i`. Applying D.2
literally would require choosing an `n` — 151,153 appearances double-counts 20,988 rows and
is not an IID sample, while the weight total 130,165 does not index the 151,153-element
array — and then inverting a weighted quantile to integer ranks. **Every resolution requires
inventing a weighted-rank median CI the draft does not define.**

Therefore: **ED2 inference for the seven folds, heldout and forward_2025 only; pooled_oof
explicitly not applicable, reason recorded; no synthetic weighted-rank extension.**
**No weighted variant was invented.** A test scans the P1 code for `weighted_median`,
`weighted_rank`, `weighted_quantile`, `effective_n` and similar and asserts none exists.

> Naming caution: the project's **D3** row-balanced construction and the draft's
> **§D.3** weighted-mean CI are unrelated. Do not conflate them.

### 3.2 Results — the Step-5 split is preserved

| stage | outcome | cells |
|---|---|---|
| Step 5 | `step5_within_pm10_stop` (\|VEI\| ≤ 10) | **117** |
| Step 5 | `step5_outside_pm10_escalate` | 279 |
| Step 6 | `ci_no_overlap_escalate` | 279 |
| Step 6 | `ci_overlap_stop` | **0** |
| Step 7 | `reject_null` | 228 |
| Step 7 | `fail_to_reject_null` | 51 |

The **117** Step-5 stops are preserved exactly; validation identified no implementation
error. Every gate was evaluated from the cell's own computed VEI — no outcome was
anticipated from an existing reported value.

**Standards-facing blocks only** (heldout + forward_2025, 88 cells):

| | cells |
|---|---|
| stop at Step 5, within ±10% | 25 |
| escalated past Step 5 | 63 |
| → `reject_null` | 52 |
| → `fail_to_reject_null` at Step 7 | **11** |
| stopped at Step 6 on CI overlap | 0 |

VEI point estimates span **−29.79 to +14.77**.

**This is the substantive value of the inferential layer.** A point estimate outside ±10%
does not by itself establish unacceptable vertical inequity: in **11 of 63** standards-facing
escalations the draft's *own* significance test fails to reject. Conversely, no cell stopped
at Step 6 — with ~3,800 observations per decile the median CIs are narrow enough that
first/last overlap essentially never occurs, so the discriminating step is Step 7's 10%
threshold, not CI overlap.

### 3.3 Fidelity

- Step 5 reproduces the canonical `vei()` **exactly**: max `|Δ| = 0.0`.
- Frozen artifact reconciliation: **234** comparisons, max `|Δ| = 7.11e-15`.
- App. D.2 `j` and both ranks re-derived independently from the verbatim rule and matched
  on all 396 cells; CI limits confirmed to be true order statistics (bitwise, on the
  parquet twin) for the reference cells × both OOS blocks.
- D.2 applied only where the draft says it applies: every percentile group has n > 30
  (smallest group 520). No rank was ever clamped.
- The deterministic bootstrap is retained **only** in separately labelled `*_sensitivity`
  columns, populated only for heldout and forward_2025, and never enters a decision.

### 3.4 Documented interpretations and deviations

Recorded in full in `configs/ed2_vei_procedure.json`; three are load-bearing.

- **ED2-I-1** — for even `n`, "count up and down the array from the median" does not fix the
  anchor ranks and the draft gives no worked large-sample example. Counting outward from the
  two central order statistics is used; it reproduces the normal-approximation interval at
  n = 100 → (40, 61) and n = 101 → (41, 61).
- **D-P1-5** — App. E Step 2 **prints** `Proxy = (0.50*SP) + (AV/Median Ratio)`, omitting the
  `0.50` on the AV term, while its own prose one line above says "equal weight". The
  equal-weight form is implemented — it is the 2013 Standard's formula, the draft's stated
  prose, and what the frozen executed code computes. **The manuscript already discloses this
  at `paper_v17_option1.tex:565`.**
- **D-P1-6** — percentile groups come from `numpy.array_split` over the proxy-sorted index,
  which is neither R6 nor R7. Required for Step 5 to reconcile with the frozen `VEI__*`
  values. Where n divides by 10 all three coincide; otherwise boundaries shift by ≤ 1
  observation.

---

## 4. Task 4 — Duan smearing sensitivity

### 4.1 The frozen estimator (not rerun; hash-validated)

`configs/smearing_estimator_frozen.json`, sha256 `6e00c9fa14b3fa5813be00691d48b36fb6917110b6cd8faa0ff9201fbd36bf18`,
frozen at `2026-09-07T02:44:25Z`.

**Sign.** The repository's residual convention is `e = y_pred_log - y_true_log`
(`utils/motivation_utils.py:1554`). Duan's factor uses the **opposite-sign** log error:

```
u_ik = y_true_log - y_pred_log = -e_ik
s    = Σ w_ik·exp(u_ik) / Σ w_ik            applied as  ŷ → s·ŷ,  i.e. y_pred_log → y_pred_log + log s
```

`exp(e)` is wrong and is asserted absent from the source. Verified: the code computes
`exp(u)`, the applied shift is `+log(s)`, and `np.exp(e)` does not appear.

A symmetric error distribution cannot detect a sign flip, since `E[exp(u)] = E[exp(-u)]`,
so the sign tests pair a lognormal magnitude benchmark with an **exact asymmetric
two-point** case. Independent corroboration from the estimates themselves: `s ≥ exp(mean u)`
by Jensen for every realization, and `s > 1` wherever `mean u > 0`.

**Weighting.** D3 row-balanced, `w_ik = 1/m_i`, so each unique development sale row carries
total weight exactly one. Identities asserted: `109,177 + 20,988 = 130,165` unique rows and
`109,177 + 2×20,988 = 151,153` appearances, max multiplicity 2. The naive duplicate-weighted
factor is reported as `s_naive_pooled` and **never applied**; max `|s − s_naive| = 3.94e-03`.
Weights sum to `130,165` and each unique row's weight sums to 1 within `1e-12`.

**Development-only, frozen before any OOS application.** Estimated from fold_1…fold_7
validation blocks only; `oos_used_in_estimation = False` on all 43 realizations. The
chronology is clean: the estimator was frozen at 22:44:25 local, before PRB (22:54), before
the smearing estimate itself (22:58) and before the VEI run that first reads heldout and
forward_2025 (23:01). The two artifacts produced *before* the freeze — the display-set
freeze and the dCor audit — read no observation-level data at all: the dCor task is a pure
source-code audit, and the display-set freeze reads only P0 tables and config maps.

**Magnitudes.** 43 realizations, `s ∈ [1.042931, 1.142692]`, median `1.060201` — a **+4.3%
to +14.3%** price-scale level shift.

### 4.2 The apply stage and the invariance audit

`--mode apply` was run once, using the frozen `s` unchanged (`s_recomputed_in_apply: false`;
the applied value is asserted equal to the frozen estimate for every realization).

**Coverage: 430 cells** = 43 realizations × 10 evaluation blocks; **6,450** metric
comparisons, of which **5,160** carry an invariance or scale assertion. **0 FLAGGED.**

| class | metrics | max relative deviation |
|---|---|---|
| `scales_by_s` | median / mean / weighted-mean ratio | **1.26e-15** |
| `invariant` | COD, COV, PRD, PRB, VEI, MKI, β_log, Cov(e, log P), dCor(e,y) | **1.00e-11** |
| `moves` | R²_price, MAE_price, MAPE | not asserted — moves by design |

Per-metric worst case: `dCor_e_y` at `1.00e-11`, then VEI `2.85e-13`, β_log `2.34e-13`,
Cov `2.34e-13`, PRB `3.69e-14`, all others `≤ 2.6e-15`. dCor is the loosest because it is
computed by an approximate-order AVL algorithm on 10⁴–10⁵ points, not because any dependence
changed; the analytic result is exact — a translation of `e` leaves every pairwise distance
`|e_i − e_j|` unchanged.

**The audit is not vacuous.** The level metrics really do move (every one by > 1e-6, and by
exactly `s`), and the price-scale accuracy metrics move materially: max |Δ| of `0.0577` for
R²_price, `20,809` for MAE_price, `0.0581` for MAPE.

**Delta_NL** (34 s per call, ~8 h for full coverage) was audited on a **pre-declared** subset —
the two included reference cells plus the realization with the **largest** `s`, the strongest
stress case — across all ten blocks: 30 cells, max relative deviation **1.81e-15**, all OK.
Its invariance is also exact by construction: `e → e + log s`, and both the affine and the
cubic-spline head carry an intercept while `Var(e)` is translation invariant, so the
cross-fitted OOF residuals are unchanged.

**The fast path is not a re-implementation.** Each metric is produced by the same canonical
function `compute_taxation_metrics` calls, and on the Delta_NL subset the fast path was
compared against the frozen P0 suite `p0_4_centered_spread.metrics_from`: **450 comparisons,
max relative difference `0.0`** — exact agreement.

**ED2 verdict stability.** The full App. E decision path was re-run on both arms:
**387/387** ED2-applicable cells keep an identical Step-5 gate, Step-6 result, Step-7 outcome
and verdict string; max VEI relative deviation `2.85e-13`. The 43 `pooled_oof` cells are
carried as `NOT_APPLICABLE_FOR_ED2_INFERENCE` with **the same reason string as Task 3**
(one source of truth, asserted) — applying the CI test there would have contradicted §3.1.
The VEI *point estimate* is still audited on pooled_oof, since that is a descriptive
statistic rather than the CI-based test.

`RMSE_log` is **not** claimed invariant — it is not invariant to adding a constant, and it is
simply not recomputed, because smearing is a post-exponentiation price-scale retransformation
that does not alter the canonical log prediction.

**Role: sensitivity only.** The canonical model is unaltered, nothing is retuned, and no
smeared value replaces a headline number. For forward_2025 the development-estimated `s` is
applied unchanged although that block's fitting set (382,897 production rows) has no
out-of-fold analogue — an **explicit assumption and stated limitation**, not a validated
property.

---

## 5. Reproducing 153/153 at the tag

Verified in a **detached git worktree** at the tag, outside the P1 tree. A fresh checkout
carries only tracked git *content*, so three classes of state the P0 suite legitimately
asserts had to be re-attached from the main working tree, where P0 actually ran. **No P0
test was modified, skipped or weakened, and no file content was altered.**

1. **Gitignored derived artifacts** — `output/paper_v6_preselection_994`,
   `output/paper_v12_lower_rho_extension_994_v2`, `output/p0_major_revision_validation`.
2. **Gitignored source data** — `data/{ATTOM,berry_cmf,CCAO,CensusData,dewey-downloads}`.
   Attached as *real directories with symlinked children*: a bare symlink is not a directory,
   so the existing `data/<X>/` gitignore patterns would not have matched it and the untracked
   path would have tripped the Stage-1 write-location guard.
3. **File mode and mtimes, which git does not record.**
   `APPROVED_EXECUTION_PLAN.md` is `0444` in the main tree and `0664` after checkout; three
   tests assert it is read-only. And
   `test_source_equivalence_precedes_reproduction_interpretation` asserts a **relative mtime
   ordering**, which no checkout can satisfy because `git worktree add` stamps every file
   with the checkout time.

Both measurements are reported, and the second is the tag's certification:

| checkout | result |
|---|---|
| content only | 152 passed, 1 failed — the mtime-ordering test |
| + mode and mtimes reconstituted (309 tracked P0 files) | **153 passed, 0 failed, 0 skipped** |

Recorded in `provenance/p0_suite_at_tag.json`. Suite sizes: 28 + 36 + 36 + 29 + 24 = 153.

> This is a genuine reproducibility limit of the P0 suite worth stating: one of its 153
> assertions is a property of the original run's filesystem metadata, not of the committed
> content, and is therefore not reproducible from the repository alone.

---

## 6. Why the P1 branch reports 152/153 — and why that is not a P0 change

`test_g2_assertions.py:325::test_only_gitignore_modified_outside_p0` runs
`git diff --name-only 2732e653~1 HEAD` and asserts the only file changed outside the P0 area
is `.gitignore`. It is **HEAD-relative**, so **any** additive commit anywhere in the
repository breaks it; a sibling directory does not help. Its purpose was to stop Stage-1/2/3
from writing outside the P0 area, and it is doing exactly that — registering that P1 exists.

**This single expected failure is recorded, not suppressed.** P0 was not modified to make the
branch report 153/153. P0 content is provably unchanged on the branch:

- `git diff p0-major-revision-final-20260907 -- analysis/p0_major_revision_validation/` → empty
- the final-gate (Stage-3B / G5b) hash index → 28/28 byte-exact
- the two early-stage indexes → the same 8 mismatches the tag already had, no new one
- all protected paths → clean
- the only files changed since the tag are the P1 directory and `.gitignore`

---

## 7. Artifact inventory

Committed evidence under `analysis/p1_inferential_reporting/`:

| area | files |
|---|---|
| tables | `dcor_estimator_audit.csv`, `dcor_estimator_facts.json`, `prb_inference.csv` (+summary), `vei_significance.csv` (+summary), `smearing_factor_provenance.csv`, `smearing_apply_invariance.csv`, `smearing_apply_ed2_stability.csv`, `smearing_apply_delta_nl_subset.csv`, `smearing_apply_fastpath_reconciliation.csv`, `smearing_apply_summary.json` |
| configs | `display_set_frozen.json` (+hash), `ed2_vei_procedure.json`, `smearing_estimator_frozen.json` (+hash) |
| provenance | `p1_artifact_hashes.json`, `p1_headline_numbers.json`, `p0_suite_at_tag.json`, `p0_suite_on_p1_branch.json`, `ed2_source_manifest.json`, `DEVIATIONS_P1.md` |
| code | `p1_common.py`, `p1_0`…`p1_4`, `p1_9_headline_numbers.py`, `p1_9_freeze_evidence.py` |
| tests | `test_p1_prb_assertions.py`, `test_smearing_sign.py`, `test_p1_vei_assertions.py`, `test_p1_smearing_apply_assertions.py`, `test_p1_headline_numbers.py` |

Parquet twins sit beside the CSVs and are gitignored by design, mirroring P0. **The
committed CSVs are text serialisations: they round-trip float64 to within 1 ULP (worst
observed absolute deviation `3.55e-15`, on `VEI_step5`). The parquet twins are bit-exact.
Any bitwise claim should reference the parquet, not the CSV** — this is asserted rather
than assumed.

The IAAO ED2 PDF is cached at a gitignored path and is **not redistributed**; only its hash,
size, URL and printed title are committed.

**Environment unchanged:** numpy 1.26.4, pandas 2.3.1, scipy 1.13.1, scikit-learn 1.6.1,
pyarrow 14.0.1, lightgbm 4.6.0, dcor 0.6, Python 3.9.19. `pypdf 5.9.0` was installed into an
isolated scratch `--target` directory purely to read the ED2 PDF and is not importable from
`fairness_env`.
