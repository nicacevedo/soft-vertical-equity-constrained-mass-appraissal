# Centered-Spread Post-hoc Comparator — Stage 2 (Gate G3)

Theorem-matched primary map, evaluated for the two frozen post-hoc bases:

* **PRIMARY** — **C**, `Custom-objective rho=0 origin`
* **SECONDARY practical** — **A**, `Ordinary LightGBM (standard raw-label native)`
* **B** — `Centered-label native L2 (initialization-aligned)`, decomposition control, **no full path**

$$f_b(x) = \bar y_T + b\,(f_0(x) - \bar y_T)$$

with `ybar_T` the mean log target of the **fitting block** for the regime (fold-k training mean for
fold k; development pool for held-out; production 2016-2024 for 2025).

**No A or C model was refit.** The Stage-1.5 cached predictions were used, with all
20 input prediction arrays hash-verified against the Stage-1.5 provenance before use.

The formal matched-beta comparison is **NOT** performed here; §7 below is explicitly descriptive.

---

## 1. Development calibration

`b_star_train = Var_T(y)/Cov_T(f0,y)` is retained **only as an in-sample theory diagnostic** and
was **not** used to define any endpoint, root or grid bound. Reason: the 994-tree learner is
substantially less regressive in sample than out of sample, so the in-sample root understates the
rescaling the data actually require.

| quantity | C | A |
|---|---|---|
| `beta_log_train` (in-sample, 9 blocks) | -0.084934 … -0.061383 | -0.085058 … -0.061710 |
| **`b_star_train`** *(diagnostic only)* | 1.065397 … 1.092817 | 1.065769 … 1.092966 |
| `beta_CVmean` at `b=1` (D1) | -0.13827123 | -0.13820396 |
| **`b_zero_cvmean`** (D1 PRIMARY root) | **1.160457945845** | **1.160367364114** |
| `beta_pooled_oof` at `b=1` (D2) | -0.14130749 | -0.14148074 |
| **`b_zero_pooled_oof`** (D2 sensitivity root) | **1.167026639895** | **1.167265678969** |

Roots computed as specified, from development fold predictions only:

* D1: `beta_k(b) = b * Cov_Vk(f0,y)/Var_Vk(y) - 1`, `beta_CVmean(b) = (1/7) sum_k beta_k(b)`,
  hence `b_zero_cvmean = 1 / mean_k[ Cov_Vk(f0,y)/Var_Vk(y) ]`.
* D2: closed-form root of the pooled coordinate after applying fold-specific training centers,
  `b = (V - P)/(Q - P)` with `V = Var_pooled(y)`, `P = mean(ybar_T(k(i)) c)`, `Q = mean(f0 c)`.
  `P` is non-zero because the fold centers correlate with `c`.

**Why the empirical roots exceed `b_star_train` by so much.** `b_star_train` ~ 1.065–1.093 while the
development roots are ~1.160–1.167. The in-sample residual–price association is only about
−0.06 to −0.09 whereas out of sample it is about −0.14 to −0.16, so an in-sample calibration would
under-correct by roughly a factor of two in `(b-1)`. This is precisely why the post-G2 refinement
moved the calibration to development out-of-sample predictions.

**Independent validation.** A's recomputed `b_zero_pooled_oof` is
**1.1672656789694**, matching the historical native `b* = 1.1672656789694134`
to 13 significant digits — while C's differs (1.167026639895). The historical
value was **not** reused; A-specific and C-specific roots were recomputed from the current verified
cached predictions and are genuinely different.

### Path upper bound

`b_max = 1 + 1.25 * (max(all four development roots) - 1)` — a true **25 % overshoot in the
adjustment `(b-1)`**, not `1.25 * b_star`.

`b_ref_max = 1.167265678969` (A's D2 root) ⇒ **`b_max = 1.209082098712`**.

Development-only coverage verification (all requirements met = **True**):

| requirement | C | A |
|---|---|---|
| `beta_CVmean` at `b=1` | -0.138271 | -0.138204 |
| `beta_CVmean` at `b_max` | +0.041901 | +0.041982 |
| reaches `beta_log = 0` | True (b=1.160458) | True (b=1.160367) |
| reaches Direct upper common support | True (b=1.069186) | True (b=1.069103) |
| EXT target −0.06 within grid | True (b=1.090830) | True (b=1.090745) |
| EXT target −0.03 within grid | True (b=1.125644) | True (b=1.125556) |
| EXT target 0 within grid | True | True |

Direct CV-mean `beta_log` range -0.139423 … -0.078651;
Surrogate -0.139298 … -0.018609.
Three-way common support on D1 = **[-0.138204, -0.078651]**,
whose lower endpoint is set by **post-hoc at b=1** — so no extension
below `b = 1` was required, and none was made.

### Grid

121 equally spaced base values from 1 to b_max (LINEAR in b, because beta_log is exactly linear in b and the interval is narrow), then force-add the exact development anchors, sort and deduplicate. Slightly more than 121 values is accepted because exact scientific anchors outrank row count.

`n_grid = 124` values on `[1, 1.209082099]`, frozen at **2026-09-06T14:34:26.405373Z**,
file sha256 `0a680e7ff8182b3ec59ddd40ea2254cc…`. The five exact anchors are `b_1`,
`b_zero_cvmean_C`, `b_zero_pooled_oof_C`, `b_zero_cvmean_A`, `b_zero_pooled_oof_A`, plus `b_max`.

**Ordering guarantee.** The `roots` mode reads *only* fold prediction files — the exact list is
recorded in `b_grid_frozen.json` under `files_read_during_root_construction`, and
`no_heldout_or_2025_outcome_read = True`. The `path` mode
re-validates the grid file hash before it runs and aborts if the file changed. The grid was not
modified after any out-of-time outcome was read.

---

## 2. Path QC

### `b = 1` reproduces `f0` bitwise

An explicit `b == 1.0` fast path returns the cached array object unchanged, so the check is exact
rather than round-off dependent:

* bitwise identical on **all 20** (reference x evaluation) blocks: **True**
* fast path returns the same object (no arithmetic at all): **True**
* max absolute deviation: **0.0**

### `beta_log` is numerically linear in `b`

Theory says `beta_V(b) = b * Cov_V(f0,y)/Var_V(y) - 1` exactly.

* max |observed − closed form| over every fold/held-out/2025 path: **4.549e-14**
* max |residual from a fitted straight line| over all 20 paths: **4.718e-16**
* fitted slope vs closed-form slope, max absolute difference: **3.697e-14**
* every path monotone increasing in `b`: **True**

### Roots zero their own coordinate

| reference | coordinate | b_zero | achieved beta_log | |beta| from zero |
|---|---|---|---|---|
| C | D1_CVmean | 1.160457945845 | -1.001e-16 | 1.001e-16 |
| C | D2_pooled_oof | 1.167026639895 | -3.625e-14 | 3.625e-14 |
| A | D1_CVmean | 1.160367364114 | +5.848e-17 | 5.848e-17 |
| A | D2_pooled_oof | 1.167265678969 | -3.612e-14 | 3.612e-14 |

Both D1 roots land within **1.00e-16** of zero and both D2 roots within
**3.63e-14**, against a target of 1e-12.

### Historical A-recalibration QC

The repository's historical centered-recalibration artifact (`V6/final_local_results/
recalibration_path.csv`, 51 b-values, native `f0`, same `ybar_T`-centered map) is used as an
independent QC reference — not as a source of grid or endpoint.

* worst **relative** metric difference over all 102 shared (b, evaluation) points and 17 metrics:
  **1.575e-11**
* `ybar_T` agreement: max absolute difference **0.000e+00**

The new A-centered implementation reproduces the historical artifact to ~1e-11 relative, i.e. to
float64 accumulation order. The old endpoint and grid were **not** adopted.

---

## 3. Centering-choice sensitivity — immaterial, frozen

Analytically, for the same `b`,

$$f_b^{\bar y}(x) - f_b^{\bar f_0}(x) = b\,(\bar f_{0,T} - \bar y_T),$$

a **constant** in `x`. Because the canonical `beta_log` convention centres `c` on the evaluation
sample, a constant log shift cannot change `beta_log`, `Cov_log_residual_log_price`, `COD`, `COV`,
`PRD`, `PRB`, `MKI`, `VEI`, `Delta_NL` or `dCor` at all; only level-sensitive metrics can move.

Gate G2 established `max |f0bar_T - ybar_T| <= 8.36e-06`, so the maximum possible shift over the
entire Stage-2 grid is **6.266e-06** in log space
(≈ 0.00063 % in price). A full duplicate
121-point path was therefore **not** generated; the sensitivity was evaluated at four anchors
(`b_1`, `b_direct_upper_common_support`, `b_zero_cvmean`, `b_max`) for A and C on CV mean,
held-out and 2025.

Maximum absolute metric movement over all anchors and regimes:

| metric | max abs difference | max relative |
|---|---|---|
| `MAE_price` | 1.54443 | 1.369e-05 |
| `R2_price` | 1.02988e-05 | 1.541e-05 |
| `weighted_mean_ratio` | 7.001e-06 | 6.266e-06 |
| `mean_ratio` | 6.78059e-06 | 6.266e-06 |
| `median_ratio` | 6.49913e-06 | 6.266e-06 |
| `MAPE` | 2.28229e-06 | 9.575e-06 |
| `RMSE_log` | 6.37272e-07 | 2.235e-06 |
| `dCor_e_y` | 6.91724e-13 | 2.365e-12 |

The largest movement anywhere is `MAE_price` = **$1.54** on an MAE of
$59k–$113k, i.e. **1.37e-05** relative.
`beta_log` moves by at most **2.78e-17** — machine epsilon, exactly
as the analytic argument requires.

> **Frozen conclusion: the centering-choice sensitivity is immaterial.** The `ybar_T`-centered
> theorem-matched map is used for the full path, and no duplicate `f0bar`-centered path is
> scheduled.

---

## 4. New repository finding — fold-6 / fold-7 validation overlap

While constructing the D2 pooled coordinate, the seven validation blocks were found **not** to be
disjoint: fold 6 and fold 7 overlap by **20,988 rows**
(151,153 concatenated vs 130,165 unique, i.e.
13.89 % duplicated; 63.4 % of fold 6's block and 60.9 % of fold 7's).

Cause: expanding-window rolling origin with a newest-10% validation rule: the fold-6 and fold-7 origins are only ~4 months apart (2023-07-01 -> 2023-11-09) while 10% of the development pool spans ~1 year, so the two validation blocks overlap.

Effect: the concatenated pooled-OOF coordinate double-counts the overlapping rows; each individual prediction is still genuinely out-of-fold (no overlapping row is in the training block of the model that predicted it).

the historical solve_b_star_validation_neutral used the same concatenation with n_oof=151153, so the historical b*=1.1672656789694134 inherits the same double-counting; our recomputed A value matches it to 13 significant digits, confirming faithful reproduction

**D1, the paper's primary CV coordinate, is unaffected** (True) because it
is an equal-weight mean of per-fold values, each computed on its own block. Only the D2 sensitivity
coordinate double-counts. Audited in `tables/cv_fold_validation_overlap_audit.csv` and
`tables/cv_fold_validation_overlap_summary.json`.

Consequence for the frozen `Delta_NL` estimator: it requires unique identifiers within a split, so
for the D2 sample only, a composite deterministic identifier `"fold|row_id"` is supplied. It remains
a function of observation identity alone and independent of the model and predictions, which is what
the frozen specification requires. Every other evaluation uses the plain `row_id`.

---

## 5. C-PRIMARY centered-spread results

### Equal-weight seven-fold CV mean

| anchor | b | R2_price | MAE_price | RMSE_log | COD | PRD | PRB | MKI | VEI | beta_log | Delta_NL | dCor_e_y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `b_1` | 1.000000 | 0.89900 | 58,951 | 0.29065 | 22.010 | 1.0687 | -0.08680 | 0.93387 | -26.762 | -0.13827 | 0.10021 | 0.38698 |
| `b_zero_cvmean_C` | 1.160458 | 0.78088 | 75,810 | 0.31497 | 22.961 | 0.9801 | +0.03428 | 1.07980 | +17.093 | -0.00000 | 0.11492 | 0.25974 |
| `b_zero_pooled_oof_C` | 1.167027 | 0.76732 | 77,260 | 0.31684 | 23.082 | 0.9766 | +0.03882 | 1.08572 | +18.868 | +0.00566 | 0.11491 | 0.26115 |
| `b_max` | 1.209082 | 0.65733 | 87,663 | 0.33014 | 23.965 | 0.9545 | +0.06729 | 1.12343 | +30.305 | +0.04190 | 0.11396 | 0.27992 |

*(CV SD columns are in `tables/centered_spread_cv_summary.csv`; they are descriptive spread over
strictly nested chronological windows, not an IID sampling distribution.)*

### Later held-out block (n = 38,290)

| anchor | b | R2_price | MAE_price | RMSE_log | COD | PRD | PRB | MKI | VEI | beta_log | Delta_NL | dCor_e_y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `b_1` | 1.000000 | 0.89281 | 75,976 | 0.28940 | 21.599 | 1.0690 | -0.08845 | 0.92305 | -26.167 | -0.14738 | 0.11645 | 0.38225 |
| `b_zero_cvmean_C` | 1.160458 | 0.80628 | 87,182 | 0.30711 | 22.587 | 0.9871 | +0.03260 | 1.07258 | +14.770 | -0.01057 | 0.13430 | 0.25175 |
| `b_zero_pooled_oof_C` | 1.167027 | 0.79408 | 88,689 | 0.30867 | 22.691 | 0.9839 | +0.03716 | 1.07866 | +16.438 | -0.00497 | 0.13436 | 0.25260 |
| `b_max` | 1.209082 | 0.69257 | 99,978 | 0.32002 | 23.495 | 0.9630 | +0.06594 | 1.11753 | +26.815 | +0.03089 | 0.13376 | 0.26813 |


### 2025 forward block (n = 26,641)

| anchor | b | R2_price | MAE_price | RMSE_log | COD | PRD | PRB | MKI | VEI | beta_log | Delta_NL | dCor_e_y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `b_1` | 1.000000 | 0.90385 | 79,139 | 0.27907 | 21.229 | 1.0775 | -0.10284 | 0.90934 | -28.407 | -0.16120 | 0.12118 | 0.41684 |
| `b_zero_cvmean_C` | 1.160458 | 0.79906 | 96,243 | 0.29392 | 21.913 | 0.9934 | +0.01872 | 1.05812 | +11.374 | -0.02661 | 0.14544 | 0.25418 |
| `b_zero_pooled_oof_C` | 1.167027 | 0.78498 | 98,248 | 0.29546 | 22.009 | 0.9900 | +0.02332 | 1.06418 | +12.974 | -0.02110 | 0.14567 | 0.25361 |
| `b_max` | 1.209082 | 0.66823 | 112,802 | 0.30690 | 22.754 | 0.9685 | +0.05238 | 1.10287 | +23.738 | +0.01418 | 0.14589 | 0.26190 |


### Reading

* **The predictive cost of post-hoc rescaling is large and monotone.** Driving development
  `beta_log` to zero (`b = 1.160458`) costs, on held-out,
  `R2_price` 0.89281 → 0.80628
  and `MAE_price` $75,976 → $87,182.
  On 2025 the same move costs `R2_price` 0.90385 → 0.79906.
* **`Delta_NL` rises** along the path (0.11645 → 0.13430
  held-out): removing the first-order slope by a global rescaling *increases* non-affine
  conditional-mean structure.
* **`dCor` falls substantially** (0.38225 → 0.25175
  held-out), then turns back up beyond the root (0.26813 at `b_max`).
* **Assessor-facing metrics overshoot into progressivity.** At the root, held-out `PRD`
  1.0690 → 0.9871, `MKI`
  0.92305 → 1.07258, `VEI`
  -26.167 → +14.770. `COD`
  worsens (21.599 → 22.587).
* **First-order neutrality does not flatten the conditional ratio profile.** In the 30-bin
  price profile on held-out, the cheapest bin's median ratio moves 1.709 → 1.474 and the most
  expensive 0.863 → 1.163: the profile is compressed and the level lifted, but a large
  low-value-to-high-value gap remains. Profiles with 90 % bootstrap CIs are in
  `tables/centered_spread_ratio_profiles.csv`.

---

## 6. A-SECONDARY practical comparator

Same map applied to standard native LightGBM — the "could a practitioner just post-process?"
question. Descriptive wording only; differences relative to A are never attributed to rho.

### Held-out

| anchor | b | R2_price | MAE_price | RMSE_log | COD | PRD | PRB | MKI | VEI | beta_log | Delta_NL | dCor_e_y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `b_1` | 1.000000 | 0.89423 | 75,655 | 0.28871 | 21.627 | 1.0695 | -0.09076 | 0.92295 | -26.460 | -0.14960 | 0.11866 | 0.38709 |
| `b_zero_cvmean_A` | 1.160367 | 0.80244 | 87,658 | 0.30599 | 22.613 | 0.9875 | +0.03081 | 1.07281 | +14.340 | -0.01322 | 0.13748 | 0.25309 |
| `b_zero_pooled_oof_A` | 1.167266 | 0.78920 | 89,277 | 0.30761 | 22.730 | 0.9840 | +0.03565 | 1.07922 | +16.185 | -0.00736 | 0.13757 | 0.25374 |
| `b_max` | 1.209082 | 0.68521 | 100,604 | 0.31882 | 23.526 | 0.9633 | +0.06445 | 1.11798 | +27.326 | +0.02820 | 0.13708 | 0.26783 |


### 2025 forward

| anchor | b | R2_price | MAE_price | RMSE_log | COD | PRD | PRB | MKI | VEI | beta_log | Delta_NL | dCor_e_y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `b_1` | 1.000000 | 0.90438 | 78,484 | 0.27819 | 21.266 | 1.0792 | -0.10604 | 0.90654 | -28.577 | -0.16408 | 0.12117 | 0.42204 |
| `b_zero_cvmean_A` | 1.160367 | 0.79881 | 96,604 | 0.29323 | 21.891 | 0.9953 | +0.01568 | 1.05487 | +11.832 | -0.03003 | 0.14639 | 0.25642 |
| `b_zero_pooled_oof_A` | 1.167266 | 0.78398 | 98,740 | 0.29487 | 21.987 | 0.9917 | +0.02051 | 1.06122 | +13.474 | -0.02426 | 0.14666 | 0.25555 |
| `b_max` | 1.209082 | 0.66763 | 113,542 | 0.30632 | 22.701 | 0.9704 | +0.04937 | 1.09960 | +23.986 | +0.01069 | 0.14708 | 0.26216 |


The A path tracks the C path very closely — the two references differ at `b=1` by the execution-path
artifact characterised at Gate G2, and rescaling does not amplify it. A practitioner
post-processing standard LightGBM would face essentially the same accuracy/equity trade-off as one
post-processing the custom rho=0 origin.

---

## 7. Descriptive juxtaposition with the retrained families — NOT the matched-beta test

**The formal matched-beta comparison is not performed in Stage 2 and is not authorized here.** It
must be matched on **development** achieved `beta_log`; the lookups below are keyed on **held-out**
`beta_log` and are therefore descriptive only, offered to show that the Stage-3 test is worth doing.

| at held-out `beta_log` ≈ | family | R2_price | MAE_price | Delta_NL | dCor |
|---|---|---|---|---|---|
| **-0.07941** | Direct (rho=100) | 0.86868 | 84,918 | 0.11535 | 0.26493 |
| | C-posthoc (b=1.080148, beta=-0.07905) | 0.89242 | 74,930 | 0.12946 | 0.28404 |
| **+0.00084** | Surrogate (rho=75.4312) | 0.88908 | 81,930 | 0.12434 | 0.26804 |
| | C-posthoc (b=1.174235, beta=+0.00118) | 0.77962 | 90,429 | 0.13438 | 0.25408 |

**The accuracy comparison appears to cross over with correction strength.** At mild correction
(held-out `beta_log` ≈ −0.079, Direct's most-corrected point) the post-hoc rescaling is the *more*
accurate of the two; at strong correction (`beta_log` ≈ 0, Surrogate's near-neutral point) the
retrained Surrogate is dramatically more accurate than post-hoc rescaling. In both regimes the
retrained families carry lower `Delta_NL` than post-hoc at comparable first-order correction.

This is exactly the structure the Stage-3 matched-beta test is designed to adjudicate properly, on
the development coordinate, with the frozen common support. Nothing here is a finding about the
manuscript's claims; no manuscript text was touched.

---

## 8. Frozen future matched-beta convention (not executed)

| role | families | matched on |
|---|---|---|
| **PRIMARY CORE** | Direct vs Surrogate vs **C-posthoc** | DEVELOPMENT achieved `beta_log` |
| SECONDARY practical | A-posthoc | separate panel/table |
| — | B | no post-hoc path |

A-posthoc must not determine the three-way common support, must not determine CORE target
selection, and must not replace C-posthoc in the primary mechanistic test. Frozen in
`configs/posthoc_comparator_convention.yaml`; `executed_in_stage_2 = {conv['future_matched_beta_convention']['executed_in_stage_2']}`.

---

## 9. Artifacts

| artifact | content |
|---|---|
| `configs/b_grid_frozen.json` + `b_grid_frozen_hash.json` | construction rule, {g['n_grid']} b values, anchors, A/C roots, coverage verification, hash |
| `configs/posthoc_comparator_convention.yaml` | frozen roles, calibration rules, future matched-beta convention |
| `tables/posthoc_development_roots.csv` | per-fold `Cov_Vk`, `Var_Vk`, `R_k`, D1/D2 roots for A and C |
| `tables/centered_spread_path.csv` | {len(path):,} rows: 2 references x 10 evaluations x {g['n_grid']} b, full canonical suite |
| `tables/centered_spread_cv_summary.csv` | equal-weight CV mean and descriptive CV SD per b |
| `tables/centered_spread_pooled_oof.csv` | D2 pooled coordinate path |
| `tables/centered_spread_b1_qc.csv` | bitwise `b=1` verification |
| `tables/centered_spread_linearity.csv` | closed-form vs observed `beta_log`, slopes, residuals |
| `tables/centered_spread_root_verification.csv` | achieved `beta_log` at each root |
| `tables/centered_spread_centering_sensitivity.csv` | anchor-level `ybar` vs `f0bar` comparison |
| `tables/centered_spread_existing_qc.csv` | historical A-recalibration reproduction |
| `tables/centered_spread_ratio_profiles.csv` | IAAO proxy-decile (90 % CI) and 30-bin price profiles at anchors |
| `tables/cv_fold_validation_overlap_audit.csv` | fold-pair validation overlap audit |
| `tables/gate_g3_checks.json` | machine-readable Gate-G3 evidence |
