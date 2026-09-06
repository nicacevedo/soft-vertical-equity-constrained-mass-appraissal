# Zero-Control Evidence — Stage 1.5 (Gate G2)

Complete zero-reference evidence for the three frozen reference roles across the seven
rolling-origin folds, the equal-weight CV mean and SD, the later held-out block and the 2025
forward block.

Built from: the 27 canonical Stage-1.5 fits (A + B + C x 9 fitting blocks, **historical**
execution settings, no Track-P pins), the frozen 82-point path table, cached predictions, and the
frozen `Delta_NL` estimator (spec hash `e85069150b…`, unchanged).

**Not included by design:** PRB standard errors / t-values, VEI significance, smearing
sensitivity. Those remain later P1 extras and were not run.

## Reference roles (frozen — `configs/post_g1_reference_convention.yaml`)

| Cell | Forward display name | Role |
|---|---|---|
| **A** | Ordinary LightGBM (standard raw-label native) | assessor-facing / workflow benchmark |
| **B** | Centered-label native L2 (initialization-aligned) | implementation-decomposition control only |
| **C** | Custom-objective rho=0 origin | PRIMARY within-path penalty-isolating reference |

Cell B's Stage-1 label *"Parity-aligned native L2"* is retained **only** as the
metadata field `legacy_stage1_label`; it is not used as a display name in any new output, because
empirical parity with C failed. Frozen Stage-1 artifacts were not rewritten.

Cell C was fitted with **one** canonical implementation: `LGBCovPenalty (Direct rho=0)`.
Stage 1 established Direct-rho0 == Surrogate-rho0 bitwise inside the custom path (T1 at every
capacity and split), so refitting both was unnecessary.

---

## 1. Reproduction QC for the 27 fits

Where exact frozen counterparts exist, the new paired-evaluation predictions were compared under
the existing R1–R4 rubric.

* **Cell A**: 9/9 blocks reproduce the corresponding native frozen artifact at **R1 — exact**
  (`max|delta| = 0.0`, 100 % of rows bitwise identical). This covers all seven CV folds plus
  held-out and 2025 — a wider check than Stage 1, which sampled only the out-of-time blocks.
* **Cell C**: 9/9 blocks reproduce the corresponding custom rho=0 frozen artifact at **R1 — exact**.
* Tier counts: {"R1": 18}. No R2, R3 or R4 anywhere; no failure class required.
* **Cell B**: no per-row Stage-1 counterpart exists (the Stage-1 ladder stored aggregates only), so
  B was cross-checked against the Stage-1 capacity-F ladder. All **6/6** aggregate cross-checks
  reproduce the Stage-1 `mean` and `max` to within 1e-12
  (`qc_ladder_crosschecks_all_match = True`).
  **B is not required to equal C** and does not.

### Independent metric-level agreement with the frozen path table

200 metric comparisons between the newly computed A / C metrics and the frozen
`combined_path_table_v4_analysis_view.csv`:

| pair | max abs difference | mean abs difference | max relative difference |
|---|---|---|---|
| A vs frozen `LightGBM` | 3.553e-15 | 7.105e-17 | 1.6e-16 |
| C vs frozen `Direct rho=0` | 3.553e-15 | 3.553e-17 | 1.6e-16 |

Agreement is at float64 round-off. The Stage-1.5 pipeline reproduces the frozen artifacts both at
the prediction level (R1 exact) and at the metric level (machine precision).

---

## 2. Zero-control results

### 2.1 Equal-weight CV mean (seven rolling-origin folds)

| cell_id | R2_price | MAE_price | MAPE | RMSE_log | COD | PRD | PRB | MKI | VEI | beta_log | Delta_NL | dCor_e_y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.89974 | 58,865 | 0.21508 | 0.29040 | 21.992 | 1.0687 | -0.08675 | 0.93367 | -27.229 | -0.13820 | 0.09957 | 0.38750 |
| B | 0.89939 | 58,891 | 0.21512 | 0.29049 | 21.997 | 1.0689 | -0.08677 | 0.93328 | -27.081 | -0.13824 | 0.09930 | 0.38739 |
| C | 0.89900 | 58,951 | 0.21530 | 0.29065 | 22.010 | 1.0687 | -0.08680 | 0.93387 | -26.762 | -0.13827 | 0.10021 | 0.38698 |


### 2.2 Equal-weight CV standard deviation across folds

| cell_id | R2_price | MAE_price | MAPE | RMSE_log | COD | PRD | PRB | MKI | VEI | beta_log | Delta_NL | dCor_e_y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.00909 | 6,916 | 0.01709 | 0.01802 | 1.845 | 0.0101 | 0.01430 | 0.02059 | 7.230 | 0.02173 | 0.01082 | 0.03473 |
| B | 0.00919 | 6,947 | 0.01711 | 0.01801 | 1.844 | 0.0101 | 0.01428 | 0.02076 | 7.262 | 0.02167 | 0.01099 | 0.03470 |
| C | 0.00928 | 6,847 | 0.01670 | 0.01780 | 1.798 | 0.0099 | 0.01364 | 0.02017 | 6.864 | 0.02078 | 0.01056 | 0.03402 |


*(Descriptive spread across seven strictly nested chronological windows, not an IID sampling
distribution — the Stage-1 nesting caveat carries forward unchanged.)*

### 2.3 Later held-out block (n = 38,290)

| cell_id | R2_price | MAE_price | MAPE | RMSE_log | COD | PRD | PRB | MKI | VEI | beta_log | Delta_NL | dCor_e_y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.89423 | 75,655 | 0.21201 | 0.28871 | 21.627 | 1.0695 | -0.09076 | 0.92295 | -26.460 | -0.14960 | 0.11866 | 0.38709 |
| B | 0.89352 | 75,727 | 0.21183 | 0.28912 | 21.587 | 1.0694 | -0.09037 | 0.92286 | -26.827 | -0.14933 | 0.11721 | 0.38647 |
| C | 0.89281 | 75,976 | 0.21192 | 0.28940 | 21.599 | 1.0690 | -0.08845 | 0.92305 | -26.167 | -0.14738 | 0.11645 | 0.38225 |


### 2.4 2025 forward block (n = 26,641)

| cell_id | R2_price | MAE_price | MAPE | RMSE_log | COD | PRD | PRB | MKI | VEI | beta_log | Delta_NL | dCor_e_y |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 0.90438 | 78,484 | 0.20779 | 0.27819 | 21.266 | 1.0792 | -0.10604 | 0.90654 | -28.577 | -0.16408 | 0.12117 | 0.42204 |
| B | 0.90438 | 78,484 | 0.20779 | 0.27820 | 21.267 | 1.0792 | -0.10604 | 0.90654 | -28.577 | -0.16408 | 0.12118 | 0.42204 |
| C | 0.90385 | 79,139 | 0.20773 | 0.27907 | 21.229 | 1.0775 | -0.10284 | 0.90934 | -28.407 | -0.16120 | 0.12118 | 0.41684 |


---

## 3. The attribution problem, quantified on every metric

This is the operational point of the zero control: **how much of an "A versus positive-rho"
difference is already present at rho = 0?**

### CV mean

| metric | A | B | C | **A − C** | A − B | B − C |
|---|---|---|---|---|---|---|
| R2_price | 0.89974 | 0.89939 | 0.89900 | **+0.00073978** | +0.00035167 | +0.00038811 |
| MAE_price | 58,865 | 58,891 | 58,951 | **-86.028** | -25.962 | -60.065 |
| MAPE | 0.21508 | 0.21512 | 0.21530 | **-0.00021691** | -3.7575e-05 | -0.00017934 |
| RMSE_log | 0.29040 | 0.29049 | 0.29065 | **-0.00025612** | -8.709e-05 | -0.00016904 |
| COD | 21.992 | 21.997 | 22.010 | **-0.017833** | -0.0044219 | -0.013411 |
| PRD | 1.0687 | 1.0689 | 1.0687 | **-8.0584e-05** | -0.00021452 | +0.00013393 |
| PRB | -0.08675 | -0.08677 | -0.08680 | **+5.4857e-05** | +2.5389e-05 | +2.9467e-05 |
| MKI | 0.93367 | 0.93328 | 0.93387 | **-0.00019667** | +0.00039782 | -0.00059449 |
| VEI | -27.229 | -27.081 | -26.762 | **-0.46757** | -0.14792 | -0.31966 |
| beta_log | -0.13820 | -0.13824 | -0.13827 | **+6.7269e-05** | +3.4515e-05 | +3.2755e-05 |
| Delta_NL | 0.09957 | 0.09930 | 0.10021 | **-0.00064298** | +0.00026581 | -0.00090879 |
| dCor_e_y | 0.38750 | 0.38739 | 0.38698 | **+0.00051374** | +0.00010954 | +0.0004042 |

### held-out

| metric | A | B | C | **A − C** | A − B | B − C |
|---|---|---|---|---|---|---|
| R2_price | 0.89423 | 0.89352 | 0.89281 | **+0.0014229** | +0.00071008 | +0.00071286 |
| MAE_price | 75,655 | 75,727 | 75,976 | **-321.02** | -71.665 | -249.36 |
| MAPE | 0.21201 | 0.21183 | 0.21192 | **+9.3193e-05** | +0.00018073 | -8.7536e-05 |
| RMSE_log | 0.28871 | 0.28912 | 0.28940 | **-0.00068748** | -0.00041505 | -0.00027243 |
| COD | 21.627 | 21.587 | 21.599 | **+0.027289** | +0.040015 | -0.012725 |
| PRD | 1.0695 | 1.0694 | 1.0690 | **+0.00054494** | +8.3511e-05 | +0.00046143 |
| PRB | -0.09076 | -0.09037 | -0.08845 | **-0.0023103** | -0.0003909 | -0.0019194 |
| MKI | 0.92295 | 0.92286 | 0.92305 | **-9.3785e-05** | +8.9866e-05 | -0.00018365 |
| VEI | -26.460 | -26.827 | -26.167 | **-0.29301** | +0.36706 | -0.66007 |
| beta_log | -0.14960 | -0.14933 | -0.14738 | **-0.0022182** | -0.0002672 | -0.001951 |
| Delta_NL | 0.11866 | 0.11721 | 0.11645 | **+0.0022129** | +0.0014494 | +0.0007635 |
| dCor_e_y | 0.38709 | 0.38647 | 0.38225 | **+0.0048419** | +0.00061598 | +0.004226 |

### 2025 forward

| metric | A | B | C | **A − C** | A − B | B − C |
|---|---|---|---|---|---|---|
| R2_price | 0.90438 | 0.90438 | 0.90385 | **+0.00052683** | -5.712e-07 | +0.0005274 |
| MAE_price | 78,484 | 78,484 | 79,139 | **-655.27** | +0.29085 | -655.56 |
| MAPE | 0.20779 | 0.20779 | 0.20773 | **+6.578e-05** | -7.1915e-07 | +6.6499e-05 |
| RMSE_log | 0.27819 | 0.27820 | 0.27907 | **-0.00087758** | -3.0727e-06 | -0.00087451 |
| COD | 21.266 | 21.267 | 21.229 | **+0.037096** | -0.00026139 | +0.037357 |
| PRD | 1.0792 | 1.0792 | 1.0775 | **+0.0017327** | -1.0456e-06 | +0.0017337 |
| PRB | -0.10604 | -0.10604 | -0.10284 | **-0.0032034** | +4.1032e-07 | -0.0032038 |
| MKI | 0.90654 | 0.90654 | 0.90934 | **-0.0027981** | +7.063e-07 | -0.0027988 |
| VEI | -28.577 | -28.577 | -28.407 | **-0.17019** | +3.1275e-06 | -0.17019 |
| beta_log | -0.16408 | -0.16408 | -0.16120 | **-0.0028813** | +1.0608e-06 | -0.0028823 |
| Delta_NL | 0.12117 | 0.12118 | 0.12118 | **-8.6323e-06** | -2.7483e-06 | -5.884e-06 |
| dCor_e_y | 0.42204 | 0.42204 | 0.41684 | **+0.0052022** | +2.8862e-06 | +0.0051993 |

### Reading

* **The metric-level footprint of the execution-path difference is small but not negligible.**
  On held-out, `A − C` is `R2_price` **+0.00142**,
  `MAE_price` **-321**,
  `beta_log` **-0.00222**,
  `dCor` **+0.00484** —
  even though the underlying mean absolute log-prediction difference is ~3.2e-2.
* **On 2025, A and B are near-identical** (`A − B` differences of order 1e-6 on every metric),
  while on held-out `A − B` is material. This reproduces the Stage-1 capacity-F asymmetry between
  the two out-of-time blocks and confirms that label/initialisation representation and
  execution-path effects are separable but block-dependent.
* On 2025, essentially the whole of `A − C` is carried by `B − C`, i.e. by the built-in-vs-custom
  execution path rather than by label representation.

**Consequence for Gate G2.** Statements about the incremental effect of rho must be made against
**Cell C**, the within-path penalty-isolating origin. Comparisons against **Cell A** remain
visible and important as the assessor-facing benchmark, but must be worded descriptively and must
not attribute the whole difference to rho.

---

## 4. b-star diagnostics (all 9 blocks x A/B/C)

Definition in force: **`b_star_train = Var_T(y) / Cov_T(f0, y) when Cov_T(f0,y) > 0`**.
`1/R2` appears only as a **theoretical diagnostic only; NOT the LightGBM b-star definition**.

| block_id | n_T | ybar_T | f0bar_T | f0bar_minus_ybar | Var_T_y_ddof0 | Cov_T_f0_y | beta_log_train | R2_log_insample | one_over_R2_log_theoretical_diagnostic_only | b_star_train |
|---|---|---|---|---|---|---|---|---|---|---|
| fold_1_train | 46888 | 12.228936 | 12.228933 | -2.848e-06 | 0.783187 | 0.735113 | -0.061383 | 0.964189 | 1.037141 | 1.065397 |
| fold_2_train | 100776 | 12.264262 | 12.264265 | +3.725e-06 | 0.756582 | 0.706282 | -0.066483 | 0.958669 | 1.043113 | 1.071218 |
| fold_3_train | 151187 | 12.294901 | 12.294899 | -1.370e-06 | 0.721772 | 0.670709 | -0.070747 | 0.954158 | 1.048044 | 1.076133 |
| fold_4_train | 200908 | 12.330395 | 12.330390 | -4.865e-06 | 0.685598 | 0.634271 | -0.074865 | 0.949747 | 1.052912 | 1.080923 |
| fold_5_train | 252486 | 12.387104 | 12.387105 | +1.795e-06 | 0.658997 | 0.607680 | -0.077872 | 0.945890 | 1.057206 | 1.084448 |
| fold_6_train | 298022 | 12.430437 | 12.430434 | -2.388e-06 | 0.641376 | 0.590179 | -0.079823 | 0.943269 | 1.060143 | 1.086747 |
| fold_7_train | 310147 | 12.438699 | 12.438698 | -7.518e-07 | 0.636096 | 0.584358 | -0.081337 | 0.941456 | 1.062185 | 1.088538 |
| development_pool | 344607 | 12.461356 | 12.461359 | +2.404e-06 | 0.627032 | 0.575009 | -0.082967 | 0.939936 | 1.063903 | 1.090473 |
| production_2016_2024 | 382897 | 12.490190 | 12.490195 | +5.183e-06 | 0.621174 | 0.568415 | -0.084934 | 0.937516 | 1.066649 | 1.092817 |


*(Cell C shown; the full A/B/C table is `tables/b_star_diagnostics.csv`.)*

Findings across all 27 fits:

* **`Cov_T(f0,y) > 0` in every block for every cell** (`cov_positive_all_cells = True`),
  so `b_star_train` is defined everywhere; `b_star_finite_all_cells = True`.
* `b_star_train` ranges **1.065397 – 1.092966**, rising
  monotonically with block size.
* **`f0bar_T − ybar_T` is negligible**: |gap| max **8.359e-06** over all
  27 fits. The theorem-matched `ybar_T`-centered map and an `f0bar_T`-centered variant therefore
  coincide for practical purposes on these blocks — recorded now so the later comparator does not
  need to re-litigate it.
* The internal identity `beta_log_train = 1/b_star_train − 1` holds to
  **8.33e-16**, confirming the moment conventions are consistent.
* **`b_star_train` is NOT `1/R2`.** For Cell C the in-sample `1/R2_log` runs
  1.0371–1.0666
  while `b_star_train` runs 1.0654–1.0928.
  The OLS identity does not transfer to a fitted LightGBM, exactly as the plan anticipated; `1/R2`
  is retained as a theoretical diagnostic only.
* In-sample `beta_log_train` (-0.0849 to
  -0.0614) is much closer to zero than the out-of-sample
  values (≈ −0.14 to −0.16), the expected signature of in-sample overfitting in a
  994-tree / 573-leaf ensemble.

---

## 5. Retained artifacts

| Artifact | Content |
|---|---|
| `tables/zero_control_full.csv` | 66 rows: A/B/C x {fold_1..7, CV_mean, CV_SD, heldout, forward_2025} with the full canonical suite, plus frozen path-table cross-reference rows |
| `tables/zero_reference_fits.csv` | the 27 fits with n_T, ybar_T, f0bar_T, variances, covariances, beta_log_train, R2_log, b_star_train, hashes |
| `tables/b_star_diagnostics.csv` | b-star diagnostics for all 27 fits |
| `tables/zero_reference_reproduction_qc.csv` | R-tier QC for A and C; aggregate ladder cross-check for B |
| `tables/zero_control_frozen_crosscheck.csv` | 200 metric comparisons against the frozen path table |
| `output/p0_major_revision_validation/zero_reference_fits/cell=*/block=*/` | in-sample and paired-evaluation log predictions (parquet, ignored by Git), `fit_meta.json` with all hashes |
