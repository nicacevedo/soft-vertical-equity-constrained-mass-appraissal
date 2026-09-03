# Berry/ATTOM validation v3 — final report

Written: 2026-09-03T15:52:06Z  
Protocol: `analysis/berry_attom_validation_v3/protocol_v3.yaml`  
**Manuscript was not edited. Frozen v1 (`analysis/berry_cmf_validation/`) and v2 (`analysis/berry_attom_validation_v2/`) were not modified.**

Berry official assessment/sale ratios and model valuation/sale ratios are different estimands.

## 1. Accidental-prompt cleanup / repository provenance

This repository is a property-assessment / mass-appraisal regressivity project.
A previous session prompt was accidentally copied from an unrelated repository.
Inspection of *this* repo found **no files, hunks, or jobs from that prompt to revert**.
Working-tree cleanup was therefore a no-op. `git reset --hard` was not used.

- HEAD at protocol write: `8729a6e7452467d57e9316da8cf391e611237335`
- Relation to `7b6adf7`: older ancestor
- Relation to `d6b07b7`: immediate parent of that HEAD (v2 freeze SHA); current HEAD is one Berry/ATTOM commit after `d6b07b7`
- Superseded v2 Slurm jobs canceled (never executed linkage/baselines/freeze): [21842728, 21842729, 21842730, 21842731, 21842732, 21842733, 21842735]
- v2 completed artifacts preserved as preliminary evidence
- New v3 job record: `21847427 21847428 21847431 21847432 21847433 21847434 21847443 21847445 21847446 21847450
21881822 21881823 21881824 21881825 21881826 21881827
21882240 21882241 21882242 21882243
21886835 21886836 21886837 21886838 21886839 21886840`
- Scheduler: `sched_mit_sloan_batch_r8` (not `mit_normal`)

## 2. New ATTOM inventory

Reused from frozen v2 with SHA256 provenance (`inventory/V2_PROVENANCE.json`). Inventory coverage rows: 37; FIPS-year rows: 711; cache-manifest rows: 6.

Dewey folders are immutable sources. Caches are FIPS-filtered to Wayne `26163`, Philadelphia `42101`, St. Louis County `29189`. St. Louis City `29510` is forbidden in the 29189 cache. Folder names are not contents: extra FIPS in the raw delivery are not modeled.

## 3. Berry/local source reproduction

Copied from v2 without re-filtering (`berry_reproduction/PROVENANCE.md`).

- Detroit: Python translation of the Rmd filters (class 401, VALID ARMS LENGTH). Native R comparison is attempted separately when `cmfproperty` can be sourced.
- Philadelphia: canonical table is the arms-length Stata file only. The alternative total file is a disjoint universe and is not stacked naively.
- St. Louis County: 2019 cumulative `sales.csv` with actual `SALEDT` and `PRICE`. This is **not** a fully reproduced Berry assessment-ratio benchmark (no official assessed-value series in that extract). ATTOM-linkage cohort is predeclared **2005-01-01 through 2019-12-31**, not 1975-2019.

## 4. Parcel linkage

Statuses: `EXACT_RAW_APN`, `EXACT_NORMALIZED_APN`, `EXACT_PREVIOUS_APN`, `AMBIGUOUS_APN`, `NO_APN_MATCH`. One-to-many APN→ATTOMID maps are never silently collapsed. Price/date is never a parcel identifier. Address is corroboration only.

See `linkage/linkage_summary.json` and per-county `*_crosswalk.parquet`.

## 5. Recorder transaction corroboration

Independent Recorder search on linked ATTOMIDs. Date windows 0 / ±1 / ±7 / ±30 days; price exact / ≤1% / ≤5%. Tiers: `TIER_1_HIGH_CONFIDENCE`, `TIER_2_PLAUSIBLE`, `AMBIGUOUS`, `CONFLICT`. Thresholds were not retuned after seeing regressivity.

## 6. Unconditional linkage waterfall

Unconditional rates use **eligible Berry/local N** as the denominator (St. Louis eligible = 2005–2019 dated sales). Conditional rates are also reported. Do not read a high history-coverage number as coverage of the full Berry sample unless the waterfall says so.

See `linkage/unconditional_waterfall.csv`. Rows: 3.

## 7. Nested linkage-selection effects

Stages 0–4: full eligible cohort → unique APN/ATTOMID → high-confidence Recorder → safe History → fully validated. `P(stage | sale-price decile)` is in `linkage/nested_selection_audit.csv` (150 rows). Figure: `figures/fully_validated_rate_by_price_decile.pdf`.

## 8. Berry regressivity preservation

Primary scientific question: does restricting to ATTOM-linkable observations materially alter the independently documented Berry pattern (COD/PRD/PRB/value-decile profile)? Sign reversals and strong value-dependent retention are flags, not gates to be loosened.

ATTOM `TAXASSESSEDVALUETOTAL` / `TAXMARKETVALUETOTAL` are **not** treated as Berry assessed value. Direct Berry-vs-ATTOM assessment-ratio comparison is skipped; see `source_concordance/STEP9_ASSESSMENT_CONCORDANCE.md`.

## 9. Standardized ATTOM modeling-cohort selection

2016–2025 qualified arms-length single-parcel residential (`PROPERTYUSESTANDARDIZED=385`) Recorder sales with strict pre-sale History. Wayne models are Wayne County, never Detroit.

Retention audit: `feature_audit/*_modeling_retention_by_decile.csv` and `modeling_table_summary.csv` (3 rows). Report `P(enters final model | Recorder price decile)` and eligible-vs-final price differences.

### 9a. Philadelphia's property-use filter is the dominant, value-dependent selection

This is a first-class limitation on the Philadelphia results, not a footnote.

`PROPERTYUSESTANDARDIZED=385` retains 8.75% of Philadelphia's eligible Recorder sales
(208,508 safe-history matches to 38,043 modeled), against roughly 85% retention in Wayne
and 82% in St. Louis. The cause is coding, not stock: Philadelphia's Assessor History is
dominated by use code 366 (5.36M rows) with 385 second (2.68M), while Wayne (13.3M) and
St. Louis (5.6M) are 385-dominated. Dewey ships no `PROPERTYUSESTANDARDIZED` code
dictionary in this delivery, so no code can be *called* residential here; the sensitivity
cohort below is defined by published structural facts instead.

The attrition is also value-dependent and non-monotone. `P(enters final model | Recorder
price decile)` in Philadelphia runs 0.028 at the bottom decile, peaks near 0.176 at the
eighth, then falls back to 0.084 at the top. **Philadelphia's vertical-equity diagnostics
therefore rest on a value-dependently filtered cohort**, and its lower held-out R2 relative
to Wayne should be read in that light.

Philadelphia was assigned `MODEL_TRANSFER_STATUS: PRIMARY` before this was visible. The
freeze is deliberately **not** revised: the protocol forbids moving a jurisdiction status on
information that arrived after the freeze, and `final_panel_freeze_v3.yaml` is byte-identical
to its frozen state. The finding is documented and probed instead.

### Property-use attrition, all three counties
| county | safe_history_sales | kept_by_385 | share_kept_by_385 | kept_by_broad_residential | share_kept_by_broad | n_broad_codes | broad_codes |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Wayne County, MI | 213274 | 182042 | 0.854 | 206251 | 0.967 | 4 | 165 366 385 401 |
| Philadelphia County, PA | 208508 | 38043 | 0.182 | 202933 | 0.973 | 7 | 169 361 366 372 385 386 401 |
| St. Louis County, MO | 172055 | 144995 | 0.843 | 168864 | 0.981 | 7 | 188 361 366 372 385 397 401 |

### Wayne County, MI: P(kept | safe-history sale-price decile)
| price_decile | n_safe_history | median_price | p_kept_primary_385 | p_kept_broad_residential |
| --- | --- | --- | --- | --- |
| 0 | 2.138e+04 | 5.85e+04 | 0.9065 | 0.9653 |
| 1 | 2.165e+04 | 7.7e+04 | 0.899 | 0.9703 |
| 2 | 2.096e+04 | 1e+05 | 0.9009 | 0.9699 |
| 3 | 2.205e+04 | 1.25e+05 | 0.9033 | 0.9787 |
| 4 | 2.233e+04 | 1.499e+05 | 0.8833 | 0.9758 |
| 5 | 1.96e+04 | 1.75e+05 | 0.8772 | 0.9789 |
| 6 | 2.14e+04 | 2.07e+05 | 0.851 | 0.9752 |
| 7 | 2.13e+04 | 2.511e+05 | 0.8381 | 0.9744 |
| 8 | 2.132e+04 | 3.25e+05 | 0.7841 | 0.9639 |
| 9 | 2.129e+04 | 5e+05 | 0.6908 | 0.9183 |

### Philadelphia County, PA: P(kept | safe-history sale-price decile)
| price_decile | n_safe_history | median_price | p_kept_primary_385 | p_kept_broad_residential |
| --- | --- | --- | --- | --- |
| 0 | 2.222e+04 | 6.5e+04 | 0.08514 | 0.9808 |
| 1 | 2.005e+04 | 1e+05 | 0.1006 | 0.9813 |
| 2 | 2.262e+04 | 1.35e+05 | 0.1177 | 0.9851 |
| 3 | 1.893e+04 | 1.67e+05 | 0.1601 | 0.9886 |
| 4 | 2.064e+04 | 2e+05 | 0.2013 | 0.9869 |
| 5 | 2.142e+04 | 2.35e+05 | 0.2455 | 0.9864 |
| 6 | 2.014e+04 | 2.79e+05 | 0.273 | 0.9853 |
| 7 | 2.089e+04 | 3.35e+05 | 0.294 | 0.9834 |
| 8 | 2.076e+04 | 4.325e+05 | 0.2198 | 0.9707 |
| 9 | 2.085e+04 | 7.25e+05 | 0.1356 | 0.8846 |

### St. Louis County, MO: P(kept | safe-history sale-price decile)
| price_decile | n_safe_history | median_price | p_kept_primary_385 | p_kept_broad_residential |
| --- | --- | --- | --- | --- |
| 0 | 1.732e+04 | 6.5e+04 | 0.8563 | 0.9912 |
| 1 | 1.722e+04 | 9.774e+04 | 0.8106 | 0.9934 |
| 2 | 1.716e+04 | 1.28e+05 | 0.7827 | 0.9925 |
| 3 | 1.829e+04 | 1.599e+05 | 0.8036 | 0.9927 |
| 4 | 1.605e+04 | 1.933e+05 | 0.8405 | 0.9936 |
| 5 | 1.719e+04 | 2.35e+05 | 0.8729 | 0.9912 |
| 6 | 1.736e+04 | 2.87e+05 | 0.896 | 0.9895 |
| 7 | 1.707e+04 | 3.6e+05 | 0.9072 | 0.9869 |
| 8 | 1.718e+04 | 4.85e+05 | 0.8993 | 0.9819 |
| 9 | 1.721e+04 | 8.65e+05 | 0.7604 | 0.9018 |

### Philadelphia cohort sensitivity, VALIDATION ONLY
| cohort | n_full | n_validation | validation_R2 | validation_COD | validation_PRD | validation_PRB | validation_beta_log | freeze_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| primary_385 (frozen) | 38043 | 3044 | 0.7795 | 25.09 | 1.113 | -0.2456 | -0.3681 | pending |
| broad_residential (sensitivity) | 202933 | 16235 | 0.1981 | 27.86 | 1.178 | -0.1254 | -0.2452 | SENSITIVITY_ONLY_NOT_A_FREEZE_REVISION |

The sensitivity cohort was never scored on the held-out test block and never entered Direct or Surrogate. It does not revise Philadelphia's frozen `MODEL_TRANSFER_STATUS`.


## 10. Validation-only baseline results

LR + ordinary LightGBM on development/validation only. Categorical LR levels from the development prefix. The chronological test block is not scored or stored before freeze. Hyperparameter selection is validation-only. A non-regressive validation baseline is **not** an exclusion criterion.

See `baselines_pre_freeze/<county>/validation_metrics.csv`.

## 11. St. Louis provider robustness

Local table uses actual `PRICE` and `SALEDT`, never `APRTOT`, with an explicit 2012 dwelling-history defect sensitivity. Validation metrics only.

**This comparison is statistically powerless and must not be read as evidence either way.** The common cohort holds 476 transactions with 38 validation rows. Two causes, both design choices rather than data limits: the cohort was cut to the standardized 2016-2025 ATTOM window, which against a local sales file ending in 2019 leaves only 2016-2019; and the join was a bespoke normalized-raw-APN plus exact-sale-date match rather than the validated linkage crosswalk, which reaches 99.6% APN match on this county over 2005-2019. Differences in the table below are within noise at this N.

| source | n_common | n_validation | R2_price | RMSE_log | COD | PRD | PRB | Beta_log |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| local_historical_dwelling | 476 | 38 | 0.08624 | 0.541 | 27.9 | 1.377 | -0.2225 | -0.5225 |
| attom_assessor_history | 476 | 38 | 0.2734 | 0.4279 | 29.04 | 1.343 | -0.1743 | -0.4031 |

Recommended follow-up: rebuild this comparison over the predeclared 2005-2019 St. Louis linkage window using `linkage/st_louis_county_crosswalk.parquet`. Not done in this pass.


## 12. Frozen model-transfer and Berry-anchor statuses

`panel_freeze/final_panel_freeze_v3.yaml`

Direct/Surrogate authorized: **True**  
Passing MODEL_TRANSFER units: ['wayne', 'philadelphia', 'st_louis_county']

Two independent statuses per jurisdiction: `MODEL_TRANSFER_STATUS` (PRIMARY / BOUNDARY / NOT_ELIGIBLE) and `BERRY_ANCHOR_STATUS` (STRONG / MODERATE / WEAK / NOT_APPLICABLE). Berry linkage is not a hard requirement for ATTOM model validity. A strong ATTOM AVM is not a claim of Berry replication.

If fewer than two MODEL_TRANSFER units are PRIMARY or BOUNDARY: STOP. Do not force penalty results.

## 13. Final untouched held-out baselines

Frozen LR and LightGBM configurations refit on development+validation, then the chronological test block scored **once**. The same test rows carry into every Direct/Surrogate comparison.

| jurisdiction | model | n_test | R2_price | R2_log | RMSE_log | MAPE | COD | PRD | PRB | MKI | VEI | Beta_log | Delta_NL | dCor_e_y |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wayne County, MI | LGBM | 36409 | 0.795 | 0.7636 | 0.3223 | 0.2427 | 24.86 | 1.091 | -0.1087 | 0.8668 | -34.64 | -0.2137 | 0.1435 | 0.4844 |
| Wayne County, MI | LR | 36409 | 0.544 | 0.6149 | 0.4115 | 0.3317 | 35.13 | 1.169 | -0.2128 | 0.7201 | -57.18 | -0.3826 | 0.07113 | 0.6138 |
| Philadelphia County, PA | LGBM | 7609 | 0.5051 | 0.6703 | 0.3229 | 0.2385 | 24.55 | 1.12 | -0.2179 | 0.7281 | -46.25 | -0.3418 | 0.05878 | 0.5596 |
| Philadelphia County, PA | LR | 7609 | -51.27 | 0.4488 | 0.4174 | 0.418 | 41.77 | 1.077 | 0.8131 | 0.8945 | -100.3 | -0.497 | 0.09358 | 0.6765 |
| St. Louis County, MO | LGBM | 28999 | 0.6807 | 0.8313 | 0.3295 | 0.2336 | 23.81 | 1.107 | -0.05627 | 0.8897 | -27.72 | -0.1456 | 0.0919 | 0.3578 |
| St. Louis County, MO | LR | 28999 | -4.158e+08 | -0.3238 | 0.9232 | 56.38 | 6586 | 0.3943 | 1309 | 1.968 | 40.64 | -0.4061 | 0.01586 | 0.3919 |

#### The LR reference model's level-space metrics are degenerate in two counties

Read `R2_log` for LR, not `R2_price`. Exponentiating a log-space linear fit extrapolates without bound, and a handful of held-out rows blow up every level-space aggregate. St. Louis County has a single LR prediction near $1.5 trillion, which alone drives its `R2_price` to about -4.2e8, `COD` to 6586 and `PRB` to 1309. Philadelphia has two rows above 100x. Those numbers are artifacts of one or two rows, not statements about vertical equity.

| jurisdiction | n_test | max_lr_prediction | max_ratio | ratio_p99_9 | n_ratio_gt_10 | n_ratio_gt_100 |
| --- | --- | --- | --- | --- | --- | --- |
| Wayne County, MI | 36409 | 6.78e+06 | 11.1 | 3.97 | 2 | 0 |
| Philadelphia County, PA | 7609 | 1.52e+08 | 232 | 6.87 | 5 | 2 |
| St. Louis County, MO | 28999 | 1.46e+12 | 1.62e+06 | 5.64 | 7 | 1 |

Nothing was clipped or winsorized. The test block is scored once, and trimming it after seeing the result would be an outcome-driven edit to a frozen evaluation. The consequences are contained: **LightGBM is the baseline that carries the science.** It is the model the panel freeze used, the model the Direct and Surrogate paths penalize, and its held-out predictions are well behaved in every county (maximum ratio 9.1 / 9.7 / 17.3). LR is a reference point only, and no jurisdiction status, rho anchor, or conclusion depends on it. A bounded LR variant should be pre-registered in a future pass rather than patched into this one.


## 14. Direct results if authorized

`LGBCovPenalty[diff]`. Rho from the rank-one mapping on the FULL PRETEST block. Portable covariance-reduction anchors 10/25/50/67/80/90/97%. Raw rho is not comparable across counties and is not compared.

### Wayne County, MI

| requested_reduction | rho | R2_price | RMSE_log | COD | PRD | PRB | MKI | VEI | Beta_log | Cov(e,logprice) | Delta_NL | dCor_e_y |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.6905 | 0.799 | 0.323 | 24.83 | 1.086 | -0.1004 | 0.8761 | -33.27 | -0.206 | -0.09058 | 0.1488 | 0.4717 |
| 0.25 | 2.072 | 0.8088 | 0.3239 | 24.83 | 1.074 | -0.07872 | 0.9005 | -26.42 | -0.185 | -0.08132 | 0.1566 | 0.4325 |
| 0.5 | 6.215 | 0.8178 | 0.3252 | 24.89 | 1.056 | -0.04513 | 0.9375 | -16.65 | -0.151 | -0.06639 | 0.164 | 0.3709 |
| 0.67 | 12.62 | 0.8155 | 0.3288 | 25.28 | 1.043 | -0.02175 | 0.9641 | -9.24 | -0.1274 | -0.056 | 0.1684 | 0.3308 |
| 0.8 | 24.86 | 0.8051 | 0.3322 | 25.54 | 1.032 | -0.004657 | 0.9859 | -4.245 | -0.1094 | -0.0481 | 0.1716 | 0.3042 |
| 0.9 | 55.93 | 0.7934 | 0.3366 | 26.09 | 1.023 | 0.01065 | 1.003 | 1.177 | -0.09475 | -0.04165 | 0.1738 | 0.2846 |
| 0.97 | 200.9 | 0.7652 | 0.3478 | 27.25 | 1.017 | 0.02636 | 1.018 | 6.408 | -0.08467 | -0.03722 | 0.1683 | 0.265 |

### Philadelphia County, PA

| requested_reduction | rho | R2_price | RMSE_log | COD | PRD | PRB | MKI | VEI | Beta_log | Cov(e,logprice) | Delta_NL | dCor_e_y |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.8791 | 0.5094 | 0.3295 | 25.04 | 1.117 | -0.209 | 0.7386 | -44.82 | -0.3391 | -0.1072 | 0.06282 | 0.5503 |
| 0.25 | 2.637 | 0.5237 | 0.3294 | 25.07 | 1.107 | -0.1762 | 0.7674 | -37.42 | -0.3161 | -0.09993 | 0.06404 | 0.5076 |
| 0.5 | 7.912 | 0.52 | 0.3344 | 25.72 | 1.093 | -0.1309 | 0.8092 | -26.46 | -0.2863 | -0.0905 | 0.07118 | 0.45 |
| 0.67 | 16.06 | 0.507 | 0.3403 | 26.35 | 1.081 | -0.08676 | 0.8446 | -17.81 | -0.2536 | -0.08017 | 0.07249 | 0.3932 |
| 0.8 | 31.65 | 0.5009 | 0.3381 | 26.2 | 1.075 | -0.0735 | 0.8605 | -14.04 | -0.2393 | -0.07564 | 0.07308 | 0.3743 |
| 0.9 | 71.21 | 0.4853 | 0.3435 | 26.61 | 1.07 | -0.05347 | 0.8735 | -9.991 | -0.2211 | -0.0699 | 0.07053 | 0.3443 |
| 0.97 | 255.8 | 0.4788 | 0.3556 | 27.28 | 1.064 | -0.02707 | 0.8883 | -2.235 | -0.2016 | -0.06372 | 0.06093 | 0.3135 |

### St. Louis County, MO

| requested_reduction | rho | R2_price | RMSE_log | COD | PRD | PRB | MKI | VEI | Beta_log | Cov(e,logprice) | Delta_NL | dCor_e_y |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.4653 | 0.6855 | 0.3311 | 23.78 | 1.098 | -0.04544 | 0.9007 | -23.2 | -0.134 | -0.08625 | 0.09207 | 0.3304 |
| 0.25 | 1.396 | 0.6955 | 0.3316 | 23.74 | 1.084 | -0.03012 | 0.9175 | -17.32 | -0.1165 | -0.07503 | 0.09497 | 0.2926 |
| 0.5 | 4.188 | 0.7066 | 0.3319 | 23.79 | 1.065 | -0.00766 | 0.9425 | -9.943 | -0.09061 | -0.05834 | 0.1004 | 0.2443 |
| 0.67 | 8.503 | 0.7094 | 0.3349 | 24.03 | 1.051 | 0.006327 | 0.9594 | -4.809 | -0.07462 | -0.04804 | 0.1034 | 0.2194 |
| 0.8 | 16.75 | 0.7103 | 0.3362 | 24.23 | 1.041 | 0.01756 | 0.9722 | -0.3465 | -0.06071 | -0.03909 | 0.1025 | 0.2035 |
| 0.9 | 37.69 | 0.7052 | 0.3418 | 24.68 | 1.034 | 0.02644 | 0.9813 | 2.027 | -0.05177 | -0.03333 | 0.1037 | 0.1932 |
| 0.97 | 135.4 | 0.6985 | 0.347 | 25.44 | 1.029 | 0.03267 | 0.9882 | 4.615 | -0.0473 | -0.03046 | 0.1016 | 0.188 |


## 15. Surrogate results if authorized

First contiguous low-rho branch only; no global sort + `np.interp`. Every UNATTAINED row now carries why it was unattained, which pass 1 could not express.

**Pass 2.** Pass 1 (`21882241`) had two pretest-diagnosable defects: a fixed `geomspace(1e-6, 1e2, 16)` grid whose ceiling sat below the Direct 97% anchor (rho up to 255.8), and a branch detector with no noise floor that opened a one-point branch on a 0.008 reduction at rho=1e-6 in St. Louis. Pass 2 ties the grid ceiling to 4x this county's largest Direct rho and requires a 1% reduction to open a branch. Rho was never chosen from a test metric in either pass. Philadelphia's test rows were scored twice and pass-1 output is preserved as `surrogate_pass1_*.csv`. Full disclosure: `panel_freeze/SURROGATE_RECALIBRATION_LOG.md`.


### Wayne County, MI

| requested_reduction | rho | status | unattained_reason | branch_terminated_by | R2_price | COD | PRD | PRB | Beta_log | Cov(e,logprice) | Delta_NL | dCor_e_y |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 |  | UNATTAINED | MATERIAL_REVERSAL | MATERIAL_REVERSAL | — | — | — | — | — | — | — | — |
| 0.25 |  | UNATTAINED | MATERIAL_REVERSAL | MATERIAL_REVERSAL | — | — | — | — | — | — | — | — |
| 0.5 |  | UNATTAINED | MATERIAL_REVERSAL | MATERIAL_REVERSAL | — | — | — | — | — | — | — | — |
| 0.67 |  | UNATTAINED | MATERIAL_REVERSAL | MATERIAL_REVERSAL | — | — | — | — | — | — | — | — |
| 0.8 |  | UNATTAINED | MATERIAL_REVERSAL | MATERIAL_REVERSAL | — | — | — | — | — | — | — | — |
| 0.9 |  | UNATTAINED | MATERIAL_REVERSAL | MATERIAL_REVERSAL | — | — | — | — | — | — | — | — |
| 0.97 |  | UNATTAINED | MATERIAL_REVERSAL | MATERIAL_REVERSAL | — | — | — | — | — | — | — | — |

Grid 0.001 to 803.8 over 25 points (max Direct rho 200.9); fit failures: 0.

### Philadelphia County, PA

| requested_reduction | rho | status | unattained_reason | branch_terminated_by | R2_price | COD | PRD | PRB | Beta_log | Cov(e,logprice) | Delta_NL | dCor_e_y |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 2.086 | interpolated_first_branch |  | MATERIAL_REVERSAL | 0.5145 | 24.63 | 1.109 | -0.1675 | -0.3087 | -0.09759 | 0.04604 | 0.4964 |
| 0.25 | 15.93 | interpolated_first_branch |  | MATERIAL_REVERSAL | 0.5174 | 25.52 | 1.092 | -0.08479 | -0.2452 | -0.07752 | 0.03982 | 0.3801 |
| 0.5 |  | UNATTAINED | MATERIAL_REVERSAL | MATERIAL_REVERSAL |  |  |  |  |  |  |  |  |
| 0.67 |  | UNATTAINED | MATERIAL_REVERSAL | MATERIAL_REVERSAL |  |  |  |  |  |  |  |  |
| 0.8 |  | UNATTAINED | MATERIAL_REVERSAL | MATERIAL_REVERSAL |  |  |  |  |  |  |  |  |
| 0.9 |  | UNATTAINED | MATERIAL_REVERSAL | MATERIAL_REVERSAL |  |  |  |  |  |  |  |  |
| 0.97 |  | UNATTAINED | MATERIAL_REVERSAL | MATERIAL_REVERSAL |  |  |  |  |  |  |  |  |

Grid 0.001 to 1023 over 25 points (max Direct rho 255.8); fit failures: 0.

Pass 1 attained 2 of 7 anchors here against 2 of 7 in pass 2; see `surrogate_pass1_heldout.csv`.

### St. Louis County, MO

| requested_reduction | rho | status | unattained_reason | branch_terminated_by | R2_price | COD | PRD | PRB | Beta_log | Cov(e,logprice) | Delta_NL | dCor_e_y |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.1 | 0.172 | interpolated_first_branch |  | GRID_CEILING | 0.6871 | 24.17 | 1.094 | -0.03828 | -0.1276 | -0.08214 | 0.08789 | 0.3126 |
| 0.25 | 0.8301 | interpolated_first_branch |  | GRID_CEILING | 0.6936 | 24.83 | 1.073 | -0.006137 | -0.09259 | -0.05961 | 0.08974 | 0.2361 |
| 0.5 | 6.673 | interpolated_first_branch |  | GRID_CEILING | 0.6923 | 26.72 | 1.045 | 0.04179 | -0.04007 | -0.0258 | 0.112 | 0.1909 |
| 0.67 |  | UNATTAINED | GRID_CEILING | GRID_CEILING |  |  |  |  |  |  |  |  |
| 0.8 |  | UNATTAINED | GRID_CEILING | GRID_CEILING |  |  |  |  |  |  |  |  |
| 0.9 |  | UNATTAINED | GRID_CEILING | GRID_CEILING |  |  |  |  |  |  |  |  |
| 0.97 |  | UNATTAINED | GRID_CEILING | GRID_CEILING |  |  |  |  |  |  |  |  |

Grid 0.001 to 541.7 over 25 points (max Direct rho 135.4); fit failures: 0.

Top-of-branch slope is 0.042 reduction per decade of rho (measured over the final 1.19 decades, ending at rho=542 with 0.668 achieved). The branch is **saturating** there. Extrapolating that slope, 67% would need about 0.0 more decades of rho (rho ~ 1e3); 80% would need about 3.1 more decades of rho (rho ~ 1e6); 90% would need about 5.5 more decades of rho (rho ~ 1e8); 97% would need about 7.2 more decades of rho (rho ~ 1e10). So these anchors are **not** unattained because the grid stopped too early. The Surrogate's first-order reduction saturates below them on this county, and widening the grid again would not deliver them. That is a finding about the method, not a missing run.

Pass 1 attained 0 of 7 anchors here against 3 of 7 in pass 2; see `surrogate_pass1_heldout.csv`.


## 16. Remaining nonlinear dependence / shape failures

Held-out `Delta_NL` and `dCor_e_y` sit in the Direct and Surrogate tables above; sale-price-decile
valuation-ratio profiles are in `final_baselines/<county>/decile_valuation_ratio_profiles.csv` with
figures at `figures/<county>_heldout_ratio_by_decile.pdf`. Residual nonlinear dependence after
first-order reduction is a scientific finding, not a reason to retune rho on test.

`HISTORY_STRUCTURAL_CORE` exists in v3 only as a validation-side metric
(`validation_lgbm_HISTORY_STRUCTURAL_CORE` in each `baselines_pre_freeze/<county>/run_meta.json`).
It has no separate modeling table, no held-out evaluation and no penalty path, so it is a
validation-only sensitivity and must not be reported as a second held-out feature family.

### 16a. Bootstrap distributions

200 monthly time-block bootstrap draws. The draw indices are built once per jurisdiction and reused across baseline, Direct and Surrogate, so method differences are not draw differences. Saved as `output/.../final_models/<county>/bootstrap_indices.npy`.

### Wayne County, MI

| method | metric | mean | std | ci_2_5 | ci_97_5 |
| --- | --- | --- | --- | --- | --- |
| baseline_lgbm | R2 | 0.7942 | 0.007185 | 0.7806 | 0.806 |
| baseline_lgbm | RMSE_log | 0.3223 | 0.001971 | 0.3187 | 0.3261 |
| baseline_lgbm | COD | 24.85 | 0.2133 | 24.47 | 25.26 |
| baseline_lgbm | VEI | -34.55 | 0.7958 | -36.22 | -32.9 |
| baseline_lgbm | PRD | 1.09 | 0.001754 | 1.087 | 1.094 |
| baseline_lgbm | PRB | -0.1084 | 0.002591 | -0.1134 | -0.1034 |
| baseline_lgbm | MKI | 0.8668 | 0.003715 | 0.8602 | 0.8731 |
| baseline_lgbm | Beta_log | -0.2137 | 0.003569 | -0.2209 | -0.2074 |
| baseline_lr | R2 | 0.5424 | 0.03042 | 0.4746 | 0.588 |
| baseline_lr | RMSE_log | 0.4114 | 0.002515 | 0.4069 | 0.4165 |
| baseline_lr | COD | 35.13 | 0.2326 | 34.71 | 35.57 |
| baseline_lr | VEI | -57.51 | 1.452 | -60.25 | -54.61 |
| baseline_lr | PRD | 1.169 | 0.002427 | 1.164 | 1.174 |
| baseline_lr | PRB | -0.2128 | 0.004193 | -0.2211 | -0.2055 |
| baseline_lr | MKI | 0.7194 | 0.005624 | 0.7085 | 0.7294 |
| baseline_lr | Beta_log | -0.383 | 0.00391 | -0.3907 | -0.3767 |
| direct_rho_0.690527_heldout | R2 | 0.7983 | 0.007231 | 0.7828 | 0.8102 |
| direct_rho_0.690527_heldout | RMSE_log | 0.323 | 0.001994 | 0.3194 | 0.3269 |
| direct_rho_0.690527_heldout | COD | 24.83 | 0.2266 | 24.43 | 25.3 |
| direct_rho_0.690527_heldout | VEI | -33.03 | 0.7118 | -34.48 | -31.65 |
| direct_rho_0.690527_heldout | PRD | 1.086 | 0.001772 | 1.083 | 1.09 |
| direct_rho_0.690527_heldout | PRB | -0.1002 | 0.002562 | -0.105 | -0.09515 |
| direct_rho_0.690527_heldout | MKI | 0.8761 | 0.00393 | 0.8688 | 0.8825 |
| direct_rho_0.690527_heldout | Beta_log | -0.2061 | 0.003822 | -0.2138 | -0.1993 |
| direct_rho_12.6178_heldout | R2 | 0.8145 | 0.007686 | 0.7992 | 0.8269 |
| direct_rho_12.6178_heldout | RMSE_log | 0.3289 | 0.002334 | 0.3246 | 0.3338 |
| direct_rho_12.6178_heldout | COD | 25.28 | 0.2432 | 24.83 | 25.75 |
| direct_rho_12.6178_heldout | VEI | -9.224 | 0.7171 | -10.54 | -7.812 |
| direct_rho_12.6178_heldout | PRD | 1.042 | 0.001906 | 1.039 | 1.046 |
| direct_rho_12.6178_heldout | PRB | -0.0215 | 0.002511 | -0.0261 | -0.01672 |
| direct_rho_12.6178_heldout | MKI | 0.9641 | 0.004112 | 0.9569 | 0.9719 |
| direct_rho_12.6178_heldout | Beta_log | -0.1274 | 0.004003 | -0.135 | -0.12 |
| direct_rho_2.07158_heldout | R2 | 0.808 | 0.007166 | 0.7934 | 0.8198 |
| direct_rho_2.07158_heldout | RMSE_log | 0.3239 | 0.002108 | 0.32 | 0.3282 |
| direct_rho_2.07158_heldout | COD | 24.83 | 0.2367 | 24.39 | 25.3 |
| direct_rho_2.07158_heldout | VEI | -26.28 | 0.8139 | -27.85 | -24.91 |
| direct_rho_2.07158_heldout | PRD | 1.074 | 0.001845 | 1.071 | 1.078 |
| direct_rho_2.07158_heldout | PRB | -0.07848 | 0.002631 | -0.08348 | -0.07322 |
| direct_rho_2.07158_heldout | MKI | 0.9006 | 0.00399 | 0.8936 | 0.9074 |
| direct_rho_2.07158_heldout | Beta_log | -0.185 | 0.003867 | -0.1929 | -0.178 |
| direct_rho_200.943_heldout | R2 | 0.7636 | 0.009069 | 0.7427 | 0.7777 |
| direct_rho_200.943_heldout | RMSE_log | 0.3479 | 0.002411 | 0.3429 | 0.3524 |
| direct_rho_200.943_heldout | COD | 27.26 | 0.2466 | 26.76 | 27.74 |
| direct_rho_200.943_heldout | VEI | 6.416 | 0.787 | 5.021 | 7.987 |
| direct_rho_200.943_heldout | PRD | 1.017 | 0.002271 | 1.012 | 1.021 |
| direct_rho_200.943_heldout | PRB | 0.02665 | 0.002662 | 0.02177 | 0.03208 |
| direct_rho_200.943_heldout | MKI | 1.018 | 0.004788 | 1.009 | 1.026 |
| direct_rho_200.943_heldout | Beta_log | -0.08474 | 0.004343 | -0.09291 | -0.07647 |
| direct_rho_24.859_heldout | R2 | 0.804 | 0.008056 | 0.7876 | 0.8175 |
| direct_rho_24.859_heldout | RMSE_log | 0.3323 | 0.002408 | 0.3278 | 0.3371 |
| direct_rho_24.859_heldout | COD | 25.54 | 0.2417 | 25.08 | 26.02 |
| direct_rho_24.859_heldout | VEI | -4.304 | 0.7276 | -5.702 | -2.76 |
| direct_rho_24.859_heldout | PRD | 1.032 | 0.002081 | 1.028 | 1.036 |
| direct_rho_24.859_heldout | PRB | -0.004407 | 0.00263 | -0.009341 | 0.0007503 |
| direct_rho_24.859_heldout | MKI | 0.9859 | 0.004372 | 0.9782 | 0.9936 |
| direct_rho_24.859_heldout | Beta_log | -0.1095 | 0.004194 | -0.1178 | -0.1018 |
| direct_rho_55.9327_heldout | R2 | 0.7924 | 0.008101 | 0.7749 | 0.805 |
| direct_rho_55.9327_heldout | RMSE_log | 0.3367 | 0.002311 | 0.3322 | 0.3413 |
| direct_rho_55.9327_heldout | COD | 26.09 | 0.2409 | 25.65 | 26.59 |
| direct_rho_55.9327_heldout | VEI | 1.331 | 0.7393 | -0.1239 | 2.667 |
| direct_rho_55.9327_heldout | PRD | 1.023 | 0.002232 | 1.019 | 1.028 |
| direct_rho_55.9327_heldout | PRB | 0.01084 | 0.002563 | 0.006126 | 0.01578 |
| direct_rho_55.9327_heldout | MKI | 1.003 | 0.004655 | 0.995 | 1.012 |
| direct_rho_55.9327_heldout | Beta_log | -0.09486 | 0.004439 | -0.1033 | -0.08647 |
| direct_rho_6.21474_heldout | R2 | 0.8169 | 0.007513 | 0.8026 | 0.8297 |
| direct_rho_6.21474_heldout | RMSE_log | 0.3253 | 0.00225 | 0.3211 | 0.33 |
| direct_rho_6.21474_heldout | COD | 24.89 | 0.2415 | 24.46 | 25.39 |
| direct_rho_6.21474_heldout | VEI | -16.58 | 0.7503 | -17.96 | -15.09 |
| direct_rho_6.21474_heldout | PRD | 1.056 | 0.001953 | 1.052 | 1.059 |
| direct_rho_6.21474_heldout | PRB | -0.04488 | 0.002706 | -0.04975 | -0.03974 |
| direct_rho_6.21474_heldout | MKI | 0.9376 | 0.004141 | 0.9301 | 0.9452 |
| direct_rho_6.21474_heldout | Beta_log | -0.1511 | 0.004088 | -0.1587 | -0.1433 |

### Philadelphia County, PA

| method | metric | mean | std | ci_2_5 | ci_97_5 |
| --- | --- | --- | --- | --- | --- |
| baseline_lgbm | R2 | 0.5337 | 0.1193 | 0.3341 | 0.735 |
| baseline_lgbm | RMSE_log | 0.3225 | 0.004554 | 0.3136 | 0.3317 |
| baseline_lgbm | COD | 24.56 | 0.4529 | 23.64 | 25.51 |
| baseline_lgbm | VEI | -46.21 | 3.169 | -52.81 | -40.73 |
| baseline_lgbm | PRD | 1.12 | 0.005921 | 1.109 | 1.132 |
| baseline_lgbm | PRB | -0.2182 | 0.008978 | -0.234 | -0.2008 |
| baseline_lgbm | MKI | 0.7295 | 0.01073 | 0.7089 | 0.7495 |
| baseline_lgbm | Beta_log | -0.3415 | 0.007313 | -0.3565 | -0.3289 |
| baseline_lr | R2 | -46.18 | 22.77 | -88.4 | -6.749 |
| baseline_lr | RMSE_log | 0.4173 | 0.007472 | 0.404 | 0.4318 |
| baseline_lr | COD | 41.71 | 3.585 | 35.72 | 48.88 |
| baseline_lr | VEI | -100.3 | 6.427 | -110.2 | -86.91 |
| baseline_lr | PRD | 1.084 | 0.05471 | 0.9609 | 1.165 |
| baseline_lr | PRB | 0.7872 | 0.6289 | -0.2062 | 2.1 |
| baseline_lr | MKI | 0.8736 | 0.1425 | 0.6617 | 1.204 |
| baseline_lr | Beta_log | -0.4975 | 0.01402 | -0.523 | -0.4664 |
| direct_rho_0.879111_heldout | R2 | 0.5353 | 0.1107 | 0.3483 | 0.7236 |
| direct_rho_0.879111_heldout | RMSE_log | 0.3293 | 0.004539 | 0.321 | 0.3384 |
| direct_rho_0.879111_heldout | COD | 25.04 | 0.4321 | 24.18 | 25.86 |
| direct_rho_0.879111_heldout | VEI | -44.97 | 3.885 | -51.4 | -36.95 |
| direct_rho_0.879111_heldout | PRD | 1.116 | 0.006192 | 1.105 | 1.128 |
| direct_rho_0.879111_heldout | PRB | -0.2096 | 0.009349 | -0.2276 | -0.1918 |
| direct_rho_0.879111_heldout | MKI | 0.7399 | 0.01217 | 0.7156 | 0.7645 |
| direct_rho_0.879111_heldout | Beta_log | -0.3392 | 0.007909 | -0.3563 | -0.3247 |
| direct_rho_16.0638_heldout | R2 | 0.5289 | 0.1081 | 0.347 | 0.707 |
| direct_rho_16.0638_heldout | RMSE_log | 0.3403 | 0.004793 | 0.3318 | 0.3498 |
| direct_rho_16.0638_heldout | COD | 26.37 | 0.4478 | 25.53 | 27.25 |
| direct_rho_16.0638_heldout | VEI | -18.22 | 2.577 | -22.92 | -13.07 |
| direct_rho_16.0638_heldout | PRD | 1.081 | 0.006117 | 1.07 | 1.092 |
| direct_rho_16.0638_heldout | PRB | -0.08697 | 0.009002 | -0.103 | -0.06975 |
| direct_rho_16.0638_heldout | MKI | 0.8461 | 0.0136 | 0.8179 | 0.874 |
| direct_rho_16.0638_heldout | Beta_log | -0.2538 | 0.00885 | -0.2722 | -0.238 |
| direct_rho_2.63733_heldout | R2 | 0.5488 | 0.1107 | 0.3624 | 0.7326 |
| direct_rho_2.63733_heldout | RMSE_log | 0.3293 | 0.004663 | 0.3209 | 0.3388 |
| direct_rho_2.63733_heldout | COD | 25.09 | 0.4366 | 24.23 | 25.94 |
| direct_rho_2.63733_heldout | VEI | -38.18 | 3.3 | -44.31 | -31.68 |
| direct_rho_2.63733_heldout | PRD | 1.106 | 0.00594 | 1.096 | 1.117 |
| direct_rho_2.63733_heldout | PRB | -0.1768 | 0.009635 | -0.1941 | -0.1587 |
| direct_rho_2.63733_heldout | MKI | 0.7688 | 0.01226 | 0.7434 | 0.7923 |
| direct_rho_2.63733_heldout | Beta_log | -0.3163 | 0.008317 | -0.3344 | -0.3003 |
| direct_rho_255.821_heldout | R2 | 0.5003 | 0.113 | 0.3119 | 0.6745 |
| direct_rho_255.821_heldout | RMSE_log | 0.3555 | 0.005382 | 0.3461 | 0.3653 |
| direct_rho_255.821_heldout | COD | 27.31 | 0.4858 | 26.42 | 28.18 |
| direct_rho_255.821_heldout | VEI | -2.172 | 3.461 | -8.733 | 4.569 |
| direct_rho_255.821_heldout | PRD | 1.063 | 0.006514 | 1.053 | 1.077 |
| direct_rho_255.821_heldout | PRB | -0.02673 | 0.006202 | -0.03816 | -0.01542 |
| direct_rho_255.821_heldout | MKI | 0.8898 | 0.01606 | 0.8608 | 0.9167 |
| direct_rho_255.821_heldout | Beta_log | -0.2015 | 0.01074 | -0.2255 | -0.1814 |
| direct_rho_31.648_heldout | R2 | 0.523 | 0.1119 | 0.3341 | 0.7023 |
| direct_rho_31.648_heldout | RMSE_log | 0.338 | 0.004797 | 0.3292 | 0.3477 |
| direct_rho_31.648_heldout | COD | 26.23 | 0.4435 | 25.38 | 27.09 |
| direct_rho_31.648_heldout | VEI | -13.99 | 3.177 | -19.51 | -8.148 |
| direct_rho_31.648_heldout | PRD | 1.075 | 0.006368 | 1.064 | 1.088 |
| direct_rho_31.648_heldout | PRB | -0.07364 | 0.00789 | -0.08725 | -0.0592 |
| direct_rho_31.648_heldout | MKI | 0.8622 | 0.01477 | 0.8321 | 0.8912 |
| direct_rho_31.648_heldout | Beta_log | -0.2394 | 0.009195 | -0.2593 | -0.2218 |
| direct_rho_7.912_heldout | R2 | 0.544 | 0.1108 | 0.3581 | 0.7232 |
| direct_rho_7.912_heldout | RMSE_log | 0.3343 | 0.00463 | 0.3263 | 0.3441 |
| direct_rho_7.912_heldout | COD | 25.72 | 0.4109 | 25 | 26.53 |
| direct_rho_7.912_heldout | VEI | -27.3 | 2.961 | -32.94 | -21.58 |
| direct_rho_7.912_heldout | PRD | 1.093 | 0.005948 | 1.083 | 1.105 |
| direct_rho_7.912_heldout | PRB | -0.1315 | 0.009285 | -0.1493 | -0.1138 |
| direct_rho_7.912_heldout | MKI | 0.8104 | 0.01308 | 0.783 | 0.8372 |
| direct_rho_7.912_heldout | Beta_log | -0.2866 | 0.008544 | -0.3053 | -0.2711 |
| direct_rho_71.208_heldout | R2 | 0.5087 | 0.1138 | 0.3173 | 0.689 |
| direct_rho_71.208_heldout | RMSE_log | 0.3433 | 0.005186 | 0.3339 | 0.3528 |
| direct_rho_71.208_heldout | COD | 26.63 | 0.4299 | 25.87 | 27.42 |
| direct_rho_71.208_heldout | VEI | -9.297 | 2.984 | -14.07 | -2.985 |
| direct_rho_71.208_heldout | PRD | 1.07 | 0.00633 | 1.059 | 1.083 |
| direct_rho_71.208_heldout | PRB | -0.05346 | 0.006351 | -0.06445 | -0.04195 |
| direct_rho_71.208_heldout | MKI | 0.8751 | 0.01505 | 0.8457 | 0.9019 |
| direct_rho_71.208_heldout | Beta_log | -0.2212 | 0.009581 | -0.2413 | -0.2032 |
| surrogate_rho_15.9314_heldout | R2 | 0.5447 | 0.1167 | 0.348 | 0.7303 |
| surrogate_rho_15.9314_heldout | RMSE_log | 0.3533 | 0.004462 | 0.3452 | 0.362 |
| surrogate_rho_15.9314_heldout | COD | 25.53 | 0.4421 | 24.78 | 26.43 |
| surrogate_rho_15.9314_heldout | VEI | -13.7 | 2.591 | -18.88 | -8.29 |
| surrogate_rho_15.9314_heldout | PRD | 1.092 | 0.006029 | 1.081 | 1.104 |
| surrogate_rho_15.9314_heldout | PRB | -0.08498 | 0.00826 | -0.0996 | -0.07067 |
| surrogate_rho_15.9314_heldout | MKI | 0.7906 | 0.01088 | 0.7666 | 0.8095 |
| surrogate_rho_15.9314_heldout | Beta_log | -0.2452 | 0.00863 | -0.2631 | -0.2312 |
| surrogate_rho_2.08642_heldout | R2 | 0.5431 | 0.1196 | 0.3429 | 0.7316 |
| surrogate_rho_2.08642_heldout | RMSE_log | 0.3329 | 0.005227 | 0.3231 | 0.3436 |
| surrogate_rho_2.08642_heldout | COD | 24.63 | 0.4921 | 23.66 | 25.61 |
| surrogate_rho_2.08642_heldout | VEI | -34.5 | 3.403 | -41.62 | -28.2 |
| surrogate_rho_2.08642_heldout | PRD | 1.108 | 0.006319 | 1.097 | 1.121 |
| surrogate_rho_2.08642_heldout | PRB | -0.1677 | 0.009174 | -0.1838 | -0.1494 |
| surrogate_rho_2.08642_heldout | MKI | 0.7526 | 0.0118 | 0.7289 | 0.7731 |
| surrogate_rho_2.08642_heldout | Beta_log | -0.3082 | 0.009084 | -0.327 | -0.2916 |
| surrogate_rho_2.18408_heldout | R2 | 0.5458 | 0.1202 | 0.3442 | 0.734 |
| surrogate_rho_2.18408_heldout | RMSE_log | 0.3339 | 0.00522 | 0.3241 | 0.3441 |
| surrogate_rho_2.18408_heldout | COD | 24.66 | 0.5057 | 23.64 | 25.61 |
| surrogate_rho_2.18408_heldout | VEI | -35.47 | 3.328 | -41.12 | -28.29 |
| surrogate_rho_2.18408_heldout | PRD | 1.107 | 0.00617 | 1.096 | 1.12 |
| surrogate_rho_2.18408_heldout | PRB | -0.1661 | 0.009323 | -0.1822 | -0.1472 |
| surrogate_rho_2.18408_heldout | MKI | 0.7536 | 0.01156 | 0.7309 | 0.7734 |
| surrogate_rho_2.18408_heldout | Beta_log | -0.3068 | 0.008911 | -0.3252 | -0.2906 |
| surrogate_rho_24.6697_heldout | R2 | 0.5405 | 0.1175 | 0.3419 | 0.7266 |
| surrogate_rho_24.6697_heldout | RMSE_log | 0.3628 | 0.004635 | 0.3541 | 0.3714 |
| surrogate_rho_24.6697_heldout | COD | 25.97 | 0.435 | 25.23 | 26.89 |
| surrogate_rho_24.6697_heldout | VEI | -8.644 | 3.348 | -14.38 | -2.316 |
| surrogate_rho_24.6697_heldout | PRD | 1.085 | 0.00592 | 1.074 | 1.097 |
| surrogate_rho_24.6697_heldout | PRB | -0.05762 | 0.008152 | -0.07314 | -0.04243 |
| surrogate_rho_24.6697_heldout | MKI | 0.8094 | 0.01076 | 0.7876 | 0.8275 |
| surrogate_rho_24.6697_heldout | Beta_log | -0.2224 | 0.008526 | -0.2403 | -0.2073 |

### St. Louis County, MO

| method | metric | mean | std | ci_2_5 | ci_97_5 |
| --- | --- | --- | --- | --- | --- |
| baseline_lgbm | R2 | 0.685 | 0.05787 | 0.5723 | 0.7796 |
| baseline_lgbm | RMSE_log | 0.3292 | 0.004787 | 0.3191 | 0.3375 |
| baseline_lgbm | COD | 23.76 | 0.2924 | 23.14 | 24.26 |
| baseline_lgbm | VEI | -27.29 | 1.365 | -29.96 | -24.78 |
| baseline_lgbm | PRD | 1.107 | 0.005298 | 1.096 | 1.117 |
| baseline_lgbm | PRB | -0.05593 | 0.003386 | -0.06219 | -0.0492 |
| baseline_lgbm | MKI | 0.89 | 0.005427 | 0.8795 | 0.8996 |
| baseline_lgbm | Beta_log | -0.1452 | 0.005284 | -0.1557 | -0.1353 |
| baseline_lr | R2 | -4.125e+08 | 4.184e+08 | -1.275e+09 | 0.3391 |
| baseline_lr | RMSE_log | 0.9231 | 0.01442 | 0.8927 | 0.9466 |
| baseline_lr | COD | 6339 | 6096 | 51.15 | 1.942e+04 |
| baseline_lr | VEI | 35.89 | 20.78 | -7.361 | 61.04 |
| baseline_lr | PRD | 0.7042 | 0.4162 | 0.3778 | 1.268 |
| baseline_lr | PRB | 1250 | 1202 | -0.06661 | 3826 |
| baseline_lr | MKI | 1.526 | 0.5938 | 0.7229 | 2.004 |
| baseline_lr | Beta_log | -0.4062 | 0.007866 | -0.4206 | -0.3903 |
| direct_rho_0.46534_heldout | R2 | 0.6897 | 0.05766 | 0.5769 | 0.7823 |
| direct_rho_0.46534_heldout | RMSE_log | 0.3308 | 0.004941 | 0.3203 | 0.3396 |
| direct_rho_0.46534_heldout | COD | 23.74 | 0.2793 | 23.17 | 24.23 |
| direct_rho_0.46534_heldout | VEI | -23.14 | 1.32 | -25.44 | -20.59 |
| direct_rho_0.46534_heldout | PRD | 1.098 | 0.005317 | 1.088 | 1.107 |
| direct_rho_0.46534_heldout | PRB | -0.04512 | 0.003661 | -0.05194 | -0.03736 |
| direct_rho_0.46534_heldout | MKI | 0.9009 | 0.005555 | 0.8904 | 0.9113 |
| direct_rho_0.46534_heldout | Beta_log | -0.1336 | 0.005474 | -0.1445 | -0.1235 |
| direct_rho_1.39602_heldout | R2 | 0.6998 | 0.05771 | 0.5862 | 0.7919 |
| direct_rho_1.39602_heldout | RMSE_log | 0.3313 | 0.004848 | 0.3209 | 0.34 |
| direct_rho_1.39602_heldout | COD | 23.71 | 0.2792 | 23.11 | 24.21 |
| direct_rho_1.39602_heldout | VEI | -17.3 | 1.36 | -19.97 | -14.74 |
| direct_rho_1.39602_heldout | PRD | 1.084 | 0.005217 | 1.074 | 1.093 |
| direct_rho_1.39602_heldout | PRB | -0.02978 | 0.003561 | -0.03661 | -0.02192 |
| direct_rho_1.39602_heldout | MKI | 0.9177 | 0.005613 | 0.9068 | 0.9281 |
| direct_rho_1.39602_heldout | Beta_log | -0.1162 | 0.005499 | -0.1272 | -0.106 |
| direct_rho_135.414_heldout | R2 | 0.7017 | 0.05487 | 0.5979 | 0.7969 |
| direct_rho_135.414_heldout | RMSE_log | 0.3468 | 0.004323 | 0.3362 | 0.3541 |
| direct_rho_135.414_heldout | COD | 25.41 | 0.2582 | 24.84 | 25.86 |
| direct_rho_135.414_heldout | VEI | 4.825 | 1.297 | 2.617 | 7.161 |
| direct_rho_135.414_heldout | PRD | 1.029 | 0.004619 | 1.021 | 1.038 |
| direct_rho_135.414_heldout | PRB | 0.03297 | 0.003448 | 0.02682 | 0.04008 |
| direct_rho_135.414_heldout | MKI | 0.9884 | 0.005634 | 0.9778 | 0.9982 |
| direct_rho_135.414_heldout | Beta_log | -0.04693 | 0.005917 | -0.05861 | -0.03682 |
| direct_rho_16.7522_heldout | R2 | 0.7141 | 0.05693 | 0.6039 | 0.8086 |
| direct_rho_16.7522_heldout | RMSE_log | 0.336 | 0.004374 | 0.3252 | 0.3433 |
| direct_rho_16.7522_heldout | COD | 24.2 | 0.2559 | 23.66 | 24.65 |
| direct_rho_16.7522_heldout | VEI | 0.09558 | 1.154 | -1.904 | 2.406 |
| direct_rho_16.7522_heldout | PRD | 1.041 | 0.004613 | 1.032 | 1.05 |
| direct_rho_16.7522_heldout | PRB | 0.01781 | 0.003338 | 0.01173 | 0.02473 |
| direct_rho_16.7522_heldout | MKI | 0.9724 | 0.005505 | 0.9616 | 0.9824 |
| direct_rho_16.7522_heldout | Beta_log | -0.06039 | 0.005638 | -0.07155 | -0.0502 |
| direct_rho_37.6925_heldout | R2 | 0.7088 | 0.05655 | 0.5992 | 0.8022 |
| direct_rho_37.6925_heldout | RMSE_log | 0.3415 | 0.004446 | 0.3313 | 0.3491 |
| direct_rho_37.6925_heldout | COD | 24.64 | 0.2541 | 24.14 | 25.08 |
| direct_rho_37.6925_heldout | VEI | 2.481 | 1.285 | 0.1583 | 5.222 |
| direct_rho_37.6925_heldout | PRD | 1.034 | 0.004692 | 1.025 | 1.043 |
| direct_rho_37.6925_heldout | PRB | 0.02668 | 0.00354 | 0.02019 | 0.0335 |
| direct_rho_37.6925_heldout | MKI | 0.9815 | 0.005652 | 0.9706 | 0.9917 |
| direct_rho_37.6925_heldout | Beta_log | -0.05141 | 0.005939 | -0.06313 | -0.04065 |
| direct_rho_4.18806_heldout | R2 | 0.7107 | 0.05736 | 0.5969 | 0.8022 |
| direct_rho_4.18806_heldout | RMSE_log | 0.3317 | 0.004524 | 0.3212 | 0.3399 |
| direct_rho_4.18806_heldout | COD | 23.76 | 0.2549 | 23.22 | 24.22 |
| direct_rho_4.18806_heldout | VEI | -9.722 | 1.272 | -11.94 | -7.23 |
| direct_rho_4.18806_heldout | PRD | 1.065 | 0.005013 | 1.055 | 1.074 |
| direct_rho_4.18806_heldout | PRB | -0.007406 | 0.003372 | -0.01387 | -5.356e-05 |
| direct_rho_4.18806_heldout | MKI | 0.9426 | 0.005663 | 0.9318 | 0.9531 |
| direct_rho_4.18806_heldout | Beta_log | -0.0903 | 0.005593 | -0.1016 | -0.07982 |
| direct_rho_8.50302_heldout | R2 | 0.7134 | 0.05802 | 0.5988 | 0.8075 |
| direct_rho_8.50302_heldout | RMSE_log | 0.3347 | 0.00449 | 0.3241 | 0.3424 |
| direct_rho_8.50302_heldout | COD | 24.01 | 0.2481 | 23.46 | 24.44 |
| direct_rho_8.50302_heldout | VEI | -4.688 | 1.256 | -7.059 | -2.23 |
| direct_rho_8.50302_heldout | PRD | 1.051 | 0.00489 | 1.042 | 1.061 |
| direct_rho_8.50302_heldout | PRB | 0.006572 | 0.00346 | -5.322e-05 | 0.01418 |
| direct_rho_8.50302_heldout | MKI | 0.9596 | 0.005718 | 0.9485 | 0.9702 |
| direct_rho_8.50302_heldout | Beta_log | -0.07431 | 0.005727 | -0.08569 | -0.06355 |
| surrogate_rho_0.171992_heldout | R2 | 0.6914 | 0.05869 | 0.5757 | 0.7855 |
| surrogate_rho_0.171992_heldout | RMSE_log | 0.3383 | 0.005589 | 0.3262 | 0.3484 |
| surrogate_rho_0.171992_heldout | COD | 24.11 | 0.3034 | 23.52 | 24.64 |
| surrogate_rho_0.171992_heldout | VEI | -21.48 | 1.406 | -23.95 | -18.86 |
| surrogate_rho_0.171992_heldout | PRD | 1.094 | 0.005432 | 1.084 | 1.104 |
| surrogate_rho_0.171992_heldout | PRB | -0.03794 | 0.003657 | -0.04458 | -0.0303 |
| surrogate_rho_0.171992_heldout | MKI | 0.9048 | 0.005654 | 0.8939 | 0.9153 |
| surrogate_rho_0.171992_heldout | Beta_log | -0.1271 | 0.005694 | -0.1383 | -0.117 |
| surrogate_rho_0.830124_heldout | R2 | 0.698 | 0.05981 | 0.5786 | 0.793 |
| surrogate_rho_0.830124_heldout | RMSE_log | 0.3534 | 0.006726 | 0.3398 | 0.3671 |
| surrogate_rho_0.830124_heldout | COD | 24.75 | 0.3314 | 24.11 | 25.32 |
| surrogate_rho_0.830124_heldout | VEI | -12.67 | 1.53 | -15.45 | -9.635 |
| surrogate_rho_0.830124_heldout | PRD | 1.073 | 0.005539 | 1.063 | 1.084 |
| surrogate_rho_0.830124_heldout | PRB | -0.005866 | 0.00371 | -0.01303 | 0.001311 |
| surrogate_rho_0.830124_heldout | MKI | 0.9279 | 0.006021 | 0.9163 | 0.9384 |
| surrogate_rho_0.830124_heldout | Beta_log | -0.09213 | 0.006253 | -0.1043 | -0.08174 |
| surrogate_rho_6.6733_heldout | R2 | 0.6966 | 0.06093 | 0.5748 | 0.7931 |
| surrogate_rho_6.6733_heldout | RMSE_log | 0.3844 | 0.006831 | 0.3711 | 0.3982 |
| surrogate_rho_6.6733_heldout | COD | 26.67 | 0.3676 | 26 | 27.33 |
| surrogate_rho_6.6733_heldout | VEI | -3.965 | 1.341 | -6.37 | -1.359 |
| surrogate_rho_6.6733_heldout | PRD | 1.045 | 0.00532 | 1.036 | 1.055 |
| surrogate_rho_6.6733_heldout | PRB | 0.04205 | 0.003672 | 0.03527 | 0.04903 |
| surrogate_rho_6.6733_heldout | MKI | 0.9588 | 0.006075 | 0.9476 | 0.9695 |
| surrogate_rho_6.6733_heldout | Beta_log | -0.03966 | 0.006373 | -0.05303 | -0.0287 |


## 17. Relationship to existing six ATTOM counties

See `source_concordance/existing_six_vs_v3_metadata.csv`. Those runs include Tax Assessor/ACS/location that v3 currently lacks. They remain a separate exploratory/sensitivity layer and were not overwritten.

## 18. Exact scientific interpretation

v3 is a **pre-rho** source-validation and standardized-AVM design. Cook County / CCAO remains the paper's primary application. Wayne County is not Detroit. St. Louis County is not St. Louis City. Official assessment ratios are not AVM valuation ratios.

## 19. Exact recommended paper additions/changes

**Do not edit the manuscript in this pass.** Recommended later:

1. Keep Cook County / CCAO as the primary application.
2. If freeze authorized transfer, add Wayne / Philadelphia / St. Louis County as a literature-anchored external-validation appendix, with Wayne **not** labeled Detroit, using HISTORY_MARKET_CORE (no Tax Assessor/ACS).
3. If freeze did not authorize transfer, report source validation + leakage-safe baselines, and state that Direct/Surrogate were not transferred because fewer than two new MODEL_TRANSFER units qualified.
4. Keep existing six-county ATTOM results as a separate sensitivity with a different feature class.
5. Do not cite Berry official assessment ratios as if they were model valuation ratios.
6. Always report unconditional linkage rates against eligible Berry N.

## 20. Remaining limitations

- Assessor History `PARCELNUMBERFORMATTED` / `PROPERTYJURISDICTIONNAME` are missing in this Dewey delivery, so Detroit-city ATTOM models are not a primary v3 product.
- St. Louis local extract does not reproduce an official assessment-ratio Berry benchmark.
- Year-end History snapshots can lag sales by more than a year; lag percentiles are reported, not tuned away.
- v3 primary features omit Tax Assessor/ACS/location; metrics are not comparable to old-six.
- The LR reference model's held-out level-space metrics are degenerate in Philadelphia and
  St. Louis County because exponentiating a log-space linear fit extrapolates without bound
  (section 13). LightGBM carries the science; no status, anchor, or conclusion uses LR.
- Philadelphia's modeled cohort survives a value-dependent property-use filter that keeps 8.75%
  of eligible sales (section 9a). Its vertical-equity numbers carry that selection.
- The St. Louis local-vs-ATTOM provider comparison has 38 validation rows and settles nothing
  (section 11).
- Surrogate anchors come from calibration pass 2; Philadelphia's held-out test rows were scored
  under both passes. Disclosed in `panel_freeze/SURROGATE_RECALIBRATION_LOG.md`.
- `HISTORY_STRUCTURAL_CORE` is validation-only in v3.
- Any Surrogate anchor still marked UNATTAINED after pass 2 carries a reason code; a
  `MATERIAL_REVERSAL` is a finding about the method on that county, not a missing run.

## Canonical jurisdiction table

| jurisdiction | ATTOM_geographic_unit | Berry_geographic_unit | Berry_N | Berry_reproduction_status | unconditional_APN_rate | unconditional_transaction_confirmed_rate | unconditional_safe_history_rate | unconditional_fully_validated_rate | Berry_PRB_full | Berry_PRB_fully_linked | model_eligible_Recorder_N | final_model_N | model_retention_decile_spread | validation_R2 | model_transfer_status | berry_anchor_status | heldout_R2 | heldout_baseline_PRB | heldout_beta_log | Direct_run | Surrogate_run | main_scientific_role | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wayne County, MI | Wayne County, MI | Detroit, MI | 9653 | PYTHON_TRANSLATION_MATCHES_V1_FILTERS | 0.9013 | 0.7078 | 0.8992 | 0.706 | -0.3495 | -0.3936 | 381624 | 182042 | 0.1476 | 0.8231 | PRIMARY | STRONG | 0.795 | -0.1087 | -0.2137 | yes | yes | Wayne County AVM + Detroit city Berry anchor | see freeze notes |
| Philadelphia County, PA | Philadelphia County, PA | Philadelphia, PA | 4.132e+04 | RESOLVED_USE_ARMS_LENGTH_FILE_NOT_RBIND | 0.9152 | 0.8691 | 0.9089 | 0.8651 | -0.1198 | -0.1245 | 434735 | 38043 | 0.1474 | 0.7795 | PRIMARY | MODERATE | 0.5051 | -0.2179 | -0.3418 | yes | yes | Philadelphia AVM + Berry validation | see freeze notes |
| St. Louis County, MO | St. Louis County, MO | St. Louis County, MO | 5.483e+05 | REBUILT_FROM_2019_CUMULATIVE_SALES_EXTRACT | 0.9958 | 0.9349 | 0.9537 | 0.9 | nan | nan | 354318 | 144995 | 0.1666 | 0.7854 | PRIMARY | MODERATE | 0.6807 | -0.05627 | -0.1456 | yes | yes | St. Louis County AVM + local source robustness; not City 29510 | see freeze notes |

