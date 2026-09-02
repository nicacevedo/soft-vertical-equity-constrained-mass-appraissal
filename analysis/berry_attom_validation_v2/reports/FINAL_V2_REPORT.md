# Berry/ATTOM validation v2 — final report

Written: 2026-09-02T21:05:39Z  
Protocol: `analysis/berry_attom_validation_v2/protocol_v2.yaml`  
**Manuscript was not edited. Frozen v1 under `analysis/berry_cmf_validation/` was not modified.**

This report distinguishes Berry *official assessment/sale* ratios from model
*valuation/sale* ratios. Those estimands are never mixed.

## 1. Berry/local external assessment-regressivity validation

See `berry_reproduction/REPRODUCTION_V2_NOTES.md` and `reproduction_summary.csv`.

- Detroit: Python translation of the Rmd filters (class 401, VALID ARMS LENGTH,
  `filter=FALSE` in `cmfproperty::reformat_data`). Native R comparison is
  attempted separately when the `cmfproperty` sources can be sourced.
- Philadelphia: canonical table is `Ratio_Analysis_arms_length.dta` only. The
  total file is a disjoint universe, not a duplicate sample to rbind.
- St. Louis County: rebuilt from the 2019 cumulative `sales.csv` extract with
  actual `SALEDT` and `PRICE`. `joined.csv` and `SALE_PRICE=APRTOT` were not used.

## 2. Berry↔ATTOM transaction/linkage validation

See `linkage/linkage_waterfall.csv`, `linkage/linkage_summary.json`, and
`linkage/transaction_concordance.parquet` (large; gitignored).

Parcel identity uses APN hierarchy only. One-to-many APN maps are `AMBIGUOUS_APN`
and are not collapsed. Price/date is never used as a parcel identifier.

## 3. Linkage-selection bias

See `linkage/matched_unmatched_balance.csv` and
`source_concordance/berry_full_vs_attom_linkable.csv`.
`figures/linkage_match_rate_by_price_decile.pdf` plots P(match | price decile).

If ATTOM matching flips the Berry PRB sign or shows a large price-decile
match-rate spread, the jurisdiction is flagged and is not quietly promoted.

## 4. ATTOM standardized baseline AVMs

Wayne County, Philadelphia County, and St. Louis County, 2016–2025, log
Recorder target, HISTORY_MARKET_CORE primary, HISTORY_STRUCTURAL_CORE secondary.
LR + ordinary LightGBM only, validation selection, 200 monthly block bootstraps.

Wayne County models are never labeled Detroit.

See `reports/BASELINE_REPORT.md`.

## 5. St. Louis local-vs-ATTOM provider robustness

See `reports/ST_LOUIS_SOURCE_ROBUSTNESS.md`. Local table uses actual PRICE and
SALEDT plus the latest dwelling snapshot strictly before sale, with an explicit
2012-folder/TAXYR=2013 sensitivity.

## 6. Frozen panel decision

`panel_freeze/final_panel_freeze_v2.yaml`

Direct/Surrogate authorized: **False**
Passing units: None

## 7. Direct/Surrogate transfer results

Not run. Either the freeze is not yet written or fewer than two independent new modeling units passed. Audit/baselines are the scientific outcome in that case.

## 8. Failure/boundary cases

- St. Louis City FIPS 29510 must not appear in the 29189 cache.
- Detroit Berry is a city sample; ATTOM AVM is Wayne County unless a validated
  city field supports a later sensitivity.
- Philadelphia v1 rbind concern is resolved: arms-length and total files are
  disjoint universes.
- Step 9 ATTOM-vs-Berry assessment-field comparison is skipped; see
  `source_concordance/STEP9_ASSESSMENT_CONCORDANCE.md`.

## 9. Relation to the existing six-county ATTOM analysis

See `source_concordance/existing_six_vs_v2_metadata.csv`. Those runs include
Tax Assessor/ACS/location features that v2 does not have. Do not present the
two pipelines as identical.

## 10. What Tax Assessor could add later

A later Tax Assessor extract could support a **separately labeled** location/ACS
enrichment sensitivity (coordinates, tract, current APN crosswalk). It must not
be folded into HISTORY_MARKET_CORE unless a new protocol is frozen first. Tax
values still must not become primary predictors.

## 11. Exact recommended paper additions (DO NOT EDIT THE PAPER IN THIS PASS)

1. Keep Cook County / CCAO as the primary application.
2. If v2 freeze authorizes transfer, add Wayne / Philadelphia / St. Louis County
   as a literature-anchored external-validation appendix, with Wayne **not**
   labeled Detroit, using HISTORY_MARKET_CORE (no Tax Assessor/ACS).
3. If the freeze does not authorize transfer, state that the new Dewey delivery
   was used to validate Berry transactions against ATTOM and to build leakage-safe
   baselines, and that Direct/Surrogate were not transferred because fewer than
   two new modeling units passed the pre-rho gates.
4. Keep the existing six-county ATTOM results as a separate sensitivity layer
   with a different feature class.
5. Do not cite Berry official assessment ratios as if they were model valuation
   ratios.

Inventory coverage rows: 37; FIPS-year rows: 711.

## Canonical table

| jurisdiction | geographic_unit | berry_role | berry_reproduction_status | berry_n | attom_recorder_n | attom_history_n | berry_parcel_match_rate | high_conf_transaction_match_rate | safe_history_match_rate | linkage_selection_risk | model_n | model_period | baseline_r2_price | baseline_prb | baseline_beta_log | baseline_delta_nl | baseline_dcor | panel_decision | direct_run | surrogate_run | main_scientific_role | confidence |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Wayne County, MI | Wayne County, MI | Detroit, MI | PYTHON_TRANSLATION_MATCHES_V1_FILTERS | 9653 | pending | pending | pending | pending | pending | pending | pending |  —  | pending | pending | pending | pending | pending | pending | no | no | Wayne County AVM + Detroit Berry validation | pending_until_gates_complete |
| Philadelphia County, PA | Philadelphia County, PA | Philadelphia, PA | RESOLVED_USE_ARMS_LENGTH_FILE_NOT_RBIND | 41325 | pending | pending | pending | pending | pending | pending | pending |  —  | pending | pending | pending | pending | pending | pending | no | no | Philadelphia AVM + Berry validation | pending_until_gates_complete |
| St. Louis County, MO | St. Louis County, MO | St. Louis County, MO | REBUILT_FROM_2019_CUMULATIVE_SALES_EXTRACT | 548307 | pending | pending | pending | pending | pending | pending | pending |  —  | pending | pending | pending | pending | pending | pending | no | no | St. Louis County AVM + Berry validation + provider robustness | pending_until_gates_complete |

