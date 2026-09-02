# Berry/ATTOM validation v2

Isolated follow-on to the frozen Berry/CMF Audit v1 (`analysis/berry_cmf_validation/`).
**Do not modify v1 artifacts.**

## What this study is

A. Validate Berry/local-government transactions and documented *assessment* regressivity against an independent ATTOM Recorder/History representation.

B. Build a standardized leakage-safe ATTOM AVM benchmark for Wayne County MI, Philadelphia County PA, and St. Louis County MO (2016–2025).

C. Run Direct/Surrogate paths **only after** `panel_freeze/final_panel_freeze_v2.yaml` is written, and only if at least two independent new modeling units pass.

Berry official assessment/sale ratios and model valuation/sale ratios are **different estimands**.

Wayne County ATTOM models must never be labeled “Detroit.” St. Louis County (FIPS 29189) must never be conflated with St. Louis City (29510).

The new Dewey folders also contain extra FIPS (notably Wayne County PA `42127`). Caches always filter by FIPS, never by folder name. Assessor History `PARCELNUMBERFORMATTED` and `PROPERTYJURISDICTIONNAME` are 100% missing in this extract.

## Layout

- Protocol (frozen before positive-rho): `protocol_v2.yaml`
- Compact tracked outputs: this directory (CSV/YAML/MD/JSON)
- Large derived tables: `output/berry_attom_validation_v2/` (gitignored)
- Raw Dewey and Berry files stay immutable under `data/`

## Canonical code reused (not duplicated)

- Sale validation: `scripts/other_counties_benchmars.py` + `data/dewey-downloads/data_dictionaries/attom_recorder_residential_avm_sale_validation_dictionary.csv`
- Metrics: `utils/motivation_utils.py` (`compute_taxation_metrics` / `_compute_extended_metrics`)
- Δ_NL: `utils/delta_nl.py`
- Direct: `soft_constrained_models.boosting_models.LGBCovPenalty`
- Surrogate: `soft_constrained_models.boosting_models.LGBSmoothPenalty`
- Rank-one rho: `scripts/theory_informed_rho_range_v2.py`
- LGBM candidate configs: `best_lgbm_baseline_configs.yaml`

## Forbidden

- Cookies, session files, or credentials in this tree
- Editing `paper/paper_v12.tex` in this pass
- Overwriting existing six-county ATTOM results
- Using positive-rho results to include/exclude a jurisdiction
- Using assessed/market/tax values as AVM predictors or targets
