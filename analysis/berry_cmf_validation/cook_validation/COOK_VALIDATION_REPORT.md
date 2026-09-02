# Cook County three-source validation

Frozen: 2026-09-02T17:09:06Z
Canonical CCAO experiment was not modified.

## Distinct sources

1. **CCAO research/modeling data** — `data/CCAO/2025/training_data.parquet` (N=444692, sales 2016-01-01 to 2025-12-29).
2. **Berry/CMF local-government files** — Box `res_sales2002.csv`–`res_sales2015.csv` plus layout xlsx; assessed value `bor_CCAO_ass`, price `NetConsideration`.
3. **ATTOM Cook construction** — `scripts/other_counties_benchmars.py` on `data/dewey-downloads/cookcounty-2016-2025-all-features` (paper exploratory benchmark).

## 10A. Source agreement

Transaction-level matching between Berry/CMF and CCAO is **not defensible**: the CMF BOR extracts stop in 2015 and the CCAO modeling sample starts in 2016. PIN fields are conceptually compatible (`pin`/`pin10` vs `meta_pin`) but there is no shared sale year.

Berry/CMF vs ATTOM Cook is likewise disjoint (2015 vs 2016+ recorder transfers).

ATTOM vs CCAO overlap in calendar time (2016+) but **not** in institutional universe (ATTOM use code 385 + $50k floor + recorder amount vs CCAO residential workflow). Existing paper already treats them as separate constructions.

## 10B. Cohort agreement

See `cook_common_cohort_summary.csv`. Definitions differ; cohorts were not forced identical.

## 10C. External regressivity benchmark

Reproduced CMF Cook code path (class filter + IQR `reformat_data`) on 2002–2015 BOR sales: N=641,147, COD=19.14, PRD=1.039, PRB=-0.007. This is an **official-assessment / sale-price** ratio, not the paper’s model valuation ratio. No local CMF published table was available to score exact numeric match; the page “Report” is CoreLogic nationwide HTML and was not used as a target.

## 10D. Harmonized-model source validation

**Not run.** A common row set with common pre-sale features does not exist across all three sources. Running LR/LightGBM on mismatched years would not be source-robustness of the same experiment.

## Interpretation

Cook remains the paper’s primary AVM application via CCAO. Berry/CMF Cook files are useful as a **pre-2016 assessment-regressivity provenance check**, not as a substitute modeling sample. ATTOM Cook remains an exploratory sensitivity layer.
