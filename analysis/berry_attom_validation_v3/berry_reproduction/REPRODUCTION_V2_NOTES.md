# Berry reproduction v2

- Detroit: Python translation of Rmd filters; Parcel Number raw preserved. Native R attempted with module R/4.4.3; tidyverse packages missing so Python remains the numeric source.
- Philadelphia: arms_length file is canonical; total is not stacked.
- St. Louis: yearly sales extracts; SALEVAL==X; SALEDT recovered.

[
  {
    "jurisdiction": "detroit_mi",
    "n": 9653,
    "q2_2016_n": 975,
    "years": "2016-2018",
    "COD": 49.2628,
    "PRD": 1.3274,
    "PRB": -0.3495,
    "status": "PYTHON_TRANSLATION_MATCHES_V1_FILTERS",
    "unique_txn_key": "Parcel Number|Sale Date|Adj.Sale$",
    "native_r_attempted": false,
    "notes": "Rmd filter=FALSE; class 401; VALID ARMS LENGTH. Parcel Number preserved raw."
  },
  {
    "jurisdiction": "philadelphia_pa",
    "n_arms_length_raw": 41325,
    "n_total_raw": 59614,
    "n_unique_txn_al": 41325,
    "n_unique_txn_tot": 59613,
    "n_overlap_txn_keys": 0,
    "n_only_arms_length": 41325,
    "n_only_total": 59613,
    "n_canonical_al_deduped": 41325,
    "al_is_subset_of_total": false,
    "v1_rbind_would_duplicate": false,
    "unique_txn_key": "parcel|saledate|sale_price|rcddt",
    "COD_iqr_on_canonical": 21.2253,
    "PRD_iqr_on_canonical": 1.0842,
    "PRB_iqr_on_canonical": -0.0923,
    "n_iqr": 40509,
    "status": "RESOLVED_USE_ARMS_LENGTH_FILE_NOT_RBIND",
    "notes": "Canonical table is Ratio_Analysis_arms_length.dta unique txn_key. Total file is a broader universe, not a second independent sample to stack.",
    "n": 41325,
    "years": "2012-2018"
  },
  {
    "jurisdiction": "st_louis_county_mo",
    "n_concat_sales_rows": 1285496,
    "n_unique_txn_key": 1281423,
    "n_saleval_x_positive_price": 548307,
    "skipped_year_files": "2018/sales.TXT: SQL stub (636 bytes), not a data table",
    "canonical_file": "/orcd/home/002/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/data/berry_cmf/raw/st_louis_county_mo/box/rabph6sd546szpwe6likep763hv8g3v3/new data/2020-stlco-assessments/2019/sales.csv",
    "unique_txn_key": "PARID|SALEDT|PRICE|INSTRUNO|SALEVAL",
    "did_not_use_joined_csv": true,
    "did_not_use_aprtot_as_price": true,
    "status": "REBUILT_FROM_2019_CUMULATIVE_SALES_EXTRACT",
    "years": "1975-2019",
    "n": 548307,
    "n_dated": 548307,
    "notes": "2019 sales.csv is a cumulative history extract with SALEDT. Earlier yearly files are growing snapshots of the same table and were not stacked. SALEVAL==X and PRICE>0."
  }
]
