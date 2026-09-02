# Berry/CMF external-data validation — final audit report

Frozen: 2026-09-02T18:05:00Z  
Protocol: `analysis/berry_cmf_validation/protocol.yaml`  
**Manuscript was not edited. Existing CCAO and ATTOM result files were not modified. Direct/Surrogate paths were not run.**

---

## 1. Repository state and exact code baseline

| Item | Value |
|---|---|
| Repository root | `/orcd/home/002/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal` (workspace alias `/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal`) |
| Branch | `testing` |
| HEAD | `a55d81e8a55b3edeb055c6e02db41ba3988587d6` |
| Tracked manuscript (not edited) | `paper/paper_v12.tex` — *Fair Property Taxation: A Covariance-Guided Correction for Assessor Workflows* |
| Working area | `analysis/berry_cmf_validation/` (no prior top-level `analysis/` convention) |
| External raw root | `data/berry_cmf/raw/<jurisdiction>/` (gitignored) |
| ATTOM raw root (untouched) | `data/dewey-downloads/` |
| Canonical CCAO sample | `data/CCAO/2025/training_data.parquet` (N=444,692; sales 2016-01-01 to 2025-12-29) |
| Canonical metrics | `utils/motivation_utils.py` (`compute_taxation_metrics`, PRD/PRB/COD/MKI/VEI, `paper_mechanism_metrics`) |
| Δ_NL | `utils/delta_nl.py::estimate_delta_nl` |
| Direct / Surrogate implementations (not executed here) | `soft_constrained_models.boosting_models` (`LGBCovPenalty`, `LGBSmoothPenalty`) |
| ATTOM benchmark code (not overwritten) | `scripts/other_counties_benchmars.py` |
| Python | `/home/nacevedo/.conda/envs/fairness_env/bin/python` (pandas 2.3.1, sklearn 1.6.1, lightgbm 4.6, dcor 0.6) |
| Stata | unavailable |
| R | available on the cluster; **not** used for numeric reproduction |

Git status at freeze of this report (audit-related): `.gitignore` modified to ignore `data/berry_cmf/` and un-ignore compact artifacts under `analysis/berry_cmf_validation/`. Unrelated dirty files that **this pass did not touch**: `paper/paper_v12.tex`, `scripts/analyze_rho_screening_region_v2_1.py`, and untracked `paper/img/generated_v12_994/*.pdf`.

HEAD SHA and protocol SHA-256:

- protocol.yaml: `5a71f358acabe1719b677d435d528f230863644e6bdbd396648531ab7a3575da`

---

## 2. Executive scientific conclusion

Berry/CMF local-government replication data **do strengthen independent provenance for documented assessment regressivity**, especially in Detroit. They **do not**, in this pass, yield a confirmatory multi-jurisdiction AVM panel on which the paper’s Direct/Surrogate method can be transferred.

Only **St. Louis County** supplies leakage-safe yearly structural snapshots plus a sale target that can support a standalone AVM. Even there, sale **day** is missing from the CMF `joined.csv`, the 2012 dwelling extract is mislabeled as tax year 2013, and the published CMF `make_report` snippet uses appraised value as if it were sale price. Cook County cannot serve as a three-source modeling overlap: CMF BOR files end in 2015 and the canonical CCAO sample starts in 2016.

No primary jurisdiction remained `D_ATTOM_ENRICHMENT_CANDIDATE`. The existing six-county ATTOM benchmark is therefore **retained as a separate exploratory sensitivity layer**, including Miami-Dade as a boundary case. Official assessed value was never treated as the AVM target.

**Step 14 (full Direct/Surrogate experiment) was not scientifically justified and was not run.**

---

## 3. What CMF/Berry actually provides for every candidate

Official hub: https://propertytaxproject.uchicago.edu/replication/  
Local reports: https://propertytaxproject.uchicago.edu/papers/

CMF states that nationwide CoreLogic microdata **cannot be redistributed**. Several jurisdiction pages labeled “Report” are CoreLogic HTML county pages, **not** local-government studies. Those HTML summaries were not used as replication targets.

| Jurisdiction | What CMF actually shipped | What it is *not* |
|---|---|---|
| Detroit, MI | Ratio-study workbook, `combined.csv`, assessor xlsx, Rmd, 2020 ratio-study PDF, CMF local-report PDF/zip | A leakage-safe AVM feature extract; not Wayne County as a whole |
| Philadelphia, PA | OPA ratio-analysis Stata files + a five-column clean CSV | A published local N/COD/PRD table; page Report is CoreLogic |
| Orleans Parish, LA | `nola.R` / scrape CSVs / RDS; README documents a GitHub scrape after a ~$75k FOIA quote | Certified historical assessor rolls |
| Franklin / Columbus, OH | `columbusonly.csv` only (city extract) | Full Franklin County `full.csv` or the cited GDB; Report link is Columbus County **NC** CoreLogic |
| St. Louis County, MO | Yearly 2009–2019 assessment extracts (~7.6–8.1 GB), `joined.csv`, `fullroll.csv`, dictionary PDF | A valid published sale-ratio `make_report` path (snippet sets `SALE_PRICE=APRTOT`) |
| Cook County, IL | `res_sales2002`–`2015.csv` BOR extracts | Overlap with CCAO 2016–2025; page Report is CoreLogic |
| NYC, Buffalo, Clark, LA, Maricopa | Inventory of Box links; large files not downloaded (Buffalo sales xlsx was) | This-pass AVM sources |

---

## 4. Raw-data provenance

Downloader: `analysis/berry_cmf_validation/scripts/download_cmf_artifacts.py`  
Manifest: `file_manifest.csv` (521 rows; **465 DOWNLOADED**, 33 FOLDER_LISTED, 22 SKIPPED as secondary-large, 1 git clone of `cmfproperty`).

Approximate downloaded bytes by jurisdiction: St. Louis 8.13 GB; Cook 593 MB; Orleans 293 MB; Philadelphia 117 MB; Detroit 21 MB; Erie/Buffalo 21 MB; Franklin 10 MB; NYC inventory snippets 2.6 MB; Clark snippets 2.2 MB. Total ≈ 9.19 GB.

Raw files were not altered after download. SHA-256 values are in `file_manifest.csv`. Selected analytic files:

| File | SHA-256 prefix | Bytes |
|---|---|---|
| Detroit `combined.csv` | `578bac7955a9cdd7…` | 2,758,158 |
| Detroit `detroit_replication_code.Rmd` | `1a6b307c4e4134af…` | 6,402 |
| Columbus `columbusonly.csv` | `b20652e97fbd660b…` | 10,352,161 |
| St. Louis `joined.csv` | `7b6f7bb624352bc7…` | 10,959,091 |
| Cook `res_sales2015.csv` | `14fcdcaa0c2d1f43…` | 5,216,924 |

Box `HEAD` on `uchicago.box.com` often 404s; GET via `uchicago.app.box.com` worked. Folder-zip API failed; recursive file download was used. Secondary large files were skipped by protocol.

**Inaccessible / not obtained by design**

- CMF CoreLogic nationwide microdata (license; not pursued).
- Franklin County `full.csv` and `FCA_SDE_Web_Prod.gdb`.
- OpenDataPhilly bulk assessment history (not acquired; timing not confirmed).
- NYC ~3.9 GB data folder, Clark yearly CSVs, LA merged/assessor files, Maricopa `PHX_*` (secondary skip).
- Official Orleans historical rolls (FOIA cost).

---

## 5. Replication success/failure

Berry/CMF **assessment-to-sale** ratios were reproduced with a Python translation of `cmfproperty::reformat_data` (default IQR “arms length”) and `iaao_stats`. This is **not** the paper’s AVM metric object. Logs: `logs/reproduction/*.log`.

| Jurisdiction | Status | Original vs reproduced |
|---|---|---|
| Detroit | **EXACT_OR_ROUNDING_MATCH** | Class 401, VALID ARMS LENGTH, `filter=FALSE`. 2016Q2–2018Q1 N=9,653; 2016Q2 N=975 exact. Before/after through 2017Q1: COD 46.2205/50.1694 vs 46.26/50.08; PRD 1.3012/1.3466 vs 1.30/1.35; PRB -0.2574/-0.4458 vs -0.26/-0.45. |
| Philadelphia | PARTIAL_REPRODUCTION | rbind both Stata files + default IQR. N=88,811 (2012–2018), COD 35.70, PRD 1.30, PRB -0.11. No local published table; rbind may duplicate. |
| Orleans | PARTIAL_REPRODUCTION | Rebuilt `nola.R` join; `/0.1` AV scaling; **skipped tidycensus tract join**. N=30,939 (2015–2019), COD 46.68, PRD 1.24, PRB -0.075. |
| Franklin/Columbus | PARTIAL_REPRODUCTION | City extract only; LAND_CLASS 510–530 + IQR. N=186,231 (2002–2018), COD 33.96, PRD 1.17, PRB -0.20. No local published target; wrong-geography CoreLogic “Report”. |
| St. Louis | PARTIAL_REPRODUCTION **with code defect** | As-written `SALE_PRICE=APRTOT` on full roll: N=3,715,468, COD≈0.012, PRD=1.0 (not a sale-ratio study). **Intended sold-only** `SALEVAL==X` + IQR: N=73,743, COD 11.87, PRD 1.015, PRB -0.007. Use sold-only for any sale-ratio claim. |
| Cook CMF | PARTIAL_REPRODUCTION | `res_sales2002–2015`, residential `bor_class`, NetConsideration / `bor_CCAO_ass` / `joinyr` + IQR. N=641,147, COD 19.14, PRD 1.039, PRB -0.007. No local published table. |

Details: `reproduction/reproduction_results.csv`, `reproduction/REPRODUCTION_NOTES.md`.

---

## 6. Feature and temporal-coverage audit

Schema inventory: `schema_inventory.parquet` (~4,973 field-rows). Coverage: `feature_coverage_matrix.csv`. Timing: `feature_coverage/temporal_integrity_audit.csv`.

Rules applied: current characteristics attached retrospectively to old sales are **not** automatically `SAFE_AS_OF_SALE`. `TIMING_UNRESOLVED` and `POST_SALE_LEAKAGE` variables were barred from any main AVM.

- **Detroit:** sale price/date and “Asd. when Sold” are usable as outcomes/diagnostics. Floor Area / style: `TIMING_UNRESOLVED` (Appr. Date often empty). No year built, lot, beds/baths in `combined.csv`.
- **Philadelphia:** `sale_price` / `assmt_at_sale` OK as outcome/diagnostic. `totlivarea`, `yrbuilt`, baths: `TIMING_UNRESOLVED`.
- **Orleans:** `properties.csv` building/land area: `POST_SALE_LEAKAGE` (scrape `updated_at`). Yearly `values.csv` can benchmark assessment-at-year, not AVM predictors.
- **Franklin:** no structural predictors in the Box extract.
- **St. Louis:** yearly `dwelling` snapshots with dictionary fields `SFLA`, `YRBLT`, `RMBED`, `FIXBATH`, `STYLE`, `GRADE`, `CDU` are `SAFE_AS_OF_SALE` **if joined on PARID + TAXYR matching sale year (or prior)**. File headers are truncated (37 vs 39 fields; 2013 comma-delimited; 2019 leading blank line). `joined.csv` **drops `SALEDT`**. `APRTOT` is `NOT_APPLICABLE` as a sale price.
- **Cook CMF:** `area` on BOR extracts is `TIMING_UNRESOLVED`. CCAO `char_*` remain the primary SAFE application features (unchanged).

St. Louis dwelling idiosyncrasy discovered during load: folder `2012/DWELLING.csv` has `TAXYR` mode **2013**, so 2012 sales have **no same-year dwelling snapshot**.

---

## 7. Standalone AVM feasibility

Classified **before** any Direct/Surrogate run (`modeling_readiness/jurisdiction_usability_matrix.csv`):

| Jurisdiction | Class |
|---|---|
| Detroit | `A_EXTERNAL_REGRESSIVITY_ONLY` |
| Philadelphia | `C_PUBLIC_ENRICHMENT_CANDIDATE` |
| Orleans | `A_EXTERNAL_REGRESSIVITY_ONLY` |
| Franklin | `A_EXTERNAL_REGRESSIVITY_ONLY` |
| **St. Louis County** | **`B_STANDALONE_AVM_READY`** |
| Cook CMF 2002–2015 | `A_EXTERNAL_REGRESSIVITY_ONLY` |
| NYC, Buffalo, Clark, LA, Maricopa | `E_NOT_RECOMMENDED` (inventory / leakage / code quality / already in ATTOM) |

Berry assessment ratios are **not** interchangeable with model valuation ratios. Official assessed value was not used as an AVM target.

---

## 8. Public enrichment feasibility

`public_enrichment/public_enrichment_audit.csv`

- Philadelphia OpenDataPhilly OPA history is the only clearly documented public enrichment path. It was **not** bulk-downloaded: yearly assessments are useful, but characteristic as-of-sale validity was not demonstrated. Usability remains `C`.
- Franklin live auditor GIS was not used as a substitute for the missing historical GDB.
- Detroit / Orleans: CMF already contains the relevant local files; additional portal scrapes would not create historical snapshots.
- St. Louis: yearly extracts already in Box; public enrichment not required.
- Cook: do not replace CCAO with CMF 2002–2015 files.

---

## 9. ATTOM hybrid feasibility

No jurisdiction remained `D_ATTOM_ENRICHMENT_CANDIDATE` after the public-enrichment gate. Hybrid ATTOM linkage was **not attempted** (`linkage/attom_join_feasibility.csv`). Existing ATTOM six-county assets were not overwritten.

---

## 10. Linkage-selection diagnostics

`linkage/matched_unmatched_balance.csv` records that no hybrid match was performed, so there is no matched/unmatched price/year/ratio selection audit for ATTOM joins.

St. Louis **dwelling join** (not ATTOM): of 78,936 `SALEVAL==X` sales with price in \[$10k, $2M\], 71,208 matched a same-year dwelling card (7,728 unmatched). After living-area / year-built filters, N=70,864. Unmatched 2012 sales are mechanically expected given the mislabeled 2012 extract. A full matched-vs-unmatched price/year balance table for those 7,728 rows was not a Step-9 ATTOM deliverable; it is a remaining construction risk if STL is promoted beyond a boundary case.

---

## 11. Cook three-source validation

See `cook_validation/COOK_VALIDATION_REPORT.md` and accompanying CSVs.

Treat as **three distinct sources**:

1. CCAO `training_data.parquet` — N=444,692, 2016-01-01 to 2025-12-29 (canonical; not modified).
2. Berry/CMF `res_sales2002–2015` — N=641,147 after class+IQR, 2002–2015.
3. ATTOM Cook exploratory construction — paper table N=109,793 eligible (not recomputed here).

**10A.** Transaction-level matching CMF↔CCAO or CMF↔ATTOM is **not defensible**: empty year overlap (2015 vs 2016+). PIN formats are conceptually compatible but unused.

**10B.** Cohorts were **not** forced identical. Definitions differ (BOR residential classes vs CCAO modeling group vs ATTOM use code 385 + $50k floor).

**10C.** CMF Cook official-assessment/sale-price ratio: COD 19.14, PRD 1.039, PRB -0.007. This is **not** the paper’s model valuation ratio. No local CMF published table was available; CoreLogic HTML was not used as a target.

**10D.** Harmonized LR/LightGBM source-robustness **was not run**: there is no common sale cohort with a common pre-sale feature set across the three sources.

Cook remains the paper’s primary AVM application via CCAO. CMF Cook is pre-2016 assessment-regressivity provenance only.

---

## 12. Harmonized feature schema

`modeling_readiness/harmonized_feature_schema.yaml`

`CORE_HARMONIZED` (do not force 100% coverage): sale price, sale date, building area, year built, lot area (optional), baths/beds (optional), residential class/style, geography id, sale-time variables.

Jurisdictions with a currently credible CORE among **new CMF sources**: **St. Louis County only**. CCAO is richer and must not be reduced to CORE. `LOCAL_ENRICHED` for STL: GRADE, CDU, STYLE, STORIES, CLASS, LUC.

Smoke-test note: `joined.csv` has no sale day and the smoke model omitted a sale-year/time feature, which contributes to held-out **level** under-prediction (median ratio 0.85). Any future STL AVM should include a prespecified sale-time variable. Lot area was not in the dwelling card; geography id was not used in the smoke feature list.

---

## 13. Baseline smoke-test results

Only St. Louis passed the AVM gate. Other primaries: `NOT_RUN_FAILED_GATE`.

Protocol: log sale price; chronological year split (dev ≤2016, val 2017–2018, held-out ≥2019); selection on validation RMSE only; canonical metrics. Features: `SFLA, YRBLT, RMBED, FIXBATH, STORIES, STYLE, GRADE, CDU`. `FIXHALF` omitted (47% missing). Modeling table SHA-256: `0fed83935e46ebd5f2d63bbb4c6641bba396867c267fd4fb110c2a1bcaf501ea`.

Selected LGBM (validation only): `n_estimators=400, learning_rate=0.05, num_leaves=63, min_child_samples=20, random_state=2025`.

**Held-out (2019, n=8,166)**

| Model | R²_price | RMSE_log | MAE | MAPE | Median ratio | COD | PRD | PRB | MKI | VEI | β_log | Δ_NL | dCor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Linear Regression | 0.742 | 0.310 | 71,719 | 0.229 | 0.840 | 21.31 | 1.061 | -0.098 | 0.936 | -36.47 | -0.178 | 0.161 | 0.415 |
| LightGBM | **0.834** | **0.275** | 60,109 | 0.199 | 0.852 | 17.78 | 1.040 | -0.049 | 0.952 | -15.59 | -0.106 | 0.093 | 0.263 |

Interpretation: STL is a **credible prediction problem** (held-out R²_price 0.83). It is **not** a production-ready level-accurate AVM in this construction (median ratio 0.85; missing time feature and year-only dates). First-order vertical pattern remains in the unpenalized LGBM (β_log -0.106, PRB -0.049, Δ_NL 0.093). That is a baseline description, not a Direct/Surrogate result.

---

## 14. Frozen recommended jurisdiction panel

`modeling_readiness/final_panel_freeze.yaml` (2026-09-02T18:05:00Z)

| Unit | Decision | Role |
|---|---|---|
| Cook County CCAO | **INCLUDE_PRIMARY** | Paper’s canonical AVM application (unchanged) |
| Existing six-county ATTOM | **RETAIN_EXISTING_ATTOM_SENSITIVITY_ONLY** | Exploratory transfer / heterogeneity / failure modes; Miami-Dade stays a boundary case |
| St. Louis County CMF | **INCLUDE_BOUNDARY_CASE** | Sole CMF standalone AVM candidate, with timing and source-code limitations |
| Cook CMF 2002–2015 | **EXCLUDE** (from AVM panel) | Pre-2016 assessment-ratio provenance |
| Detroit | **EXCLUDE** (from AVM panel) | Best Berry reproduction; assessment-regressivity benchmark only |
| Philadelphia, Orleans, Franklin | **EXCLUDE** | Timing, scrape leakage, or incomplete source |
| NYC, Buffalo, Clark, LA, Maricopa CMF | **EXCLUDE** | Secondary inventory; Maricopa already in ATTOM |

No inclusion/exclusion used Direct/Surrogate performance.

---

## 15. Recommended role of each jurisdiction

- **Cook / CCAO:** keep as the sole confirmatory application in the paper.
- **Cook / CMF BOR:** cite as independent local-government evidence of assessment/sale ratios **before** the modeling window; do not numerically equate those ratios to model valuation ratios.
- **Detroit:** strongest external replication of documented assessment regressivity; do not train an AVM on the ratio-study workbook.
- **St. Louis County:** optional boundary-case AVM **if** sale day is recovered from yearly `sales` extracts, the 2012 snapshot gap is handled, and the APRTOT code defect is disclosed. Not confirmatory transfer evidence.
- **Philadelphia:** wait for dated OPA characteristic history.
- **Orleans / Franklin:** assessment-ratio documentation only, with provenance caveats.
- **ATTOM six:** keep the current exploratory benchmark language; do not replace it with CMF.
- **Miami-Dade ATTOM:** keep as the nonlinear/boundary case already used in the paper.

---

## 16. Remaining risks

1. Berry ratio ≠ model valuation ratio (estimand confusion in any paper revision).
2. St. Louis published CMF code is scientifically invalid as a sale-ratio study.
3. St. Louis `joined.csv` dropped `SALEDT`; year-only chronology is coarser than CCAO.
4. 2012 dwelling extract is tax year 2013; 2012 sales lack same-year cards.
5. Dwelling headers truncated; 37- vs 39-field mapping depends on the data dictionary, not filenames.
6. Cook CMF vs CCAO empty overlap — cannot validate source robustness of the *same* experiment.
7. Several CMF “Report” links are CoreLogic, not local studies.
8. Orleans scrape leakage if characteristics are used as predictors.
9. Default `cmfproperty` IQR filter is not the same as official arms-length (Detroit correctly used `filter=FALSE`).
10. Philadelphia rbind may duplicate rows.
11. No ATTOM hybrid selection-bias audit because no D-class jurisdiction.
12. Secondary jurisdictions unexplored; LA page code is internally inconsistent; AZ `lpv` is not market value.
13. Smoke LGBM level bias (median ratio 0.85) if time is omitted.
14. Stata unavailable; Detroit match used a Python translation (close enough for rounding, still a translation).

---

## 17. Whether a full method experiment was scientifically justified

**No.** Protocol Step 14: if unresolved major methodological issues remain, stop rather than force rho paths.

Justifications for stopping:

- Only one new AVM-ready CMF jurisdiction, so this is not a multi-jurisdiction transfer test.
- That jurisdiction has year-only dates and a missing 2012 snapshot.
- Cook cannot support three-source model robustness.
- No approved ATTOM hybrid.
- Running Direct/Surrogate on St. Louis alone, then choosing whether to “keep” it, would violate the freeze rule in spirit even if the freeze file existed.

The existing ATTOM six-county Direct/Surrogate paths already in the paper remain the exploratory transfer layer.

---

## 18. Direct/Surrogate transfer results

**Not run.** No rho grid, no penalty paths, no deployment rho.

---

## 19. Exact recommended changes to the paper (DO NOT EDIT IN THIS PASS)

These are recommendations only.

1. **Keep** the CCAO application as the confirmatory core. Do not replace it with CMF Cook 2002–2015 files.
2. **Keep** the six-county ATTOM section as exploratory. Do not overwrite those tables or drop Miami-Dade.
3. **Sharpen estimand language** wherever Berry/CMF is cited: official assessment/sale ratios are related to, but not the same object as, model valuation/sale ratios.
4. **Do not** add a confirmatory “six CMF counties” AVM transfer claim. The audit does not support it.
5. If external local-government provenance is added, **Detroit** is the only exact Berry reproduction; report COD/PRD/PRB as assessment-ratio diagnostics, not as method results.
6. If Cook CMF numbers are mentioned, state years **2002–2015**, N=641,147 after the CMF class+IQR path, and the **absence** of overlap with the CCAO modeling sample.
7. **St. Louis** may be added later only as a *boundary-case construction* after recovering `SALEDT`, documenting the 2012 snapshot defect, and refusing the published APRTOT-as-price path. Do not add it now as a clean replication county.
8. Do not treat CMF page “Report” CoreLogic HTML as local-government results (Philadelphia, Franklin, Cook, Maricopa, LA).
9. Do not train the proposed method to reproduce assessed values.
10. Any future method-transfer experiment should be a **new isolated directory** after a revised freeze, not mixed with this audit.

---

## Canonical jurisdiction table

`jurisdiction | geographic_unit | cmf_replication_available | reproduction_status | original_government_source | sale_target_quality | assessment_benchmark | parcel_id_quality | years | n_eligible | rich_local_features | temporal_integrity | public_enrichment | attom_enrichment | hybrid_match_rate | hybrid_selection_risk | baseline_avm_quality | recommended_role | final_include | confidence`

| jurisdiction | geographic_unit | cmf_replication_available | reproduction_status | original_government_source | sale_target_quality | assessment_benchmark | parcel_id_quality | years | n_eligible | rich_local_features | temporal_integrity | public_enrichment | attom_enrichment | hybrid_match_rate | hybrid_selection_risk | baseline_avm_quality | recommended_role | final_include | confidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| detroit_mi | Detroit, MI | yes | EXACT_OR_ROUNDING_MATCH | City of Detroit Office of the Assessor | high (Adj. Sale $, VALID ARMS LENGTH) | yes (Asd. when Sold) | present in workbook | 2016Q2–2018Q1 | 9653 (ratio study window) | thin (floor area, style, district) | TIMING_UNRESOLVED for chars | not acquired; not needed for ratio study | no Detroit ATTOM in repo | n/a | n/a | not run (failed gate) | external assessment-regressivity benchmark | EXCLUDE from AVM panel | high |
| philadelphia_pa | Philadelphia County, PA | yes (Stata + clean CSV) | PARTIAL_REPRODUCTION | OPA / OpenDataPhilly | good sale_price; possible rbind duplicates | assmt_at_sale | parcel id in DTA | 2012–2018 | 88811 (IQR path) | DTA has area/yrbuilt/baths | TIMING_UNRESOLVED | candidate, not acquired | not attempted | n/a | n/a | not run | public-enrichment candidate | EXCLUDE | medium |
| orleans_la | Orleans Parish, LA | yes (scrape + nola.R) | PARTIAL_REPRODUCTION | Assessor site scrape (bhelx); FOIA blocked | sales.price usable | yearly values; /0.1 scaling | property ids in scrape | 2015–2019 | 30939 (no ACS join) | land/bldg area; no year built/beds | POST_SALE_LEAKAGE if properties used | no official history | no Orleans ATTOM | n/a | n/a | not run | scrape provenance only | EXCLUDE | high |
| franklin_oh | Columbus city, OH (not full county) | partial (columbusonly.csv) | PARTIAL_REPRODUCTION | Franklin County Auditor (GDB missing) | PRICE + SALE_YEAR only | ASSESSED_VALUE; no stage | PID present | 2002–2018 | 186231 (IQR path) | none in Box file | n/a (no chars) | live GIS not historical | not attempted | n/a | n/a | not run | incomplete source | EXCLUDE | high |
| st_louis_county_mo | St. Louis County, MO | yes (large yearly extracts) | PARTIAL_REPRODUCTION (as-written path invalid) | St. Louis County Assessor | PRICE on SALEVAL==X; SALEDT dropped in joined.csv | APRTOT/ASMTOT diagnostic only | PARID high quality | 2009–2019 (no 2012 snapshot) | 70864 modeled; 73743 sold-only IQR ratio N | dwelling cards: SFLA, YRBLT, beds/baths, grade | SAFE if year-matched; 2012 folder is 2013 | not required | not required (class B) | n/a (no hybrid) | dwelling unmatched 7728/78936 | LGBM held-out R²_price 0.834; median ratio 0.85 | CMF AVM boundary case | INCLUDE_BOUNDARY_CASE | medium |
| cook_il_cmf | Cook County, IL (BOR res_sales) | yes | PARTIAL_REPRODUCTION | Cook BOR / CCAO extracts | NetConsideration | bor_CCAO_ass | pin/pin10 | 2002–2015 | 641147 (class+IQR) | area/class/township only | TIMING_UNRESOLVED for area | n/a | disjoint years vs ATTOM Cook | n/a | n/a | not run | pre-2016 assessment provenance | EXCLUDE from AVM | high |
| cook_il_ccao | Cook County, IL (canonical) | n/a (not a CMF extract) | n/a | CCAO research extract | high | not the AVM target | meta_pin | 2016–2025 | 444692 | rich char_* | SAFE_AS_OF_SALE (existing protocol) | n/a | separate ATTOM sensitivity | n/a | n/a | existing paper experiment | primary application | INCLUDE_PRIMARY | high |
| nyc_ny | New York City, NY | inventory only | not run | NYC DOF (unconfirmed in files) | unknown | unknown | unknown | unknown | not downloaded | unknown | unknown | n/a | n/a | n/a | n/a | not run | secondary inventory | EXCLUDE | high |
| erie_ny | Buffalo City, NY | inventory; sales xlsx downloaded | not run | City of Buffalo rolls | sales 2009–2019 | 2020 disclosure on 2018+ sales | SBL | 2018+ in page filter | not modeled | 2020 snapshot | POST_SALE_LEAKAGE risk | n/a | n/a | n/a | n/a | not run | secondary inventory | EXCLUDE | high |
| clark_nv | Las Vegas / Clark, NV | inventory | not run | Clark County Assessor (inferred) | unknown | unknown | unknown | 2011–2014 files listed | large files skipped | unknown | unknown | n/a | n/a | n/a | n/a | not run | secondary inventory | EXCLUDE | high |
| los_angeles_ca | Los Angeles County, CA | inventory | not run | LA County Assessor | code overwrites SALE_PRICE | inconsistent in page code | AIN | Year>2004 in snippet | not downloaded | assessor dta exists | unknown | n/a | n/a | n/a | n/a | not run | code-quality risk | EXCLUDE | high |
| maricopa_az | Maricopa County, AZ | inventory | not run | Maricopa Assessor (PHX_* files) | unknown | lpv (statutory LPV, not market) | parcelid | unknown | ~3.2 GB skipped | unknown | unknown | n/a | existing ATTOM Maricopa | n/a | n/a | ATTOM sensitivity already | ATTOM sensitivity only | EXCLUDE CMF; RETAIN ATTOM | high |
| attom_six | Cook, Allegheny, Maricopa, King, Miami-Dade, Middlesex | n/a | n/a | ATTOM/Dewey commercial | recorder TRANSFERAMOUNT | not official assessment | ATTOM ids | 2016–2025 pipeline | paper: 1,231,194 / 246,241 held-out | ATTOM assessor history | existing paper protocol | n/a | this *is* the ATTOM layer | n/a | n/a | existing exploratory paths | exploratory sensitivity | RETAIN_EXISTING_ATTOM_SENSITIVITY_ONLY | high |
