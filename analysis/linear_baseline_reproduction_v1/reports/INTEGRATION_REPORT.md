# Ordinary-linear baseline reproduction for v20 — integration report

**Stage** `analysis/linear_baseline_reproduction_v1/`
**Branch / commit at start** `paper-major-revision-write` @ `79ac963e`
**Manuscript of record** `paper/paper_v20.tex` (untracked working-tree file, sha256 in the
reproducibility record), rendered as `paper/paper_v20.pdf`.
**Nothing in `paper/` was edited in this pass.** No existing figure or frozen artifact was
overwritten. Everything below is a proposal.

---

## 0. Repository state and the "supplied v20 draft"

`paper/paper_v20.tex` and `paper/paper_v20.pdf` are both present and both **untracked** — v20 has
never been committed on this branch, so there is no git-side v20 to diff against. The working-tree
file is the only v20 the repository has.

**No separate v20 draft was attached to this task**, so the requested comparison between "the
repository's v20" and "the v20 draft supplied for this task" could not be performed. Everything in
this report is resolved against the working-tree file. Two statements in the task brief that appear
to come from that draft are contradicted by the repository and are corrected here:

| Brief says | Repository |
|---|---|
| "the graphic is absent from this source packet" (also a comment at `paper_v20.tex:656-658`) | `paper/img/generated_v12_994/baseline_models_motivation_2024_2025.pdf` **exists** (20,884 bytes) and `paper_v20.pdf` renders it: the embedded figure text is `βlog = -0.150`, `Sale price (log10 scale)`, `βlog = -0.164`. |
| "v15's rendered figure showed both models, whereas the v20 caption describes LightGBM alone" | True historically, already reconciled. `paper_v15.tex:1323` and `paper_v20.tex:658` reference the **same path**. Commit `c630994d` ("[Final] Close independent referee P1 findings") replaced the asset in place by vector surgery, removing the Linear column (`paper/paper_analysis/tier_b_validation/remove_linear_baseline_visuals.py`). The v20 caption is accurate for the current asset. |

That commit's message states the reason for the surgery: *"the executed run roots that produced
those series exist in no checkout."* **That premise is false for the main checkout.** The executed
linear run roots are present — see §1.

---

## 1. (a) Did the old linear predictions pass the audit, or was a rerun needed?

**They passed. No rerun was needed, and none was performed.**

Row-level ordinary-linear predictions exist for all nine canonical blocks under
`output/paper_v6_preselection_994/`, resolved through the frozen P0 config maps
(`analysis/p0_major_revision_validation/configs/frozen_{config,cv_run}_map.csv`), config
`fd63507d2456c789`, `LinearRegression`:

* held-out — `baseline_reporting/analysis/data_id=d4929d43ec19badf/split_id=3d464d4a611b131b/test_run_predictions/fd63507d2456c789.parquet` (38,290 rows)
* 2025 forward — `.../assess_run_predictions/fd63507d2456c789.parquet` (26,641 rows)
* folds 1–7 — `predictions/.../fold_id={0..6}/<run_id>.parquet`

These are *not* merely an old summary table or a two-model PDF: each carries `row_id`, `sale_date`,
`y_true_log`, `y_pred_log`, `y_true`, `y_pred`.

Compatibility gates (`provenance/lb2_compatibility.json`), all **PASS**:

| Gate | Result |
|---|---|
| C1 extract identity | `data/CCAO/2025/training_data.parquet`, 215,400,916 bytes, sha256 `b1fc00b5…3a7b51` — both match the v20 record exactly |
| C2 samples | 344,607 / 38,290 / 382,897 / 26,641 via the imported `run_temporal_cv._load_and_split_data`; 95 predictors, 23 categorical; filters `ind_pin_is_multicard == False`, `sv_is_outlier == False`; date ranges 2016-01-01–2023-11-09, 2023-11-09–2024-12-31, 2025-01-01–2025-12-29 |
| C3 folds | seven expanding-window folds rebuilt; all archived `train_index_hash` / `val_index_hash` verified |
| C4 sale identity | for all 18 model×block artifacts: `row_id` equals the canonical positional index, `y_true_log` is **bitwise** equal to `np.log(meta_sale_price)` on the canonical split, `sale_date` is exactly equal, and `y_pred == exp(y_pred_log)` exactly (direct exponentiation confirmed, not inferred) |
| C5 predictors | see §1.1 |

Metric audit (`tables/lb_cached_prediction_audit.csv`): re-executing the frozen metric code on the
cached row-level predictions reproduces every value the frozen runs recorded, across **324**
comparisons (144 linear, 180 LightGBM, the latter audited against both the baseline-report CSVs
and `zero_control_full.csv` cell A), to a maximum relative deviation of **6.96 × 10⁻¹⁴**.
**253 of the 324 comparisons (78.1%) are exactly equal.** Of the 71 that are not, the closed-form
measures deviate by at most **7.6 × 10⁻¹⁵** relative — last-ULP noise, consistent with the frozen
values being re-read from their stored decimal text rather than from float64 bytes. The larger
residual (up to 7 × 10⁻¹⁴) is confined to `dCor(e,y)`, `Delta_NL` and `Delta_NL_raw`, the only
iteratively estimated measures in the suite (the `dcor` library's O(n log n) path and five
cross-fitted spline fits). PRD and COD are exactly equal everywhere. Nothing here is large enough
to move a displayed digit.

### 1.1 Predictor-information audit — the one substantive asymmetry

The linear design matrix realised on the development pool is **344,607 × 148**
(`provenance/lb2_compatibility.json`, `tables/lb_predictor_representation_audit.csv`).

* **Excluded from the linear model: 2 of the 95 predictors** — `loc_census_tract_geoid` and
  `loc_tax_municipality_name`. `preprocessing.recipes_pipelined.InitialColumnDropper` drops any
  column that starts with `loc_`, does not start with `loc_school_`, and is non-numeric in the
  extract. Both are strings. LightGBM consumes both natively.
* **Added to the linear model, not available to LightGBM in that form**: target encodings of five
  high-cardinality fields (`meta_nbhd_code`, `meta_township_code`, `char_class`, both
  `loc_school_*_district_geoid`), one-hot encoding of the remaining 16 categoricals, median
  imputation on 72 numeric columns and most-frequent imputation on 21 nominal ones, 1%/99%
  winsorising of `char_land_sf` and `char_bldg_sf`, Box-Cox transforms on 8 columns, `+0.001`
  offset copies of three distance/rate variables, squared terms in `char_yrblt`, `char_bldg_sf`
  and `char_land_sf`, and standardisation of 68 columns. The near-zero-variance step ran and
  removed nothing.
* Preprocessing is **refit per fitting block** (`run_temporal_cv.py:1359-1364`), so each fold's
  encodings, imputers, winsor limits, Box-Cox parameters and scaler come from that fold's
  training rows only. No held-out or 2025 outcome enters any linear fitting decision.

**The two representations are not nested in either direction.** Any accuracy statement should say
so; it is not a clean functional-form contrast.

---

## 2. (b) Frozen-LightGBM parity

**PASS — bitwise identical on all nine blocks** (`tables/lb_frozen_lightgbm_parity.csv`,
`provenance/lb1_verdict.json`).

The ordinary-LightGBM series that sits alongside the linear predictions
(config `252a25d9c0ce796b`) is the *same object* as the frozen zero-penalty benchmark,
`output/p0_major_revision_validation/zero_reference_fits/cell=A`, which is what
`analysis/p0_major_revision_validation/tables/zero_control_full.csv` cell A reports:

* `y_pred_log` max absolute difference **0.0** on every block;
* recomputed `sha256(float64 bytes)` of the prediction vector equals the frozen
  `eval_pred_sha256` on every block (held-out `b4aad84e…6bd5c`, 2025 `97e9d856…a042e3`);
* `y_eval_log_sha256` and the archived index hashes agree on every block;
* `lgbm_params_sha256` = `8f0f2acd…60585b` throughout, the expected frozen vector.

This was established on unrounded float64 values and content hashes — **not** by agreement after
rounding \(R_P^2\) to 0.894. Because the linear series comes from the same artifact directory and
shares the same `row_id`, `sale_date` and bitwise-equal log-price vectors, it inherits that sample
identity exactly. No LightGBM control was regenerated: refitting an exact match would only add
execution-path noise.

*Identity limitation.* These artifacts carry `row_id` and `sale_date`, not `meta_pin`, so sale
identity is positional against the canonical split, corroborated by the bitwise log-price and
sale-date vectors. A parcel-identifier join is not possible from the retained artifacts.

---

## 3. (c) Full-precision comparison, both periods

Machine-readable, full precision: `tables/lb_table1_main_baseline.csv`,
`tables/lb_table2_complementary_baseline.csv`, and the all-blocks superset
`tables/lb_metrics_full_precision_all_blocks.csv`.

### Held-out evaluation (n = 38,290; both fit on the 344,607-sale development pool)

| Measure | Ordinary linear | Ordinary LightGBM |
|---|---:|---:|
| $R^2_P$ | 0.798980441733524 | 0.894228690294242 |
| $\mathrm{MAE}_P$ | 90092.34575272 | 75655.0882843753 |
| $\mathrm{MAPE}_P$ | 0.241232023700879 | 0.212008815010044 |
| $\mathrm{RMSE}_{\log P}$ | 0.32174841531268 | 0.288709037061319 |
| PRD | 1.04175335958494 | 1.06949738016592 |
| PRB | -0.0163231314225898 | -0.0907588766108404 |
| MKI | 0.974783625265852 | 0.922952737240953 |
| VEI (%) | -11.5696831061445 | -26.4596342178749 |
| Median ratio | 0.968704367321864 | 0.928770750289118 |
| Mean ratio | 1.02031809094669 | 0.988646904389737 |
| Weighted-mean ratio | 0.979423854561131 | 0.924403297029454 |
| COD (%) | 24.7343533752551 | 21.6265818314926 |
| COV (%) | 45.1608358064802 | 39.7275759726444 |
| $\beta_{\log}$ | -0.091782004441005 | -0.149599333833048 |
| $\Delta_{\mathrm{NL}}$ | 0.131214913727407 | 0.118661762377438 |
| $\mathrm{dCor}(e,y)$ | 0.250134520719612 | 0.387087194041816 |

### 2025 forward evaluation (n = 26,641; both refit on all 382,897 eligible 2016–2024 sales)

| Measure | Ordinary linear | Ordinary LightGBM |
|---|---:|---:|
| $R^2_P$ | 0.7989136660075 | 0.904377393694208 |
| $\mathrm{MAE}_P$ | 99371.4105375113 | 78484.1593225354 |
| $\mathrm{MAPE}_P$ | 0.24898198599462 | 0.207793487824236 |
| $\mathrm{RMSE}_{\log P}$ | 0.313147361456118 | 0.278194976306063 |
| PRD | 1.0522387447188 | 1.07924761088269 |
| PRB | -0.0289365575031892 | -0.106044033424992 |
| MKI | 0.954145010277287 | 0.906539668006541 |
| VEI (%) | -17.2999101659209 | -28.5773464436935 |
| Median ratio | 1.0204784109757 | 0.950316022231903 |
| Mean ratio | 1.08071793756613 | 1.01547176107898 |
| Weighted-mean ratio | 1.02706533378501 | 0.940907119774355 |
| COD (%) | 24.3342072523613 | 21.2663706641781 |
| COV (%) | 42.0506234871339 | 37.1876226271675 |
| $\beta_{\log}$ | -0.108596005089028 | -0.164083693002414 |
| $\Delta_{\mathrm{NL}}$ | 0.122394256022544 | 0.121174861465389 |
| $\mathrm{dCor}(e,y)$ | 0.268996146770448 | 0.422038047033507 |

*Reading.* On both evaluations LightGBM is ahead on all four predictive measures and on COD/COV,
while linear has PRD, PRB, MKI and VEI closer to neutral and a smaller \(|\beta_{\log}|\). The
pattern does **not** generalise to the residual-structure diagnostics: linear's
\(\Delta_{\mathrm{NL}}\) is *larger* (more non-affine conditional-mean structure) on both periods.
Nothing here licenses a statement that linear regression is fairer in general, and nothing here is
evidence that boosting causes regressivity.

*PRB/VEI value proxy* (`tables/lb_prb_vei_value_proxy.csv`, recomputed per model and sample under
the existing convention \(V^{\mathrm{proxy}}_i = 0.5P_i + 0.5\widehat P_i/m_S\)): the proxy is
**model-specific**, because it embeds that model's own valuations and its own sample median ratio.
The two columns are therefore read against different value axes. That is a property of the adopted
PRB definition and the exposure-draft VEI grouping, not a defect of this reproduction, but it
should be stated wherever the two PRB or VEI columns are set side by side.

*Guidance status is preserved throughout*: PRD and PRB carry adopted IAAO 2013 guidance; MKI and
VEI carry May-2026 exposure-draft (proposed, not adopted) guidance; \(\beta_{\log}\),
\(\Delta_{\mathrm{NL}}\) and dCor carry no reference range. No compliance is inferred from any
point estimate.

### Seven chronological folds (descriptive)

`tables/lb_metrics_full_precision_all_blocks.csv` (rows with `kind = cv_validation`) and
`tables/lb_cv_fold_summary.csv`.

| Fold | n | Linear \(R^2_P\) | LGBM \(R^2_P\) | Linear PRB | LGBM PRB | Linear VEI | LGBM VEI |
|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | 5,209 | 0.8447 | 0.9115 | −0.0766 | −0.0849 | −38.03 | −32.36 |
| 2 | 11,197 | 0.8323 | 0.8991 | −0.0542 | −0.0776 | −24.86 | −28.32 |
| 3 | 16,798 | 0.8166 | 0.9104 | −0.0176 | −0.0784 | −11.76 | −23.96 |
| 4 | 22,323 | 0.8229 | 0.9024 | +0.0173 | −0.0709 | +3.56 | −15.02 |
| 5 | 28,053 | 0.7958 | 0.8866 | +0.0486 | −0.0821 | +12.89 | −22.53 |
| 6 | 33,113 | 0.7884 | 0.8947 | +0.0078 | −0.1058 | −5.06 | −32.38 |
| 7 | 34,460 | 0.7569 | 0.8935 | −0.0044 | −0.1076 | −11.78 | −36.05 |

Equal-weight summary: linear \(R^2_P\) mean 0.8082 (SD 0.0299), PRB mean −0.0113 (SD 0.0428), VEI
mean −10.72 (SD 17.03); LightGBM \(R^2_P\) mean 0.8997 (SD 0.0091), PRB mean −0.0867 (SD 0.0143),
VEI mean −27.23 (SD 7.23).

**These seven folds are expanding-window and overlapping. They are not independent replications**
and the spread above is descriptive, not a sampling distribution. Two facts from them matter for
any text that gets written: on **fold 1 the linear VEI is worse than LightGBM's** (−38.0% vs
−32.4%), so the equity ordering is not uniform; and the linear fold-to-fold spread on VEI is more
than twice LightGBM's.

### Retransformation sensitivity

`tables/lb_smearing_{factors,sensitivity,invariance_audit,ordering_verdict}.csv`,
`provenance/lb4_smearing.json`. The frozen development-only Duan factor (estimator and D3
row-balanced weights imported from `analysis/p1_inferential_reporting`, 151,153 appearances over
130,165 unique development rows) is \(s = 1.081596\) for linear and \(s = 1.062323\) for LightGBM —
the latter reproducing the frozen P1 value to 2.2 × 10⁻¹⁶.

Applying it unchanged to both evaluations: **the qualitative accuracy ordering is unchanged in all
six comparisons** (\(R^2_P\), \(\mathrm{MAE}_P\), \(\mathrm{MAPE}\) × two periods) — LightGBM
remains ahead, and the gap widens. Invariance of COD, COV, PRD, PRB, MKI, VEI, \(\beta_{\log}\),
\(\Delta_{\mathrm{NL}}\) and dCor holds to 8.1 × 10⁻¹²; the three level ratios scale by \(s\) to
3.3 × 10⁻¹⁶. This belongs in the reproduction record, not in the main figures.

---

## 4. (d) Differences from v15

**None.** All **64** displayed baseline cells in `paper_v15.tex` — both models × both evaluations ×
16 measures, from `tab:ccao_baseline_results` (l.1236-1260) and the complementary
`oldrevisionblock` table (l.1276-1310) — are reproduced exactly at v15's own display precision
(`tables/lb_v15_comparison.csv`). That includes every quantity the brief singled out: linear
\(R^2_P\) 0.799 / 0.799, PRB −0.016 / −0.029, VEI −11.6% / −17.3%, COD 24.7% / 24.3%,
\(\beta_{\log}\) −0.092 / −0.109.

One provenance discrepancy is worth recording even though no number moved. The v15 comment at
`paper_v15.tex:1224` says the table was "regenerated from `output/paper_v6_preselection`
baseline-report predictions". The **linear** column is the same in both roots (identical config
`fd63507d2456c789`, byte-identical metrics), but the **LightGBM** column printed in v15 is the
`_994` run (\(R^2_P\) 0.894, PRB −0.091, \(\beta_{\log}\) −0.150), not the non-994 run
(\(R^2_P\) 0.882, PRB −0.103, \(\beta_{\log}\) −0.164). So the v15 provenance comment is stale for
half the table. The v20 comment at `paper_v20.tex:610-613` already flags exactly this and re-sources
the table to the frozen control; that correction stands and should be carried into any new version.

The retired asset `paper/img/generated_v6_preselection/baseline_models_motivation_2024_2025.pdf`
carries \(\beta_{\log}\) annotations −0.092 / −0.150 / −0.109 / −0.164, which match this
reproduction exactly. The root-level `paper/img/baseline_models_motivation_2024_2025.pdf` carries
−0.092 / −0.163 / −0.109 / −0.174 — its linear panels match, its LightGBM panels come from a
different (non-994) fit and should not be reused.

---

## 5. (e) Files created or changed

**Nothing under `paper/`, `utils/`, `scripts/`, `preprocessing/`, `data/`, the P0/P1/B0 analysis
subtrees, or the frozen `output/paper_v6_*` / `output/paper_v12_*` roots was written.**
`lb_common._assert_write_allowed` refuses those paths by assertion.

Created:

```
analysis/linear_baseline_reproduction_v1/
  code/  lb_common.py  lb1_audit_and_metrics.py  lb2_compatibility_and_predictors.py
         lb3_figure.py  lb4_smearing_sensitivity.py  lb5_paper_assets.py
         lb6_reproducibility_record.py
  tables/ (csv + parquet twin)
         lb_frozen_lightgbm_parity.csv          lb_metrics_full_precision_all_blocks.csv
         lb_cached_prediction_audit.csv         lb_cv_fold_summary.csv
         lb_prb_vei_value_proxy.csv             lb_sale_identity_check.csv
         lb_predictor_representation_audit.csv  lb_table1_main_baseline.csv
         lb_table2_complementary_baseline.csv   lb_v15_comparison.csv
         lb_figure_ratio_profile_bins.csv       lb_smearing_factors.csv
         lb_smearing_sensitivity.csv            lb_smearing_invariance_audit.csv
         lb_smearing_ordering_verdict.csv       lb_restricted_input_manifest.csv
  figures/ baseline_models_motivation_linear_vs_lgbm_2024_2025_v20r1.{pdf,png}
  snippets/ tab_ccao_baseline_results_panelled.tex  tab_ccao_baseline_results_wide.tex
            tab_ccao_baseline_complementary.tex     fig_baseline_motivation.tex
  provenance/ lb1_verdict.json  lb2_compatibility.json  lb3_figure.json
              lb4_smearing.json lb5_assets.json  lb_reproducibility_record.json
  reports/ INTEGRATION_REPORT.md   (this file)
output/linear_baseline_reproduction_v1/   (created, empty — nothing bulky was produced)
data/CCAO                                  -> symlink to the main checkout (read-only)
output/p0_major_revision_validation        -> symlink to the main checkout (read-only)
output/paper_v6_preselection               -> symlink to the main checkout (read-only)
output/paper_v6_preselection_994           -> symlink to the main checkout (read-only)
output/paper_v12_lower_rho_extension_994_v2 -> symlink to the main checkout (read-only)
```

The five symlinks exist because `output/` and `data/CCAO/` are git-ignored and live only in the
main checkout, while `p0_common` resolves them relative to the repository root. They let the
canonical loader and the frozen config maps be **imported verbatim** rather than re-implemented.
Remove them with:

```
rm analysis/linear_baseline_reproduction_v1 -r   # if the stage is not being kept
rm data/CCAO output/p0_major_revision_validation output/paper_v6_preselection \
   output/paper_v6_preselection_994 output/paper_v12_lower_rho_extension_994_v2
```

### Two repository-hygiene consequences you need to decide on

1. **Tier-B write scope.** `paper/paper_analysis/tier_b_validation/tb_scope.py`
   `uncommitted_write_scope()` counts **untracked** paths and requires every one to be under
   `paper/`. `analysis/linear_baseline_reproduction_v1/` and `data/CCAO` are untracked and outside
   `paper/`, so running `validate.py` on this branch right now will fail that check until they are
   removed. `cumulative_write_scope()` is worse: committing this directory on
   `paper-major-revision-write` would fail the validator **permanently** for that branch. This
   stage should be committed on a separate branch, or kept uncommitted, or moved under `paper/` —
   your call, not mine.
2. **`.gitignore`.** Inside this stage `*.csv`, `*.json` and `*.parquet` are caught by the global
   ignores. Tracking the evidence would need a narrow exception block mirroring the existing P0 and
   P1 blocks (`.gitignore:131-155`), keeping `*.parquet` ignored.

---

### 5.1 End-to-end determinism

The whole chain (`lb1` → `lb6`) was re-run from scratch after the first pass. All 15 result CSVs,
their parquet twins, the figure PNG and all four LaTeX snippets came back **byte-identical**. The
only file that changed is the figure PDF, because matplotlib stamps `/CreationDate` into it; the
drawn content is unchanged (the PNG hash is stable, and the plotted profile is itself archived in
`tables/lb_figure_ratio_profile_bins.csv`). Verify the figure by the PNG hash or that CSV, not by
the PDF hash.

---

## 6. (f) Unresolved limitations

* No v20 draft was supplied, so the requested repository-vs-draft diff is outstanding.
* Sale identity is positional plus bitwise log-price and sale-date agreement; `meta_pin` is not
  carried in the retained prediction artifacts, so a parcel-level join cannot be done.
* The seven folds overlap; their spread is descriptive only.
* The Duan factor is estimated on development out-of-fold residuals and applied unchanged to the
  2025 block, which has no out-of-fold analogue — the assumption the frozen P1 design already
  records.
* Linear and LightGBM predictor representations are not nested (§1.1).
* The linear specification is a research benchmark on this extract and design. **It is not the
  office's historical production model**, and nothing here supports a statement about how the
  township-level workflow actually performed.
* No PRB or VEI confidence intervals were computed for the linear arm. The v20 main text quotes a
  95% CI for the LightGBM 2025 PRB (`paper_v20.tex:606`), which comes from the frozen P1 inference
  stage; a symmetric linear CI would require a new P1-style run and is out of scope here.

---

# 7. Integration proposals — text that becomes inaccurate once a linear result is added

Every item below is **proposed only**. Nothing was applied. Line numbers are from the current
untracked `paper/paper_v20.tex`.

### 7.1 Statements that become false

**L646 — `tab:ccao_baseline_results` notes. Hard contradiction.**

> Current: `The benchmark is defined in Section~\ref{subsec:comparators}. No linear-regression model is estimated or reported in this study. The guidance status...`

> Proposed: `The benchmark is defined in Section~\ref{subsec:comparators}. The linear column is an ordinary least-squares fit of log sale price on the same samples under the same temporal design; it is a research benchmark, not a reconstruction of the office's historical township-level production models. The guidance status...`

**L1169 — comparators, "The workflow benchmark" paragraph. Hard contradiction.**

> Current: `... The earlier township-level linear-regression workflow is historical context only: it is not re-estimated, and no quantitative linear-versus-boosted comparison is reported.`

> Proposed: `... The office's earlier township-level linear-regression workflow is historical context only and is not re-estimated. The linear benchmark reported in Section~\ref{subsec:baseline_tension} is a research specification fitted on this study's samples, not that workflow; the two should not be conflated.`

**L2393 — `tab:ccao_baseline_complementary` notes. Hard contradiction.**

> Current: `Every entry reports the same unpenalized workflow benchmark, which carries all eight complementary measures on both evaluations. The earlier township-level linear-regression workflow is discussed only as historical context and is neither re-estimated nor reported quantitatively in this study.`

> Proposed: `Both unpenalized specifications carry all eight complementary measures on both evaluations. The office's earlier township-level linear-regression workflow is discussed only as historical context and is not re-estimated; the linear column here is a research benchmark fitted on this study's samples.`

**L224 — CCAO background. Needs narrowing, not deletion.**

> Current: `Our analysis compares ordinary LightGBM with corrections to the same model class. We do not re-estimate the earlier linear-regression workflow or reconstruct official historical assessments.`

> Proposed: `Our analysis compares ordinary LightGBM with corrections to the same model class, against an ordinary linear-regression benchmark fitted on the same samples. We do not re-estimate the office's earlier linear-regression workflow or reconstruct official historical assessments.`

**L664 — prose after the figure. The second clause is directly refuted.**

> Current: `... The baseline results describe the observed sample, not standards compliance or how a linear model would perform under this study's design.`

> Proposed: `... The baseline results describe the observed sample and do not establish standards compliance. They also do not describe how the office's historical township-level models performed: the linear benchmark is a research specification estimated here, on this extract and this design.`

**L133 — introduction. Becomes misleading rather than false.**

> Current: `... The earlier linear-regression workflow provides historical context; this paper evaluates corrections within the LightGBM model class.`

> Proposed: `... The office's earlier linear-regression workflow provides historical context. An ordinary linear benchmark fitted on the same samples is reported alongside the LightGBM baseline in Section~\ref{subsec:baseline_tension}; the corrections this paper develops are evaluated within the LightGBM model class.`

### 7.2 Captions and one-model framing

**L621 — main table caption.** Replace with the caption in
`snippets/tab_ccao_baseline_results_panelled.tex`.

**L2364(caption, physical line 2365 `\caption{...}`) — complementary table caption.** The clause
`A level statement about one model class, not a comparison between model classes.` is no longer
true. Replace with the caption in `snippets/tab_ccao_baseline_complementary.tex`.

**L659 — figure caption.** `... for ordinary LightGBM in the held-out and 2025 forward
evaluations` → the caption in `snippets/fig_baseline_motivation.tex`.

**L608 — prose introducing the table.**

> Current: `Table~\ref{tab:ccao_baseline_results} reports the unpenalized LightGBM benchmark on the held-out sample and after refitting...`

> Proposed: `Table~\ref{tab:ccao_baseline_results} reports both unpenalized baselines on the held-out sample and after refitting...`

**L647 — the fitting-sample note says "the model" (singular).**

> Proposed: `For the held-out evaluation both models are fit on the 344,607-sale development pool; for the 2025 forward evaluation both are refit on all 382,897 eligible 2016--2024 sales.`

**L2476-2477 — cross-reference wording.**

> Current: `... the ordinary-LightGBM values are reported in Appendix Table~\ref{tab:ccao_baseline_complementary} instead.`

> Proposed: `... the two unpenalized baselines' values are reported in Appendix Table~\ref{tab:ccao_baseline_complementary} instead.` (Label unchanged.)

### 7.3 Path and cross-reference updates

* **L658 figure path.** `img/generated_v12_994/baseline_models_motivation_2024_2025.pdf` →
  `img/generated_v20_linear_baseline/baseline_models_motivation_linear_vs_lgbm_2024_2025_v20r1.pdf`.
  The new asset must be **copied** from
  `analysis/linear_baseline_reproduction_v1/figures/`; do **not** overwrite the v12_994 asset —
  that is the LightGBM-only figure produced by the P1-1 vector surgery in `c630994d`, and it is
  still the correct asset for any build that keeps the one-model framing. Width also wants to go
  from `0.45\textwidth` to `0.8\textwidth` (four panels, as in v15).
* **L656-658 stale comment.** The `% Before submission, inspect the actual PDF at this path... The
  graphic is absent from this source packet.` comment should be deleted: the graphic is present and
  the caption already matches it.
* **L610-613 provenance comment.** Extend the Tier-B2.2 note to record the second source:
  `% The linear column is sourced from output/paper_v6_preselection_994/baseline_reporting/...`
  `% test/assess_run_predictions/fd63507d2456c789.parquet, verified bitwise against the same`
  `% sample identity as cell A in analysis/linear_baseline_reproduction_v1/.`
* **Labels `tab:ccao_baseline_results`, `tab:ccao_baseline_complementary`,
  `fig:baseline_motivation` are preserved in all four snippets**, so every existing `\ref` keeps
  resolving. No other cross-reference needs to change.
* **Appendix~\ref{app:implementation}** (`L2119`, "Linear and Boosting Implementations") currently
  derives the penalized linear estimator but documents no linear *pipeline*. If the linear column
  enters the paper, the two excluded predictors and the engineered terms of §1.1 need a short
  subsection there — the main-table note proposed in
  `snippets/tab_ccao_baseline_results_panelled.tex` forward-references it.

### 7.4 Validator work the integration will require

* **Numeric ledger.** Each newly printed linear number needs an entry in
  `paper/paper_analysis/tier_b_validation/ledger/tier_b_numeric_ledger.yaml` with the seven
  required fields; `tb_ledger.verify_entry` reopens the artifact, verifies its sha256 against
  `FINAL_EVIDENCE_MANIFEST.json`, executes the selector and re-derives the rendered value. The
  frozen `manuscript_numeric_map.csv` indexes only Tier-A baseline numbers and will not resolve
  them. Note that the current manifest does not index this stage's tables, so they would need
  registering first — that is a Tier-B0-side change, not a paper-side one.
* **C02** (no flagged unsupported numeric tokens) will fail at `subsec:baseline_tension` and at the
  appendix anchor until those ledger entries exist.
* **C07** is not at risk: `spec/tier_b_required_statements.yaml` pins four
  covariance-is-not-fairness denials, none of which is among the sentences above.
* **C03** guards reference-cell semantics (cell A may not carry `PENALTY_ISOLATING`). The linear
  benchmark is neither cell A nor cell C; it is a *different model class*, outside the A/B/C
  zero-reference taxonomy. It needs its own `reference_cell` value (e.g. `LINEAR_BENCHMARK`) with
  `comparison_purpose: MODEL_CLASS_CONTRAST`, so that adding it cannot be mistaken for a
  penalty-isolating comparison. **The custom-objective \(\rho=0\) origin (cell C) remains distinct
  from both ordinary LightGBM and the ordinary-linear benchmark; nothing in this stage touches it.**

### 7.5 What must not be written

For the record, and consistent with the brief: this stage's evidence does **not** support saying
that linear regression is universally fairer (fold 1 reverses the VEI ordering;
\(\Delta_{\mathrm{NL}}\) favours LightGBM on both periods), that boosting *causes* regressivity,
that any penalty strength is preferred, or that a penalized model dominates linear regression. No
penalty-path plot was touched and no linear marker was added to one.

---

# 8. Gap inventory — the v15 linear surfaces this pass did NOT cover

Raised after the first hand-back, and correct: v15 used the linear model far more widely than the
baseline pair. This pass covered **1 of 7 figure assets** and **2 of 6 table locations**, because the
brief scoped the rest out ("Do not edit or add linear markers to the penalty-path plots in this
run"; "do not expand the main figures for it"). The full inventory, with what is already in hand:

## 8.1 Tables — all six locations need ZERO new computation

The `Linear regression` row in v15's path tables is **\(\rho\)-independent**: it prints the same
constants as the v15 baseline table. Every cell is already in
`tables/lb_table1_main_baseline.csv` and `tables/lb_table2_complementary_baseline.csv` at full
precision, and all of them are among the 64 cells already validated in `tables/lb_v15_comparison.csv`.

| v15 location | Columns in the Linear row | Covered by |
|---|---|---|
| `paper_v15.tex:2531` / `:2553` — primary path summary, Panels A/B | \(R^2_P\), MAE, PRD, PRB, MKI, VEI, \(\beta_{\log}\), \(\Delta_{\mathrm{NL}}\), dCor | lb_table1 + lb_table2 |
| `:2700` / `:2724` — path anchor variant | adds MAPE, \(\mathrm{RMSE}_{\log P}\) | lb_table1 |
| `:3905` / `:3933` — complementary path anchor | median / mean / weighted-mean ratio, COD, COV, \(\beta_{\log}\), \(\Delta_{\mathrm{NL}}\), dCor | lb_table2 |

v20's counterparts are `tab:path_anchor_frozen` (l.1214), `tab:path_anchor_standards` (l.1278) and
`tab:path_anchor_complementary` (l.2414).

**One editorial decision is not mine to make.** v15's path tables keyed their marking convention to
linear — *"an asterisk identifies those that also strictly improve upon Linear Regression"*
(`:2681`). v20 replaced that with bold = improves on the workflow benchmark, dagger = improves on
the custom-objective origin (l.1265). Restoring a Linear row therefore means either restoring a
third comparison class across the whole path table — which re-introduces "improves on linear" as a
claim class on every penalised row — or printing the Linear row **unmarked** as a context row. The
second is the smaller claim and the one consistent with §7.5. Either way it is a claim-scope
decision, not formatting.

## 8.2 Figures — six assets outstanding

Enumerated by `paper/paper_analysis/tier_b_validation/remove_linear_baseline_visuals.py`, which
records exactly what it removed from each:

| Asset | Linear content removed by `c630994d` | Status |
|---|---|---|
| `baseline_models_motivation_2024_2025.pdf` | whole Linear column, \(\beta_{\log}\) −0.092 / −0.109 | **done** — `figures/…_v20r1.pdf` |
| `vei_percentile_group_profile.pdf` | whole "Held-out Linear" / "2025 Linear" column | **not done** — needs the linear VEI decile profile |
| `accuracy_equity_trajectories_inprocessing_only.pdf` | 8 gray Linear diamonds + Linear legend entry | **not done** |
| `tradeoff_equity_vs_accuracy_heldout.pdf` | 16 Linear diamonds | **not done** |
| `tradeoff_equity_vs_accuracy_2025.pdf` | 16 | **not done** |
| `tradeoff_mechanism_vs_accuracy_heldout.pdf` | 12 | **not done** |
| `tradeoff_mechanism_vs_accuracy_2025.pdf` | 12 | **not done** |

In v20 these are `fig:vei_group_profile_placeholder` (l.3060) and the four tradeoff atlases
(l.2597, 2602, 2607, 2612); the trajectories figure is at l.1344.

**For the five marker figures the linear coordinates are already in hand.** Each Linear diamond is a
single \((\text{accuracy}, \text{equity})\) or \((\text{accuracy}, \text{mechanism})\) point per
panel — \((R^2_P, \mathrm{VEI})\), \((R^2_P, \mathrm{PRB})\), \((R^2_P, \beta_{\log})\) and so on —
all of which are columns of `lb_table1` / `lb_table2`. What is missing is only the regeneration run,
which also needs the Direct and Surrogate series.

**`vei_percentile_group_profile.pdf` is the one genuinely uncomputed quantity.** The linear decile
profile with deterministic 90% bootstrap intervals was not produced here. It is cheap:
`utils.motivation_utils.vei_percentile_group_profile` runs directly on the verified row-level
predictions this stage already resolves, with no refit.

## 8.3 The removal script's second false premise

`remove_linear_baseline_visuals.py` justifies the surgery with: *"the executed run roots
(`output/paper_v12_lower_rho_extension_994_v2/`, `output/paper_v6_preselection_994/`) and the
combined path table exist in no checkout, and no artifact in the frozen evidence set reproduces
them."*

**All three of those objects exist in the main checkout**, verified in this pass:

* `output/paper_v6_preselection_994/analysis/combined_path_table.csv` (+ `.parquet`, +
  `paper_outputs/tables/combined_path_table.csv`)
* `output/paper_v12_lower_rho_extension_994_v2/analysis/.../transition_regions_v2_lower_rho/tables/combined_path_table_v2.csv`
  and `…_v4_analysis_view.csv`
* both run roots in full

This is the **same worktree-blindness** as the premise addressed in §0: `output/` is git-ignored and
so is empty in a paper-only worktree. The practical consequence is that the six outstanding figures
are **regenerable from evidence** — the Direct/Surrogate path series and the linear baseline points
are both available — rather than recoverable only by reverting vector surgery on a presentation
asset. If the linear comparison is reinstated, regenerating them from the path tables is the
defensible route, and `git show c630994d^:<path>` gives the pre-surgery assets as a visual check.
