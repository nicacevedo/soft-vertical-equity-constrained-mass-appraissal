# P0 Major-Revision Validation — Execution Plan (rev. 3, FINAL)

## Context

`paper/paper_v17_option1.tex` is the canonical manuscript. An external scientific audit
(`ADDITIONAL CONTEXT/FINAL_SCIENTIFIC_AUDIT.md`) plus a follow-up adjudication
(`CHATGPT_ASSESSMENT_OF_CLAUDE_AUDIT.md`) concluded that the paper's theory is sound but that
**four empirical/implementation blockers must be resolved before any Results/Discussion rewrite**:

1. the executed Direct/Surrogate objective code has never been numerically audited against the
   manuscript equations (is the Direct supplied Hessian effectively inert?);
2. native LightGBM and the custom-objective ρ=0 controls are **not** prediction-identical
   (held-out mean |Δŷ| = 3.24e−2, max 3.09e−1), which contaminates every "vs Ordinary LightGBM"
   claim in `tab:path_anchor_summary`;
3. the ρ=0 control table reports only 3 of ~17 metrics, so a reader cannot repair the comparison;
4. the paper proves (Cor. `cor:path_scaling`) that Direct **is** a centered-spread rescaling in the
   fixed-prediction-space benchmark, then declines to run the centered-spread post-hoc comparator.

Plus a bounded temporal-integrity audit (same-date split boundaries; repeat-parcel robustness).

This plan is the minimal, rigorous, reproducible execution sequence to close those blockers.
**No manuscript prose is rewritten during execution.** All new work is additive under
`analysis/p0_major_revision_validation/`.

### Decisions already taken (do not re-litigate during execution)

1. **ρ=0 reference.** Subject to the tiered parity criterion in §F.4, **Cell B — "Parity-aligned native
   L2"** may serve as the **within-implementation causal reference** for P0 contrasts *if justified by the
   evidence*. It is **never silently relabelled "Ordinary LightGBM."** The three cells keep fixed names
   in every table, figure and report:
   **Cell A = "Ordinary LightGBM (standard raw-label native)"**,
   **Cell B = "Parity-aligned native L2"**,
   **Cell C = "Custom ρ=0 origin"**.
2. **Full path rerun.** The 82-point Direct/Surrogate path is regenerated **only when a regeneration
   trigger in §F.6 fires** — see Gates G2, G5a and G5b. It is not run unconditionally, and
   **A↔B or A↔C non-parity can never trigger it on its own.**
3. **P1 extras included:** PRB standard errors/t-values, IAAO ED2 VEI Significance, and one
   smearing-corrected retransformation sensitivity with **development-only** smearing estimation.
   **Subgroup/township stratification is deferred** and will be reported as unmeasured.

### Amendments incorporated (rev. 2 and rev. 3)

| # | Amendment |
|---|---|
| A1 | **Historical-setting parity and pinned-determinism reproducibility are separate experiments.** A pinned B/C equivalence result may never be cited as validating the historical positive-ρ path artifacts. |
| A2 | The **theorem-matched comparator `f_b = ȳ_T + b(f₀ − ȳ_T)` is primary** (so `b=1` reproduces `f₀` exactly). The `f̄_{0,T}`-recentered form is retained only as a labelled sensitivity. |
| A3 | **No interpolation of held-out/2025 metrics between fitted ρ.** Matching uses actual fitted configurations within a frozen development β_log tolerance, with targeted new fits only where the tolerance cannot be met. |
| A4 | **Repeat-PIN robustness preserves the evaluation sets** and purges from *training* every PIN appearing in the corresponding evaluation block. The unseen-parcel evaluation-subset design is secondary. |
| A5 | **D-SNAP is strict-date robustness**, not a replacement design, unless it materially changes conclusions. |
| A6 | **Smearing factors are estimated on development data only**; evaluation-set calibration is prohibited. |
| A7 | The hard `1e-9` parity-or-rerun rule is replaced by a **tiered exact / numerical / material / non-parity criterion (T1–T4)**. |
| A8 | **Cell B is "Parity-aligned native L2"**; **Cell A remains "Ordinary LightGBM (standard raw-label native)"**; **Cell C is "Custom ρ=0 origin"**. |
| A9 | The claim that equal-weight CV means are dominated by larger later folds is **withdrawn and corrected**. |
| A10 | The Surrogate fixed-λ diagnostic is kept; `min_sum_hessian_in_leaf` is **not** implied to matter unless demonstrated. |
| **A11** | **Regeneration logic fixed (§F.6).** A↔B / A↔C non-parity **never** triggers full positive-ρ regeneration by itself. Regeneration fires only on a material B↔C failure, a defect in the custom objective actually used for the frozen paths, an **R4** historical-reproduction failure, or an explicit decision to make the pinned implementation canonical. |
| **A12** | **Source-equivalence / provenance check (§F.2b) precedes any interpretation of reproduction failure.** Failures are classified as source drift / unreconstructable dirty-state uncertainty / environment nondeterminism / numerical nondeterminism / genuine implementation discrepancy — never collapsed into "nondeterminism". A temporary read-only Git worktree at the recorded provenance commit is used where needed. |
| **A13** | **Historical artifact reproduction is tiered R1–R4 (§F.3a-2)**, not bitwise-or-fail. Only **R4** contributes to the regeneration gate. R1–R4 are kept strictly separate from the A/B/C parity tiers T1–T4. |
| **A14** | **D-SNAP promotion requires confirmation (§J.6).** The 21-point run is a *screening* grid: **G5a** triggers targeted local refinement on the missing original-grid ρ values around the affected region; **G5b** promotes and mandates full regeneration only if the trigger survives refinement. A very large qualitative reversal may go straight to promotion. |
| **A15** | Three precision corrections: CORE matched-β targets are defined **algorithmically** from actual fitted Direct configurations (§I.3-2); **D-PURGE is an oracle overlap-removal robustness diagnostic**, not a proposed deployment split (§J.3); the fixed-λ leaf-shrinkage calculation is a **stylized implementation diagnostic** unless actual trained-leaf Hessian sums are measured (§E.3-7). |

---

## A. Repository state and canonical experiment map

### A.1 Git / working tree

| Item | Value |
|---|---|
| Branch | `testing` |
| HEAD | `b878b00886584d8d81402a77fd19b93599204dd9` — "[Update] Small update of context files" (2026-09-05) |
| Clean? | Working tree clean except **one untracked file**: `ADDITIONAL CONTEXT/CLAUDE_CODE_HANDOFF.md` |
| Submodules | none (`.gitmodules` absent) |
| `CLAUDE.md` | none anywhere in repo |
| Baseline commit named in handoff | `b878b008…` — **matches HEAD** ✅ |
| Provenance commits | `d3ef45f2`, `508dc1c2`, `2aa0346a`, `725832f8` are **all ancestors of HEAD** (verified via `git merge-base --is-ancestor`) |

Note: the git status in the session preamble (deleted PDFs, untracked `paper_v15/v17*`) is **stale**;
those were committed in `a1b4f369` / `b878b008`.

### A.2 Environment

| Item | Value |
|---|---|
| Canonical conda env | `fairness_env` → `/home/nacevedo/.conda/envs/fairness_env` |
| Python | 3.9.19 |
| LightGBM | **4.6.0** (frozen provenance records the same) |
| numpy / pandas / sklearn / scipy / dcor | 1.26.4 / 2.3.1 / 1.6.1 / 1.13.1 / 0.6 |
| Repo pin | `requirements.txt` only says `lightgbm>=4.0` — **must be pinned to 4.6.0 for the P0 pass** |

### A.3 Canonical data (all present locally — no HPC-only paths)

| Path | Size | Role |
|---|---|---|
| `data/CCAO/2025/training_data.parquet` | 215 MB, 444,692 rows × 203 cols | **THE canonical CCAO extract** (`DATA_PATH`, `scripts/final_local_results_994.py:47`; `cv_config.yaml:13`) |
| `data/CCAO/2025/training_data_2016_append_manifest.json` | — | 2016 append provenance (396,480 → 444,692 rows) |
| `data/CCAO/2025/{assessment,char,complex_id,hie,land_nbhd_rate}_data.parquet` | 420 MB / 843 MB / … | supporting extracts |

After `~ind_pin_is_multicard & ~sv_is_outlier`: **409,538 rows**.
`data/berry_cmf`, `data/dewey-downloads`, `data/ATTOM` are external-jurisdiction work, out of P0 scope.

### A.4 Canonical experiment provenance chain (994-tree paper results)

| Stage | Exact artifact |
|---|---|
| **Data** | `data/CCAO/2025/training_data.parquet`; filters `~ind_pin_is_multicard & ~sv_is_outlier`; target `meta_sale_price`, date `meta_sale_date` |
| **Split construction** | `utils/motivation_utils.py::split_ccao_assessment_universe` (L190-231) and `::build_rolling_origin_protocol` (L234-365, Mode A); driven from `run_temporal_cv.py::_load_and_split_data` (L750-773); frozen in `output/paper_v6_preselection_994/protocol/**/folds.json` |
| **Preprocessing/features** | predictor columns from `params.yaml`; categoricals cast to pandas `category` and auto-detected (`utils/motivation_utils.py:712-715`, `run_temporal_cv.py:1248-1251`) |
| **Baseline fit (994-tree freeze)** | `scripts/setup_paper_v6_preselection_994.py` → `output/paper_v6_preselection_994/lgbm_config.json` (`config_id 407d47775760c14d`, `lgbm_params_sha256 8f0f2acd…585b`); search winner from `output/robust_rolling_origin_cv_v2/baseline_lgbm_search/**/baseline_lgbm_best_params.json` |
| **Direct objective** | `soft_constrained_models/boosting_models.py::canonical_direct_scaled_grad_hess` (L143-186), class `LGBCovPenalty` (fit L502-527, predict L529-533, fobj L535-554) |
| **Surrogate objective** | same file `::canonical_surrogate_scaled_grad_hess` (L198-232), class `LGBSmoothPenalty` (fit L1102-1140, predict L1142-1148, fobj L1150-1176) |
| **Native baseline** | `run_temporal_cv.py::_native_lgbm_estimator` (L876-889) |
| **ρ grid — original 50** | `cv_config.yaml` `rho_values:[1e-1,100.0]`, `rho_count:50`, `rho_scale:"geom"` → `[0.0] + geomspace(0.1,100,50)` |
| **ρ grid — lower-tail 32** | `scripts/setup_paper_v12_lower_rho_v2.py::lower_rho_extension` (L60-95), same geometric ratio, 0.0010985 → 0.0868511; frozen in `V12/protocol/lower_rho_grid_v2.json` (hash `072e4ea94c…`) |
| **Rolling-origin CV** | `scripts/run_paper_v6_preselection_994_cv.sbatch` (104 cfg × 7 folds = 728 fits) + `scripts/run_paper_v12_lower_rho_cv.sbatch` (64 cfg × 7 folds = 448 fits) |
| **Held-out + 2025 forward** | `scripts/run_paper_v6_preselection_994_oos_array.sbatch`, `scripts/run_paper_v12_lower_rho_oos_array.sbatch` (`--stage test` / `--stage forward`) |
| **Postprocessing/metrics** | `utils/motivation_utils.py::compute_taxation_metrics` (L1585) + `::paper_mechanism_metrics` (L1544); `utils/delta_nl.py::estimate_delta_nl` (L126-192) |
| **Path assembly** | `scripts/final_local_results_994.py`, `utils/transition_regions.py`, `utils/transition_paper_assets.py`, `scripts/paper_v12_followup_assets_v4.py` |
| **FROZEN canonical path table** | `V12/analysis/data_id=d4929d43ec19badf/split_id=3d464d4a611b131b/penalty_path_analysis/transition_regions_paper_assets_v4_delta_nl_bends/tables/combined_path_table_v4_analysis_view.csv` — **168 rows × 197 cols** (Direct 83 + Surrogate 83 + LightGBM 1 + Linear 1) |
| **Promoted paper figures** | `paper/img/generated_v12_994/` — 20 referenced by the `.tex`, all present; sources span three asset generations (v3_followup Aug 29, v4_delta_nl_bends Aug 30, candidate_regions_v2_1 Aug 31) |
| **Promoted paper tables** | hard-coded literal numbers in the `.tex` (no `\input` of generated files); population provenance in `output/paper_v12_population_994/provenance/{PRE,POST}_POPULATION.json` |

### A.5 Frozen 994-tree LightGBM parameter vector (`lgbm_params_sha256 = 8f0f2acd…585b`)

```
boosting_type gbdt | objective mse | n_estimators 994 | learning_rate 0.04430617038989053
num_leaves 573 | max_depth 11 | min_child_samples 93 | max_bin 424
colsample_bytree 0.5410105713520937 | reg_alpha 0.06172892626785128 | reg_lambda 10.857207234911057
min_split_gain 0.0010638716225202418 | cat_l2 0.6440381513216918 | cat_smooth 148.99648221517137
max_cat_threshold 81 | min_data_per_group 86 | random_state 2025 | n_jobs 1 | verbosity -1
```

**Never set anywhere:** `min_sum_hessian_in_leaf` / `min_child_weight` (LightGBM default 1e-3),
`bagging_fraction` / `subsample` / `subsample_freq` (bagging OFF), `deterministic`,
`force_row_wise` / `force_col_wise`, `num_threads`. Early stopping is **OFF**
(`early_stopping_rounds: null`); all 994 trees are always grown, so `best_iteration` is not a
parity suspect. Threading is constrained only by `n_jobs=1` plus `OMP_NUM_THREADS=1` in the sbatch.

### A.6 Split design (as executed)

- Dev/CV pool = oldest 90 % of eligible 2016–2024 sales = **344,607** (through 2023-11-09).
- Held-out = newest 10 % = **38,290** (2023-11-09 → 2024-12-31).
- 2025 forward = **26,641**, evaluated after refit on all **382,897** 2016–2024 sales.
- 7 expanding-window folds, 15-month origin steps from 2016-01-01; in each fold the newest 10 % of
  the cumulative window is validation (fold 7: 310,147 train / 34,460 val).
- **Both the fold train/val cut and the dev/held-out cut are positional, not date-based** — so rows
  sharing a boundary date fall on both sides at **8 boundaries** (7 CV internal + 1 dev/held-out).
  The 2025 boundary is year-based and clean. Measured exposure in §J.1.

---

## B. Existing artifacts that can be reused

Root abbreviations: `V6 = output/paper_v6_preselection_994`,
`V12 = output/paper_v12_lower_rho_extension_994_v2`,
`PPA = V12/analysis/data_id=d4929d43ec19badf/split_id=3d464d4a611b131b/penalty_path_analysis`.

| Artifact | Location | What it contains | Reuse? | Why | Risk |
|---|---|---|---|---|---|
| **Frozen 82-point path table** | `PPA/transition_regions_paper_assets_v4_delta_nl_bends/tables/combined_path_table_v4_analysis_view.csv` | 168 rows × 197 cols: Direct 83 + Surrogate 83 + LightGBM + Linear; 17 metrics × (fold_1..7, CV_mean, CV_sd, heldout, forward_2025) | **YES — primary input** | Complete canonical suite already computed for every ρ on all 3 regimes | Produced under **historical (unpinned) settings**; its validity is a question for the historical-reproduction check (§F.3a), *not* for the pinned ladder |
| **ρ=0 control rows inside that table** | same file, `family∈{Direct,Surrogate} & rho==0`, plus `family=='LightGBM'` | **The full metric suite at the ρ=0 origin already exists** on held-out, 2025 and CV_mean | **YES — P0-3 is an extraction, not a computation** | Verified by direct read (values in §G) | Δ_NL CV_mean is `NaN` for ρ=0 / LightGBM / Linear — needs 4 configs × 7 folds of cheap recompute |
| **ρ=0 parity audit (CCAO)** | `V6/final_local_results/rho0_split_audit.{csv,json}`, `FINAL_LOCAL_RESULTS_STATUS.json` | mean/median/p95/max \|Δlog\|, Pearson vs native, per split; diagnosis text; `gate: PASS_REPORT_CODE_PATH_DIFFERENCE` | **YES as evidence** | Exactly the numbers in `tab:rho_zero_control` | Diagnosis is asserted, not demonstrated — no decomposition of the cause |
| **ρ=0 parity audit (external, TINY config)** | `analysis/external_jurisdiction_benchmark_v1/audits/zero_rho_parity.csv` + `objective_scaling_audit.{md,json}` | native-vs-Direct mean \|Δ\| ≈ **1.3–1.8e-8**, max ≈ **1.0–1.3e-7**; Direct≡Surrogate exactly (0.0); Cook included | **YES as a diagnostic clue only** | Shows the code paths agree to ~1e-7 at low capacity | **Run with `TINY_PARAMS` (60 trees, 15 leaves, depth 4, n=4,000, numeric-only features)** — it does **not** establish parity for the 994-tree/573-leaf/310K-row CCAO config, and it is not a historical-setting test of the CCAO artifacts |
| **Existing centered-recalibration path** | `V6/final_local_results/recalibration_path.{csv,parquet}` + `recalibration_spec.json` | 51 b-values × {heldout, forward_2025} = 102 rows, full metric suite incl. Δ_NL, β_log, dCor | **YES — extend, do not rebuild** | It already implements the **theorem-matched `ȳ_T`-centered map** that A2 makes primary, so `b=1` reproduces `f₀` | Two gaps only: the grid stops **at** `b*` (no overshoot) and there are **no CV-fold rows** |
| **Recalibration code** | `scripts/final_local_results_994.py` L442-676 (`load_native_oof`, `reconstruct_fold_training_means`, `solve_b_star_validation_neutral`, `centered_map` L586-587, `run_recalibration`) | Working, hash-verified against the archived fold protocol | **YES — wrap/extend** | `reconstruct_fold_training_means` re-verifies every fold index hash and asserts split counts 344,607/38,290/26,641 | Hard-codes `RESULT_ROOT = V6`; needs parameterisation |
| **Cached CV fold predictions** | `V6/predictions/**/fold_id={0..6}/<run_id>.parquet` (104 cfg × 7 = 728) and `V12/predictions/**` (64 cfg × 7 = 448) | `row_id, sale_date, y_true_log, y_pred_log, y_true, y_pred` | **YES** | b-path on all 7 folds, pooled-OOF smearing factors, and the D-UNSEEN robustness variant all need **zero refits** | run_id→config mapping lives in `runs/**.parquet`; must be resolved, not guessed |
| **Cached held-out + 2025 predictions for every ρ** | `V6/reporting_preview/{heldout,forward_2025}/{LGBCovPenalty,LGBSmoothPenalty}/chunk_*/**/{test,assess}_run_predictions/` and `V12/…` (83 per family per split) | same schema plus `config_id, model_name` | **YES** | Any *new* metric (PRB SE, VEI significance, smearing, ratio profile, Surrogate weight shares) is computable with **zero refits** | 8 + 4 chunk dirs per family/split — needs a resolver |
| **Native + Linear held-out/2025 predictions** | `V6/baseline_reporting/analysis/**/{test,assess}_run_predictions/{252a25d9c0ce796b,fd63507d2456c789}.parquet` | Cell-A native `f₀` on both out-of-time blocks | **YES — the comparator base** | The `f₀` the manuscript's corollary refers to | Base cell may change under Gate G2 |
| **Δ_NL estimator** | `utils/delta_nl.py` + specs `V6/final_local_results/delta_nl_estimator.json` and `PPA/…v4…/delta_nl_cv/estimator_spec.json` (hash `e85069150b…`) | Frozen 5-fold cross-fitted spline (cubic, 8 quantile knots), id-hash fold rule, salt `paper_v6_delta_nl_v1` | **YES — freeze unchanged** | Prediction-independent fold assignment ⇒ new configs directly comparable | — |
| **CV Δ_NL by fold** | `PPA/…v4_delta_nl_bends/delta_nl_cv/delta_nl_cv_{by_fold,mean}.csv` | 164 positive-ρ configs × 7 folds | **YES** | Already the CV mechanism coordinate | **Excludes ρ=0, LightGBM, Linear** — 4 configs × 7 folds to add |
| **Metric suite** | `utils/motivation_utils.py::compute_taxation_metrics` / `::paper_mechanism_metrics`; `cod` L1146, `cov_iaao` L1060, `vei` L1182, `vei_percentile_group_profile` L1267, `prd` L1340, `prb` L1390, `mki` L1500, `distance_correlation_e_y` L1572 | Single canonical implementation used by every paper-pipeline stage | **YES — import, never re-implement** | Guarantees P0 outputs sit on the same footing as the frozen table | Legacy duplicates exist (`spatial_analysis_wrapup.py`, `final_market_value_1_target_correction.py`, `quick_test_models.py`) with **different** MdAPE/dCor definitions — must not be used |
| **Split builders** | `utils/motivation_utils.py::split_ccao_assessment_universe`, `::build_rolling_origin_protocol`; archived `V6/protocol/**/folds.json` | The exact executed chronological design + index hashes | **YES — wrap, never edit** | Robustness variants build on these | Editing them in place would invalidate every frozen hash |
| **Frozen 994-tree config** | `V6/lgbm_config.json` | Parameter vector + versions + git provenance | **YES — verbatim** | Also satisfies manuscript `\todo` L1966 | Recorded from a **dirty** tree at `d3ef45f2…`; only `git_diff_sha256` was stored, not the diff text |
| **Provenance helpers** | `canonical_experiment.py::{git_state, lgbm_params_hash, package_versions}` | Standard provenance stamping | **YES** | House convention | — |
| **Objective unit tests** | `tests/test_canonical_objectives.py`, `tests/test_paper_v6_guards.py`, `tests/test_final_local_results_994.py`, `tests/test_delta_nl.py`, `tests/test_canonical_metrics.py` | Finite-difference checks of both gradients/Hessians; the exact `I + (ρ/2n)ccᵀ` identity; ρ=0 equality of families | **YES — ready-made harness** | — | `test_native_custom_rho0_parity_after_mean_init` asserts only `mean|Δ| < 5e-3` on synthetic data — far too loose to have caught the CCAO gap |
| **Analysis-folder + Slurm conventions** | `analysis/external_jurisdiction_benchmark_v1/` | protocol yaml, `scripts/`, `slurm/`, `audits/`, `tables/`, `figures/`, `reports/`, `tests/`, `README.md` | **YES — copy the pattern** | Consistency with the newest frozen analysis | — |

**Artifacts that must NOT be reused:** `output/paper_baseline_2024_2025/table2_metrics_unrounded.csv`
(superseded: 382,900 rows, 2025 R² 0.891 vs the paper's 0.904); `output/paper_v6_preselection/` (pre-994);
`analysis/berry_attom_validation_v2/v3` (quarantined); every `quick_test*`, `attom_*`, `county_bench_*`
directory; and all legacy call sites that omit the canonical kwargs
(`ratio_mode="diff"`, `match_native_init=True`, and for the Surrogate `weighting_proxy_mode="identity"`) —
notably `quick_test_models.py:1136-1163`, `analysis/berry_attom_validation_v{2,3}/scripts/run_direct_surrogate.py`,
`scripts/other_counties_benchmars.py:1598`, `spatial_analysis.py:899`.

---

## C. Disagreements between repository and context documents

| # | Context-document claim | Repository evidence | Resolution |
|---|---|---|---|
| **C-1** | Audit §G.2 / §M and manuscript `\todo` L3679: "a later initialization-aligned parity experiment … achieved near-numerical native/custom agreement on the tested path" | `analysis/external_jurisdiction_benchmark_v1/scripts/objective_scaling_audit.py:50-53` — that run used `TINY_PARAMS = dict(n_estimators=60, num_leaves=15, max_depth=4, learning_rate=0.1, min_child_samples=20, n_jobs=2)` on **n=4,000 deterministic subsamples with numeric-only features**. `grep -r parity` finds **no CCAO-scale parity artifact anywhere**. | **The existing parity evidence does not transfer.** It shows the two code paths agree to ~1e-7 at low capacity; it says nothing about the 994-tree / 573-leaf / 310,147-row configuration. The manuscript `\todo` L3679 overstates what exists. |
| **C-2** | Audit P0-2 / H-1: MAE, MAPE, PRD, PRB, MKI, VEI, COD at the custom-objective origin are "not reported anywhere" | Verified: `combined_path_table_v4_analysis_view.csv` **already contains the complete 17-metric suite** for Direct ρ=0, Surrogate ρ=0 and native on held-out, 2025 and CV_mean (Δ_NL CV_mean excepted). | The gap is **manuscript reporting**, not computation. P0-3 costs ~0 compute; its only dependency is the parity gate. |
| **C-3** | Audit §G.1: use `b_max = 1/R²_T`, extended ~20 % | (i) The assessment doc correctly rejects `1/R²` as the general endpoint. (ii) The existing artifact solves a *development-neutral* `b* = 1.16727` from pooled OOF predictions (`recalibration_spec.json`), the value that actually zeroes development β_log; an **in-sample** `b*` for a near-interpolating 994-tree ensemble will sit close to 1. | Compute **both** `b*_train = Var_T(y)/Cov_T(f₀,y)` (spec form) and `b*_oof` (existing). Path spans `[1, 1.25·max(b*_train, b*_oof)]`. Both development-only, neither leaks. |
| **C-4** | Audit §G.1: match at `β_log ∈ {−0.15, −0.12, −0.09, −0.06, −0.03, 0}` | Frozen table: **development (CV-mean) β_log ranges are Direct [−0.1394, −0.0787], Surrogate [−0.1393, −0.0186], native −0.1382.** Direct never reaches −0.06, −0.03 or 0 at any ρ ≤ 100 (held-out max −0.0794 at ρ=100). | **Four of six proposed targets are unattainable by Direct**, and −0.15 is outside both families' CV-mean range. The matched grid must be derived from measured common support (§I). |
| **C-5** | Audit §E: Direct's supplied Hessian `1 + (ρ/2n)c_i²` is "numerically inert"; the assessment demotes this to a hypothesis | Code confirms the algebra exactly: `boosting_models.py:185` supplies `hess = 1 + (ρ/(2n))c_i²`; `canonical_direct_exact_scaled_hessian` (L189-195) confirms the exact scaled Hessian is `I + (ρ/(2n))ccᵀ`, eigenvalue along `c` equal to `1 + (ρ/2)V̂ar(y)`. A synthetic check at `V̂ar(y) ≈ 0.4955`, `n = 310,147` gives supplied median `h` 1.0000000038 (ρ=0.0105) → 1.0000363 (ρ=100), max 1.0018, vs exact directional curvature 1.0026 → **25.78**. | **Hypothesis very likely correct**, but those numbers are synthetic. P0-1 computes them on the real `c` of every training fold at every displayed ρ. Decision rule in §E.4. |
| **C-6** | Neither document mentions it | **New finding.** `reg_lambda = 10.857` is an **absolute** quantity entering `leaf = −G/(H+λ)`. The Surrogate's Hessian is `1 + ρc_i²`, which grows with ρ; the Direct's stays ≈1. So Surrogate ρ both reweights the loss **and** shrinks the *effective* L2 leaf penalty relative to `H`. Direct ρ does neither. | A genuine, previously undocumented **implementation asymmetry at matched ρ**, and a candidate mechanism for the high-ρ Surrogate ratio trough (audit H-7). P0-1 **measures** it (§E.3-7). |
| **C-6b** | — | `min_sum_hessian_in_leaf` defaults to 1e-3 and is never set. With `h_i ≥ 1` and `min_child_samples = 93`, any admissible leaf has `ΣH ≥ 93 ≫ 1e-3`. | **Reported as checked-and-non-binding**, not as a mechanism. Per A10, the plan makes no claim that it matters unless the diagnostic demonstrates a binding case. |
| **C-7** | Audit §I, §J: document-integrity and bibliography work | Confirmed: 19 rendering `\todo`s in `paper_v17_option1.tex`; `\usepackage[textsize=tiny]{todonotes}` at L20 with no `disable`; `\oldtext` renders blue struck-through (L59-61). The `.tex` names the **superseded** 50-ρ `combined_path_table.csv` at L3758 while its figures come from the 82-ρ v4 table; `V6/paper_outputs/paper_results_manifest.json` still points at the abandoned `img/generated_v6_preselection/`. | Out of P0 scope, but the **two provenance mis-references** are recorded in the P0 manifest so the eventual fix is mechanical. |
| **C-8** | (agent claim) the artifact-generating commit `d3ef45f2…` is unreachable from HEAD | `git merge-base --is-ancestor` returns **YES** for all four provenance commits. | The commits **are** reachable. What is genuinely missing is the archived working-tree **diff text**. P0 archives its own. |
| **C-9** | Audit §A.4-5 / P1-1: the seven expanding-window folds are "summarized by equal-weight means/SDs" | Confirmed nested (`train_mode: expanding`, `step_months: 15`, `val_fraction: 0.1`): fold 7 trains on 310,147 of the 344,607 pool; folds 1–6 validation blocks sit inside fold 7's training block. | **Correction (A9):** because fold means are **equal-weighted**, larger later folds do **not** dominate the CV mean — that earlier statement is withdrawn. The real issue is **nesting and non-exchangeability**: the seven folds share most of their training data, so CV_mean/CV_sd describe variation across nested chronological windows, not independent replications, and 7/7 LOFO agreement is close to mechanical. A secondary consequence of equal weighting is the opposite of size domination: small early folds (n_val 5,209) carry the same weight as fold 7 (n_val 34,460). Both facts are reported; neither invalidates the coordinate, and per-fold paths are shown alongside the mean. |
| **C-10** | Audit §G.3: same-date crossing is "disclosed"; repeat-sale effect "likely second-order" | Measured by re-running the split code read-only (**every archived fold index hash reproduced exactly**). Same-date crossing at all 8 boundaries; per-boundary train/val counts on the boundary date range from 46/46 to 158/2, dev/held-out 43/105. Repeat-parcel exposure: share of validation PINs also in that fold's training block rises **9.5 % → 26.8 %**; dev↔held-out **10,972 / 36,994 (29.7 %)**; (dev+test)↔2025 **8,805 / 26,048 (33.8 %)**. | Same-date fix is confirmed trivial (≤166 rows/boundary). Repeat-parcel exposure is **large enough that the robustness variant is genuinely informative**. Neither replaces the primary design by default (§J.3, §J.5, A4, A5). |
| **C-11** | Neither document mentions it | **Reproducibility hazard.** `run_temporal_cv._load_and_split_data` (L750-773) reads with pyarrow row-group **pushdown** → 409,538 rows. Reading without pushdown and applying the equivalent pandas mask gives 409,541 → development becomes 344,610 and **every fold index hash changes**. | All P0 code loads via `run_temporal_cv._load_and_split_data`, never re-implementing the filter, and keeps the 344,607 / 38,290 / 26,641 assertions from `scripts/final_local_results_994.py:543-544`. |
| **C-12** | Audit §G.1: "reuse the same `f₀`" | **No in-sample training predictions are cached anywhere** (`utils/motivation_utils.py:867-887` writes validation rows only; the OOS path writes test rows only). | `f̄_{0,T}` and `Cov_T(f₀,y)` require **9 native refits with in-sample prediction** (7 fold-training blocks + the 344,607 dev block + the 382,897 production block). Shared with P0-2, so not additional. |
| **C-13** | Audit §L1/§L3 treat the objective audit and the parity diagnosis as one task | The frozen artifacts were produced **without** `deterministic` / `force_row_wise` / `num_threads` pins. | **(A1)** Two distinct questions must be separated: (a) *does the historical configuration reproduce, and what is the historical A↔C gap?*; (b) *are the code paths algebraically equivalent under pinned determinism?* Answer (b) may never be used to certify the historical positive-ρ artifacts produced under (a)'s settings. §F implements the separation. |

---

## D. P0 dependency graph

```
                  ┌───────────────────────────────────────────────┐
                  │ S1  Scaffold + provenance freeze (no compute)  │
                  └───────────────┬───────────────────────────────┘
                                  │
        ┌─────────────────────────┼──────────────────────────────┐
        │                         │                              │
┌───────▼────────┐   ┌────────────▼─────────────┐     ┌──────────▼───────────┐
│ S2  P0-1       │   │ S4-0 SOURCE-EQUIVALENCE  │     │ S3  P0-6a same-date  │
│ objective      │   │      / provenance audit  │     │ + repeat-parcel      │
│ arithmetic     │   │      (F.2b, NO FITS)     │     │ EXPOSURE AUDIT       │
│ NO FITS        │   │        ↓                 │     │ NO FITS              │
└───────┬────────┘   │ S4a HISTORICAL-setting   │     └──────────┬───────────┘
        │            │     reproduction (R1-R4) │                │
        │            │     + A/B/C ladder       │                │
        │            │ S4b PINNED-determinism   │                │
        │            │     A/B/C ladder (T1-T4) │                │
        │            │ (SEPARATE experiments)   │                │
        │            └────────────┬─────────────┘                │
        │                         │                              │
        │            ┌────────────▼─────────────┐                │
        │            │ GATE G1  reproduction    │                │
        │            │ tier + failure class     │                │
        │            └────────────┬─────────────┘                │
        │                         │                              │
        │            ┌────────────▼──────────────────────────┐   │
        │            │ S5  9 native refits (canonical cfg,   │   │
        │            │     historical settings) with         │   │
        │            │     IN-SAMPLE prediction retained     │   │
        │            │  ⇒ also yields f̄₀,T and Cov_T(f₀,y)   │   │
        │            └────────────┬──────────────────────────┘   │
        │                         │                              │
        │            ┌────────────▼─────────────────────┐        │
        │            │ GATE G2  T-tiers + R-tiers +     │        │
        │            │ F.6 RG-1..RG-4 trigger table     │        │
        │            │ (A-B / A-C can NEVER trigger)    │        │
        │            └───┬──────────────────────────┬───┘        │
        │                │                          │            │
        │      ┌─────────▼──────┐   ┌───────────────▼────────┐   │
        │      │ S6  P0-3 zero  │   │ (only if an RG fired)  │   │
        │      │ control table  │   │ S12 full path regen    │   │
        │      └────────┬───────┘   └────────────────────────┘   │
        │               │                                        │
        │      ┌────────▼──────────────────┐                     │
        └─────►│ S7  P0-4 centered-spread  │                     │
               │ comparator (post-proc)    │                     │
               └────────┬──────────────────┘                     │
                        │                                        │
               ┌────────▼──────────────────┐                     │
               │ GATE G3  common support   │                     │
               └────────┬──────────────────┘                     │
                        │                                        │
        ┌───────────────▼───────────┐              ┌─────────────▼──────────┐
        │ S8  P0-5 matched-β:       │              │ S10 D-SNAP SCREENING   │
        │  algorithmic CORE targets │              │     (21-pt, strict-    │
        │  nearest FITTED configs   │              │      date robustness)  │
        │  within frozen tolerance  │              │ S11 D-PURGE (oracle) / │
        │  (+ targeted fits if req) │              │     D-UNSEEN (0 fits)  │
        └───────────────┬───────────┘              └─────────────┬──────────┘
                        │                                        │
                        │                              ┌─────────▼─────────┐
                        │                              │ GATE G5a  alert?  │
                        │                              │  → S10b targeted  │
                        │                              │    local refine   │
                        │                              └─────────┬─────────┘
                        │                                        │
                        │                              ┌─────────▼─────────┐
                        │                              │ GATE G5b  survives│
                        │                              │ refinement? then  │
                        │                              │ promote → S12'    │
                        │                              └─────────┬─────────┘
                        └──────────────┬─────────────────────────┘
                                       │
                              ┌────────▼─────────┐
                              │ S13 FREEZE + G4  │
                              └────────┬─────────┘
                                       │
                              ┌────────▼──────────────────┐
                              │ S14 MANUSCRIPT_IMPACT_MEMO│
                              │ (no manuscript edits)     │
                              └───────────────────────────┘
```

**Parallelism.** `S2`, `S3` and `S4-0` start together; `S4a` follows `S4-0` (which decides whether the
reproduction check runs at HEAD or in a provenance worktree), and `S4b` follows `S4a`. `S10`/`S11` depend
only on `S3` and the frozen screening grid, so they run in parallel with the comparator chain. `S6` and
`S7` both depend on `S5`. `S8` depends on `S7` plus the frozen path table. `S10b` runs only on a G5a
alert; `S12`/`S12'` run only on a fired §F.6 RG trigger or a G5b promotion respectively.

**Correction to the intended order in the task brief.** Steps 5 (matched β) and 6 (temporal) are
listed sequentially but are independent — run them in parallel. Step 3 (full ρ=0 control metrics)
correctly precedes step 4, but note it costs essentially **zero compute** (§C-2); its only real
dependency is Gate G2.

---

## E. P0-1 — implementation audit

**Deliverables:** `reports/P0_IMPLEMENTATION_AUDIT.md`, `tables/objective_scaling_audit.csv`,
`tables/direct_hessian_magnitudes.csv`, `tables/surrogate_weight_distribution.csv`,
`tables/effective_leaf_shrinkage.csv`.

### E.1 Files to audit (read-only)

| Concern | Exact location |
|---|---|
| Direct grad/hess | `soft_constrained_models/boosting_models.py:143-186` |
| Direct exact dense Hessian | `boosting_models.py:189-195` |
| Surrogate grad/hess | `boosting_models.py:198-232` |
| Direct fit/predict/fobj | `boosting_models.py:502-554` |
| Surrogate fit/predict/fobj | `boosting_models.py:1102-1176` |
| Canonical-kwargs gate | `boosting_models.py:133-140`, `:122-130` |
| Native baseline | `run_temporal_cv.py:876-889` |
| Spec construction | `run_temporal_cv.py:925-933` / `:940-967` / `:1035-1061` |
| Frozen params | `V6/lgbm_config.json` |
| Existing tests | `tests/test_canonical_objectives.py`, `tests/test_paper_v6_guards.py` |

### E.2 Facts established by inspection (record, then re-verify numerically)

- Target `y = log(meta_sale_price)`, computed per fold (`utils/motivation_utils.py:707-710`, `run_temporal_cv.py:1248-1251`).
- Centering: `base_score_ = mean(y_fit_block)`; labels passed to LightGBM are `y − base_score_`;
  `y_mean_ = mean(centered) ≈ 0`; `boost_from_average=False`; `init_score = zeros(n)`;
  `base_score_` added back in `predict`. So `c_i = y_i − ȳ_T` exactly, per fold.
- The `n/2` scaling is applied **analytically**: the squared-error block yields `grad = e`, `hess = 1`
  (not `2e`, `2`). Pinned by `tests/test_canonical_objectives.py:17-24`.
- Direct supplies `grad_i = e_i + (ρ/2)·C·c_i` (exact) and `hess_i = 1 + (ρ/2n)c_i²`, which is exactly
  `diag(I + (ρ/2n)ccᵀ)`. Surrogate supplies `grad_i = e_i(1+ρc_i²)`, `hess_i = 1+ρc_i²` — exactly diagonal.
- **No `if rho == 0` branch exists**; ρ=0 reduces analytically. `zero_grad_tol` flooring is unreachable
  on the canonical branch.
- Early stopping OFF; bagging OFF; `deterministic` / `force_row_wise` / `force_col_wise` /
  `num_threads` unset; `min_sum_hessian_in_leaf` unset (default 1e-3); `reg_lambda = 10.857` absolute.
- **Manuscript App. E matches the code exactly — no discrepancy found between the manuscript
  equations and the implementation.** This is itself a reportable audit result.

### E.3 Required numerical diagnostics

For each training block `T ∈ {fold1…fold7, dev-pool, production}` and each ρ on the full 82-point grid
(with the five display anchors `{0.0104811, 0.1, 0.954095, 10.4811, 100}` broken out), using the **real**
`c_T = y_T − ȳ_T` (indices rebuilt via `run_temporal_cv._load_and_split_data` +
`build_rolling_origin_protocol`, index-hash verified against `V6/protocol/**/folds.json`):

1. `n_T`, `ȳ_T`, `V̂ar_T(y) = mean(c_T²)` (ddof=0, matching `paper_mechanism_metrics`).
2. Direct supplied Hessian `h_i = 1 + (ρ/2n)c_i²` → min / median / mean / p99 / max.
3. Penalty-only diagonal increment `(ρ/2n)c_i²` → min / median / max, and its share of total curvature.
4. Full rank-one curvature eigenvalue in the centered-target direction:
   `1 + (ρ/2n)‖c_T‖² = 1 + (ρ/2)V̂ar_T(y)`.
5. Ratio of exact to supplied directional curvature along `ĉ = c/‖c‖`:
   `R_T(ρ) = [1 + (ρ/2)V̂ar_T(y)] / [1 + (ρ/2n)·(Σc_i⁴ / Σc_i²)]`.
6. Surrogate weights `w_i = 1 + ρc_i²`: min/median/max, **top-1 % weight share**, and effective sample
   size `(Σw)²/Σw²` (closes audit H-7).
7. **Fixed-λ leaf-shrinkage diagnostic — a STYLIZED implementation diagnostic (A10, A15).**
   `reg_lambda = 10.857207234911057` is absolute and enters `leaf = −G/(H+λ)`. Because the Surrogate's
   `h_i = 1 + ρc_i²` grows with ρ while the Direct's stays ≈1, the *effective* leaf shrinkage
   `H_leaf/(H_leaf+λ)` differs between the families at matched ρ.
   **This calculation is explicitly stylized.** Report it for a *nominal* leaf of
   `min_child_samples = 93` observations drawn (i) uniformly at random from the training block and
   (ii) from the upper/lower `c²` deciles, and label every such number
   `diagnostic_type = stylized_nominal_leaf`. It is an illustration of a mechanism that *could* operate,
   **not** a measurement of what the fitted trees actually do.
   To upgrade it beyond stylized, the **actual trained-leaf Hessian sums** must be measured — obtainable
   from `booster.predict(X, pred_leaf=True)` on the training block for the relevant fitted models,
   aggregating `Σ_leaf h_i` per tree/leaf. That measurement is **optional** and is scheduled only if the
   stylized diagnostic shows a large family asymmetry that the reports would otherwise be tempted to
   interpret. Until it is run, no claim of the form "the Surrogate's ρ effect is partly implicit
   de-regularization" may be made — only "a stylized calculation indicates this is possible and has not
   been measured."
8. **`min_sum_hessian_in_leaf` check (A10).** Report `min_i h_i ≥ 1` and note that any leaf satisfying
   `min_child_samples = 93` has `ΣH ≥ 93 ≫ 1e-3`. Record the constraint as **checked and non-binding**.
   Only if some block/ρ produced `ΣH < 1e-3` would it be escalated — which cannot happen while `h_i ≥ 1`.

**Cost: zero fits.** One column read plus index reconstruction. Minutes, locally.

### E.4 Decision rule for the "effectively gradient-only" characterisation

Let `M(ρ) = max_i (ρ/2n)c_i²` and `R_T(ρ)` from E.3-5.

- **Accept** "effectively gradient-only, retaining essentially native curvature" if at every displayed ρ
  and every training block `M(ρ) < 1e-2` **and** `R_T(ρ) ≥ 1.2` for ρ ≥ 1.
- **Reject** if `M(ρ) ≥ 5e-2` at any displayed ρ, or if `R_T(ρ) < 1.05` throughout (in which case the
  diagonal is a good approximation and the manuscript's current wording stands).
- **Indeterminate** otherwise — describe the implementation with the measured numbers, not a label.

*Indicative only* (synthetic draw, `V̂ar(y) ≈ 0.4955`, `n = 310,147`; **to be replaced by real numbers**):
supplied `h` median 1.0000000038 → 1.0000363, max 1.0018; exact directional curvature 1.0026 → 25.78;
`R ≈ 1.00 / 1.02 / 1.24 / 3.60 / 25.78` at ρ = 0.0105 / 0.1 / 0.954 / 10.48 / 100.

---

## F. P0-2 — native vs custom ρ=0 parity

### F.0 The A1 separation (governing rule for this whole section)

Two experiments, never conflated:

| | **Track H — historical-setting** | **Track P — pinned-determinism** |
|---|---|---|
| Settings | Exactly as executed for the frozen artifacts: `n_jobs=1`, `OMP_NUM_THREADS=1`, **no** `deterministic`, **no** `force_row_wise`/`force_col_wise`, **no** `num_threads` | Adds `deterministic=True`, `force_row_wise=True`, `num_threads=1` to **every** cell, native and custom alike |
| Question answered | Do the frozen artifacts reproduce, and what is the historical native↔custom gap under the settings that actually produced the paper's numbers? | Are the two code paths algebraically/numerically equivalent once environment nondeterminism is removed? |
| What it can certify | The historical positive-ρ path artifacts | The implementation, prospectively |
| What it may **not** be used for | — | **A pinned Track-P equivalence result may never be cited as validating the historical positive-ρ path artifacts.** Any such claim must come from Track H (or from regeneration under pinned settings). |

Both tracks are reported. Every parity number in every table carries a `track ∈ {historical, pinned}`
column, and `reports/RHO_ZERO_PARITY_REPORT.md` states the rule above verbatim.

### F.1 What is already known (do not re-derive)

| Evidence | Value | Source |
|---|---|---|
| CCAO held-out mean / median / p95 / max \|Δlog\| | 3.239e-2 / 2.418e-2 / 9.095e-2 / 3.089e-1 | `V6/final_local_results/rho0_split_audit.csv` |
| CCAO 2025 mean / max | 3.056e-2 / 2.825e-1 | same |
| Pearson vs native | 0.99774 (held-out), 0.99796 (2025) | same |
| Direct ρ=0 vs Surrogate ρ=0 | **bit-identical on both splits** | same + external audit (`0.0`) |
| Same frozen parameter vector on all three rows | `lgbm_params_sha256 8f0f2acd…` | same |
| Low-capacity real-data agreement (Cook/Wayne/Philadelphia, n=4,000, 60 trees/15 leaves/depth 4) | mean \|Δ\| **1.3–1.8e-8**, max **1.0–1.3e-7** | `analysis/external_jurisdiction_benchmark_v1/audits/zero_rho_parity.csv` |
| Early stopping | OFF both paths; both always grow 994 trees | `experiment_spec.json`, `run_temporal_cv.py:876-889` |
| Seeds | `random_state=2025` both; bagging OFF; `colsample_bytree` seed derived from the same `seed` | `V6/lgbm_config.json` |

**Already ruled out:** `best_iteration`, early stopping, seeds, row subsampling, differing tree counts,
and any positive-ρ objective bug (Direct ≡ Surrogate at ρ=0 exactly).

### F.2 Ranked diagnostic tree

| Rank | Hypothesis | Evidence present | File | Minimal diagnostic | Refit? |
|---|---|---|---|---|---|
| **H1** | **Float32 label quantisation × capacity amplification.** LightGBM stores `Dataset` labels as 32-bit. Native (Cell A) sees `float32(y)` with `y ≈ 12.44` (ULP ≈ 9.5e-7); the custom path sees `float32(y − ȳ)` with `|·| ≲ 2` (ULP ≈ 1.2e-7). Effective labels differ by ~2.4e-7 mean / 5.0e-7 max **before the first tree**. At 60 trees/15 leaves this shows as ~1e-7; at 994 trees/573 leaves with `min_split_gain = 1.06e-3`, near-tied split gains flip and the ensembles diverge structurally. | The two existing parity results differ by **6 orders of magnitude** and differ **only in capacity and n** — the code path is identical (`match_native_init=True` in both). Float64 check: native label error mean 2.38e-7 / max 4.77e-7 vs custom 1.21e-8 / 1.19e-7. | `boosting_models.py:504-527`; `run_temporal_cv.py:876-889` | (a) free arithmetic: emit `float32(y)` vs `float32(y−ȳ)+ȳ` error distributions on the real dev pool; (b) **Cell B** — native L2 on the **same centered labels**, `boost_from_average=False`, `init_score=0`, `ȳ` added back — run in **both tracks**. | Yes (Cell B) |
| **H2** | **Environment/threading nondeterminism.** `force_row_wise`/`force_col_wise` auto-selected by a timing test; `deterministic` unset. Different accumulation order ⇒ different FP sums ⇒ different splits. | Untested. `n_jobs=1` reduces but does not eliminate auto-selection. | `V6/lgbm_config.json`; sbatch env | **Track H replicate test:** run Cell A twice under historical settings and compare bitwise; and re-run a sample of frozen configs against their cached predictions (§F.3a). This is the *only* way to bound how much of the historical gap is nondeterminism rather than H1. | Yes |
| **H3** | `boost_from_average` init-value precision (LightGBM computes it from float32 labels; the custom path uses float64 `np.mean(y)`). | `base_score_matches_mean_y = True` in the external audit is an `np.isclose` check, not bit equality. | `boosting_models.py:504` | Read the booster's own init score / first-iteration raw prediction and compare with float64 `mean(y)`. | No (read from a Cell-A booster) |
| **H4** | Hessian-scale traps (`min_sum_hessian_in_leaf`, `lambda_l2`, `min_gain_to_split`) | **Eliminated at ρ=0** — both paths supply `hess = 1` identically. Relevant only for ρ>0 (§C-6). | — | none | No |
| **H5** | Different feature matrix / categorical set / row order | Both paths receive the same `X` from the same call site (`run_temporal_cv.py:1357, 1370-1371`); categoricals are pandas `category` in both. | `run_temporal_cv.py:1240-1275` | Hash `X` (dtypes, category orders, row order) once; assert equality across cells. | No |
| **H6** | Prediction transformation | Both use `np.exp`, no smearing (`run_temporal_cv.py:1430-1431`); custom adds `base_score_` in log space first. | `boosting_models.py:529-533` | Symbolic assertion. | No |
| **H7** | Sample weights | None used anywhere. | — | Assert absent. | No |

### F.3 The experiments

**Cells** (naming per A8):

| Cell | **Fixed label — used verbatim in every table, figure and report** | Objective | Labels | Init |
|---|---|---|---|---|
| **A** | **Ordinary LightGBM (standard raw-label native)** | native `objective='mse'` | raw `y` | `boost_from_average=True` — *the paper's current baseline; the learner an assessor would actually run* |
| **B** | **Parity-aligned native L2** | native `objective='mse'` | centered `y − ȳ_T` | `boost_from_average=False`, `init_score=0`, `ȳ_T` added back |
| **C** | **Custom ρ=0 origin** | `LGBCovPenalty(rho=0, ratio_mode='diff', match_native_init=True)` and the Surrogate twin | centered | `boost_from_average=False`, `init_score=0`, `ȳ_T` added back |

These three labels are **fixed strings**. Cell B may become the within-implementation **causal reference**
if §F.4 justifies it, but it is **never** printed as "Ordinary LightGBM", and Cell A is never dropped from
a table in which a native baseline is reported.

Capacities: **T** (60 trees / 15 leaves / depth 4 — reproduces the external audit), **M** (200 / 63 / 8),
**F** (the frozen 994 / 573 / 11 vector).

### F.2b Source-equivalence / provenance check — MUST precede any interpretation of reproduction failure (A12)

*(Logically this precedes §F.3a and is executed first, as job J2b / step 3b. It is placed here so that the
check and the failure taxonomy it produces read together with the reproduction criterion that consumes them.)*

The frozen artifacts were generated from **earlier provenance commits** (`508dc1c2` for the 994-tree
config; `2aa0346a` for the lower-ρ spec; `d3ef45f2` for `final_local_results`) and, in every recorded
case, from a **dirty working tree** whose diff text was **not preserved** — only `git_diff_sha256` was
stored (`2268bae616…`, `fa45001c23…`, `4efe92e877…`). Attributing any reproduction failure to
nondeterminism before ruling out source drift would be an error.

**Procedure (read-only; runs before the reproduction check is interpreted):**

1. **Diff the executed training/prediction path against each recorded provenance commit.**
   `git diff <provenance_commit>..HEAD --stat` restricted to the files that can affect a fit:
   `soft_constrained_models/boosting_models.py`, `run_temporal_cv.py`, `utils/motivation_utils.py`,
   `utils/delta_nl.py`, `canonical_experiment.py`, `params.yaml`, `cv_config.yaml`, `model_params.yaml`.
2. **Classify each hunk** as *touches the executed path* / *does not*. Any hunk inside the objective
   functions, the fit/predict methods, the split builders, the metric functions, the parameter plumbing
   or the data loader is `touches_executed_path = true`.
3. **Inspect every stored code/config hash** and re-verify what can be re-verified:
   `lgbm_params_sha256 8f0f2acd…585b`, `config_id 407d47775760c14d`, `frozen_baseline_hash fcf614d60048cae7`,
   `canonical_model_grid_hash 2ceba22cb08138b1`, `model_grid_hash 23d0e88535bb65ce`,
   `grid_hash 072e4ea94c…`, `delta_nl estimator_spec_hash e85069150b…`, plus the `data_id d4929d43ec19badf`
   and `split_id 3d464d4a611b131b` partitions and the archived fold index hashes.
4. **If any hunk touches the executed path**, run the historical reproduction check inside a
   **temporary read-only Git worktree checked out at the recorded provenance commit**
   (`git worktree add --detach <tmp> <commit>`), with the same `fairness_env` interpreter, and repeat.
   The worktree is created outside the repository tree, is never committed to, and is removed afterwards.
5. **Explicitly acknowledge the irreducible limit:** because the historical dirty diff was never
   archived, exact source reconstruction **may be impossible**. Where that is the case, say so; do not
   present a worktree-based reproduction as proof of the exact historical source state.

**Failure taxonomy — every reproduction failure must be classified into exactly one of these, never
collapsed into "nondeterminism":**

| Class | Definition | Evidence that establishes it |
|---|---|---|
| **F-SRC** *source drift* | A code change between the provenance commit and HEAD touches the executed training/prediction path | Step 2 classification; failure disappears when re-run in the provenance worktree |
| **F-DIRTY** *unreconstructable dirty-state uncertainty* | HEAD and the provenance commit agree on the executed path, but the recorded `git_diff_sha256` shows the generating tree was dirty and the diff text is unavailable | Recorded `git_dirty: true` + missing diff; failure persists in the provenance worktree; magnitude cannot be attributed further |
| **F-ENV** *environment nondeterminism* | Library/BLAS/thread-count/host differences | Failure varies across hosts or thread settings but is stable within a host |
| **F-NUM** *numerical nondeterminism* | Run-to-run variation on the **same** host and settings | The Cell-A duplicate run (F.3a-3) differs from itself |
| **F-IMP** *genuine implementation discrepancy* | A systematic, reproducible difference in what the code computes | Stable, repeatable, host-independent, and traceable to a named code path |

The classification is recorded per config in `tables/frozen_artifact_reproduction.csv`
(`failure_class` column) and drives §F.6.

**#### F.3a Track H — historical-setting (runs first; it is the one that speaks to the paper)**

1. **Frozen-artifact reproduction check.** Under historical settings — and, if §F.2b step 4 applies,
   inside the provenance worktree — re-fit a sample of frozen configurations: Cell A native
   (`252a25d9c0ce796b`), Direct ρ=0 (`1fb838f7d6bfda88`), Surrogate ρ=0 (`5b7875e55e58ac62`), and
   Direct/Surrogate at ρ ≈ 0.954095 and ρ = 100, on the dev pool and the production block. Compare
   against the cached predictions in `V6/baseline_reporting/**` and `V6|V12/reporting_preview/**`.

2. **Tiered reproduction criterion R1–R4 (A13) — kept strictly separate from the parity tiers T1–T4.**
   T1–T4 (§F.4) grade **A/B/C parity comparisons**; R1–R4 grade **reproduction of a frozen artifact by a
   re-run of the same configuration**. The two are never combined into one column or one verdict.

   | Tier | Name | Criterion | Consequence |
   |---|---|---|---|
   | **R1** | **Exact reproduction** | `max\|Δŷ\| = 0` | The frozen artifact is bit-reproducible. Strongest. |
   | **R2** | **Numerical reproduction** | `max\|Δŷ\| ≤ 1e-6` **and** every reported metric identical at displayed precision | Reproducible for every scientific purpose. Frozen artifacts stand. |
   | **R3** | **Materially equivalent reproduction** | `1e-6 < max\|Δŷ\| ≤ 1e-3` **and** a **named benign cause** (from the F.2b taxonomy) **and** no displayed metric changes **and** no scientific conclusion changes | Frozen artifacts stand, with the cause disclosed in the appendix. |
   | **R4** | **Material non-reproduction** | Larger discrepancy, **or** any displayed metric changes, **or** any change capable of altering the paper's scientific interpretation | The frozen positive-ρ artifacts cannot be relied on as-is. **Only R4 contributes to the regeneration gate (§F.6).** |

   A tier is assigned per configuration per split, with the F.2b `failure_class` attached. An **R3
   assigned as F-DIRTY** is acceptable and is reported as such — unreconstructable dirty state is a
   disclosure item, not automatically a regeneration trigger.

3. **A/B/C × T/M/F ladder** under historical settings, on the dev pool and the production block.
4. **Cell A run twice** under historical settings, same host and settings (the F-NUM test for H2).

**#### F.3b Track P — pinned-determinism (runs second; characterises the implementation)**

4. The same A/B/C × T/M/F ladder with `deterministic=True, force_row_wise=True, num_threads=1` on every
   cell. This answers whether the custom-objective API is equivalent to native L2 *in principle*.

**Reported for every pair (A,B), (B,C), (A,C), at every capacity, in every track:** mean / median / p95 /
max |Δlog|, Pearson, and the full metric suite.

**Interpretation:**
- **Track P `B ≡ C` at capacity F (T1/T2)** ⇒ the custom-objective API introduces nothing under pinned
  determinism. This is a statement about the implementation **only**. It does *not* certify the
  historical artifacts (A1).
- **Track H `B ↔ C` at T1/T2, plus frozen-artifact reproduction at R1/R2 (or R3 with a named benign
  cause)** ⇒ the historical A↔C gap is attributable to label representation amplified by ensemble
  capacity, and the frozen positive-ρ path artifacts stand. This is the branch that permits reusing the
  frozen table, and **no regeneration is triggered.**
- **Track P or Track H `B ≢ C` at capacity T** ⇒ a candidate genuine custom-objective defect; root-cause
  by per-iteration raw-score tracing before anything else proceeds. If it resolves to a defect in the
  objective **actually used to generate the frozen positive-ρ paths**, regeneration trigger RG-2 fires.
- **Track H reproduction at R4** ⇒ regeneration trigger RG-3 fires (§F.6), after the F.2b failure class
  has been established. An R4 classified **F-SRC** is first re-tested in the provenance worktree; if it
  becomes R1–R3 there, RG-3 does **not** fire and the finding is recorded as source drift.
- **A↔B or A↔C non-parity at any tier** ⇒ **never a regeneration trigger on its own** (A11). It is a
  characterisation of the label-representation difference and is reported as such.

### F.4 Tiered parity criterion (replaces the single-threshold rule, per A7)

Applied per comparison pair, per track, per capacity, per split. `Δ` is on the log-prediction scale.

| Tier | Name | Criterion | Consequence |
|---|---|---|---|
| **T1** | **Exact parity** | `max\|Δŷ\| = 0` (bitwise identical predictions) | Strongest. The two objects are the same computation. |
| **T2** | **Numerical parity** | `0 < max\|Δŷ\| ≤ 1e-6` **and** every reported metric identical at displayed precision | Attributable to floating-point accumulation. Fully acceptable; contrast is clean. |
| **T3** | **Material agreement** | `1e-6 < max\|Δŷ\| ≤ 1e-3` (≈0.1 % in price) **and** every reported metric agrees at displayed precision **and** a **named, documented, scientifically innocuous cause** | Acceptable **with disclosure**. Contrast may be reported, with the cause stated in the appendix. |
| **T4** | **Non-parity** | Anything larger, **or** any of T2/T3 achieved without an identified cause, **or** any metric differing at displayed precision | Native-to-penalized contrasts are reported **against the within-family ρ=0 origin only**; Cell A is retained descriptively. **T4 on A↔B or A↔C does not trigger regeneration** (A11); only a T4 on **B↔C** can, and then only via §F.6 RG-1. |

**Applying the tiers to the expected comparisons:**
- **B ↔ C** is expected at **T1/T2**: the supplied `(grad, hess) = (e, 1)` is algebraically what
  LightGBM's `RegressionL2loss` computes, on identically represented labels with identical init.
- **A ↔ B** and **A ↔ C** are **not** expected to reach T1–T3 at capacity F. A 1e-6 target there is not
  achievable in principle: the two paths are fed labels differing by up to ~5e-7 before a single tree is
  grown, and 994 rounds over 573-leaf trees amplify that discretely. For A↔C the requirement is therefore
  **not a tolerance but an explanation** — a decomposition into (i) a measured label-representation
  difference and (ii) measured capacity amplification across T→M→F, with A↔B failing by the same
  mechanism, and with the measured **F-NUM** floor (the same-host Cell-A replicate, §F.3a-4) subtracted.
- The tier assignment is recorded per pair in `tables/parity_ladder.csv` and drives Gate G2; **no single
  numeric threshold triggers a rerun on its own.**

**Reference decision (Gate G2).** If Track H shows (a) frozen-artifact reproduction at **R1/R2**, or R3
with a named benign cause, and (b) **B↔C** at **T1 or T2**, then **Cell B — "Parity-aligned native L2" —
may serve as the within-implementation causal reference** for P0 contrasts, with the justification
recorded. **Cell A remains reported throughout as "Ordinary LightGBM (standard raw-label native)"** — the
learner an assessor would actually run. **Cell B is never printed as "Ordinary LightGBM."** Tables carry
both reference columns; prose states which contrast each claim rests on. If the justification does not
hold, contrasts fall back to the within-family ρ=0 origin (Cell C) and Cell A stays the only native row.

### F.6 Regeneration triggers — the ONLY conditions that mandate a full positive-ρ path rerun (A11)

Full 82-point regeneration under pinned settings (job J11 / step 16) is scheduled **if and only if at
least one of the following holds.** Nothing else — in particular **no A↔B or A↔C result at any tier** —
triggers it.

| ID | Trigger | Evidence required |
|---|---|---|
| **RG-1** | **Historical-setting `B↔C` shows scientifically material non-parity** that cannot be attributed to an innocuous numerical effect | Track H `B↔C` at **T4**, *or* at T3 without a named innocuous cause. A T3 with a named benign cause does **not** fire RG-1. |
| **RG-2** | **A defect is found in the custom-objective implementation actually used to generate the frozen positive-ρ paths** | A reproducible, host-independent discrepancy traced to a named code path in `canonical_direct_scaled_grad_hess` / `canonical_surrogate_scaled_grad_hess` / `LGBCovPenalty.fit` / `LGBSmoothPenalty.fit` **as they stood at the provenance commit** — i.e. classified **F-IMP** in §F.2b, not F-SRC in code that post-dates the frozen run |
| **RG-3** | **The frozen positive-ρ artifacts fail the historical reproduction test by a scientifically material amount** | Reproduction tier **R4** (§F.3a-2), after F.2b classification, and — where the class is **F-SRC** — after re-testing in the provenance worktree fails to recover R1–R3 |
| **RG-4** | **An explicit decision that the pinned deterministic implementation becomes the new canonical implementation** | A recorded decision in `protocol_p0_validation.yaml`, taken deliberately for reproducibility reasons, not as a consequence of any single measurement |

**Explicitly NOT triggers:** A↔B or A↔C non-parity at any tier; a Track-P-only B↔C result (A1);
an **R3/F-DIRTY** classification (unreconstructable dirty state is a disclosure item); a T4 arising solely
from the label-representation difference between Cell A and the centered-label cells.

If no trigger fires, the frozen 82-point path table is reused as-is and the P0 pass proceeds on it, with
the reproduction tier and failure class recorded alongside every table that draws on it.

### F.7 Outputs

`reports/RHO_ZERO_PARITY_REPORT.md` (states the A1 rule verbatim and reports T- and R-tiers in separate
tables); `tables/parity_ladder.csv` (`track`, `cell_pair`, `capacity`, `split`, `parity_tier ∈ T1..T4`,
full |Δ| distribution, metric suite); `tables/frozen_artifact_reproduction.csv` (`config_id`, `split`,
`worktree ∈ {HEAD, provenance}`, `reproduction_tier ∈ R1..R4`, `failure_class ∈ {none, F-SRC, F-DIRTY,
F-ENV, F-NUM, F-IMP}`, named cause, |Δ| distribution, metric-level agreement flags);
`tables/source_equivalence_audit.csv` (per file: provenance commit, hunks, `touches_executed_path`,
hash re-verification result); `tables/parity_prediction_deltas.parquet`; `tables/label_quantization.csv`;
`tables/regeneration_triggers.csv` (RG-1..RG-4, fired yes/no, evidence pointer);
`provenance/parity_config_hashes.json`.

---

## G. P0-3 — complete ρ=0 control metrics

**Finding that reshapes this task:** the complete canonical suite at the ρ=0 origin **already exists** in
the frozen path table (verified by direct read):

| Sample | Model | R²_P | MAE | MAPE | RMSE_log | med r | mean r | w-mean r | COD | COV | PRD | PRB | MKI | VEI | β_log | Δ_NL | dCor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Held-out | Ordinary LightGBM (Cell A) | 0.89423 | 75,655 | 0.21201 | 0.28871 | 0.92877 | 0.98865 | 0.92440 | 21.627 | 0.39728 | 1.0695 | −0.09076 | 0.92295 | −26.46 | −0.14960 | 0.11866 | 0.38709 |
| Held-out | Direct ρ=0 | 0.89281 | 75,976 | 0.21192 | 0.28940 | 0.92755 | 0.98612 | 0.92251 | 21.599 | 0.39553 | 1.0690 | −0.08845 | 0.92305 | −26.17 | −0.14738 | 0.11645 | 0.38225 |
| Held-out | Surrogate ρ=0 | *(bit-identical to Direct ρ=0)* | | | | | | | | | | | | | | | |
| 2025 | Ordinary LightGBM (Cell A) | 0.90438 | 78,484 | 0.20779 | 0.27819 | 0.95032 | 1.01550 | 0.94091 | 21.266 | 0.37188 | 1.0792 | −0.10604 | 0.90654 | −28.58 | −0.16408 | 0.12117 | 0.42204 |
| 2025 | Direct/Surrogate ρ=0 | 0.90385 | 79,139 | 0.20773 | 0.27907 | 0.94644 | 1.00900 | 0.93645 | 21.229 | 0.37122 | 1.0775 | −0.10284 | 0.90934 | −28.41 | −0.16120 | 0.12118 | 0.41684 |

CV_mean is present for all metrics except `Δ_NL` (NaN for the ρ=0 / LightGBM / Linear rows —
`delta_nl_cv` covers only the 164 positive-ρ configs).

**Useful magnitude for the manuscript:** the Cell-A↔custom ρ=0 gap is −0.0014 in R²_P on held-out,
against a claimed Direct ρ≈1 gain of +0.005 over native — the implementation gap is roughly **28 % of
the headline effect and of the same sign**. That is precisely why the control matters, and it is now
quantifiable on every metric.

**Work required:**

| Item | Cost |
|---|---|
| Extract the ρ=0 / Cell-A rows on held-out, 2025 and CV_mean from the frozen table | none |
| Add per-fold columns, CV_sd, `Cov_log_residual_log_price` | none |
| Compute `Δ_NL` CV_mean for the 4 missing configs (Cell A, Direct ρ=0, Surrogate ρ=0, Linear) × 7 folds from **cached fold predictions** using the frozen estimator (spec hash `e85069150b…`) | 28 evaluations, minutes, local |
| Add the **Cell B ("Parity-aligned native L2")** reference row on all three regimes, alongside the retained **Cell A ("Ordinary LightGBM (standard raw-label native)")** row | covered by the 9 refits in step 7 |
| PRB SE/t, VEI Significance, smearing sensitivity (§K J11) | ~2 core-h, zero refits |

**Output schema — `tables/zero_control_full.csv`**
`reference_cell ∈ {A, B}, family, rho, config_id, track, evaluation ∈ {fold_1..fold_7, CV_mean, CV_sd, heldout, forward_2025}, n, R2_price, MAE_price, MAPE, RMSE_log, median_ratio, mean_ratio, weighted_mean_ratio, COD, COV, PRD, PRB, PRB_se, PRB_t, MKI, VEI, VEI_significance, Beta_log, Cov_log_residual_log_price, Delta_NL, dCor_e_y, mean_abs_delta_log, median_abs_delta_log, p95_abs_delta_log, max_abs_delta_log, pearson_vs_reference, parity_tier, source_artifact, lgbm_params_sha256`

---

## H. P0-4 — centered-spread comparator

### H.1 Primary form (A2) and the algebra

**Primary — theorem-matched.** Following Cor. `cor:path_scaling` (`f_ρ = ȳ1 + b_ρ(f₀ − ȳ1)`):

```
f_b(x) = ȳ_T + b·(f₀(x) − ȳ_T)
```

At `b = 1` this is **exactly `f₀`**, so the path starts at the baseline by construction — the property
that makes the comparator a path through the fitted model rather than a shifted family. This is also the
map already implemented at `scripts/final_local_results_994.py:586-587`, so the existing
`recalibration_path.csv` is directly reusable as the primary variant.

**Sensitivity — `f₀`-mean-recentered (labelled, secondary).**
`f_b^{(f₀)}(x) = ȳ_T + b·(f₀(x) − f̄_{0,T})`, which at `b=1` equals `f₀ + (ȳ_T − f̄_{0,T})` — i.e. it does
**not** reproduce `f₀`. Reported only as a sensitivity, clearly labelled `variant = center_f0bar`, to
show whether the level shift `ȳ_T − f̄_{0,T}` matters.

**Zero-covariance scale.** With `c = y − ȳ_T` (so `Σ_T c_i = 0`):

```
Cov_T(e_b, y) = mean_T[(ȳ_T + b(f₀ − ȳ_T) − y)·c] = b·Cov_T(f₀, y) − Var_T(y)
```

so `Cov_T(e_b, y) = 0 ⟺ b*_train = Var_T(y)/Cov_T(f₀, y)`, provided `Cov_T(f₀,y) > 0`. ✔ Matches the
P0 spec. **Constant shifts drop out of the covariance, so `b*` is identical for both variants** — the two
differ only by a level shift, which moves level-sensitive metrics (median/mean/weighted-mean ratio, PRD,
MKI, VEI, MAE, MAPE, R²_P) but not β_log or Cov. That is exactly what the sensitivity measures.

`1/R²` is used nowhere. For OLS with intercept `Cov(f₀,y) = Var(f₀)`, which collapses `b*` to `1/R²`;
a fitted LightGBM is not an orthogonal projection, so that identity appears only as a remark.

**β_log is linear in `b` on any evaluation sample.** On evaluation set `V`, `paper_mechanism_metrics`
centers `c` on `V`'s own mean, so constants drop and
`β_V(b) = b·Cov_V(f₀,y)/Var_V(y) − 1`. This means a target β can be hit by **solving for `b` exactly**
and then **evaluating the actual map at that `b`** — no metric interpolation is ever required for the
post-hoc family (relevant to A3).

### H.2 Design decisions and the evidence behind each

| Decision | Choice | Why |
|---|---|---|
| Baseline `f₀` | Per Gate G2: Cell B **"Parity-aligned native L2"** as the within-implementation base **if justified**, **and** Cell A **"Ordinary LightGBM (standard raw-label native)"** reported alongside — never relabelled, never dropped | The comparator base must match the penalized families' origin, but the assessor-facing baseline must remain visible |
| Centering | **`ȳ_T` primary** (A2); `f̄_{0,T}` as labelled sensitivity | `b=1` must reproduce `f₀` |
| Is `f̄_{0,T} − ȳ_T` material? | **Measured, not assumed** — no in-sample training predictions are cached (§C-12) | Reported per training block in `tables/b_star_diagnostics.csv` |
| `b*` | Report **both** `b*_train = Var_T(y)/Cov_T(f₀,y)` and `b*_oof = 1.16727` (existing pooled-OOF neutral value) | §C-3; both development-only |
| Path bounds | `b ∈ [1, 1.25·max(b*_train, b*_oof)]`, never less than the `b` reaching development β_log = 0 | Spans the full common support with Direct and Surrogate, plus overshoot |
| Resolution | **121 points**, geometric in `(b − 1)`, with `b = 1`, `b*_train`, `b*_oof` forced as exact grid points | Matches the 82-point ρ resolution; dense near `b=1` where Direct lives |
| Cross-fold coordinate | Report `u = (b−1)/(b*_T − 1)` alongside raw `b` | Folds have different `b*_T`; `u` overlays them. Analogous to the manuscript's `ρ̃ = ρ·V̂ar(y)` |
| Refit vs post-process | **Post-process cached predictions**; only the 9 native training-block refits are new | The map is a closed-form affine transform of `y_pred_log` |
| Evaluation sets | 7 CV validation folds (**new**), held-out, 2025 forward | The existing `recalibration_path.csv` has held-out and 2025 only |

### H.3 Leakage controls

1. `ȳ_T`, `f̄_{0,T}`, `Var_T(y)`, `Cov_T(f₀,y)`, `b*_T` are computed **only** on the fitting block of the
   relevant regime: fold-`k` training rows for fold `k`; the 344,607 dev pool for held-out; the 382,897
   production block for 2025. Fold means come from `reconstruct_fold_training_means`, which asserts
   index-hash equality against the archived protocol.
2. The **entire b-grid is written to `configs/b_grid_frozen.json` and hashed before any held-out or 2025
   row is scored**; the evaluation script refuses to run if the hash changes.
3. No held-out/2025 outcome enters `b` selection, grid construction, or matching (§I).
4. Disclosure preserved: the held-out block was inspected during earlier development (manuscript L1888),
   so the comparator is described as *frozen-before-evaluation*, **not** preregistered.

### H.4 Output schemas

**`tables/centered_spread_path.csv`** — one row per (variant, evaluation, b):
`variant ∈ {center_ybar (PRIMARY), center_f0bar (SENSITIVITY)}, f0_reference_cell ∈ {A,B}, b, u_normalized, is_b1, is_b_star_train, is_b_star_oof, evaluation ∈ {fold_1..fold_7, heldout, forward_2025}, n, ybar_T, f0bar_T, Var_T_y, Cov_T_f0_y, b_star_train, b_star_oof, R2_price, MAE_price, MAPE, RMSE_log, median_ratio, mean_ratio, weighted_mean_ratio, COD, COV, PRD, PRB, PRB_se, MKI, VEI, Beta_log, Cov_log_residual_log_price, Delta_NL, Delta_NL_raw, dCor_e_y, source_prediction_sha256`

**`tables/centered_spread_ratio_profiles.csv`** — 30 equal-count sale-price bins plus the IAAO
proxy-decile profile with 90 % CIs, at `b = 1`, `b*_train`, `b*_oof` and each matched-β target
(reusing `utils/motivation_utils.py::vei_percentile_group_profile`, L1267-1337).

**`tables/b_star_diagnostics.csv`** — per training block:
`block, n_T, ybar_T, f0bar_T, gap = f0bar_T − ybar_T, Var_T_y, Cov_T_f0_y, Var_T_f0, b_star_train, R2_train, one_over_R2_train, b_star_oof`.

**Report:** `reports/CENTERED_SPREAD_COMPARATOR_REPORT.md`.

---

## I. P0-5 — matched first-order comparison

### I.1 Matching coordinate — development information only

- **D1 (primary): CV-mean β_log** across the 7 rolling-origin validation blocks — already frozen for
  Direct/Surrogate in `combined_path_table_v4_analysis_view.csv`, and produced for the post-hoc path by S7.
- **D2 (sensitivity): pooled out-of-fold β_log** over the concatenated 7 validation blocks (n = 151,153).

Neither touches held-out or 2025 outcomes. D2 guards against the nesting non-exchangeability noted in
C-9; note (A9) that equal weighting means larger later folds do **not** dominate D1 — if anything the
small early folds are over-weighted relative to their sample size, which is why D2 is carried.

### I.2 Attainable common support (measured — this overturns the suggested grid)

| Family | CV-mean β_log range | Held-out | 2025 |
|---|---|---|---|
| Ordinary LightGBM (Cell A) | −0.1382 (point) | −0.1496 | −0.1641 |
| **Direct** | **[−0.1394, −0.0787]** | [−0.1508, −0.0794] | [−0.1640, −0.0894] |
| **Surrogate** | **[−0.1393, −0.0186]** | [−0.1510, +0.0008] | [−0.1646, −0.0148] |
| Post-hoc | continuous; reaches 0 by construction | — | — |

Per-fold CV ranges vary widely (Direct fold 2 spans [−0.1226, −0.0505]; folds 6–7 span ≈[−0.172, −0.107]).

**Consequence:** of the audit's suggested targets `{−0.15, −0.12, −0.09, −0.06, −0.03, 0}`, **−0.15 is
outside both families' CV-mean range and −0.06, −0.03, 0 are unattainable by Direct at any ρ ≤ 100.**

### I.3 Frozen matching procedure — actual fitted configurations, no metric interpolation (A3)

1. **Compute the three-way common support** on D1 (expected ≈ `[−0.1393, −0.0787]`, recomputed at
   execution time, never hard-coded).
2. **Anchor the CORE targets on Direct's own fitted grid — by a deterministic algorithm, not by
   judgement (A15).** Direct has the narrowest support and is therefore the binding family. The six CORE
   targets are produced by the following procedure, which involves no discretion and no held-out
   information, and is written into `configs/matched_beta_frozen.json` before any out-of-time read:

   ```
   INPUT : S = [L, U]              # three-way common support on the D1 coordinate (step 1)
           D = { fitted Direct configs c : beta_dev(c) in S }, sorted ascending by beta_dev
   ASSERT: |D| >= 6                # else reduce K to |D| and record the reduction
   K = 6
   for j = 0 .. K-1:
       q_j      = L + (j / (K-1)) * (U - L)        # equally spaced probe values, endpoints included
       c_j      = argmin_{c in D, c unused} | beta_dev(c) - q_j |   # ties -> smaller rho
       target_j = beta_dev(c_j)                     # THE TARGET IS THE ACHIEVED VALUE, not q_j
       mark c_j used
   OUTPUT: targets = { target_0 .. target_5 }, anchor configs = { c_0 .. c_5 }
   ```

   Because each target is *defined as* the achieved development β_log of a specific fitted Direct
   configuration, Direct matches exactly by construction — zero new fits, zero interpolation, and no room
   to choose favourable points. The probe values `q_j`, the selected `c_j`, the achieved `target_j`, and
   the `|beta_dev(c_j) − q_j|` residuals are all recorded so the selection is fully auditable.
3. **Freeze a matching tolerance `τ` before any out-of-time evaluation.** Default `τ = 0.002` in β_log
   units — roughly 2–3 steps of the 82-point grid, whose mean β_log step across Direct's CV-mean range
   is ≈ 0.00074. `τ` is written into `configs/matched_beta_frozen.json` and hashed.
4. **Selection rule per family per target `q`:**
   - **Post-hoc:** solve `b` exactly from the linear relation in §H.1 and **evaluate the actual map at
     that `b`**. Exact by construction; no interpolation, no refit.
   - **Direct / Surrogate:** select the **actual fitted configuration** whose development β_log is
     nearest to `q`. Accept if `|β_dev − q| ≤ τ`. **Never interpolate held-out or 2025 metrics between
     fitted ρ.** Every reported number is a measured value of a configuration that was actually fitted.
   - If no fitted configuration lies within `τ`: run a **targeted new fit** at a ρ solved from the
     *development* path only (log-linear solve in `(log ρ, β_dev)` between the two bracketing fitted
     points), then evaluate that new configuration on all regimes. The solved ρ, its provenance and its
     achieved β_dev are recorded before out-of-time evaluation.
   - If a targeted fit still misses by more than `τ` (nonmonotone or flat region), emit
     `attained = false` with the achieved value; do not fabricate a match.
5. **Always report the achieved development β_log next to the target**, plus `abs_gap` and
   `match_mode ∈ {exact_solve, nearest_fitted, targeted_fit, not_attained}`.
6. **Nonmonotone paths.** If a target is attained more than once, take the **smallest** parameter value
   (least-intervention convention), record `n_crossings`, and emit the alternatives in
   `tables/matched_beta_crossings.csv`.
7. **EXT targets `{−0.06, −0.03, 0.00}`** are reported for Surrogate and post-hoc only, with Direct rows
   marked `NOT_ATTAINED (max β_log = −0.0787 at ρ = 86.85)`. The non-attainment is itself a result and
   must be stated, never silently dropped. A complementary comparison at each family's own maximum
   attainable correction is also reported.
8. **Freezing.** The matched-configuration table (family, target, selected ρ or b, match_mode, achieved
   β_dev, τ) is written to `configs/matched_beta_frozen.json` and **hashed before any held-out or 2025
   metric is read**. The evaluation script asserts the hash.

### I.4 What is compared at each matched target

Predictive (`R²_P`, MAE, MAPE, RMSE_logP); assessor-facing (median/mean/weighted-mean ratio, COD, COV,
PRD, PRB [+SE], MKI, VEI [+significance]); mechanism (achieved out-of-sample β_log, `Δ_NL`, dCor); ratio
profile (30 equal-count sale-price bins + IAAO proxy deciles with 90 % CIs); temporal transfer (the same
quantities on CV → held-out → 2025 and the CV→2025 change).

**The scientific question, operationally:** at equal *achieved* development-sample β_log, does retraining
(Direct, Surrogate) differ from global rescaling (post-hoc) in accuracy, ratio shape, `Δ_NL`/dCor, or
temporal transfer? The comparator is **descriptive**; the acceptance criterion is completeness.

**Outputs:** `tables/matched_beta_comparison.csv`, `tables/matched_beta_crossings.csv`,
`configs/matched_beta_frozen.json`, `figures/matched_beta_{ratio_profiles,accuracy_equity,mechanism}.pdf`,
`reports/MATCHED_BETA_REPORT.md`.

---

## J. P0-6 — temporal robustness

### J.1 Same-date boundary integrity — measured, not inferred

The split code was re-run read-only and **reproduced every archived fold index hash exactly**:

| Boundary | Boundary date | Rows on that date in train | in val |
|---|---|---|---|
| fold 1 | 2017-02-16 | 46 | 46 |
| fold 2 | 2018-04-23 | 156 | 35 |
| fold 3 | 2019-06-03 | 121 | 54 |
| fold 4 | 2020-08-04 | 123 | 47 |
| fold 5 | 2021-08-17 | 35 | 166 |
| fold 6 | 2022-08-01 | 132 | 38 |
| fold 7 | 2022-11-21 | 158 | 2 |
| dev / held-out | **2023-11-09** | 43 (dev) | 105 (held-out) |
| 2025 forward | year-based | **clean by construction** | — |

Mechanism (`utils/motivation_utils.py:281-302`): origins advance on 15-month date offsets, but the
train/val cut inside each cumulative window is **positional** (`val_size = floor(n·0.10)`,
`train_end_pos = n − val_size`, `.iloc[...]`). The dev/held-out cut is likewise positional
(`split_idx = int(0.9·len(universe))`, L220-223). The guard at `scripts/run_paper_baseline.py:882` uses
`<`, so equal dates pass it.

**Minimal deterministic correction — "snap-back":** move `train_end_pos` backwards to the first index
whose `meta_sale_date` differs, so `max(train date) < min(val date)` strictly. This moves ≤166 rows per
boundary (≤0.5 % of any validation block; 0.39 % at the dev/held-out boundary). Implement as a **new**
function in the P0 folder wrapping `build_rolling_origin_protocol` — **never** edit
`utils/motivation_utils.py`, which would invalidate every frozen index hash.

Design precedents already in-repo: `build_rolling_origin_protocol` **Mode B** (L323-365, activated by
`val_fraction: null`) is a pure-date design, and
`analysis/external_jurisdiction_benchmark_v1/scripts/run_baseline_cv.py:78-109` asserts
`train.max() <= val.min()`. Mode B changes the design more than snap-back does, so snap-back is the
minimal correction and Mode B is the fallback if snap-back proves ambiguous.

### J.2 Repeat parcels — measured exposure

`meta_pin` exists in the parquet and is declared in `params.yaml:277-283`, but **is not loaded by the
pipeline** (`run_temporal_cv.py:741`), so the P0 code re-reads that one column.

Share of validation-block PINs that also appear in that fold's training block:

| fold | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| overlap | 496/5,206 (9.5 %) | 1,338/11,172 (12.0 %) | 2,674/16,741 (16.0 %) | 4,682/22,201 (21.1 %) | 6,564/27,668 (23.7 %) | 8,369/32,305 (25.9 %) | 8,987/33,535 (26.8 %) |

dev ↔ held-out: **10,972 of 36,994** held-out PINs (29.7 %). (dev+held-out) ↔ 2025: **8,805 of 26,048**
(33.8 %). This is larger than the audit's "likely second-order" language implies, and justifies a real
robustness variant — while still not justifying replacing the primary design, which does not block
repeat sales, matching production practice and Candogan et al.

### J.3 The three robustness designs

| Design | Construction | Status | Evaluation sets |
|---|---|---|---|
| **D-SNAP** | Primary design + snap-back at all 8 boundaries | **Strict-date robustness (A5).** Not a replacement for the primary design unless Gates G5a → G5b promote it. | Change only by the ≤166 boundary rows per boundary |
| **D-PURGE** | **Primary repeat-parcel diagnostic (A4).** Evaluation blocks are **preserved exactly** as in the frozen design; from each *training* block, remove every row whose `meta_pin` appears in the **corresponding evaluation block**. Applied to each CV fold, to the dev pool (vs held-out), and to the production block (vs 2025). | **ORACLE overlap-removal robustness DIAGNOSTIC — explicitly not a deployable split (A15)** | **Identical to the frozen design** — so every metric is directly comparable, same denominators, same rows |
| **D-UNSEEN** | **Secondary (A4).** No new fits. Take the **frozen** models' cached predictions and restrict the evaluation to the subset of each evaluation block whose `meta_pin` never appears in that model's training block. | Secondary; evaluation-set-changing, therefore not comparable to the frozen numbers on the same base | Restricted subsets (≈70 % of held-out, ≈66 % of 2025) |

**D-PURGE is an oracle construction and must be labelled as one everywhere it appears.** Building each
training block requires knowing which PINs appear in the *future* evaluation block, which no assessor
could know at training time. It is therefore **a diagnostic that isolates the contribution of
parcel-specific history to the reported results — not a proposed sample-construction rule, not a
deployment split, and not a recommendation to CCAO.** Every table and figure carries
`design_type = oracle_diagnostic`, and the report states in its opening paragraph that D-PURGE could not
be implemented prospectively. Its purpose is precisely that it *over*-corrects: if the path conclusions
survive an oracle removal of all repeat-parcel information, they are not driven by that information.

Rationale for A4: purging training while preserving evaluation answers "does parcel-specific history in
training inflate the result?" **without** confounding the answer with a change of evaluation sample.
D-UNSEEN answers the complementary question "how do the frozen models behave on genuinely unseen
parcels?" but on a different denominator, so it is reported separately and never differenced against the
frozen numbers.

D-PURGE training-set sizes will shrink (retained-n reported per block); D-UNSEEN costs **zero fits**
because it is a row subset of already-cached predictions.

### J.4 Grid for the robustness runs

A **21-point ρ SCREENING grid per family** drawn from the frozen 82-point grid: ρ = 0 plus every 4th grid
index, force-including the five display anchors `{0.0104811, 0.1, 0.954095, 10.4811, 100}` and the four
candidate-region endpoints `{0.202359, 0.355648, 2.222996, 2.559548}`.
Per fitted design: `1 native + 21 Direct + 21 Surrogate = 43 configs × (7 folds + held-out + 2025) = 387 fits`.
Fitted designs: D-SNAP and D-PURGE. D-UNSEEN: 0 fits.

**This is explicitly a screening grid (A14).** It is dense enough to reveal the β_log path shape and
ordering, the Δ_NL valley/rebound, the dCor rebound, the accuracy/equity anchors and the approximate
candidate-region endpoints — but it is 4× coarser than the frozen path, so an apparent shift in a
turning point or a rebound can be a grid artifact rather than a real change. Any trigger that fires on
this grid must therefore pass the **G5a local refinement** in §J.6 before it is believed.

### J.5 What would force the manuscript's temporal claims to be revised

Applied to D-SNAP vs the frozen primary, and to D-PURGE vs the frozen primary:

1. **Sign or ordering change in the β_log path** — Direct and Surrogate swapping which family attains
   more first-order correction at comparable ρ, or loss of monotonicity.
2. **Loss of the Surrogate dCor rebound** (currently held-out 0.382 → 0.250 → 0.267). This is the paper's
   single most valuable empirical object.
3. **Loss of the Surrogate Δ_NL rebound** (currently 0.099 at ρ≈1 → 0.124 at ρ=100).
4. **Candidate-region endpoints moving by more than roughly a factor of two in ρ**
   (Direct `[0.3556, 2.5595]`, Surrogate `[0.2024, 2.2230]`).
5. **Reversal of "moderate regularization need not impose an accuracy cost"** — the R²/MAE improvement
   near ρ≈1 disappearing.

Ordinary shifts in absolute accuracy under D-PURGE are **expected** (parcel history is genuinely
informative) and threaten nothing, because every claim is within-path. That distinction is stated
explicitly in the report.

### J.6 Gates G5a / G5b — two-stage promotion rule for D-SNAP (A5 + A14)

D-SNAP is reported as **strict-date robustness**. A trigger firing on the 21-point *screening* grid is an
**alert, not a verdict** — a coarse grid can move an apparent turning point or flatten a rebound purely
by omitting the intervening ρ values.

**G5a — coarse-grid alert → targeted local refinement.**
If any of the five §J.5 triggers fires on the D-SNAP screening grid:
1. Identify the affected region on the ρ axis (the screening indices bracketing the changed turning
   point, rebound, ordering swap or accuracy anchor).
2. **Fill in the missing original-grid ρ values inside that region** — i.e. the frozen 82-point grid
   points that the every-4th-index screening grid skipped, extended by one screening step on each side.
   Typically 6–12 additional ρ per affected family per region.
3. Refit those points under the D-SNAP protocol on all 9 evaluations
   (≈ 9 fits per ρ; a 12-ρ refinement for both families ≈ 216 fits ≈ 9 core-hours).
4. Re-evaluate the trigger on the refined path.

**G5b — confirmed material change → promotion.**
Only if the trigger **survives** the refinement is D-SNAP promoted to the primary temporal design, and
only then does **full 82-point regeneration under the D-SNAP protocol become mandatory** (step 17:
168 configs × 9 evaluations), because the frozen path would then no longer describe the paper's design.

**Direct-promotion exception.** A **very large qualitative reversal** may proceed straight to G5b without
refinement — specifically: a sign change in the β_log path, a reversal of Direct/Surrogate ordering that
holds across the whole screening grid, or complete disappearance of the Surrogate dCor rebound (dCor
monotone across all 21 screening points). **Minor candidate-region, Δ_NL or dCor differences must first
survive local refinement** and may never promote on the screening grid alone.

If no trigger fires, or every trigger dies at G5a, the primary design stands and D-SNAP is reported as a
robustness table plus one sentence in the temporal section. The G5a refinement outcome is recorded in
`tables/dsnap_refinement.csv` regardless.

**Outputs:** `reports/TEMPORAL_ROBUSTNESS_REPORT.md`; `tables/boundary_exposure_audit.csv`;
`tables/repeat_pin_exposure.csv`; `tables/robustness_path_dsnap.csv`; `tables/dsnap_refinement.csv`;
`tables/robustness_path_dpurge.csv` (carrying `design_type = oracle_diagnostic`);
`tables/robustness_unseen_subset.csv`; `tables/robustness_vs_frozen_deltas.csv`;
`configs/split_protocol_dsnap.json`, `configs/split_protocol_dpurge.json` (with index hashes),
`configs/dsnap_refinement_grid.json`.

---

## J-bis. Smearing sensitivity — development-only estimation (A6)

The manuscript retransforms by direct exponentiation (§2.2 L358), correctly disclaimed; audit H-8 asks
for one smearing-corrected sensitivity. Duan's factor is `s = mean(exp(residual))`.

**Estimation rule (mandatory):**
- For every configuration, `s` is estimated from **pooled development out-of-fold residuals** — the
  concatenated 7 CV validation blocks, which are cached for every config. This is development-only and
  avoids in-sample optimism, and requires **zero refits**.
- For the **held-out** evaluation, `s` comes from that development pool. For the **2025 forward**
  evaluation, the fitting block is the 382,897-row production block, which has no out-of-fold analogue;
  the same development-estimated `s` is applied and **the assumption is stated explicitly** as a
  limitation in the report.
- **Estimating `s` on the held-out or 2025 block is prohibited.** The P0 code asserts that the residual
  source for `s` is a development artifact and refuses to run otherwise; `tests/test_p0_assertions.py`
  covers this.
- Both uncorrected and smearing-corrected values are reported side by side at the baselines and the five
  display anchors; the correction is a **sensitivity**, never the headline number.

Output: `tables/smearing_sensitivity.csv` with columns `config_id, family, rho, s_source ∈ {dev_oof}, s, evaluation, metric, value_uncorrected, value_smeared`.

---

## K. Compute / Slurm plan

### K.1 Measured unit costs (from the historical logs — not estimates)

| Operation | Measured |
|---|---|
| Native LightGBM 994-tree fit on 344,607 rows + predict 38,290 | `fit_predict_sec = 148.1` (single-threaded) |
| Native 994-tree fit on 382,897 rows + predict 26,641 | `fit_predict_sec = 155.0` |
| Custom-objective (Direct) 994-tree fit, same blocks | `fit_predict_sec ≈ 179.4` |
| Full 7-fold CV over 104 configs (728 fits) | 6,245 s wall on 16 workers ⇒ **≈137 core-s per fold fit** |
| Memory | 180 GB requested for 16 workers ⇒ **≈11 GB per worker** |

Everything is **CPU-only** — no GPU code exists in the repo (zero `--gres`/`--gpus`); `n_jobs=1` and
`OMP_NUM_THREADS=1` throughout.

### K.2 Job plan

| # | Job | Where | Fits | Core-h | Wall (16 workers) | Mem | Depends on | Reuse instead of refit? |
|---|---|---|---|---|---|---|---|---|
| J1 | P0-1 objective arithmetic | **local** | 0 | <0.1 | ~5 min | 8 GB | — | Entirely |
| J2 | P0-6a boundary + repeat-PIN exposure audit | **local** | 0 | <0.1 | ~5 min | 16 GB | — | Entirely |
| J2b | **Source-equivalence / provenance audit** (§F.2b): diff HEAD vs each provenance commit over the executed path; classify hunks; re-verify all stored hashes; create a temporary read-only Git worktree at the provenance commit **only if** a hunk touches the executed path | **local** | 0 | <0.1 | ~10 min | 8 GB | — | Entirely — read-only Git and hash checks |
| J3a | **Track H** — frozen-artifact reproduction (6 configs × 2 blocks, at HEAD and, if J2b requires, in the provenance worktree) + A/B/C × T/M/F ladder + same-host Cell-A replicate, historical settings | Slurm | ~34 (+12 if the worktree re-test is needed) | ~1.5–2 | ~15–20 min | 64 GB | J1, **J2b** | No — this is the diagnostic |
| J3b | **Track P** — A/B/C × T/M/F ladder, pinned determinism | Slurm | ~18 | ~0.7 | ~10 min | 64 GB | J1 | No |
| J4 | **9 native refits at the canonical config, historical settings, in-sample prediction retained** (7 fold-training blocks + dev pool + production block) | Slurm | 9 | ~0.5 | ~10 min | 64 GB | Gate G1 | No — `f̄_{0,T}`, `Cov_T(f₀,y)` are not cached (§C-12). **Shared by P0-2 Cell B, P0-3 and P0-4.** |
| J5 | P0-3 zero-control assembly + 28 missing `Δ_NL` evaluations | **local** | 0 | ~0.3 | ~15 min | 32 GB | J4, Gate G2 | Yes |
| J6 | P0-4 centered-spread path: 121 b × 9 evaluations × 2 variants, full metric suite | Slurm array (9 tasks) | 0 | ~4 | ~30 min | 32 GB/task | J4 | **Yes — pure post-processing of cached predictions** |
| J7 | P0-5 matched-β selection + **targeted fits where `τ` cannot be met** (worst case ~6 configs × 9 evaluations) | local + Slurm | 0–54 | ~2.5 | ~20 min | 32 GB | J6 | Mostly — Direct is exact by construction; post-hoc solves exactly; only Surrogate may need targeted fits |
| J8 | **D-SNAP** 21-point **screening** path | Slurm | 387 | ~16 | ~1 h | 180 GB | J2 | No |
| J9 | **D-PURGE** 21-point screening path (**oracle diagnostic**) | Slurm | 387 | ~15 | ~1 h | 180 GB | J2 | No (training sets slightly smaller) |
| J8b | *(conditional, **Gate G5a**)* D-SNAP **targeted local refinement**: fill in the skipped original-grid ρ values around each affected region (~6–12 ρ per affected family) | Slurm | ~110–220 | ~5–9 | ~30–40 min | 180 GB | J8 + a fired J.5 trigger | No |
| J9b | **D-UNSEEN** subset evaluation | **local** | 0 | ~0.3 | ~15 min | 32 GB | J2 | **Yes — row subset of cached predictions** |
| J10 | Inferential extras: PRB SE/t, VEI Significance, development-only smearing sensitivity | Slurm array | 0 | ~2 | ~15 min | 32 GB | J4 | Yes — entirely post-processing |
| J11 | *(contingent, **§F.6 RG-1 / RG-2 / RG-3 / RG-4 only**)* full path regeneration under pinned settings: 168 configs × 7 folds + 168 × 2 OOS. **Never scheduled by an A↔B or A↔C result.** | Slurm | ~1,510 | **~62** | ~4 h | 180 GB | Gate G2 + a fired RG trigger | No |
| J12 | *(contingent, **Gate G5b only** — i.e. after a G5a refinement confirms the change, or after a very large qualitative reversal)* full path regeneration under the **D-SNAP** protocol | Slurm | ~1,510 | **~62** | ~4 h | 180 GB | Gate G5b | No |
| J13a | *(optional, §E.3-7)* actual trained-leaf Hessian sums via `pred_leaf=True` on the training block, to upgrade the fixed-λ diagnostic beyond stylized | Slurm | 0 | ~1 | ~10 min | 64 GB | J4 | Yes — leaf indices from already-fitted models where retained; otherwise a small number of refits |
| J13 | Freeze, hash, manifest, figures | **local** | 0 | ~0.3 | ~20 min | 16 GB | all | — |

**Expected path total: ≈ 44 core-hours, ~4–5 h wall end-to-end** (J1–J10, J13; no contingency fires).
**With the G5a refinement only: ≈ 50–53 core-hours, ~5–6 h wall.**
**Worst case with one full regeneration (J11 *or* J12): ≈ 112–115 core-hours, ~9 h wall.**
**Both regenerations: ≈ 175 core-hours, ~13 h wall.**
Dominant costs are the two fitted robustness designs (J8 + J9 ≈ 31 core-h), not the parity or comparator
work. The full-regeneration contingencies remain affordable, which is precisely why they are gated behind
§F.6 and G5b rather than run by default.

### K.3 Partitions and templates

**Preferred:** `sched_mit_sloan_batch_r8` — 64 cores, 507 GB, rocky8, 4-day limit; the repo default in
~39 of 41 sbatch scripts.
**Overflow mix:** `-p sched_mit_sloan_batch_r8,ou_sloan_batch` (the repo's `PART_MIX` pattern, minus
`mit_normal` per the brief).
**Long jobs:** `sched_mit_sloan_interactive_r8` (14-day limit) if a contingency regeneration is run
un-sharded.
**Shardable, restartable work only:** `mit_preemptable` with `--requeue`, exactly as
`scripts/run_final_local_delta_nl.sbatch` already does.
**No GPU partitions are relevant.**

**Exact scripts to adapt (copy into the P0 folder; do not edit in place):**

| New job | Template |
|---|---|
| J2b | no template needed — read-only `git diff` / `git worktree add --detach` plus hash re-verification in `code/p0_2_source_equivalence.py` |
| J3a, J3b, J4, J13a | `scripts/run_paper_v6_preselection_994_baseline_report.sbatch` (8c / 120 G / 12 h) |
| J6, J10 | `scripts/run_final_local_recalibrate.sbatch` (2c / 32 G / 2 h); `scripts/run_final_local_delta_nl.sbatch` (array `0-15%12`, `--requeue`) |
| J8, J8b, J9, J11, J12 | `scripts/run_paper_v6_preselection_994_cv.sbatch` (16c / 180 G / 36 h) + `scripts/run_paper_v6_preselection_994_oos_array.sbatch` (array `0-7`, 2c / 80 G / 8 h); J8b reuses the same pair with an explicit `--rho-values` list |
| DAG glue | `scripts/run_paper_v12_lower_rho_stage.sbatch` (2c / 16 G / 2 h); drivers `scripts/submit_paper_v6_preselection_994.sh`, `scripts/submit_paper_v12_lower_rho_v2.sh` |

**Safe parallelism:** J1 ∥ J2 ∥ J2b; J3a after J2b (the source-equivalence result determines whether the
reproduction check runs at HEAD or in the provenance worktree); J3b after J3a (so the historical result is
never contaminated by a pinned rerun of the same cell); then J8 ∥ J9 ∥ J9b ∥ (J4 → J5 ∥ J6 → J7).
J8b runs only if a screening trigger fires. J6 shards by evaluation block (9 tasks); J8/J8b/J9 shard by
ρ-chunk via the existing `--rho-chunk INDEX/N` flag.

**Git worktree hygiene (J2b).** The provenance worktree is created with `git worktree add --detach` into a
path **outside** the repository tree, is used read-only, is never committed to or checked out over the
working branch, and is removed with `git worktree remove` when the check completes. Its path and the
commit it points at are recorded in `provenance/PREFLIGHT.json`.

**Environment pinning.** Every job uses `/home/nacevedo/.conda/envs/fairness_env/bin/python`,
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, LightGBM pinned to 4.6.0.
`deterministic` / `force_row_wise` / `num_threads` are set **only** in Track P and in the contingency
regenerations — **never** in Track H or in J4, which must reproduce the historical configuration exactly.

---

## L. Proposed file tree under `analysis/p0_major_revision_validation/`

```
analysis/p0_major_revision_validation/
├── README.md                                  # scope, status, canonical artifact list, supersession note
├── protocol_p0_validation.yaml                # THE frozen protocol: baseline commit, env pins, config hashes,
│                                              #   Track H / Track P separation rule, tiered parity criterion,
│                                              #   gates G1-G5, matching tolerance tau, forbidden actions
├── configs/
│   ├── frozen_lgbm_994.json                   # copy of V6/lgbm_config.json + sha256 re-verification
│   ├── parity_cells.json                      # Cells A/B/C x capacities T/M/F x tracks H/P, exact params each
│   ├── b_grid_frozen.json                     # 121-point b-grid  (HASHED BEFORE OOS EVALUATION)
│   ├── matched_beta_frozen.json               # CORE + EXT targets, tau, selected configs, match_mode
│   │                                          #   (HASHED BEFORE OOS EVALUATION)
│   ├── split_protocol_dsnap.json              # snap-back indices + hashes
│   ├── split_protocol_dpurge.json             # training-purged indices + hashes; evaluation sets UNCHANGED
│   ├── unseen_subset_definition.json          # D-UNSEEN row masks per (config, evaluation block)
│   ├── robustness_rho_grid.json               # the 21-point SCREENING grid
│   └── dsnap_refinement_grid.json             # G5a: the skipped original-grid rho values per affected region
├── code/
│   ├── p0_common.py                           # loaders (via run_temporal_cv._load_and_split_data ONLY),
│   │                                          #   provenance stamping, index-hash verification,
│   │                                          #   canonical-metric wrappers (imports utils/, never reimplements)
│   ├── p0_1_objective_audit.py
│   ├── p0_2_source_equivalence.py             # F.2b: provenance diff, hunk classification, hash re-verify,
│   │                                          #   temporary read-only worktree management
│   ├── p0_2_parity_tracks.py                  # Track H (incl. R1-R4 artifact reproduction) and Track P
│   ├── p0_3_zero_control.py
│   ├── p0_4_centered_spread.py                # extends scripts/final_local_results_994.py::run_recalibration
│   ├── p0_5_matched_beta.py                   # nearest-fitted selection, tau enforcement, targeted-fit solver
│   ├── p0_6_temporal.py                       # exposure audit + D-SNAP / D-PURGE / D-UNSEEN builders
│   ├── p0_7_inferential_extras.py             # PRB SE/t, VEI Significance, dev-only smearing
│   └── p0_freeze.py
├── slurm/
│   ├── 01_parity_track_h.sbatch
│   ├── 02_parity_track_p.sbatch
│   ├── 03_native_refits.sbatch
│   ├── 04_centered_spread_array.sbatch        # array 0-8, one per evaluation block
│   ├── 05_matched_beta_targeted_fits.sbatch
│   ├── 06_temporal_dsnap.sbatch               # 21-point SCREENING grid; + _oos_array variant
│   ├── 07_temporal_dpurge.sbatch              # oracle diagnostic; + _oos_array variant
│   ├── 08_inferential_extras.sbatch
│   ├── 09_dsnap_refinement.sbatch             # CONDITIONAL (Gate G5a) — targeted local refinement
│   ├── 10_full_path_regen.sbatch              # CONTINGENT (only a fired F.6 RG-1/2/3/4) — not default
│   ├── 11_full_path_regen_dsnap.sbatch        # CONTINGENT (only Gate G5b) — not default
│   ├── 12_trained_leaf_hessians.sbatch        # OPTIONAL (E.3-7 upgrade beyond stylized)
│   └── submit_p0.sh                           # DAG driver with explicit gate stops; contains NO
│                                              #   unconditional submission of 10_ or 11_
├── logs/
├── tables/                                    # csv + parquet twins
│   ├── objective_scaling_audit.csv
│   ├── direct_hessian_magnitudes.csv
│   ├── surrogate_weight_distribution.csv
│   ├── effective_leaf_shrinkage.csv           # diagnostic_type = stylized_nominal_leaf
│   ├── trained_leaf_hessians.csv              # OPTIONAL; only if E.3-7 is upgraded
│   ├── label_quantization.csv
│   ├── source_equivalence_audit.csv           # F.2b: per-file hunks, touches_executed_path, hash checks
│   ├── frozen_artifact_reproduction.csv       # reproduction_tier R1-R4 + failure_class
│   ├── regeneration_triggers.csv              # RG-1..RG-4 fired yes/no + evidence pointer
│   ├── parity_ladder.csv                      # carries track + parity_tier (T1-T4) columns
│   ├── parity_prediction_deltas.parquet
│   ├── zero_control_full.csv
│   ├── b_star_diagnostics.csv
│   ├── centered_spread_path.csv
│   ├── centered_spread_ratio_profiles.csv
│   ├── matched_beta_comparison.csv
│   ├── matched_beta_crossings.csv
│   ├── boundary_exposure_audit.csv
│   ├── repeat_pin_exposure.csv
│   ├── robustness_path_dsnap.csv
│   ├── dsnap_refinement.csv                   # G5a outcome (present even when no trigger fires)
│   ├── robustness_path_dpurge.csv             # design_type = oracle_diagnostic
│   ├── robustness_unseen_subset.csv
│   ├── robustness_vs_frozen_deltas.csv
│   ├── prb_inference.csv
│   ├── vei_significance.csv
│   └── smearing_sensitivity.csv
├── figures/                                   # P0-only; NEVER writes into paper/img/
│   ├── matched_beta_ratio_profiles.pdf
│   ├── matched_beta_accuracy_equity.pdf
│   ├── matched_beta_mechanism.pdf
│   ├── parity_capacity_ladder.pdf             # both tracks overlaid, clearly labelled
│   └── centered_spread_vs_rho_paths.pdf
├── reports/
│   ├── P0_IMPLEMENTATION_AUDIT.md
│   ├── RHO_ZERO_PARITY_REPORT.md              # states the A1 rule verbatim
│   ├── ZERO_CONTROL_REPORT.md
│   ├── CENTERED_SPREAD_COMPARATOR_REPORT.md
│   ├── MATCHED_BETA_REPORT.md
│   ├── TEMPORAL_ROBUSTNESS_REPORT.md
│   └── MANUSCRIPT_IMPACT_MEMO.md              # WRITTEN LAST, after Gate G4
├── provenance/
│   ├── PREFLIGHT.json                         # branch, HEAD, env versions, provenance-worktree path+commit
│   ├── POSTFLIGHT.json
│   ├── worktree_diff.patch                    # OUR OWN full diff text (closes the gap left by
│   │                                          #   git_diff_sha256 in the historical runs)
│   ├── provenance_commit_diffs/               # HEAD vs each recorded provenance commit, executed-path only
│   ├── DIRTY_STATE_LIMITATION.md              # explicit statement that the historical dirty diff was never
│   │                                          #   archived and exact source reconstruction may be impossible
│   ├── input_artifact_hashes.json
│   ├── output_artifact_hashes.json
│   └── slurm_graph.json
└── tests/
    └── test_p0_assertions.py                  # split counts 344,607/38,290/26,641; fold index hashes;
                                               #   b=1 reproduces f0 EXACTLY under the ybar-centered map;
                                               #   b* zeroes Cov_T; grid- and matching-hash freeze ordering;
                                               #   no held-out/2025 metric is ever interpolated;
                                               #   smearing residual source is a development artifact;
                                               #   D-PURGE evaluation sets identical to the frozen design;
                                               #   T-tiers and R-tiers never share a column or a verdict;
                                               #   no RG trigger can be set by an A-B or A-C comparison;
                                               #   10_/11_ regen sbatch never submitted without a fired trigger;
                                               #   CORE targets equal achieved beta_dev of the anchor configs;
                                               #   cell labels are the three fixed strings, verbatim;
                                               #   no writes outside this dir and output/p0_major_revision_validation/
```

**Non-negotiable isolation rules:**
- Nothing under `output/paper_v6_preselection_994/`, `output/paper_v12_*`, `paper/img/`, `paper/*.tex`,
  `utils/`, `soft_constrained_models/`, `run_temporal_cv.py` or `scripts/` is modified. New code lives in
  `analysis/p0_major_revision_validation/code/` and **imports** the existing modules.
- Any refit writes to a **new result root** `output/p0_major_revision_validation/` mirroring the standard
  `runs/ predictions/ analysis/ protocol/ logs/` layout, never into a frozen root.
- `tests/test_p0_assertions.py` asserts no P0 process writes outside those two directories.

---

## M. Explicit stop/go gates

### Gate G1 — before the canonical refits (after J1–J3b)
- [ ] P0-1 diagnostics emitted for all 9 training blocks × the full ρ grid; the §E.4 rule returns
      Accept / Reject / Indeterminate **with numbers**; the fixed-λ calculation is labelled
      `stylized_nominal_leaf` (A15); `min_sum_hessian_in_leaf` recorded as checked-and-non-binding (A10).
- [ ] `X` matrix identity hash equal across parity cells (H5 eliminated).
- [ ] **§F.2b source-equivalence audit complete**: executed-path diffs against every provenance commit
      classified; all stored hashes re-verified; `DIRTY_STATE_LIMITATION.md` written; the provenance
      worktree created and removed if it was needed.
- [ ] **Track H frozen-artifact reproduction recorded with an R-tier (R1–R4) and, where not R1, an F.2b
      `failure_class`.** No failure is recorded as "nondeterminism" without ruling out F-SRC and F-DIRTY.
- [ ] Track H and Track P results stored **separately**, each labelled, with the A1 rule recorded in
      `protocol_p0_validation.yaml`.
**Stop if:** Track P **or** Track H shows `B ≢ C` at capacity **T** — a candidate custom-objective defect
that must be root-caused (per-iteration raw-score tracing) before any further fits.

### Gate G2 — before comparator execution (after J4)
- [ ] A **parity tier T1–T4** is assigned per comparison pair, per track, per capacity, per split (§F.4),
      in a table **separate** from the R1–R4 reproduction tiers.
- [ ] The A↔C historical gap is **decomposed and explained**: measured label-representation difference +
      measured capacity amplification across T→M→F, net of the measured F-NUM floor.
- [ ] Reference decision recorded: if Track H gives reproduction at **R1/R2** (or R3 with a named benign
      cause) **and** `B↔C` at **T1/T2**, **Cell B — "Parity-aligned native L2"** may serve as the
      within-implementation **causal reference**, with the justification written down. **Cell A remains
      reported as "Ordinary LightGBM (standard raw-label native)" and Cell B is never printed under that
      name.** Otherwise contrasts fall back to the Cell C within-family ρ=0 origin.
- [ ] **`tables/regeneration_triggers.csv` completed**: RG-1, RG-2, RG-3, RG-4 each marked fired/not-fired
      with an evidence pointer. **J11 is scheduled if and only if at least one RG trigger fired.**
      An A↔B or A↔C result — at any tier, including T4 — **must not** set any RG flag (A11).
- [ ] `f̄_{0,T}`, `Var_T(y)`, `Cov_T(f₀,y)`, `b*_train` emitted for all 9 training blocks;
      `Cov_T(f₀,y) > 0` verified everywhere.
- [ ] P0-3 `zero_control_full.csv` complete on all three regimes for Cell A, Cell B and Cell C.
**Stop if:** `Cov_T(f₀,y) ≤ 0` on any block (the comparator is undefined there), or an R4 reproduction
result has no F.2b classification.

### Gate G3 — before matched-β comparison (after J6)
- [ ] Under the **primary `ȳ_T`-centered map**, `b = 1` reproduces `f₀` **exactly (bitwise)** on every
      evaluation block. *(This is a hard assertion, not a tolerance — it is what makes the map
      theorem-matched.)*
- [ ] `b*_train` and `b*_oof` each drive `Cov_T(e_b, y)` to |value| < 1e-12 on their own defining sample.
- [ ] The b-grid spans the full three-way common support in development β_log with ≥20 % overshoot beyond
      `max(b*_train, b*_oof)`.
- [ ] `configs/b_grid_frozen.json` hashed **before** any held-out/2025 row was scored — verified from the
      manifest and file mtimes.
- [ ] The measured `f̄_{0,T} − ȳ_T` gap is reported and the `center_f0bar` sensitivity is present and
      labelled as such.

### Gate G4 — before **any** manuscript revision
- [ ] All six reports written; all `tables/*.csv` frozen with `output_artifact_hashes.json`.
- [ ] `configs/matched_beta_frozen.json` (targets, `τ`, selected configs, `match_mode`) demonstrably
      hashed before out-of-time evaluation, and **no held-out/2025 metric in any output is interpolated**
      (asserted in `tests/test_p0_assertions.py`).
- [ ] Temporal robustness completed for D-SNAP (screening) and D-PURGE (oracle diagnostic); D-UNSEEN
      reported; **Gates G5a and G5b resolved**, with `tables/dsnap_refinement.csv` present whether or not
      a trigger fired.
- [ ] Each of the five §J.5 revision triggers evaluated with an explicit yes/no and numbers, and — where
      one fired on the screening grid — an explicit statement of whether it survived G5a refinement.
- [ ] D-PURGE labelled `design_type = oracle_diagnostic` everywhere, with the "not implementable
      prospectively" statement in the report's opening paragraph.
- [ ] Smearing factors demonstrably estimated from development out-of-fold residuals only (A6).
- [ ] `provenance/worktree_diff.patch` present.
- [ ] `MANUSCRIPT_IMPACT_MEMO.md` written, stating per prior conclusion: survives / fails / remains
      experiment-dependent.
**Then and only then** may Results, Discussion, Abstract, Conclusion, title or contribution statements be
touched.

### Gate G5a — D-SNAP coarse-grid alert (after J8)
- [ ] Each of the five §J.5 triggers evaluated on the 21-point **screening** grid.
- [ ] **If none fires:** the primary design stands; D-SNAP is reported as strict-date robustness; record a
      no-trigger `dsnap_refinement.csv` and skip to Gate G4.
- [ ] **If any fires:** identify the affected ρ region, build `configs/dsnap_refinement_grid.json` from the
      skipped original-grid ρ values plus one screening step either side, and run **J8b**.
- [ ] **Direct-promotion exception:** a very large qualitative reversal (β_log sign change; Direct/Surrogate
      ordering reversal across the whole screening grid; dCor monotone across all 21 screening points) may
      go straight to G5b, with the exception invoked explicitly and recorded.

### Gate G5b — confirmed material change → D-SNAP promotion (after J8b)
- [ ] The trigger is re-evaluated on the **refined** path.
- [ ] **If it does not survive refinement:** it was a screening-grid artifact. The primary design stands;
      record the refinement result and proceed. **No regeneration.**
- [ ] **If it survives:** D-SNAP is promoted to the primary temporal design and **J12 (full 82-point
      regeneration under the D-SNAP protocol) becomes mandatory** before any manuscript claim rests on the
      temporal design.
- [ ] Minor candidate-region, Δ_NL or dCor differences may **never** promote on the screening grid alone.

---

## N. Risks and unresolved questions

**Genuinely unresolvable by repository inspection:**

1. **Which reproduction tier (R1–R4) the frozen artifacts achieve under their own historical settings, and
   which failure class applies if they miss R1.** This is the single most consequential unknown, because
   everything reused from the frozen path table depends on it, and no pinned experiment can answer it (A1).
1b. **Whether the historical source state can be reconstructed at all.** The generating trees were dirty
   and their diffs were never archived (only `git_diff_sha256`). If the executed path differs between HEAD
   and the provenance commits, a provenance worktree narrows the question but cannot eliminate
   **F-DIRTY** uncertainty. The plan's answer is to name that limit explicitly
   (`provenance/DIRTY_STATE_LIMITATION.md`) rather than to disguise it as nondeterminism.
2. **Whether `B ≡ C` holds at capacity F in either track.** Everything points to yes (the derivatives are
   algebraically identical and Direct ≡ Surrogate already agrees bitwise), but it must be run.
3. **Whether `Cov_T(f₀, y) > 0` on every training block** — near-certain, but `b*_train` is undefined
   otherwise, and no in-sample training predictions exist to check without the refits.
4. **How far `f̄_{0,T}` sits from `ȳ_T`.** Determines whether the `center_f0bar` sensitivity is material.
5. **Whether `b*_train` for a near-interpolating 994-tree ensemble is so close to 1 as to be
   scientifically uninformative** — in which case `b*_oof` is the reportable endpoint and `b*_train`
   becomes a remark about in-sample overfitting.
6. **Whether D-PURGE materially shifts the candidate-region endpoints.** Repeat-parcel exposure of
   27–34 % makes the outcome genuinely uncertain. (Note this is an *oracle* diagnostic: a shift bounds the
   contribution of parcel history, it does not indicate a better deployable split.)
7. **How many Surrogate targeted fits the `τ = 0.002` tolerance will require.** Bounded above by 6
   configs × 9 evaluations, so the risk is cost-bounded, not scientific.
8. **Whether any D-SNAP screening trigger survives G5a refinement.** A 4×-coarser grid can manufacture
   apparent turning-point shifts, so the screening result alone is not informative about promotion.

**Risks to manage during execution:**

- **The pyarrow pushdown trap (§C-11).** Loading the parquet any other way silently changes the sample by
  3 rows and invalidates every fold hash. Mitigation: load only via `run_temporal_cv._load_and_split_data`
  and keep the 344,607 / 38,290 / 26,641 assertions.
- **Track cross-contamination.** Running a pinned cell and citing it against historical artifacts is the
  precise error A1 forbids. Mitigation: `track` is a required column in every parity table; the report
  states the rule verbatim; `submit_p0.sh` runs Track H to completion before Track P.
- **Tier confusion.** Mixing the A/B/C parity tiers (T1–T4) with the artifact-reproduction tiers (R1–R4)
  would let a parity result masquerade as a reproduction result. Mitigation: they live in different tables
  with different column names (`parity_tier`, `reproduction_tier`), and a test asserts neither name
  appears in the other table.
- **Over-triggering regeneration.** The natural failure mode is to let the large, expected A↔C gap
  cascade into a ~62 core-hour rerun. Mitigation: §F.6 enumerates the only four triggers;
  `tables/regeneration_triggers.csv` must show a fired trigger before `10_full_path_regen.sbatch` may be
  submitted, and `submit_p0.sh` contains no unconditional submission of it.
- **Promoting a grid artifact.** A screening-grid trigger promoted straight to a full D-SNAP regeneration
  would spend 62 core-hours on a resolution artifact. Mitigation: G5a refinement is mandatory except for
  the three enumerated large qualitative reversals.
- **Reading the stylized leaf diagnostic as a measurement.** Mitigation: `diagnostic_type =
  stylized_nominal_leaf` on every such row, and an explicit prohibition on the corresponding claim until
  trained-leaf Hessian sums are measured (J13a).
- **Accidental mutation of frozen artifacts.** Mitigation: the isolation rules in §L plus a write-scope
  test.
- **Leakage through the matching step.** Mitigation: grid- and matching-hash freezing before out-of-time
  scoring, asserted in code.
- **Silent metric interpolation.** Mitigation: `match_mode` is a required column; a test asserts no
  held-out/2025 value carries `match_mode = interpolated` (that value is not permitted to exist).
- **`Δ_NL` estimator drift.** Mitigation: pin `estimator_spec_hash = e85069150b509a3518eeb2abff02b91d589bc13fd4c1f265da8978c9798c5243`
  and refuse to run if it changes.
- **dCor estimator variant still unrecorded** (manuscript `\todo` L3204). The code uses
  `dcor.distance_correlation(..., method="auto")` from `dcor 0.6`. P0 records this explicitly, closing
  P1-9 at zero cost.
- **Direct never reaching β_log ≈ 0** makes some comparisons structurally one-sided. Mitigation: the
  `attained = false` convention in §I.3-7.

---

## O. Final recommended execution sequence

| # | Step | Inputs | Code to reuse / modify | Outputs | Compute | Depends on | Acceptance criterion |
|---|---|---|---|---|---|---|---|
| **1** | Scaffold the isolated analysis folder; freeze provenance including the full diff text | HEAD `b878b008`, env versions, `V6/lgbm_config.json` | new `code/p0_common.py`; pattern from `analysis/external_jurisdiction_benchmark_v1/` | `README.md`, `protocol_p0_validation.yaml` (incl. the A1 rule and tiered criterion), `configs/frozen_lgbm_994.json`, `provenance/PREFLIGHT.json`, `provenance/worktree_diff.patch` | none | — | `lgbm_params_sha256` recomputes to `8f0f2acd…585b`; diff text archived |
| **2** | **P0-1** objective / curvature / weight / fixed-λ audit | one column of `training_data.parquet` + rebuilt fold indices; `boosting_models.py` | new `code/p0_1_objective_audit.py`; reuse `run_temporal_cv._load_and_split_data`, `build_rolling_origin_protocol`, `tests/test_canonical_objectives.py` | `tables/{objective_scaling_audit, direct_hessian_magnitudes, surrogate_weight_distribution, effective_leaf_shrinkage, label_quantization}.csv`; `reports/P0_IMPLEMENTATION_AUDIT.md` | local, <5 min | 1 | Fold index hashes match the archive; §E.4 returns a definite verdict; `min_sum_hessian_in_leaf` recorded non-binding; manuscript-vs-code equation check emitted |
| **3** | **P0-6a** boundary + repeat-PIN exposure audit | same load + `meta_pin` | new `code/p0_6_temporal.py` (audit mode) | `tables/{boundary_exposure_audit, repeat_pin_exposure}.csv` | local, <5 min | 1 | Reproduces the measured counts (dev/held-out 43 / 105 on 2023-11-09); split counts assert 344,607/38,290/26,641 |
| **3b** | **§F.2b source-equivalence / provenance audit** — diff HEAD against each recorded provenance commit over the executed training/prediction path; classify every hunk `touches_executed_path`; re-verify all stored code/config hashes; create a temporary read-only Git worktree at the provenance commit **only if** a hunk touches that path; write the dirty-state limitation | `git`, provenance commits `508dc1c2` / `2aa0346a` / `d3ef45f2`, all stored hashes | new `code/p0_2_source_equivalence.py` | `tables/source_equivalence_audit.csv`, `provenance/provenance_commit_diffs/`, `provenance/DIRTY_STATE_LIMITATION.md` | local, ~10 min | 1 | Every executed-path file classified; every re-verifiable hash re-verified; worktree path + commit recorded in `PREFLIGHT.json` and the worktree removed afterwards |
| **4** | **P0-2 Track H** — frozen-artifact reproduction (Cell A native, Direct/Surrogate ρ=0, ρ≈0.954, ρ=100 on both OOS blocks; at HEAD and, if step 3b requires it, in the provenance worktree) **and** the A/B/C × T/M/F ladder **and** the same-host Cell-A replicate, all under **historical settings** | dev pool + production block; cached predictions for comparison | new `code/p0_2_parity_tracks.py` (track=historical); sbatch from `run_paper_v6_preselection_994_baseline_report.sbatch` | `tables/frozen_artifact_reproduction.csv` (with `reproduction_tier` and `failure_class`), `tables/parity_ladder.csv` (`track=historical`, `parity_tier`), `figures/parity_capacity_ladder.pdf` | Slurm, ~34 fits (+12 if the worktree re-test runs), ~1.5–2 core-h, ~15–20 min wall | 2, 3b | Every config assigned an **R1–R4** tier and, where not R1, an F.2b `failure_class`; **no failure recorded as "nondeterminism" without ruling out F-SRC and F-DIRTY**; parity tiers assigned per pair in a separate table |
| **5** | **P0-2 Track P** — the same A/B/C × T/M/F ladder with `deterministic/force_row_wise/num_threads` pinned | same blocks | `code/p0_2_parity_tracks.py` (track=pinned) | `tables/parity_ladder.csv` (`track=pinned`) | Slurm, ~18 fits, ~0.7 core-h, ~10 min wall | 4 | `B ≡ C` at capacity T; result recorded **as an implementation statement only**, never cited against historical artifacts, and never able to set an RG flag |
| **6** | **GATE G1** | steps 2, 3b, 4, 5 | — | gate record in `protocol_p0_validation.yaml` | none | 5 | §M G1 checklist |
| **7** | **9 native refits at the canonical config, historical settings, in-sample prediction retained** (7 fold-training blocks, dev pool, production block); plus Direct ρ=0 / Surrogate ρ=0 confirmations at capacity F | frozen 994 params; verified index sets | `code/p0_2_parity_tracks.py` (F mode) + `code/p0_4_centered_spread.py` (b* mode); `slurm/03_native_refits.sbatch` | `tables/parity_prediction_deltas.parquet`, `tables/b_star_diagnostics.csv`; in-sample predictions under `output/p0_major_revision_validation/` | Slurm, ~11 fits, ~0.5 core-h, ~10 min wall | 6 | `Cov_T(f₀,y) > 0` on every block; parity tier assigned for B↔C at F |
| **8** | **GATE G2** — parity (T1–T4) and reproduction (R1–R4) verdicts, reference decision, and the **§F.6 regeneration-trigger table** | step 7 | — | `reports/RHO_ZERO_PARITY_REPORT.md` (states the A1 rule verbatim; T- and R-tiers in separate tables), `tables/regeneration_triggers.csv`; gate record | none | 7 | §M G2 checklist. **Step 16 is scheduled if and only if RG-1, RG-2, RG-3 or RG-4 fired.** An A↔B or A↔C result at any tier — including T4 — must not set any RG flag |
| **9** | **P0-3** complete zero-control table (+ 28 missing `Δ_NL` CV evaluations) for **both** reference cells | frozen path table; cached fold predictions; step 7 | new `code/p0_3_zero_control.py`; reuse `utils/delta_nl.py`, `utils/motivation_utils.py` | `tables/zero_control_full.csv`, `reports/ZERO_CONTROL_REPORT.md` | local, ~0.3 core-h | 8 | Every metric present for {Cell A, Cell B, Direct ρ=0, Surrogate ρ=0} × {7 folds, CV mean/sd, held-out, 2025}; reproduces the frozen table where they overlap |
| **10** | **P0-4** centered-spread comparator: build and **hash-freeze** the 121-point b-grid, then evaluate the **primary `ȳ_T`-centered** map and the labelled `f̄_{0,T}` sensitivity on 7 folds + held-out + 2025 | cached predictions (all regimes); step 7 `b*` diagnostics | extend `scripts/final_local_results_994.py::{load_native_oof, reconstruct_fold_training_means, solve_b_star_validation_neutral, centered_map, run_recalibration}` into `code/p0_4_centered_spread.py` | `configs/b_grid_frozen.json`, `tables/{centered_spread_path, centered_spread_ratio_profiles}.csv`, `figures/centered_spread_vs_rho_paths.pdf`, `reports/CENTERED_SPREAD_COMPARATOR_REPORT.md` | Slurm array 0-8, ~4 core-h, ~30 min wall | 8 | `b=1` reproduces `f₀` **bitwise** under the primary map; grid hash written before any OOS row scored; existing `recalibration_path.csv` reproduced at its 51 shared b-values |
| **11** | **GATE G3** | step 10 | — | gate record | none | 10 | §M G3 checklist |
| **12** | **P0-5** matched-β: derive common support, generate the six CORE targets by the **deterministic algorithm in §I.3-2** from actual fitted Direct configs, freeze `τ` and the selection table, then select **nearest fitted** configs (exact solve for post-hoc) and run targeted fits only where `τ` cannot be met | `combined_path_table_v4_analysis_view.csv`; step 10 path; step 9 origin | new `code/p0_5_matched_beta.py`; `slurm/05_matched_beta_targeted_fits.sbatch` | `configs/matched_beta_frozen.json` (probe values `q_j`, anchor configs `c_j`, achieved targets, residuals, `τ`), `tables/{matched_beta_comparison, matched_beta_crossings}.csv`, three `figures/matched_beta_*.pdf`, `reports/MATCHED_BETA_REPORT.md` | local + Slurm, 0–54 fits, ~2.5 core-h, ~20 min wall | 11 | Targets produced by the algorithm, with `q_j`, `c_j` and residuals recorded so no selection discretion is possible; **every held-out/2025 number is a measured value of an actually fitted configuration**; `match_mode` and achieved β_dev on every row; Direct-unattained targets flagged, never dropped; freeze hash precedes OOS reads |
| **13** | **D-SNAP** strict-date robustness **screening** run: snap-back protocol + 21-point screening grid | verified index sets; frozen 994 params | `code/p0_6_temporal.py`; sbatch from the CV + OOS-array pair | `configs/split_protocol_dsnap.json`, `tables/robustness_path_dsnap.csv` | Slurm, 387 fits, ~16 core-h, ~1 h wall | 3 (parallel with 7–12) | `max(train date) < min(val date)` asserted at all 8 boundaries; ≤0.5 % of any block moved; results labelled `grid = screening` |
| **14** | **D-PURGE** repeat-parcel **oracle diagnostic**: evaluation sets preserved exactly; purge from each training block every PIN present in the corresponding evaluation block; 21-point screening grid. Plus **D-UNSEEN** (zero fits) from cached predictions | D-SNAP-independent; frozen indices + `meta_pin` | same | `configs/split_protocol_dpurge.json`, `configs/unseen_subset_definition.json`, `tables/{robustness_path_dpurge, robustness_unseen_subset}.csv` | Slurm 387 fits ~15 core-h ~1 h wall; D-UNSEEN local ~0.3 core-h | 3 | **Evaluation sets bitwise identical to the frozen design** (asserted); zero training PIN appears in the paired evaluation block; retained training-n per block; every row carries `design_type = oracle_diagnostic` and the report opens with the "not implementable prospectively" statement |
| **15** | **GATE G5a** — evaluate the five §J.5 triggers on the screening grid | steps 13, 14, frozen path table | `code/p0_6_temporal.py` (compare mode) | `tables/robustness_vs_frozen_deltas.csv`, `configs/dsnap_refinement_grid.json` (if a trigger fires) | local, ~0.2 core-h | 13, 14 | Each trigger answered yes/no with numbers; if any fires, the affected ρ region is identified and the refinement grid built; if none fires, a no-trigger `dsnap_refinement.csv` is written |
| **15b** | *(CONDITIONAL — Gate G5a alert)* **targeted local refinement**: refit the skipped original-grid ρ values around each affected region under D-SNAP, then re-evaluate the trigger | `configs/dsnap_refinement_grid.json` | `slurm/09_dsnap_refinement.sbatch` (CV + OOS-array pair with an explicit `--rho-values` list) | `tables/dsnap_refinement.csv` | Slurm, ~110–220 fits, ~5–9 core-h, ~30–40 min wall | 15 | The trigger is re-evaluated on the refined path and recorded as *survived* or *screening-grid artifact* |
| **15c** | **GATE G5b** + temporal report | steps 15, 15b | `code/p0_6_temporal.py` | `reports/TEMPORAL_ROBUSTNESS_REPORT.md` | local, ~0.1 core-h | 15b | Promotion decision recorded; promotion only if a trigger survived refinement, or a very large qualitative reversal was invoked explicitly; if promoted, step 17 is scheduled |
| **16** | *(CONTINGENT — only if §F.6 **RG-1, RG-2, RG-3 or RG-4** fired)* full path regeneration under pinned settings | frozen 82-point grid; corrected implementation | `run_temporal_cv.py --stage {cv,test,forward}` with a new result root | regenerated `combined_path_table` under `output/p0_major_revision_validation/` | Slurm, ~1,510 fits, ~62 core-h, ~4 h wall | 8 + a fired RG trigger | `tables/regeneration_triggers.csv` shows the firing trigger and its evidence; regenerated table either reproduces the frozen one within its assigned R-tier or the differences are fully characterised |
| **17** | *(CONTINGENT — only after **Gate G5b** promotion)* full path regeneration under the **D-SNAP** protocol | D-SNAP split protocol; frozen 82-point grid | same | D-SNAP `combined_path_table` under `output/p0_major_revision_validation/` | Slurm, ~1,510 fits, ~62 core-h, ~4 h wall | 15c | The paper's temporal design and its reported path come from the same protocol |
| **18** | Inferential extras from cached predictions: PRB SE/t; VEI Significance (ED2 App. E Steps 6–7); **development-only** smearing sensitivity | cached predictions (83 ρ × 2 families × 2 OOS splits) | new `code/p0_7_inferential_extras.py`; reuse `utils/motivation_utils.py::{prb, vei, vei_percentile_group_profile}` | `tables/{prb_inference, vei_significance, smearing_sensitivity}.csv`; a section in `reports/ZERO_CONTROL_REPORT.md` | Slurm, ~2 core-h, ~15 min wall | 9 | Closes audit P1-2, P1-3 and H-8 with zero refits; smearing residual source asserted to be a development artifact; H-16 (subgroup) stays open and is stated as unmeasured |
| **18b** | *(OPTIONAL — §E.3-7 upgrade)* measure **actual trained-leaf Hessian sums** via `pred_leaf=True` on the training block, to replace the stylized fixed-λ diagnostic with a measured one | fitted models from steps 7/13/14 | `code/p0_1_objective_audit.py` (leaf mode); `slurm/12_trained_leaf_hessians.sbatch` | `tables/trained_leaf_hessians.csv` | Slurm, ~1 core-h, ~10 min wall | 7 | Only after this may the report drop the `stylized_nominal_leaf` label or claim the Surrogate's ρ effect includes implicit de-regularization |
| **19** | Freeze: hash every input and output; write the manifest and artifact index | all | `code/p0_freeze.py`; reuse `canonical_experiment.py::{git_state, package_versions, lgbm_params_hash}` | `provenance/{POSTFLIGHT, input_artifact_hashes, output_artifact_hashes, slurm_graph}.json` | local, ~20 min | 12, 15c, 18 | Every table referenced by a report appears in the hash index |
| **20** | **GATE G4** | all | — | gate record | none | 19 | §M G4 checklist |
| **21** | `MANUSCRIPT_IMPACT_MEMO.md` — which prior conclusions survive, which fail, what must change, what stays experiment-dependent | all reports | — | `reports/MANUSCRIPT_IMPACT_MEMO.md` | local | 20 | Maps each finding to the §P sections; **proposes no prose** |

### P. Manuscript sections that will depend on each P0 result (mapping only — no edits this pass)

| P0 result | Manuscript locations that depend on it |
|---|---|
| P0-1 curvature verdict | §3.2 L1509; App. E L3626–3650 and `\todo` L3681; `subsec:comparators` L2041; Abstract L123. The fixed-λ leaf-shrinkage material may enter only as a **stylized** remark unless step 18b runs |
| P0-2 parity (Track H) | `tab:rho_zero_control` L3659–3676 and `\todo` L3679; `subsec:zero_control_results` L3653; Results opening L2092; `tab:path_anchor_summary` L2671 and `tab:path_anchor_complementary` L3890; Discussion L2978–3002; Limitations L3061 |
| P0-2 parity (Track P) | App. E implementation description only — **may not be used to support any Results claim about the frozen paths** |
| P0-2 reproduction tier + source-equivalence | App. E reproducibility appendix and `\todo` L3681 (which asks for the executed code/config hash and the archived diff); Limitations L3061 if the class is **F-DIRTY**, which must be disclosed as an irreducible provenance limitation |
| P0-3 zero control | `tab:rho_zero_control`; every "improves on Ordinary LightGBM" sentence in §5.2 (esp. L2750) |
| P0-4 comparator | `subsec:comparators` L2031–2032 and `\todo` L2046; `\todo` L2969; App. D L3381 and `\todo` L3382; Future Work L3071; Limitations L3061; new Results subsection; Related Work L277–287 |
| P0-5 matched-β | Discussion L3002; contribution ¶ L204; Abstract; title (P1-14) |
| P0-6 temporal | §4.1 L1890 (same-date disclosure); §4.2 L1947–1955; App. F `tab:fold_structure` L4202; Limitations L3061; `tab:rho_candidate_regions` L2759 and prose L2756. D-PURGE must be described as an **oracle diagnostic**, never as a recommended split |
| P1 extras | `tab:assessment_metrics_summary` PRB/VEI rows and notes; §2.3 VEI L626–639; §2.2 L358 and Limitations L3057 (smearing) |
| Nesting/CV wording (C-9, A9) | §4.2 L1968–1971; App. F; P1-1 — reframe as variation across **nested** chronological windows, not independent replications; do **not** claim later folds dominate the equal-weight mean |
| (side effects, zero cost) | `\todo` L1966 — the frozen parameter vector and hash already exist in `V6/lgbm_config.json`; `\todo` L3204 — dCor is `dcor 0.6 distance_correlation(method="auto")`; L3758 names the superseded 50-ρ `combined_path_table.csv` |

Bibliography and document-integrity work (audit §I, §J: the six missing references, duplicate/stale
BibTeX, 19 rendering `\todo`s, struck `\oldtext` regions, six-county promises, the MKI band) is
**experiment-independent** and should proceed on a **separate branch and separate commits**, never mixed
into the experiment commits.

---

PLAN FINALIZED — READY FOR EXECUTION
