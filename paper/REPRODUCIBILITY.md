# Reproducibility guide — soft vertical-equity constrained mass appraisal (CCAO paper)

This guide belongs to one paper and one manuscript source:

| | |
|---|---|
| Manuscript source | `paper/paper_v17_option1.tex` |
| Compiled article | `paper/paper_v17_option1.pdf` |
| Frozen empirical tag | `tier-b-final-20260909` (commit `a89e7469b0244efc10f7c2ed4cab417cead78e8b`) |
| Frozen evidence tags | `p0-major-revision-final-20260907`, `p1-inferential-reporting-final-20260907`, `tier-b0-final-20260907` |

**The repository root `README.md` is not the replication protocol for this paper.** It documents a
broader, older development workflow — two baselines including linear regression, a 2024 assessment
split, convex stacking, block-bootstrap uncertainty — none of which is the frozen design reported
here. In particular, do not treat `quick_test_models.py` as the reproduction recipe: it is a
development smoke test and it does not reproduce this paper. This file is the paper-specific entry
point.

---

## 1. What this guide does and does not claim

It claims three things, separately, and does not merge them into a single word:

* **Level A — auditability.** Every number printed in the paper can be re-derived from tracked
  artifacts, mechanically, without refitting anything. **Supported.**
* **Level B — reconstruction from frozen derived artifacts.** The derived tables behind every
  reported comparison are tracked and can be re-examined without the raw extract. **Supported, with
  one exclusion: the per-configuration prediction arrays are not released.**
* **Level C — full end-to-end rerun.** Refitting the models requires the restricted CCAO extract
  and the executed run roots. **Not publicly reproducible.**

It does **not** claim: that this repository is "fully reproducible" end to end; that an external
reader can obtain the identical CCAO extract; that a single command regenerates the paper's
empirical results; that the recorded environment is a lockfile; that the exact source state that
generated the frozen artifacts can be reconstructed; or that a DOI, Zenodo deposit or release
archive exists for this work. None of those is true.

---

## 2. The restricted input

Every sample reported in the paper is constructed from a single file:

```
path   data/CCAO/2025/training_data.parquet
bytes  215,400,916
sha256 b1fc00b514041af5aa7135d85ed59a028afe40569bba1014c4dcf3ad2f3a7b51
```

recorded in `analysis/p0_major_revision_validation/provenance/PREFLIGHT.json`
(`input_artifact_hashes.canonical_data_parquet`).

* It is **not redistributed** and is **not tracked** in this repository.
* **No public availability is claimed for it**, and no download location is asserted.
* **No build or retrieval date was recorded** when the extract was taken, so none is asserted. The
  hash fixes *which* extract was used, not *when* it was produced, and it does not entitle another
  party to obtain a byte-identical copy.
* No dataset version string and no license claim are asserted here.

The executed run roots that consumed it — `output/paper_v6_preselection_994/` and
`output/paper_v12_lower_rho_extension_994_v2/` — are likewise absent from this repository. They hold
the stored per-configuration predictions and the combined path table. They are recorded by path,
byte length and content hash in the frozen evidence manifest (see §5), and nowhere else.

**Per-record prediction artifacts are not released.** They are row-level outputs derived from
restricted assessor data, and no promise of public release is made for them anywhere in the paper.

---

## 3. Level A — auditing every number in the paper

The binding artifact is the Tier-B numeric ledger:

```
paper/paper_analysis/tier_b_validation/ledger/tier_b_numeric_ledger.yaml
```

683 entries covering 32 distinct tracked artifacts. Each entry names the artifact path, its sha256,
a selector that locates the value inside it, the raw value byte-exactly as it appears, the transform
applied, the rounding rule, and the string the manuscript prints.

To recompute all of them and check the rest of the manuscript contract:

```bash
cd paper/paper_analysis/tier_b_validation
python validate.py --stage B4.3 --compile --json runs/B4.3.json
```

The pass criterion is `UNEXPECTED_VALIDATOR_FAILURES == 0`. Check C01 reopens each artifact,
verifies its hash against both the file and the frozen manifest, re-executes the selector, re-extracts
the raw value, applies the transform and rounding with `decimal.Decimal` and `ROUND_HALF_UP`, and
compares with the `.tex`. Checks C02–C13 cover numeric-token support, reference-cell and
development-coordinate semantics, `NOT_ATTAINED` preservation, Exposure-Draft status language,
forbidden and required wording, visual provenance, figure existence and git tracking, label and
citation integrity, TODO closure, and math-mode claims. The same run re-verifies that the three
frozen subtrees are byte-identical to their tags.

`--compile` additionally runs `latexmk -pdf -interaction=nonstopmode -file-line-error` on
`paper/paper_v17_option1.tex` and reports undefined references, undefined citations, missing
bibliography entries and LaTeX errors, which a zero exit code alone does not.

The executed metric code is tracked and is byte-identical at every provenance commit
(`analysis/p0_major_revision_validation/tables/source_equivalence_verdict.json`,
`executed_path_drift_resolution.worktree_evidence`):

| what | tracked path |
|---|---|
| ratio-study measures (PRD, PRB, COD, MKI, VEI), dCor call site | `utils/motivation_utils.py` |
| nonlinearity gap `Delta_NL` | `utils/delta_nl.py` |
| custom boosting objectives | `soft_constrained_models/boosting_models.py` |

---

## 4. Level B — reconstruction from frozen derived artifacts

All paths below are tracked and frozen under their tags. This is where the paper's comparisons come
from; none of it requires the raw extract.

**Matched first-order comparison** (`analysis/p0_major_revision_validation/tables/`)
`matched_beta_comparison.csv`, `matched_beta_ext_targets.csv`, `matched_beta_pairwise_deltas.csv`,
`matched_beta_crossings.csv`, `matched_beta_origin_identity.csv`, `matched_beta_ratio_profiles.csv`,
and the `_d3` twin of each; `matched_beta_d3_sensitivity.csv`,
`matched_beta_d3_sensitivity_summary.json`, `matched_beta_d3_sign_agreement.csv`. Design frozen in
`configs/matched_beta_frozen.json` and `configs/matched_beta_d3_frozen.json` with their `_hash`
files. Report: `reports/MATCHED_BETA_REPORT.md`.

**Post-hoc centered-spread comparator** `centered_spread_path.csv` (plus the per-cell, per-fold,
`heldout`, `forward_2025` and `pooled_oof` shards), `centered_spread_cv_summary.csv`,
`centered_spread_b1_qc.csv`, `centered_spread_linearity.csv`, `centered_spread_root_verification.csv`,
`centered_spread_centering_sensitivity.csv`, `centered_spread_ratio_profiles.csv`,
`posthoc_development_roots.csv`, `b_star_diagnostics.csv`. Convention frozen in
`configs/posthoc_comparator_convention.yaml` and `configs/post_g1_reference_convention.yaml`.
Report: `reports/CENTERED_SPREAD_COMPARATOR_REPORT.md`.

**Zero-penalty control and native/custom parity** `zero_control_full.csv`,
`zero_control_frozen_crosscheck.csv`, `zero_reference_fits.csv` (+ `block0`–`block8`),
`zero_reference_reproduction_qc.csv`, `parity_ladder_pinned.csv`, `parity_ladder_historical.csv`,
`fnum_same_host_replicate.csv`, `label_quantization.csv`, `effective_leaf_shrinkage.csv`,
`direct_hessian_magnitudes.csv`, `objective_scaling_audit.csv`. Reports:
`reports/RHO_ZERO_PARITY_REPORT.md`, `reports/ZERO_CONTROL_REPORT.md`,
`reports/P0_IMPLEMENTATION_AUDIT.md`.

**Temporal design and robustness** `robustness_path_dsnap.csv`, `robustness_path_dpurge.csv` and
their per-fold shards, `robustness_vs_frozen_deltas.csv`, `robustness_unseen_subset.csv`,
`dsnap_refinement.csv`, `dsnap_boundary_audit.csv`, `dpurge_purge_audit.csv`,
`cv_fold_validation_overlap_audit.csv` / `_summary.json`,
`temporal_validation_overlap_audit.csv` / `_summary.json`,
`development_beta_coordinate_audit.csv` / `_summary.json`. Splits frozen in
`configs/split_protocol_dsnap.json` and `configs/split_protocol_dpurge.json`; grid in
`configs/robustness_rho_grid.json`; configuration and CV run maps in `configs/frozen_config_map.csv`
and `configs/frozen_cv_run_map.csv`. Report: `reports/TEMPORAL_ROBUSTNESS_REPORT.md`.

**Reproduction audit of the frozen fits** `frozen_artifact_reproduction.csv` — 14 refits (7 labels ×
{held-out, 2025 forward}) compared against the cached predictions, all at reproduction tier `R1`.
This audit itself needed the restricted input and the executed run roots; it is reported as a frozen
result, not as something a public reader can repeat.

**Inferential reporting** (`analysis/p1_inferential_reporting/tables/`) `prb_inference.csv` /
`prb_inference_summary.json`, `vei_significance.csv` / `vei_significance_summary.json`,
`smearing_factor_provenance.csv`, `smearing_apply_invariance.csv`,
`smearing_apply_delta_nl_subset.csv`, `smearing_apply_ed2_stability.csv`,
`smearing_apply_fastpath_reconciliation.csv`, `smearing_apply_summary.json`,
`dcor_estimator_facts.json`, `dcor_estimator_audit.csv`. Frozen configs:
`configs/display_set_frozen.json`, `configs/smearing_estimator_frozen.json`,
`configs/ed2_vei_procedure.json` (each with its `_hash` file).

**Figures** `analysis/p0_major_revision_validation/figures/*.pdf` and the manuscript figure tree
under `paper/img/`, whose existence and git tracking check C09 verifies on every run.

**Not at this level:** the combined path table
(`…/transition_regions_paper_assets_v4_delta_nl_bends/tables/combined_path_table_v4_analysis_view.csv`,
640,795 bytes) and the per-configuration prediction parquets. Hash-identified only; see §2 and §5.

---

## 5. Machine-readable inventory

No new manifest is introduced by this guide. The canonical machine-readable inventory already exists
and is tracked:

```
analysis/final_manuscript_evidence/FINAL_EVIDENCE_MANIFEST.json
```

370 artifacts. For each: `role`, `scientific_stage`, `bytes`, `sha256`, `source_commit_or_tag`, and
`present_in_worktree`. 359 are present with a **recomputed** hash; 11 are absent with a hash
**quoted** from a frozen index (`hash_source`), and absent files are never read. Those 11 are exactly
the restricted and unreleased inputs: the CCAO parquet, the combined path table, both experiment
specs, `lgbm_config.json`, `folds.json`, `delta_nl_estimator.json`, `recalibration_path.csv`,
`recalibration_spec.json`, and `rho0_split_audit.{csv,json}`.

The manifest also records `excluded_streams` — the external-jurisdiction and ATTOM benchmark trees —
so that the exclusion is explicit. No external-benchmark artifact backs any claim in this paper.

Companion documents: `FINAL_EVIDENCE_INDEX.md`, `SELECTOR_GRAMMAR.md`,
`certification/CERTIFICATION.md`, `manuscript_numeric_map.csv`, `manuscript_claim_map.csv`.

---

## 6. Environment

Recorded independently and identically in
`analysis/p0_major_revision_validation/provenance/PREFLIGHT.json` (`.environment`) and
`…/POSTFLIGHT.json` (`.environment`), and again in
`analysis/p0_major_revision_validation/tables/source_equivalence_verdict.json`
(`.provenance.versions`):

| package | executed version | status |
|---|---|---|
| Python | 3.9.19 | exact executed version verified |
| LightGBM | 4.6.0 | exact executed version verified |
| numpy | 1.26.4 | exact executed version verified |
| pandas | 2.3.1 | exact executed version verified |
| scipy | 1.13.1 | exact executed version verified |
| scikit-learn | 1.6.1 | exact executed version verified |
| dcor | 0.6 | exact executed version verified |
| pyarrow | 14.0.1 | exact executed version verified |

Host `sloan-login001.mit.edu`; conda environment `fairness_env`. Threading at the P0 stage:
`OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`.

**The root `requirements.txt` is not this environment.** It states lower bounds
(`numpy>=1.23`, `lightgbm>=4.0`, `scikit-learn>=1.2`, `scipy>=1.9`, `pandas>=1.5`, `dcor>=0.6,<0.7`)
and is a development specification, not a lockfile. It is not merely looser than the executed
environment; for at least one package it disagrees with it — it requires `pyarrow>=16.0` while the
executed environment ran pyarrow 14.0.1. Installing from `requirements.txt` therefore does **not**
reconstruct the executed environment. No lockfile for it exists in this repository.

---

## 7. Source-state provenance, and its irreducible limit

Three commits generated the frozen artifacts:

| commit | role |
|---|---|
| `508dc1c2` | 994-tree baseline/config + v6 experiment spec |
| `2aa0346a` | lower-rho extension experiment spec |
| `d3ef45f2` | final local results: rho=0 split audit, recalibration path, `Delta_NL` |

What is established (`tables/source_equivalence_audit.csv`, `tables/source_equivalence_verdict.json`):

* all three are ancestors of the frozen provenance head;
* `boosting_models.py`, `motivation_utils.py`, `canonical_experiment.py`, `params.yaml`,
  `cv_config.yaml` and `model_params.yaml` are **byte-identical at every one of them**;
* the two executed-path files that did drift — `run_temporal_cv.py` and `utils/delta_nl.py` — have
  per-file patches archived under `provenance/provenance_commit_diffs/`, and both drifts are
  adjudicated behaviour-neutral for prediction reproduction (the first is an additive flag that
  defaults off and is bitwise identical to the prior function on the exact frozen grids; the second
  consumes predictions and takes no part in training or prediction);
* the executed LightGBM parameter vector is archived in full in `PREFLIGHT.json` under
  `frozen_lgbm.lgbm_params`, with configuration id `407d47775760c14d` and parameter-vector sha256
  `8f0f2acd83118de782604b5ca7143acfbd2af3fd186ea9376588f9bcf560585b`, re-verified against the
  executed configuration file and both experiment specifications.

What is **not** established, and is disclosed rather than resolved
(`provenance/DIRTY_STATE_LIMITATION.md`): all three generating trees were **dirty**. A diff hash was
recorded for two of them and none for the third, and **the diff text was archived for none**. The
uncommitted state therefore cannot be reconstructed. `provenance/worktree_diff.patch` is the
worktree diff of the P0 validation pass itself, not of the original generating runs. Checking out a
provenance commit reproduces the committed state at that point, not the state that actually ran.
The frozen record classifies the residual as `F-DIRTY` — unreconstructable dirty-state uncertainty —
and treats it as a disclosure item, not a regeneration trigger.

---

## 8. Distance-correlation estimator

Executed call, recorded in `analysis/p1_inferential_reporting/tables/dcor_estimator_facts.json`:

```python
dcor.distance_correlation(e, y_true_log, method="auto")   # dcor 0.6
# e = log(P_hat) - log(P)   (first argument),  y = log P   (second)
```

Full evaluation sample, no subsampling; default unit exponent; `bias_corrected` not passed. This is
the usual **biased** distance correlation computed from double-centered distance matrices. It is
**not** the `U`-centered bias-corrected squared-distance-correlation estimator, which dcor also
exports (`dcor.u_distance_correlation_sqr`) and which this project never calls. The frozen record
carries a numerical check on n = 400: the executed value agrees with a hand-rolled double-centered
statistic to 1.3e-15 and differs from the U-centered statistic by 7.5e-3.

---

## 9. Standards status

The adopted reference throughout is the **2013 IAAO Standard on Ratio Studies**. The **May 2026**
ratio-studies document is an **Exposure Draft** — proposed, not adopted — and every VEI or
band attribution in the paper is marked as such. Provenance:
`analysis/p1_inferential_reporting/provenance/ed2_source_manifest.json`, which also records that the
exposure-draft PDF is **not committed** to this repository because redistribution permission has not
been established.

---

## 10. What cannot be done from this repository

* Refit any model, penalized or not.
* Regenerate the combined path table or any per-configuration prediction array.
* Recompute any reported metric from raw sales.
* Reconstruct the exact executed environment from `requirements.txt`.
* Reconstruct the exact uncommitted source state that generated the frozen artifacts.
* Obtain the CCAO extract.

Each of these requires either authorized access to the restricted input and the executed run roots,
or information that was never recorded. The paper states this rather than working around it.
