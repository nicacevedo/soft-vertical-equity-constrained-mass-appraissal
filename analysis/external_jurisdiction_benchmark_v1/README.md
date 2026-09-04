# External-jurisdiction benchmark v1

Isolated, paper-ready external-jurisdiction benchmark for the property-regressivity paper.
Supersedes both the manuscript's exploratory six-county ATTOM benchmark and
`analysis/berry_attom_validation_v3/` as the paper's external-jurisdiction Results evidence.
Neither predecessor is modified.

Protocol: `protocol_external_benchmark_v1.yaml` (frozen before cohort/baseline/penalty outcomes;
the objective-scaling/parity audit is a pre-outcome correctness gate, not a scientific result).

**Unattended execution rule** (governs all of Phase B): no scientific criterion may be altered
automatically. Any correctness failure, missing input, schema surprise, uninterpretable parity, or
missing residential mapping stops the workflow and writes `BLOCKER.md`. Scientifically unfavorable
results (null candidate region, failed portability test, UNATTAINED targets) are never blockers.

## Status

**Residential mapping FROZEN.** `cohort/residential_code_mapping.yaml`:
`PRIMARY_RESIDENTIAL = {363, 376, 377, 380, 382, 383, 384, 385, 386, 390}`.

**Development/CV complete; 2025 forward locked until `path_freeze/FORWARD_FREEZE.yaml`.**
Canonical compact artifacts (do not look under `output/` for the paper record):

- sources: `scripts/v1_common.py`, `audits/history_source_resolution.yaml`
- cohort: `cohort/residential_code_mapping.yaml`, `cohort/modeling_table_summary_dev.csv`
- temporal design: `temporal_design.yaml` (dev 2016–2024, forward 2025, folds 2018–2024)
- baseline freeze: `baseline/BASELINE_FREEZE.yaml`, `baseline/all_jurisdiction_baseline_summary.csv`
- normalized CV: `cv/*_normalized_cv_path_summary.csv`
- candidate regions: `candidate_regions/candidate_regions.csv`
- portability/overlap: `tables/normalization_portability.csv`, `candidate_regions/cross_jurisdiction_band.csv`
- forward freeze: `path_freeze/FORWARD_FREEZE.yaml`

```
/home/nacevedo/.conda/envs/fairness_env/bin/python analysis/external_jurisdiction_benchmark_v1/scripts/run_v1_tests.py
```

Partition: `sched_mit_sloan_batch_r8` (Sloan). Never `mit_normal`.
