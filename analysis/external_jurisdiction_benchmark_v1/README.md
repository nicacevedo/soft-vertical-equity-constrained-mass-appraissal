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

**Phase A (runs without external input): COMPLETE.**
- Step 0 preflight: `audits/preflight.json`
- Step 1 objective-scaling + rho=0 parity gate: **PASSED** — `audits/objective_scaling_audit.md`.
  Real-data parity ~1.5e-08 across Wayne/Philadelphia/Cook (vs. the frozen manuscript's 3.24e-02
  gap), because every fit here uses the already-validated `match_native_init=True` configuration.
- County caches for all 9 jurisdictions: `output/cache/<key>/{recorder,history}.parquet`,
  `audits/county_cache_manifest.csv`. One corrupt shard in Allegheny's Assessor History
  (`assessor-history_1_7_0.snappy.parquet`) skipped gracefully, not fatal.
- Step 3 temporal completeness audit: `audits/monthly_completeness.csv`,
  `audits/temporal_completeness_decision.json` (see `temporal_design.yaml` once frozen).
- Step 4 Berry/local external regressivity: `berry/berry_external_metrics.csv`,
  `berry/berry_linkage_preservation.csv`, `figures/berry_ratio_profiles.pdf`. St. Louis County
  included (corrects v3's overstated "no official assessed-value series" claim — true only of the
  file v3 chose).
- Test suite: `scripts/run_v1_tests.py` — 21/21 passing.

**Phase B: residential mapping FROZEN.** `cohort/residential_code_mapping.yaml`:
`PRIMARY_RESIDENTIAL = {363, 376, 377, 380, 382, 383, 384, 385, 386, 390}` (Single Family
Residence=385, Townhouse=386, Seasonal/Cabin=384, etc.), from the user-verified ATTOM PropertyType
reference. An initial transcription (385=Seasonal/Cabin, 386=SFR) was caught by an empirical
jurisdiction x code frequency check as inconsistent with the observed volume/geography pattern
(385 was 54.7% of all rows, dominant in Cook/Maricopa/Miami-Dade — not plausible for a seasonal
code) and flagged per the "stop if semantically inconsistent" rule; the user re-checked the source
and corrected it. Full audit trail in `cohort/residential_code_mapping.yaml::correction_history` and
`cohort/UNCLASSIFIED_CODES_REPORT.md`.

Cohort retention under the frozen mapping: `cohort/cohort_retention.csv`. Primary-residential share
ranges from 25.7% (Philadelphia) to 76.5% (Wayne) — reported as a real housing-composition
difference (Philadelphia's excluded condo code 366 covers 41.3% there), per the frozen rule that
membership is never adjusted for retention or model outcomes.

`BROAD_RESIDENTIAL_APPENDIX` remains partially specified (only code 366=Condominium confirmed) and
is not used for any modeling outcome until complete.

```
/home/nacevedo/.conda/envs/fairness_env/bin/python analysis/external_jurisdiction_benchmark_v1/scripts/run_v1_tests.py
```

Partition: `sched_mit_sloan_batch_r8` (Sloan). Never `mit_normal`.
