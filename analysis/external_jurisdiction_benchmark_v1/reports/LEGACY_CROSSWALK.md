# Legacy crosswalk: v1 vs. six-county ATTOM benchmark vs. berry_attom_validation_v3

Qualitative only. No statistical merging, no legacy reruns. This benchmark supersedes both
predecessors as the paper's external-jurisdiction evidence but does not modify either.

## Design comparison

| Dimension | Six-county benchmark (paper `subsec:attom_design`) | v3 | v1 (this benchmark) |
| --- | --- | --- | --- |
| Jurisdictions | Cook, Allegheny, Maricopa, King, Miami-Dade, Middlesex (6) | Wayne, Philadelphia, St. Louis County (3) | Wayne, Philadelphia, St. Louis, Allegheny, Maricopa, King, Miami-Dade, Middlesex (8 external) + Cook (bridge) |
| Property-use cohort | code 385 only, no dictionary | code 385 primary (18.2% retention in Philadelphia); `BROAD_RESIDENTIAL_RULE` v2 structural sensitivity | **Gated on external ATTOM code dictionary.** No structural-proxy fallback. code-385 retained only as a legacy bridge for comparability, never primary. |
| Features | Tax Assessor + ACS + location enrichment included | Tax Assessor/ACS excluded (HISTORY_MARKET_CORE) | Tax Assessor/ACS/location excluded (same exclusion method as v3: à la carte imports, never `main()`) |
| Temporal design | Chronological 80/20 split per county, no CV path | Chronological dev/val/test, no penalty-path CV | Calendar-year expanding CV folds (2018-2024) + forward=2025 once frozen; **explicitly not the CCAO rolling-origin design** |
| Baseline HPs | Not confirmed shared/frozen across counties before test | Frozen CCAO-tuned configs (3), CV-selected | One shared, non-CCAO-tuned 12-config grid, CV-selected per jurisdiction, frozen before penalty sweep |
| Direct/Surrogate | Direct only (`\newtext` main-text: "the substantive benchmark compares standard LightGBM with positive covariance-penalty values") | Direct: protocol-conforming within v3. Surrogate: `EXPLORATORY_METHOD_DIAGNOSTIC` (calibration rule amended after pass 1; Philadelphia's held-out rows scored twice) | Both families, on a `rho_tilde` grid normalized by training-block `Var(y)` (ddof=0) instead of a raw-rho grid, so cross-jurisdiction comparison is a stated design goal rather than an afterthought |
| Candidate-region screen | Not applied | Not applied | `utils/rho_screening_v2.py` (the CCAO v2.1 engine), reused unchanged, applied to `log10(rho_tilde)` |
| rho=0 parity | Not audited per-jurisdiction | Not audited | Explicit gate (Step 1), passed at ~1.5e-08 real-data mean gap using `match_native_init=True` |
| Confirmatory status | Manuscript's own disclosure: "not all frozen under one confirmatory protocol before test evaluation" | Direct path protocol-conforming *within v3*; not elevated to a whole-paper confirmatory claim | Protocol frozen (`protocol_external_benchmark_v1.yaml`) before any cohort/baseline/penalty outcome |

## Why numbers will differ, county by county

- **Feature information.** Six-county results include Tax Assessor/ACS/location signal that v1
  excludes; a lower v1 R² for the same county is not a regression, it is a smaller and more
  portable feature set by design.
- **Cohort definition.** Six-county and v3-primary both use code-385 alone; whatever v1's
  ATTOM-dictionary-based mapping turns out to be, retention and therefore the modeled population
  will differ, sometimes substantially (Philadelphia's code-385 retention is 18.2% vs. ~85% in
  Wayne/St. Louis under the same rule — a known, cohort-driven artifact, not a modeling difference).
- **Temporal design.** Six-county uses a single 80/20 split; v3 uses dev/val/test with no CV path;
  v1 uses calendar-year expanding CV plus a single forward evaluation. Metrics computed on
  differently-drawn test blocks are not directly comparable even for the same county and cohort.
- **Baseline hyperparameters.** Six-county's search-space provenance across counties is not
  confirmed identical; v3 reuses CCAO-tuned configs; v1 declares and freezes one shared,
  non-CCAO-tuned grid specifically so cross-jurisdiction comparisons are not confounded by
  per-county hyperparameter provenance.
- **rho normalization.** Only v1 defines `rho_tilde = rho * Var(y)` and tests whether it reduces
  cross-jurisdiction candidate-region dispersion. Six-county and v3 report raw rho only, which by
  construction cannot be compared across counties with different price-variance scales.
- **St. Louis official ratios.** v3 states no official assessed-value series exists; this benchmark
  corrects that to "true only of the file v3 chose" and reconstructs a genuine 2009-2019 benchmark
  from `joined.csv` under the already-published `intended_sold` specification. The resulting St.
  Louis official-ratio PRB (~-0.007, near-neutral) is a real and materially different finding from
  what v3's silence implied, not a discrepancy to reconcile.

## Not done here

No legacy experiment was rerun. No six-county or v3 artifact was modified, regenerated, or used as
training/evaluation input to v1's own baselines or penalty paths — only v3's already-computed
Berry/local transaction tables and linkage crosswalks were read for Step 4 (Berry evidence), which
is explicitly a read-only reuse, not a rerun.
