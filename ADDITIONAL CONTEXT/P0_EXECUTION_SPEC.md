# P0 Execution Specification

## Objective
Resolve the empirical and implementation blockers identified in the final scientific audit before rewriting the manuscript.

Do not rewrite Results, Discussion, Abstract, Conclusion, or title during this pass.

## P0-1 — Audit Direct and Surrogate implementation

Verify against the executed code:

- exact training objective;
- target centering;
- initialization / base score;
- gradient scaling;
- Hessian scaling;
- `n/2` normalization;
- LightGBM custom-objective interface;
- all model parameters shared with native LightGBM.

For Direct, compute the actual magnitude of:

    1 + (rho / (2n)) * c_i^2

for every displayed rho and training fold, and compare it with the omitted rank-one curvature along the target-centered direction.

Do not accept the audit report's "effectively gradient-only" conclusion until verified numerically from the actual implementation.

Deliver:
- `P0_IMPLEMENTATION_AUDIT.md`
- machine-readable diagnostics.

## P0-2 — Native vs custom rho=0 parity

At rho=0, custom Direct/Surrogate should reproduce the corresponding unpenalized squared-error learner as closely as the implementation permits.

Identify the exact source of any discrepancy.

Align and audit:
- training rows/order;
- features;
- categorical handling;
- initialization/base score;
- number of trees;
- early stopping;
- learning rate;
- seeds;
- row/column subsampling;
- Hessian scale;
- regularization parameters;
- split/gain thresholds;
- prediction transformation.

Report:
- mean/median/max |prediction difference|;
- full distribution of prediction differences;
- best iteration if applicable;
- complete metric suite for:
  - native LightGBM;
  - Direct rho=0;
  - Surrogate rho=0;
on held-out and 2025 samples.

Do not interpret positive-rho improvements until this control is resolved.

Deliver:
- `RHO_ZERO_PARITY_REPORT.md`
- full rho=0 metrics CSV.

## P0-3 — Centered-spread post-hoc comparator

For each training block T and fitted baseline f0, define

    f_b(x) = ybar_T + b * (f0(x) - f0bar_T)

where both means are calculated on T only.

Compute the training-sample zero-covariance scale directly as

    b_star = Var_T(y) / Cov_T(f0, y)

when Cov_T(f0,y) > 0.

Do not use 1/R^2 as the general LightGBM endpoint; that identity is specific to the linear/projection case.

Trace a dense b-path from b=1 through and moderately beyond b_star.

Use no held-out or 2025 outcomes for:
- selecting b;
- defining the grid;
- matching configurations.

Evaluate:
- seven rolling-origin validation folds;
- held-out block;
- 2025 forward block.

Metrics:
- R2 price;
- MAE;
- MAPE;
- RMSE log-price;
- median/mean/weighted-mean ratio;
- COD;
- PRD;
- PRB;
- MKI;
- VEI;
- beta_log;
- Delta_NL;
- distance correlation;
- ratio-profile diagnostics.

Deliver:
- `CENTERED_SPREAD_COMPARATOR_REPORT.md`
- full path CSV.

## P0-4 — Matched first-order comparison

Compare Direct, Surrogate, and centered-spread post-hoc correction at matched development-sample beta_log values.

Do not match arbitrary rho values to arbitrary b values.

Freeze the matching procedure before held-out and 2025 evaluation.

Report whether, at comparable first-order correction, methods differ in:
- predictive accuracy;
- assessor-facing metrics;
- Delta_NL;
- dCor;
- ratio shape;
- temporal transfer.

## P0-5 — Temporal robustness

Primary design:
- enforce clean date boundaries so one sale date never appears on both sides of a chronological split.

Robustness variant:
- test parcel/repeat-sale blocking;
- do not replace the primary CCAO-style design with parcel blocking unless results show it is necessary.

A coarse rho grid is sufficient for the parcel-blocked robustness test.

Recheck:
- beta_log path;
- Delta_NL path;
- dCor path;
- key accuracy/equity anchors;
- candidate-region conclusions.

Deliver:
- `TEMPORAL_ROBUSTNESS_REPORT.md`

## Provenance and safety

- Work in a new isolated analysis folder.
- Do not overwrite frozen prior outputs.
- Record branch, commit, configuration, seeds, hashes, commands, and generated files.
- Do not invent missing experimental outcomes.
- Do not edit scientific conclusions before all P0 outputs are complete.

## Final output

After all P0 work, produce:

    MANUSCRIPT_IMPACT_MEMO.md

This memo should state:
- which prior conclusions survive;
- which fail;
- what changes are required in the manuscript;
- which claims remain experiment-dependent.

Do not implement the manuscript rewrite yet.