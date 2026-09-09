# Charter

## Project

Covariance-guided regressivity correction in a research translation of the Cook County
Assessor's Office (CCAO) residential LightGBM valuation workflow.

## Scope

The project studies a deliberately narrow, training-time intervention on one fixed base
learner. Regressivity is formalised as an undesired first-order dependence between
log-price residuals and log sale price, `Cov(e, y)` with `e = log P_hat - log P`. Two
objectives are compared:

- **Direct** — a squared-covariance penalty `(rho/2) * Cov(e,y)^2`. LightGBM receives the
  exact gradient but only the diagonal of a dense rank-one Hessian.
- **Surrogate** — a sample-additive Jensen upper bound that is exactly weighted squared
  error with weights `w_i(rho) = 1 + rho * (y_i - ybar)^2`, so gradient and Hessian are both
  exact.

Evaluation traces the **complete regularization path** over a penalty grid frozen in
advance, across seven rolling-origin folds (development n = 344,607), a later held-out
block (n = 38,290) and a separate 2025 forward sample (n = 26,641), with the 994-tree
LightGBM configuration held constant.

## Estimand

A global first-order price-related pattern **among observed, verified sales**. Not latent
market value, not the full assessment roll, not realised tax liability.

## Reference convention (binding)

- **Cell C** — custom-objective `rho = 0` origin — is the primary *within-path
  penalty-isolating* reference. Only comparisons against C may be attributed to `rho`.
- **Cell A** — ordinary LightGBM — is an assessor-facing *workflow benchmark*.
  Comparisons against A are descriptive only and must never be attributed to `rho` alone.
- **Cell B** — centered-label native L2 (initialization-aligned) — is an
  implementation-decomposition control. It is never printed as "Ordinary LightGBM".

## What this project is NOT

- Not a general fairness method or guarantee.
- Not a selected operating point: **no penalty strength is selected anywhere.**
- Not a compliance claim. COD lies outside the adopted IAAO `[5,15]` range at every
  reported configuration, including the path origin.
- Not a claim that training-time regularization is necessary or superior to post-hoc
  rescaling. The frozen matched-correction comparison does not support a ranking.
- Not an external-validity claim. The multi-jurisdiction / ATTOM stream is excluded
  (see `DEC-0002`).
- Not an adoption claim. CCAO has not adopted the correction or placed it in production.

## Authority for this capsule

Every object is grounded only in artifacts reachable from commit `993db93`
(`[Tier B2.5] Close temporal robustness and portability interpretation`) on branch
`research-os/r0-dogfood`. See `STATE.md` for what that snapshot does and does not settle.
