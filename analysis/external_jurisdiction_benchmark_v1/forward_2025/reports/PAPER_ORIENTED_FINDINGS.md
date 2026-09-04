# Paper-oriented findings (frozen 2025 forward pass)

This note is interpretive. The canonical numbers are the CSVs under `forward_2025/`.

```
FORWARD_SUPPORTS_CV: 2025 equity correction has the same sign as CV at activity
and at the 25% A_beta anchor; 50% A_beta remains attained or still reduces
|beta|; no reversal of PRB/beta toward worse regressivity; Delta_NMSE has the
same sign as CV (or remains near zero).
FORWARD_PARTIAL: some frozen anchors transfer, others do not, with no wholesale
reversal of the correction mechanism.
FORWARD_WEAKENS_CV: correction remains same-signed but is materially smaller, or
predictive cost is materially larger, at the frozen activity/25%/50% coordinates.
FORWARD_REVERSAL: at a frozen activity or 25%/50% anchor, 2025 PRB or beta_log
moves away from the ideal relative to that split's own baseline.
```

Direct common band frozen at [0.387, 0.562].
Surrogate all-jurisdiction intersection remains `NO_NONDEGENERATE_ALL_JURISDICTION_INTERSECTION`.

Family-jurisdiction rubric: SUPPORTS=4, PARTIAL=14, WEAKENS=0, REVERSAL=0.

Predeclared paper figures (counties were not chosen from 2025 outcomes):

1. `figures/paper/accuracy_mechanism_frontier_cv_vs_2025.pdf`
2. `figures/paper/forward_key_metric_paths_9jurisdictions.pdf`
3. `figures/paper/forward_ratio_profile_examples.pdf` (Philadelphia, St. Louis County, Middlesex — chosen from pre-2025 roles).

Do not cherry-pick 2025 counties for the main figures.

Level shifts that matter for interpretation, not for re-estimating regions:

- Maricopa and Middlesex 2025 price-R2 sit well below their CV fold-means; the Direct correction still moves beta_log toward 0 at frozen activity/25%/50% coordinates.
- Cook and Maricopa 2025 baselines are more regressive than CV; activity still reduces |beta| at near-zero Delta_NMSE.
- Middlesex Direct 50% A_beta has a wild interpolated CV-mean R2 near the diverged tail; use the finite 2025 metric, not the interpolated CV R2.
- Wayne is not Detroit. Official-assessment and AVM ratio profiles remain separate constructs.

Do not write a 2025-derived candidate region. Allegheny Direct remains `NO_STABLE_CANDIDATE_REGION`. Surrogate remains `NO_NONDEGENERATE_ALL_JURISDICTION_INTERSECTION`.
