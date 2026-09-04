# Objective-scaling + rho=0 parity audit
Written: 2026-09-03T20:08:16Z
## rho_tilde = rho * Vy_T equivalence
| family    |   rho_direct |     Vy_T |   rho_tilde |   rho_recovered |   max_abs_prediction_diff | exact_match   | county_key   |
|:----------|-------------:|---------:|------------:|----------------:|--------------------------:|:--------------|:-------------|
| direct    |          0.5 | 0.409739 |    0.204869 |             0.5 |                         0 | True          | wayne        |
| surrogate |          0.5 | 0.409739 |    0.204869 |             0.5 |                         0 | True          | wayne        |
| direct    |          3   | 0.409739 |    1.22922  |             3   |                         0 | True          | wayne        |
| surrogate |          3   | 0.409739 |    1.22922  |             3   |                         0 | True          | wayne        |
| direct    |          0.5 | 0.326786 |    0.163393 |             0.5 |                         0 | True          | philadelphia |
| surrogate |          0.5 | 0.326786 |    0.163393 |             0.5 |                         0 | True          | philadelphia |
| direct    |          3   | 0.326786 |    0.980359 |             3   |                         0 | True          | philadelphia |
| surrogate |          3   | 0.326786 |    0.980359 |             3   |                         0 | True          | philadelphia |
| direct    |          0.5 | 0.441914 |    0.220957 |             0.5 |                         0 | True          | cook         |
| surrogate |          0.5 | 0.441914 |    0.220957 |             0.5 |                         0 | True          | cook         |
| direct    |          3   | 0.441914 |    1.32574  |             3   |                         0 | True          | cook         |
| surrogate |          3   | 0.441914 |    1.32574  |             3   |                         0 | True          | cook         |

## Native-vs-custom rho=0 parity (real pilot-county data)
      county    n  direct_vs_surrogate_max_abs_diff  native_vs_direct_mean_abs_diff  native_vs_direct_max_abs_diff  beta_native  beta_direct  abs_beta_diff  base_score_matches_mean_y
       wayne 4000                               0.0                    1.721411e-08                   1.331190e-07    -0.638100    -0.638100   2.183224e-09                       True
philadelphia 4000                               0.0                    1.312524e-08                   1.301178e-07    -0.702043    -0.702043   2.828176e-09                       True
        cook 4000                               0.0                    1.822829e-08                   9.810651e-08    -0.316158    -0.316158   3.992524e-09                       True

**Gate passed: True**

This reuses the exact canonical configuration validated on synthetic data by `tests/test_paper_v6_guards.py::test_native_custom_rho0_parity_after_mean_init` (ratio_mode='diff', match_native_init=True, and for Surrogate weighting_proxy_mode='identity'). `plan_rho_grid` normalizes by A = Var(baseline predictions), a DIFFERENT quantity from Vy_T used here; rho_tilde is never derived from A.
