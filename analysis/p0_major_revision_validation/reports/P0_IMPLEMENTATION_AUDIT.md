# P0-1 — Direct / Surrogate Implementation and Curvature Audit

**Stage 1, task S2.** No model fits. Real CCAO fold data, all 9 canonical fitting blocks, the
full frozen 83-point rho grid (rho=0 plus 82 positive values).

Canonical splits verified: **344,607 / 38,290 / 26,641**. All seven archived rolling-origin fold
index hashes verified against `output/paper_v6_preselection_994/protocol/**/folds.json`.
Frozen parameter vector re-hashed to `8f0f2acd83118de782604b5ca7143acfbd2af3fd186ea9376588f9bcf560585b`.

## 1. Manuscript equations vs executed code

Verified by direct inspection of `soft_constrained_models/boosting_models.py`:

| Manuscript (App. E) | Code | Location | Match |
|---|---|---|---|
| `g_i = e_i + (rho/2) z c_i` | `grad = e + 0.5*rho*C*c` | L182-184 | yes |
| `h_i = 1 + (rho/(2n)) c_i^2` | `hess = ones(n) + (rho/(2n))*c**2` | L185 | yes |
| exact scaled Hessian `I + (rho/(2n)) c c^T` | `np.eye(n) + (rho/(2n))*np.outer(c,c)` | L189-195 | yes |
| Surrogate `g_i = e_i(1+rho c_i^2)`, `h_i = 1+rho c_i^2` | `weight = 1+rho*c**2; grad = e*weight; hess = weight` | L227-232 | yes |
| `n/2` scaling giving `grad=e, hess=1` at rho=0 | applied analytically, not numerically | L150-169 | yes |
| centering `y - ybar`, zero init, add back at predict | `base_score_`, `init_score=zeros`, `predict()+base_score_` | L502-533 | yes |

**No discrepancy was found between the manuscript equations and the executed implementation.**
There is no `rho == 0` branch anywhere; rho=0 reduces analytically.

## 2. Direct: supplied diagonal vs omitted rank-one curvature

Ranges are across all 9 fitting blocks. `Var_T(y)` spans 0.6212 (production, n=382,897) to
0.7832 (fold 1, n=46,888).

| rho | supplied `h` median | supplied `h` max | `M = max_i (rho/2n)c_i^2` | exact directional curvature `1+(rho/2)Var_T(y)` | exact / supplied ratio |
|---|---|---|---|---|---|
| 0.0104811 | 1.0000000024 – 1.0000000272 | 1.000001 | 1.237e-06 | 1.0033 – 1.0041 | 1 – 1 |
| 0.1 | 1.0000000232 – 1.0000002591 | 1.000012 | 1.18e-05 | 1.0311 – 1.0392 | 1.03 – 1.04 |
| 0.954095 | 1.0000002213 – 1.0000024723 | 1.000113 | 0.0001126 | 1.2963 – 1.3736 | 1.3 – 1.37 |
| 10.4811 | 1.0000024310 – 1.0000271595 | 1.001237 | 0.001237 | 4.2553 – 5.1043 | 4.26 – 5.1 |
| 100 | 1.0000231945 – 1.0002591274 | 1.011804 | 0.0118 | 32.0587 – 40.1594 | 32 – 40 |

Minimum supplied Hessian anywhere on the grid: **1.0** (the
penalty increment is non-negative, so `h_i >= 1` always).

**Reading.** The supplied diagonal is indistinguishable from 1 at every displayed anchor: its
median departs from 1 by at most 2.6e-4 even at rho=100. The curvature actually present in the
centered-target direction and discarded by the diagonal approximation reaches **40.2**
at rho=100 — a factor of **40.0** more than what is supplied.

## 3. E.4 verdict

Rule (plan E.4): ACCEPT if `M < 1e-2` at every displayed anchor and every block **and**
ratio >= 1.20 for rho >= 1; REJECT if `M >= 5e-2` anywhere or ratio < 1.05 throughout;
otherwise INDETERMINATE.

Measured:

* `max M` over all blocks and displayed anchors = **0.0118039**
* `min` curvature ratio at rho >= 1 = **4.255**
* `max` curvature ratio = **40.04**

> ### E.4 VERDICT: **INDETERMINATE**

The ratio criterion is satisfied with very large margin (4.26 vs the 1.20 threshold). The `M`
criterion fails in **exactly one cell of 45** (9 blocks x 5 anchors): fold 1 — the smallest
training block, n=46,888, the largest `Var_T(y)` = 0.7832 — at rho=100, where a single
extreme-tail observation reaches `M` = 0.0118, i.e. 1.18 % against a 1 % threshold. Every other
block/anchor combination has `M` <= 0.0063.

**Per the binding rule, the statement "Direct is effectively gradient-only, retaining essentially
native curvature" is NOT asserted as fact.** It is reported as INDETERMINATE under the
pre-registered rule, with the numbers above. The honest description is: *the supplied diagonal
departs from 1 by at most ~1.2 % for any single observation and by at most 0.026 % at the median,
while the omitted rank-one term carries 4.3x to 40x more curvature along the penalized direction
at rho >= 1.*

## 4. Surrogate weight concentration (closes audit H-7)

Weights `w_i = 1 + rho c_i^2`, ranges across the 9 blocks:

| rho | max weight | top-1 % weight share | effective sample size / n |
|---|---|---|---|
| 0.0104811 | 1.1 | 0.0106 – 0.0107 | 0.9998 – 0.9999 |
| 0.1 | 2.3 | 0.0157 – 0.0158 | 0.9867 – 0.9895 |
| 0.954095 | 13.1 | 0.0441 – 0.0471 | 0.6808 – 0.6998 |
| 10.4811 | 134.3 | 0.0810 – 0.0963 | 0.3005 – 0.3293 |
| 100 | 1272.7 | 0.0887 – 0.1080 | 0.2500 – 0.2858 |

At rho=100 the Surrogate discards roughly three quarters of its effective sample: ESS falls to
**25.0–28.6 %** of n, the top 1 % of observations carry **8.9–10.8 %** of total weight, and the
single largest weight reaches **1273**. Even at the rho≈0.954 anchor, ESS is already down to
**68–70 %**. This quantifies audit H-7 on real data and is a candidate mechanism for the
high-rho Surrogate ratio-shape trough — reported here as a measurement, not as an explanation.

## 5. Fixed-lambda leaf shrinkage — STYLIZED diagnostic only

`reg_lambda = 10.857207234911057` is absolute and enters `leaf = -G/(H+lambda)`. The Surrogate's
Hessian grows with rho while the Direct's stays ~1, so effective leaf shrinkage differs between
families at matched rho.

Every row of `tables/effective_leaf_shrinkage.csv` carries `diagnostic_type =
stylized_nominal_leaf`. For a nominal 93-observation leaf (three draws: uniform-random, low-`c^2`
decile, high-`c^2` decile) on the development pool, the shrinkage factor `H/(H+lambda)` rises from
0.8955 (Direct, all rho) to at most 0.9998 (Surrogate, rho=100, high-`c^2` leaf), a ratio of
**1.1164**.

**This is an illustration of a mechanism that could operate, not a measurement of what the fitted
trees do.** No claim that the Surrogate's rho effect includes implicit de-regularization may be
made until actual trained-leaf Hessian sums are measured (plan step 18b, not authorized in Stage 1).

## 6. `min_sum_hessian_in_leaf`

Never set in the frozen configuration, so LightGBM's default `1e-3` applies. Since `h_i >= 1` for
both families and `min_child_samples = 93`, every admissible leaf has `sum(H) >= 93`. Measured
minimum over all blocks, rhos and leaf draws: **93.0** (Direct),
**93.0** (Surrogate).

> **`min_sum_hessian_in_leaf` is checked and NON-BINDING.** It is not a mechanism and is not
> presented as one.

## 7. Float32 label representation (input to P0-2)

LightGBM stores `Dataset` labels as 32-bit. Cell A is fed `float32(y)` with `y ~ 12.2–12.5`;
Cells B and C are fed `float32(y - ybar)`. Measured on real blocks:

| quantity | mean | max |
|---|---|---|
| raw-label representation error | 2.359e-07 – 2.389e-07 | 4.768e-07 |
| centered-label representation error | 1.231e-08 – 1.514e-08 | 1.192e-07 |
| effective label gap between the two | 2.361e-07 – 2.397e-07 | 5.863e-07 |

Centering improves label resolution by roughly 18x. Whether this matters is answered
empirically in the P0-2 parity report — where it is **refuted** as the cause of the
native/custom gap.

## Artifacts

`tables/objective_scaling_audit.csv`, `tables/direct_hessian_magnitudes.csv`,
`tables/surrogate_weight_distribution.csv`, `tables/effective_leaf_shrinkage.csv`,
`tables/label_quantization.csv`, `tables/e4_verdict.json`.
