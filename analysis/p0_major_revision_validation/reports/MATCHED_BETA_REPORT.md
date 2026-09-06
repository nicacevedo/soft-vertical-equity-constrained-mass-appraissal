# MATCHED-BETA REPORT — what retraining buys at equal first-order correction

Stage 3A. Development coordinate **D1** (equal-weight seven-fold mean `β_log`) is PRIMARY.
Primary matched triple: **Direct / Surrogate / C-posthoc**. **A-posthoc** is SECONDARY.
**Cell B has no post-hoc path** and does not appear here.

Frozen before any held-out or 2025 metric was read:
`configs/matched_beta_frozen.json`,
`file_sha256 = ace12b09e79b92511b01979620b48a49e0765f990de4b93ad6025174a069a7a1`,
frozen `2026-09-06T16:38:45Z`. The evaluation script re-validates that hash before it runs.

**No out-of-sample metric anywhere in this report is interpolated.** Every number is either a
measured value of an actually fitted configuration (read from the frozen 82-point path table) or
an actual evaluation of the theorem-matched transformation at an exactly solved `b`.
`match_mode` is carried on every row.

---

## 1. The matched design

Three-way common support on D1: **[−0.138271, −0.078651]**. Direct is the binding family.
Six CORE targets are produced by the frozen deterministic rule — probe `q_j = L + j/5·(U−L)`,
then take the *achieved* `β_log` of the unused fitted Direct configuration nearest `q_j`
(ties → smaller ρ). The target is the achieved value, never the probe, so Direct matches exactly
by construction and no selection discretion exists.

| j | target `β_log` | Direct ρ | Surrogate ρ (gap) | C-posthoc `b` | A-posthoc `b` |
|---|---|---|---|---|---|
| 0 | −0.138271 | 0 | 0 (0) | 1.000000 | 0.999922 |
| 1 | −0.126490 | 0.719686 | 0.132571 (5.6e−04) | 1.013671 | 1.013592 |
| 2 | −0.114733 | 1.676833 | 0.409492 (1.5e−04) | 1.027316 | 1.027235 |
| 3 | −0.102033 | 3.906940 | 0.828643 (4.0e−05) | 1.042053 | 1.041972 |
| 4 | −0.090902 | 9.102982 | 1.456348 (1.1e−03) | 1.054971 | 1.054888 |
| 5 | −0.078651 | 86.851137 | 2.559548 (3.1e−04) | 1.069186 | 1.069103 |

All 24 CORE cells are attained within the frozen tolerance **τ = 0.002** (worst gap 1.06e−03).
`match_mode`: Direct `exact_anchor`, Surrogate `nearest_fitted` throughout, C/A-posthoc
`exact_posthoc_solve`. **Zero targeted new fits were required.**

**Origin identity (QC).** At j=0 the three primary families are the same object: Direct ρ=0 ≡
Surrogate ρ=0 ≡ Cell C, and C-posthoc at `b = 1` returns `f₀` through the bitwise fast path.
Measured agreement across all 17 metrics and all four evaluations: **max relative 4.42e−12,
max absolute 2.73e−12** (`tables/matched_beta_origin_identity.csv`). The residual is CSV
serialization precision, because the two sides come from *different sources* — Direct/Surrogate
from the frozen path CSV, C-posthoc recomputed from cached predictions. So this check does double
duty: it confirms the origin identity **and** independently validates the Stage-3 recomputation
against the frozen table. Because these deltas are structurally zero, j=0 is excluded from
crossing detection and asserted instead.

The post-hoc solve is exact because `β_log` is exactly linear in `b`. On D1,
`β(b) = −1 + b·mean_k R_k` (intercept recovers to −1.0 within 1e−15). On the *pooled* coordinates
the centered map applies a different `ȳ_T` per fold, so the relation picks up a
`(1−b)·Cov_pool(ȳ_T, y)` term and the intercept is **not** −1; it is recovered by an exact
two-point solve with a residual assertion ≤1e−12 (measured ≤2.2e−16). That distinction matters
for the D3 sensitivity and is implemented, not assumed.

---

## 2. The eight questions

### Q1 — At MILD correction, does retraining beat C-posthoc, or is global rescaling as good or better?

**At the mildest correction retraining wins by a small margin on held-out, and the margin does not
survive to 2025.** At j=1 (`β_log = −0.1265`) held-out: Direct − C-posthoc = **+0.00122 R²**,
**−0.00050 RMSE_log** — Direct better on both. On CV_mean the same sign (+0.00112 R², −0.00006
RMSE_log). On 2025, Direct is still better in RMSE_log (−0.00058) and R² (+0.00072).
Surrogate at j=1 is *worse* than C-posthoc on held-out (−0.00183 R², +0.00163 RMSE_log).

So at mild correction the honest answer is: **global rescaling is essentially as good.** The Direct
advantage is ~0.001 in R² — the same order as the Cell-A↔Cell-C implementation gap (0.0014) that
Gate G2 characterised, and well inside the fold-to-fold variation. It is a real sign, not a
material margin.

### Q2 — At MODERATE correction, where do the accuracy frontiers cross?

**Between j=2 and j=3, i.e. `β_log ≈ −0.115` to −0.102 (Direct ρ ≈ 1.68 → 3.91).**
`tables/matched_beta_crossings.csv` locates the Direct−C-posthoc R² sign change at exactly that
bracket on both CV_mean (+0.00107 → −0.00057) and held-out (+0.00064 → −0.00167). In RMSE_log the
crossing is one step earlier, between j=1 and j=2 (−0.00050 → +0.00056 held-out).

Past that point **C-posthoc is uniformly more accurate than Direct**, and the gap widens
monotonically: held-out R² deficit for Direct is −0.0017 (j=3), −0.0042 (j=4), **−0.0206 (j=5)**.
At j=5 Direct is not merely behind, it has degraded outright — held-out RMSE_log 0.3170 versus
0.2920, COD 23.97 versus 21.61. Reaching `β_log = −0.0787` costs Direct ρ = 86.85, and at that
penalty the fit is damaged.

Surrogate crosses differently: it is *behind* C-posthoc in R² across j=1..4 on held-out
(−0.0018 to −0.0044) but recovers to parity at j=5 (−0.00035), and on 2025 it *overtakes*
C-posthoc at j=5 (+0.0031).

### Q3 — At STRONG correction / near β=0, how much predictive performance does Surrogate preserve relative to C-posthoc?

**Substantially more, and the gap grows as the target tightens — this is the clearest accuracy
result in the stage.** From `tables/matched_beta_ext_targets.csv` (held-out):

| dev target | Surrogate R² | C-posthoc R² | Surrogate advantage | Surrogate RMSE_log | C-posthoc RMSE_log |
|---|---|---|---|---|---|
| −0.06 | 0.89340 (ρ=5.96) | 0.88679 (b=1.0908) | **+0.0066** | 0.3230 | 0.2944 |
| −0.03 | 0.89168 (ρ=28.12) | 0.85686 (b=1.1256) | **+0.0348** | 0.3443 | 0.2999 |
| 0.00 | **NOT ATTAINED** (max −0.0186) | 0.80628 (b=1.1605) | — | — | 0.3071 |

In `R²` on price, Surrogate preserves far more than global rescaling: at `β_dev = −0.03`,
C-posthoc has given up 0.036 of R² relative to the origin while Surrogate has given up 0.001.
Pushing C-posthoc all the way to `β_dev = 0` costs it **0.087 of held-out R²** (0.8928 → 0.8063).

Two qualifications the numbers force, and neither should be dropped:
**(a)** In `RMSE_log` the ranking reverses — Surrogate is worse (0.3443 vs 0.2999 at −0.03).
The two accuracy metrics disagree because they weight the price distribution differently;
`R²` on price is dominated by the expensive tail that the Surrogate's `w_i = 1 + ρc_i²` weighting
protects, while `RMSE_log` weights all sales equally in logs. Both are reported; neither alone
settles "predictive performance."
**(b)** Surrogate **cannot reach `β_dev = 0`** at any fitted ρ ≤ 100 (its maximum is −0.0186).
Only the post-hoc families attain the full range, and they do so by construction.

### Q4 — At equal β, which method has lower Δ_NL?

**Surrogate, decisively and at every matched target, on every evaluation block.** This is the
strongest and most consistent finding of the stage. Surrogate − C-posthoc in Δ_NL, held-out:
−0.0060 (j=1), −0.0183, −0.0260, −0.0324, **−0.0368 (j=5)**. On 2025 it reaches **−0.0490**.
In levels (held-out): Surrogate takes Δ_NL from 0.1165 at the origin *down* to **0.0914** at j=5,
while C-posthoc *rises* to 0.1282 and Direct to 0.1198.

The direction is what matters. **Matching the first-order correction does not equalize the
nonlinear residual structure — it moves the two families in opposite directions.** Global spread
rescaling, being affine, cannot change nonlinear residual–price structure except incidentally, and
what it does incidentally is make it worse. Direct, despite retraining, behaves like the post-hoc
comparator here (it also increases Δ_NL at j=1..4 on held-out) and only turns down at j=5 where
its overall fit has degraded. So the Δ_NL result is specifically a **Surrogate** result, not a
generic retraining result.

### Q5 — At equal β, which method has lower dCor?

**The retrained families, at moderate-to-strong correction, with Surrogate lowest at high
correction — but the margins are an order of magnitude smaller than for Δ_NL.**
Held-out Direct − C-posthoc: +0.0012 (j=1), then −0.0014, −0.0022, −0.0079, **−0.0214** (j=5).
Surrogate − C-posthoc: +0.0034 (j=1), then −0.0080, −0.0114, −0.0157, **−0.0230** (j=5).
At mild correction both retrained families are *higher* (worse) than the comparator; the ordering
flips by j=2 and then widens. `tables/matched_beta_crossings.csv` records the j=1→2 sign change
for both pairs on held-out.

In levels all three families fall together from ≈0.382 to ≈0.27–0.29 on held-out, so this is a
second-order distinction between broadly similar paths — unlike Δ_NL, where the families separate
in *direction*.

### Q6 — Does Direct do anything that cannot be reproduced by C-posthoc in its attainable β range?

**On these coordinates, no.** Inside Direct's attainable range (`β_log ∈ [−0.1383, −0.0787]`),
C-posthoc matches or beats Direct on essentially every axis:
- **Accuracy:** C-posthoc better from j=3 onward, by up to 0.021 R² and 0.025 RMSE_log at j=5.
- **Assessor-facing:** C-posthoc lower COD at every j≥1 (held-out gap +0.05 to **+2.36** at j=5).
- **Δ_NL:** Direct is *lower* than C-posthoc by only 0.0024–0.0050 at j=1..4 on held-out, and at
  j=1..3 the sign is the wrong way (Direct higher). Direct's one clear Δ_NL advantage (−0.0083 at
  j=5) is bought at ρ=86.85, where its R² has collapsed by 0.021.
- **dCor:** Direct's advantage is ≤0.021 and only at j=4–5.
- **Ratio shape:** the IAAO proxy-group profiles at j=0/3/5 (`figures/matched_beta_ratio_profiles.pdf`)
  show Direct and C-posthoc tracking within overlapping 90% bootstrap intervals across groups at
  j=3; the visible divergence at j=5 is Direct's degradation, not a differently-shaped correction.

The conclusion is specific and should be stated plainly: **the Direct family's empirical effect
in the fixed-prediction-space benchmark is, to the resolution of these six matched targets,
reproducible by a one-parameter global rescaling.** That is exactly what Cor. `cor:path_scaling`
predicts, so this is theory confirmed rather than a defect — but it does remove any claim that
Direct's benefit comes from retraining the nonlinear learner.

### Q7 — Does Surrogate's high-correction behaviour constitute evidence of shape correction beyond global first-order spread rescaling?

**Yes — this is the one place where retraining demonstrably does something rescaling cannot.**
Three independent strands, all pointing the same way:

1. **Δ_NL moves in the opposite direction** (Q4): −0.037 versus the comparator at j=5 held-out,
   −0.049 on 2025, and in levels Surrogate *reduces* Δ_NL below the origin while both other
   families raise it. An affine map of `f₀` cannot do this; only a differently-shaped fit can.
2. **Accuracy preservation at strong correction** (Q3): at `β_dev = −0.03` Surrogate holds
   held-out R² at 0.8917 while C-posthoc has fallen to 0.8569. Global rescaling buys first-order
   correction by inflating spread, which necessarily costs accuracy; Surrogate does not pay that
   price on price-scale R².
3. **COD** is lower for Surrogate than C-posthoc at **every** matched target on held-out
   (−0.046 to −0.216) and 2025, so the ratio distribution is genuinely tighter, not merely shifted.

The honest boundary on this claim: it is **descriptive**, it rests on the Δ_NL/dCor/COD triple
rather than on any inferential test, and it is a **Surrogate** finding that does **not** extend to
Direct. And the `RMSE_log` reversal in Q3(a) must travel with it.

### Q8 — Do these conclusions transfer from development to held-out and 2025?

**The two headline conclusions transfer cleanly; the mild-correction accuracy detail does not.**

| Conclusion | CV_mean | held-out | 2025 |
|---|---|---|---|
| Surrogate has lower Δ_NL at every matched target | yes (−0.008 → −0.022) | yes (−0.006 → −0.037) | yes (−0.010 → −0.049) |
| C-posthoc ≥ Direct in accuracy from moderate correction on | yes (crosses j=2→3) | yes (crosses j=2→3) | yes (crosses j=1→2) |
| Direct degrades at j=5 | yes (R² −0.027) | yes (−0.021) | yes (−0.026) |
| Surrogate lower COD than C-posthoc at all j≥1 | yes | yes | yes |
| Direct better than C-posthoc at j=1 | yes (+0.0011 R²) | yes (+0.0012) | yes (+0.0007) |
| dCor ordering at mild correction | retrained higher | retrained higher | mixed (Direct already lower) |

The crossing point moves one grid step earlier on 2025, and the mild-correction `dCor` sign is not
stable across blocks. Everything that carries a material margin transfers; what does not transfer
is confined to differences at or below ~0.001 in R² and ~0.002 in dCor.

---

## 3. D3 bounded sensitivity (required: `D3_MATERIAL = TRUE`)

D1 remains PRIMARY. Re-running the identical K=6 construction on the row-balanced D3 coordinate
re-selects **13 of 24** cells (Direct's j=2 anchor moves ρ 1.6768 → 1.9307; the post-hoc `b` values
shift because the pooled relation has a different slope and intercept). The D3 CORE selection
reproduces the targets recorded in `tables/development_beta_coordinate_summary.json` **before** any
OOS read, digit for digit — so the D3 branch is provably not outcome-selected.

Of **315** headline pairwise-delta sign comparisons (`tables/matched_beta_d3_sign_agreement.csv`):

| classification | n |
|---|---|
| sign agrees | 259 |
| both below materiality floor | 41 |
| **material sign flip** | **10** |
| structural zero (D1 match forces an exact zero) | 5 |

The 10 material flips are all in small differences — the largest is held-out MAE at j=2
(+15.3 → −149.0, against an MAE of ≈76,000, i.e. 0.2%). **No Δ_NL comparison flips sign under D3
anywhere**, and no flip touches Direct's j=5 degradation. Largest headline metric shifts:
R² 0.0015, RMSE_log 0.0009, Δ_NL 0.0025, dCor 0.0062, COD 0.089.

**Conclusion: every conclusion in §2 that carries a material margin survives the D3 coordinate.**
What D3 perturbs is the fine ordering of near-tied configurations — expected, because both Direct
and Surrogate `β_log` paths are non-monotone in ρ under *both* coordinates, so a 0.002–0.005
coordinate shift reorders configurations that were already nearly tied.

---

## 4. Attainability asymmetry — a result, not a gap

| family | max attained development `β_log` | reaches −0.06? | −0.03? | 0.00? |
|---|---|---|---|---|
| Direct | **−0.078651** (ρ=86.85) | no | no | no |
| Surrogate | **−0.018609** | yes (ρ=5.96) | yes (ρ=28.12) | no |
| C-posthoc / A-posthoc | unbounded by construction | yes | yes | yes |

Direct cannot reach any EXT target at ρ ≤ 100; Surrogate reaches two of three. These rows are
carried explicitly as `match_mode = NOT_ATTAINED` with the maximum achieved correction, never
dropped. The asymmetry is itself informative: the post-hoc comparator's ability to hit any `β`
target is not an advantage in isolation, because at `β_dev = 0` it costs 0.087 of held-out R².

---

## 5. Artifacts

Tables: `matched_beta_comparison.csv` (96 rows), `matched_beta_pairwise_deltas.csv` (72),
`matched_beta_crossings.csv` (26), `matched_beta_ext_targets.csv` (48),
`matched_beta_origin_identity.csv` (8), `matched_beta_ratio_profiles.csv` (480 IAAO proxy-group
rows), `matched_beta_ratio_profiles_price_bins.csv` (1,440 rows, 30 equal-count price bins),
`matched_beta_d3_sensitivity.csv`, `matched_beta_d3_sign_agreement.csv`, plus the `_d3` twins of
the comparison/pairwise/ext/crossings tables and `.parquet` twins throughout.

Configs: `matched_beta_frozen.json` (+`_hash.json`), `matched_beta_d3_frozen.json` (+`_hash.json`).

Figures: `matched_beta_accuracy_equity.pdf`, `matched_beta_mechanism.pdf`,
`matched_beta_ratio_profiles.pdf` — all with the matched development `β_log` target on the
horizontal axis.

Provenance: `provenance/POST_G3_ADJUDICATION.md`.

## 6. Language boundary

Every comparison here is **descriptive**. No difference is called statistically significant; no
inferential procedure was run in this stage (PRB standard errors and VEI significance remain
explicitly out of scope). The `dCor`, `Δ_NL` and `COD` margins are reported as measured
differences between fitted configurations, with the coordinate sensitivity in §3 as the available
robustness statement.
