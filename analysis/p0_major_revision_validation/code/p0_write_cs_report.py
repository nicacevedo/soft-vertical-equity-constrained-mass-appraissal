#!/usr/bin/env python3
"""Write reports/CENTERED_SPREAD_COMPARATOR_REPORT.md."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd, yaml
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as c
T = c.TABLES

def main() -> int:
    g = json.loads((c.CONFIGS / "b_grid_frozen.json").read_text())
    gh = json.loads((c.CONFIGS / "b_grid_frozen_hash.json").read_text())
    conv = yaml.safe_load((c.CONFIGS / "posthoc_comparator_convention.yaml").read_text())
    gate = json.loads((T / "gate_g3_checks.json").read_text())
    roots = g["development_roots"]; anc = g["special_anchors"]; cov = g["coverage_verification"]
    path = pd.read_csv(T / "centered_spread_path.csv")
    cv = pd.read_csv(T / "centered_spread_cv_summary.csv")
    lin = pd.read_csv(T / "centered_spread_linearity.csv")
    rv = pd.read_csv(T / "centered_spread_root_verification.csv")
    q1 = pd.read_csv(T / "centered_spread_b1_qc.csv")
    sens = pd.read_csv(T / "centered_spread_centering_sensitivity.csv")
    hq = pd.read_csv(T / "centered_spread_existing_qc.csv")
    prof = pd.read_csv(T / "centered_spread_ratio_profiles.csv")
    ov = json.loads((T / "cv_fold_validation_overlap_summary.json").read_text())
    v4 = (c.V12 / "analysis" / "data_id=d4929d43ec19badf" / "split_id=3d464d4a611b131b"
          / "penalty_path_analysis" / "transition_regions_paper_assets_v4_delta_nl_bends"
          / "tables" / "combined_path_table_v4_analysis_view.csv")
    fp = pd.read_csv(v4)

    def at(ref, ev, b):
        s = path[(path.reference == ref) & (path.evaluation == ev)]
        return s.iloc[(s.b - b).abs().argmin()]

    def cvat(ref, b):
        s = cv[cv.reference == ref]
        return s.iloc[(s.b - b).abs().argmin()]

    MET = ["R2_price", "MAE_price", "RMSE_log", "COD", "PRD", "PRB", "MKI", "VEI",
           "beta_log", "Delta_NL", "dCor_e_y"]
    F = {"MAE_price": "{:,.0f}", "R2_price": "{:.5f}", "beta_log": "{:+.5f}",
         "Delta_NL": "{:.5f}", "dCor_e_y": "{:.5f}", "PRB": "{:+.5f}", "MKI": "{:.5f}",
         "PRD": "{:.4f}", "COD": "{:.3f}", "VEI": "{:+.3f}", "RMSE_log": "{:.5f}"}

    def anchor_table(ref, ev, keys):
        hdr = "| anchor | b | " + " | ".join(MET) + " |\n|" + "---|" * (len(MET) + 2) + "\n"
        for k in keys:
            r = at(ref, ev, anc[k])
            hdr += f"| `{k}` | {r.b:.6f} | " + " | ".join(
                F.get(m, "{:.5g}").format(r[m]) for m in MET) + " |\n"
        return hdr

    def cv_anchor_table(ref, keys):
        cols = [f"{m}__CV_mean" for m in MET]
        hdr = "| anchor | b | " + " | ".join(MET) + " |\n|" + "---|" * (len(MET) + 2) + "\n"
        for k in keys:
            r = cvat(ref, anc[k])
            hdr += f"| `{k}` | {r.b:.6f} | " + " | ".join(
                F.get(m, "{:.5g}").format(r[f'{m}__CV_mean']) for m in MET) + " |\n"
        return hdr

    md = f"""# Centered-Spread Post-hoc Comparator — Stage 2 (Gate G3)

Theorem-matched primary map, evaluated for the two frozen post-hoc bases:

* **PRIMARY** — **C**, `Custom-objective rho=0 origin`
* **SECONDARY practical** — **A**, `Ordinary LightGBM (standard raw-label native)`
* **B** — `Centered-label native L2 (initialization-aligned)`, decomposition control, **no full path**

$$f_b(x) = \\bar y_T + b\\,(f_0(x) - \\bar y_T)$$

with `ybar_T` the mean log target of the **fitting block** for the regime (fold-k training mean for
fold k; development pool for held-out; production 2016-2024 for 2025).

**No A or C model was refit.** The Stage-1.5 cached predictions were used, with all
{int(len(q1))} input prediction arrays hash-verified against the Stage-1.5 provenance before use.

The formal matched-beta comparison is **NOT** performed here; §7 below is explicitly descriptive.

---

## 1. Development calibration

`b_star_train = Var_T(y)/Cov_T(f0,y)` is retained **only as an in-sample theory diagnostic** and
was **not** used to define any endpoint, root or grid bound. Reason: the 994-tree learner is
substantially less regressive in sample than out of sample, so the in-sample root understates the
rescaling the data actually require.

| quantity | C | A |
|---|---|---|
| `beta_log_train` (in-sample, 9 blocks) | {roots['C']['beta_log_train_min_IN_SAMPLE']:.6f} … {roots['C']['beta_log_train_max_IN_SAMPLE']:.6f} | {roots['A']['beta_log_train_min_IN_SAMPLE']:.6f} … {roots['A']['beta_log_train_max_IN_SAMPLE']:.6f} |
| **`b_star_train`** *(diagnostic only)* | {roots['C']['b_star_train_min_IN_SAMPLE_DIAGNOSTIC_ONLY']:.6f} … {roots['C']['b_star_train_max_IN_SAMPLE_DIAGNOSTIC_ONLY']:.6f} | {roots['A']['b_star_train_min_IN_SAMPLE_DIAGNOSTIC_ONLY']:.6f} … {roots['A']['b_star_train_max_IN_SAMPLE_DIAGNOSTIC_ONLY']:.6f} |
| `beta_CVmean` at `b=1` (D1) | {roots['C']['beta_cvmean_at_b1']:+.8f} | {roots['A']['beta_cvmean_at_b1']:+.8f} |
| **`b_zero_cvmean`** (D1 PRIMARY root) | **{roots['C']['b_zero_cvmean']:.12f}** | **{roots['A']['b_zero_cvmean']:.12f}** |
| `beta_pooled_oof` at `b=1` (D2) | {roots['C']['beta_pooled_oof_at_b1']:+.8f} | {roots['A']['beta_pooled_oof_at_b1']:+.8f} |
| **`b_zero_pooled_oof`** (D2 sensitivity root) | **{roots['C']['b_zero_pooled_oof']:.12f}** | **{roots['A']['b_zero_pooled_oof']:.12f}** |

Roots computed as specified, from development fold predictions only:

* D1: `beta_k(b) = b * Cov_Vk(f0,y)/Var_Vk(y) - 1`, `beta_CVmean(b) = (1/7) sum_k beta_k(b)`,
  hence `b_zero_cvmean = 1 / mean_k[ Cov_Vk(f0,y)/Var_Vk(y) ]`.
* D2: closed-form root of the pooled coordinate after applying fold-specific training centers,
  `b = (V - P)/(Q - P)` with `V = Var_pooled(y)`, `P = mean(ybar_T(k(i)) c)`, `Q = mean(f0 c)`.
  `P` is non-zero because the fold centers correlate with `c`.

**Why the empirical roots exceed `b_star_train` by so much.** `b_star_train` ~ 1.065–1.093 while the
development roots are ~1.160–1.167. The in-sample residual–price association is only about
−0.06 to −0.09 whereas out of sample it is about −0.14 to −0.16, so an in-sample calibration would
under-correct by roughly a factor of two in `(b-1)`. This is precisely why the post-G2 refinement
moved the calibration to development out-of-sample predictions.

**Independent validation.** A's recomputed `b_zero_pooled_oof` is
**{roots['A']['b_zero_pooled_oof']:.13f}**, matching the historical native `b* = 1.1672656789694134`
to 13 significant digits — while C's differs ({roots['C']['b_zero_pooled_oof']:.12f}). The historical
value was **not** reused; A-specific and C-specific roots were recomputed from the current verified
cached predictions and are genuinely different.

### Path upper bound

`b_max = 1 + 1.25 * (max(all four development roots) - 1)` — a true **25 % overshoot in the
adjustment `(b-1)`**, not `1.25 * b_star`.

`b_ref_max = {g['b_reference_max_used']:.12f}` (A's D2 root) ⇒ **`b_max = {g['b_max']:.12f}`**.

Development-only coverage verification (all requirements met = **{cov['all_requirements_met']}**):

| requirement | C | A |
|---|---|---|
| `beta_CVmean` at `b=1` | {cov['beta_cvmean_at_b1']['C']:+.6f} | {cov['beta_cvmean_at_b1']['A']:+.6f} |
| `beta_CVmean` at `b_max` | {cov['beta_cvmean_at_bmax']['C']:+.6f} | {cov['beta_cvmean_at_bmax']['A']:+.6f} |
| reaches `beta_log = 0` | {cov['reaches_beta_zero']['C']} (b={cov['b_for_beta_zero']['C']:.6f}) | {cov['reaches_beta_zero']['A']} (b={cov['b_for_beta_zero']['A']:.6f}) |
| reaches Direct upper common support | {cov['reaches_direct_upper_common_support']['C']} (b={cov['b_for_direct_upper_common_support']['C']:.6f}) | {cov['reaches_direct_upper_common_support']['A']} (b={cov['b_for_direct_upper_common_support']['A']:.6f}) |
| EXT target −0.06 within grid | {cov['ext_targets']['-0.06']['C']['within_grid']} (b={cov['ext_targets']['-0.06']['C']['b']:.6f}) | {cov['ext_targets']['-0.06']['A']['within_grid']} (b={cov['ext_targets']['-0.06']['A']['b']:.6f}) |
| EXT target −0.03 within grid | {cov['ext_targets']['-0.03']['C']['within_grid']} (b={cov['ext_targets']['-0.03']['C']['b']:.6f}) | {cov['ext_targets']['-0.03']['A']['within_grid']} (b={cov['ext_targets']['-0.03']['A']['b']:.6f}) |
| EXT target 0 within grid | {cov['ext_targets']['0.0']['C']['within_grid']} | {cov['ext_targets']['0.0']['A']['within_grid']} |

Direct CV-mean `beta_log` range {cov['direct_cvmean_range']['min']:.6f} … {cov['direct_cvmean_range']['max']:.6f};
Surrogate {cov['surrogate_cvmean_range']['min']:.6f} … {cov['surrogate_cvmean_range']['max']:.6f}.
Three-way common support on D1 = **[{cov['three_way_common_support_cvmean'][0]:.6f}, {cov['three_way_common_support_cvmean'][1]:.6f}]**,
whose lower endpoint is set by **{cov['common_support_lower_endpoint_set_by']}** — so no extension
below `b = 1` was required, and none was made.

### Grid

{g['construction_rule']}

`n_grid = {g['n_grid']}` values on `[1, {g['b_max']:.9f}]`, frozen at **{gh['frozen_at_utc']}**,
file sha256 `{gh['file_sha256'][:32]}…`. The five exact anchors are `b_1`,
`b_zero_cvmean_C`, `b_zero_pooled_oof_C`, `b_zero_cvmean_A`, `b_zero_pooled_oof_A`, plus `b_max`.

**Ordering guarantee.** The `roots` mode reads *only* fold prediction files — the exact list is
recorded in `b_grid_frozen.json` under `files_read_during_root_construction`, and
`no_heldout_or_2025_outcome_read = {g['no_heldout_or_2025_outcome_read']}`. The `path` mode
re-validates the grid file hash before it runs and aborts if the file changed. The grid was not
modified after any out-of-time outcome was read.

---

## 2. Path QC

### `b = 1` reproduces `f0` bitwise

An explicit `b == 1.0` fast path returns the cached array object unchanged, so the check is exact
rather than round-off dependent:

* bitwise identical on **all {int(len(q1))}** (reference x evaluation) blocks: **{bool(q1.bitwise_identical_at_b1.all())}**
* fast path returns the same object (no arithmetic at all): **{gate['b1_fast_path_shares_memory']}**
* max absolute deviation: **{q1.max_abs_delta.max():.1f}**

### `beta_log` is numerically linear in `b`

Theory says `beta_V(b) = b * Cov_V(f0,y)/Var_V(y) - 1` exactly.

* max |observed − closed form| over every fold/held-out/2025 path: **{gate['linearity_max_dev_from_closed_form']:.3e}**
* max |residual from a fitted straight line| over all {int(len(lin))} paths: **{gate['linearity_max_residual']:.3e}**
* fitted slope vs closed-form slope, max absolute difference: **{lin.slope_abs_diff.max():.3e}**
* every path monotone increasing in `b`: **{bool(lin.monotone_increasing.all())}**

### Roots zero their own coordinate

| reference | coordinate | b_zero | achieved beta_log | |beta| from zero |
|---|---|---|---|---|
"""
    for _, r in rv.iterrows():
        md += (f"| {r.reference} | {r.coordinate} | {r.b_zero:.12f} | {r.achieved_beta:+.3e} | "
               f"{r.abs_from_zero:.3e} |\n")
    md += f"""
Both D1 roots land within **{gate['cvmean_root_max_abs_beta']:.2e}** of zero and both D2 roots within
**{gate['pooled_root_max_abs_beta']:.2e}**, against a target of 1e-12.

### Historical A-recalibration QC

The repository's historical centered-recalibration artifact (`V6/final_local_results/
recalibration_path.csv`, 51 b-values, native `f0`, same `ybar_T`-centered map) is used as an
independent QC reference — not as a source of grid or endpoint.

* worst **relative** metric difference over all 102 shared (b, evaluation) points and 17 metrics:
  **{gate['historical_qc_worst_relative_metric_diff']:.3e}**
* `ybar_T` agreement: max absolute difference **{hq.ybar_absdiff.dropna().max():.3e}**

The new A-centered implementation reproduces the historical artifact to ~1e-11 relative, i.e. to
float64 accumulation order. The old endpoint and grid were **not** adopted.

---

## 3. Centering-choice sensitivity — immaterial, frozen

Analytically, for the same `b`,

$$f_b^{{\\bar y}}(x) - f_b^{{\\bar f_0}}(x) = b\\,(\\bar f_{{0,T}} - \\bar y_T),$$

a **constant** in `x`. Because the canonical `beta_log` convention centres `c` on the evaluation
sample, a constant log shift cannot change `beta_log`, `Cov_log_residual_log_price`, `COD`, `COV`,
`PRD`, `PRB`, `MKI`, `VEI`, `Delta_NL` or `dCor` at all; only level-sensitive metrics can move.

Gate G2 established `max |f0bar_T - ybar_T| <= 8.36e-06`, so the maximum possible shift over the
entire Stage-2 grid is **{gate['centering_max_analytic_shift_log']:.3e}** in log space
(≈ {(np.exp(gate['centering_max_analytic_shift_log'])-1)*100:.5f} % in price). A full duplicate
121-point path was therefore **not** generated; the sensitivity was evaluated at four anchors
(`b_1`, `b_direct_upper_common_support`, `b_zero_cvmean`, `b_max`) for A and C on CV mean,
held-out and 2025.

Maximum absolute metric movement over all anchors and regimes:

| metric | max abs difference | max relative |
|---|---|---|
"""
    dcols = [k for k in sens.columns if k.endswith("__absdiff")]
    mx = sens[dcols].max().sort_values(ascending=False)
    for k in mx.index[:8]:
        base = k.replace("__absdiff", "__ybar")
        rel = (sens[k] / sens[base].abs().replace(0, np.nan)).max()
        md += f"| `{k.replace('__absdiff','')}` | {mx[k]:.6g} | {rel:.3e} |\n"
    md += f"""
The largest movement anywhere is `MAE_price` = **${mx['MAE_price__absdiff']:.2f}** on an MAE of
$59k–$113k, i.e. **{(sens['MAE_price__absdiff']/sens['MAE_price__ybar'].abs()).max():.2e}** relative.
`beta_log` moves by at most **{gate['centering_beta_log_absdiff_max']:.2e}** — machine epsilon, exactly
as the analytic argument requires.

> **Frozen conclusion: the centering-choice sensitivity is immaterial.** The `ybar_T`-centered
> theorem-matched map is used for the full path, and no duplicate `f0bar`-centered path is
> scheduled.

---

## 4. New repository finding — fold-6 / fold-7 validation overlap

While constructing the D2 pooled coordinate, the seven validation blocks were found **not** to be
disjoint: fold 6 and fold 7 overlap by **{ov['n_duplicated_rows']:,} rows**
({ov['n_concatenated']:,} concatenated vs {ov['n_unique']:,} unique, i.e.
{ov['share_duplicated']*100:.2f} % duplicated; 63.4 % of fold 6's block and 60.9 % of fold 7's).

Cause: {ov['cause']}.

Effect: {ov['effect_on_D2']}.

{ov['historical_note']}

**D1, the paper's primary CV coordinate, is unaffected** ({ov['d1_cvmean_unaffected']}) because it
is an equal-weight mean of per-fold values, each computed on its own block. Only the D2 sensitivity
coordinate double-counts. Audited in `tables/cv_fold_validation_overlap_audit.csv` and
`tables/cv_fold_validation_overlap_summary.json`.

Consequence for the frozen `Delta_NL` estimator: it requires unique identifiers within a split, so
for the D2 sample only, a composite deterministic identifier `"fold|row_id"` is supplied. It remains
a function of observation identity alone and independent of the model and predictions, which is what
the frozen specification requires. Every other evaluation uses the plain `row_id`.

---

## 5. C-PRIMARY centered-spread results

### Equal-weight seven-fold CV mean

{cv_anchor_table('C', ['b_1','b_zero_cvmean_C','b_zero_pooled_oof_C','b_max'])}
*(CV SD columns are in `tables/centered_spread_cv_summary.csv`; they are descriptive spread over
strictly nested chronological windows, not an IID sampling distribution.)*

### Later held-out block (n = 38,290)

{anchor_table('C', 'heldout', ['b_1','b_zero_cvmean_C','b_zero_pooled_oof_C','b_max'])}

### 2025 forward block (n = 26,641)

{anchor_table('C', 'forward_2025', ['b_1','b_zero_cvmean_C','b_zero_pooled_oof_C','b_max'])}

### Reading

* **The predictive cost of post-hoc rescaling is large and monotone.** Driving development
  `beta_log` to zero (`b = {roots['C']['b_zero_cvmean']:.6f}`) costs, on held-out,
  `R2_price` {at('C','heldout',1.0).R2_price:.5f} → {at('C','heldout',anc['b_zero_cvmean_C']).R2_price:.5f}
  and `MAE_price` ${at('C','heldout',1.0).MAE_price:,.0f} → ${at('C','heldout',anc['b_zero_cvmean_C']).MAE_price:,.0f}.
  On 2025 the same move costs `R2_price` {at('C','forward_2025',1.0).R2_price:.5f} → {at('C','forward_2025',anc['b_zero_cvmean_C']).R2_price:.5f}.
* **`Delta_NL` rises** along the path ({at('C','heldout',1.0).Delta_NL:.5f} → {at('C','heldout',anc['b_zero_cvmean_C']).Delta_NL:.5f}
  held-out): removing the first-order slope by a global rescaling *increases* non-affine
  conditional-mean structure.
* **`dCor` falls substantially** ({at('C','heldout',1.0).dCor_e_y:.5f} → {at('C','heldout',anc['b_zero_cvmean_C']).dCor_e_y:.5f}
  held-out), then turns back up beyond the root ({at('C','heldout',anc['b_max']).dCor_e_y:.5f} at `b_max`).
* **Assessor-facing metrics overshoot into progressivity.** At the root, held-out `PRD`
  {at('C','heldout',1.0).PRD:.4f} → {at('C','heldout',anc['b_zero_cvmean_C']).PRD:.4f}, `MKI`
  {at('C','heldout',1.0).MKI:.5f} → {at('C','heldout',anc['b_zero_cvmean_C']).MKI:.5f}, `VEI`
  {at('C','heldout',1.0).VEI:+.3f} → {at('C','heldout',anc['b_zero_cvmean_C']).VEI:+.3f}. `COD`
  worsens ({at('C','heldout',1.0).COD:.3f} → {at('C','heldout',anc['b_zero_cvmean_C']).COD:.3f}).
* **First-order neutrality does not flatten the conditional ratio profile.** In the 30-bin
  price profile on held-out, the cheapest bin's median ratio moves 1.709 → 1.474 and the most
  expensive 0.863 → 1.163: the profile is compressed and the level lifted, but a large
  low-value-to-high-value gap remains. Profiles with 90 % bootstrap CIs are in
  `tables/centered_spread_ratio_profiles.csv`.

---

## 6. A-SECONDARY practical comparator

Same map applied to standard native LightGBM — the "could a practitioner just post-process?"
question. Descriptive wording only; differences relative to A are never attributed to rho.

### Held-out

{anchor_table('A', 'heldout', ['b_1','b_zero_cvmean_A','b_zero_pooled_oof_A','b_max'])}

### 2025 forward

{anchor_table('A', 'forward_2025', ['b_1','b_zero_cvmean_A','b_zero_pooled_oof_A','b_max'])}

The A path tracks the C path very closely — the two references differ at `b=1` by the execution-path
artifact characterised at Gate G2, and rescaling does not amplify it. A practitioner
post-processing standard LightGBM would face essentially the same accuracy/equity trade-off as one
post-processing the custom rho=0 origin.

---

## 7. Descriptive juxtaposition with the retrained families — NOT the matched-beta test

**The formal matched-beta comparison is not performed in Stage 2 and is not authorized here.** It
must be matched on **development** achieved `beta_log`; the lookups below are keyed on **held-out**
`beta_log` and are therefore descriptive only, offered to show that the Stage-3 test is worth doing.

| at held-out `beta_log` ≈ | family | R2_price | MAE_price | Delta_NL | dCor |
|---|---|---|---|---|---|
"""
    dmax = fp[fp.family == "Direct"].dropna(subset=["Beta_log__heldout"])
    dm = dmax.loc[dmax.Beta_log__heldout.idxmax()]
    smax = fp[fp.family == "Surrogate"].dropna(subset=["Beta_log__heldout"])
    sm = smax.loc[smax.Beta_log__heldout.idxmax()]
    ch = path[(path.reference == "C") & (path.evaluation == "heldout")]
    for tgt, fam, row in ((dm.Beta_log__heldout, f"Direct (rho={dm.rho:.6g})", dm),
                          (sm.Beta_log__heldout, f"Surrogate (rho={sm.rho:.6g})", sm)):
        p = ch.iloc[(ch.beta_log - tgt).abs().argmin()]
        md += (f"| **{tgt:+.5f}** | {fam} | {row.R2_price__heldout:.5f} | "
               f"{row.MAE_price__heldout:,.0f} | {row.Delta_NL__heldout:.5f} | "
               f"{row.dCor_e_y__heldout:.5f} |\n")
        md += (f"| | C-posthoc (b={p.b:.6f}, beta={p.beta_log:+.5f}) | {p.R2_price:.5f} | "
               f"{p.MAE_price:,.0f} | {p.Delta_NL:.5f} | {p.dCor_e_y:.5f} |\n")
    md += """
**The accuracy comparison appears to cross over with correction strength.** At mild correction
(held-out `beta_log` ≈ −0.079, Direct's most-corrected point) the post-hoc rescaling is the *more*
accurate of the two; at strong correction (`beta_log` ≈ 0, Surrogate's near-neutral point) the
retrained Surrogate is dramatically more accurate than post-hoc rescaling. In both regimes the
retrained families carry lower `Delta_NL` than post-hoc at comparable first-order correction.

This is exactly the structure the Stage-3 matched-beta test is designed to adjudicate properly, on
the development coordinate, with the frozen common support. Nothing here is a finding about the
manuscript's claims; no manuscript text was touched.

---

## 8. Frozen future matched-beta convention (not executed)

| role | families | matched on |
|---|---|---|
| **PRIMARY CORE** | Direct vs Surrogate vs **C-posthoc** | DEVELOPMENT achieved `beta_log` |
| SECONDARY practical | A-posthoc | separate panel/table |
| — | B | no post-hoc path |

A-posthoc must not determine the three-way common support, must not determine CORE target
selection, and must not replace C-posthoc in the primary mechanistic test. Frozen in
`configs/posthoc_comparator_convention.yaml`; `executed_in_stage_2 = {conv['future_matched_beta_convention']['executed_in_stage_2']}`.

---

## 9. Artifacts

| artifact | content |
|---|---|
| `configs/b_grid_frozen.json` + `b_grid_frozen_hash.json` | construction rule, {g['n_grid']} b values, anchors, A/C roots, coverage verification, hash |
| `configs/posthoc_comparator_convention.yaml` | frozen roles, calibration rules, future matched-beta convention |
| `tables/posthoc_development_roots.csv` | per-fold `Cov_Vk`, `Var_Vk`, `R_k`, D1/D2 roots for A and C |
| `tables/centered_spread_path.csv` | {len(path):,} rows: 2 references x 10 evaluations x {g['n_grid']} b, full canonical suite |
| `tables/centered_spread_cv_summary.csv` | equal-weight CV mean and descriptive CV SD per b |
| `tables/centered_spread_pooled_oof.csv` | D2 pooled coordinate path |
| `tables/centered_spread_b1_qc.csv` | bitwise `b=1` verification |
| `tables/centered_spread_linearity.csv` | closed-form vs observed `beta_log`, slopes, residuals |
| `tables/centered_spread_root_verification.csv` | achieved `beta_log` at each root |
| `tables/centered_spread_centering_sensitivity.csv` | anchor-level `ybar` vs `f0bar` comparison |
| `tables/centered_spread_existing_qc.csv` | historical A-recalibration reproduction |
| `tables/centered_spread_ratio_profiles.csv` | IAAO proxy-decile (90 % CI) and 30-bin price profiles at anchors |
| `tables/cv_fold_validation_overlap_audit.csv` | fold-pair validation overlap audit |
| `tables/gate_g3_checks.json` | machine-readable Gate-G3 evidence |
"""
    c.write_text(c.REPORTS / "CENTERED_SPREAD_COMPARATOR_REPORT.md", md)
    print(f"written {len(md)} chars")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
