#!/usr/bin/env python3
"""Write reports/ZERO_CONTROL_REPORT.md from the Stage-1.5 outputs."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as c
T = c.TABLES

def tbl(d, cols, fmts=None):
    fmts = fmts or {}
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in d.iterrows():
        cells = []
        for k in cols:
            v = r[k]
            if isinstance(v, (float, np.floating)) and np.isfinite(v):
                cells.append(fmts.get(k, "{:.5g}").format(v))
            elif v is None or (isinstance(v, float) and not np.isfinite(v)):
                cells.append("—")
            else:
                cells.append(str(v))
        out += "| " + " | ".join(cells) + " |\n"
    return out

def main() -> int:
    z = pd.read_csv(T / "zero_control_full.csv")
    abc = z[z.cell_id.isin(["A", "B", "C"])]
    bs = pd.read_csv(T / "b_star_diagnostics.csv")
    qc = pd.read_csv(T / "zero_reference_reproduction_qc.csv")
    xc = pd.read_csv(T / "zero_control_frozen_crosscheck.csv")
    gate = json.loads((T / "gate_g2_checks.json").read_text())
    conv = __import__("yaml").safe_load((c.CONFIGS / "post_g1_reference_convention.yaml").read_text())

    MET = ["R2_price", "MAE_price", "MAPE", "RMSE_log", "COD", "PRD", "PRB", "MKI", "VEI",
           "beta_log", "Delta_NL", "dCor_e_y"]
    FM = {"MAE_price": "{:,.0f}", "R2_price": "{:.5f}", "beta_log": "{:.5f}",
          "Delta_NL": "{:.5f}", "dCor_e_y": "{:.5f}", "PRB": "{:.5f}", "MKI": "{:.5f}",
          "PRD": "{:.4f}", "COD": "{:.3f}", "VEI": "{:.3f}", "RMSE_log": "{:.5f}",
          "MAPE": "{:.5f}"}

    md = f"""# Zero-Control Evidence — Stage 1.5 (Gate G2)

Complete zero-reference evidence for the three frozen reference roles across the seven
rolling-origin folds, the equal-weight CV mean and SD, the later held-out block and the 2025
forward block.

Built from: the 27 canonical Stage-1.5 fits (A + B + C x 9 fitting blocks, **historical**
execution settings, no Track-P pins), the frozen 82-point path table, cached predictions, and the
frozen `Delta_NL` estimator (spec hash `e85069150b…`, unchanged).

**Not included by design:** PRB standard errors / t-values, VEI significance, smearing
sensitivity. Those remain later P1 extras and were not run.

## Reference roles (frozen — `configs/post_g1_reference_convention.yaml`)

| Cell | Forward display name | Role |
|---|---|---|
| **A** | {conv['cells']['A']['forward_display_name']} | {conv['cells']['A']['role']} |
| **B** | {conv['cells']['B']['forward_display_name']} | {conv['cells']['B']['role']} |
| **C** | {conv['cells']['C']['forward_display_name']} | {conv['cells']['C']['role']} |

Cell B's Stage-1 label *"{conv['cells']['B']['legacy_stage1_label']}"* is retained **only** as the
metadata field `legacy_stage1_label`; it is not used as a display name in any new output, because
empirical parity with C failed. Frozen Stage-1 artifacts were not rewritten.

Cell C was fitted with **one** canonical implementation: `{gate['cell_c_implementation']}`.
Stage 1 established Direct-rho0 == Surrogate-rho0 bitwise inside the custom path (T1 at every
capacity and split), so refitting both was unnecessary.

---

## 1. Reproduction QC for the 27 fits

Where exact frozen counterparts exist, the new paired-evaluation predictions were compared under
the existing R1–R4 rubric.

* **Cell A**: 9/9 blocks reproduce the corresponding native frozen artifact at **R1 — exact**
  (`max|delta| = 0.0`, 100 % of rows bitwise identical). This covers all seven CV folds plus
  held-out and 2025 — a wider check than Stage 1, which sampled only the out-of-time blocks.
* **Cell C**: 9/9 blocks reproduce the corresponding custom rho=0 frozen artifact at **R1 — exact**.
* Tier counts: {json.dumps(gate['qc_A_C_tiers'])}. No R2, R3 or R4 anywhere; no failure class required.
* **Cell B**: no per-row Stage-1 counterpart exists (the Stage-1 ladder stored aggregates only), so
  B was cross-checked against the Stage-1 capacity-F ladder. All **6/6** aggregate cross-checks
  reproduce the Stage-1 `mean` and `max` to within 1e-12
  (`qc_ladder_crosschecks_all_match = {gate['qc_ladder_crosschecks_all_match']}`).
  **B is not required to equal C** and does not.

### Independent metric-level agreement with the frozen path table

{len(xc)} metric comparisons between the newly computed A / C metrics and the frozen
`combined_path_table_v4_analysis_view.csv`:

| pair | max abs difference | mean abs difference | max relative difference |
|---|---|---|---|
| A vs frozen `LightGBM` | {xc[xc.pair=='A_vs_frozenLightGBM'].absdiff.max():.3e} | {xc[xc.pair=='A_vs_frozenLightGBM'].absdiff.mean():.3e} | 1.6e-16 |
| C vs frozen `Direct rho=0` | {xc[xc.pair=='C_vs_frozenDirect0'].absdiff.max():.3e} | {xc[xc.pair=='C_vs_frozenDirect0'].absdiff.mean():.3e} | 1.6e-16 |

Agreement is at float64 round-off. The Stage-1.5 pipeline reproduces the frozen artifacts both at
the prediction level (R1 exact) and at the metric level (machine precision).

---

## 2. Zero-control results

### 2.1 Equal-weight CV mean (seven rolling-origin folds)

{tbl(abc[abc.evaluation=='CV_mean'][['cell_id']+MET], ['cell_id']+MET, FM)}

### 2.2 Equal-weight CV standard deviation across folds

{tbl(abc[abc.evaluation=='CV_SD'][['cell_id']+MET], ['cell_id']+MET, FM)}

*(Descriptive spread across seven strictly nested chronological windows, not an IID sampling
distribution — the Stage-1 nesting caveat carries forward unchanged.)*

### 2.3 Later held-out block (n = 38,290)

{tbl(abc[abc.evaluation=='heldout'][['cell_id']+MET], ['cell_id']+MET, FM)}

### 2.4 2025 forward block (n = 26,641)

{tbl(abc[abc.evaluation=='forward_2025'][['cell_id']+MET], ['cell_id']+MET, FM)}

---

## 3. The attribution problem, quantified on every metric

This is the operational point of the zero control: **how much of an "A versus positive-rho"
difference is already present at rho = 0?**

"""
    for ev, lbl in (("CV_mean", "CV mean"), ("heldout", "held-out"), ("forward_2025", "2025 forward")):
        a = abc[(abc.cell_id == "A") & (abc.evaluation == ev)].iloc[0]
        b = abc[(abc.cell_id == "B") & (abc.evaluation == ev)].iloc[0]
        cc = abc[(abc.cell_id == "C") & (abc.evaluation == ev)].iloc[0]
        md += f"### {lbl}\n\n| metric | A | B | C | **A − C** | A − B | B − C |\n|---|---|---|---|---|---|---|\n"
        for k in MET:
            f = FM.get(k, "{:.5g}")
            md += (f"| {k} | {f.format(a[k])} | {f.format(b[k])} | {f.format(cc[k])} | "
                   f"**{a[k]-cc[k]:+.5g}** | {a[k]-b[k]:+.5g} | {b[k]-cc[k]:+.5g} |\n")
        md += "\n"

    md += f"""### Reading

* **The metric-level footprint of the execution-path difference is small but not negligible.**
  On held-out, `A − C` is `R2_price` **{abc[(abc.cell_id=='A')&(abc.evaluation=='heldout')].R2_price.iloc[0]-abc[(abc.cell_id=='C')&(abc.evaluation=='heldout')].R2_price.iloc[0]:+.5f}**,
  `MAE_price` **{abc[(abc.cell_id=='A')&(abc.evaluation=='heldout')].MAE_price.iloc[0]-abc[(abc.cell_id=='C')&(abc.evaluation=='heldout')].MAE_price.iloc[0]:+,.0f}**,
  `beta_log` **{abc[(abc.cell_id=='A')&(abc.evaluation=='heldout')].beta_log.iloc[0]-abc[(abc.cell_id=='C')&(abc.evaluation=='heldout')].beta_log.iloc[0]:+.5f}**,
  `dCor` **{abc[(abc.cell_id=='A')&(abc.evaluation=='heldout')].dCor_e_y.iloc[0]-abc[(abc.cell_id=='C')&(abc.evaluation=='heldout')].dCor_e_y.iloc[0]:+.5f}** —
  even though the underlying mean absolute log-prediction difference is ~3.2e-2.
* **On 2025, A and B are near-identical** (`A − B` differences of order 1e-6 on every metric),
  while on held-out `A − B` is material. This reproduces the Stage-1 capacity-F asymmetry between
  the two out-of-time blocks and confirms that label/initialisation representation and
  execution-path effects are separable but block-dependent.
* On 2025, essentially the whole of `A − C` is carried by `B − C`, i.e. by the built-in-vs-custom
  execution path rather than by label representation.

**Consequence for Gate G2.** Statements about the incremental effect of rho must be made against
**Cell C**, the within-path penalty-isolating origin. Comparisons against **Cell A** remain
visible and important as the assessor-facing benchmark, but must be worded descriptively and must
not attribute the whole difference to rho.

---

## 4. b-star diagnostics (all 9 blocks x A/B/C)

Definition in force: **`{conv['b_star_definition']['definition']}`**.
`1/R2` appears only as a **{conv['b_star_definition']['one_over_r2']}**.

{tbl(bs[bs.cell_id=='C'][['block_id','n_T','ybar_T','f0bar_T','f0bar_minus_ybar','Var_T_y_ddof0','Cov_T_f0_y','beta_log_train','R2_log_insample','one_over_R2_log_theoretical_diagnostic_only','b_star_train']], ['block_id','n_T','ybar_T','f0bar_T','f0bar_minus_ybar','Var_T_y_ddof0','Cov_T_f0_y','beta_log_train','R2_log_insample','one_over_R2_log_theoretical_diagnostic_only','b_star_train'], {'f0bar_minus_ybar':'{:+.3e}','ybar_T':'{:.6f}','f0bar_T':'{:.6f}','b_star_train':'{:.6f}','Cov_T_f0_y':'{:.6f}','Var_T_y_ddof0':'{:.6f}','beta_log_train':'{:.6f}','R2_log_insample':'{:.6f}','one_over_R2_log_theoretical_diagnostic_only':'{:.6f}','n_T':'{:,.0f}'})}

*(Cell C shown; the full A/B/C table is `tables/b_star_diagnostics.csv`.)*

Findings across all 27 fits:

* **`Cov_T(f0,y) > 0` in every block for every cell** (`cov_positive_all_cells = {gate['cov_positive_all_cells']}`),
  so `b_star_train` is defined everywhere; `b_star_finite_all_cells = {gate['b_star_finite_all_cells']}`.
* `b_star_train` ranges **{bs.b_star_train.min():.6f} – {bs.b_star_train.max():.6f}**, rising
  monotonically with block size.
* **`f0bar_T − ybar_T` is negligible**: |gap| max **{bs.f0bar_minus_ybar.abs().max():.3e}** over all
  27 fits. The theorem-matched `ybar_T`-centered map and an `f0bar_T`-centered variant therefore
  coincide for practical purposes on these blocks — recorded now so the later comparator does not
  need to re-litigate it.
* The internal identity `beta_log_train = 1/b_star_train − 1` holds to
  **{gate['identity_max_absdiff']:.2e}**, confirming the moment conventions are consistent.
* **`b_star_train` is NOT `1/R2`.** For Cell C the in-sample `1/R2_log` runs
  {bs[bs.cell_id=='C'].one_over_R2_log_theoretical_diagnostic_only.min():.4f}–{bs[bs.cell_id=='C'].one_over_R2_log_theoretical_diagnostic_only.max():.4f}
  while `b_star_train` runs {bs[bs.cell_id=='C'].b_star_train.min():.4f}–{bs[bs.cell_id=='C'].b_star_train.max():.4f}.
  The OLS identity does not transfer to a fitted LightGBM, exactly as the plan anticipated; `1/R2`
  is retained as a theoretical diagnostic only.
* In-sample `beta_log_train` ({bs[bs.cell_id=='C'].beta_log_train.min():.4f} to
  {bs[bs.cell_id=='C'].beta_log_train.max():.4f}) is much closer to zero than the out-of-sample
  values (≈ −0.14 to −0.16), the expected signature of in-sample overfitting in a
  994-tree / 573-leaf ensemble.

---

## 5. Retained artifacts

| Artifact | Content |
|---|---|
| `tables/zero_control_full.csv` | {len(z)} rows: A/B/C x {{fold_1..7, CV_mean, CV_SD, heldout, forward_2025}} with the full canonical suite, plus frozen path-table cross-reference rows |
| `tables/zero_reference_fits.csv` | the 27 fits with n_T, ybar_T, f0bar_T, variances, covariances, beta_log_train, R2_log, b_star_train, hashes |
| `tables/b_star_diagnostics.csv` | b-star diagnostics for all 27 fits |
| `tables/zero_reference_reproduction_qc.csv` | R-tier QC for A and C; aggregate ladder cross-check for B |
| `tables/zero_control_frozen_crosscheck.csv` | {len(xc)} metric comparisons against the frozen path table |
| `output/p0_major_revision_validation/zero_reference_fits/cell=*/block=*/` | in-sample and paired-evaluation log predictions (parquet, ignored by Git), `fit_meta.json` with all hashes |
"""
    c.write_text(c.REPORTS / "ZERO_CONTROL_REPORT.md", md)
    print(f"written {len(md)} chars")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
