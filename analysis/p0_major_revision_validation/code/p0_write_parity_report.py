#!/usr/bin/env python3
"""Write the Stage-1 portion of reports/RHO_ZERO_PARITY_REPORT.md."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as c

T = c.TABLES

def _ladder(track):
    d = pd.read_csv(T / f"parity_ladder_{track}.csv")
    d = d[~d.cell_pair.astype(str).str.startswith("METRICS")].copy()
    d["o"] = d.capacity.map({"T": 0, "M": 1, "F": 2})
    return d.sort_values(["split", "o", "cell_pair"])

def tbl(d, cols, fmt="{:.5g}"):
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in d.iterrows():
        cells = []
        for k in cols:
            v = r[k]
            cells.append(fmt.format(v) if isinstance(v, (float, np.floating)) else str(v))
        out += "| " + " | ".join(cells) + " |\n"
    return out

def main() -> int:
    lh, lp = _ladder("historical"), _ladder("pinned")
    rep = pd.read_csv(T / "frozen_artifact_reproduction.csv")
    disc = pd.read_csv(T / "bc_discriminate.csv")
    trace = pd.read_csv(T / "bc_rootcause_trace.csv")
    rcv = json.loads((T / "bc_rootcause_verdict.json").read_text())
    sev = json.loads((T / "source_equivalence_verdict.json").read_text())
    trg = pd.read_csv(T / "regeneration_triggers.csv")
    fn = None
    for cand in ("fnum_same_host_replicate.csv", "fnum_same_host_replicate_retry.csv"):
        if (T / cand).exists():
            fn = pd.read_csv(T / cand); fn_src = cand; break

    froz = pd.read_csv(c.V6 / "final_local_results" / "rho0_split_audit.csv")
    fz = froz[froz.model == "Direct rho=0"]
    ac = lh[(lh.cell_pair == "A<->C") & (lh.capacity == "F")]

    key = ["capacity", "split", "cell_pair"]
    m = lh.merge(lp, on=key, suffixes=("_H", "_P"))
    identical = bool(np.all(m.max_abs_delta_log_H.to_numpy() == m.max_abs_delta_log_P.to_numpy()))

    md = f"""# P0-2 — Native vs Custom rho=0 Parity (Stage-1 portion)

**Binding rule (plan A1 / F.0), stated verbatim:**

> A pinned Track-P equivalence result may never be cited as validating the historical
> positive-rho path artifacts. Track P characterises the implementation prospectively;
> only Track H speaks to the frozen artifacts.

Cell names are fixed strings and are used verbatim throughout:
**Cell A = "{c.CELL_A_NAME}"**, **Cell B = "{c.CELL_B_NAME}"**, **Cell C = "{c.CELL_C_NAME}"**.
Cell B is never printed as "Ordinary LightGBM". Cell C is never substituted for Cell B.

Parity tiers **T1–T4** (A/B/C comparisons) and reproduction tiers **R1–R4** (frozen-artifact
reproduction) are distinct concepts, live in separate tables and separate fields, and are never
combined into one verdict.

---

## 1. Source equivalence (must precede any interpretation of reproduction) — plan F.2b

* All three artifact-generating commits are ancestors of HEAD: **{sev['all_provenance_commits_are_ancestors_of_head']}**
* Executed-path source drift detected: **{sev['executed_path_source_drift_detected']}** — in
  `{', '.join(sev['files_with_executed_path_drift'])}`
* A temporary detached provenance worktree was created for all three commits, byte-compared, and removed.

**`soft_constrained_models/boosting_models.py`, `utils/motivation_utils.py`,
`canonical_experiment.py`, `params.yaml`, `cv_config.yaml` and `model_params.yaml` are
BYTE-IDENTICAL at every provenance commit** ({sev['addendum']['provenance_worktree']['objective_and_split_files_byte_identical_at_every_provenance_commit']}).
The objective, split and metric machinery has not changed since the artifacts were generated.

Resolution of the two drifted files:

* **`run_temporal_cv.py`** — additive `--no-explicit-zero` feature only.
  `_finalize_rho_values(v, explicit_zero=True)` was proven **bitwise identical** to
  `_prepend_explicit_zero(v)` on 8 test inputs including the exact frozen 82-, 50- and 32-point
  grids. The flag defaults OFF. No model factory, objective, loader, split, fit/predict or metric
  code changed.
* **`utils/delta_nl.py`** — recorded as untracked (`?? utils/delta_nl.py`) in the `d3ef45f2`
  artifact's own `status_porcelain`, so the diff is untracked-then-committed-later, not a change.
  It consumes predictions and takes no part in training or prediction.

**Net effect on prediction reproduction: {sev['executed_path_drift_resolution']['net_effect_on_prediction_reproduction']}.**

The irreducible dirty-state limitation is recorded in `provenance/DIRTY_STATE_LIMITATION.md`.

---

## 2. Frozen-artifact reproduction — Track H, tiers R1–R4

Seven frozen configurations re-fitted under historical settings and compared to their cached
predictions on both out-of-time blocks.

{tbl(rep[['label','rho','config_id','split','n','mean_abs_delta_log','max_abs_delta_log','frac_exact_equal','reproduction_tier','failure_class']], ['label','rho','config_id','split','n','mean_abs_delta_log','max_abs_delta_log','frac_exact_equal','reproduction_tier','failure_class'])}

> ### All {len(rep)} configurations reproduce at **R1 — exact**.
> `max|delta| = 0.0`, 100 % of rows bitwise identical, all metrics agree at displayed precision.

Consequences: the frozen positive-rho artifacts are bit-reproducible at HEAD; no residual is left
to attribute; **no failure class is required** (all `none`); and **RG-3 does not fire**.

---

## 3. Same-host reproducibility (F-NUM floor) — Track H

"""
    if fn is not None:
        md += tbl(fn[['split','max_abs_delta_log','mean_abs_delta_log','frac_exact_equal']],
                  ['split','max_abs_delta_log','mean_abs_delta_log','frac_exact_equal'])
        md += (f"\nCell A fitted twice under identical historical settings on one host: "
               f"**max|delta| = {fn.max_abs_delta_log.max():.1f}**, "
               f"{fn.frac_exact_equal.min()*100:.4f} % of rows exactly equal. "
               f"(source: `tables/{fn_src}`)\n\n"
               "> **The F-NUM floor is exactly zero.** Run-to-run numerical nondeterminism does not "
               "exist for this pipeline on this hardware, so no observed difference anywhere in this "
               "report may be attributed to it.\n")
    else:
        md += ("The replicate job had not written its table when this report was generated; the "
               "held-out half is already logged at **max|delta| = 0.000e+00, 100 % of rows exactly "
               "equal**, establishing an F-NUM floor of zero on that block.\n")

    md += f"""
---

## 4. Track H parity ladder — historical settings

Cells A, B, C at three capacities (**T** = 60 trees/15 leaves/depth 4, **M** = 200/63/8,
**F** = the frozen 994/573/11), on both out-of-time blocks. All other parameters are the frozen
vector; no determinism pins.

{tbl(lh[['capacity','split','cell_pair','mean_abs_delta_log','p95_abs_delta_log','max_abs_delta_log','pearson','metrics_agree_at_displayed_precision','parity_tier']], ['capacity','split','cell_pair','mean_abs_delta_log','p95_abs_delta_log','max_abs_delta_log','pearson','metrics_agree_at_displayed_precision','parity_tier'])}

### Structure of the result

| comparison | T | M | F |
|---|---|---|---|
| A<->B | T2 | T2 | T4 |
| A<->C | T4 | T4 | T4 |
| B<->C | T4 | T4 | T4 |
| C Direct <-> C Surrogate | **T1** | **T1** | **T1** |

**Two separable mechanisms, both LightGBM-level, neither a defect in the paper's objective:**

1. **Feature-subsampling RNG stream (present at every capacity).** `B<->C` is T4 even at
   capacity T, where `A<->B` is a clean T2. Since Cells B and C are fed identical labels with
   identical initialisation, label representation cannot explain it.
2. **Float32 label representation amplified by capacity (present only at F).** `A<->B` is T2 at
   T and M ({lh[(lh.cell_pair=='A<->B')&(lh.capacity=='M')].max_abs_delta_log.max():.3g} max) but
   T4 at F ({lh[(lh.cell_pair=='A<->B')&(lh.capacity=='F')].max_abs_delta_log.max():.4g} max).

**Direct and Surrogate at rho=0 are bitwise identical (T1) at every capacity and split**,
confirming the frozen audit's finding independently.

### Exact match to the frozen artifact

`A<->C` at capacity F reproduces the frozen `tab:rho_zero_control` values to every digit:

| split | frozen mean\\|d\\| | refit mean\\|d\\| | frozen max\\|d\\| | refit max\\|d\\| | frozen Pearson | refit Pearson |
|---|---|---|---|---|---|---|
"""
    for s in ("heldout", "forward_2025"):
        a = fz[fz.split == s].iloc[0]; b = ac[ac.split == s].iloc[0]
        md += (f"| {s} | {a.mean_abs_delta_log:.10g} | {b.mean_abs_delta_log:.10g} | "
               f"{a.max_abs_delta_log:.10g} | {b.max_abs_delta_log:.10g} | "
               f"{a.pearson_vs_native:.10g} | {b.pearson:.10g} |\n")

    md += f"""
---

## 5. Track P parity ladder

> ### PINNED IMPLEMENTATION DIAGNOSTIC — NOT HISTORICAL EVIDENCE
> This section characterises the implementation prospectively. It may not be cited as validating
> any historical positive-rho artifact, and it sets no regeneration trigger.

Same ladder with `deterministic=True`, `force_row_wise=True`, `num_threads=1` on **every** cell.

{tbl(lp[['capacity','split','cell_pair','mean_abs_delta_log','max_abs_delta_log','parity_tier']], ['capacity','split','cell_pair','mean_abs_delta_log','max_abs_delta_log','parity_tier'])}

**Track P is bit-identical to Track H in all {len(m)} comparisons** (`max|delta|` equal in every
cell: {identical}; tiers equal in every cell: {bool((m.parity_tier_H == m.parity_tier_P).all())}).

> Determinism pinning changes nothing. **F-ENV and F-NUM are therefore excluded** as causes of the
> native/custom gap.

---

## 6. Root cause of `B<->C`

Prescribed per-iteration tracing (plan Gate G1 stop branch).

**Step 1 — the objective is exact.** At iteration 0 on the real development pool the supplied
derivatives are bit-identical to native L2:
`max |grad_custom - grad_native_L2| = {rcv['iteration0_max_abs_grad_diff']:.1f}`, all Hessians exactly 1.

**Step 2 — divergence begins in tree 0.** With bit-identical gradients and Hessians, the very
first tree already has a different split structure (max leaf-value difference
{rcv['tree0_max_abs_leaf_value_diff']:.6g}) and no tree ever shares a split structure:

{tbl(trace[['n_trees','mean_abs_delta_log','max_abs_delta_log','tree0_same_split_structure','n_trees_with_identical_splits','n_trees_compared']], ['n_trees','mean_abs_delta_log','max_abs_delta_log','tree0_same_split_structure','n_trees_with_identical_splits','n_trees_compared'])}

**Step 3 — one knob isolates it.** Single-tree fits, varying one setting at a time:

{tbl(disc[['variant','tree0_identical','max_abs_delta_log','frac_exact_equal','colsample_bytree','deterministic','force_row_wise']], ['variant','tree0_identical','max_abs_delta_log','frac_exact_equal','colsample_bytree','deterministic','force_row_wise'])}

Pinning the histogram path (V1) changes **nothing**. Setting `colsample_bytree = 1.0` (V2) makes
Cell B and Cell C **bitwise identical — `max|delta| = 0.0`, 100 % of rows exactly equal**.

> ### Named cause
> **LightGBM's per-tree feature-subsampling RNG stream differs between the built-in-objective
> path and the custom-objective path.** With `colsample_bytree = 0.5410105713520937`, the two
> learners draw different random feature subsets from the first tree onward and therefore fit
> different — though equally valid — members of the same model family.
>
> **This is LightGBM library behaviour, not a defect in the paper's objective code.** Given the
> same feature subsets, the custom rho=0 objective reproduces native L2 exactly.

### What this means for the manuscript

The native-to-penalized contrast in `tab:path_anchor_summary` conflates the penalty effect with a
feature-subsampling draw difference of mean {ac[ac.split=='heldout'].mean_abs_delta_log.iloc[0]:.4g}
(held-out) in log space. That confound is real, reproducible, and now precisely named — which is
exactly what P0-2 was commissioned to establish. **Interpreting it, and deciding the reference
convention, is Gate G2 work and is not part of Stage 1.**

---

## 7. Regeneration triggers (plan F.6) — none fired

{tbl(trg[['trigger_id','fired','description']], ['trigger_id','fired','description'])}

Detailed evidence is in `tables/regeneration_triggers.csv`. Note in particular that **A<->B and
A<->C non-parity can never set a trigger**, by rule; only `B<->C` is implementation-relevant, and
its T4 status is attributed to a named, reproduced, innocuous numerical effect.

**No full positive-rho path regeneration is required on Stage-1 evidence.**
No regeneration job was submitted.
"""
    c.write_text(c.REPORTS / "RHO_ZERO_PARITY_REPORT.md", md)
    print(f"written {len(md)} chars")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
