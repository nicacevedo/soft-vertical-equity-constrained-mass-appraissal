#!/usr/bin/env python3
"""P1 Task 2: observation-level PRB inference (IAAO 2013 p.19 structure).

utils/motivation_utils.prb returns a bare np.polyfit slope with no standard error.
This module adds the inferential structure the 2013 Standard ties PRB to -- standard
error, t, p, and a 95% confidence interval -- computed at the OBSERVATION level.

The seven CV folds are NEVER treated as IID replicates.  No mean +/- SD/sqrt(7)
appears anywhere in the output.  Fold rows are descriptive chronological-window
values, each carrying its own observation-level standard error.

Classification follows the 2013 reading that the ENTIRE confidence interval must lie
outside a band before that band is deemed exceeded; a CI that merely crosses a
threshold is labelled as overlapping, never as exceeding.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1_common as c                                                   # noqa: E402

CI_LEVEL = 0.95
BAND_05 = 0.05
BAND_10 = 0.10


# ---------------------------------------------------------------- PRB design
def prb_design(assessed: np.ndarray, sale_price: np.ndarray):
    """Reproduce utils.motivation_utils.prb's preprocessing EXACTLY.

    prb() is the function that produced every frozen PRB value, so its conventions
    are binding here: the median ratio is taken on the UNFILTERED ratio vector, the
    only mask is inner_term > 0, and the regressor is the UNCENTERED log2 of the
    equal-weight value proxy.  (iaao_prb() masks av>0 & sp>0 first and centers; the
    slope is identical but the retained sample can differ.)
    """
    assessed = np.asarray(assessed, dtype=float)
    sale_price = np.asarray(sale_price, dtype=float)
    ok = np.isfinite(assessed) & np.isfinite(sale_price)          # prb(..., na_rm=True)
    assessed, sale_price = assessed[ok], sale_price[ok]
    ratio = assessed / sale_price
    med = float(np.median(ratio))
    if med == 0 or not np.isfinite(med):
        return None
    lhs = (ratio - med) / med
    inner = ((assessed / med) + sale_price) * 0.5
    valid = inner > 0
    if not np.any(valid):
        return None
    return lhs[valid], np.log2(inner[valid]), med, valid


def _ols(y, x, w=None):
    """(W)LS of y on [1, x].  Returns beta, X, resid, XtWX_inv, w."""
    n = y.size
    X = np.column_stack([np.ones(n), x])
    w = np.ones(n) if w is None else np.asarray(w, dtype=float)
    XtW = X.T * w
    XtWX = XtW @ X
    XtWX_inv = np.linalg.inv(XtWX)
    beta = XtWX_inv @ (XtW @ y)
    resid = y - X @ beta
    return beta, X, resid, XtWX_inv, w


def prb_inference(assessed, sale_price, weights=None, clusters=None) -> dict | None:
    """PRB slope with classical, HC1 and (optionally) cluster-robust standard errors."""
    d = prb_design(assessed, sale_price)
    if d is None:
        return None
    y, x, med, valid = d
    n = y.size
    if n < 3:
        return None
    w = None if weights is None else np.asarray(weights, dtype=float)[valid]
    beta, X, r, XtWX_inv, w = _ols(y, x, w)
    slope = float(beta[1])
    k = 2
    df = n - k

    # classical (homoskedastic) -- the standard's literal reading, and the display SE
    ssr = float(np.sum(w * r ** 2))
    sw = float(np.sum(w))
    s2 = ssr / df if weights is None else ssr / df
    V_cl = s2 * XtWX_inv
    se_cl = float(np.sqrt(V_cl[1, 1]))

    # HC1 heteroskedasticity-robust -- reported beside it as the robustness column
    meat = (X * (w * r)[:, None]).T @ (X * (w * r)[:, None])
    V_hc1 = XtWX_inv @ meat @ XtWX_inv * (n / df)
    se_hc1 = float(np.sqrt(V_hc1[1, 1]))

    se_cluster = None
    n_clusters = None
    if clusters is not None:
        g = np.asarray(clusters)[valid]
        order = np.argsort(g, kind="mergesort")
        gs, Xs, rs, ws = g[order], X[order], r[order], w[order]
        bounds = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1], True])
        n_clusters = len(bounds) - 1
        # Vectorised per-cluster score sums: a Python loop over ~130k clusters is
        # the dominant cost otherwise.  reduceat gives the identical (n_clusters, k)
        # matrix of within-cluster sums.
        S = Xs * (ws * rs)[:, None]
        sums = np.add.reduceat(S, bounds[:-1], axis=0)
        meat_c = sums.T @ sums
        corr = (n_clusters / max(n_clusters - 1, 1)) * ((n - 1) / df)
        V_cluster = XtWX_inv @ meat_c @ XtWX_inv * corr
        se_cluster = float(np.sqrt(V_cluster[1, 1]))

    tcrit = float(stats.t.ppf(0.5 + CI_LEVEL / 2, df))
    tstat = slope / se_cl if se_cl > 0 else np.nan
    pval = float(2 * stats.t.sf(abs(tstat), df)) if np.isfinite(tstat) else np.nan
    lo, hi = slope - tcrit * se_cl, slope + tcrit * se_cl
    return {"PRB": slope, "n": int(n), "PRB_df": int(df),
            "PRB_se_classical": se_cl, "PRB_se_hc1": se_hc1,
            "PRB_se_cluster_rowid": se_cluster, "n_clusters": n_clusters,
            "PRB_t": float(tstat), "PRB_p": pval,
            "PRB_ci_lo": float(lo), "PRB_ci_hi": float(hi),
            "median_ratio": med, "t_crit": tcrit}


def classify(lo: float, hi: float) -> str:
    """2013 reading: the ENTIRE CI must lie outside a band before it counts against it."""
    if lo > BAND_10 or hi < -BAND_10:
        return "outside_pm010"
    if lo > BAND_05 or hi < -BAND_05:
        return "outside_pm005_but_not_pm010"
    if lo >= -BAND_05 and hi <= BAND_05:
        return "within_pm005"
    return "overlaps_pm005"


ROLE_FOR_EVAL = {**{f"fold_{k}": "descriptive" for k in range(1, 8)},
                 "pooled_oof": "sensitivity", "heldout": "standards_facing",
                 "forward_2025": "standards_facing"}


def _frozen_prb(v4: pd.DataFrame, entry: dict, ev: str):
    """Frozen PRB from the manuscript-facing path table, where directly resolvable."""
    if entry["kind"] != "fitted" or ev == "pooled_oof":
        return None
    fam = {"Direct": "Direct", "Surrogate": "Surrogate", "A-native": "LightGBM"}.get(entry["family"])
    if fam is None:
        return None
    s = v4[v4.family == fam] if fam == "LightGBM" else v4[
        (v4.family == fam) & (v4.rho.astype(float) - float(entry["rho"])).abs().le(1e-12)]
    if len(s) != 1:
        return None
    col = f"PRB__{ev}"
    if col not in v4.columns:
        return None
    v = s.iloc[0][col]
    return None if pd.isna(v) else float(v)


def _write_summary(df: pd.DataFrame, max_recon: float) -> dict:
    fz = df.frozen_minus_recomputed.dropna().abs()
    summary = {
        "ci_level": CI_LEVEL,
        "bands": {"acceptable_pm005": BAND_05, "unacceptable_pm010": BAND_10},
        "standard": "IAAO Standard on Ratio Studies (2013), p.19 -- adopted guidance",
        "classification_rule": ("the ENTIRE 95% CI must lie outside a band before that band is "
                               "deemed exceeded; a CI that merely crosses a threshold is "
                               "'overlaps_pm005', never evidence of exceeding it"),
        "classification_states": ["within_pm005", "overlaps_pm005",
                                  "outside_pm005_but_not_pm010", "outside_pm010"],
        "display_se": "classical (homoskedastic); HC1 reported beside it as robustness",
        "pooled_oof": ("D3 row-balanced WLS (w_ik = 1/m_i) with standard errors clustered on "
                       "row_id -- reported as a SENSITIVITY, not the standards-facing result"),
        "fold_as_iid_replicates": "NEVER -- no mean +/- SD/sqrt(7) appears in this table",
        "rows": int(len(df)),
        "rows_not_attained": int((~df.attained.astype(bool)).sum()),
        "reconciliation_unweighted_rows": {
            "_meaning": ("|P1 OLS slope - utils.motivation_utils.prb()| on the SAME observations, "
                         "for every unweighted evaluation. Must be ~0: it proves the P1 regression "
                         "reproduces the canonical function that produced the frozen values."),
            "max_abs": float(max_recon)},
        "d3_pooled_oof_reweighting_effect": {
            "_meaning": ("|D3 row-balanced WLS slope - unweighted canonical prb()| on pooled_oof. "
                         "This is NOT an error: it is the measured effect of giving each unique "
                         "sale one vote instead of letting the 20,988 fold_6/fold_7 rows count "
                         "twice. Reported as a sensitivity."),
            "max_abs": None, "median_abs": None, "n": None},
        "frozen_reconciliation": {
            "n_compared": int(fz.size),
            "max_abs_frozen_minus_recomputed": float(fz.max()) if fz.size else None,
            "median_abs": float(fz.median()) if fz.size else None},
        "class_counts": {("NOT_ATTAINED" if pd.isna(k) else str(k)): int(v)
                         for k, v in df.iaao_2013_class.value_counts(dropna=False).items()},
        "class_counts_standards_facing": {
            ("NOT_ATTAINED" if pd.isna(k) else str(k)): int(v)
            for k, v in df[df.evaluation_role == "standards_facing"]
                          .iaao_2013_class.value_counts(dropna=False).items()},
        "provenance": c.preflight_block(),
    }
    c.write_json(c.TABLES / "prb_inference_summary.json", summary)
    return summary


def main_summary_only() -> int:
    df = pd.read_csv(c.TABLES / "prb_inference.csv")
    unw = df[(df.weighting == "none")].ols_minus_canonical.dropna().abs()
    d3 = df[(df.weighting == "D3_row_balanced")].ols_minus_canonical.dropna().abs()
    s = _write_summary(df, float(unw.max()))
    s["d3_pooled_oof_reweighting_effect"].update(
        {"max_abs": float(d3.max()), "median_abs": float(d3.median()), "n": int(d3.size)})
    c.write_json(c.TABLES / "prb_inference_summary.json", s)
    print("d3 pooled-OOF reweighting effect:", s["d3_pooled_oof_reweighting_effect"]["max_abs"])
    print("rows:", s["rows"], "| not attained:", s["rows_not_attained"])
    print("max |OLS - canonical| (unweighted rows):", f'{s["reconciliation_unweighted_rows"]["max_abs"]:.3e}')
    print("frozen reconciliation       :", s["frozen_reconciliation"])
    print("class counts (all)          :", s["class_counts"])
    print("class counts (standards)    :", s["class_counts_standards_facing"])
    return 0


def main() -> int:
    from utils.motivation_utils import prb as canonical_prb
    c.assert_d3_multiplicity_identities()
    D = c.display_set()
    v4 = pd.read_csv(c.V12 / "analysis" / "data_id=d4929d43ec19badf"
                     / "split_id=3d464d4a611b131b" / "penalty_path_analysis"
                     / "transition_regions_paper_assets_v4_delta_nl_bends" / "tables"
                     / "combined_path_table_v4_analysis_view.csv")

    entries = D["entries"]
    by_real: dict = {}
    for e in entries:
        by_real.setdefault(e["realization_key"], []).append(e)

    rows = []
    max_recon = 0.0
    for rk, group in by_real.items():
        if rk == "NOT_ATTAINED":
            for e in group:                       # preserved explicitly, never dropped
                for ev in c.ALL_EVALS:
                    rows.append({
                        "realization_key": rk, "display_kind": e["display_kind"],
                        "reference_cell": e["reference_cell"], "family": e["family"],
                        "role_label": e["role"], "j": e["j"], "ext_target": e["ext_target"],
                        "target": e["target"], "rho": e["rho"], "b": e["b"],
                        "attained": False, "match_mode": e["match_mode"],
                        "match_gap": e["match_gap"],
                        "max_achieved_dev_correction": e["max_achieved_dev_correction"],
                        "evaluation": ev, "evaluation_role": ROLE_FOR_EVAL[ev],
                        "weighting": None, "n": None, "n_unique": None,
                        "PRB": None, "PRB_se_classical": None, "PRB_se_hc1": None,
                        "PRB_se_cluster_rowid": None, "n_clusters": None, "PRB_t": None,
                        "PRB_df": None, "PRB_p": None, "PRB_ci_lo": None, "PRB_ci_hi": None,
                        "iaao_2013_class": None, "canonical_prb_fn": None,
                        "ols_minus_canonical": None, "frozen_PRB_value": None,
                        "frozen_minus_recomputed": None,
                        "source": "NOT_ATTAINED within the frozen design"})
            continue

        head = group[0]
        for ev in c.ALL_EVALS:
            d = c.load_observations(head, ev)
            y_true_log = d.y_true_log.to_numpy()
            y_pred_log = d.y_pred_log.to_numpy()
            assessed = np.exp(y_pred_log)          # exactly as compute_taxation_metrics does
            sale_price = np.exp(y_true_log)
            rid = d.row_id.to_numpy()

            if ev == "pooled_oof":
                w = c.d3_weights_for_pooled(rid)
                res = prb_inference(assessed, sale_price, weights=w, clusters=rid)
                weighting = "D3_row_balanced"
                n_unique = int(pd.unique(rid).size)
            else:
                res = prb_inference(assessed, sale_price)
                weighting = "none"
                n_unique = int(pd.unique(rid).size)

            canon = float(canonical_prb(assessed, sale_price, na_rm=True))
            # unweighted OLS must reproduce the canonical function bit-for-bit
            if weighting == "none":
                delta = abs(res["PRB"] - canon)
                max_recon = max(max_recon, delta)
                if delta > 1e-10:
                    raise c.ProtocolViolation(
                        f"PRB OLS does not reproduce canonical prb() for {rk}/{ev}: {delta:.3e}")
            frozen = _frozen_prb(v4, head, ev)
            for e in group:
                rows.append({
                    "realization_key": rk, "display_kind": e["display_kind"],
                    "reference_cell": e["reference_cell"], "family": e["family"],
                    "role_label": e["role"], "j": e["j"], "ext_target": e["ext_target"],
                    "target": e["target"], "rho": e["rho"], "b": e["b"],
                    "attained": True, "match_mode": e["match_mode"],
                    "match_gap": e["match_gap"],
                    "max_achieved_dev_correction": e["max_achieved_dev_correction"],
                    "evaluation": ev, "evaluation_role": ROLE_FOR_EVAL[ev],
                    "weighting": weighting, "n": res["n"], "n_unique": n_unique,
                    "PRB": res["PRB"], "PRB_se_classical": res["PRB_se_classical"],
                    "PRB_se_hc1": res["PRB_se_hc1"],
                    "PRB_se_cluster_rowid": res["PRB_se_cluster_rowid"],
                    "n_clusters": res["n_clusters"], "PRB_t": res["PRB_t"],
                    "PRB_df": res["PRB_df"], "PRB_p": res["PRB_p"],
                    "PRB_ci_lo": res["PRB_ci_lo"], "PRB_ci_hi": res["PRB_ci_hi"],
                    "iaao_2013_class": classify(res["PRB_ci_lo"], res["PRB_ci_hi"]),
                    "canonical_prb_fn": canon,
                    "ols_minus_canonical": res["PRB"] - canon,
                    "frozen_PRB_value": frozen,
                    "frozen_minus_recomputed": None if frozen is None else frozen - canon,
                    "source": head["artifacts"].get(ev) if ev != "pooled_oof"
                              else "concatenated fold_1..fold_7 (D3 row-balanced)"})
        print(f"  done {rk}  ({len(rows)} rows so far)", flush=True)

    df = pd.DataFrame(rows)
    c.write_table(df, c.TABLES / "prb_inference.csv")

    fz = df.frozen_minus_recomputed.dropna().abs()
    summary = {
        "ci_level": CI_LEVEL,
        "bands": {"acceptable_pm005": BAND_05, "unacceptable_pm010": BAND_10},
        "standard": "IAAO Standard on Ratio Studies (2013), p.19 -- adopted guidance",
        "classification_rule": ("the ENTIRE 95% CI must lie outside a band before that band is "
                               "deemed exceeded; a CI that merely crosses a threshold is "
                               "'overlaps_pm005', never evidence of exceeding it"),
        "display_se": "classical (homoskedastic); HC1 reported beside it as robustness",
        "pooled_oof": ("D3 row-balanced WLS (w_ik = 1/m_i) with standard errors clustered on "
                       "row_id -- reported as a SENSITIVITY, not the standards-facing result"),
        "fold_as_iid_replicates": "NEVER -- no mean +/- SD/sqrt(7) appears in this table",
        "rows": len(df),
        "rows_not_attained": int((~df.attained).sum()),
        "reconciliation_unweighted_rows": {
            "_meaning": ("|P1 OLS slope - utils.motivation_utils.prb()| on the SAME observations, "
                         "for every unweighted evaluation. Must be ~0: it proves the P1 regression "
                         "reproduces the canonical function that produced the frozen values."),
            "max_abs": float(max_recon)},
        "d3_pooled_oof_reweighting_effect": {
            "_meaning": ("|D3 row-balanced WLS slope - unweighted canonical prb()| on pooled_oof. "
                         "This is NOT an error: it is the measured effect of giving each unique "
                         "sale one vote instead of letting the 20,988 fold_6/fold_7 rows count "
                         "twice. Reported as a sensitivity."),
            "max_abs": None, "median_abs": None, "n": None},
        "frozen_reconciliation": {
            "n_compared": int(fz.size),
            "max_abs_frozen_minus_recomputed": float(fz.max()) if fz.size else None,
            "median_abs": float(fz.median()) if fz.size else None},
        "class_counts": {("NOT_ATTAINED" if pd.isna(k) else str(k)): int(v)
                         for k, v in df.iaao_2013_class.value_counts(dropna=False).items()},
        "provenance": c.preflight_block(),
    }
    c.write_json(c.TABLES / "prb_inference_summary.json", summary)
    print("\nrows:", len(df), "| not attained:", summary["rows_not_attained"])
    print("max |OLS - canonical prb()| :", f"{max_recon:.3e}")
    print("frozen reconciliation       :", summary["frozen_reconciliation"])
    print("class counts                :", summary["class_counts"])
    return 0


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary-only", action="store_true",
                    help="regenerate the summary JSON from the written prb_inference.csv")
    a = ap.parse_args()
    raise SystemExit(main_summary_only() if a.summary_only else main())
