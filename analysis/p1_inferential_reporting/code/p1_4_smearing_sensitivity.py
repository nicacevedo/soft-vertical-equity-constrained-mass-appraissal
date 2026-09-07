#!/usr/bin/env python3
"""P1 Task 4: development-only Duan smearing sensitivity (D3 row-balanced).

Three modes, run in order:
  freeze    write configs/smearing_estimator_frozen.json (+ hash).  Reads NOTHING.
  estimate  compute s from DEVELOPMENT out-of-fold residuals only.
  apply     apply the frozen s unchanged to every evaluation; invariance audit.

SIGN.  The canonical residual convention in this repository is
    e = y_pred_log - y_true_log            (utils/motivation_utils.py:1554)
Duan's smearing factor uses the OPPOSITE-SIGN log error
    u = y_true_log - y_pred_log = -e
so that  s = E[exp(u)]  and  y_hat = s * exp(f(x)).   exp(e) is WRONG.

WEIGHTING.  s is row-balanced consistently with D3: w_ik = 1/m_i, so each unique
development sale row carries total weight exactly one.  The naive duplicate-weighted
pooled-OOF factor (the D2 construction) is reported alongside as s_naive_pooled but
is never applied.

This is a SENSITIVITY.  The canonical model is unaltered, nothing is retuned, and no
smeared value replaces a headline number.
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p1_common as c                                                   # noqa: E402

FROZEN = "smearing_estimator_frozen.json"
DEV_EVALS = [f"fold_{k}" for k in range(1, 8)]

# Behaviour of each metric under y_hat -> s * y_hat  (s > 0 constant).
INVARIANCE = {
    "median_ratio": "scales_by_s", "mean_ratio": "scales_by_s",
    "weighted_mean_ratio": "scales_by_s",
    "COD": "invariant", "COV": "invariant", "PRD": "invariant", "PRB": "invariant",
    "VEI": "invariant", "MKI": "invariant", "beta_log": "invariant",
    "dCor_e_y": "invariant", "Delta_NL": "invariant",
    "Cov_log_residual_log_price": "invariant",
    "R2_price": "moves", "MAE_price": "moves", "MAPE": "moves",
}
TOL_REL = 1e-9


# ------------------------------------------------------------------- freeze
def mode_freeze() -> int:
    payload = {
        "schema_version": 1,
        "stage": "P1_TASK4_DUAN_SMEARING",
        "frozen_before_any_heldout_or_2025_output_is_read": True,
        "residual_convention_in_repo": "e = y_pred_log - y_true_log",
        "residual_convention_source": "utils/motivation_utils.py:1554 (paper_mechanism_metrics)",
        "duan_log_error": "u_ik = y_true_log - y_pred_log = -e_ik",
        "formula": "s = sum_ik( w_ik * exp(u_ik) ) / sum_ik( w_ik )",
        "formula_expanded": "s = sum_ik( w_ik * exp(-(y_pred_log - y_true_log)) ) / sum_ik( w_ik )",
        "forbidden_formula": "s = mean(exp(e_ik))  -- WRONG SIGN, must never appear",
        "weights": "w_ik = 1 / m_i, m_i = number of development validation folds containing unique row i",
        "weighting_name": "D3 row-balanced (one sale row, one vote)",
        "application": "y_hat_level = s * exp(f(x));  equivalently y_pred_log -> y_pred_log + log(s)",
        "estimation_sample": "DEVELOPMENT out-of-fold residuals only (fold_1..fold_7 validation blocks)",
        "prohibited": ("estimating s on the heldout block or the 2025 forward block is prohibited "
                       "and is refused by assertion, not by convention"),
        "applied_unchanged_to": ["fold_1..fold_7", "pooled_oof", "heldout", "forward_2025"],
        "forward_2025_assumption": (
            "The 2025 forward evaluation's fitting set is the 382,897-row production block, which has "
            "no out-of-fold analogue. The development-estimated s is applied there unchanged. This is "
            "an explicit assumption and a stated limitation, not a validated property."),
        "also_reported_never_applied": {
            "s_naive_pooled": ("duplicate-weighted pooled-OOF factor, i.e. the D2 construction that the "
                               "approved P0 plan section J-bis mandated; reported for transparency only")},
        "d3_multiplicity": {
            "n_appearances": c.D3_N_APPEARANCES, "n_unique": c.D3_N_UNIQUE,
            "unique_rows_multiplicity_1": c.D3_UNIQUE_ROWS_MULT_1,
            "duplicated_unique_rows": c.D3_DUPLICATED_UNIQUE_ROWS,
            "duplicated_appearances": c.D3_DUPLICATED_APPEARANCES,
            "max_multiplicity": c.D3_MAX_MULTIPLICITY,
            "identities": ["109177 + 20988 == 130165 (unique rows)",
                           "109177 + 2*20988 == 151153 (appearances)"],
            "warning": ("the P0 artifact key m_i_distribution={'1':109177,'2':41976} is valued in "
                        "APPEARANCES; 41,976 appearances come from 20,988 duplicated UNIQUE rows. "
                        "Never write 'multiplicity 2 -> 41,976 rows'.")},
        "log_scale_metrics": {
            "beta_log": "INVARIANT to adding a constant to the log predictions (c_y is centered)",
            "RMSE_log": ("NOT mathematically invariant to adding a constant. It is simply not "
                         "recomputed: Duan smearing is a post-exponentiation price-scale "
                         "retransformation sensitivity and does not alter the canonical log "
                         "prediction.")},
        "role": "SENSITIVITY ONLY -- canonical model unaltered, nothing retuned, no headline replaced",
        "provenance": c.preflight_block(),
    }
    out = c.write_json(c.CONFIGS / FROZEN, payload)
    c.write_json(c.CONFIGS / FROZEN.replace(".json", "_hash.json"),
                 {"file": out.name, "file_sha256": c.sha256_file(out),
                  "frozen_at_utc": pd.Timestamp.utcnow().isoformat(),
                  "frozen_before_any_oos_read": True})
    print(f"[freeze] wrote {out}  sha256={c.sha256_file(out)[:16]}...")
    return 0


def _check_frozen() -> dict:
    p = c.CONFIGS / FROZEN
    h = __import__("json").loads((c.CONFIGS / FROZEN.replace(".json", "_hash.json")).read_text())
    if h["file_sha256"] != c.sha256_file(p):
        raise c.ProtocolViolation(f"{FROZEN} changed after being hashed")
    return __import__("json").loads(p.read_text())


# ----------------------------------------------------------------- estimate
def smearing_factor(u: np.ndarray, w: np.ndarray) -> float:
    """s = sum(w * exp(u)) / sum(w),  u = y_true_log - y_pred_log."""
    u = np.asarray(u, dtype=float); w = np.asarray(w, dtype=float)
    return float(np.sum(w * np.exp(u)) / np.sum(w))


def mode_estimate() -> int:
    F = _check_frozen()
    print(f"[estimate] frozen estimator hash validated; formula: {F['formula']}")
    c.assert_d3_multiplicity_identities()
    D = c.display_set()
    by_real = {}
    for e in D["entries"]:
        by_real.setdefault(e["realization_key"], []).append(e)

    rows = []
    for rk, group in by_real.items():
        if rk == "NOT_ATTAINED":
            continue
        head = group[0]
        parts = []
        for ev in DEV_EVALS:                       # development folds ONLY
            d = c.load_observations(head, ev)
            parts.append(d[["row_id", "y_true_log", "y_pred_log"]])
        dev = pd.concat(parts, ignore_index=True)
        rid = dev.row_id.to_numpy()
        u = dev.y_true_log.to_numpy() - dev.y_pred_log.to_numpy()      # = -e
        w = c.d3_weights_for_pooled(rid)
        s = smearing_factor(u, w)
        s_naive = smearing_factor(u, np.ones_like(w))
        uniq, cnt = np.unique(rid, return_counts=True)
        rows.append({
            "realization_key": rk, "family": head["family"], "rho": head["rho"], "b": head["b"],
            "config_id": head.get("config_id"), "reference_cell": head["reference_cell"],
            "s_source": "dev_oof_row_balanced_D3", "s": s, "s_naive_pooled": s_naive,
            "s_minus_s_naive": s - s_naive,
            "n_appearances": int(rid.size), "n_unique": int(uniq.size),
            "unique_rows_multiplicity_1": int((cnt == 1).sum()),
            "duplicated_unique_rows": int((cnt == 2).sum()),
            "duplicated_appearances": int(2 * (cnt == 2).sum()),
            "max_multiplicity": int(cnt.max()),
            "sum_weights": float(w.sum()),
            "max_abs_rowweight_minus_one": float(
                np.abs(pd.Series(w).groupby(pd.Series(rid)).sum().to_numpy() - 1.0).max()),
            "mean_u": float(np.average(u, weights=w)),
            "estimation_evaluations": "|".join(DEV_EVALS),
            "oos_used_in_estimation": False,
        })
        # hard guards
        r = rows[-1]
        if r["n_appearances"] != c.D3_N_APPEARANCES or r["n_unique"] != c.D3_N_UNIQUE:
            raise c.ProtocolViolation(f"{rk}: dev OOF structure {r['n_appearances']}/{r['n_unique']} "
                                      f"!= {c.D3_N_APPEARANCES}/{c.D3_N_UNIQUE}")
        if r["duplicated_unique_rows"] != c.D3_DUPLICATED_UNIQUE_ROWS:
            raise c.ProtocolViolation(f"{rk}: duplicated unique rows {r['duplicated_unique_rows']}")
        if r["max_abs_rowweight_minus_one"] > 1e-12:
            raise c.ProtocolViolation(f"{rk}: D3 weights do not sum to one per unique row")
        print(f"  {rk:52s} s={s:.6f}  s_naive={s_naive:.6f}", flush=True)

    df = pd.DataFrame(rows)
    c.write_table(df, c.TABLES / "smearing_factor_provenance.csv")
    print(f"\n[estimate] {len(df)} realizations | s range "
          f"[{df.s.min():.6f}, {df.s.max():.6f}] | max |s - s_naive| "
          f"{df.s_minus_s_naive.abs().max():.3e}")
    return 0


# ------------------------------------------------------------------- apply
# Delta_NL costs ~34 s per call (five OOF spline fits).  Auditing it on all
# 43 x 10 cells in both arms would cost ~8 h, so it is verified on a
# PRE-DECLARED subset -- the two included reference cells plus the realization
# with the LARGEST s (the strongest stress on invariance) -- across all ten
# evaluation blocks.  Its invariance is also exact by construction: Delta_NL is
# built from OOF residuals of e on Z where both the affine and the spline head
# carry an intercept, and Var(e) is translation invariant, so adding log(s) to
# every prediction is absorbed by the intercept.
DELTA_NL_SUBSET_RULE = ("the two included reference cells (C, A) plus the realization with the "
                        "largest frozen s; all ten evaluation blocks; both arms")

_LEVEL = ("median_ratio", "mean_ratio", "weighted_mean_ratio")


def _pooled_oof_ed2_reason() -> str:
    import p1_3_vei_ed2_inference as v
    return v.POOLED_OOF_REASON


POOLED_OOF_ED2_REASON = _pooled_oof_ed2_reason()


def canonical_suite_fast(y_log: np.ndarray, p_log: np.ndarray) -> dict:
    """The canonical metric suite MINUS Delta_NL.

    Every value is produced by the same canonical function that
    utils.motivation_utils.compute_taxation_metrics calls, with the three
    level metrics written exactly as that function writes them.  Verified
    against p0_4_centered_spread.metrics_from on the Delta_NL subset.
    """
    import utils.motivation_utils as mu
    from sklearn.metrics import mean_absolute_error, r2_score

    y_log = np.asarray(y_log, float); p_log = np.asarray(p_log, float)
    y_price, p_price = np.exp(y_log), np.exp(p_log)
    ratios = p_price / y_price
    out = {
        "R2_price": float(r2_score(y_price, p_price)),
        "MAE_price": float(mean_absolute_error(y_price, p_price)),
        "MAPE": float(np.mean(np.abs(p_price - y_price) / y_price)),
        "median_ratio": float(np.median(ratios)),
        "mean_ratio": float(np.mean(ratios)),
        "weighted_mean_ratio": float(np.sum(p_price) / np.sum(y_price)),
        "COD": float(mu.cod(ratios, na_rm=True)),
        "COV": float(mu.cov_iaao(p_price, y_price, na_rm=True)),
        "PRD": float(mu.prd(p_price, y_price, na_rm=True)),
        "PRB": float(mu.prb(p_price, y_price, na_rm=True)),
        "MKI": float(mu.mki(p_price, y_price, na_rm=True)),
        "VEI": float(mu.vei(p_price, y_price, na_rm=True)),
    }
    mech = mu.paper_mechanism_metrics(y_log, p_log)
    out["beta_log"] = float(mech["Beta_log"])
    out["Cov_log_residual_log_price"] = float(mech["Cov_log_residual_log_price"])
    out["dCor_e_y"] = float(mech["dCor_e_y"])
    return out


def _rel(a: float, b: float) -> float:
    """Relative discrepancy |a-b| / max(|b|, 1e-12), with an absolute floor so that
    metrics legitimately near zero cannot manufacture a spurious FLAG."""
    d = abs(a - b)
    scale = max(abs(b), 1e-8)
    return float(d / scale)


def _ed2_verdicts(y_log: np.ndarray, p_log: np.ndarray) -> dict:
    """Re-run the full ED2 App. E decision path (Task 3 code, imported)."""
    import p1_3_vei_ed2_inference as v
    r = v.evaluate_ed2(np.exp(np.asarray(p_log, float)), np.exp(np.asarray(y_log, float)))
    if r is None:
        return {}
    return {k: r.get(k) for k in ("VEI_step5", "step5_gate", "step6_result",
                                  "VEI_significance_step7", "step7_outcome", "ed2_verdict",
                                  "first_pg_ci_lo", "first_pg_ci_hi",
                                  "last_pg_ci_lo", "last_pg_ci_hi", "sample_median_ratio")}


def mode_apply() -> int:
    F = _check_frozen()
    print(f"[apply] frozen estimator hash VALIDATED  ({c.sha256_file(c.CONFIGS / FROZEN)[:16]}...)")
    print(f"[apply] formula: {F['formula']}")
    if not F.get("frozen_before_any_heldout_or_2025_output_is_read"):
        raise c.ProtocolViolation("frozen estimator does not assert pre-OOS freeze")
    c.assert_d3_multiplicity_identities()

    prov = pd.read_csv(c.TABLES / "smearing_factor_provenance.csv")
    if bool(prov.oos_used_in_estimation.any()):
        raise c.ProtocolViolation("a smearing factor claims OOS data in estimation")
    if not (prov.s_source == "dev_oof_row_balanced_D3").all():
        raise c.ProtocolViolation("non-D3 smearing source present")
    S = dict(zip(prov.realization_key, prov.s.astype(float)))
    print(f"[apply] {len(S)} frozen factors, s in [{min(S.values()):.6f}, {max(S.values()):.6f}]")

    D = c.display_set()
    by_real = {}
    for e in D["entries"]:
        by_real.setdefault(e["realization_key"], []).append(e)

    max_s_rk = max(S, key=lambda k: S[k])
    subset = {e["realization_key"] for e in D["entries"]
              if e.get("reference_cell") in ("C", "A")} | {max_s_rk}
    print(f"[apply] Delta_NL subset ({len(subset)}): {sorted(subset)}")

    rows, ed2_rows, dnl_rows, fastcheck = [], [], [], []
    for rk, group in by_real.items():
        if rk == "NOT_ATTAINED":
            continue
        if rk not in S:
            raise c.ProtocolViolation(f"no frozen smearing factor for realization {rk}")
        s = S[rk]
        if not (s > 0):
            raise c.ProtocolViolation(f"non-positive s for {rk}")
        head = group[0]
        for ev in c.ALL_EVALS:
            d = c.load_observations(head, ev)
            y = d.y_true_log.to_numpy()
            p = d.y_pred_log.to_numpy()
            p_s = p + math.log(s)                      # y_hat -> s * y_hat
            base, smear = canonical_suite_fast(y, p), canonical_suite_fast(y, p_s)

            for m, cls in INVARIANCE.items():
                if m not in base:
                    continue                            # Delta_NL handled below
                b, a = base[m], smear[m]
                if cls == "scales_by_s":
                    exp_v, rel = s * b, _rel(a, s * b)
                    flag = "FLAG" if rel > TOL_REL else "OK"
                elif cls == "invariant":
                    exp_v, rel = b, _rel(a, b)
                    flag = "FLAG" if rel > TOL_REL else "OK"
                else:
                    exp_v, rel, flag = None, None, "NA_MOVES_BY_DESIGN"
                rows.append({
                    "realization_key": rk, "family": head["family"], "rho": head["rho"],
                    "b": head["b"], "reference_cell": head["reference_cell"],
                    "evaluation": ev, "n": int(len(y)), "s": s, "log_s": math.log(s),
                    "metric": m, "invariance_class": cls,
                    "baseline": b, "smeared": a, "expected": exp_v,
                    "abs_diff_from_expected": (None if exp_v is None else abs(a - exp_v)),
                    "rel_diff_from_expected": rel, "flag": flag,
                    "delta_smeared_minus_baseline": a - b,
                })

            # ED2 App. E inference is NOT applicable to the D3 row-balanced pooled-OOF
            # construction (Task 3 adjudication: App. D.2 is a rank-based order statistic
            # on distinct observations and the draft defines no weighted-median analogue).
            # Applying it here would contradict that adjudication, so it is refused with the
            # same reason recorded.  The VEI POINT ESTIMATE above is unaffected: it is a
            # descriptive statistic, not the CI-based ED2 test.
            if ev == "pooled_oof":
                ed2_rows.append({
                    "realization_key": rk, "family": head["family"], "rho": head["rho"],
                    "b": head["b"], "evaluation": ev, "s": s,
                    "ed2_applicability": "NOT_APPLICABLE_FOR_ED2_INFERENCE",
                    "ed2_not_applicable_reason": POOLED_OOF_ED2_REASON,
                    "verdict_unchanged": None, "vei_rel_diff": None})
            else:
                v0, v1 = _ed2_verdicts(y, p), _ed2_verdicts(y, p_s)
                if v0 and v1:
                    ed2_rows.append({
                        "realization_key": rk, "family": head["family"], "rho": head["rho"],
                        "b": head["b"], "evaluation": ev, "s": s,
                        "ed2_applicability": "ED2_APPLIED",
                        "ed2_not_applicable_reason": None,
                        **{f"base_{k}": v for k, v in v0.items()},
                        **{f"smeared_{k}": v for k, v in v1.items()},
                        "verdict_unchanged": bool(v0["ed2_verdict"] == v1["ed2_verdict"]
                                                  and v0["step5_gate"] == v1["step5_gate"]
                                                  and v0["step6_result"] == v1["step6_result"]
                                                  and v0["step7_outcome"] == v1["step7_outcome"]),
                        "vei_rel_diff": _rel(v1["VEI_step5"], v0["VEI_step5"]),
                    })

            if rk in subset:
                import sys as _s
                _s.path.insert(0, str(c.P0_DIR / "code"))
                import p0_4_centered_spread as cs
                rid = d.row_id.to_numpy()
                dnl_ids = (np.array([f"{int(t)}|{int(r)}" for t, r in zip(d.fold.to_numpy(), rid)],
                                    dtype=object) if ev == "pooled_oof" else None)
                m0 = cs.metrics_from(y, p, y, rid, dnl_ids=dnl_ids)
                m1 = cs.metrics_from(y, p_s, y, rid, dnl_ids=dnl_ids)
                dnl_rows.append({
                    "realization_key": rk, "evaluation": ev, "n": int(len(y)), "s": s,
                    "Delta_NL_baseline": m0["Delta_NL"], "Delta_NL_smeared": m1["Delta_NL"],
                    "Delta_NL_rel_diff": _rel(m1["Delta_NL"], m0["Delta_NL"]),
                    "Delta_NL_raw_baseline": m0["Delta_NL_raw"],
                    "Delta_NL_raw_smeared": m1["Delta_NL_raw"],
                    "Delta_NL_raw_rel_diff": _rel(m1["Delta_NL_raw"], m0["Delta_NL_raw"]),
                    "flag": ("FLAG" if _rel(m1["Delta_NL"], m0["Delta_NL"]) > TOL_REL else "OK"),
                })
                # the fast path must reproduce the frozen canonical suite exactly
                ren = {"COV": "COV", "median_ratio": "median_ratio"}
                for k in base:
                    if k in m0:
                        fastcheck.append({"realization_key": rk, "evaluation": ev, "metric": k,
                                          "fast": base[k], "canonical": float(m0[k]),
                                          "rel_diff": _rel(base[k], float(m0[k]))})
            print(f"  {rk[:44]:44s} {ev:13s} n={len(y):7d}", flush=True)

    df = pd.DataFrame(rows)
    ed2 = pd.DataFrame(ed2_rows)
    dnl = pd.DataFrame(dnl_rows)
    fc = pd.DataFrame(fastcheck)
    c.write_table(df, c.TABLES / "smearing_apply_invariance.csv")
    c.write_table(ed2, c.TABLES / "smearing_apply_ed2_stability.csv")
    c.write_table(dnl, c.TABLES / "smearing_apply_delta_nl_subset.csv")
    c.write_table(fc, c.TABLES / "smearing_apply_fastpath_reconciliation.csv")

    tested = df[df.flag != "NA_MOVES_BY_DESIGN"]
    _ap = (ed2[ed2.ed2_applicability == "ED2_APPLIED"] if len(ed2) else ed2)
    flagged = tested[tested.flag == "FLAG"]
    summary = {
        "role": F["role"],
        "applied_s_source": "configs/smearing_estimator_frozen.json (hash-validated) + "
                            "tables/smearing_factor_provenance.csv (frozen estimate, unchanged)",
        "application": F["application"],
        "duan_log_error": F["duan_log_error"],
        "s_recomputed_in_apply": False,
        "n_realizations": int(df.realization_key.nunique()),
        "n_evaluations": int(df.evaluation.nunique()),
        "n_cells": int(df.groupby(["realization_key", "evaluation"]).ngroups),
        "n_metric_checks": int(len(df)),
        "n_invariance_or_scale_checks": int(len(tested)),
        "n_flagged": int(len(flagged)),
        "max_rel_diff_scales_by_s": float(
            tested[tested.invariance_class == "scales_by_s"].rel_diff_from_expected.max()),
        "max_rel_diff_invariant": float(
            tested[tested.invariance_class == "invariant"].rel_diff_from_expected.max()),
        "per_metric_max_rel_diff": {
            m: float(g.rel_diff_from_expected.max())
            for m, g in tested.groupby("metric")},
        "level_metrics_scale_by_s": sorted(_LEVEL),
        "tolerance_rel": TOL_REL,
        "ed2_stability": {
            "scope": "the nine ED2-applicable evaluation blocks (fold_1..fold_7, heldout, "
                     "forward_2025); pooled_oof is NOT_APPLICABLE_FOR_ED2_INFERENCE",
            "n_rows_total": int(len(ed2)),
            "n_cells_ed2_applied": int(len(_ap)),
            "n_cells_not_applicable": int(len(ed2) - len(_ap)),
            "not_applicable_reason": POOLED_OOF_ED2_REASON,
            "n_verdict_unchanged": int(_ap.verdict_unchanged.sum()) if len(_ap) else 0,
            "all_verdicts_unchanged": bool(_ap.verdict_unchanged.all()) if len(_ap) else None,
            "max_vei_rel_diff": float(_ap.vei_rel_diff.max()) if len(_ap) else None,
        },
        "delta_nl_subset": {
            "rule": DELTA_NL_SUBSET_RULE,
            "realizations": sorted(dnl.realization_key.unique().tolist()) if len(dnl) else [],
            "n_cells": int(len(dnl)),
            "max_rel_diff": float(dnl.Delta_NL_rel_diff.max()) if len(dnl) else None,
            "all_ok": bool((dnl.flag == "OK").all()) if len(dnl) else None,
            "analytic_reason": ("e -> e + log(s); the affine and spline heads both carry an "
                                "intercept and Var(e) is translation invariant, so the OOF "
                                "residuals and hence Delta_NL are unchanged by construction"),
        },
        "fastpath_reconciliation": {
            "n_comparisons": int(len(fc)),
            "max_rel_diff_vs_p0_metrics_from": float(fc.rel_diff.max()) if len(fc) else None,
        },
        "moves_by_design": {
            m: {"max_abs_delta": float(df[(df.metric == m)].delta_smeared_minus_baseline.abs().max())}
            for m, cl in INVARIANCE.items() if cl == "moves" and (df.metric == m).any()},
        "RMSE_log_note": F["log_scale_metrics"]["RMSE_log"],
        "forward_2025_assumption": F["forward_2025_assumption"],
        "provenance": c.preflight_block(),
    }
    c.write_json(c.TABLES / "smearing_apply_summary.json", summary)

    print(f"\n[apply] cells={summary['n_cells']}  checks={summary['n_metric_checks']}  "
          f"tested={summary['n_invariance_or_scale_checks']}  FLAGGED={summary['n_flagged']}")
    print(f"[apply] max rel diff  scales_by_s={summary['max_rel_diff_scales_by_s']:.3e}  "
          f"invariant={summary['max_rel_diff_invariant']:.3e}")
    print(f"[apply] ED2 verdicts unchanged: {summary['ed2_stability']['all_verdicts_unchanged']} "
          f"({summary['ed2_stability']['n_verdict_unchanged']}/"
          f"{summary['ed2_stability']['n_cells_ed2_applied']} ED2-applicable cells; "
          f"{summary['ed2_stability']['n_cells_not_applicable']} pooled_oof NOT_APPLICABLE)")
    print(f"[apply] Delta_NL subset max rel diff: {summary['delta_nl_subset']['max_rel_diff']}")
    print(f"[apply] fast-path vs P0 metrics_from max rel diff: "
          f"{summary['fastpath_reconciliation']['max_rel_diff_vs_p0_metrics_from']}")
    if len(flagged):
        print("\n[apply] FLAGGED ROWS:")
        print(flagged[["realization_key", "evaluation", "metric", "invariance_class",
                       "baseline", "smeared", "expected", "rel_diff_from_expected"]].to_string())
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["freeze", "estimate", "apply"])
    a = ap.parse_args()
    if a.mode == "freeze":
        raise SystemExit(mode_freeze())
    if a.mode == "estimate":
        raise SystemExit(mode_estimate())
    raise SystemExit(mode_apply())
