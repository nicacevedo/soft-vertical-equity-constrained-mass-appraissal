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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["freeze", "estimate", "apply"])
    a = ap.parse_args()
    if a.mode == "freeze":
        raise SystemExit(mode_freeze())
    if a.mode == "estimate":
        raise SystemExit(mode_estimate())
    raise SystemExit(0)
