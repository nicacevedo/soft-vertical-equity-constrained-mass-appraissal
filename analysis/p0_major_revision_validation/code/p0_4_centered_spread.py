#!/usr/bin/env python3
"""Stage 2 -- centered-spread post-hoc comparator (theorem-matched primary map).

PRIMARY reference  = C  "Custom-objective rho=0 origin"
SECONDARY practical = A  "Ordinary LightGBM (standard raw-label native)"
B receives NO full path.

Primary map (only map used for the full path):

    f_b(x) = ybar_T + b * (f0(x) - ybar_T)

with ybar_T the mean log target of the FITTING BLOCK for that regime:
  CV fold k -> fold-k training mean;  held-out -> development pool;  2025 -> production block.

b == 1 takes an explicit fast path returning the cached array unchanged, so the QC is bitwise.

Calibration is DEVELOPMENT OUT-OF-SAMPLE.  b_star_train (in-sample) is carried only as a
labelled theory diagnostic and never defines the empirical endpoint.

Modes
-----
  roots     compute A/C development roots + b_max, freeze & hash configs/b_grid_frozen.json
            (reads FOLD predictions only -- never held-out or 2025)
  path      evaluate the frozen grid for one (ref, evaluation); asserts the grid hash first
  assemble  merge, CV summary, linearity, centering sensitivity, historical QC, profiles
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as c

T = c.TABLES
REFS = {"C": "Custom-objective rho=0 origin", "A": "Ordinary LightGBM (standard raw-label native)"}
REF_ROLE = {"C": "PRIMARY within-path penalty-isolating reference (post-hoc base)",
            "A": "SECONDARY practical assessor-facing post-hoc comparator"}
FOLD_EVALS = [f"fold_{k}" for k in range(1, 8)]
OOS_EVALS = ["heldout", "forward_2025"]
ALL_EVALS = FOLD_EVALS + ["pooled_oof"] + OOS_EVALS

BLOCK_FOR = {**{f"fold_{k}": f"fold_{k}_train" for k in range(1, 8)},
             "heldout": "development_pool", "forward_2025": "production_2016_2024"}

N_BASE = 121
OVERSHOOT = 1.25


# --------------------------------------------------------------------------- io
def zref() -> pd.DataFrame:
    return pd.read_csv(T / "zero_reference_fits.csv")


def load_eval(ref: str, block_id: str) -> tuple[pd.DataFrame, float, str]:
    """Cached paired-evaluation predictions + the fitting-block ybar_T, hash-verified."""
    zf = zref()
    r = zf[(zf.cell_id == ref) & (zf.block_id == block_id)]
    if len(r) != 1:
        raise c.ProtocolViolation(f"no unique Stage-1.5 fit for {ref}/{block_id}")
    r = r.iloc[0]
    d = pd.read_parquet(c.REPO / r.eval_predictions)
    h = hashlib.sha256(np.ascontiguousarray(d.y_pred_log.to_numpy(), dtype=np.float64)
                       .tobytes()).hexdigest()
    if h != r.eval_pred_sha256:
        raise c.ProtocolViolation(f"{ref}/{block_id}: eval prediction hash mismatch")
    return d.sort_values("row_id").reset_index(drop=True), float(r.ybar_T), h


# --------------------------------------------------------- theorem-matched map
def centered_map(f0: np.ndarray, ybar_T: float, b: float) -> np.ndarray:
    """f_b = ybar_T + b (f0 - ybar_T).  b == 1 returns f0 unchanged (bitwise)."""
    if b == 1.0:
        return f0
    return float(ybar_T) + float(b) * (np.asarray(f0, dtype=float) - float(ybar_T))


def beta_ratio(y: np.ndarray, f0: np.ndarray) -> tuple[float, float, float]:
    """Return (Cov_V(f0,y), Var_V(y), R = Cov/Var) in the canonical beta_log convention."""
    cy = y - float(np.mean(y))
    var = float(np.mean(cy ** 2))
    cov = float(np.mean(f0 * cy))          # == mean((f0 - f0bar_V) * cy) since mean(cy)=0
    return cov, var, cov / var


# --------------------------------------------------------------------- metrics
def metrics_from(y_log, p_log, y_train_log, row_ids, dnl_ids=None) -> dict:
    from utils.motivation_utils import _compute_extended_metrics, paper_mechanism_metrics
    from utils.delta_nl import estimate_delta_nl
    m = _compute_extended_metrics(y_true_log=np.asarray(y_log, float),
                                  y_pred_log=np.asarray(p_log, float),
                                  y_train_log=np.asarray(y_train_log, float),
                                  ratio_mode="diff")
    ren = {"Median ratio": "median_ratio", "Mean ratio": "mean_ratio",
           "W. Mean ratio": "weighted_mean_ratio", "COV_IAAO": "COV"}
    out = {}
    for k in ["R2_price", "MAE_price", "MAPE", "RMSE_log", "Median ratio", "Mean ratio",
              "W. Mean ratio", "COD", "COV_IAAO", "PRD", "PRB", "MKI", "VEI"]:
        if k in m:
            try:
                out[ren.get(k, k)] = float(m[k])
            except (TypeError, ValueError):
                pass
    mech = paper_mechanism_metrics(np.asarray(y_log, float), np.asarray(p_log, float))
    out["beta_log"] = float(mech["Beta_log"])
    out["Cov_log_residual_log_price"] = float(mech["Cov_log_residual_log_price"])
    out["dCor_e_y"] = float(mech["dCor_e_y"])
    if "RMSE_log" not in out:
        out["RMSE_log"] = float(np.sqrt(np.mean((np.asarray(p_log, float)
                                                 - np.asarray(y_log, float)) ** 2)))
    # The frozen Delta_NL estimator requires unique identifiers within a split (its fold
    # rule is sha256(salt|str(id)) % 5).  The concatenated pooled-OOF (D2) sample contains
    # 20,988 rows twice because the fold-6 and fold-7 validation blocks overlap, so a
    # composite deterministic identifier is supplied there.  It is still a function of the
    # observation identity only and is independent of the model and predictions, which is
    # exactly what the frozen specification requires.
    dn = estimate_delta_nl(np.asarray(y_log, float), np.asarray(p_log, float),
                           row_ids if dnl_ids is None else dnl_ids)
    out["Delta_NL"] = float(dn["Delta_NL"])
    out["Delta_NL_raw"] = float(dn["Delta_NL_raw"])
    return out


# =========================================================== MODE: roots + grid
def mode_roots() -> int:
    opened = []
    rows, roots = [], {}
    for ref in ("C", "A"):
        Rk, fold_info = [], []
        for k in range(1, 8):
            blk = f"fold_{k}_train"
            d, ybar, h = load_eval(ref, blk)
            opened.append(str(c.REPO.joinpath(zref().query(
                "cell_id==@ref and block_id==@blk").iloc[0].eval_predictions)))
            y = d.y_true_log.to_numpy(); f0 = d.y_pred_log.to_numpy()
            cov, var, R = beta_ratio(y, f0)
            Rk.append(R)
            fold_info.append({"fold": k, "n_val": int(len(y)), "ybar_T": ybar,
                              "Cov_Vk_f0_y": cov, "Var_Vk_y": var, "R_k": R,
                              "beta_k_at_b1": R - 1.0, "eval_pred_sha256": h})
        Rk = np.asarray(Rk, dtype=float)
        b_zero_cvmean = float(1.0 / np.mean(Rk))

        # pooled OOF (D2): concatenate the 7 validation blocks WITH fold-specific centers
        ys, f0s, ybars, rids = [], [], [], []
        for k in range(1, 8):
            d, ybar, _ = load_eval(ref, f"fold_{k}_train")
            ys.append(d.y_true_log.to_numpy()); f0s.append(d.y_pred_log.to_numpy())
            ybars.append(np.full(len(d), ybar)); rids.append(d.row_id.to_numpy())
        y = np.concatenate(ys); f0 = np.concatenate(f0s); yb = np.concatenate(ybars)
        cy = y - float(np.mean(y))
        V = float(np.mean(cy ** 2))
        P = float(np.mean(yb * cy))        # non-zero: fold centers correlate with c
        Q = float(np.mean(f0 * cy))
        denom = Q - P
        if abs(denom) < 1e-18:
            raise c.ProtocolViolation(f"{ref}: degenerate pooled-OOF denominator")
        b_zero_pooled = float((V - P) / denom)
        beta_pooled_b1 = float(((1 - 1.0) * P + 1.0 * Q - V) / V)

        bs = pd.read_csv(T / "b_star_diagnostics.csv")
        bsr = bs[bs.cell_id == ref]
        roots[ref] = {
            "b_zero_cvmean": b_zero_cvmean,
            "b_zero_pooled_oof": b_zero_pooled,
            "mean_R_k": float(np.mean(Rk)),
            "beta_cvmean_at_b1": float(np.mean(Rk) - 1.0),
            "beta_pooled_oof_at_b1": beta_pooled_b1,
            "pooled_V_var_y": V, "pooled_P_cov_ybar_c": P, "pooled_Q_cov_f0_c": Q,
            "n_pooled_oof": int(len(y)),
            "b_star_train_min_IN_SAMPLE_DIAGNOSTIC_ONLY": float(bsr.b_star_train.min()),
            "b_star_train_max_IN_SAMPLE_DIAGNOSTIC_ONLY": float(bsr.b_star_train.max()),
            "beta_log_train_min_IN_SAMPLE": float(bsr.beta_log_train.min()),
            "beta_log_train_max_IN_SAMPLE": float(bsr.beta_log_train.max()),
            "folds": fold_info,
        }
        for fi in fold_info:
            rows.append({"reference": ref, "display_name": REFS[ref], "role": REF_ROLE[ref],
                         "coordinate": "D1_fold", **fi})
        rows.append({"reference": ref, "display_name": REFS[ref], "role": REF_ROLE[ref],
                     "coordinate": "D1_CVmean", "fold": None,
                     "n_val": int(sum(f["n_val"] for f in fold_info)),
                     "R_k": float(np.mean(Rk)), "beta_k_at_b1": float(np.mean(Rk) - 1.0),
                     "b_zero": b_zero_cvmean})
        rows.append({"reference": ref, "display_name": REFS[ref], "role": REF_ROLE[ref],
                     "coordinate": "D2_pooled_oof", "fold": None, "n_val": int(len(y)),
                     "R_k": None, "beta_k_at_b1": beta_pooled_b1, "b_zero": b_zero_pooled})
    c.write_table(pd.DataFrame(rows), T / "posthoc_development_roots.csv")

    # ------------------------------------------------------------------ b_max
    cand = {"b_zero_cvmean_C": roots["C"]["b_zero_cvmean"],
            "b_zero_pooled_oof_C": roots["C"]["b_zero_pooled_oof"],
            "b_zero_cvmean_A": roots["A"]["b_zero_cvmean"],
            "b_zero_pooled_oof_A": roots["A"]["b_zero_pooled_oof"]}
    b_ref_max = max(cand.values())
    b_max = 1.0 + OVERSHOOT * (b_ref_max - 1.0)

    # ---- coverage verification, development information only -----------------
    v4 = (c.V12 / "analysis" / "data_id=d4929d43ec19badf" / "split_id=3d464d4a611b131b"
          / "penalty_path_analysis" / "transition_regions_paper_assets_v4_delta_nl_bends"
          / "tables" / "combined_path_table_v4_analysis_view.csv")
    fp = pd.read_csv(v4)
    fam_rng = {}
    for fam in ("Direct", "Surrogate"):
        s = fp[fp.family == fam]["Beta_log__CV_mean"].dropna()
        fam_rng[fam] = {"min": float(s.min()), "max": float(s.max())}

    def beta_of(ref, b):     # D1 CV-mean coordinate
        return b * roots[ref]["mean_R_k"] - 1.0

    def b_of(ref, beta):
        return (1.0 + beta) / roots[ref]["mean_R_k"]

    three_way_low = max(fam_rng["Direct"]["min"], fam_rng["Surrogate"]["min"],
                        beta_of("C", 1.0), beta_of("A", 1.0))
    three_way_high = min(fam_rng["Direct"]["max"], fam_rng["Surrogate"]["max"],
                         beta_of("C", b_max), beta_of("A", b_max))
    ext = [-0.06, -0.03, 0.0]
    coverage = {
        "direct_cvmean_range": fam_rng["Direct"], "surrogate_cvmean_range": fam_rng["Surrogate"],
        "beta_cvmean_at_b1": {r: beta_of(r, 1.0) for r in ("C", "A")},
        "beta_cvmean_at_bmax": {r: beta_of(r, b_max) for r in ("C", "A")},
        "three_way_common_support_cvmean": [three_way_low, three_way_high],
        "common_support_lower_endpoint_set_by": (
            "post-hoc at b=1" if abs(three_way_low - max(beta_of("C", 1.0), beta_of("A", 1.0))) < 1e-12
            else "Direct/Surrogate family minimum"),
        "reaches_beta_zero": {r: bool(beta_of(r, b_max) >= 0.0) for r in ("C", "A")},
        "b_for_beta_zero": {r: b_of(r, 0.0) for r in ("C", "A")},
        "reaches_direct_upper_common_support": {
            r: bool(beta_of(r, b_max) >= fam_rng["Direct"]["max"]) for r in ("C", "A")},
        "b_for_direct_upper_common_support": {r: b_of(r, fam_rng["Direct"]["max"])
                                              for r in ("C", "A")},
        "ext_targets": {str(e): {r: {"b": b_of(r, e), "within_grid": bool(b_of(r, e) <= b_max)}
                                 for r in ("C", "A")} for e in ext},
        "all_requirements_met": None,
    }
    ok = (all(coverage["reaches_beta_zero"].values())
          and all(coverage["reaches_direct_upper_common_support"].values())
          and all(v[r]["within_grid"] for v in coverage["ext_targets"].values()
                  for r in ("C", "A"))
          and three_way_high >= three_way_low)
    coverage["all_requirements_met"] = bool(ok)
    if not ok:
        raise c.ProtocolViolation(f"b_max fails development coverage: {json.dumps(coverage, indent=2)}")

    # ------------------------------------------------------------------- grid
    base = np.linspace(1.0, b_max, N_BASE)
    anchors = {"b_1": 1.0, **cand, "b_max": b_max}
    grid = np.unique(np.round(np.concatenate([base, np.array(list(cand.values()))]), 15))
    grid = np.sort(grid)
    grid = grid[grid >= 1.0 - 1e-15]
    if grid[0] != 1.0:
        grid = np.concatenate([[1.0], grid])
    payload = {
        "schema_version": 1,
        "stage": "STAGE_2_CENTERED_SPREAD",
        "frozen_at_utc": datetime.datetime.utcnow().isoformat() + "Z",
        "construction_rule": (
            f"{N_BASE} equally spaced base values from 1 to b_max (LINEAR in b, because "
            "beta_log is exactly linear in b and the interval is narrow), then force-add the "
            "exact development anchors, sort and deduplicate. Slightly more than "
            f"{N_BASE} values is accepted because exact scientific anchors outrank row count."),
        "b_max_rule": ("b_max = 1 + 1.25 * (max(all four development roots) - 1)  "
                       "== true 25% overshoot in the adjustment (b-1). "
                       "NOT 1.25 * b_star, and NOT derived from b_star_train."),
        "b_star_train_status": ("IN-SAMPLE THEORY DIAGNOSTIC ONLY -- explicitly NOT used to "
                               "define any empirical endpoint, root or grid bound"),
        "primary_map": "f_b(x) = ybar_T + b * (f0(x) - ybar_T)",
        "b_equals_one_fast_path": "returns the cached prediction array unchanged (bitwise)",
        "development_only": True,
        "files_read_during_root_construction": sorted(set(opened)),
        "no_heldout_or_2025_outcome_read": True,
        "development_roots": roots,
        "b_reference_max_used": b_ref_max,
        "b_max": float(b_max),
        "special_anchors": anchors,
        "coverage_verification": coverage,
        "n_grid": int(grid.size),
        "b_values": [float(x) for x in grid],
    }
    body = json.dumps(payload, indent=2, sort_keys=True, default=str)
    gh = hashlib.sha256(body.encode()).hexdigest()
    payload["grid_sha256"] = gh
    c.write_json(c.CONFIGS / "b_grid_frozen.json", payload)
    c.write_json(c.CONFIGS / "b_grid_frozen_hash.json",
                 {"grid_sha256_of_payload_without_hash_field": gh,
                  "file_sha256": c.sha256_file(c.CONFIGS / "b_grid_frozen.json"),
                  "frozen_at_utc": payload["frozen_at_utc"],
                  "n_grid": payload["n_grid"], "b_max": payload["b_max"],
                  "assertion": ("this hash was written BEFORE any held-out or 2025 comparator "
                                "outcome was read; the path mode validates it before running")})
    print(json.dumps({k: payload[k] for k in
                      ("b_reference_max_used", "b_max", "n_grid", "special_anchors")},
                     indent=2, default=str))
    print("coverage:", json.dumps(coverage, indent=2, default=str))
    print("grid_sha256:", gh)
    return 0


def _grid():
    p = c.CONFIGS / "b_grid_frozen.json"
    if not p.exists():
        raise c.ProtocolViolation("b_grid_frozen.json missing -- run --mode roots first")
    d = json.loads(p.read_text())
    h = json.loads((c.CONFIGS / "b_grid_frozen_hash.json").read_text())
    if h["file_sha256"] != c.sha256_file(p):
        raise c.ProtocolViolation("b_grid_frozen.json changed after being hashed")
    return d


# ================================================================ MODE: path
def mode_path(ref: str, ev: str) -> int:
    g = _grid()
    print(f"[path] grid hash validated ({g['n_grid']} values, b_max={g['b_max']:.8f})", flush=True)
    bvals = g["b_values"]
    anchors = g["special_anchors"]

    if ev == "pooled_oof":
        ys, f0s, ybs, rids = [], [], [], []
        for k in range(1, 8):
            d, yb, _ = load_eval(ref, f"fold_{k}_train")
            ys.append(d.y_true_log.to_numpy()); f0s.append(d.y_pred_log.to_numpy())
            ybs.append(np.full(len(d), yb)); rids.append(d.row_id.to_numpy())
        y = np.concatenate(ys); f0 = np.concatenate(f0s)
        yb_vec = np.concatenate(ybs); rid = np.concatenate(rids)
        fold_tag = np.concatenate([np.full(len(v), k) for k, v in enumerate(ys, start=1)])
        dnl_ids = np.array([f"{int(t)}|{int(r)}" for t, r in zip(fold_tag, rid)], dtype=object)
        ytr = y  # pooled OOF has no single fitting block; use its own sample for level ref
        per_row_center = True
    else:
        blk = BLOCK_FOR[ev]
        d, yb, _ = load_eval(ref, blk)
        y = d.y_true_log.to_numpy(); f0 = d.y_pred_log.to_numpy(); rid = d.row_id.to_numpy()
        zf = zref(); r = zf[(zf.cell_id == ref) & (zf.block_id == blk)].iloc[0]
        ytr = pd.read_parquet(c.REPO / r.train_predictions).y_true_log.to_numpy()
        yb_vec = None
        dnl_ids = None
        per_row_center = False

    cov, var, R = beta_ratio(y, f0)
    rows = []
    for b in bvals:
        t0 = time.perf_counter()
        if per_row_center:
            p = f0 if b == 1.0 else (yb_vec + float(b) * (f0 - yb_vec))
        else:
            p = centered_map(f0, yb, b)
        m = metrics_from(y, p, ytr, rid, dnl_ids=dnl_ids)
        tags = [k for k, v in anchors.items() if abs(v - b) < 1e-15]
        rows.append({
            "reference": ref, "display_name": REFS[ref], "role": REF_ROLE[ref],
            "map": "ybar_T-centered (theorem-matched primary)",
            "evaluation": ev, "b": float(b), "b_minus_1": float(b - 1.0),
            "is_b1": bool(b == 1.0), "special_anchors": ";".join(sorted(tags)),
            "n": int(len(y)), "ybar_T": (None if per_row_center else float(yb)),
            "Cov_V_f0_y": cov, "Var_V_y": var, "R_ratio": R,
            "beta_log_predicted_linear": (None if per_row_center else float(b * R - 1.0)),
            **m, "seconds": time.perf_counter() - t0,
        })
        if len(rows) % 25 == 0:
            print(f"[path] {ref}/{ev} {len(rows)}/{len(bvals)}", flush=True)
    df = pd.DataFrame(rows)
    c.write_table(df, T / f"centered_spread_path__{ref}__{ev}.csv")
    print(f"[path] wrote centered_spread_path__{ref}__{ev}.csv ({len(df)} rows)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["roots", "path"])
    ap.add_argument("--ref", choices=["A", "C"])
    ap.add_argument("--eval", dest="ev", choices=ALL_EVALS)
    ap.add_argument("--shard", type=int, default=-1)
    a = ap.parse_args()
    if a.mode == "roots":
        return mode_roots()
    if a.shard >= 0:
        combos = [(r, e) for r in ("C", "A") for e in ALL_EVALS]
        r, e = combos[a.shard]
        return mode_path(r, e)
    return mode_path(a.ref, a.ev)


if __name__ == "__main__":
    raise SystemExit(main())
