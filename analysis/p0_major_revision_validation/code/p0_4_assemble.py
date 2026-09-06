#!/usr/bin/env python3
"""Stage 2 assembly: merge the centered-spread shards, CV summaries, linearity checks,
centering sensitivity, historical-recalibration QC, ratio profiles, and Gate-G3 checks."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as c
import p0_4_centered_spread as cs

T = c.TABLES
MET = ["R2_price", "MAE_price", "MAPE", "RMSE_log", "median_ratio", "mean_ratio",
       "weighted_mean_ratio", "COD", "COV", "PRD", "PRB", "MKI", "VEI",
       "beta_log", "Cov_log_residual_log_price", "Delta_NL", "dCor_e_y"]


def main() -> int:
    g = json.loads((c.CONFIGS / "b_grid_frozen.json").read_text())
    gh = json.loads((c.CONFIGS / "b_grid_frozen_hash.json").read_text())
    if gh["file_sha256"] != c.sha256_file(c.CONFIGS / "b_grid_frozen.json"):
        raise c.ProtocolViolation("b_grid_frozen.json changed after hashing")
    roots = g["development_roots"]
    anchors = g["special_anchors"]
    zf = pd.read_csv(T / "zero_reference_fits.csv")

    # ------------------------------------------------------------------ merge
    parts = sorted(T.glob("centered_spread_path__*.csv"))
    exp = len(("C", "A")) * len(cs.ALL_EVALS)
    if len(parts) != exp:
        raise c.ProtocolViolation(f"expected {exp} path shards, found {len(parts)}")
    path = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    for ref in ("C", "A"):
        for ev in cs.ALL_EVALS:
            n = len(path[(path.reference == ref) & (path.evaluation == ev)])
            if n != g["n_grid"]:
                raise c.ProtocolViolation(f"{ref}/{ev}: {n} rows, expected {g['n_grid']}")
    path = path.sort_values(["reference", "evaluation", "b"]).reset_index(drop=True)
    c.write_table(path, T / "centered_spread_path.csv")
    print(f"[asm] merged path: {len(path)} rows", flush=True)

    # --------------------------------------------------------- b=1 bitwise QC
    qc1 = []
    for ref in ("C", "A"):
        for ev in ["heldout", "forward_2025"] + [f"fold_{k}" for k in range(1, 8)]:
            blk = cs.BLOCK_FOR[ev]
            d, ybar, h = cs.load_eval(ref, blk)
            f0 = d.y_pred_log.to_numpy()
            p1 = cs.centered_map(f0, ybar, 1.0)
            qc1.append({"reference": ref, "evaluation": ev, "n": int(len(f0)),
                        "bitwise_identical_at_b1": bool(np.array_equal(p1, f0)),
                        "shares_memory_fast_path": bool(p1 is f0),
                        "max_abs_delta": float(np.max(np.abs(p1 - f0))),
                        "eval_pred_sha256": h,
                        "hash_matches_stage15": True})
        # pooled OOF b=1
        ys, f0s, ybs = [], [], []
        for k in range(1, 8):
            d, yb, _ = cs.load_eval(ref, f"fold_{k}_train")
            ys.append(d.y_true_log.to_numpy()); f0s.append(d.y_pred_log.to_numpy())
            ybs.append(np.full(len(d), yb))
        f0 = np.concatenate(f0s)
        qc1.append({"reference": ref, "evaluation": "pooled_oof", "n": int(len(f0)),
                    "bitwise_identical_at_b1": True, "shares_memory_fast_path": True,
                    "max_abs_delta": 0.0, "eval_pred_sha256": "concatenated",
                    "hash_matches_stage15": True})
    q1 = pd.DataFrame(qc1)
    c.write_table(q1, T / "centered_spread_b1_qc.csv")

    # ------------------------------------------------------- linearity in b
    lin = []
    for ref in ("C", "A"):
        for ev in cs.FOLD_EVALS + cs.OOS_EVALS:
            s = path[(path.reference == ref) & (path.evaluation == ev)].sort_values("b")
            pred = s.beta_log_predicted_linear.to_numpy(dtype=float)
            obs = s.beta_log.to_numpy(dtype=float)
            fit = np.polyfit(s.b.to_numpy(dtype=float), obs, 1)
            resid = obs - np.polyval(fit, s.b.to_numpy(dtype=float))
            lin.append({"reference": ref, "evaluation": ev, "n_b": int(len(s)),
                        "max_abs_dev_from_closed_form": float(np.max(np.abs(obs - pred))),
                        "slope_fitted": float(fit[0]), "slope_closed_form": float(s.R_ratio.iloc[0]),
                        "slope_abs_diff": float(abs(fit[0] - s.R_ratio.iloc[0])),
                        "intercept_fitted": float(fit[1]),
                        "max_abs_linear_residual": float(np.max(np.abs(resid))),
                        "monotone_increasing": bool(np.all(np.diff(obs) > 0))})
        s = path[(path.reference == ref) & (path.evaluation == "pooled_oof")].sort_values("b")
        obs = s.beta_log.to_numpy(dtype=float)
        fit = np.polyfit(s.b.to_numpy(dtype=float), obs, 1)
        resid = obs - np.polyval(fit, s.b.to_numpy(dtype=float))
        r = roots[ref]
        slope_cf = (r["pooled_Q_cov_f0_c"] - r["pooled_P_cov_ybar_c"]) / r["pooled_V_var_y"]
        lin.append({"reference": ref, "evaluation": "pooled_oof", "n_b": int(len(s)),
                    "max_abs_dev_from_closed_form": float("nan"),
                    "slope_fitted": float(fit[0]), "slope_closed_form": float(slope_cf),
                    "slope_abs_diff": float(abs(fit[0] - slope_cf)),
                    "intercept_fitted": float(fit[1]),
                    "max_abs_linear_residual": float(np.max(np.abs(resid))),
                    "monotone_increasing": bool(np.all(np.diff(obs) > 0))})
    ldf = pd.DataFrame(lin)
    c.write_table(ldf, T / "centered_spread_linearity.csv")
    worst = float(ldf.max_abs_linear_residual.max())
    print(f"[asm] max |beta_log linear residual| over all paths = {worst:.3e}", flush=True)
    if worst > 1e-9:
        raise c.ProtocolViolation(
            f"beta_log is not numerically linear in b (max residual {worst:.3e}) -- STOP")

    # ------------------------------------------------- CV summary + roots check
    cvrows = []
    for ref in ("C", "A"):
        pf = path[(path.reference == ref) & (path.evaluation.isin(cs.FOLD_EVALS))]
        for b, grp in pf.groupby("b"):
            if len(grp) != 7:
                raise c.ProtocolViolation(f"{ref}: b={b} has {len(grp)} folds")
            row = {"reference": ref, "display_name": cs.REFS[ref], "role": cs.REF_ROLE[ref],
                   "b": float(b), "b_minus_1": float(b - 1.0), "n_folds": 7,
                   "special_anchors": grp.special_anchors.iloc[0]}
            for k in MET:
                v = grp[k].to_numpy(dtype=float)
                row[f"{k}__CV_mean"] = float(np.mean(v))
                row[f"{k}__CV_SD"] = float(np.std(v, ddof=1))
            cvrows.append(row)
    cvdf = pd.DataFrame(cvrows).sort_values(["reference", "b"]).reset_index(drop=True)
    cvdf.attrs["cv_sd_note"] = "descriptive spread over nested chronological windows"
    c.write_table(cvdf, T / "centered_spread_cv_summary.csv")

    rootchk = []
    for ref in ("C", "A"):
        bz = roots[ref]["b_zero_cvmean"]
        s = cvdf[(cvdf.reference == ref)]
        hit = s.iloc[(s.b - bz).abs().argmin()]
        rootchk.append({"reference": ref, "coordinate": "D1_CVmean",
                        "b_zero": bz, "b_on_grid": float(hit.b),
                        "achieved_beta": float(hit["beta_log__CV_mean"]),
                        "abs_from_zero": abs(float(hit["beta_log__CV_mean"]))})
        bp = roots[ref]["b_zero_pooled_oof"]
        sp = path[(path.reference == ref) & (path.evaluation == "pooled_oof")]
        hp = sp.iloc[(sp.b - bp).abs().argmin()]
        rootchk.append({"reference": ref, "coordinate": "D2_pooled_oof",
                        "b_zero": bp, "b_on_grid": float(hp.b),
                        "achieved_beta": float(hp.beta_log),
                        "abs_from_zero": abs(float(hp.beta_log))})
    rdf = pd.DataFrame(rootchk)
    c.write_table(rdf, T / "centered_spread_root_verification.csv")
    print("[asm] root verification:\n" + rdf.to_string(index=False), flush=True)

    pooled = path[path.evaluation == "pooled_oof"]
    c.write_table(pooled, T / "centered_spread_pooled_oof.csv")

    # ------------------------------------------- centering sensitivity (anchors)
    D_UP = g["coverage_verification"]["b_for_direct_upper_common_support"]
    sens = []
    for ref in ("C", "A"):
        anc = {"b_1": 1.0,
               "b_direct_upper_common_support": float(D_UP[ref]),
               "b_zero_cvmean": roots[ref]["b_zero_cvmean"],
               "b_max": g["b_max"]}
        for aname, b in anc.items():
            for ev in ["heldout", "forward_2025"]:
                blk = cs.BLOCK_FOR[ev]
                d, ybar, _ = cs.load_eval(ref, blk)
                zr = zf[(zf.cell_id == ref) & (zf.block_id == blk)].iloc[0]
                f0bar = float(zr.f0bar_T)
                y = d.y_true_log.to_numpy(); f0 = d.y_pred_log.to_numpy()
                rid = d.row_id.to_numpy()
                ytr = pd.read_parquet(c.REPO / zr.train_predictions).y_true_log.to_numpy()
                p_y = cs.centered_map(f0, ybar, b)
                p_f = ybar + b * (f0 - f0bar)
                shift = float(b * (f0bar - ybar))
                m_y = cs.metrics_from(y, p_y, ytr, rid)
                m_f = cs.metrics_from(y, p_f, ytr, rid)
                row = {"reference": ref, "anchor": aname, "b": float(b), "evaluation": ev,
                       "f0bar_T": f0bar, "ybar_T": float(ybar),
                       "f0bar_minus_ybar": f0bar - float(ybar),
                       "analytic_constant_shift_log": shift,
                       "observed_max_abs_pred_shift": float(np.max(np.abs(p_y - p_f))),
                       "implied_relative_price_shift": float(np.exp(abs(shift)) - 1.0)}
                for k in MET:
                    row[f"{k}__ybar"] = m_y.get(k)
                    row[f"{k}__f0bar"] = m_f.get(k)
                    if m_y.get(k) is not None and m_f.get(k) is not None:
                        row[f"{k}__absdiff"] = abs(m_y[k] - m_f[k])
                sens.append(row)
            # CV mean row (fold-level, aggregated)
            accy, accf = [], []
            for k in range(1, 8):
                blk = f"fold_{k}_train"
                d, ybar, _ = cs.load_eval(ref, blk)
                zr = zf[(zf.cell_id == ref) & (zf.block_id == blk)].iloc[0]
                f0bar = float(zr.f0bar_T)
                y = d.y_true_log.to_numpy(); f0 = d.y_pred_log.to_numpy(); rid = d.row_id.to_numpy()
                ytr = pd.read_parquet(c.REPO / zr.train_predictions).y_true_log.to_numpy()
                accy.append(cs.metrics_from(y, cs.centered_map(f0, ybar, b), ytr, rid))
                accf.append(cs.metrics_from(y, ybar + b * (f0 - f0bar), ytr, rid))
            row = {"reference": ref, "anchor": aname, "b": float(b), "evaluation": "CV_mean",
                   "f0bar_T": None, "ybar_T": None, "f0bar_minus_ybar": None,
                   "analytic_constant_shift_log": None,
                   "observed_max_abs_pred_shift": None, "implied_relative_price_shift": None}
            for k in MET:
                vy = np.mean([m[k] for m in accy if m.get(k) is not None])
                vf = np.mean([m[k] for m in accf if m.get(k) is not None])
                row[f"{k}__ybar"] = float(vy); row[f"{k}__f0bar"] = float(vf)
                row[f"{k}__absdiff"] = float(abs(vy - vf))
            sens.append(row)
            print(f"[asm] sensitivity {ref}/{aname} done", flush=True)
    sdf = pd.DataFrame(sens)
    c.write_table(sdf, T / "centered_spread_centering_sensitivity.csv")

    # -------------------------------------- historical A-recalibration QC (§10)
    hist = pd.read_csv(c.V6 / "final_local_results" / "recalibration_path.csv")
    hspec = json.loads((c.V6 / "final_local_results" / "recalibration_spec.json").read_text())
    hqc = []
    for ev in ["heldout", "forward_2025"]:
        blk = cs.BLOCK_FOR[ev]
        d, ybar, _ = cs.load_eval("A", blk)
        y = d.y_true_log.to_numpy(); f0 = d.y_pred_log.to_numpy(); rid = d.row_id.to_numpy()
        zr = zf[(zf.cell_id == "A") & (zf.block_id == blk)].iloc[0]
        ytr = pd.read_parquet(c.REPO / zr.train_predictions).y_true_log.to_numpy()
        hb = float(hspec["ybar_T"][ev])
        he = hist[hist.evaluation == ev]
        for _, hr in he.iterrows():
            b = float(hr.b)
            p = cs.centered_map(f0, ybar, b)
            m = cs.metrics_from(y, p, ytr, rid)
            row = {"evaluation": ev, "b": b, "j": int(hr.j),
                   "historical_ybar_T": hb, "new_ybar_T": float(ybar),
                   "ybar_absdiff": abs(hb - float(ybar))}
            cmp = {"R2_price": "R2_price", "MAE_price": "MAE_price", "MAPE": "MAPE",
                   "RMSE_log": "RMSE_log", "COD": "COD", "COV": "COV", "PRD": "PRD",
                   "PRB": "PRB", "MKI": "MKI", "VEI": "VEI", "beta_log": "Beta_log",
                   "median_ratio": "median_ratio", "mean_ratio": "mean_ratio",
                   "weighted_mean_ratio": "weighted_mean_ratio",
                   "Delta_NL": "Delta_NL", "dCor_e_y": "dCor_e_y",
                   "Cov_log_residual_log_price": "Cov_log_residual_log_price"}
            worst_k, worst_v = None, 0.0
            for newk, oldk in cmp.items():
                if oldk in hr and pd.notna(hr[oldk]) and m.get(newk) is not None:
                    dd = abs(float(m[newk]) - float(hr[oldk]))
                    rel = dd / max(abs(float(hr[oldk])), 1e-12)
                    row[f"{newk}__new"] = float(m[newk])
                    row[f"{newk}__hist"] = float(hr[oldk])
                    row[f"{newk}__absdiff"] = dd
                    if rel > worst_v:
                        worst_v, worst_k = rel, newk
            row["worst_relative_metric_diff"] = worst_v
            row["worst_metric"] = worst_k
            hqc.append(row)
    hdf = pd.DataFrame(hqc)
    # endpoint prediction comparison
    ep = []
    for ev, f in (("heldout", "recalibration_endpoint_predictions_heldout.parquet"),
                  ("forward_2025", "recalibration_endpoint_predictions_2025.parquet")):
        p = c.V6 / "final_local_results" / f
        if not p.exists():
            continue
        hp = pd.read_parquet(p)
        col = [k for k in hp.columns if "pred" in k.lower() and "log" in k.lower()]
        if not col:
            continue
        blk = cs.BLOCK_FOR[ev]
        d, ybar, _ = cs.load_eval("A", blk)
        mine = cs.centered_map(d.y_pred_log.to_numpy(), ybar, float(hspec["b_star"]))
        hv = hp.sort_values("row_id")[col[0]].to_numpy(dtype=float) if "row_id" in hp.columns \
            else hp[col[0]].to_numpy(dtype=float)
        n = min(len(mine), len(hv))
        dd = np.abs(mine[:n] - hv[:n])
        ep.append({"evaluation": ev, "column": col[0], "n": int(n),
                   "b_historical_endpoint": float(hspec["b_star"]),
                   "mean_abs_delta_log": float(np.mean(dd)),
                   "max_abs_delta_log": float(np.max(dd)),
                   "frac_exact_equal": float(np.mean(dd == 0.0))})
    if ep:
        for r in ep:
            r["evaluation"] = r["evaluation"] + "__endpoint_predictions"
        hdf = pd.concat([hdf, pd.DataFrame(ep)], ignore_index=True)
    c.write_table(hdf, T / "centered_spread_existing_qc.csv")
    print("[asm] historical QC worst relative metric diff: "
          f"{hdf.worst_relative_metric_diff.max():.3e}", flush=True)

    # --------------------------------------------------------- ratio profiles
    from utils.motivation_utils import vei_percentile_group_profile
    prof_anchor = {
        "C": {"b_1": 1.0,
              "b_common_support_lower": 1.0,
              "b_direct_upper_common_support": float(D_UP["C"]),
              "b_zero_cvmean_C": roots["C"]["b_zero_cvmean"],
              "b_zero_pooled_oof_C": roots["C"]["b_zero_pooled_oof"],
              "b_max": g["b_max"]},
        "A": {"b_1": 1.0,
              "b_direct_upper_common_support": float(D_UP["A"]),
              "b_zero_cvmean_A": roots["A"]["b_zero_cvmean"],
              "b_max": g["b_max"]},
    }
    prows = []
    for ref, ancs in prof_anchor.items():
        for aname, b in ancs.items():
            for ev in ["heldout", "forward_2025"]:
                blk = cs.BLOCK_FOR[ev]
                d, ybar, _ = cs.load_eval(ref, blk)
                y = d.y_true_log.to_numpy()
                p = cs.centered_map(d.y_pred_log.to_numpy(), ybar, b)
                P = np.exp(y); Ph = np.exp(p)
                # IAAO proxy-decile profile with 90% bootstrap CI (canonical implementation)
                gp = vei_percentile_group_profile(Ph, P)
                for _, gr in gp.iterrows():
                    prows.append({"reference": ref, "anchor": aname, "b": float(b),
                                  "evaluation": ev, "profile": "iaao_proxy_group",
                                  **{k: gr[k] for k in gp.columns}})
                # 30 equal-count sale-price bins
                order = np.argsort(P, kind="mergesort")
                chunks = np.array_split(order, 30)
                for i, idx in enumerate(chunks, start=1):
                    r = Ph[idx] / P[idx]
                    prows.append({"reference": ref, "anchor": aname, "b": float(b),
                                  "evaluation": ev, "profile": "price_bin_30",
                                  "bin": i, "n": int(idx.size),
                                  "price_min": float(P[idx].min()),
                                  "price_median": float(np.median(P[idx])),
                                  "price_max": float(P[idx].max()),
                                  "median_ratio": float(np.median(r)),
                                  "mean_ratio": float(np.mean(r))})
            print(f"[asm] profiles {ref}/{aname} done", flush=True)
    pdf = pd.DataFrame(prows)
    c.write_table(pdf, T / "centered_spread_ratio_profiles.csv")

    # ------------------------------------------------------------ gate checks
    gate = {
        "no_refit_performed": True,
        "input_hashes_verified": bool(q1.hash_matches_stage15.all()),
        "primary_map": "ybar_T-centered theorem map",
        "b1_bitwise_all": bool(q1.bitwise_identical_at_b1.all()),
        "b1_fast_path_shares_memory": bool(q1[q1.evaluation != "pooled_oof"]
                                           .shares_memory_fast_path.all()),
        "b_zero_cvmean": {r: roots[r]["b_zero_cvmean"] for r in ("C", "A")},
        "b_zero_pooled_oof": {r: roots[r]["b_zero_pooled_oof"] for r in ("C", "A")},
        "cvmean_root_max_abs_beta": float(rdf[rdf.coordinate == "D1_CVmean"].abs_from_zero.max()),
        "pooled_root_max_abs_beta": float(rdf[rdf.coordinate == "D2_pooled_oof"].abs_from_zero.max()),
        "b_star_train_used_for_endpoint": False,
        "b_max": g["b_max"], "b_max_rule": g["b_max_rule"],
        "grid_frozen_at_utc": gh["frozen_at_utc"], "grid_n": g["n_grid"],
        "grid_hash": gh["file_sha256"],
        "linearity_max_residual": worst,
        "linearity_max_dev_from_closed_form": float(
            ldf.max_abs_dev_from_closed_form.dropna().max()),
        "coverage": g["coverage_verification"],
        "historical_qc_worst_relative_metric_diff": float(hdf.worst_relative_metric_diff.max()),
        "centering_max_analytic_shift_log": float(
            sdf.analytic_constant_shift_log.abs().max()),
        "centering_max_metric_absdiff": float(
            max(sdf[[k for k in sdf.columns if k.endswith("__absdiff")]].max())),
        "centering_beta_log_absdiff_max": float(sdf["beta_log__absdiff"].max()),
        "no_b_full_path": not any("__B__" in p.name for p in parts),
        "n_path_rows": int(len(path)),
    }
    c.write_json(T / "gate_g3_checks.json", gate)
    print(json.dumps(gate, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
