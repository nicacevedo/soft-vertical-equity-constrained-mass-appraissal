#!/usr/bin/env python3
"""Stage-3: formal matched-beta comparison on the frozen D1 development coordinate.

PRIMARY triple : Direct / Surrogate / C-posthoc      SECONDARY: A-posthoc     B: excluded.

Modes
-----
  freeze    build the deterministic K=6 CORE targets and the matched configuration table
            from DEVELOPMENT information only, then hash configs/matched_beta_frozen.json.
            This mode never opens a held-out or 2025 column.
  evaluate  read the frozen table and report development / held-out / 2025 metrics.
            Every number is an actual fit or an actually evaluated transformation.
            NOTHING is interpolated between rho values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as c
import p0_4_centered_spread as cs

T = c.TABLES
TAU = 0.002
K_CORE = 6
EXT_TARGETS = [-0.06, -0.03, 0.00]
PRIMARY = ["Direct", "Surrogate", "C-posthoc"]
SECONDARY = ["A-posthoc"]

MET = ["R2_price", "MAE_price", "MAPE", "RMSE_log", "median_ratio", "mean_ratio",
       "weighted_mean_ratio", "COD", "COV", "PRD", "PRB", "MKI", "VEI",
       "beta_log", "Cov_log_residual_log_price", "Delta_NL", "dCor_e_y"]
# frozen-path-table column base names
FP = {"R2_price": "R2_price", "MAE_price": "MAE_price", "MAPE": "MAPE", "RMSE_log": "RMSE_log",
      "median_ratio": "median_ratio", "mean_ratio": "mean_ratio",
      "weighted_mean_ratio": "weighted_mean_ratio", "COD": "COD", "COV": "COV", "PRD": "PRD",
      "PRB": "PRB", "MKI": "MKI", "VEI": "VEI", "beta_log": "Beta_log",
      "Cov_log_residual_log_price": "Cov_log_residual_log_price",
      "Delta_NL": "Delta_NL", "dCor_e_y": "dCor_e_y"}


def _v4():
    return pd.read_csv(c.V12 / "analysis" / "data_id=d4929d43ec19badf"
                       / "split_id=3d464d4a611b131b" / "penalty_path_analysis"
                       / "transition_regions_paper_assets_v4_delta_nl_bends" / "tables"
                       / "combined_path_table_v4_analysis_view.csv")


def _dev():
    return pd.read_csv(T / "development_beta_coordinate_audit.csv")


def _posthoc_linear(ref: str, coord: str) -> tuple:
    """Exact (alpha, slope) of the post-hoc relation beta_coord(b) = alpha + slope*b.

    beta_log is exactly linear in b on any fixed evaluation sample. On the D1
    coordinate (equal-weight mean of per-fold beta) the relation is exactly
    alpha = -1, slope = mean_k R_k, because each fold uses its own ybar_T.
    On the POOLED coordinates (D2/D3) the centered map applies a DIFFERENT ybar_T
    per fold, so the pooled covariance picks up a (1-b)*Cov_pool(ybar_T, y) term
    and alpha != -1. The relation stays exactly linear, so it is recovered by an
    exact two-point solve on the frozen b-grid and residual-checked.
    Development information only; no outcome is involved.
    """
    fam = {"C": "C-posthoc", "A": "A-posthoc"}[ref]
    d = _dev()
    t = d[(d.family == fam) & d.b.notna()].sort_values("b")
    b = t.b.to_numpy(dtype=float); y = t[coord].to_numpy(dtype=float)
    if len(t) < 2:
        raise c.ProtocolViolation(f"cannot fit the post-hoc {coord} relation for {fam}")
    slope = (y[-1] - y[0]) / (b[-1] - b[0])
    alpha = y[0] - slope * b[0]
    resid = float(np.max(np.abs(y - (alpha + slope * b))))
    if resid > 1e-12:
        raise c.ProtocolViolation(
            f"{fam} {coord} is not linear in b (max resid {resid:.3e}); the exact "
            f"post-hoc solve is only valid on an exactly linear relation")
    if coord == "beta_D1":
        g = json.loads((c.CONFIGS / "b_grid_frozen.json").read_text())
        mR = float(g["development_roots"][ref]["mean_R_k"])
        if abs(slope - mR) > 1e-12 or abs(alpha + 1.0) > 1e-12:
            raise c.ProtocolViolation(
                f"D1 post-hoc relation disagrees with the frozen mean_R_k "
                f"(slope {slope!r} vs {mR!r}, alpha {alpha!r})")
        return -1.0, mR          # canonical frozen values on the primary coordinate
    return alpha, slope


# ============================================================== MODE: freeze
def mode_freeze(coord: str = "beta_D1", out: str = "matched_beta_frozen.json") -> int:
    d = _dev()
    L = max(d[d.family == f][coord].min() for f in PRIMARY)
    U = min(d[d.family == f][coord].max() for f in PRIMARY)
    print(f"[freeze] {coord} three-way common support = [{L:.8f}, {U:.8f}]", flush=True)

    D = d[(d.family == "Direct") & (d[coord] >= L - 1e-15) & (d[coord] <= U + 1e-15)].copy()
    if len(D) < K_CORE:
        raise c.ProtocolViolation(f"only {len(D)} Direct configs inside the support")

    used, core = set(), []
    for j in range(K_CORE):
        q = L + j / (K_CORE - 1) * (U - L)
        cand = D[~D.rho.isin(used)].copy()
        cand["d"] = (cand[coord] - q).abs()
        cand = cand.sort_values(["d", "rho"])          # tie-break: smaller rho
        r = cand.iloc[0]
        used.add(float(r.rho))
        core.append({"j": j, "q_j": float(q), "direct_rho": float(r.rho),
                     "target": float(r[coord]), "abs_target_minus_q": float(abs(r[coord] - q))})

    rows = []
    for t in core:
        tgt = t["target"]
        # Direct: exact anchor, no fit
        rows.append({"j": t["j"], "target": tgt, "family": "Direct", "role": "PRIMARY",
                     "rho": t["direct_rho"], "b": None, "achieved_dev_beta": tgt,
                     "match_gap": 0.0, "match_mode": "exact_anchor", "attained": True,
                     "requires_new_fit": False})
        # Surrogate: nearest actually-fitted config
        S = d[d.family == "Surrogate"].copy()
        S["gap"] = (S[coord] - tgt).abs()
        s = S.sort_values(["gap", "rho"]).iloc[0]
        ok = bool(s.gap <= TAU)
        rows.append({"j": t["j"], "target": tgt, "family": "Surrogate", "role": "PRIMARY",
                     "rho": float(s.rho), "b": None, "achieved_dev_beta": float(s[coord]),
                     "match_gap": float(s.gap),
                     "match_mode": "nearest_fitted" if ok else "targeted_fit_required",
                     "attained": ok, "requires_new_fit": (not ok)})
        # C / A post-hoc: exact solve on the linear relation, then actual evaluation
        for fam, ref in (("C-posthoc", "C"), ("A-posthoc", "A")):
            al, sl = _posthoc_linear(ref, coord)
            b = (tgt - al) / sl
            rows.append({"j": t["j"], "target": tgt, "family": fam,
                         "role": "PRIMARY" if fam == "C-posthoc" else "SECONDARY",
                         "rho": None, "b": float(b),
                         "achieved_dev_beta": float(al + b * sl),
                         "match_gap": float(abs(al + b * sl - tgt)),
                         "match_mode": "exact_posthoc_solve", "attained": True,
                         "requires_new_fit": False})

    # ------------------------------------------------------------ EXT targets
    ext = []
    for e in EXT_TARGETS:
        S = d[d.family == "Surrogate"].copy()
        S["gap"] = (S[coord] - e).abs()
        s = S.sort_values(["gap", "rho"]).iloc[0]
        smax = float(d[d.family == "Surrogate"][coord].max())
        sok = bool(s.gap <= TAU)
        ext.append({"target": e, "family": "Surrogate", "role": "PRIMARY",
                    "rho": float(s.rho) if sok else None, "b": None,
                    "achieved_dev_beta": float(s[coord]) if sok else smax,
                    "match_gap": float(s.gap), "attained": sok,
                    "match_mode": "nearest_fitted" if sok else "NOT_ATTAINED",
                    "max_achieved_dev_correction": smax})
        dmax = float(d[d.family == "Direct"][coord].max())
        ext.append({"target": e, "family": "Direct", "role": "PRIMARY", "rho": None, "b": None,
                    "achieved_dev_beta": dmax, "match_gap": float(abs(dmax - e)),
                    "attained": False, "match_mode": "NOT_ATTAINED",
                    "max_achieved_dev_correction": dmax})
        for fam, ref in (("C-posthoc", "C"), ("A-posthoc", "A")):
            al, sl = _posthoc_linear(ref, coord); b = (e - al) / sl
            ext.append({"target": e, "family": fam,
                        "role": "PRIMARY" if fam == "C-posthoc" else "SECONDARY",
                        "rho": None, "b": float(b), "achieved_dev_beta": float(al + b * sl),
                        "match_gap": float(abs(al + b * sl - e)), "attained": True,
                        "match_mode": "exact_posthoc_solve",
                        "max_achieved_dev_correction": None})

    payload = {
        "schema_version": 1, "stage": "STAGE_3_MATCHED_BETA",
        "frozen_at_utc": __import__("datetime").datetime.utcnow().isoformat() + "Z",
        "development_coordinate": coord,
        "coordinate_role": ("D1 = FROZEN PRIMARY equal-weight seven-fold mean"
                            if coord == "beta_D1" else "D3 = row-balanced robustness sensitivity"),
        "primary_families": PRIMARY, "secondary_families": SECONDARY,
        "excluded": {"B": "Centered-label native L2 (initialization-aligned) -- decomposition only"},
        "A_does_not_determine": ["common support", "CORE target selection",
                                 "whether a CORE target exists"],
        "tau": TAU, "tau_source": "frozen P0 matching tolerance (plan rev.3 I.3)",
        "K_core": K_CORE,
        "three_way_common_support": [float(L), float(U)],
        "core_construction": ("q_j = L + j/(K-1)*(U-L); c_j = unused fitted Direct config "
                              "minimizing |beta_dev - q_j| (tie-break smaller rho); "
                              "target_j = ACHIEVED beta of c_j, not q_j"),
        "posthoc_linear_relation": {
            r: {"alpha": _posthoc_linear(r, coord)[0], "slope": _posthoc_linear(r, coord)[1],
                "form": "beta(b) = alpha + slope*b (exact; residual <= 1e-12)"}
            for r in ("C", "A")},
        "core_targets": core, "matched_configurations": rows,
        "ext_targets": EXT_TARGETS, "ext_matched": ext,
        "no_oos_read_during_freeze": True,
        "no_oos_interpolation": ("every reported OOS number is an actual fit or an actually "
                                 "evaluated transformation; no metric is interpolated between "
                                 "rho values"),
        "provenance": c.preflight_block(),
    }
    body = json.dumps(payload, indent=2, sort_keys=True, default=str)
    payload["payload_sha256"] = hashlib.sha256(body.encode()).hexdigest()
    p = c.CONFIGS / out
    c.write_json(p, payload)
    c.write_json(c.CONFIGS / out.replace(".json", "_hash.json"),
                 {"file_sha256": c.sha256_file(p), "payload_sha256": payload["payload_sha256"],
                  "frozen_at_utc": payload["frozen_at_utc"],
                  "assertion": ("hashed BEFORE any held-out or 2025 comparator outcome was read; "
                                "the evaluate mode validates this hash before running")})
    print(json.dumps({"support": [L, U], "core": core}, indent=2))
    need = [r for r in rows if r["requires_new_fit"]]
    print(f"[freeze] targeted Surrogate fits required: {len(need)}")
    return 0


# ============================================================ MODE: evaluate
def _fp_lookup(v4, family, rho, reg, met):
    """Actual measured value from the frozen path table (an actual fit)."""
    s = v4[(v4.family == family) & (np.isclose(v4.rho.astype(float), float(rho), rtol=0, atol=1e-12))]
    if len(s) != 1:
        raise c.ProtocolViolation(f"frozen table lookup failed: {family} rho={rho} n={len(s)}")
    col = f"{FP[met]}__{reg}"
    if col not in v4.columns:
        return None
    v = s.iloc[0][col]
    return None if pd.isna(v) else float(v)


_PH_CACHE = {}


def _posthoc_metrics(ref, b, reg):
    """Actual evaluated transformation at the solved b (never interpolated).

    CV_mean and CV_SD are two summaries of the SAME seven fold evaluations, so the
    seven-fold work is computed once per (ref, b) and cached. The cache is keyed on
    development-side quantities only and never on an outcome.
    """
    key = (ref, float(b), reg if reg in ("heldout", "forward_2025") else "cv")
    if key in _PH_CACHE:
        acc = _PH_CACHE[key]
        if reg == "CV_mean":
            return {k: float(np.mean([a[k] for a in acc])) for k in sorted({k for a in acc for k in a})}
        if reg == "CV_SD":
            return {k: float(np.std([a[k] for a in acc], ddof=1)) for k in sorted({k for a in acc for k in a})}
        return acc
    if reg in ("heldout", "forward_2025"):
        blk = cs.BLOCK_FOR[reg]
        d, yb, _ = cs.load_eval(ref, blk)
        zf = pd.read_csv(T / "zero_reference_fits.csv")
        r = zf[(zf.cell_id == ref) & (zf.block_id == blk)].iloc[0]
        ytr = pd.read_parquet(c.REPO / r.train_predictions).y_true_log.to_numpy()
        y = d.y_true_log.to_numpy(); p = cs.centered_map(d.y_pred_log.to_numpy(), yb, b)
        m = cs.metrics_from(y, p, ytr, d.row_id.to_numpy())
        _PH_CACHE[key] = m
        return m
    acc = []
    for k in range(1, 8):
        blk = f"fold_{k}_train"
        d, yb, _ = cs.load_eval(ref, blk)
        zf = pd.read_csv(T / "zero_reference_fits.csv")
        r = zf[(zf.cell_id == ref) & (zf.block_id == blk)].iloc[0]
        ytr = pd.read_parquet(c.REPO / r.train_predictions).y_true_log.to_numpy()
        y = d.y_true_log.to_numpy(); p = cs.centered_map(d.y_pred_log.to_numpy(), yb, b)
        acc.append(cs.metrics_from(y, p, ytr, d.row_id.to_numpy()))
    _PH_CACHE[key] = acc
    keys = sorted({k for a in acc for k in a})
    if reg == "CV_mean":
        return {k: float(np.mean([a[k] for a in acc])) for k in keys}
    return {k: float(np.std([a[k] for a in acc], ddof=1)) for k in keys}


def mode_evaluate(cfg: str = "matched_beta_frozen.json", tag: str = "") -> int:
    p = c.CONFIGS / cfg
    h = json.loads((c.CONFIGS / cfg.replace(".json", "_hash.json")).read_text())
    if h["file_sha256"] != c.sha256_file(p):
        raise c.ProtocolViolation(f"{cfg} changed after being hashed")
    F = json.loads(p.read_text())
    print(f"[eval] frozen config hash validated ({F['frozen_at_utc']})", flush=True)
    v4 = _v4()
    zc = pd.read_csv(T / "zero_control_full.csv")
    regs = ["CV_mean", "CV_SD", "heldout", "forward_2025"]
    fpreg = {"CV_mean": "CV_mean", "CV_SD": "CV_sd", "heldout": "heldout",
             "forward_2025": "forward_2025"}

    def eval_row(rec, target_field):
        fam = rec["family"]; out = []
        for reg in regs:
            base = {"target": rec[target_field], "j": rec.get("j"), "family": fam,
                    "role": rec["role"], "rho": rec.get("rho"), "b": rec.get("b"),
                    "achieved_dev_beta": rec["achieved_dev_beta"],
                    "match_gap": rec["match_gap"], "match_mode": rec["match_mode"],
                    "attained": rec["attained"], "evaluation": reg}
            if not rec["attained"]:
                out.append({**base, **{m: None for m in MET},
                            "source": "NOT_ATTAINED within the frozen design"})
                continue
            if fam in ("Direct", "Surrogate"):
                vals = {m: _fp_lookup(v4, fam, rec["rho"], fpreg[reg], m) for m in MET}
                # rho=0 CV Delta_NL is absent from the frozen table; take the Gate-G2 value
                if vals.get("Delta_NL") is None and float(rec["rho"]) == 0.0 \
                        and reg in ("CV_mean", "CV_SD"):
                    z = zc[(zc.cell_id == "C") & (zc.evaluation == ("CV_mean" if reg == "CV_mean"
                                                                    else "CV_SD"))]
                    if len(z):
                        vals["Delta_NL"] = float(z.iloc[0]["Delta_NL"])
                out.append({**base, **vals,
                            "source": "frozen 82-point path table (actual fit)"})
            else:
                ref = "C" if fam == "C-posthoc" else "A"
                m = _posthoc_metrics(ref, float(rec["b"]), reg)
                out.append({**base, **{k: m.get(k) for k in MET},
                            "source": "actual evaluated theorem-matched transformation"})
        return out

    rows = []
    for rec in F["matched_configurations"]:
        rows += eval_row(rec, "target")
        print(f"[eval] CORE j={rec['j']} {rec['family']} done", flush=True)
    df = pd.DataFrame(rows)
    c.write_table(df, T / f"matched_beta_comparison{tag}.csv")

    erows = []
    for rec in F["ext_matched"]:
        erows += eval_row(rec, "target")
        print(f"[eval] EXT {rec['target']} {rec['family']} done", flush=True)
    edf = pd.DataFrame(erows)
    c.write_table(edf, T / f"matched_beta_ext_targets{tag}.csv")

    # ------------------------------------------------- pairwise primary deltas
    pw = []
    for j in sorted({r["j"] for r in F["matched_configurations"]}):
        for reg in regs:
            g = df[(df.j == j) & (df.evaluation == reg)]
            def get(f):
                s = g[g.family == f]
                return s.iloc[0] if len(s) else None
            D, S, C = get("Direct"), get("Surrogate"), get("C-posthoc")
            for a, b_, lab in ((D, C, "Direct_minus_Cposthoc"),
                               (S, C, "Surrogate_minus_Cposthoc"),
                               (S, D, "Surrogate_minus_Direct")):
                if a is None or b_ is None:
                    continue
                row = {"j": int(j), "target": float(a["target"]), "evaluation": reg,
                       "pair": lab}
                for m in MET:
                    va, vb = a[m], b_[m]
                    row[m] = (float(va) - float(vb)) if (va is not None and vb is not None
                                                         and pd.notna(va) and pd.notna(vb)) else None
                pw.append(row)
    pdf = pd.DataFrame(pw)
    c.write_table(pdf, T / f"matched_beta_pairwise_deltas{tag}.csv")

    # ------------------------------------------- origin (j=0) identity QC
    # At j=0 the three PRIMARY families are the SAME object by construction:
    # Direct rho=0 == Surrogate rho=0 == Cell C, and C-posthoc b=1 returns f0
    # bitwise via the fast path. Their matched-beta deltas are therefore
    # structurally zero and carry no sign information, so j=0 is excluded from
    # crossing detection and asserted instead.
    ident = []
    for reg in regs:
        g = df[(df.j == 0) & (df.evaluation == reg)]
        base = g[g.family == "C-posthoc"]
        if not len(base):
            continue
        for fam in ("Direct", "Surrogate"):
            o = g[g.family == fam]
            if not len(o):
                continue
            worst, worst_m, worst_rel, worst_rel_m = 0.0, None, 0.0, None
            for m in MET:
                a, b_ = base.iloc[0][m], o.iloc[0][m]
                if a is None or b_ is None or pd.isna(a) or pd.isna(b_):
                    continue
                a, b_ = float(a), float(b_)
                dv = abs(a - b_)
                rel = dv / max(abs(a), abs(b_), 1e-300)
                if dv > worst:
                    worst, worst_m = dv, m
                if rel > worst_rel:
                    worst_rel, worst_rel_m = rel, m
            ident.append({"evaluation": reg, "pair": f"{fam}_vs_Cposthoc_at_j0",
                          "max_abs_metric_difference": worst, "worst_metric_abs": worst_m,
                          "max_rel_metric_difference": worst_rel,
                          "worst_metric_rel": worst_rel_m,
                          "agrees_within_1e9_relative": bool(worst_rel <= 1e-9),
                          "identity": ("Direct rho=0 == Surrogate rho=0 == Cell C, and "
                                       "C-posthoc at b=1 returns f0 via the bitwise fast path"),
                          "why_not_bitwise": ("the two sides come from DIFFERENT sources: "
                                              "Direct/Surrogate rho=0 metrics are read from the "
                                              "frozen 82-point path CSV, C-posthoc b=1 metrics are "
                                              "recomputed here from cached predictions; the "
                                              "residual is CSV serialization precision, so this "
                                              "check also validates the Stage-3 recomputation "
                                              "against the frozen table")})
    idf = pd.DataFrame(ident)
    c.write_table(idf, T / f"matched_beta_origin_identity{tag}.csv")
    bad = idf[~idf.agrees_within_1e9_relative]
    if len(bad):
        raise c.ProtocolViolation(
            f"j=0 origin identity violated in {len(bad)} rows "
            f"(max relative {bad.max_rel_metric_difference.max():.3e} > 1e-9)")
    print(f"[eval] origin identity holds on all {len(idf)} j=0 primary comparisons "
          f"(max relative {idf.max_rel_metric_difference.max():.3e}, "
          f"max absolute {idf.max_abs_metric_difference.max():.3e}) -- this also validates "
          f"the Stage-3 recomputation against the frozen path table")

    # ------------------------------------------------------------- crossings
    # Detected on j>=1 only (see the origin-identity note above).
    pdf_x = pdf[pdf.j >= 1]
    cr = []
    for reg in regs:
        for m in ("R2_price", "MAE_price", "RMSE_log", "Delta_NL", "dCor_e_y", "COD", "VEI"):
            s = pdf_x[(pdf_x.evaluation == reg) & (pdf_x.pair == "Direct_minus_Cposthoc")].sort_values("j")
            v = s[m].to_numpy(dtype=float)
            if np.all(np.isfinite(v)) and len(v) > 1 and np.any(np.sign(v[:-1]) != np.sign(v[1:])):
                idx = int(np.where(np.sign(v[:-1]) != np.sign(v[1:]))[0][0])
                jl = int(s.j.iloc[idx]); jh = int(s.j.iloc[idx + 1])
                cr.append({"evaluation": reg, "pair": "Direct_minus_Cposthoc", "metric": m,
                           "crosses_between_j": f"{jl}->{jh}",
                           "target_lo": float(s.target.iloc[idx]),
                           "target_hi": float(s.target.iloc[idx + 1]),
                           "value_lo": float(v[idx]), "value_hi": float(v[idx + 1])})
            s2 = pdf_x[(pdf_x.evaluation == reg) & (pdf_x.pair == "Surrogate_minus_Cposthoc")].sort_values("j")
            v2 = s2[m].to_numpy(dtype=float)
            if np.all(np.isfinite(v2)) and len(v2) > 1 and np.any(np.sign(v2[:-1]) != np.sign(v2[1:])):
                idx = int(np.where(np.sign(v2[:-1]) != np.sign(v2[1:]))[0][0])
                jl = int(s2.j.iloc[idx]); jh = int(s2.j.iloc[idx + 1])
                cr.append({"evaluation": reg, "pair": "Surrogate_minus_Cposthoc", "metric": m,
                           "crosses_between_j": f"{jl}->{jh}",
                           "target_lo": float(s2.target.iloc[idx]),
                           "target_hi": float(s2.target.iloc[idx + 1]),
                           "value_lo": float(v2[idx]), "value_hi": float(v2[idx + 1])})
    c.write_table(pd.DataFrame(cr) if cr else pd.DataFrame(
        [{"evaluation": None, "pair": None, "metric": None, "crosses_between_j": None,
          "target_lo": None, "target_hi": None, "value_lo": None, "value_hi": None}]),
        T / f"matched_beta_crossings{tag}.csv")
    print(f"[eval] wrote matched-beta tables{tag} ({len(df)} core rows, {len(edf)} ext rows, "
          f"{len(pdf)} pairwise rows, {len(cr)} crossings)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["freeze", "evaluate"])
    ap.add_argument("--coord", default="beta_D1", choices=["beta_D1", "beta_D3"])
    ap.add_argument("--config", default="matched_beta_frozen.json")
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    if a.mode == "freeze":
        return mode_freeze(a.coord, a.config)
    return mode_evaluate(a.config, a.tag)


if __name__ == "__main__":
    raise SystemExit(main())
