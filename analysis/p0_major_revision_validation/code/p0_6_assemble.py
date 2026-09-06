#!/usr/bin/env python3
"""Stage-3 temporal assembly: merge the D-SNAP / D-PURGE screening shards, run D-UNSEEN
(zero fits), audit validation-block overlap under each design, difference against the
frozen primary path, and evaluate the five material-change triggers (Gate G5a).

Modes: merge | overlap | unseen | deltas | triggers | all
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as c
import p0_6_temporal_designs as td
import p0_4_centered_spread as cs

T = c.TABLES
DESIGNS = ["dsnap", "dpurge"]
BLOCKS = [f"fold_{k}" for k in range(1, 8)] + ["heldout", "forward_2025"]
FAMS = ["native", "direct", "surrogate"]
FAM_DISPLAY = {"native": "LightGBM", "direct": "Direct", "surrogate": "Surrogate"}
MET = ["R2_price", "MAE_price", "MAPE", "RMSE_log", "median_ratio", "mean_ratio",
       "weighted_mean_ratio", "COD", "COV", "PRD", "PRB", "MKI", "VEI", "beta_log",
       "Cov_log_residual_log_price", "dCor_e_y", "Delta_NL"]
ANCHORS = [0.010481131341546852, 0.1, 0.9540954763499944, 10.481131341546874, 100.0]


# ============================================================== merge
def mode_merge() -> int:
    exp = [(d, b, f) for d in DESIGNS for b in BLOCKS for f in FAMS]
    got, miss = [], []
    for d, b, f in exp:
        p = T / f"robustness_shard__{d}__{b}__{f}.csv"
        (got if p.exists() else miss).append((d, b, f, p))
    print(f"[merge] shards present {len(got)}/{len(exp)}")
    if miss:
        print("[merge] MISSING:", [f"{d}/{b}/{f}" for d, b, f, _ in miss])
        raise c.ProtocolViolation(f"{len(miss)} shards missing; merge is not complete")
    for d in DESIGNS:
        parts = [pd.read_csv(p) for dd, b, f, p in got if dd == d]
        df = pd.concat(parts, ignore_index=True)
        df["family_display"] = df.family.map(FAM_DISPLAY)
        df = df.sort_values(["family", "rho", "block"], na_position="first")
        c.write_table(df, T / f"robustness_path_{d}.csv")
        print(f"[merge] {d}: {len(df)} rows, "
              f"{df.block.nunique()} blocks, {df.family.nunique()} families, "
              f"{df[df.family=='direct'].rho.nunique()} Direct rhos")
    return 0


# ============================================================== overlap audit
def mode_overlap() -> int:
    """Validation-block overlap under the primary, D-SNAP and D-PURGE designs.

    D-PURGE preserves evaluation sets exactly, so it MUST equal the primary; that is
    asserted, not assumed. D-SNAP moves boundary-date rows to the evaluation side, so
    its overlap is measured.
    """
    df_tv, df_test, df_assess, _, _ = c.load_canonical_splits(verbose=False)
    folds, _ = c.rebuild_folds(df_tv)
    prim = {f"fold_{int(r['fold_id'])+1}": np.asarray(r["val_indices"], dtype=int)
            for r in folds}
    snap = json.loads((c.CONFIGS / "split_protocol_dsnap.json").read_text())
    purge = json.loads((c.CONFIGS / "split_protocol_dpurge.json").read_text())
    sv = {f"fold_{r['fold_1based']}": td._load_idx(r["val_idx"]) for r in snap["folds"]}
    pv = {f"fold_{r['fold_1based']}": td._load_idx(r["val_idx"]) for r in purge["folds"]}

    rows = []
    for design, vb in (("primary", prim), ("dsnap", sv), ("dpurge", pv)):
        allv = np.concatenate([vb[f"fold_{k}"] for k in range(1, 8)])
        uniq, cnt = np.unique(allv, return_counts=True)
        rows.append({"design": design, "scope": "all_seven_folds_pooled",
                     "pair": None,
                     "n_appearances": int(allv.size), "n_unique": int(uniq.size),
                     "n_duplicated_appearances": int(allv.size - uniq.size),
                     "share_duplicated": float((allv.size - uniq.size) / allv.size),
                     "max_appearances_of_any_row": int(cnt.max())})
        for a in range(1, 8):
            for b in range(a + 1, 8):
                n = int(np.intersect1d(vb[f"fold_{a}"], vb[f"fold_{b}"]).size)
                if n:
                    rows.append({"design": design, "scope": "fold_pair",
                                 "pair": f"fold_{a}&fold_{b}", "n_appearances": None,
                                 "n_unique": None, "n_duplicated_appearances": n,
                                 "share_duplicated": None,
                                 "max_appearances_of_any_row": None})
    au = pd.DataFrame(rows)
    c.write_table(au, T / "temporal_validation_overlap_audit.csv")

    # ---- assertions
    p0 = au[(au.design == "primary") & (au.scope == "all_seven_folds_pooled")].iloc[0]
    g0 = au[(au.design == "dpurge") & (au.scope == "all_seven_folds_pooled")].iloc[0]
    s0 = au[(au.design == "dsnap") & (au.scope == "all_seven_folds_pooled")].iloc[0]
    same = all(np.array_equal(np.sort(pv[f"fold_{k}"]), np.sort(prim[f"fold_{k}"]))
               for k in range(1, 8))
    if not same:
        raise c.ProtocolViolation(
            "D-PURGE evaluation sets differ from the frozen primary design; D-PURGE is "
            "defined to purge TRAINING only and must preserve evaluation sets exactly")
    summary = {
        "primary": {k: (None if pd.isna(p0[k]) else p0[k]) for k in
                    ("n_appearances", "n_unique", "n_duplicated_appearances",
                     "share_duplicated", "max_appearances_of_any_row")},
        "dsnap": {k: (None if pd.isna(s0[k]) else s0[k]) for k in
                  ("n_appearances", "n_unique", "n_duplicated_appearances",
                   "share_duplicated", "max_appearances_of_any_row")},
        "dpurge": {k: (None if pd.isna(g0[k]) else g0[k]) for k in
                   ("n_appearances", "n_unique", "n_duplicated_appearances",
                    "share_duplicated", "max_appearances_of_any_row")},
        "dpurge_evaluation_sets_identical_to_primary": True,
        "dpurge_assertion": ("verified elementwise for all seven folds -- D-PURGE purges "
                             "TRAINING rows only"),
        "overlapping_fold_pairs": {
            d: au[(au.design == d) & (au.scope == "fold_pair")].pair.tolist()
            for d in ("primary", "dsnap", "dpurge")},
        "interpretation": (
            "The seven validation blocks are not mutually disjoint under the primary "
            "design, so the CV mean is a mean over overlapping evaluation evidence. "
            "D-SNAP moves boundary-date rows to the evaluation side and therefore "
            "changes the overlap slightly; D-PURGE leaves it exactly unchanged by "
            "construction."),
    }
    c.write_json(T / "temporal_validation_overlap_summary.json", summary)
    print(json.dumps(summary, indent=2, default=str))
    return 0


# ============================================================== D-UNSEEN
def _unseen_masks():
    """Recompute the D-UNSEEN masks from the frozen rule and verify each mask_hash."""
    df_tv, df_test, df_assess, _, _ = c.load_canonical_splits(verbose=False)
    pin_dev, pin_test, pin_as, _ = td.load_pins(df_tv, df_test, df_assess)
    folds, _ = c.rebuild_folds(df_tv)
    spec = json.loads((c.CONFIGS / "unseen_subset_definition.json").read_text())
    out = {}
    for rec in folds:
        tr = np.asarray(rec["train_indices"], dtype=int)
        va = np.asarray(rec["val_indices"], dtype=int)
        m = ~np.isin(pin_dev[va], np.unique(pin_dev[tr]))
        k = f"fold_{int(rec['fold_id'])+1}"
        h = td._hash_idx(va[m])
        if h != spec["blocks"][k]["mask_hash"]:
            raise c.ProtocolViolation(f"D-UNSEEN mask hash mismatch for {k}")
        out[k] = (va, m)
    m = ~np.isin(pin_test, np.unique(pin_dev))
    if td._hash_idx(np.arange(pin_test.size)[m]) != spec["blocks"]["heldout"]["mask_hash"]:
        raise c.ProtocolViolation("D-UNSEEN mask hash mismatch for heldout")
    out["heldout"] = (np.arange(pin_test.size), m)
    prod = np.concatenate([pin_dev, pin_test])
    m = ~np.isin(pin_as, np.unique(prod))
    if td._hash_idx(np.arange(pin_as.size)[m]) != spec["blocks"]["forward_2025"]["mask_hash"]:
        raise c.ProtocolViolation("D-UNSEEN mask hash mismatch for forward_2025")
    out["forward_2025"] = (np.arange(pin_as.size), m)
    print(f"[unseen] all {len(out)} mask hashes reproduce the frozen definition")
    return out


def _train_logs():
    """y_train_log for each evaluation block's own fitting block (never an eval set)."""
    df_tv, df_test, df_assess, _, _ = c.load_canonical_splits(verbose=False)
    folds, _ = c.rebuild_folds(df_tv)
    ydev = np.log(df_tv[c.TARGET_COL].to_numpy(dtype=float))
    ytest = np.log(df_test[c.TARGET_COL].to_numpy(dtype=float))
    out = {}
    for r in folds:
        out[f"fold_{int(r['fold_id'])+1}"] = ydev[np.asarray(r["train_indices"], dtype=int)]
    out["heldout"] = ydev                              # development pool
    out["forward_2025"] = np.concatenate([ydev, ytest])  # production block
    return out


def mode_unseen() -> int:
    masks = _unseen_masks()
    trlog = _train_logs()
    grid = json.loads((c.CONFIGS / "robustness_rho_grid.json").read_text())
    # the frozen screening grid ALREADY force-includes the five display anchors, so
    # unioning ANCHORS again only risks ULP-duplicate rhos; use the grid verbatim
    rhos = sorted(float(r) for r in grid["positive_rhos"])
    if len(rhos) != len(grid["positive_rhos"]):
        raise c.ProtocolViolation("screening grid contains duplicate rhos")
    cvm = pd.read_csv(c.CONFIGS / "frozen_cv_run_map.csv")
    oos = pd.read_csv(c.CONFIGS / "frozen_config_map.csv")
    MODEL = {"Direct": "LGBCovPenalty", "Surrogate": "LGBSmoothPenalty"}
    rows = []
    for fam in ("LightGBM", "Direct", "Surrogate"):
        todo = [None] if fam == "LightGBM" else rhos
        for rho in todo:
            for blk in BLOCKS:
                if blk.startswith("fold_"):
                    k = int(blk.split("_")[1])
                    s = cvm[cvm.fold_1based == k]
                    s = (s[s.model_name == "LGBMRegressor"] if fam == "LightGBM"
                         else s[(s.model_name == MODEL[fam])
                                & np.isclose(s.rho.astype(float), rho, rtol=0, atol=1e-12)])
                else:
                    s = oos[oos.stage == blk]
                    s = (s[s.model_name == "LGBMRegressor"] if fam == "LightGBM"
                         else s[(s.model_name == MODEL[fam])
                                & np.isclose(s.rho.astype(float), rho, rtol=0, atol=1e-12)])
                if len(s) != 1:
                    continue
                d = pd.read_parquet(c.REPO / s.iloc[0].pred_file).sort_values("row_id")
                va, m = masks[blk]
                if len(d) != len(m):
                    raise c.ProtocolViolation(
                        f"{fam} rho={rho} {blk}: cached n={len(d)} vs mask n={len(m)}")
                y = d.y_true_log.to_numpy()[m]; p = d.y_pred_log.to_numpy()[m]
                mm = cs.metrics_from(y, p, trlog[blk], d.row_id.to_numpy()[m])
                rows.append({"design": "dunseen",
                             "design_type": "evaluation_subset_secondary",
                             "family": fam, "rho": rho, "block": blk,
                             "n_eval_full": int(len(m)), "n_eval_unseen": int(m.sum()),
                             "share_unseen": float(m.mean()),
                             "not_differenced_against_frozen": True,
                             **{k2: mm.get(k2) for k2 in MET}})
                print(f"[unseen] {fam:<10} rho={rho} {blk:<13} n={int(m.sum())}", flush=True)
    df = pd.DataFrame(rows)
    c.write_table(df, T / "robustness_unseen_subset.csv")
    print(f"[unseen] wrote {len(df)} rows")
    return 0




# ============================================================== deltas
def _frozen_screening():
    """The frozen 82-point path restricted to the screening rhos, CV_mean/heldout/2025."""
    v4 = pd.read_csv(c.V12 / "analysis" / "data_id=d4929d43ec19badf"
                     / "split_id=3d464d4a611b131b" / "penalty_path_analysis"
                     / "transition_regions_paper_assets_v4_delta_nl_bends" / "tables"
                     / "combined_path_table_v4_analysis_view.csv")
    grid = json.loads((c.CONFIGS / "robustness_rho_grid.json").read_text())
    rhos = [0.0] + [float(r) for r in grid["positive_rhos"]]
    FP = {"R2_price": "R2_price", "MAE_price": "MAE_price", "MAPE": "MAPE",
          "RMSE_log": "RMSE_log", "median_ratio": "median_ratio", "mean_ratio": "mean_ratio",
          "weighted_mean_ratio": "weighted_mean_ratio", "COD": "COD", "COV": "COV",
          "PRD": "PRD", "PRB": "PRB", "MKI": "MKI", "VEI": "VEI", "beta_log": "Beta_log",
          "Cov_log_residual_log_price": "Cov_log_residual_log_price",
          "Delta_NL": "Delta_NL", "dCor_e_y": "dCor_e_y"}
    out = []
    for fam in ("Direct", "Surrogate", "LightGBM"):
        s = v4[v4.family == fam]
        for _, r in s.iterrows():
            rv = float(r.rho) if pd.notna(r.rho) else 0.0
            if fam != "LightGBM" and not any(abs(rv - x) < 1e-12 for x in rhos):
                continue
            for blk, suf in [(f"fold_{k}", f"fold_{k}") for k in range(1, 8)] + \
                            [("CV_mean", "CV_mean"), ("heldout", "heldout"),
                             ("forward_2025", "forward_2025")]:
                rec = {"family": fam, "rho": rv, "block": blk}
                for m, base in FP.items():
                    col = f"{base}__{suf}"
                    rec[m] = (float(r[col]) if col in v4.columns and pd.notna(r[col])
                              else None)
                out.append(rec)
    return pd.DataFrame(out)


def _cvmean(df):
    """Equal-weight seven-fold mean, the D1 convention, computed from fold rows."""
    f = df[df.block.str.startswith("fold_")]
    g = f.groupby(["family", "rho"], dropna=False)[MET].mean().reset_index()
    g["block"] = "CV_mean"
    return g


def mode_deltas() -> int:
    froz = _frozen_screening()
    froz_cv = froz[froz.block == "CV_mean"].copy()
    rows = []
    for design in DESIGNS:
        rb = pd.read_csv(T / f"robustness_path_{design}.csv")
        rb["family"] = rb.family_display
        rb_cv = _cvmean(rb)
        for blk, R, Fz in (("CV_mean", rb_cv, froz_cv),
                           ("heldout", rb[rb.block == "heldout"], froz[froz.block == "heldout"]),
                           ("forward_2025", rb[rb.block == "forward_2025"],
                            froz[froz.block == "forward_2025"])):
            for _, r in R.iterrows():
                f = Fz[(Fz.family == r.family)
                       & np.isclose(Fz.rho.astype(float), float(r.rho), rtol=0, atol=1e-12)]
                if len(f) != 1:
                    continue
                f = f.iloc[0]
                rec = {"design": design,
                       "design_type": ("strict_date_robustness" if design == "dsnap"
                                       else "oracle_diagnostic"),
                       "family": r.family, "rho": float(r.rho), "evaluation": blk,
                       "grid": "screening"}
                for m in MET:
                    a = r.get(m); b = f.get(m)
                    rec[f"{m}__frozen"] = (None if b is None or pd.isna(b) else float(b))
                    rec[f"{m}__robust"] = (None if a is None or pd.isna(a) else float(a))
                    rec[f"{m}__delta"] = (None if (a is None or b is None
                                                   or pd.isna(a) or pd.isna(b))
                                          else float(a) - float(b))
                rows.append(rec)
    df = pd.DataFrame(rows)
    c.write_table(df, T / "robustness_vs_frozen_deltas.csv")
    print(f"[deltas] wrote {len(df)} rows")
    for design in DESIGNS:
        s = df[df.design == design]
        print(f"\n[{design}] max |delta| by headline metric (CV_mean):")
        for m in ("R2_price", "RMSE_log", "beta_log", "Delta_NL", "dCor_e_y", "COD"):
            v = s[s.evaluation == "CV_mean"][f"{m}__delta"].astype(float).abs()
            print(f"    {m:<12} max={np.nanmax(v):.6f}")
    return 0


# ============================================================== triggers (Gate G5a)
# Frozen, transparent, grid-robust criteria. Each is applied IDENTICALLY to the frozen
# path restricted to the screening rhos and to the robustness path, so the comparison is
# valid for CHANGE DETECTION even where the absolute value differs from the published
# 82-point estimator.
REBOUND_RETAIN = 0.25       # a rebound "survives" if >= 25% of the frozen rebound remains
ENDPOINT_FACTOR = 2.0       # candidate-region endpoint move that counts as material
ACTIVITY_FRAC = 0.10        # beta_log activity threshold, as a share of attained range
# G5b materiality rule. It keys off TAU_MATCH -- the beta_log matching tolerance already
# frozen in configs/matched_beta_frozen.json at the start of this stage -- rather than a new
# constant chosen after the refinement was seen. Two configurations closer than TAU_MATCH in
# beta_log were declared equivalent for matching purposes, so:
#   * an ordering "flip" is material only if the two families are separated by at least
#     TAU_MATCH on BOTH the frozen and the D-SNAP path (a reversal of a tie is not a
#     path change);
#   * a change in "all negative" status is material only if the path maximum that crosses
#     zero exceeds TAU_MATCH in absolute value (otherwise the path merely grazes zero).
TAU_MATCH = 0.002


def _path(df, fam, blk):
    s = df[(df.family == fam) & (df.block == blk)].sort_values("rho")
    return s.rho.to_numpy(dtype=float), s


def _rebound(rho, v):
    """Interior-minimum rebound: (value at max rho) - (interior minimum)."""
    ok = np.isfinite(v)
    rho, v = rho[ok], v[ok]
    if len(v) < 3:
        return None
    i = int(np.argmin(v))
    if i == len(v) - 1:
        return {"min_rho": float(rho[i]), "min_value": float(v[i]),
                "end_value": float(v[-1]), "rebound": 0.0, "interior_min": False}
    return {"min_rho": float(rho[i]), "min_value": float(v[i]),
            "end_value": float(v[-1]), "rebound": float(v[-1] - v[i]),
            "interior_min": bool(i > 0)}


def _endpoints(rho, s):
    """Screening-grid PROXY for the two published candidate-region endpoints."""
    b = s.beta_log.to_numpy(dtype=float); rl = s.RMSE_log.to_numpy(dtype=float)
    ok = np.isfinite(b) & np.isfinite(rl)
    rho, b, rl = rho[ok], b[ok], rl[ok]
    if len(rho) < 3:
        return {"activity_rho": None, "guardrail_rho": None}
    b0, r0 = b[0], rl[0]
    rng = float(np.nanmax(b) - np.nanmin(b))
    act = next((float(rho[i]) for i in range(1, len(rho))
                if abs(b[i] - b0) >= ACTIVITY_FRAC * rng), None)
    gr = next((float(rho[i]) for i in range(1, len(rho))
               if rl[i] > r0 and (act is None or rho[i] >= act)), None)
    return {"activity_rho": act, "guardrail_rho": gr}


def mode_triggers() -> int:
    froz = _frozen_screening()
    froz = pd.concat([froz, _cvmean(froz)], ignore_index=True)
    out, detail = [], {}
    for design in DESIGNS:
        rb = pd.read_csv(T / f"robustness_path_{design}.csv")
        rb["family"] = rb.family_display
        rb = pd.concat([rb, _cvmean(rb)], ignore_index=True)
        gate = (design == "dsnap")   # only D-SNAP feeds the G5a/G5b promotion gate

        # ---- T1 beta_log path ordering / monotonicity
        chg = []
        for blk in ("CV_mean", "heldout", "forward_2025"):
            rf, sf = _path(froz, "Direct", blk); rs, ss = _path(froz, "Surrogate", blk)
            rf2, sf2 = _path(rb, "Direct", blk); rs2, ss2 = _path(rb, "Surrogate", blk)
            common = sorted(set(np.round(rf, 12)) & set(np.round(rs, 12))
                            & set(np.round(rf2, 12)) & set(np.round(rs2, 12)))
            def at(s, r):
                q = s[np.isclose(s.rho.astype(float), r, rtol=0, atol=1e-12)]
                return float(q.beta_log.iloc[0]) if len(q) else np.nan
            flips = 0
            for r in common:
                a = np.sign(at(sf, r) - at(ss, r)); b2 = np.sign(at(sf2, r) - at(ss2, r))
                if np.isfinite(a) and np.isfinite(b2) and a != 0 and b2 != 0 and a != b2:
                    flips += 1
            fmono = {f: bool(np.all(np.diff(_path(froz, f, blk)[1].beta_log
                                            .to_numpy(dtype=float)) >= -1e-12))
                     for f in ("Direct", "Surrogate")}
            rmono = {f: bool(np.all(np.diff(_path(rb, f, blk)[1].beta_log
                                            .to_numpy(dtype=float)) >= -1e-12))
                     for f in ("Direct", "Surrogate")}
            sgn_f = {f: bool(np.all(_path(froz, f, blk)[1].beta_log.to_numpy(dtype=float) < 0))
                     for f in ("Direct", "Surrogate")}
            sgn_r = {f: bool(np.all(_path(rb, f, blk)[1].beta_log.to_numpy(dtype=float) < 0))
                     for f in ("Direct", "Surrogate")}
            chg.append({"evaluation": blk, "n_common_rho": len(common),
                        "ordering_flips": flips,
                        "frozen_monotone": fmono, "robust_monotone": rmono,
                        "frozen_all_negative": sgn_f, "robust_all_negative": sgn_r,
                        "sign_change": sgn_f != sgn_r})
        t1 = any(x["ordering_flips"] > 0 or x["sign_change"] for x in chg)
        out.append({"design": design, "trigger": "T1_beta_log_sign_or_ordering",
                    "fired": bool(t1), "feeds_g5_gate": gate,
                    "evidence": json.dumps(chg, default=str)})
        detail[f"{design}_T1"] = chg

        # ---- T2 Surrogate dCor rebound / T3 Surrogate Delta_NL rebound
        for tid, met in (("T2_surrogate_dcor_rebound", "dCor_e_y"),
                         ("T3_surrogate_delta_nl_rebound", "Delta_NL")):
            ev = []
            for blk in ("CV_mean", "heldout", "forward_2025"):
                rf, sf = _path(froz, "Surrogate", blk)
                rr, sr = _path(rb, "Surrogate", blk)
                a = _rebound(rf, sf[met].to_numpy(dtype=float))
                b2 = _rebound(rr, sr[met].to_numpy(dtype=float))
                lost = None
                if a and b2 and a["rebound"] > 0:
                    lost = bool(b2["rebound"] < REBOUND_RETAIN * a["rebound"])
                ev.append({"evaluation": blk, "frozen": a, "robust": b2,
                           "retained_share": (None if not (a and b2) or a["rebound"] <= 0
                                              else b2["rebound"] / a["rebound"]),
                           "rebound_lost": lost})
            fired = any(x["rebound_lost"] for x in ev if x["rebound_lost"] is not None)
            out.append({"design": design, "trigger": tid, "fired": bool(fired),
                        "feeds_g5_gate": gate, "evidence": json.dumps(ev, default=str)})
            detail[f"{design}_{tid}"] = ev

        # ---- T4 candidate-region endpoints
        ev = []
        for fam in ("Direct", "Surrogate"):
            rf, sf = _path(froz, fam, "CV_mean"); rr, sr = _path(rb, fam, "CV_mean")
            a = _endpoints(rf, sf); b2 = _endpoints(rr, sr)
            moved = {}
            for k in ("activity_rho", "guardrail_rho"):
                if a[k] and b2[k] and a[k] > 0 and b2[k] > 0:
                    f_ = max(a[k] / b2[k], b2[k] / a[k])
                    moved[k] = {"frozen": a[k], "robust": b2[k], "factor": float(f_),
                                "material": bool(f_ > ENDPOINT_FACTOR)}
                else:
                    moved[k] = {"frozen": a[k], "robust": b2[k], "factor": None,
                                "material": bool((a[k] is None) != (b2[k] is None))}
            ev.append({"family": fam, "endpoints": moved})
        fired = any(m["material"] for e in ev for m in e["endpoints"].values())
        out.append({"design": design, "trigger": "T4_candidate_region_endpoints",
                    "fired": bool(fired), "feeds_g5_gate": gate,
                    "evidence": json.dumps(ev, default=str),
                    "caveat": ("screening-grid PROXY endpoints, applied identically to "
                               "both paths; NOT a reproduction of the published 82-point "
                               "smoothed changepoint estimator")})
        detail[f"{design}_T4"] = ev

        # ---- T5 accuracy at moderate rho
        ev = []
        for fam in ("Direct", "Surrogate"):
            for blk in ("CV_mean", "heldout", "forward_2025"):
                for src, dd in (("frozen", froz), ("robust", rb)):
                    r_, s_ = _path(dd, fam, blk)
                    o = s_[np.isclose(s_.rho.astype(float), 0.0, rtol=0, atol=1e-12)]
                    m1 = s_[np.isclose(s_.rho.astype(float), 0.9540954763499944,
                                       rtol=0, atol=1e-9)]
                    if not len(o) or not len(m1):
                        continue
                    ev.append({"family": fam, "evaluation": blk, "source": src,
                               "dR2_at_rho_0.954": float(m1.R2_price.iloc[0]
                                                          - o.R2_price.iloc[0]),
                               "dMAE_at_rho_0.954": float(m1.MAE_price.iloc[0]
                                                           - o.MAE_price.iloc[0])})
        fired = False
        for fam in ("Direct", "Surrogate"):
            for blk in ("CV_mean", "heldout", "forward_2025"):
                f_ = [x for x in ev if x["family"] == fam and x["evaluation"] == blk
                      and x["source"] == "frozen"]
                r_ = [x for x in ev if x["family"] == fam and x["evaluation"] == blk
                      and x["source"] == "robust"]
                if f_ and r_ and f_[0]["dR2_at_rho_0.954"] > 0 >= r_[0]["dR2_at_rho_0.954"]:
                    fired = True
        out.append({"design": design, "trigger": "T5_no_accuracy_cost_at_moderate_rho",
                    "fired": bool(fired), "feeds_g5_gate": gate,
                    "evidence": json.dumps(ev, default=str)})
        detail[f"{design}_T5"] = ev

    tg = pd.DataFrame(out)
    c.write_table(tg, T / "temporal_material_change_triggers.csv")
    c.write_json(T / "temporal_material_change_triggers_detail.json",
                 {"criteria": {"REBOUND_RETAIN": REBOUND_RETAIN,
                               "ENDPOINT_FACTOR": ENDPOINT_FACTOR,
                               "ACTIVITY_FRAC": ACTIVITY_FRAC},
                  "detail": detail})
    gate_fired = tg[(tg.design == "dsnap") & tg.fired]
    g5a = {
        "gate": "G5a",
        "screening_grid": "27 positive screening rhos + rho=0 (every 4th of the frozen 82 "
                          "plus the five display anchors and four candidate endpoints)",
        "dsnap_triggers_fired": gate_fired.trigger.tolist(),
        "n_dsnap_triggers_fired": int(len(gate_fired)),
        "dpurge_triggers_fired": tg[(tg.design == "dpurge") & tg.fired].trigger.tolist(),
        "dpurge_note": ("D-PURGE is an ORACLE diagnostic and does NOT feed the D-SNAP "
                        "promotion gate; its triggers are reported for information only"),
        "g5a_outcome": ("NO_TRIGGER -- the primary design stands and D-SNAP is reported as "
                        "strict-date robustness" if not len(gate_fired) else
                        "ALERT -- targeted local refinement required before any promotion"),
        "hard_stop": ("Full 82-point D-SNAP regeneration is NOT launched in this run under "
                      "any outcome; a confirmed promotion requires separate authorization."),
    }
    c.write_json(T / "gate_g5a_outcome.json", g5a)
    print(tg[["design", "trigger", "fired", "feeds_g5_gate"]].to_string(index=False))
    print()
    print(json.dumps(g5a, indent=2))
    return 0



# ============================================================== Gate G5b
def _refined_dsnap():
    """Screening D-SNAP path + the G5a refinement shards, deduplicated on (block, family, rho)."""
    rb = pd.read_csv(T / "robustness_path_dsnap.csv")
    rb["family"] = rb.family_display
    parts = [rb]
    for p in sorted(T.glob("dsnap_refine_shard__*.csv")):
        d = pd.read_csv(p)
        d["family_display"] = d.family.map(FAM_DISPLAY)
        d["family"] = d.family_display
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    n0 = len(out)
    out = out.drop_duplicates(["block", "family", "rho"], keep="first")

    # The gate may never run on partial refinement evidence: every rho of every region
    # must be present for every one of the nine blocks, on each family the frozen
    # refinement config declares. Missing fits would silently shrink the tested support.
    g = json.loads((c.CONFIGS / "dsnap_refinement_grid.json").read_text())
    scr = {round(float(x), 12)
           for x in json.loads((c.CONFIGS / "robustness_rho_grid.json").read_text())
           ["positive_rhos"]}
    missing = []
    for reg in g["regions"]:
        for fam in reg["families"]:
            disp = FAM_DISPLAY[fam]
            for rho in (round(float(x), 12) for x in reg["refinement_rhos"]):
                if rho in scr:
                    raise c.ProtocolViolation(
                        f"refinement rho {rho} is already a screening rho; the refinement "
                        "grid must contain only original-grid points the screen skipped")
                for blk in BLOCKS:
                    if not len(out[(out.family == disp) & (out.block == blk)
                                   & np.isclose(out.rho.astype(float), rho,
                                                rtol=0, atol=1e-12)]):
                        missing.append(f"{reg['id']}/{blk}/{fam}/rho={rho}")
    if missing:
        raise c.ProtocolViolation(
            f"{len(missing)} refinement fits missing, e.g. {missing[:5]}; "
            "Gate G5b cannot be evaluated on partial evidence")
    return out.sort_values(["family", "rho", "block"]), n0 - len(out)


def mode_g5b() -> int:
    g5a = json.loads((T / "gate_g5a_outcome.json").read_text())
    if g5a["n_dsnap_triggers_fired"] == 0:
        c.write_json(T / "gate_g5b_outcome.json", {
            "gate": "G5b", "status": "NOT_APPLICABLE",
            "reason": "no D-SNAP screening trigger fired at G5a",
            "full_dsnap_regeneration_launched_in_this_run": False})
        print("[g5b] no G5a alert; nothing to confirm")
        return 0

    tau_frozen = json.loads((c.CONFIGS / "matched_beta_frozen.json").read_text())["tau"]
    if abs(tau_frozen - TAU_MATCH) > 0:
        raise c.ProtocolViolation(
            f"TAU_MATCH {TAU_MATCH} does not equal the frozen matching tolerance {tau_frozen}")
    refined, ndup = _refined_dsnap()
    froz = _frozen_screening()
    # The frozen table carries all 82 points, so both sides are restricted to the rho
    # values the refined D-SNAP path actually contains. Region A refined BOTH families;
    # region B refined the Surrogate only (it exists to test the Surrogate held-out sign
    # alert). A single Direct-derived rho set would therefore silently discard every
    # region-B fit, so the ordering test uses the rhos common to both families and the
    # sign test uses each family's own refined support.
    rho_by_fam = {fam: sorted(set(np.round(
        refined[refined.family == fam].rho.astype(float), 12)))
        for fam in ("Direct", "Surrogate")}
    keep = sorted(set(rho_by_fam["Direct"]) & set(rho_by_fam["Surrogate"]))
    fullf = pd.read_csv(c.V12 / "analysis" / "data_id=d4929d43ec19badf"
                        / "split_id=3d464d4a611b131b" / "penalty_path_analysis"
                        / "transition_regions_paper_assets_v4_delta_nl_bends" / "tables"
                        / "combined_path_table_v4_analysis_view.csv")

    def froz_beta(fam, blk, rho):
        s = fullf[(fullf.family == fam)
                  & np.isclose(fullf.rho.astype(float).fillna(0.0), rho, rtol=0, atol=1e-12)]
        if not len(s):
            return np.nan
        col = "Beta_log__CV_mean" if blk == "CV_mean" else f"Beta_log__{blk}"
        v = s.iloc[0].get(col)
        return float(v) if pd.notna(v) else np.nan

    ref_cv = _cvmean(refined)
    ref_all = pd.concat([refined, ref_cv], ignore_index=True)

    rows, ev = [], []
    for blk in ("CV_mean", "heldout", "forward_2025"):
        flips_all, flips_material = 0, 0
        n = 0
        for rho in keep:
            fd, fs = froz_beta("Direct", blk, rho), froz_beta("Surrogate", blk, rho)
            rd = ref_all[(ref_all.family == "Direct") & (ref_all.block == blk)
                         & np.isclose(ref_all.rho.astype(float), rho, rtol=0, atol=1e-12)]
            rs = ref_all[(ref_all.family == "Surrogate") & (ref_all.block == blk)
                         & np.isclose(ref_all.rho.astype(float), rho, rtol=0, atol=1e-12)]
            if not len(rd) or not len(rs) or not np.isfinite(fd) or not np.isfinite(fs):
                continue
            gf = fd - fs
            gr = float(rd.iloc[0].beta_log) - float(rs.iloc[0].beta_log)
            n += 1
            if np.sign(gf) != np.sign(gr) and gf != 0 and gr != 0:
                flips_all += 1
                # MATERIAL only if the two families are actually separated at this rho
                if min(abs(gf), abs(gr)) >= TAU_MATCH:
                    flips_material += 1
                    rows.append({"evaluation": blk, "rho": rho, "gap_frozen": gf,
                                 "gap_dsnap": gr, "material": True})
        # sign status of each family's path
        sf, sr = {}, {}
        for fam in ("Direct", "Surrogate"):
            fam_rhos = rho_by_fam[fam]
            fv = np.array([froz_beta(fam, blk, r) for r in fam_rhos], dtype=float)
            rv = np.array([float(ref_all[(ref_all.family == fam) & (ref_all.block == blk)
                                         & np.isclose(ref_all.rho.astype(float), r,
                                                      rtol=0, atol=1e-12)].beta_log.iloc[0])
                           if len(ref_all[(ref_all.family == fam) & (ref_all.block == blk)
                                          & np.isclose(ref_all.rho.astype(float), r,
                                                       rtol=0, atol=1e-12)]) else np.nan
                           for r in fam_rhos], dtype=float)
            m = np.isfinite(fv) & np.isfinite(rv)
            sf[fam] = {"all_negative": bool(np.all(fv[m] < 0)), "max": float(np.nanmax(fv[m])),
                       "n_rho": int(m.sum())}
            sr[fam] = {"all_negative": bool(np.all(rv[m] < 0)), "max": float(np.nanmax(rv[m])),
                       "n_rho": int(m.sum())}
        sign_changed = any(sf[f]["all_negative"] != sr[f]["all_negative"]
                           for f in ("Direct", "Surrogate"))
        sign_material = any(
            sf[f]["all_negative"] != sr[f]["all_negative"]
            and max(abs(sf[f]["max"]), abs(sr[f]["max"])) >= TAU_MATCH
            for f in ("Direct", "Surrogate"))
        ev.append({"evaluation": blk, "n_rho_compared": n,
                   "ordering_flips_any": flips_all,
                   "ordering_flips_material": flips_material,
                   "tau_match": TAU_MATCH,
                   "frozen_sign": sf, "dsnap_sign": sr,
                   "sign_status_changed": sign_changed,
                   "sign_change_material": sign_material})

    survives = any(x["ordering_flips_material"] > 0 or x["sign_change_material"] for x in ev)
    ref_tab = pd.DataFrame(rows) if rows else pd.DataFrame(
        [{"evaluation": None, "rho": None, "gap_frozen": None, "gap_dsnap": None,
          "material": False}])
    c.write_table(ref_tab, T / "dsnap_refinement.csv")

    out = {
        "gate": "G5b",
        "trigger_re_evaluated": "T1_beta_log_sign_or_ordering",
        "refinement_rhos_added": {fam: int(len(v) - 28) for fam, v in rho_by_fam.items()},
        "duplicate_rows_dropped_on_merge": int(ndup),
        "n_rho_after_refinement": {fam: int(len(v)) for fam, v in rho_by_fam.items()},
        "n_rho_ordering_test_both_families": int(len(keep)),
        "materiality_rule": {
            "TAU_MATCH": TAU_MATCH,
            "source": ("the beta_log matching tolerance frozen in "
                       "configs/matched_beta_frozen.json BEFORE this stage read any outcome; "
                       "no new constant was introduced at decision time"),
            "ordering": ("material only if min(|gap_frozen|, |gap_dsnap|) >= TAU_MATCH, i.e. "
                         "the families are genuinely separated on BOTH paths and the ordering "
                         "genuinely reversed"),
            "sign": ("material only if the path maximum crossing zero exceeds TAU_MATCH in "
                     "absolute value")},
        "evidence": ev,
        "status": ("CONFIRMED_MATERIAL_CHANGE" if survives else "NOT_CONFIRMED"),
        "promotion": ("D-SNAP promoted; the temporal branch is REQUIRES_FULL_DSNAP_REGEN"
                      if survives else
                      "no promotion -- the screening alert was a near-tie / near-zero artifact "
                      "that did not survive local refinement; the primary design stands"),
        "full_dsnap_regeneration_launched_in_this_run": False,
        "hard_stop": ("Per the stage authorization, the full 82-point D-SNAP regeneration is "
                      "NOT launched in this run even under a confirmed promotion; it requires "
                      "separate authorization."),
    }
    c.write_json(T / "gate_g5b_outcome.json", out)
    print(json.dumps(out, indent=2, default=str))
    return 0

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True,
                    choices=["merge", "overlap", "unseen", "deltas", "triggers",
                             "g5b", "all"])
    a = ap.parse_args()
    if a.mode == "merge":
        return mode_merge()
    if a.mode == "overlap":
        return mode_overlap()
    if a.mode == "unseen":
        return mode_unseen()
    if a.mode == "deltas":
        return mode_deltas()
    if a.mode == "triggers":
        return mode_triggers()
    if a.mode == "g5b":
        return mode_g5b()
    if a.mode == "all":
        for f in (mode_merge, mode_overlap, mode_unseen, mode_deltas, mode_triggers):
            f()
        return 0
    raise SystemExit(f"unknown mode {a.mode}")


if __name__ == "__main__":
    raise SystemExit(main())
