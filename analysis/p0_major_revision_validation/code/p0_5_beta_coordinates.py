#!/usr/bin/env python3
"""Stage-3 step 1: the D1 / D2 / D3 development beta_log coordinate audit.

D1  equal-weight mean of the seven fold-specific beta_log values  -- FROZEN PRIMARY
D2  concatenated pooled OOF                                      -- duplicate-weighted,
                                                                    historical/reproducibility
                                                                    sensitivity ONLY
D3  row-balanced pooled OOF: every fold-row appearance gets w_ik = 1/m_i, so each UNIQUE
    sale carries total weight exactly one                        -- robustness sensitivity

Development data only.  No held-out or 2025 outcome is read anywhere in this module.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as c
import p0_4_centered_spread as cs

T = c.TABLES


# ------------------------------------------------------------------ weights
def appearance_weights(fold_rids: list[np.ndarray]):
    """w_ik = 1/m_i with m_i the number of validation folds containing unique row i."""
    allr = np.concatenate(fold_rids)
    uniq, cnt = np.unique(allr, return_counts=True)
    m = dict(zip(uniq.tolist(), cnt.tolist()))
    w = [np.asarray([1.0 / m[int(r)] for r in fr], dtype=float) for fr in fold_rids]
    tot = {}
    for fr, wk in zip(fold_rids, w):
        for r, x in zip(fr.tolist(), wk.tolist()):
            tot[int(r)] = tot.get(int(r), 0.0) + x
    mx = max(abs(v - 1.0) for v in tot.values())
    if mx > 1e-12:
        raise c.ProtocolViolation(f"D3 weights do not sum to one per unique row (max dev {mx:.2e})")
    return w, uniq.size, int(allr.size), mx


def beta_weighted(y, f0, w) -> float:
    """Canonical beta_log convention under weights w: c = y - weighted mean(y)."""
    W = float(np.sum(w))
    ybar = float(np.sum(w * y) / W)
    cy = y - ybar
    var = float(np.sum(w * cy ** 2) / W)
    e = f0 - y
    cov = float(np.sum(w * e * cy) / W)
    return cov / var


def beta_unweighted(y, f0) -> float:
    cy = y - float(np.mean(y))
    return float(np.mean((f0 - y) * cy) / np.mean(cy ** 2))


# ------------------------------------------------- frozen family fold preds
def family_fold_predictions():
    """(family, rho) -> list of 7 (y, f0, row_id) from the frozen CV artifacts."""
    m = pd.read_csv(c.CONFIGS / "frozen_cv_run_map.csv")
    m = m[m.model_name.isin(["LGBCovPenalty", "LGBSmoothPenalty"])]
    fam = {"LGBCovPenalty": "Direct", "LGBSmoothPenalty": "Surrogate"}
    out = {}
    for (mn, rho), grp in m.groupby(["model_name", "rho"]):
        if grp.fold_1based.nunique() != 7:
            continue
        blocks = []
        for k in range(1, 8):
            r = grp[grp.fold_1based == k].iloc[0]
            d = pd.read_parquet(c.REPO / r.pred_file).sort_values("row_id")
            blocks.append((d.y_true_log.to_numpy(float), d.y_pred_log.to_numpy(float),
                           d.row_id.to_numpy()))
        out[(fam[mn], float(rho))] = blocks
    return out


def main() -> int:
    # ---- fold row ids and D3 weights (identical across families/configs) ---
    ref_blocks = [cs.load_eval("C", f"fold_{k}_train")[0] for k in range(1, 8)]
    fold_rids = [d.row_id.to_numpy() for d in ref_blocks]
    w, n_unique, n_concat, wdev = appearance_weights(fold_rids)
    overlap = {
        "n_concatenated": n_concat, "n_unique": int(n_unique),
        "n_duplicated_appearances": int(n_concat - n_unique),
        "share_duplicated": float((n_concat - n_unique) / n_concat),
        "max_abs_unique_row_weight_minus_one": wdev,
        "m_i_distribution": {str(k): int(v) for k, v in
                             zip(*np.unique(np.concatenate(
                                 [np.asarray([1.0 / x for x in wk]) for wk in w]).astype(int),
                                 return_counts=True))},
    }

    rows = []

    # ---------------------------------------------- Direct / Surrogate paths
    fam = family_fold_predictions()
    print(f"[coord] frozen family configs with 7 folds: {len(fam)}", flush=True)
    for (family, rho), blocks in sorted(fam.items()):
        b1 = float(np.mean([beta_unweighted(y, f) for y, f, _ in blocks]))
        y = np.concatenate([b[0] for b in blocks]); f0 = np.concatenate([b[1] for b in blocks])
        b2 = beta_unweighted(y, f0)
        b3 = beta_weighted(y, f0, np.concatenate(w))
        rows.append({"family": family, "config": f"rho={rho:.12g}", "rho": rho, "b": None,
                     "beta_D1": b1, "beta_D2": b2, "beta_D3": b3,
                     "D1_minus_D3": b1 - b3, "D2_minus_D3": b2 - b3,
                     "source": "frozen CV fold predictions"})

    # -------------------------------------------------- C / A post-hoc paths
    grid = json.loads((c.CONFIGS / "b_grid_frozen.json").read_text())
    for ref, label in (("C", "C-posthoc"), ("A", "A-posthoc")):
        bl = [cs.load_eval(ref, f"fold_{k}_train") for k in range(1, 8)]
        ys = [d.y_true_log.to_numpy(float) for d, _, _ in bl]
        f0s = [d.y_pred_log.to_numpy(float) for d, _, _ in bl]
        ybars = [yb for _, yb, _ in bl]
        yc = np.concatenate(ys); f0c = np.concatenate(f0s)
        ybv = np.concatenate([np.full(len(v), yb) for v, yb in zip(ys, ybars)])
        wc = np.concatenate(w)
        for b in grid["b_values"]:
            pk = [(yb + b * (f - yb)) if b != 1.0 else f
                  for f, yb in zip(f0s, ybars)]
            b1 = float(np.mean([beta_unweighted(y, p) for y, p in zip(ys, pk)]))
            pc = f0c if b == 1.0 else (ybv + b * (f0c - ybv))
            b2 = beta_unweighted(yc, pc)
            b3 = beta_weighted(yc, pc, wc)
            rows.append({"family": label, "config": f"b={b:.12g}", "rho": None, "b": float(b),
                         "beta_D1": b1, "beta_D2": b2, "beta_D3": b3,
                         "D1_minus_D3": b1 - b3, "D2_minus_D3": b2 - b3,
                         "source": "Stage-1.5 cached fold predictions + theorem map"})
        print(f"[coord] {label} done", flush=True)

    df = pd.DataFrame(rows)
    c.write_table(df, T / "development_beta_coordinate_audit.csv")

    # ------------------------------------------------------ support + shift
    def support(coord):
        prim = ["Direct", "Surrogate", "C-posthoc"]
        lo = max(df[df.family == f][coord].min() for f in prim)
        hi = min(df[df.family == f][coord].max() for f in prim)
        return [float(lo), float(hi)]

    s1, s3 = support("beta_D1"), support("beta_D3")
    prim = df[df.family.isin(["Direct", "Surrogate", "C-posthoc"])]
    maxshift = float(prim.D1_minus_D3.abs().max())

    # order / monotonicity under each coordinate
    order_ok = True
    for f in ("Direct", "Surrogate"):
        s = df[df.family == f].sort_values("rho")
        r1 = s.beta_D1.rank().to_numpy(); r3 = s.beta_D3.rank().to_numpy()
        order_ok = order_ok and bool(np.array_equal(r1, r3))
    mono = {}
    for f in ("Direct", "Surrogate"):
        s = df[df.family == f].sort_values("rho")
        mono[f] = {"D1_monotone": bool(np.all(np.diff(s.beta_D1) > 0)),
                   "D3_monotone": bool(np.all(np.diff(s.beta_D3) > 0))}
    for f in ("C-posthoc", "A-posthoc"):
        s = df[df.family == f].sort_values("b")
        mono[f] = {"D1_monotone": bool(np.all(np.diff(s.beta_D1) > 0)),
                   "D3_monotone": bool(np.all(np.diff(s.beta_D3) > 0))}

    # would the six deterministic CORE anchors differ under D3?
    def core(coord, sup):
        L, U = sup
        D = df[(df.family == "Direct") & (df[coord] >= L - 1e-15) & (df[coord] <= U + 1e-15)]
        D = D.sort_values([coord, "rho"])
        used, picks = set(), []
        for j in range(6):
            q = L + j / 5.0 * (U - L)
            cand = D[~D.rho.isin(used)].copy()
            cand["d"] = (cand[coord] - q).abs()
            cand = cand.sort_values(["d", "rho"])
            r = cand.iloc[0]
            used.add(r.rho)
            picks.append({"j": j, "q": float(q), "rho": float(r.rho),
                          "achieved": float(r[coord]), "gap": float(abs(r[coord] - q))})
        return picks

    core1, core3 = core("beta_D1", s1), core("beta_D3", s3)
    same_cfg = [a["rho"] == b["rho"] for a, b in zip(core1, core3)]
    tau = 0.002
    within_tau = [abs(a["achieved"] - b["achieved"]) <= tau for a, b in zip(core1, core3)]
    d3_material = (not all(same_cfg)) or (not all(within_tau)) or (not order_ok)

    summary = {
        "overlap_structure": overlap,
        "D1": {"role": "FROZEN PRIMARY development coordinate",
               "definition": "equal-weight mean of seven fold-specific beta_log values",
               "interpretation": ("every fold-level prediction remains genuinely "
                                  "out-of-training-sample, and D1's mathematical definition is "
                                  "unchanged; but the validation blocks are not fully disjoint, "
                                  "so fold-level statistics are NOT independent and some "
                                  "observations contribute to more than one fold-specific value"),
               "no_iid_interpretation": True,
               "three_way_common_support": s1},
        "D2": {"role": ("duplicate-weighted historical / reproducibility sensitivity ONLY"),
               "note": ("the concatenation explicitly gives overlapping rows multiple "
                        "observation weight")},
        "D3": {"role": "row-balanced robustness sensitivity",
               "construction": "w_ik = 1/m_i so each unique sale carries total weight exactly one",
               "weights_sum_to_one_max_dev": wdev,
               "development_only": True,
               "three_way_common_support": s3},
        "max_abs_D1_minus_D3": maxshift,
        "max_abs_D2_minus_D3": float(prim.D2_minus_D3.abs().max()),
        "rank_order_preserved_D1_vs_D3": order_ok,
        "monotonicity": mono,
        "core_targets_D1": core1, "core_targets_D3": core3,
        "core_same_config_under_D3": same_cfg,
        "core_achieved_within_tau": within_tau,
        "tau": tau,
        "D3_MATERIAL": bool(d3_material),
    }
    c.write_json(T / "development_beta_coordinate_summary.json", summary)
    print(json.dumps({k: summary[k] for k in
                      ("max_abs_D1_minus_D3", "max_abs_D2_minus_D3",
                       "rank_order_preserved_D1_vs_D3", "core_same_config_under_D3",
                       "core_achieved_within_tau", "D3_MATERIAL")}, indent=2))
    print("D1 support:", s1, " D3 support:", s3)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
