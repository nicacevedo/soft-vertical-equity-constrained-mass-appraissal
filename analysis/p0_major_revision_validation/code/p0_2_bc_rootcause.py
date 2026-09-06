#!/usr/bin/env python3
"""Root-cause probe for the Cell B <-> Cell C discrepancy (plan Gate G1 stop branch).

Per-iteration raw-score tracing on the real development pool.  Establishes WHERE the
native-L2 and custom-objective paths first diverge and WHY, so the finding can be
classified rather than described as 'nondeterminism'.

Emits tables/bc_rootcause_trace.csv and tables/bc_rootcause_verdict.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as c


def main() -> int:
    from soft_constrained_models.boosting_models import (
        LGBCovPenalty, canonical_direct_scaled_grad_hess)

    df_tv, df_test, df_assess, pred_cols, cat_cols = c.load_canonical_splits()
    X = df_tv[pred_cols].copy()
    Xte = df_test[pred_cols].copy()
    for k in [k for k in cat_cols if k in X.columns]:
        X[k] = X[k].astype("category"); Xte[k] = Xte[k].astype("category")
    y = np.log(df_tv[c.TARGET_COL].to_numpy())
    base = float(np.mean(y)); yc = y - base
    params = dict(c.frozen_lgbm_config()["lgbm_params"])

    rows = []
    verdict = {}

    # ---- (0) analytic check: supplied derivatives at iteration 0 vs native L2 -----
    g, h, _ = canonical_direct_scaled_grad_hess(yc, np.zeros_like(yc), y_mean=float(np.mean(yc)), rho=0.0)
    native_g = np.zeros_like(yc) - yc     # score - label at score=0
    verdict["iteration0_gradient_identical_to_native_l2"] = bool(np.array_equal(g, native_g))
    verdict["iteration0_hessian_all_ones"] = bool(np.all(h == 1.0))
    verdict["iteration0_max_abs_grad_diff"] = float(np.max(np.abs(g - native_g)))

    # ---- (1) tree-count ladder: where does divergence appear? --------------------
    for n_trees in (1, 2, 5, 20, 60):
        p = dict(params); p.update({"n_estimators": n_trees, "num_leaves": 15, "max_depth": 4})

        pb = {k: v for k, v in p.items() if k not in {"early_stopping_rounds", "early_stopping_round"}}
        b = lgb.LGBMRegressor(boost_from_average=False, **pb)
        b.fit(X, yc, init_score=np.zeros(yc.shape[0], dtype=float))
        pred_b = np.asarray(b.predict(Xte), dtype=float) + base

        cm = LGBCovPenalty(rho=0.0, ratio_mode="diff", match_native_init=True,
                           zero_grad_tol=1e-12, early_stopping_rounds=None,
                           lgbm_params=dict(p), verbose=False)
        cm.fit(X, y)
        pred_c = np.asarray(cm.predict(Xte), dtype=float)

        d = np.abs(pred_b - pred_c)
        # structural comparison of the boosters
        mb = b.booster_.dump_model(); mc = cm.model.booster_.dump_model()
        def leaves(m):
            out = []
            for t in m["tree_info"]:
                st = [t["tree_structure"]]; lv = []
                while st:
                    nd = st.pop()
                    if "leaf_value" in nd:
                        lv.append(float(nd["leaf_value"]))
                    else:
                        st.append(nd["left_child"]); st.append(nd["right_child"])
                out.append(sorted(lv))
            return out
        def splits(m):
            out = []
            for t in m["tree_info"]:
                st = [t["tree_structure"]]; sp = []
                while st:
                    nd = st.pop()
                    if "leaf_value" not in nd:
                        sp.append((nd.get("split_feature"), round(float(nd.get("threshold", 0)) if not isinstance(nd.get("threshold"), str) else 0.0, 10)))
                        st.append(nd["left_child"]); st.append(nd["right_child"])
                out.append(sorted(sp, key=lambda z: (str(z[0]), str(z[1]))))
            return out
        lb, lc = leaves(mb), leaves(mc)
        sb, sc = splits(mb), splits(mc)
        first_tree_leaf_diff = float(np.max(np.abs(np.array(lb[0]) - np.array(lc[0])))) if len(lb[0]) == len(lc[0]) else float("nan")
        same_splits_tree0 = bool(sb[0] == sc[0])
        n_trees_same_splits = sum(1 for i in range(min(len(sb), len(sc))) if sb[i] == sc[i])
        rows.append({
            "n_trees": n_trees, "n_eval": int(d.size),
            "mean_abs_delta_log": float(np.mean(d)), "max_abs_delta_log": float(np.max(d)),
            "frac_exact_equal": float(np.mean(d == 0.0)),
            "tree0_same_split_structure": same_splits_tree0,
            "tree0_max_abs_leaf_value_diff": first_tree_leaf_diff,
            "n_trees_with_identical_splits": n_trees_same_splits,
            "n_trees_compared": int(min(len(sb), len(sc))),
            "cellB_n_leaves_tree0": len(lb[0]), "cellC_n_leaves_tree0": len(lc[0]),
        })
        print(f"[trace] trees={n_trees:>3d} mean|d|={np.mean(d):.4e} max|d|={np.max(d):.4e} "
              f"tree0_same_splits={same_splits_tree0} identical_split_trees={n_trees_same_splits}/{min(len(sb),len(sc))}",
              flush=True)

    df = pd.DataFrame(rows)
    c.write_table(df, c.TABLES / "bc_rootcause_trace.csv")

    first = df.iloc[0]
    verdict.update({
        "divergence_present_at_1_tree": bool(first["max_abs_delta_log"] > 0),
        "tree0_split_structure_identical": bool(first["tree0_same_split_structure"]),
        "tree0_max_abs_leaf_value_diff": float(first["tree0_max_abs_leaf_value_diff"]),
        "trace": df.to_dict(orient="records"),
    })
    c.write_json(c.TABLES / "bc_rootcause_verdict.json", verdict)
    print(json.dumps({k: v for k, v in verdict.items() if k != "trace"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
