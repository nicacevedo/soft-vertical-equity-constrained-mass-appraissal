#!/usr/bin/env python3
"""Discriminate the mechanism behind the Cell B <-> Cell C tree-0 split difference.

Gradients/Hessians are bit-identical (p0_2_bc_rootcause), so the divergence is in tree
construction.  Test, at 1 tree, which single knob makes tree 0 identical:

  V0  frozen params as-is                       (baseline reproduction of the finding)
  V1  + deterministic=True, force_row_wise=True (histogram path pinned)
  V2  + colsample_bytree=1.0                    (feature subsampling removed)
  V3  + both

Emits tables/bc_discriminate.csv and tables/bc_discriminate_verdict.json
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


def tree_signature(model):
    m = model.dump_model()
    sig = []
    for t in m["tree_info"]:
        st = [t["tree_structure"]]; nodes = []
        while st:
            nd = st.pop()
            if "leaf_value" in nd:
                nodes.append(("L", round(float(nd["leaf_value"]), 12)))
            else:
                thr = nd.get("threshold")
                thr = thr if isinstance(thr, str) else round(float(thr), 12)
                nodes.append(("S", nd.get("split_feature"), thr))
                st.append(nd["left_child"]); st.append(nd["right_child"])
        sig.append(sorted(map(str, nodes)))
    return sig


def main() -> int:
    from soft_constrained_models.boosting_models import LGBCovPenalty

    df_tv, df_test, _a, pred_cols, cat_cols = c.load_canonical_splits()
    X = df_tv[pred_cols].copy(); Xte = df_test[pred_cols].copy()
    for k in [k for k in cat_cols if k in X.columns]:
        X[k] = X[k].astype("category"); Xte[k] = Xte[k].astype("category")
    y = np.log(df_tv[c.TARGET_COL].to_numpy()); base = float(np.mean(y)); yc = y - base
    frozen = dict(c.frozen_lgbm_config()["lgbm_params"])

    variants = {
        "V0_frozen_as_is": {},
        "V1_pin_histogram": {"deterministic": True, "force_row_wise": True, "num_threads": 1},
        "V2_no_feature_subsampling": {"colsample_bytree": 1.0},
        "V3_pin_and_no_subsampling": {"deterministic": True, "force_row_wise": True,
                                      "num_threads": 1, "colsample_bytree": 1.0},
    }
    rows = []
    for name, over in variants.items():
        p = dict(frozen); p.update({"n_estimators": 1, "num_leaves": 15, "max_depth": 4}); p.update(over)
        pb = {k: v for k, v in p.items() if k not in {"early_stopping_rounds", "early_stopping_round"}}
        b = lgb.LGBMRegressor(boost_from_average=False, **pb)
        b.fit(X, yc, init_score=np.zeros(yc.shape[0], dtype=float))
        pred_b = np.asarray(b.predict(Xte), dtype=float) + base

        cm = LGBCovPenalty(rho=0.0, ratio_mode="diff", match_native_init=True,
                           zero_grad_tol=1e-12, early_stopping_rounds=None,
                           lgbm_params=dict(p), verbose=False)
        cm.fit(X, y)
        pred_c = np.asarray(cm.predict(Xte), dtype=float)

        sb, sc = tree_signature(b.booster_), tree_signature(cm.model.booster_)
        d = np.abs(pred_b - pred_c)
        rows.append({
            "variant": name, "overrides": json.dumps(over),
            "tree0_identical": bool(sb[0] == sc[0]),
            "max_abs_delta_log": float(np.max(d)),
            "mean_abs_delta_log": float(np.mean(d)),
            "frac_exact_equal": float(np.mean(d == 0.0)),
            "colsample_bytree": p.get("colsample_bytree"),
            "deterministic": p.get("deterministic", False),
            "force_row_wise": p.get("force_row_wise", False),
        })
        print(f"[disc] {name:<28s} tree0_identical={sb[0]==sc[0]!s:<5s} "
              f"max|d|={np.max(d):.4e} exact_frac={np.mean(d==0.0):.4f}", flush=True)

    df = pd.DataFrame(rows)
    c.write_table(df, c.TABLES / "bc_discriminate.csv")

    def got(n): return df[df.variant == n].iloc[0]
    v0, v1, v2, v3 = (got(k) for k in variants)
    if v1["tree0_identical"] and not v0["tree0_identical"]:
        mech = "histogram_construction_path_unpinned (deterministic/force_row_wise)"
        fclass = "F-ENV"
    elif v2["tree0_identical"] and not v0["tree0_identical"]:
        mech = "feature_subsampling_rng_stream_differs_between_builtin_and_custom_objective"
        fclass = "F-IMP"
    elif v3["tree0_identical"]:
        mech = "combination_of_histogram_path_and_feature_subsampling"
        fclass = "F-ENV+F-IMP"
    else:
        mech = "not_isolated_by_these_knobs"
        fclass = "PENDING_CLASSIFICATION"
    verdict = {
        "gradients_and_hessians_bit_identical": True,
        "divergence_first_appears_in": "tree 0 split structure",
        "isolating_variant": mech, "candidate_failure_class": fclass,
        "results": df.to_dict(orient="records"),
        "note": ("Classification is provisional until the Track P ladder confirms it at "
                 "every capacity. A Track-P result characterises the implementation only "
                 "and may never certify a historical artifact."),
    }
    c.write_json(c.TABLES / "bc_discriminate_verdict.json", verdict)
    print(json.dumps({k: v for k, v in verdict.items() if k != "results"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
