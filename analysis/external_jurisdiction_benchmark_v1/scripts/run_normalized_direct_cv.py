#!/usr/bin/env python3
"""Step 7 (Direct): normalized-rho CV path over the frozen seven folds.

raw_rho = rho_tilde / Var_training_fold(y), y = log(sale_price), ddof=0.
Canonical objective only: ratio_mode="diff", match_native_init=True (the
configuration whose real-data parity was verified in the Step 1 gate at
~1.5e-8). Uses the SELECTED baseline config for this jurisdiction (frozen in
baseline/<key>_baseline_config.json) -- no new hyperparameter search here.

No forward/2025 data is read. No penalty-path artifact from this script is
used to select a jurisdiction or a deployment rho.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from soft_constrained_models.boosting_models import LGBCovPenalty  # noqa: E402
from utils.rho_screening_v2 import BENEFIT_METRICS, PREDICTIVE_COST_METRICS  # noqa: E402
from analysis.external_jurisdiction_benchmark_v1.scripts.run_baseline_cv import (  # noqa: E402
    build_folds, enrich, v1_features,
)
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import (  # noqa: E402
    ANALYSIS, DIRECT_CANONICAL_KWARGS, LGBM_CONFIG_PATH, OUTPUT, population_variance,
    normalized_rho_tilde_grid, write_json,
)

FORWARD_LOCK_DATE = pd.Timestamp("2025-01-01")

# Metrics the Step 8-9 screen actually reads. A path point is only usable if
# every one of them is finite; see `screening_metrics_finite` below.
REQUIRED_SCREENING_METRICS = tuple(BENEFIT_METRICS) + tuple(PREDICTIVE_COST_METRICS)

# Training-support divergence bound (user-approved 2026-09-04, applied uniformly
# to every jurisdiction on the Direct path). Predicted log price must stay within
# the training label range extended by one full range width on each side. The
# rule is stated ONLY in terms of the model's own training support -- never in
# terms of R2, PRD, PRB, MKI, VEI or beta -- so it cannot select on the outcome
# of interest, and it is deliberately generous (a whole data range of headroom
# either way).
SUPPORT_SLACK_IN_RANGE_WIDTHS = 1.0


def training_support_window(y_log_train: np.ndarray) -> tuple[float, float]:
    y_min = float(np.min(y_log_train))
    y_max = float(np.max(y_log_train))
    width = y_max - y_min
    slack = SUPPORT_SLACK_IN_RANGE_WIDTHS * width
    return y_min - slack, y_max + slack


def diverged_outside_training_support(pred_log: np.ndarray, lo: float, hi: float) -> dict | None:
    """Return divergence diagnostics if any prediction leaves the support window.

    Motivation: at the top of the shared rho_tilde grid a Direct fit can pass the
    frozen gradient guard AND the prediction-finiteness guard and still be a
    numerically diverged optimisation rather than a worse model. Middlesex Direct
    fold 5 at rho_tilde=71.22 predicted log prices around 24 against a training
    range of roughly [10, 16] -- i.e. prices near $4e10 -- giving R2_price
    ~ -1.5e12 and MAE_price ~ $41bn. Those numbers are finite, so treating them
    as a legitimate "prediction deteriorated" signal moved the detected activity
    onset from 0.2669 to 2.4935 (a full decade, and a lone outlier against the
    other eight jurisdictions' 0.18-0.39), which in turn emptied the
    cross-jurisdiction Direct band. Divergence is a numerical-validity question,
    not a performance question, so it is tested on the predictions themselves.
    """
    p_min = float(np.min(pred_log))
    p_max = float(np.max(pred_log))
    if p_min >= lo and p_max <= hi:
        return None
    return {
        "pred_log_min": p_min, "pred_log_max": p_max,
        "support_lo": lo, "support_hi": hi,
    }


def screening_metrics_finite(metrics: dict) -> list[str]:
    """Names of required screening metrics that are missing or non-finite.

    A fit can "succeed" (finite gradients, finite predictions) at very large
    raw_rho and still yield unusable metrics: exp() of a large-but-finite log
    prediction stays finite in float64, so the guards on the fit and on the
    predictions both pass, yet the resulting price scale is absurd and the
    metrics computed from it blow up (observed for Middlesex Direct at
    rho_tilde=71.22: PRB~4e290, VEI~1e93, MAPE~3e290, R2_price=NaN). Left
    unflagged, such a row poisons the screen twice over -- the NaN metric fails
    the engine's whole-curve finiteness check and is silently dropped, and the
    ~1e290 values overflow when squared in select_pwl's SSE -- which is how a
    numerical artifact can masquerade as NO_STABLE_CANDIDATE_REGION. Requiring
    strict finiteness of every screening metric catches this without inventing
    any magnitude threshold.
    """
    bad = []
    for m in REQUIRED_SCREENING_METRICS:
        if m not in metrics:
            bad.append(m)
            continue
        v = metrics[m]
        if v is None or not np.isfinite(float(v)):
            bad.append(m)
    return bad


def main() -> int:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--county-key", required=True)
    parser.add_argument("--lgbm-threads", type=int, default=8)
    parser.add_argument("--extra-decades", type=float, default=0.0)
    args = parser.parse_args()
    key = args.county_key

    config_path = ANALYSIS / "baseline" / f"{key}_baseline_config.json"
    table_path = OUTPUT / "modeling_tables" / key / "history_market_core_dev.parquet"
    if not (config_path.exists() and table_path.exists()):
        print(json.dumps({"county_key": key, "path": "direct", "skipped": True, "reason": "no baseline/table"}))
        return 0
    cfg = json.loads(config_path.read_text())
    grid = yaml.safe_load(LGBM_CONFIG_PATH.read_text())["lgbm_grid"]
    params = dict(grid[cfg["selected_lgbm_config"]])
    params["n_jobs"] = args.lgbm_threads

    data = pd.read_parquet(table_path).sort_values("sale_date").reset_index(drop=True)
    data["sale_date"] = pd.to_datetime(data["sale_date"])
    assert data["sale_date"].max() < FORWARD_LOCK_DATE, f"{key}: table contains >=2025 data"

    rho_tilde_grid = normalized_rho_tilde_grid(extra_decades=args.extra_decades)
    rows = []
    for fold_idx, (dev_start, train_end, val_end) in enumerate(build_folds(data["sale_date"]), start=1):
        train_mask = data["sale_date"] < train_end
        val_mask = (data["sale_date"] >= train_end) & (data["sale_date"] < val_end)
        train = data.loc[train_mask].reset_index(drop=True)
        val = data.loc[val_mask].reset_index(drop=True)
        if len(train) < 500 or len(val) < 50:
            continue
        assert val["sale_date"].max() < FORWARD_LOCK_DATE
        assert train["sale_date"].max() <= val["sale_date"].min()

        combined = pd.concat([train, val], ignore_index=True)
        features, cats = v1_features(combined, len(train), True)
        y_log_train = np.log(train["sale_price"].astype(float))
        vy_train = population_variance(y_log_train)
        train_price = train["sale_price"].to_numpy()
        val_price = val["sale_price"].to_numpy()
        support_lo, support_hi = training_support_window(y_log_train.to_numpy())

        for rho_tilde in rho_tilde_grid:
            raw_rho = float(rho_tilde) / vy_train if vy_train > 0 else float("nan")
            model = LGBCovPenalty(
                rho=raw_rho, early_stopping_rounds=None, zero_grad_tol=1e-12,
                lgbm_params=dict(params), verbose=False, **DIRECT_CANONICAL_KWARGS,
            )
            row = {
                "county_key": key, "fold": fold_idx, "validation_year": train_end.year,
                "rho_tilde": float(rho_tilde), "raw_rho": raw_rho, "Var_training_y": vy_train,
                "n_train": len(train), "n_val": len(val), "fit_status": "OK",
            }
            try:
                # LGBCovPenalty.fit(X,y) has no categorical_feature kwarg; category dtype is auto-detected.
                # At the top of the shared rho_tilde grid, raw_rho = rho_tilde / Var_training(y) can be very
                # large for a jurisdiction/fold whose Var_training(y) is well below the CCAO reference used
                # to set the grid ceiling (confirmed for Middlesex fold 1: Vy=0.27 vs reference 0.50). This
                # can either overflow the penalty gradient directly (raises FloatingPointError inside
                # boosting_models.py) or leave the fit "successful" but with a blown-up log-price prediction
                # that overflows in exp() and then fails a downstream sklearn finiteness check as ValueError
                # -- both are the same underlying numerical-instability outcome, so both are checked for
                # explicitly rather than relying on whichever exception type happens to surface. This mirrors
                # the FIT_FAILURE outcome the frozen v3 Surrogate calibrator (first_branch_calibrate) already
                # treats as a valid, recorded result rather than a crash -- Direct just has no early-stopping
                # branch to hide behind, since the protocol requires attempting every point on the full grid.
                model.fit(features.iloc[:len(train)], y_log_train)
                pred_log = model.predict(features.iloc[len(train):])
                if not np.all(np.isfinite(pred_log)):
                    raise FloatingPointError("non-finite predicted log price")
                # Training-support divergence bound: checked BEFORE any metric is
                # computed, so a diverged optimisation can never be recorded as a
                # legitimate predictive-deterioration signal.
                divergence = diverged_outside_training_support(pred_log, support_lo, support_hi)
                if divergence is not None:
                    row["fit_status"] = "DIVERGED_OUTSIDE_TRAINING_SUPPORT"
                    row["fit_error"] = (
                        "pred_log in [{pred_log_min:.4g},{pred_log_max:.4g}] left training-support "
                        "window [{support_lo:.4g},{support_hi:.4g}]".format(**divergence)
                    )
                    row.update({k: v for k, v in divergence.items()})
                    rows.append(row)
                    continue
                pred_price = np.exp(pred_log)
                if not np.all(np.isfinite(pred_price)):
                    raise FloatingPointError("non-finite predicted price (exp overflow)")
                metrics = enrich(val_price, pred_price, train_price)
                bad_metrics = screening_metrics_finite(metrics)
                if bad_metrics:
                    raise FloatingPointError(
                        "non-finite screening metric(s): " + ",".join(bad_metrics)
                    )
                row.update(metrics)
            except (FloatingPointError, ValueError) as exc:
                row["fit_status"] = "NUMERICALLY_UNSTABLE_RHO"
                row["fit_error"] = str(exc)
            rows.append(row)
        print(f"{key} direct fold {fold_idx} ({train_end.year}) done, {len(rho_tilde_grid)} rho points", flush=True)

    df = pd.DataFrame(rows)
    ANALYSIS.joinpath("cv").mkdir(parents=True, exist_ok=True)
    OUTPUT.joinpath("cv").mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT / "cv" / f"{key}_direct_normalized_cv_path.parquet", index=False)
    df.to_csv(ANALYSIS / "cv" / f"{key}_direct_normalized_cv_path_summary.csv", index=False)
    write_json(ANALYSIS / "cv" / f"{key}_direct_cv_meta.json", {
        "county_key": key, "family": "direct", "n_folds": int(df["fold"].nunique()) if len(df) else 0,
        "n_rho_points": len(rho_tilde_grid), "selected_lgbm_config": cfg["selected_lgbm_config"],
        "rho_tilde_min": float(rho_tilde_grid.min()), "rho_tilde_max": float(rho_tilde_grid.max()),
        "no_2025_data_used": True,
    })
    print(json.dumps({"county_key": key, "path": "direct", "n_rows": len(df)}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
