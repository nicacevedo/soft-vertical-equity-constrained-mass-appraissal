"""Canonical model-grid and split-protocol tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from run_temporal_cv import _build_model_specs, _build_rho_values, _prepend_explicit_zero
from utils.motivation_utils import build_rolling_origin_protocol, split_ccao_assessment_universe


def _canonical_specs():
    lgbm_params = {
        "n_estimators": 10,
        "learning_rate": 0.1,
        "num_leaves": 8,
        "max_depth": 3,
        "random_state": 2025,
        "n_jobs": 1,
        "verbosity": -1,
    }
    rhos = _prepend_explicit_zero(_build_rho_values([0.1, 100.0], rho_count=50, rho_scale="geom"))
    return _build_model_specs(
        lgbm_params=lgbm_params,
        rho_values_smooth=rhos,
        rho_values_cov=rhos,
        keep_values=[1.0],
        ratio_modes=["diff"],
        fairness_ratio_mode="diff",
        include_cvar_models=False,
        include_logistic_proxy=False,
    ), rhos


def test_canonical_grid_families_and_rho_controls():
    specs, rhos = _canonical_specs()
    names = [s["name"] for s in specs]
    assert names.count("LinearRegression") == 1
    assert names.count("LGBMRegressor") == 1
    assert "LGBSmoothPenaltyLogisticProxy" not in names
    assert not any("CVaR" in n for n in names)
    assert not any("Group" in n for n in names)
    smooth = [s for s in specs if s["name"] == "LGBSmoothPenalty"]
    cov = [s for s in specs if s["name"] == "LGBCovPenalty"]
    assert len(smooth) == 51
    assert len(cov) == 51
    assert rhos[0] == 0.0
    assert 0.0 not in rhos[1:]
    assert len(rhos) == 51
    np.testing.assert_allclose(min(rhos[1:]), 0.1)
    np.testing.assert_allclose(max(rhos), 100.0)
    assert all(s["config"].get("ratio_mode", "diff") == "diff" for s in smooth + cov)
    assert all(s["config"].get("weighting_proxy_mode") == "identity" for s in smooth)


def test_development_test_forward_are_disjoint_on_synthetic_dates():
    dates = pd.date_range("2016-01-01", "2025-12-15", freq="7D")
    df = pd.DataFrame({"meta_sale_date": dates, "x": range(len(dates))})
    parts = split_ccao_assessment_universe(
        df,
        "meta_sale_date",
        split_prop=0.9,
        universe_start="2016-01-01",
        pre_assessment_end="2024-12-31",
        assessment_year=2025,
    )
    dev = set(map(tuple, parts["development"][["meta_sale_date"]].to_numpy()))
    test = set(map(tuple, parts["test"][["meta_sale_date"]].to_numpy()))
    fwd = set(map(tuple, parts["assessment"][["meta_sale_date"]].to_numpy()))
    assert len(dev.intersection(test)) == 0
    assert len(dev.intersection(fwd)) == 0
    assert len(test.intersection(fwd)) == 0
    assert len(parts["development"]) + len(parts["test"]) == len(parts["production"])


def test_seven_fold_protocol_on_real_development_sample():
    data_path = Path("data/CCAO/2025/training_data.parquet")
    if not data_path.is_file():
        print("SKIP training parquet not available")
        return
    dates = pd.read_parquet(data_path, columns=["meta_sale_date", "ind_pin_is_multicard", "sv_is_outlier"])
    dates = dates[(~dates["ind_pin_is_multicard"].astype("bool").fillna(True)) & (~dates["sv_is_outlier"].astype("bool").fillna(True))]
    parts = split_ccao_assessment_universe(
        dates,
        "meta_sale_date",
        split_prop=0.9,
        universe_start="2016-01-01",
        pre_assessment_end="2024-12-31",
        assessment_year=2025,
    )
    dev = parts["development"]
    folds = build_rolling_origin_protocol(
        dev,
        "meta_sale_date",
        train_mode="expanding",
        initial_train_months=15,
        val_fraction=0.10,
        step_months=15,
        min_train_rows=200,
        min_val_rows=100,
    )
    assert len(folds) == 7
    expected = [
        (46888, 5209),
        (100776, 11197),
        (151187, 16798),
        (200908, 22323),
        (252487, 28054),
        (298024, 33113),
        (310149, 34461),
    ]
    for fold, (n_tr, n_va) in zip(folds, expected):
        assert int(fold["train_size"]) == n_tr
        assert int(fold["val_size"]) == n_va
