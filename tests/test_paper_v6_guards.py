"""Focused paper-v6 pre-selection guards: init, freeze, grid, CV gate, metrics."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

from analyze_penalty_paths import (
    SECTION2_EXPORT_COLUMNS,
    _load_cv_runs,
    _plot_section2_baseline_bins,
    _sale_price_bin_profile,
    _section2_row,
    resolve_experiment_ids,
)
from canonical_experiment import (
    BaselineFreezeError,
    build_cv_completion,
    require_complete_cv,
    write_frozen_baseline,
    write_json,
)
from run_temporal_cv import _build_model_specs, _build_rho_values, _prepend_explicit_zero
from soft_constrained_models.boosting_models import (
    LGBCovPenalty,
    LGBSmoothPenalty,
    canonical_direct_scaled_grad_hess,
    canonical_surrogate_scaled_grad_hess,
)
from utils.motivation_utils import (
    compute_taxation_metrics,
    distance_correlation_e_y,
    mki,
    paper_mechanism_metrics,
)


@contextmanager
def _raises(exc_type, match=None):
    try:
        yield
    except exc_type as err:
        if match and match not in str(err):
            raise AssertionError(f"expected {match!r} in {err}") from err
        return
    raise AssertionError(f"expected {exc_type.__name__}")


def _tiny_lgbm_params():
    return {
        "n_estimators": 25,
        "learning_rate": 0.2,
        "num_leaves": 8,
        "max_depth": 3,
        "min_child_samples": 5,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "reg_lambda": 0.0,
        "reg_alpha": 0.0,
        "random_state": 7,
        "n_jobs": 1,
        "verbosity": -1,
    }


def test_centered_init_algebra_matches_original_coordinates():
    y = np.array([11.0, 12.0, 13.0, 14.0])
    base = float(np.mean(y))
    y_c = y - base
    pred_c = np.array([0.1, -0.2, 0.0, 0.3])
    np.testing.assert_allclose(pred_c - y_c, (pred_c + base) - y)
    g, h, extra = canonical_direct_scaled_grad_hess(y_c, pred_c, y_mean=float(np.mean(y_c)), rho=2.0)
    g2, h2, extra2 = canonical_direct_scaled_grad_hess(y, pred_c + base, y_mean=base, rho=2.0)
    np.testing.assert_allclose(g, g2)
    np.testing.assert_allclose(h, h2)
    np.testing.assert_allclose(extra["C"], extra2["C"])


def test_rho0_direct_and_surrogate_derivatives_match_residual():
    rng = np.random.default_rng(0)
    y = rng.normal(12.0, 0.4, size=20)
    pred = y + rng.normal(0.0, 0.1, size=20)
    y_mean = float(np.mean(y))
    e = pred - y
    gd, hd, _ = canonical_direct_scaled_grad_hess(y, pred, y_mean=y_mean, rho=0.0)
    gs, hs, _ = canonical_surrogate_scaled_grad_hess(y, pred, y_mean=y_mean, rho=0.0)
    np.testing.assert_allclose(gd, e)
    np.testing.assert_allclose(gs, e)
    np.testing.assert_allclose(hd, np.ones_like(e))
    np.testing.assert_allclose(hs, np.ones_like(e))
    np.testing.assert_allclose(gd, gs)
    np.testing.assert_allclose(hd, hs)


def test_native_custom_rho0_parity_after_mean_init():
    import lightgbm as lgb
    from sklearn.metrics import r2_score

    rng = np.random.default_rng(21)
    n = 120
    X = rng.normal(size=(n, 4))
    y = 12.0 + 0.4 * X[:, 0] - 0.2 * X[:, 1] + rng.normal(0.0, 0.15, size=n)
    params = _tiny_lgbm_params()
    native = lgb.LGBMRegressor(**params)
    native.fit(X, y)
    p_native = np.asarray(native.predict(X), dtype=float)

    direct = LGBCovPenalty(
        rho=0.0,
        ratio_mode="diff",
        match_native_init=True,
        early_stopping_rounds=None,
        lgbm_params=dict(params),
        verbose=False,
    )
    surrogate = LGBSmoothPenalty(
        rho=0.0,
        ratio_mode="diff",
        weighting_proxy_mode="identity",
        match_native_init=True,
        early_stopping_rounds=None,
        lgbm_params=dict(params),
        verbose=False,
    )
    direct.fit(X, y)
    surrogate.fit(X, y)
    p_direct = np.asarray(direct.predict(X), dtype=float)
    p_surr = np.asarray(surrogate.predict(X), dtype=float)

    np.testing.assert_allclose(p_direct, p_surr, rtol=1e-8, atol=1e-8)
    mean_abs = float(np.mean(np.abs(p_native - p_direct)))
    max_abs = float(np.max(np.abs(p_native - p_direct)))
    r2_n = r2_score(np.exp(y), np.exp(p_native))
    r2_d = r2_score(np.exp(y), np.exp(p_direct))
    rmse_n = float(np.sqrt(np.mean((p_native - y) ** 2)))
    rmse_d = float(np.sqrt(np.mean((p_direct - y) ** 2)))
    beta_n = paper_mechanism_metrics(y, p_native)["Beta_log"]
    beta_d = paper_mechanism_metrics(y, p_direct)["Beta_log"]
    assert mean_abs < 5e-3, {"mean_abs": mean_abs, "max_abs": max_abs, "delta_r2": r2_n - r2_d}
    assert abs(r2_n - r2_d) < 1e-3
    assert abs(rmse_n - rmse_d) < 1e-3
    assert abs(beta_n - beta_d) < 1e-3
    assert np.isclose(direct.base_score_, float(np.mean(y)))
    assert surrogate.early_stopping_rounds is None
    assert direct.early_stopping_rounds is None


def test_fallback_does_not_write_frozen_baseline():
    with TemporaryDirectory() as td:
        target = Path(td) / "frozen_baseline.json"
        with _raises(BaselineFreezeError):
            write_frozen_baseline(target, {"best_lgbm_params": {}}, fallback_used=True)
        assert not target.exists()
        write_frozen_baseline(target, {"best_lgbm_params": {"n_estimators": 10}}, fallback_used=False)
        assert target.is_file()


def test_canonical_grid_composition_and_early_stopping():
    rhos = _prepend_explicit_zero(_build_rho_values([0.1, 100.0], rho_count=50, rho_scale="geom"))
    specs = _build_model_specs(
        lgbm_params=_tiny_lgbm_params(),
        rho_values_smooth=rhos,
        rho_values_cov=rhos,
        keep_values=[1.0],
        ratio_modes=["diff"],
        fairness_ratio_mode="diff",
        include_cvar_models=False,
        include_logistic_proxy=False,
    )
    names = [s["name"] for s in specs]
    assert names.count("LinearRegression") == 1
    assert names.count("LGBMRegressor") == 1
    assert names.count("LGBCovPenalty") == 51
    assert names.count("LGBSmoothPenalty") == 51
    assert not any("CVaR" in n or "Logistic" in n or "Group" in n for n in names)
    cov0 = next(s for s in specs if s["name"] == "LGBCovPenalty" and float(s["config"]["rho"]) == 0.0)
    sm0 = next(s for s in specs if s["name"] == "LGBSmoothPenalty" and float(s["config"]["rho"]) == 0.0)
    cov_m = cov0["factory"]()
    sm_m = sm0["factory"]()
    assert cov_m.early_stopping_rounds is None
    assert sm_m.early_stopping_rounds is None
    assert cov_m.match_native_init is True
    assert sm_m.match_native_init is True
    assert sm0["config"]["weighting_proxy_mode"] == "identity"


def test_cv_completion_gate_and_invalid_propagation():
    with TemporaryDirectory() as td:
        result_root = Path(td) / "exp"
        run_records = pd.DataFrame(
            {"config_id": ["cfgA", "cfgA", "cfgB", "cfgB"], "fold_id": [1, 2, 1, 2]}
        )
        complete = build_cv_completion(
            data_id="d1",
            split_id="s1",
            expected_config_ids=["cfgA", "cfgB"],
            expected_fold_ids=[1, 2],
            run_records=run_records,
            failed_records=pd.DataFrame(),
            invalid_config_ids=["cfgB"],
            frozen_baseline_sha="abc",
            model_grid_sha="def",
        )
        assert complete["status"] == "complete"
        write_json(result_root / "cv_completion.json", complete)
        loaded = require_complete_cv(
            str(result_root),
            data_id="d1",
            split_id="s1",
            frozen_baseline_sha="abc",
            model_grid_sha="def",
        )
        assert loaded["invalid_config_ids"] == ["cfgB"]
        incomplete = build_cv_completion(
            data_id="d1",
            split_id="s1",
            expected_config_ids=["cfgA", "cfgB"],
            expected_fold_ids=[1, 2],
            run_records=pd.DataFrame({"config_id": ["cfgA"], "fold_id": [1]}),
            failed_records=pd.DataFrame(),
            invalid_config_ids=[],
            frozen_baseline_sha="abc",
            model_grid_sha="def",
        )
        assert incomplete["status"] == "incomplete"
        write_json(result_root / "cv_completion.json", incomplete)
        with _raises(RuntimeError, match="incomplete"):
            require_complete_cv(
                str(result_root),
                data_id="d1",
                split_id="s1",
                frozen_baseline_sha="abc",
                model_grid_sha="def",
            )


def test_analysis_does_not_mix_experiment_ids():
    with TemporaryDirectory() as td:
        root = Path(td) / "result"
        for data_id, split_id, val in (("aaa", "111", 1.0), ("bbb", "222", 9.0)):
            d = root / "runs" / f"data_id={data_id}" / f"split_id={split_id}" / "fold_id=1"
            d.mkdir(parents=True)
            pd.DataFrame(
                {
                    "model_name": ["LGBCovPenalty"],
                    "rho": [0.0],
                    "R2_price": [val],
                }
            ).to_parquet(d / "metrics.parquet", index=False)
        with _raises(SystemExit):
            resolve_experiment_ids(root, None, None)
        did, sid = resolve_experiment_ids(root, "aaa", "111")
        assert (did, sid) == ("aaa", "111")
        cv = _load_cv_runs(root, "aaa", "111")
        assert len(cv) == 1
        assert float(cv["R2_price"].iloc[0]) == 1.0


def test_dcor_fast_matches_naive_small_vector():
    import dcor

    rng = np.random.default_rng(3)
    y = rng.normal(size=40)
    e = 0.3 * y + rng.normal(scale=0.2, size=40)
    fast = float(dcor.distance_correlation(e, y, method="auto"))
    naive = float(dcor.distance_correlation(e, y, method="naive"))
    wrapped = distance_correlation_e_y(y, y + e)
    np.testing.assert_allclose(fast, naive, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(wrapped, naive, rtol=1e-10, atol=1e-10)


def _mki_paper(phat, p):
    phat = np.asarray(phat, dtype=float)
    p = np.asarray(p, dtype=float)
    order = np.argsort(p, kind="mergesort")
    phat_s, p_s = phat[order], p[order]
    n = len(p_s)
    q = (np.arange(1, n + 1) - 0.5) / n
    C = 2.0 / (n * phat_s.mean()) * np.sum(phat_s * q) - 1.0
    G = 2.0 / (n * p_s.mean()) * np.sum(p_s * q) - 1.0
    return C / G


def test_mki_matches_paper_formula_and_handles_ties():
    rng = np.random.default_rng(0)
    p = rng.uniform(50, 400, size=20)
    phat = p * rng.uniform(0.7, 1.3, size=20)
    np.testing.assert_allclose(mki(phat, p, na_rm=True), _mki_paper(phat, p), rtol=1e-12, atol=1e-12)
    p2 = np.array([1.0, 1.0, 2.0, 2.0, 3.0])
    phat2 = np.array([1.1, 0.8, 2.2, 1.7, 3.3])
    np.testing.assert_allclose(mki(phat2, p2, na_rm=True), _mki_paper(phat2, p2), rtol=1e-12, atol=1e-12)


def test_mki_adversarial_ties_use_documented_secondary_key():
    """Equal sale prices in an order that conflicts with AV-descending tie-breaks.

    Production MKI sorts sale price ascending, then assessed/predicted descending
    (Quintos). The paper formula is silent on ties; after applying that secondary
    key it matches production. Naive stable argsort on price alone need not.
    Production numerics are unchanged.
    """
    p = np.array([100.0, 100.0, 100.0, 200.0, 300.0])
    phat = np.array([80.0, 150.0, 120.0, 210.0, 290.0])
    prod = mki(phat, p, na_rm=True)
    order = np.lexsort((-phat, p))
    paper_ordered = _mki_paper(phat[order], p[order])
    np.testing.assert_allclose(prod, paper_ordered, rtol=1e-12, atol=1e-12)
    naive = _mki_paper(phat, p)
    assert not np.isclose(prod, naive)


def test_vei_profile_defaults_match_production_grouping():
    import inspect
    from utils.motivation_utils import vei, vei_percentile_group_profile

    sig = inspect.signature(vei_percentile_group_profile)
    assert sig.parameters["n_bootstrap"].default == 1000
    assert sig.parameters["ci"].default == 0.90
    assert sig.parameters["rng_seed"].default == 2025
    rng = np.random.default_rng(2025)
    sale = rng.uniform(80.0, 400.0, size=600)
    assessed = sale * rng.uniform(0.8, 1.2, size=600)
    point = vei(assessed, sale, na_rm=True)
    profile = vei_percentile_group_profile(assessed, sale)
    assert not profile.empty
    assert int(profile["n_groups"].iloc[0]) == 10
    first = float(profile.loc[profile["group"] == 1, "median_ratio"].iloc[0])
    last = float(profile.loc[profile["group"] == 10, "median_ratio"].iloc[0])
    med = float(profile["overall_median_ratio"].iloc[0])
    reconstructed = 100.0 * (last - first) / med
    np.testing.assert_allclose(point, reconstructed, rtol=1e-10, atol=1e-10)


def test_native_lgbm_factory_disables_early_stopping():
    rhos = _prepend_explicit_zero(_build_rho_values([0.1, 100.0], rho_count=2, rho_scale="geom"))
    specs = _build_model_specs(
        lgbm_params=_tiny_lgbm_params(),
        rho_values_smooth=rhos,
        rho_values_cov=rhos,
        keep_values=[1.0],
        ratio_modes=["diff"],
        fairness_ratio_mode="diff",
        include_cvar_models=False,
        include_logistic_proxy=False,
    )
    native = next(s for s in specs if s["name"] == "LGBMRegressor")["factory"]()
    params = native.get_params()
    for key in ("early_stopping_rounds", "early_stopping_round"):
        if key in params:
            assert params[key] in (None, 0, False)


def test_section2_table_schema_and_units():
    y = np.log(np.array([100.0, 200.0, 400.0, 800.0]))
    pred = y + np.array([0.05, -0.02, 0.01, -0.04])
    metrics = compute_taxation_metrics(y, pred, scale="log")
    row = _section2_row(metrics, evaluation="Held-out evaluation", model="LGBMRegressor")
    assert set(SECTION2_EXPORT_COLUMNS) <= set(row)
    assert abs(row["MAPE_price_pct"] - 100.0 * metrics["MAPE"]) < 1e-12
    assert abs(row["COD_pct"] - metrics["COD"]) < 1e-12
    assert np.isfinite(row["dCor_e_y"])


def test_section2_binning_figure_smoke():
    rng = np.random.default_rng(4)
    rows = []
    for evaluation in ("Held-out evaluation", "2025 forward evaluation"):
        for model in ("LinearRegression", "LGBMRegressor"):
            sale = np.exp(rng.normal(12.2, 0.4, size=90))
            ratio = 1.05 - 0.04 * (np.log(sale) - np.log(sale).mean())
            rows.append(
                pd.DataFrame(
                    {
                        "evaluation": evaluation,
                        "model": model,
                        "sale_price": sale,
                        "valuation_to_sale_ratio": ratio,
                        "beta_log": -0.04,
                    }
                )
            )
    profile = _sale_price_bin_profile(pd.concat(rows, ignore_index=True), n_bins=30)
    assert set(profile["bin"].unique()) == set(range(1, 31))
    with TemporaryDirectory() as td:
        out = Path(td) / "section2.png"
        _plot_section2_baseline_bins(profile, out)
        assert out.is_file() and out.stat().st_size > 0
