"""Hand-checked canonical metric tests."""

from __future__ import annotations

import numpy as np

from utils.motivation_utils import compute_taxation_metrics, paper_mechanism_metrics


def test_price_and_log_r2_and_oos_r2():
    y_price = np.array([100.0, 200.0, 300.0, 400.0])
    y_pred = np.array([110.0, 190.0, 305.0, 380.0])
    y_train = np.array([90.0, 210.0, 250.0])
    m = compute_taxation_metrics(y_price, y_pred, scale="price", y_train=y_train)

    ss_res = np.sum((y_pred - y_price) ** 2)
    ss_tot = np.sum((y_price - y_price.mean()) ** 2)
    np.testing.assert_allclose(m["R2_price"], 1.0 - ss_res / ss_tot)

    y_log = np.log(y_price)
    p_log = np.log(y_pred)
    ss_res_log = np.sum((p_log - y_log) ** 2)
    ss_tot_log = np.sum((y_log - y_log.mean()) ** 2)
    np.testing.assert_allclose(m["R2_log"], 1.0 - ss_res_log / ss_tot_log)

    train_price_mean = float(np.mean(y_train))
    oos_den = np.sum((y_price - train_price_mean) ** 2)
    np.testing.assert_allclose(m["OOS_R2_price"], 1.0 - ss_res / oos_den)

    train_log_mean = float(np.mean(np.log(y_train)))
    oos_den_log = np.sum((y_log - train_log_mean) ** 2)
    np.testing.assert_allclose(m["OOS_R2_log"], 1.0 - ss_res_log / oos_den_log)


def test_rmse_mae_mape_mdape_proportions():
    y = np.array([100.0, 200.0, 400.0])
    p = np.array([110.0, 180.0, 500.0])
    m = compute_taxation_metrics(y, p, scale="price")
    ape = np.abs(p - y) / y
    np.testing.assert_allclose(m["RMSE_price"], float(np.sqrt(np.mean((p - y) ** 2))))
    np.testing.assert_allclose(m["MAE"], float(np.mean(np.abs(p - y))))
    np.testing.assert_allclose(m["MAPE"], float(np.mean(ape)))
    np.testing.assert_allclose(m["MdAPE"], float(np.median(ape)))
    assert m["MAPE"] < 1.0  # stored as proportion, not percent


def test_mechanism_cov_beta_corr():
    y_log = np.array([11.0, 12.0, 13.0, 14.0])
    p_log = np.array([11.2, 11.9, 13.1, 13.7])
    e = p_log - y_log
    c = y_log - y_log.mean()
    cov = float(np.mean(e * c))
    var = float(np.mean(c ** 2))
    var_e = float(np.mean((e - e.mean()) ** 2))
    mech = paper_mechanism_metrics(y_log, p_log)
    np.testing.assert_allclose(mech["Cov_log_residual_log_price"], cov)
    np.testing.assert_allclose(mech["Beta_log"], cov / var)
    np.testing.assert_allclose(mech["Corr_log_residual_log_price"], cov / np.sqrt(var_e * var))

    m = compute_taxation_metrics(y_log, p_log, scale="log")
    np.testing.assert_allclose(m["Beta_log"], mech["Beta_log"])
    assert "R2" in m and "R2_price" in m
    np.testing.assert_allclose(m["R2"], m["R2_price"])
