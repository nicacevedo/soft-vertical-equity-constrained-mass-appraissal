#!/usr/bin/env python3
"""EXPERIMENTAL / TOY / NON-CANONICAL V2: quadratic vs Huber vs absolute.

Fixed-level only. Dense low-to-moderate lambda path. Training signed-slope
matching. No CV, no LI, no capped-quadratic, no paper/canonical edits.

Writes only to output/toy_surrogate_ablation_v2/. Does not overwrite V1.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import toy_surrogate_ablation as v1
from canonical_experiment import git_state, lgbm_params_hash, package_versions, read_json, write_json
from soft_constrained_models.toy_surrogate_models import (
    EXPERIMENT_LABEL,
    ToyLGBSurrogate,
    covariance_C,
    fairness_penalty_gradient_unit_rho_from_ec,
    huber_delta_from_q,
    huber_linear_share,
)
from utils.motivation_utils import compute_taxation_metrics, paper_mechanism_metrics

DEFAULT_OUTPUT = REPO / "output" / "toy_surrogate_ablation_v2"
V1_OUTPUT = REPO / "output" / "toy_surrogate_ablation"
PYTHON_PATH = "/home/nacevedo/.conda/envs/fairness_env/bin/python"
FAMILIES = ("quadratic", "huber", "absolute")
LAMBDA_GRID: Tuple[float, ...] = (
    0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00, 1.15, 1.30, 1.50,
)
S_TARGETS: Tuple[float, ...] = (0.90, 0.80, 0.70, 0.60)
S_TARGETS_COMBINED: Tuple[float, ...] = (
    0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10,
)
S_TARGETS_ABSOLUTE_EXT: Tuple[float, ...] = (0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30)
RATIO_DISPLAY_S: Tuple[float, ...] = (1.00, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10)
S_TOL = 0.015
S_PLOT_LO, S_PLOT_HI = 0.50, 1.05
RHO_EXTRAP_CAP_MULT = 12.0
RHO_SEED_FALLBACK = {
    "quadratic": {0.30: 1.7, 0.25: 2.3, 0.20: 3.2, 0.15: 4.7, 0.10: 7.9},
    "huber": {0.25: 10.0, 0.20: 14.0, 0.15: 20.0, 0.10: 32.0},
    "absolute": {0.40: 2.7, 0.30: 5.5},
}
HUBER_Q_QUANTILE = 0.80


def _log(msg: str, **fields: Any) -> None:
    suffix = ""
    if fields:
        suffix = " | " + " | ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[toy_surrogate_ablation_v2] {msg}{suffix}", flush=True)


def _finite_num(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def classify_status(metrics: Dict[str, Any], pred: Optional[np.ndarray]) -> str:
    if pred is not None and (not np.all(np.isfinite(pred))):
        return "numerical_failure"
    keys = [
        "R2_price", "MAE_price", "MAPE", "RMSE_log", "PRD", "PRB", "MKI", "VEI",
        "Beta_log", "dCor_e_y", "C_train", "Beta_log_train", "s_train",
    ]
    for k in keys:
        if k in metrics and metrics[k] is not None and not _finite_num(metrics[k]):
            return "numerical_failure"
    return "ok"


def train_mechanism(y_train: np.ndarray, pred_train: np.ndarray) -> Dict[str, float]:
    e = pred_train - y_train
    c = y_train - float(np.mean(y_train))
    mech = paper_mechanism_metrics(y_train, pred_train)
    return {
        "C_train": covariance_C(e, c),
        "Beta_log_train": float(mech["Beta_log"]),
        "dCor_e_y_train": float(mech["dCor_e_y"]),
    }


def family_metrics_path(root: Path, family: str) -> Path:
    return root / "families" / family / "metrics.csv"


def family_status_path(root: Path, family: str) -> Path:
    return root / "families" / family / "fit_status.csv"


def pred_path(root: Path, family: str, tag: str) -> Path:
    return root / "families" / family / f"pred_{tag}.parquet"


def fit_one(
    *,
    data: Dict[str, Any],
    lgbm_params: dict,
    family: str,
    rho: float,
    huber_delta: Optional[float],
    verbose: bool = False,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    rss0 = v1._peak_rss_gb()
    status = "ok"
    error = ""
    model = ToyLGBSurrogate(
        rho=float(rho),
        penalty_shape=family,
        level_invariant=False,
        huber_delta=huber_delta if family == "huber" else None,
        lgbm_params=dict(lgbm_params),
        verbose=verbose,
        match_native_init=True,
        early_stopping_rounds=None,
    )
    try:
        model.fit(data["X_train"], data["y_train"])
        pred_eval = np.asarray(model.predict(data["X_eval"]), dtype=float).reshape(-1)
        pred_train = np.asarray(model.predict(data["X_train"]), dtype=float).reshape(-1)
        if (not np.all(np.isfinite(pred_eval))) or (not np.all(np.isfinite(pred_train))):
            status = "numerical_failure"
            error = "non_finite_prediction"
    except Exception as exc:
        status = "numerical_failure"
        error = f"{type(exc).__name__}: {exc}"
        pred_eval = np.full(data["y_eval"].shape, np.nan)
        pred_train = np.full(data["y_train"].shape, np.nan)
        model = None
    runtime = float(time.perf_counter() - t0)
    held = {}
    train_m = {"C_train": np.nan, "Beta_log_train": np.nan, "dCor_e_y_train": np.nan}
    p_linear = np.nan
    if status != "numerical_failure":
        raw = compute_taxation_metrics(data["y_eval"], pred_eval, scale="log", y_train=data["y_train"])
        held = v1.extract_required_metrics(raw)
        train_m = train_mechanism(data["y_train"], pred_train)
        if family == "huber" and huber_delta is not None:
            p_linear = huber_linear_share(pred_train - data["y_train"], data["y_train"] - float(np.mean(data["y_train"])), huber_delta)
        status = classify_status({**held, **train_m}, pred_eval)
    else:
        held = {k: float("nan") for k in v1.METRIC_KEYS}
    pred_df = pd.DataFrame(
        {
            "row_id": data["df_eval"].index.to_numpy(),
            "sale_date": pd.to_datetime(data["df_eval"]["meta_sale_date"]).to_numpy(),
            "y_true_log": data["y_eval"],
            "y_pred_log": pred_eval,
            "y_true": np.exp(data["y_eval"]),
            "y_pred": np.exp(pred_eval) if np.all(np.isfinite(pred_eval)) else np.full_like(pred_eval, np.nan),
        }
    )
    return {
        "status": status,
        "error": error,
        "runtime_sec": runtime,
        "peak_rss_gb": max(v1._peak_rss_gb(), rss0),
        "held": held,
        "train": train_m,
        "pred_eval": pred_df,
        "pred_train_log": pred_train,
        "p_linear": p_linear,
        "best_iteration": None if model is None else model.best_iteration(),
        "n_estimators": int(lgbm_params.get("n_estimators", -1)),
    }


def load_or_fit_lambda0(root: Path, args, data, lgbm_params) -> Dict[str, Any]:
    v1._ensure_dir(root / "predictions")
    v1_resid = V1_OUTPUT / "predictions" / "lambda0_train_residuals.parquet"
    v1_pred = V1_OUTPUT / "predictions" / "lambda0_shared.parquet"
    reuse = False
    if v1_resid.is_file() and v1_pred.is_file() and not bool(args.refit_lambda0):
        resid = pd.read_parquet(v1_resid)
        if int(len(resid)) == int(len(data["y_train"])):
            reuse = True
        else:
            _log(
                "V1 rho=0 residual count mismatch; refitting",
                v1_n=int(len(resid)),
                n_train=int(len(data["y_train"])),
            )
    if reuse:
        resid = pd.read_parquet(v1_resid)
        e0 = resid["e0"].to_numpy(dtype=float)
        c = resid["c"].to_numpy(dtype=float)
        pred_eval = pd.read_parquet(v1_pred)
        pred_train = resid["y_pred_train_log"].to_numpy(dtype=float)
        _log("reused V1 shared rho=0 fit", path=str(v1_resid), n=int(e0.size))
        source = "reused_v1_lambda0"
        runtime = 0.0
        peak = float("nan")
    else:
        _log("fitting shared rho=0 custom-objective model on full development")
        fit = fit_one(data=data, lgbm_params=lgbm_params, family="quadratic", rho=0.0, huber_delta=None)
        if fit["status"] != "ok":
            raise RuntimeError(f"lambda0 fit failed: {fit['error']}")
        pred_train = fit["pred_train_log"]
        e0 = pred_train - data["y_train"]
        c = data["y_train"] - float(np.mean(data["y_train"]))
        pred_eval = fit["pred_eval"]
        runtime = fit["runtime_sec"]
        peak = fit["peak_rss_gb"]
        source = "fitted_here"
    pred_eval = _ensure_price_cols(pred_eval)
    pred_eval.to_parquet(root / "predictions" / "lambda0_shared.parquet", index=False)
    pd.DataFrame({"e0": e0, "c": c, "y_train_log": data["y_train"], "y_pred_train_log": pred_train}).to_parquet(
        root / "predictions" / "lambda0_train_residuals.parquet", index=False
    )
    return {
        "e0": e0,
        "c": c,
        "pred_eval": pred_eval,
        "pred_train": pred_train,
        "source": source,
        "runtime_sec": runtime,
        "peak_rss_gb": peak,
    }


def write_calibration(root: Path, e0: np.ndarray, c: np.ndarray, lambdas: Sequence[float]) -> Tuple[pd.DataFrame, float, float]:
    q0 = e0 * c
    delta = huber_delta_from_q(q0, quantile=HUBER_Q_QUANTILE)
    p0 = huber_linear_share(e0, c, delta)
    rows = []
    g_ref = None
    for fam in FAMILIES:
        g_pen = fairness_penalty_gradient_unit_rho_from_ec(
            e0, c, penalty_shape=fam, level_invariant=False, huber_delta=delta if fam == "huber" else None
        )
        gm = float(np.sqrt(np.mean(g_pen ** 2)))
        if fam == "quadratic":
            g_ref = gm
        rows.append({"family": fam, "G_m": gm, "huber_delta": delta if fam == "huber" else float("nan"), "p_linear_baseline": p0 if fam == "huber" else float("nan")})
    if g_ref is None or g_ref <= 0:
        raise RuntimeError("Quadratic G_m is not positive.")
    out = []
    for row in rows:
        mult = float(g_ref / row["G_m"]) if row["G_m"] > 0 else float("nan")
        for lam in lambdas:
            rec = dict(row)
            rec["G_ref"] = float(g_ref)
            rec["multiplier_GQ_over_Gm"] = mult
            rec["lambda"] = float(lam)
            rec["raw_rho"] = float(lam) * mult
            rec["note"] = "Equal lambda is not equal regularization; RMS fairness-gradient scale only."
            out.append(rec)
    calib = pd.DataFrame(out)
    calib.to_csv(root / "rho_scale_calibration.csv", index=False)
    _log("calibration written", huber_delta=f"{delta:.6g}", p_linear0=f"{p0:.4f}")
    return calib, float(delta), float(p0)


def lookup_rho(calib: pd.DataFrame, family: str, lam: float) -> float:
    sub = calib.loc[(calib["family"] == family) & np.isclose(calib["lambda"].astype(float), float(lam), atol=1e-12)]
    if sub.empty:
        raise KeyError(f"missing rho for {family} lambda={lam}")
    return float(sub["raw_rho"].iloc[0])


def row_from_fit(
    *,
    family: str,
    lam: float,
    raw_rho: float,
    run_type: str,
    target_s: Optional[float],
    fit: Dict[str, Any],
    beta0: float,
    c0: float,
    delta: float,
    n_train: int,
    n_eval: int,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    beta_tr = float(fit["train"]["Beta_log_train"]) if _finite_num(fit["train"].get("Beta_log_train")) else float("nan")
    s = float(beta_tr / beta0) if (_finite_num(beta_tr) and _finite_num(beta0) and abs(beta0) > 0) else float("nan")
    metrics = {
        "experiment_label": EXPERIMENT_LABEL,
        "family": family,
        "level_invariant": False,
        "lambda": float(lam) if lam is not None else float("nan"),
        "raw_rho": float(raw_rho),
        "run_type": run_type,
        "target_s": float(target_s) if target_s is not None else float("nan"),
        "huber_delta": delta if family == "huber" else float("nan"),
        "p_linear": fit["p_linear"] if family == "huber" else float("nan"),
        "C_train": fit["train"]["C_train"],
        "C_train_0": c0,
        "Beta_log_train": beta_tr,
        "Beta_log_train_0": beta0,
        "s_train": s,
        "n_train": int(n_train),
        "n_eval": int(n_eval),
        "runtime_sec": fit["runtime_sec"],
        "peak_rss_gb": fit["peak_rss_gb"],
        "best_iteration": fit["best_iteration"] if fit["best_iteration"] is not None else float("nan"),
        "n_estimators": fit["n_estimators"],
        **fit["held"],
    }
    pathological = bool(
        fit["status"] == "ok"
        and _finite_num(metrics.get("R2_price"))
        and float(metrics.get("R2_price")) < -1.0
    )
    status_label = "numerical_failure" if fit["status"] == "numerical_failure" else (
        "pathological_finite" if pathological else "ok"
    )
    metrics["status"] = status_label
    metrics["error"] = fit["error"]
    status = {
        "family": family,
        "run_type": run_type,
        "lambda": metrics["lambda"],
        "raw_rho": raw_rho,
        "target_s": metrics["target_s"],
        "s_train": s,
        "status": status_label,
        "error": fit["error"],
        "pathological_finite": pathological,
        "R2_price": metrics.get("R2_price"),
        "Beta_log": metrics.get("Beta_log"),
        "dCor_e_y": metrics.get("dCor_e_y"),
        "C_train": metrics.get("C_train"),
        "p_linear": metrics.get("p_linear"),
    }
    return metrics, status


def run_calibrate(args) -> None:
    root = v1._ensure_dir(Path(args.output_root))
    v1._ensure_dir(root / "logs")
    params = v1.load_params(Path(args.params))
    n_jobs = v1._allocated_cpus()
    lgbm_params = v1.load_frozen_lgbm_params(Path(args.lgbm_config_json), n_jobs=n_jobs, n_estimators=args.n_estimators)
    data = v1.load_experiment_data(data_path=Path(args.data_path), params=params, sample_frac=args.sample_frac, seed=int(args.seed))
    _log("data ready (development -> held-out; no CV)", n_train=data["n_train"], n_eval=data["n_eval"], n_estimators=int(lgbm_params["n_estimators"]), n_jobs=n_jobs)
    lam0 = load_or_fit_lambda0(root, args, data, lgbm_params)
    calib, delta, p0 = write_calibration(root, lam0["e0"], lam0["c"], list(args.lambdas))
    train_m = train_mechanism(data["y_train"], lam0["pred_train"])
    held = v1.extract_required_metrics(
        compute_taxation_metrics(data["y_eval"], lam0["pred_eval"]["y_pred_log"].to_numpy(dtype=float), scale="log", y_train=data["y_train"])
    )
    fake_fit = {
        "status": "ok",
        "error": "",
        "runtime_sec": lam0["runtime_sec"],
        "peak_rss_gb": lam0["peak_rss_gb"] if _finite_num(lam0["peak_rss_gb"]) else 0.0,
        "held": held,
        "train": train_m,
        "pred_eval": lam0["pred_eval"],
        "pred_train_log": lam0["pred_train"],
        "p_linear": p0,
        "best_iteration": None,
        "n_estimators": int(lgbm_params["n_estimators"]),
    }
    rows = []
    stats = []
    for fam in FAMILIES:
        m, st = row_from_fit(
            family=fam, lam=0.0, raw_rho=0.0, run_type="shared_lambda0", target_s=None,
            fit=fake_fit, beta0=train_m["Beta_log_train"], c0=train_m["C_train"], delta=delta,
            n_train=data["n_train"], n_eval=data["n_eval"],
        )
        rows.append(m)
        stats.append(st)
        v1._ensure_dir(root / "families" / fam)
        pd.DataFrame([m]).to_csv(root / "families" / fam / "metrics_lambda0.csv", index=False)
        pred0 = _ensure_price_cols(lam0["pred_eval"])
        keep = [c for c in ("row_id", "sale_date", "y_true_log", "y_pred_log", "y_true", "y_pred") if c in pred0.columns]
        v1.save_pred(
            pred_path(root, fam, "lambda_0"),
            pred0[keep].copy(),
            {"family": fam, "lambda": 0.0, "raw_rho": 0.0, "run_type": "shared_lambda0"},
        )
    pd.DataFrame(rows).to_csv(root / "metrics_lambda0.csv", index=False)
    pd.DataFrame(stats).to_csv(root / "fit_status_lambda0.csv", index=False)
    write_config(root, args, data, lgbm_params, calib, delta, p0, extra={"lambda0_source": lam0["source"], "Beta_log_train_0": train_m["Beta_log_train"], "C_train_0": train_m["C_train"]})
    _log("calibrate done", beta0=f"{train_m['Beta_log_train']:.6g}")


def run_fit_family(args) -> None:
    root = Path(args.output_root)
    family = str(args.family)
    if family not in FAMILIES:
        raise SystemExit(f"family must be one of {FAMILIES}")
    calib = pd.read_csv(root / "rho_scale_calibration.csv")
    delta = float(calib.loc[calib["family"] == "huber", "huber_delta"].iloc[0])
    params = v1.load_params(Path(args.params))
    n_jobs = v1._allocated_cpus()
    lgbm_params = v1.load_frozen_lgbm_params(Path(args.lgbm_config_json), n_jobs=n_jobs, n_estimators=args.n_estimators)
    data = v1.load_experiment_data(data_path=Path(args.data_path), params=params, sample_frac=args.sample_frac, seed=int(args.seed))
    lam0_all = pd.read_csv(root / "metrics_lambda0.csv")
    lam0_row = lam0_all.loc[lam0_all["family"] == family].iloc[0]
    beta0 = float(lam0_row["Beta_log_train_0"]) if "Beta_log_train_0" in lam0_row.index else float(lam0_row["Beta_log_train"])
    c0 = float(lam0_row["C_train_0"]) if "C_train_0" in lam0_row.index else float(lam0_row["C_train"])
    rows = [lam0_row.to_dict()]
    st0 = pd.read_csv(root / "fit_status_lambda0.csv")
    stats = [st0.loc[st0["family"] == family].iloc[0].to_dict()]
    lambdas = [float(x) for x in args.lambdas if float(x) > 0.0]
    _log("family dense start", family=family, n_lambdas=len(lambdas), n_train=data["n_train"], n_estimators=int(lgbm_params["n_estimators"]))
    for lam in lambdas:
        rho = lookup_rho(calib, family, lam)
        _log("fit start", family=family, lam=lam, raw_rho=f"{rho:.6g}")
        fit = fit_one(data=data, lgbm_params=lgbm_params, family=family, rho=rho, huber_delta=delta if family == "huber" else None)
        m, st = row_from_fit(
            family=family, lam=lam, raw_rho=rho, run_type="dense", target_s=None, fit=fit,
            beta0=beta0, c0=c0, delta=delta, n_train=data["n_train"], n_eval=data["n_eval"],
        )
        rows.append(m)
        stats.append(st)
        if fit["status"] != "numerical_failure":
            v1.save_pred(pred_path(root, family, f"lambda_{lam:g}"), fit["pred_eval"], {"family": family, "lambda": lam, "raw_rho": rho, "run_type": "dense"})
        _log("fit done", family=family, lam=lam, status=st["status"], s=m["s_train"], runtime=f"{fit['runtime_sec']:.1f}")
    dense = pd.DataFrame(rows)
    if not bool(args.skip_refine):
        refine_rows, refine_stats = run_refinements(family, dense, data, lgbm_params, delta, beta0, c0, root)
        rows.extend(refine_rows)
        stats.extend(refine_stats)
    out_m = pd.DataFrame(rows)
    out_s = pd.DataFrame(stats)
    dest_m = family_metrics_path(root, family)
    dest_s = family_status_path(root, family)
    v1._ensure_dir(dest_m.parent)
    out_m.to_csv(dest_m, index=False)
    out_s.to_csv(dest_s, index=False)
    _log("family written", family=family, n=int(len(out_m)))


def _usable_match_rows(dense: pd.DataFrame) -> pd.DataFrame:
    work = dense.copy()
    if "status" in work.columns:
        work = work.loc[work["status"].astype(str) != "numerical_failure"]
    work = work.loc[work["s_train"].apply(_finite_num)]
    work = work.loc[work["raw_rho"].apply(_finite_num)]
    work = work.loc[work["raw_rho"].astype(float) >= 0.0]
    work = work.loc[work["s_train"].astype(float) > 0.0]
    work = work.loc[work["R2_price"].apply(_finite_num)]
    return work.sort_values("raw_rho")


def _first_straddle_rho(usable: pd.DataFrame, sstar: float) -> Optional[float]:
    if usable is None or len(usable) < 2:
        return None
    s = usable["s_train"].astype(float).to_numpy()
    r = usable["raw_rho"].astype(float).to_numpy()
    for i in range(len(s) - 1):
        if (s[i] - float(sstar)) * (s[i + 1] - float(sstar)) <= 0.0:
            s1, s2 = float(s[i]), float(s[i + 1])
            r1, r2 = float(r[i]), float(r[i + 1])
            if abs(s1 - s2) < 1e-12:
                return 0.5 * (r1 + r2)
            return r1 + (r2 - r1) * (float(sstar) - s1) / (s2 - s1)
    return None


def run_refinements(family, dense: pd.DataFrame, data, lgbm_params, delta, beta0, c0, root) -> Tuple[List[dict], List[dict]]:
    usable = _usable_match_rows(dense)
    rows: List[dict] = []
    stats: List[dict] = []
    for sstar in S_TARGETS:
        if usable.empty:
            stats.append(
                {
                    "family": family,
                    "run_type": "match",
                    "target_s": sstar,
                    "status": "not_attained",
                    "error": "no_stable_positive_s",
                    "s_train": np.nan,
                    "lambda": np.nan,
                    "raw_rho": np.nan,
                    "pathological_finite": False,
                }
            )
            _log("target not attained", family=family, sstar=sstar)
            continue
        cand = usable.copy()
        cand["abs_ds"] = (cand["s_train"].astype(float) - float(sstar)).abs()
        nearest = cand.sort_values("abs_ds").iloc[0]
        if float(nearest["abs_ds"]) <= S_TOL:
            stats.append(
                {
                    "family": family,
                    "run_type": "match_reuse",
                    "target_s": sstar,
                    "status": "ok",
                    "error": "dense_within_tol",
                    "s_train": float(nearest["s_train"]),
                    "lambda": float(nearest["lambda"]),
                    "raw_rho": float(nearest["raw_rho"]),
                    "pathological_finite": False,
                }
            )
            _log("match reuse dense", family=family, sstar=sstar, s=float(nearest["s_train"]), rho=float(nearest["raw_rho"]))
            continue
        rho_star = _first_straddle_rho(usable, float(sstar))
        if rho_star is None or rho_star <= 0.0:
            stats.append(
                {
                    "family": family,
                    "run_type": "match",
                    "target_s": sstar,
                    "status": "not_attained",
                    "error": "not_bracketed" if rho_star is None else "interpolated_rho_nonpositive",
                    "s_train": float(nearest["s_train"]),
                    "lambda": float(nearest["lambda"]),
                    "raw_rho": float("nan") if rho_star is None else float(rho_star),
                    "pathological_finite": False,
                }
            )
            _log("target not bracketed", family=family, sstar=sstar, nearest_s=float(nearest["s_train"]))
            continue
        _log("refinement fit", family=family, sstar=sstar, raw_rho=f"{rho_star:.6g}")
        fit = fit_one(
            data=data,
            lgbm_params=lgbm_params,
            family=family,
            rho=float(rho_star),
            huber_delta=delta if family == "huber" else None,
        )
        m, st = row_from_fit(
            family=family,
            lam=float("nan"),
            raw_rho=float(rho_star),
            run_type="refinement",
            target_s=float(sstar),
            fit=fit,
            beta0=beta0,
            c0=c0,
            delta=delta,
            n_train=data["n_train"],
            n_eval=data["n_eval"],
        )
        rows.append(m)
        stats.append(st)
        if fit["status"] != "numerical_failure":
            v1.save_pred(
                pred_path(root, family, f"refine_s_{sstar:g}"),
                fit["pred_eval"],
                {"family": family, "raw_rho": rho_star, "run_type": "refinement", "target_s": sstar},
            )
        _log("refinement done", family=family, sstar=sstar, s=m["s_train"], status=st["status"])
    return rows, stats


def _log_linear_rho_guess(usable: pd.DataFrame, sstar: float) -> Optional[float]:
    sub = usable.loc[usable["raw_rho"].astype(float) > 1e-12].sort_values("raw_rho")
    if len(sub) < 3:
        return None
    tail = sub.tail(min(8, len(sub)))
    x = np.log(tail["raw_rho"].astype(float).to_numpy())
    y = tail["s_train"].astype(float).to_numpy()
    b, a = np.polyfit(x, y, 1)
    if (not np.isfinite(b)) or abs(float(b)) < 1e-12:
        return None
    rho = float(np.exp((float(sstar) - float(a)) / float(b)))
    if (not np.isfinite(rho)) or rho <= 0.0:
        return None
    max_rho = float(sub["raw_rho"].max())
    return min(rho, RHO_EXTRAP_CAP_MULT * max_rho)


def _seed_rho(family: str, sstar: float) -> Optional[float]:
    table = RHO_SEED_FALLBACK.get(family, {})
    for k, v in table.items():
        if abs(float(k) - float(sstar)) < 1e-9:
            return float(v)
    return None


def _propose_rho(family: str, usable: pd.DataFrame, sstar: float) -> Tuple[Optional[float], str]:
    interp = _first_straddle_rho(usable, float(sstar))
    if interp is not None and interp > 0.0:
        return float(interp), "interpolate"
    extra = _log_linear_rho_guess(usable, float(sstar))
    if extra is not None and extra > 0.0:
        return float(extra), "extrapolate"
    seed = _seed_rho(family, float(sstar))
    if seed is not None and seed > 0.0:
        return float(seed), "seed_fallback"
    return None, "no_rho_guess"


def _nearest_within_tol(usable: pd.DataFrame, sstar: float) -> Optional[pd.Series]:
    if usable is None or usable.empty:
        return None
    cand = usable.loc[usable["s_train"].astype(float) > 0.0].copy()
    if cand.empty:
        return None
    cand["abs_ds"] = (cand["s_train"].astype(float) - float(sstar)).abs()
    nearest = cand.sort_values("abs_ds").iloc[0]
    if float(nearest["abs_ds"]) <= S_TOL:
        return nearest
    return None


def run_extend_targets(args) -> None:
    """Append missing training-s refinements. Does not rerun dense/calibration/rho=0."""
    root = Path(args.output_root)
    family = str(args.family)
    if family not in FAMILIES:
        raise SystemExit(f"family must be one of {FAMILIES}")
    targets = [float(x) for x in args.targets]
    dest_m = family_metrics_path(root, family)
    dest_s = family_status_path(root, family)
    if not dest_m.is_file():
        raise FileNotFoundError(f"missing family metrics {dest_m}")
    existing = pd.read_csv(dest_m)
    existing_status = pd.read_csv(dest_s) if dest_s.is_file() else pd.DataFrame()
    calib = pd.read_csv(root / "rho_scale_calibration.csv")
    delta = float(calib.loc[calib["family"] == "huber", "huber_delta"].iloc[0])
    params = v1.load_params(Path(args.params))
    n_jobs = v1._allocated_cpus()
    lgbm_params = v1.load_frozen_lgbm_params(Path(args.lgbm_config_json), n_jobs=n_jobs, n_estimators=args.n_estimators)
    data = v1.load_experiment_data(
        data_path=Path(args.data_path), params=params, sample_frac=args.sample_frac, seed=int(args.seed)
    )
    beta0 = float(existing["Beta_log_train_0"].dropna().iloc[0])
    c0 = float(existing["C_train_0"].dropna().iloc[0])
    usable = _usable_match_rows(existing)
    new_rows: List[dict] = []
    new_stats: List[dict] = []
    _log(
        "extend-targets start",
        family=family,
        n_targets=len(targets),
        n_existing=int(len(existing)),
        n_train=data["n_train"],
        huber_delta=f"{delta:.6g}",
        n_estimators=int(lgbm_params["n_estimators"]),
    )

    def _combined_metrics() -> pd.DataFrame:
        return pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True) if new_rows else existing

    def _persist() -> None:
        out_m = _combined_metrics()
        out_s = existing_status
        if new_stats:
            out_s = (
                pd.concat([existing_status, pd.DataFrame(new_stats)], ignore_index=True)
                if not existing_status.empty
                else pd.DataFrame(new_stats)
            )
        out_m.to_csv(dest_m, index=False)
        out_s.to_csv(dest_s, index=False)

    def _one_fit(rho: float, note: str, tag: str) -> Tuple[Optional[dict], Optional[dict]]:
        if (not np.isfinite(rho)) or float(rho) <= 0.0:
            return None, None
        _log("extend fit", family=family, sstar=sstar, raw_rho=f"{float(rho):.6g}", note=note)
        fit = fit_one(
            data=data,
            lgbm_params=lgbm_params,
            family=family,
            rho=float(rho),
            huber_delta=delta if family == "huber" else None,
        )
        m, st = row_from_fit(
            family=family,
            lam=float("nan"),
            raw_rho=float(rho),
            run_type="refinement",
            target_s=float(sstar),
            fit=fit,
            beta0=beta0,
            c0=c0,
            delta=delta,
            n_train=data["n_train"],
            n_eval=data["n_eval"],
        )
        m["pred_tag"] = tag
        m["rho_note"] = note
        new_rows.append(m)
        new_stats.append(st)
        if fit["status"] != "numerical_failure":
            v1.save_pred(
                pred_path(root, family, tag),
                fit["pred_eval"],
                {"family": family, "raw_rho": float(rho), "run_type": "refinement", "target_s": sstar, "rho_note": note},
            )
        _log("extend fit done", family=family, sstar=sstar, s=m["s_train"], status=st["status"], rho=f"{float(rho):.6g}")
        return m, st

    for sstar in targets:
        hit = _nearest_within_tol(usable, float(sstar))
        if hit is not None:
            new_stats.append(
                {
                    "family": family,
                    "run_type": "match_reuse",
                    "target_s": float(sstar),
                    "status": "ok",
                    "error": "existing_within_tol",
                    "s_train": float(hit["s_train"]),
                    "lambda": float(hit["lambda"]) if _finite_num(hit.get("lambda")) else float("nan"),
                    "raw_rho": float(hit["raw_rho"]),
                    "pathological_finite": False,
                }
            )
            _log("reuse existing", family=family, sstar=sstar, s=float(hit["s_train"]), rho=float(hit["raw_rho"]))
            continue
        rho, how = _propose_rho(family, usable, float(sstar))
        if rho is None:
            new_stats.append(
                {
                    "family": family,
                    "run_type": "match",
                    "target_s": float(sstar),
                    "status": "not_attained",
                    "error": how,
                    "s_train": np.nan,
                    "lambda": np.nan,
                    "raw_rho": np.nan,
                    "pathological_finite": False,
                }
            )
            _log("target not attained", family=family, sstar=sstar, reason=how)
            continue
        m, st = _one_fit(float(rho), how, f"refine_s_{sstar:g}")
        _persist()
        if m is None:
            continue
        achieved = m.get("s_train")
        ok_s = _finite_num(achieved) and float(achieved) > 0.0
        if ok_s and abs(float(achieved) - float(sstar)) <= S_TOL:
            usable = _usable_match_rows(_combined_metrics())
            continue
        if (not ok_s) or float(achieved) <= 0.0 or str(st.get("status")) == "numerical_failure":
            new_stats.append(
                {
                    "family": family,
                    "run_type": "match",
                    "target_s": float(sstar),
                    "status": "not_attained",
                    "error": "first_fit_unstable",
                    "s_train": achieved,
                    "lambda": np.nan,
                    "raw_rho": float(rho),
                    "pathological_finite": bool(st.get("pathological_finite", False)),
                }
            )
            usable = _usable_match_rows(_combined_metrics())
            _persist()
            continue
        usable = _usable_match_rows(_combined_metrics())
        rho2, how2 = _propose_rho(family, usable, float(sstar))
        if rho2 is None or abs(float(rho2) - float(rho)) < 1e-12:
            _log("no corrective rho", family=family, sstar=sstar, achieved=achieved)
            continue
        m2, st2 = _one_fit(float(rho2), f"corrective_{how2}", f"refine_s_{sstar:g}_c")
        if m2 is not None:
            s2 = m2.get("s_train")
            if (not _finite_num(s2)) or float(s2) <= 0.0 or str(st2.get("status")) == "numerical_failure":
                new_stats.append(
                    {
                        "family": family,
                        "run_type": "match",
                        "target_s": float(sstar),
                        "status": "not_attained",
                        "error": "corrective_fit_unstable",
                        "s_train": s2,
                        "lambda": np.nan,
                        "raw_rho": float(rho2),
                        "pathological_finite": bool(st2.get("pathological_finite", False)),
                    }
                )
        usable = _usable_match_rows(_combined_metrics())
        _persist()
    _persist()
    _log("extend-targets written", family=family, n_new=int(len(new_rows)))


def write_config(root, args, data, lgbm_params, calib, delta, p0, extra=None):
    git = git_state(REPO)
    payload = {
        "experiment_label": EXPERIMENT_LABEL,
        "experiment": "toy_surrogate_ablation_v2",
        "scientific_status": "toy_fixed_level_quadratic_huber_absolute_not_canonical",
        **git,
        "python_path": PYTHON_PATH,
        "environment": "fairness_env",
        "versions": package_versions(),
        "data_path": str(Path(args.data_path).resolve()),
        "data_version": "CCAO/2025/training_data.parquet",
        "heldout_test_mode": "pre_assessment_tail",
        "protocol": {
            "train": "full current development sample",
            "evaluate": "current primary held-out sample only",
            "no_cv_folds": True,
            "no_rolling_origin": True,
            "no_early_stopping": True,
            "no_level_invariant": True,
            "no_capped_quadratic": True,
            "quantities_varied": ["family", "lambda", "raw_rho"],
        },
        "train_n": data["n_train"],
        "eval_n": data["n_eval"],
        "train_period": [data["train_start"], data["train_end"]],
        "eval_period": [data["eval_start"], data["eval_end"]],
        "feature_count": data["n_features"],
        "lgbm_config_json": str(Path(args.lgbm_config_json).resolve()),
        "frozen_lgbm_params": dict(lgbm_params),
        "lgbm_params_sha256": lgbm_params_hash(lgbm_params),
        "seed": int(args.seed),
        "lambda_grid": [float(x) for x in args.lambdas],
        "matched_retention_targets": list(S_TARGETS),
        "matched_retention_tolerance": S_TOL,
        "huber_delta": float(delta),
        "huber_delta_rule": "Q_0.80(|e0 * c|) on the shared full-development rho=0 residuals",
        "huber_p_linear_baseline": float(p0),
        "rho_scale_factors": {
            fam: {
                "G_m": float(calib.loc[calib["family"] == fam, "G_m"].iloc[0]),
                "multiplier_GQ_over_Gm": float(calib.loc[calib["family"] == fam, "multiplier_GQ_over_Gm"].iloc[0]),
            }
            for fam in FAMILIES
        },
        "objectives": {
            "quadratic": {"Psi": "(1/n) sum (e c)^2", "g": "e + rho e c^2", "h": "1 + rho c^2", "matches_canonical": True},
            "absolute": {"Psi": "(1/n) sum |e c|", "g": "e + (rho/2) |c| sign(e)", "h": "1", "sign0": "sign(0)=0"},
            "huber": {
                "Psi": "(1/n) sum phi_delta(e c)",
                "phi": "q^2 if |q|<=delta else 2 delta |q| - delta^2",
                "inside": "g = e + rho e c^2 ; h = 1 + rho c^2",
                "outside": "g = e + rho delta |c| sign(e) ; h = 1",
                "smoothness": "C^1; Hessian jumps at |q|=delta",
            },
        },
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION") or "sched_mit_sloan_batch_r8",
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK") or "8",
            "requested": {
                "partition": "sched_mit_sloan_batch_r8",
                "cpus_per_task": 8,
                "mem_gb": 24,
                "time_calibrate": "02:00:00",
                "time_family": "04:00:00",
                "gpus": 0,
                "basis": "V1 probe peak RSS 2.53 GB and ~50s per 994-tree fit; 24G is a safety margin, not another oversized probe.",
            },
        },
        "does_not_overwrite_v1": str(V1_OUTPUT),
    }
    if extra:
        payload.update(extra)
    write_json(root / "config.json", payload)


def _collect(root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ms, ss = [], []
    for fam in FAMILIES:
        mp = family_metrics_path(root, fam)
        sp = family_status_path(root, fam)
        if mp.is_file():
            ms.append(pd.read_csv(mp))
        if sp.is_file():
            ss.append(pd.read_csv(sp))
    if not ms:
        raise FileNotFoundError("No family metrics found.")
    metrics = pd.concat(ms, ignore_index=True)
    status = pd.concat(ss, ignore_index=True) if ss else pd.DataFrame()
    return metrics, status


def _targets_for(fam: str) -> Tuple[float, ...]:
    if fam == "absolute":
        return S_TARGETS_ABSOLUTE_EXT
    return S_TARGETS_COMBINED


def _is_old_target(sstar: float) -> bool:
    return any(abs(float(sstar) - float(t)) < 1e-9 for t in S_TARGETS)


def _lookup_frozen_row(prev: Optional[pd.DataFrame], fam: str, sstar: float) -> Optional[pd.Series]:
    if prev is None or prev.empty:
        return None
    sub = prev.loc[prev["family"] == fam]
    if "attained" in sub.columns:
        sub = sub.loc[sub["attained"] == True]
    sub = sub.loc[np.isclose(sub["target_s"].astype(float), float(sstar), atol=1e-12)]
    if sub.empty:
        return None
    return sub.iloc[0]


def compute_nl_shape(pred: pd.DataFrame, n_bins: int = 30) -> float:
    pred = _ensure_price_cols(pred)
    sale = pred["y_true"].to_numpy(dtype=float)
    ratio = pred["y_pred"].to_numpy(dtype=float) / np.clip(sale, 1e-12, None)
    keep = np.isfinite(sale) & np.isfinite(ratio) & (sale > 0) & (ratio > 0)
    sale, ratio = sale[keep], ratio[keep]
    if sale.size < n_bins:
        return float("nan")
    order = np.argsort(sale, kind="mergesort")
    sale, ratio = sale[order], ratio[order]
    log_r = np.log(ratio)
    log_sale = np.log(sale)
    m_b = []
    x_b = []
    for idx in np.array_split(np.arange(len(sale)), n_bins):
        if idx.size == 0:
            continue
        m_b.append(float(np.median(log_r[idx])))
        x_b.append(float(np.median(log_sale[idx])))
    m_b = np.asarray(m_b, dtype=float)
    x_b = np.asarray(x_b, dtype=float)
    if m_b.size < 3 or (not np.all(np.isfinite(m_b))) or (not np.all(np.isfinite(x_b))):
        return float("nan")
    coef = np.polyfit(x_b, m_b, 1)
    mhat = np.polyval(coef, x_b)
    return float(np.sqrt(np.mean((m_b - mhat) ** 2)))


def _pred_tag_of(chosen: pd.Series, sstar: float, note: str) -> str:
    if "pred_tag" in chosen.index:
        tag = chosen.get("pred_tag")
        if isinstance(tag, str) and tag.strip() and tag.strip().lower() not in {"nan", "none"}:
            return str(tag).strip()
    if note == "refinement":
        return f"refine_s_{float(sstar):g}"
    if _finite_num(chosen.get("lambda")):
        return f"lambda_{float(chosen['lambda']):g}"
    return f"refine_s_{float(sstar):g}"


def select_matched(metrics: pd.DataFrame, status: pd.DataFrame, previous: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    rows = []
    for fam in FAMILIES:
        fam_m = metrics.loc[metrics["family"] == fam].copy()
        if "status" in fam_m.columns:
            fam_m = fam_m.loc[fam_m["status"].astype(str) != "numerical_failure"]
        fam_m = fam_m.loc[fam_m["s_train"].apply(_finite_num)]
        fam_m = fam_m.loc[fam_m["s_train"].astype(float) > 0.0]
        fam_m = fam_m.loc[fam_m["R2_price"].apply(_finite_num)]
        for sstar in _targets_for(fam):
            if _is_old_target(sstar):
                frozen = _lookup_frozen_row(previous, fam, float(sstar))
                if frozen is not None:
                    rec = frozen.to_dict()
                    rec["family"] = fam
                    rec["target_s"] = float(sstar)
                    rec["attained"] = True
                    rec["frozen_v2_original"] = True
                    rows.append(rec)
                    continue
            chosen = None
            note = ""
            ref = fam_m.loc[
                (fam_m["run_type"] == "refinement")
                & np.isclose(fam_m["target_s"].astype(float), float(sstar), atol=1e-12)
            ]
            if not ref.empty:
                ref = ref.copy()
                ref["abs_ds"] = (ref["s_train"].astype(float) - float(sstar)).abs()
                cand_ref = ref.sort_values("abs_ds").iloc[0]
                if float(cand_ref["s_train"]) > 0.0 and str(cand_ref.get("status", "ok")) != "numerical_failure":
                    if float(cand_ref["abs_ds"]) <= S_TOL:
                        chosen = cand_ref
                        note = "refinement"
            if chosen is None:
                cand = fam_m.copy()
                if cand.empty:
                    rows.append({"family": fam, "target_s": sstar, "attained": False, "note": "not_attained"})
                    continue
                cand["abs_ds"] = (cand["s_train"].astype(float) - float(sstar)).abs()
                nearest = cand.sort_values("abs_ds").iloc[0]
                if float(nearest["abs_ds"]) <= S_TOL:
                    chosen = nearest
                    note = "dense_within_tol" if str(nearest.get("run_type", "")) != "refinement" else "refinement"
                elif not ref.empty:
                    rows.append(
                        {
                            "family": fam,
                            "target_s": sstar,
                            "attained": False,
                            "note": "not_attained",
                            "nearest_s_train": float(nearest["s_train"]),
                            "nearest_raw_rho": float(nearest["raw_rho"]),
                        }
                    )
                    continue
                else:
                    rows.append(
                        {
                            "family": fam,
                            "target_s": sstar,
                            "attained": False,
                            "note": "not_attained",
                            "nearest_s_train": float(nearest["s_train"]),
                            "nearest_raw_rho": float(nearest["raw_rho"]),
                        }
                    )
                    continue
            rec = {
                "family": fam,
                "target_s": float(sstar),
                "attained": True,
                "note": note,
                "achieved_s_train": float(chosen["s_train"]),
                "raw_rho": float(chosen["raw_rho"]),
                "lambda": float(chosen["lambda"]) if _finite_num(chosen.get("lambda")) else float("nan"),
                "huber_delta": float(chosen["huber_delta"]) if fam == "huber" else float("nan"),
                "p_linear": float(chosen["p_linear"]) if fam == "huber" and _finite_num(chosen.get("p_linear")) else float("nan"),
                "R2_price": float(chosen["R2_price"]),
                "MAE_price": float(chosen["MAE_price"]),
                "PRD": float(chosen["PRD"]),
                "PRB": float(chosen["PRB"]),
                "MKI": float(chosen["MKI"]),
                "VEI": float(chosen["VEI"]),
                "Beta_log": float(chosen["Beta_log"]),
                "dCor_e_y": float(chosen["dCor_e_y"]),
                "run_type": chosen["run_type"],
                "pred_tag": _pred_tag_of(chosen, sstar, note),
                "frozen_v2_original": False,
            }
            rows.append(rec)
    return pd.DataFrame(rows)


def attach_nl_shape(root: Path, matched: pd.DataFrame) -> pd.DataFrame:
    out = matched.copy()
    vals = []
    for _, rec in out.iterrows():
        if not bool(rec.get("attained", False)):
            vals.append(float("nan"))
            continue
        tag = rec.get("pred_tag")
        p = pred_path(root, rec["family"], tag) if isinstance(tag, str) else None
        if p is None or (not p.is_file()):
            vals.append(float("nan"))
            continue
        try:
            pred = pd.read_parquet(p)
            vals.append(compute_nl_shape(pred))
        except Exception:
            vals.append(float("nan"))
    out["NL_shape"] = vals
    return out


def _set_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10.5,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _ensure_price_cols(pred: pd.DataFrame) -> pd.DataFrame:
    out = pred.copy()
    if "y_pred" not in out.columns and "y_pred_log" in out.columns:
        out["y_pred"] = np.exp(out["y_pred_log"].to_numpy(dtype=float))
    if "y_true" not in out.columns and "y_true_log" in out.columns:
        out["y_true"] = np.exp(out["y_true_log"].to_numpy(dtype=float))
    return out


def plot_matched_ratio(root: Path, matched: pd.DataFrame, metrics: pd.DataFrame) -> Optional[Path]:
    import matplotlib.pyplot as plt
    _set_style()
    frames = []
    base = pred_path(root, "quadratic", "lambda_0")
    if not base.is_file():
        base = root / "predictions" / "lambda0_shared.parquet"
    if base.is_file():
        pred = _ensure_price_cols(pd.read_parquet(base))
        sale = pred["y_true"].to_numpy(dtype=float)
        ratio = pred["y_pred"].to_numpy(dtype=float) / sale
        b = v1.equal_count_bins(sale, ratio)
        b["family"] = "baseline"
        b["target_s"] = 1.0
        frames.append(b)
    attained = matched.loc[matched["attained"] == True] if "attained" in matched.columns else matched
    for _, rec in attained.iterrows():
        if not bool(rec.get("attained", False)):
            continue
        p = pred_path(root, rec["family"], rec["pred_tag"])
        if not p.is_file():
            continue
        pred = _ensure_price_cols(pd.read_parquet(p))
        sale = pred["y_true"].to_numpy(dtype=float)
        ratio = pred["y_pred"].to_numpy(dtype=float) / np.clip(sale, 1e-12, None)
        if not np.all(np.isfinite(ratio)):
            continue
        b = v1.equal_count_bins(sale, ratio)
        b["family"] = rec["family"]
        b["target_s"] = float(rec["target_s"])
        frames.append(b)
    if not frames:
        return None
    bins = pd.concat(frames, ignore_index=True)
    # Keep all matched bins for reproducibility; plot only the display anchors.
    bins.to_csv(root / "matched_ratio_bins.csv", index=False)
    display_s = list(RATIO_DISPLAY_S)
    plot_bins = bins.loc[
        (bins["family"] == "baseline")
        | bins["target_s"].apply(lambda s: any(np.isclose(float(s), float(t), atol=1e-12) for t in display_s))
    ].copy()
    axis_bins = plot_bins.loc[plot_bins["family"].isin(["baseline", "quadratic", "huber"])]
    abs_ok = []
    if "attained" in matched.columns:
        abs_rec = matched.loc[(matched["family"] == "absolute") & (matched["attained"] == True)]
        abs_ok_s = [
            float(r["target_s"])
            for _, r in abs_rec.iterrows()
            if _finite_num(r.get("R2_price")) and float(r["R2_price"]) >= 0.0
        ]
        if abs_ok_s:
            axis_bins = pd.concat(
                [
                    axis_bins,
                    plot_bins.loc[
                        (plot_bins["family"] == "absolute")
                        & plot_bins["target_s"].apply(lambda s: any(np.isclose(float(s), t, atol=1e-12) for t in abs_ok_s))
                    ],
                ],
                ignore_index=True,
            )
    yvals = axis_bins["median_ratio"].to_numpy(dtype=float)
    yvals = yvals[np.isfinite(yvals)]
    ymin, ymax = float(np.min(yvals)), float(np.max(yvals))
    pad = 0.05 * max(ymax - ymin, 0.05)
    ylim = (ymin - pad, ymax + pad)
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), sharex=True, sharey=True)
    titles = {"quadratic": "Quadratic", "huber": "Huber", "absolute": "Absolute"}
    cmap = plt.cm.viridis
    x_all = plot_bins["median_sale_price"].to_numpy(dtype=float)
    xmin, xmax = float(np.min(x_all)), float(np.max(x_all))
    display_curve_s = [t for t in display_s if abs(float(t) - 1.0) > 1e-12]
    clip_rows: List[dict] = []
    for ax, fam in zip(axes, FAMILIES):
        baseb = plot_bins.loc[plot_bins["family"] == "baseline"].sort_values("bin")
        if not baseb.empty:
            ax.plot(
                baseb["median_sale_price"],
                baseb["median_ratio"],
                color=cmap(0.08),
                lw=1.6,
                marker="o",
                ms=2.0,
                label=r"$s=1$",
            )
        sub = plot_bins.loc[plot_bins["family"] == fam]
        for i, sstar in enumerate(display_curve_s):
            part = sub.loc[np.isclose(sub["target_s"].astype(float), float(sstar), atol=1e-12)].sort_values("bin")
            if part.empty:
                continue
            y = part["median_ratio"].to_numpy(dtype=float)
            y_plot = np.clip(y, ylim[0], ylim[1])
            clipped = ~np.isclose(y, y_plot, rtol=0.0, atol=1e-12)
            ax.plot(
                part["median_sale_price"],
                y_plot,
                color=cmap(0.18 + 0.75 * i / max(len(display_curve_s) - 1, 1)),
                lw=1.5,
                marker="o",
                ms=2.0,
                label=rf"$s={sstar:.2f}$",
            )
            if np.any(clipped):
                ax.scatter(part["median_sale_price"].to_numpy()[clipped], y_plot[clipped], marker="^", s=18, color="#B91C1C", zorder=5)
                clip_rows.append({"figure": "ratio", "family": fam, "target_s": float(sstar), "n_clipped": int(np.sum(clipped))})
        ax.axhline(1.0, color="#111827", ls=(0, (2, 2)), lw=0.8)
        ax.set_xscale("log")
        ax.set_ylim(*ylim)
        ax.set_xlim(xmin / 1.05, xmax * 1.05)
        ax.set_title(titles[fam])
        ax.grid(True, color="#E5E7EB", lw=0.7)
        ax.set_axisbelow(True)
        ax.set_xlabel("Sale price")
        if fam == "quadratic":
            ax.set_ylabel("Valuation-to-sale ratio")
        if fam == "absolute":
            ax.legend(fontsize=6.5, frameon=False, loc="best", ncol=2)
    fig.suptitle("EXPERIMENTAL / TOY matched signed-retention ratio shapes (held-out)", fontsize=11)
    fig.tight_layout()
    out = root / "figures" / "matched_ratio_shape"
    v1._ensure_dir(out.parent)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    _record_clips(root, clip_rows)
    return out.with_suffix(".pdf")


def _in_s_window(df: pd.DataFrame) -> pd.DataFrame:
    sub = df.loc[df["s_train"].apply(_finite_num)].copy()
    if "status" in sub.columns:
        sub = sub.loc[sub["status"].astype(str) != "numerical_failure"]
    sub = sub.loc[sub["s_train"].astype(float).between(S_PLOT_LO, S_PLOT_HI)]
    sub = sub.loc[sub["Beta_log"].apply(_finite_num)]
    sub = sub.loc[sub["dCor_e_y"].apply(_finite_num)]
    sub = sub.loc[sub["R2_price"].apply(_finite_num)]
    return sub.sort_values(["family", "s_train"], ascending=[True, False])


def _axis_source(window: pd.DataFrame) -> pd.DataFrame:
    src = window.copy()
    if "status" in src.columns:
        src = src.loc[src["status"].astype(str) == "ok"]
    good = src.loc[src["R2_price"].astype(float) >= 0.0] if "R2_price" in src.columns else src
    return good if not good.empty else src


def _padded_lim(series) -> Tuple[float, float]:
    v = np.asarray(series, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (-1.0, 1.0)
    lo, hi = float(np.min(v)), float(np.max(v))
    pad = 0.05 * max(hi - lo, 1e-3)
    return lo - pad, hi + pad


def _record_clips(root: Path, rows: List[dict]) -> None:
    dest = root / "plot_clipped_points.csv"
    extra = pd.DataFrame(rows)
    if extra.empty:
        return
    if dest.is_file():
        extra = pd.concat([pd.read_csv(dest), extra], ignore_index=True)
    extra.to_csv(dest, index=False)


def plot_mechanism(root: Path, matched: pd.DataFrame) -> Path:
    import matplotlib.pyplot as plt
    _set_style()
    fig, axes = plt.subplots(3, 3, figsize=(10.6, 7.6), sharex=True)
    work = matched.loc[matched.get("attained", True) == True].copy() if "attained" in matched.columns else matched.copy()
    work = work.loc[work["achieved_s_train"].apply(_finite_num)]
    work = work.loc[work["achieved_s_train"].astype(float) > 0.0]
    axis = work.loc[work["family"].isin(["quadratic", "huber"])]
    if "R2_price" in axis.columns:
        axis = axis.loc[axis["R2_price"].apply(_finite_num) & (axis["R2_price"].astype(float) >= 0.0)]
    beta_lim = _padded_lim(axis["Beta_log"])
    dcor_lim = _padded_lim(axis["dCor_e_y"])
    nl_lim = _padded_lim(axis["NL_shape"]) if "NL_shape" in axis.columns else (0.0, 0.05)
    titles = {"quadratic": "Quadratic", "huber": "Huber", "absolute": "Absolute"}
    clip_notes: List[dict] = []
    panels = (
        ("Beta_log", r"held-out $\beta_{\log}$", beta_lim),
        ("dCor_e_y", r"held-out $\mathrm{dCor}(e,y)$", dcor_lim),
        ("NL_shape", r"$\mathrm{NL}_{\mathrm{shape}}$", nl_lim),
    )
    for c, fam in enumerate(FAMILIES):
        sub = work.loc[work["family"] == fam].sort_values("achieved_s_train", ascending=False)
        for r, (col, ylab, ylim) in enumerate(panels):
            ax = axes[r, c]
            if sub.empty or col not in sub.columns:
                ax.set_title(titles[fam] if r == 0 else "")
                continue
            y = sub[col].to_numpy(dtype=float)
            x = sub["achieved_s_train"].to_numpy(dtype=float)
            finite = np.isfinite(y) & np.isfinite(x)
            x, y = x[finite], y[finite]
            y_plot = np.clip(y, ylim[0], ylim[1])
            clipped = ~np.isclose(y, y_plot, rtol=0.0, atol=1e-12)
            ax.plot(x, y_plot, color="#1D4ED8", marker="o", ms=4, lw=1.4)
            if np.any(clipped):
                ax.scatter(x[clipped], y_plot[clipped], marker="^", s=28, color="#B91C1C", zorder=5)
                for i in np.flatnonzero(clipped):
                    clip_notes.append(
                        {
                            "figure": "mechanism",
                            "family": fam,
                            "metric": col,
                            "s_train": float(x[i]),
                            "exact_value": float(y[i]),
                            "displayed_value": float(y_plot[i]),
                        }
                    )
            ax.set_xlim(1.05, -0.02)
            ax.set_ylim(*ylim)
            ax.grid(True, color="#E5E7EB", lw=0.7)
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(titles[fam])
                ax.axhline(0.0, color="#111827", lw=0.8, ls=":")
            if c == 0:
                ax.set_ylabel(ylab)
            if r == 2:
                ax.set_xlabel(r"Training signed retention $s$  ($1\rightarrow 0$)")
    fig.suptitle("EXPERIMENTAL / TOY matched mechanism vs training signed retention", fontsize=11)
    fig.tight_layout()
    out = root / "figures" / "mechanism_vs_s"
    v1._ensure_dir(out.parent)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    _record_clips(root, clip_notes)
    return out.with_suffix(".pdf")


def plot_tradeoff(root: Path, matched: pd.DataFrame) -> Path:
    import matplotlib.pyplot as plt
    _set_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.5))
    work = matched.loc[matched.get("attained", True) == True].copy() if "attained" in matched.columns else matched.copy()
    work = work.loc[work["achieved_s_train"].apply(_finite_num)]
    work = work.loc[work["achieved_s_train"].astype(float) > 0.0]
    axis = work.loc[work["family"].isin(["quadratic", "huber"])]
    if "R2_price" in axis.columns:
        axis = axis.loc[axis["R2_price"].apply(_finite_num) & (axis["R2_price"].astype(float) >= 0.0)]
    colors = {"quadratic": "#1D4ED8", "huber": "#047857", "absolute": "#B45309"}
    r2_lim = _padded_lim(axis["R2_price"])
    dcor_lim = _padded_lim(axis["dCor_e_y"])
    nl_lim = _padded_lim(axis["NL_shape"]) if "NL_shape" in axis.columns else (0.0, 0.05)
    beta_lim = _padded_lim(axis["Beta_log"])
    panels = (
        (axes[0], "R2_price", r"held-out $R^2_P$", r2_lim),
        (axes[1], "dCor_e_y", r"held-out $\mathrm{dCor}(e,y)$", dcor_lim),
        (axes[2], "NL_shape", r"$\mathrm{NL}_{\mathrm{shape}}$", nl_lim),
    )
    clip_rows: List[dict] = []
    for ax, col, ylab, ylim in panels:
        for fam in FAMILIES:
            sub = work.loc[work["family"] == fam].sort_values("achieved_s_train", ascending=False)
            if sub.empty or col not in sub.columns:
                continue
            x = sub["Beta_log"].to_numpy(dtype=float)
            y = sub[col].to_numpy(dtype=float)
            keep = np.isfinite(x) & np.isfinite(y)
            x, y = x[keep], y[keep]
            y_plot = np.clip(y, ylim[0], ylim[1])
            x_plot = np.clip(x, beta_lim[0], beta_lim[1])
            clipped = (~np.isclose(y, y_plot, rtol=0.0, atol=1e-12)) | (~np.isclose(x, x_plot, rtol=0.0, atol=1e-12))
            ax.plot(x_plot, y_plot, color=colors[fam], marker="o", ms=4, lw=1.3, label=fam)
            if np.any(clipped):
                ax.scatter(x_plot[clipped], y_plot[clipped], marker="^", s=28, color="#B91C1C", zorder=5)
                for i in np.flatnonzero(clipped):
                    clip_rows.append(
                        {
                            "figure": "tradeoff",
                            "family": fam,
                            "metric": col,
                            "s_train": float(sub["achieved_s_train"].iloc[i]),
                            "exact_x": float(x[i]),
                            "exact_y": float(y[i]),
                            "displayed_x": float(x_plot[i]),
                            "displayed_y": float(y_plot[i]),
                        }
                    )
        ax.set_xlabel(r"held-out $\beta_{\log}$")
        ax.set_ylabel(ylab)
        ax.set_xlim(*beta_lim)
        ax.set_ylim(*ylim)
        ax.grid(True, color="#E5E7EB", lw=0.7)
        ax.set_axisbelow(True)
    axes[2].legend(frameon=False, fontsize=8)
    fig.suptitle("EXPERIMENTAL / TOY matched tradeoffs", fontsize=10)
    fig.tight_layout()
    out = root / "figures" / "tradeoff_vs_beta"
    v1._ensure_dir(out.parent)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    _record_clips(root, clip_rows)
    return out.with_suffix(".pdf")

def render_matched_table(root: Path, matched: pd.DataFrame) -> Path:
    import matplotlib.pyplot as plt
    _set_style()
    cols = [
        ("family", "family"),
        ("target_s", r"$s^\star$"),
        ("achieved_s_train", r"$s$"),
        ("raw_rho", r"$\rho$"),
        ("p_linear", r"$p_{\mathrm{lin}}$"),
        ("R2_price", r"$R^2_P$"),
        ("MAE_price", "MAE"),
        ("PRD", "PRD"),
        ("PRB", "PRB"),
        ("MKI", "MKI"),
        ("VEI", "VEI"),
        ("Beta_log", r"$\beta_{\log}$"),
        ("dCor_e_y", "dCor"),
        ("NL_shape", r"$\mathrm{NL}$"),
    ]
    show = matched.copy()
    cell = []
    for _, rec in show.iterrows():
        if not bool(rec.get("attained", False)):
            cell.append(
                [str(rec.get("family", "")), f"{float(rec['target_s']):.2f}", "not attained"]
                + [""] * (len(cols) - 3)
            )
            continue
        row = []
        for key, _lab in cols:
            val = rec.get(key, np.nan)
            if key == "family":
                row.append(str(val))
            elif not _finite_num(val):
                row.append("—")
            elif key in {"target_s", "achieved_s_train", "p_linear", "R2_price", "PRD", "dCor_e_y", "NL_shape"}:
                row.append(f"{float(val):.3f}")
            elif key in {"PRB", "MKI", "VEI", "Beta_log"}:
                row.append(f"{float(val):.4f}")
            elif key == "MAE_price":
                row.append(f"{float(val):,.0f}")
            else:
                row.append(f"{float(val):.4g}")
        cell.append(row)
    fig, ax = plt.subplots(figsize=(11.6, 0.45 + 0.32 * max(len(cell), 1)))
    ax.axis("off")
    table = ax.table(cellText=cell, colLabels=[lab for _k, lab in cols], loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.25)
    ax.set_title("EXPERIMENTAL / TOY matched signed-retention comparison (training $s$ only)", fontsize=10, pad=8)
    fig.tight_layout()
    out = root / "figures" / "matched_correction_table"
    v1._ensure_dir(out.parent)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".pdf")


def assemble(args) -> None:
    root = Path(args.output_root)
    clip_path = root / "plot_clipped_points.csv"
    if clip_path.is_file():
        clip_path.unlink()
    metrics, status = _collect(root)
    metrics.to_csv(root / "metrics.csv", index=False)
    status.to_csv(root / "fit_status.csv", index=False)
    prev_path = root / "matched_correction_metrics.csv"
    backup = root / "matched_correction_metrics_pre_extend.csv"
    previous = None
    if prev_path.is_file():
        previous = pd.read_csv(prev_path)
        if not backup.is_file():
            previous.to_csv(backup, index=False)
    matched = select_matched(metrics, status, previous=previous)
    matched = attach_nl_shape(root, matched)
    matched.to_csv(root / "matched_correction_metrics.csv", index=False)
    dense_summary = []
    for fam in FAMILIES:
        sub = metrics.loc[metrics["family"] == fam]
        dense = sub.loc[sub["run_type"].isin(["dense", "shared_lambda0"])]
        st = status.loc[status["family"] == fam] if not status.empty and "family" in status.columns else pd.DataFrame()
        n_fail = int((st["status"].astype(str) == "numerical_failure").sum()) if not st.empty else 0
        n_path = int((st["status"].astype(str) == "pathological_finite").sum()) if not st.empty else 0
        s_ok = dense.loc[dense["s_train"].apply(_finite_num), "s_train"].astype(float)
        dense_summary.append(
            {
                "family": fam,
                "n_dense_rows": int(len(dense)),
                "n_numerical_failure": n_fail,
                "n_pathological_finite": n_path,
                "s_min": float(s_ok.min()) if not s_ok.empty else float("nan"),
                "s_max": float(s_ok.max()) if not s_ok.empty else float("nan"),
            }
        )
    pd.DataFrame(dense_summary).to_csv(root / "dense_path_summary.csv", index=False)
    ratio = plot_matched_ratio(root, matched, metrics)
    mech = plot_mechanism(root, matched)
    trade = plot_tradeoff(root, matched)
    table = render_matched_table(root, matched)
    cfg_path = root / "config.json"
    cfg = read_json(cfg_path) if cfg_path.is_file() else {}
    cfg["assembled_utc"] = pd.Timestamp.utcnow().isoformat()
    cfg["n_metric_rows"] = int(len(metrics))
    cfg["figure_paths"] = {
        "matched_ratio_shape": None if ratio is None else str(ratio),
        "mechanism": str(mech),
        "tradeoff": str(trade),
        "matched_table": str(table),
    }
    cfg["dense_path_summary"] = dense_summary
    cfg["plot_caption_note"] = (
        "Extreme configurations do not determine the displayed common axis range; "
        "exact values remain available in the machine-readable results."
    )
    cfg["plot_clip_log"] = str(clip_path) if clip_path.is_file() else None
    write_json(cfg_path, cfg)
    _log("assemble complete", metrics=str(root / "metrics.csv"), matched=str(root / "matched_correction_metrics.csv"))


def _parse_floats(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EXPERIMENTAL / TOY V2 quadratic/Huber/absolute comparison.")
    p.add_argument("--mode", required=True, choices=["calibrate", "fit-family", "extend-targets", "assemble"])
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    p.add_argument("--data-path", default=str(v1.DEFAULT_DATA))
    p.add_argument("--params", default=str(v1.DEFAULT_PARAMS))
    p.add_argument("--lgbm-config-json", default=str(v1.DEFAULT_LGBM_CONFIG))
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--lambdas", type=str, default=",".join(str(x) for x in LAMBDA_GRID))
    p.add_argument("--family", type=str, default=None)
    p.add_argument("--targets", type=str, default=None)
    p.add_argument("--skip-refine", action="store_true")
    p.add_argument("--refit-lambda0", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    args.lambdas = _parse_floats(args.lambdas)
    if args.targets:
        args.targets = _parse_floats(args.targets)
    os.environ.setdefault("MPLBACKEND", "Agg")
    _log("start", mode=args.mode, label=EXPERIMENT_LABEL)
    if args.mode == "calibrate":
        run_calibrate(args)
    elif args.mode == "fit-family":
        if not args.family:
            raise SystemExit("--family is required")
        run_fit_family(args)
    elif args.mode == "extend-targets":
        if not args.family:
            raise SystemExit("--family is required")
        if not args.targets:
            raise SystemExit("--targets is required")
        run_extend_targets(args)
    else:
        assemble(args)
    _log("done", mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
