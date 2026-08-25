#!/usr/bin/env python3
"""EXPERIMENTAL / TOY / NON-CANONICAL 3x2 surrogate-mechanism ablation.

Not a CV experiment and not a paper-method redesign.

Protocol (scientific run)
  * Frozen paper LightGBM hyperparameters from
    output/paper_v6_preselection_994/lgbm_config.json (994 trees).
  * Train each toy variant once on the full current development sample.
  * Evaluate on the same primary held-out sample.
  * No rolling-origin folds, no CV aggregation, no hyperparameter search,
    no early stopping, no 2025 forward sample.
  * Only surrogate type, level-invariant flag, and lambda/rho change.
  * Rho-scale calibration uses one shared custom-objective rho=0 fit on
    the full development sample.

``_load_and_split_data`` is imported only to reuse that development/held-out
split and preprocessing. This script never calls the CV stage.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from canonical_experiment import git_state, lgbm_params_hash, package_versions, read_json, write_json
from run_temporal_cv import _load_and_split_data, _load_lgbm_params_from_config_json
from soft_constrained_models.toy_surrogate_models import (
    CAP_QUANTILE,
    EXPERIMENT_LABEL,
    LAMBDA_GRID,
    RATIO_SHAPE_LAMBDAS,
    VARIANT_SPECS,
    ToyLGBSurrogate,
    capped_tau,
    fairness_penalty_gradient_unit_rho_from_ec,
    parse_variant_name,
    variant_name,
)
from utils.motivation_utils import compute_taxation_metrics

DEFAULT_OUTPUT = REPO / "output" / "toy_surrogate_ablation"
DEFAULT_LGBM_CONFIG = REPO / "output" / "paper_v6_preselection_994" / "lgbm_config.json"
DEFAULT_DATA = REPO / "data" / "CCAO" / "2025" / "training_data.parquet"
DEFAULT_PARAMS = REPO / "params.yaml"
PYTHON_PATH = "/home/nacevedo/.conda/envs/fairness_env/bin/python"

METRIC_KEYS = [
    "R2_price",
    "MAE_price",
    "MAPE",
    "RMSE_log",
    "Median ratio",
    "Mean ratio",
    "W. Mean ratio",
    "COD",
    "COV_IAAO",
    "PRD",
    "PRB",
    "MKI",
    "VEI",
    "Beta_log",
    "dCor_e_y",
]


def _log(message: str, **fields: Any) -> None:
    suffix = ""
    if fields:
        suffix = " | " + " | ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[toy_surrogate_ablation] {message}{suffix}", flush=True)


def _peak_rss_gb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (1024.0 ** 2)


def _allocated_cpus() -> int:
    env = os.environ.get("SLURM_CPUS_PER_TASK") or os.environ.get("SLURM_CPUS_ON_NODE")
    if env:
        try:
            return max(1, int(str(env).split()[0]))
        except Exception:
            pass
    return max(1, int(os.cpu_count() or 1))


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_params(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_frozen_lgbm_params(config_json: Path, n_jobs: int, n_estimators: Optional[int] = None) -> Dict[str, Any]:
    params = dict(_load_lgbm_params_from_config_json(config_json))
    params["n_jobs"] = int(n_jobs)
    params["verbosity"] = -1
    if n_estimators is not None:
        params["n_estimators"] = int(n_estimators)
    return params


def prepare_xy(
    df: pd.DataFrame,
    predictor_cols: Sequence[str],
    categorical_cols: Sequence[str],
    target_col: str,
) -> Tuple[pd.DataFrame, np.ndarray]:
    X = df[list(predictor_cols)].copy()
    for col in categorical_cols:
        if col in X.columns:
            X[col] = X[col].astype("category")
    y_log = np.log(df[target_col].to_numpy(dtype=float))
    return X, y_log


def equal_count_bins(sale: np.ndarray, ratio: np.ndarray, n_bins: int = 30) -> pd.DataFrame:
    """Current-paper construction: 30 equal-count sale-price bins, median ratio."""
    sale = np.asarray(sale, dtype=float).reshape(-1)
    ratio = np.asarray(ratio, dtype=float).reshape(-1)
    order = np.argsort(sale, kind="mergesort")
    sale, ratio = sale[order], ratio[order]
    rows = []
    for i, idx in enumerate(np.array_split(np.arange(len(sale)), n_bins), start=1):
        if idx.size == 0:
            continue
        r = ratio[idx]
        rows.append(
            {
                "bin": int(i),
                "n": int(idx.size),
                "median_sale_price": float(np.median(sale[idx])),
                "median_ratio": float(np.median(r)),
                "ratio_q25": float(np.quantile(r, 0.25)),
                "ratio_q75": float(np.quantile(r, 0.75)),
            }
        )
    return pd.DataFrame(rows)


def extract_required_metrics(raw: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    aliases = {
        "MAE_price": ("MAE_price", "MAE"),
        "COD": ("COD",),
        "COV_IAAO": ("COV_IAAO",),
        "Median ratio": ("Median ratio",),
        "Mean ratio": ("Mean ratio",),
        "W. Mean ratio": ("W. Mean ratio",),
    }
    for key in METRIC_KEYS:
        names = aliases.get(key, (key,))
        val = None
        for name in names:
            if name in raw and raw[name] is not None:
                val = raw[name]
                break
        out[key] = float(val) if val is not None and np.isfinite(float(val)) else float("nan")
    return out


def lambda0_pred_path(root: Path) -> Path:
    return root / "predictions" / "lambda0_shared.parquet"


def lambda0_train_resid_path(root: Path) -> Path:
    return root / "predictions" / "lambda0_train_residuals.parquet"


def calibration_path(root: Path) -> Path:
    return root / "rho_scale_calibration.csv"


def variant_metrics_path(root: Path, name: str) -> Path:
    return root / "variants" / name / "metrics.csv"


def variant_pred_path(root: Path, name: str, lam: float) -> Path:
    return root / "variants" / name / f"pred_lambda_{lam:g}.parquet"


def load_experiment_data(
    *,
    data_path: Path,
    params: dict,
    sample_frac: Optional[float],
    seed: int,
) -> Dict[str, Any]:
    # Development vs primary held-out only. The 2025 assessment frame is
    # returned by the helper and discarded; this experiment never uses folds.
    df_train, df_test, _df_assess, predictor_cols, categorical_cols = _load_and_split_data(
        data_path=str(data_path),
        params=params,
        target_column="meta_sale_price",
        date_column="meta_sale_date",
        assessment_year=2025,
        heldout_test_mode="pre_assessment_tail",
        sample_frac=sample_frac,
        sample_seed=int(seed),
        universe_start="2016-01-01",
        pre_assessment_end="2024-12-31",
    )
    X_train, y_train = prepare_xy(df_train, predictor_cols, categorical_cols, "meta_sale_price")
    X_eval, y_eval = prepare_xy(df_test, predictor_cols, categorical_cols, "meta_sale_price")
    return {
        "df_train": df_train,
        "df_eval": df_test,
        "predictor_cols": list(predictor_cols),
        "categorical_cols": list(categorical_cols),
        "X_train": X_train,
        "y_train": y_train,
        "X_eval": X_eval,
        "y_eval": y_eval,
        "n_train": int(len(df_train)),
        "n_eval": int(len(df_test)),
        "train_start": str(pd.to_datetime(df_train["meta_sale_date"]).min()),
        "train_end": str(pd.to_datetime(df_train["meta_sale_date"]).max()),
        "eval_start": str(pd.to_datetime(df_test["meta_sale_date"]).min()),
        "eval_end": str(pd.to_datetime(df_test["meta_sale_date"]).max()),
        "n_features": int(len(predictor_cols)),
        "n_categorical": int(len(categorical_cols)),
    }


def fit_toy(
    *,
    X_train,
    y_train,
    X_eval,
    y_eval,
    df_eval: pd.DataFrame,
    lgbm_params: dict,
    penalty_shape: str,
    level_invariant: bool,
    rho: float,
    verbose: bool = False,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    rss0 = _peak_rss_gb()
    model = ToyLGBSurrogate(
        rho=float(rho),
        penalty_shape=penalty_shape,
        level_invariant=level_invariant,
        lgbm_params=dict(lgbm_params),
        verbose=verbose,
        match_native_init=True,
        early_stopping_rounds=None,
    )
    model.fit(X_train, y_train)
    pred_eval = np.asarray(model.predict(X_eval), dtype=float).reshape(-1)
    pred_train = np.asarray(model.predict(X_train), dtype=float).reshape(-1)
    runtime = float(time.perf_counter() - t0)
    metrics_raw = compute_taxation_metrics(y_eval, pred_eval, scale="log", y_train=y_train)
    metrics = extract_required_metrics(metrics_raw)
    c_train = y_train - float(np.mean(y_train))
    tau = float("nan")
    if penalty_shape == "capped_quadratic":
        tau = float(model.tau_)
        if not np.isfinite(tau):
            tau = capped_tau(c_train)
    pred_df = pd.DataFrame(
        {
            "row_id": df_eval.index.to_numpy(),
            "sale_date": pd.to_datetime(df_eval["meta_sale_date"]).to_numpy(),
            "y_true_log": y_eval,
            "y_pred_log": pred_eval,
            "y_true": np.exp(y_eval),
            "y_pred": np.exp(pred_eval),
        }
    )
    return {
        "model": model,
        "metrics": metrics,
        "pred_eval": pred_df,
        "pred_train_log": pred_train,
        "runtime_sec": runtime,
        "peak_rss_gb": max(_peak_rss_gb(), rss0),
        "best_iteration": model.best_iteration(),
        "tau": tau,
        "n_estimators": int(lgbm_params.get("n_estimators", -1)),
    }


def _g_stats(g_pen: np.ndarray) -> Dict[str, float]:
    abs_g = np.abs(g_pen)
    return {
        "G_m": float(np.sqrt(np.mean(g_pen ** 2))),
        "abs_g_rms": float(np.sqrt(np.mean(abs_g ** 2))),
        "abs_g_median": float(np.median(abs_g)),
        "abs_g_p95": float(np.quantile(abs_g, 0.95)),
    }


def compute_calibration(e0: np.ndarray, c: np.ndarray, lambdas: Sequence[float]) -> pd.DataFrame:
    rows = []
    g_ref = None
    for shape, li in VARIANT_SPECS:
        g_pen = fairness_penalty_gradient_unit_rho_from_ec(
            e0, c, penalty_shape=shape, level_invariant=li
        )
        stats = _g_stats(g_pen)
        if shape == "quadratic" and not li:
            g_ref = stats["G_m"]
        tau = capped_tau(c) if shape == "capped_quadratic" else float("nan")
        rows.append(
            {
                "surrogate_type": shape,
                "level_invariant": bool(li),
                "variant": variant_name(shape, li),
                **stats,
                "tau": tau,
                "cap_quantile": CAP_QUANTILE if shape == "capped_quadratic" else float("nan"),
            }
        )
    if g_ref is None or not np.isfinite(g_ref) or g_ref <= 0.0:
        raise RuntimeError("Reference G_m for quadratic/fixed is not positive.")
    out = []
    for row in rows:
        multiplier = float(g_ref / row["G_m"]) if row["G_m"] > 0 else float("nan")
        for lam in lambdas:
            rec = dict(row)
            rec["G_ref"] = float(g_ref)
            rec["multiplier_Gref_over_Gm"] = multiplier
            rec["lambda"] = float(lam)
            rec["raw_rho"] = float(lam) * multiplier
            rec["note"] = (
                "Normalization puts initial fairness-gradient RMS on a common scale; "
                "not exact objective equivalence."
            )
            out.append(rec)
    return pd.DataFrame(out)


def lookup_raw_rho(calib: pd.DataFrame, shape: str, li: bool, lam: float) -> float:
    sub = calib.loc[
        (calib["surrogate_type"] == shape)
        & (calib["level_invariant"] == bool(li))
        & np.isclose(calib["lambda"].astype(float), float(lam), rtol=0.0, atol=1e-12)
    ]
    if sub.empty:
        raise KeyError(f"Missing calibration row for {shape} li={li} lambda={lam}")
    return float(sub["raw_rho"].iloc[0])


def lookup_tau(calib: pd.DataFrame, shape: str) -> float:
    if shape != "capped_quadratic":
        return float("nan")
    sub = calib.loc[calib["surrogate_type"] == "capped_quadratic"]
    return float(sub["tau"].iloc[0])


def metrics_row(
    *,
    shape: str,
    li: bool,
    lam: float,
    raw_rho: float,
    tau: float,
    fit: Dict[str, Any],
    n_train: int,
    n_eval: int,
    shared_lambda0: bool,
) -> Dict[str, Any]:
    return {
        "experiment_label": EXPERIMENT_LABEL,
        "surrogate_type": shape,
        "level_invariant": bool(li),
        "variant": variant_name(shape, li),
        "lambda": float(lam),
        "raw_rho": float(raw_rho),
        "cap_tau": float(tau) if np.isfinite(tau) else float("nan"),
        "cap_quantile": CAP_QUANTILE if shape == "capped_quadratic" else float("nan"),
        "n_train": int(n_train),
        "n_eval": int(n_eval),
        "runtime_sec": float(fit["runtime_sec"]),
        "peak_rss_gb": float(fit["peak_rss_gb"]),
        "best_iteration": fit["best_iteration"] if fit["best_iteration"] is not None else float("nan"),
        "n_estimators": int(fit["n_estimators"]),
        "shared_lambda0_fit": bool(shared_lambda0),
        **fit["metrics"],
    }


def save_pred(path: Path, pred: pd.DataFrame, meta: Dict[str, Any]) -> None:
    _ensure_dir(path.parent)
    out = pred.copy()
    for k, v in meta.items():
        out[k] = v
    out.to_parquet(path, index=False)


def run_calibrate_or_probe(args: argparse.Namespace, *, probe: bool) -> Dict[str, Any]:
    root = _ensure_dir(Path(args.output_root))
    params = load_params(Path(args.params))
    n_jobs = _allocated_cpus()
    lgbm_params = load_frozen_lgbm_params(Path(args.lgbm_config_json), n_jobs=n_jobs, n_estimators=args.n_estimators)
    data = load_experiment_data(
        data_path=Path(args.data_path),
        params=params,
        sample_frac=args.sample_frac,
        seed=int(args.seed),
    )
    _log(
        "data ready (development -> held-out; no CV folds)",
        n_train=data["n_train"],
        n_eval=data["n_eval"],
        n_features=data["n_features"],
        n_estimators=int(lgbm_params["n_estimators"]),
        mode=("probe" if probe else "calibrate"),
        n_jobs=n_jobs,
    )
    fit = fit_toy(
        X_train=data["X_train"],
        y_train=data["y_train"],
        X_eval=data["X_eval"],
        y_eval=data["y_eval"],
        df_eval=data["df_eval"],
        lgbm_params=lgbm_params,
        penalty_shape="quadratic",
        level_invariant=False,
        rho=0.0,
        verbose=bool(args.verbose),
    )
    _log(
        "lambda0 fit done",
        runtime_sec=f"{fit['runtime_sec']:.1f}",
        peak_rss_gb=f"{fit['peak_rss_gb']:.2f}",
        n_jobs=n_jobs,
    )
    e0 = fit["pred_train_log"] - data["y_train"]
    c = data["y_train"] - float(np.mean(data["y_train"]))
    calib = compute_calibration(e0, c, lambdas=list(args.lambdas))
    calib.to_csv(calibration_path(root), index=False)
    save_pred(
        lambda0_pred_path(root),
        fit["pred_eval"],
        {
            "surrogate_type": "shared_lambda0",
            "level_invariant": False,
            "lambda": 0.0,
            "raw_rho": 0.0,
            "experiment_label": EXPERIMENT_LABEL,
        },
    )
    pd.DataFrame({"e0": e0, "c": c, "y_train_log": data["y_train"], "y_pred_train_log": fit["pred_train_log"]}).to_parquet(
        lambda0_train_resid_path(root), index=False
    )
    timing = {
        "mode": "probe" if probe else "calibrate",
        "runtime_sec": fit["runtime_sec"],
        "peak_rss_gb": fit["peak_rss_gb"],
        "n_jobs": n_jobs,
        "n_train": data["n_train"],
        "n_eval": data["n_eval"],
        "n_estimators": int(lgbm_params["n_estimators"]),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "recommended_cpus": int(n_jobs),
        "recommended_mem_gb": int(max(48, np.ceil(fit["peak_rss_gb"] * 1.8) + 16)),
        "recommended_time_hours_per_variant": float(max(4.0, (fit["runtime_sec"] * 6.5) / 3600.0 * 1.6)),
    }
    write_json(root / ("probe_timing.json" if probe else "calibrate_timing.json"), timing)
    write_partial_config(root, args, data, lgbm_params, calib, extra={"timing": timing})
    # Shared lambda=0 metrics for every variant label (same fit).
    rows = []
    tau_cap = lookup_tau(calib, "capped_quadratic")
    for shape, li in VARIANT_SPECS:
        tau = tau_cap if shape == "capped_quadratic" else float("nan")
        row = metrics_row(
            shape=shape,
            li=li,
            lam=0.0,
            raw_rho=0.0,
            tau=tau,
            fit={**fit, "runtime_sec": fit["runtime_sec"] if (shape == "quadratic" and not li) else 0.0},
            n_train=data["n_train"],
            n_eval=data["n_eval"],
            shared_lambda0=True,
        )
        rows.append(row)
        vdir = _ensure_dir(root / "variants" / variant_name(shape, li))
        pd.DataFrame([row]).to_csv(vdir / "metrics_lambda0.csv", index=False)
        save_pred(
            variant_pred_path(root, variant_name(shape, li), 0.0),
            fit["pred_eval"],
            {
                "surrogate_type": shape,
                "level_invariant": bool(li),
                "lambda": 0.0,
                "raw_rho": 0.0,
                "shared_lambda0_fit": True,
            },
        )
    pd.DataFrame(rows).to_csv(root / "metrics_lambda0.csv", index=False)
    _log("calibration written", path=str(calibration_path(root)))
    return timing


def run_fit_variant(args: argparse.Namespace) -> Path:
    root = Path(args.output_root)
    shape, li = parse_variant_name(args.variant)
    name = variant_name(shape, li)
    calib = pd.read_csv(calibration_path(root))
    params = load_params(Path(args.params))
    n_jobs = _allocated_cpus()
    lgbm_params = load_frozen_lgbm_params(Path(args.lgbm_config_json), n_jobs=n_jobs, n_estimators=args.n_estimators)
    data = load_experiment_data(
        data_path=Path(args.data_path),
        params=params,
        sample_frac=args.sample_frac,
        seed=int(args.seed),
    )
    lambdas = [float(x) for x in args.lambdas if float(x) > 0.0]
    if args.lambda_chunk:
        chunk_i, n_chunks = [int(x) for x in str(args.lambda_chunk).split("/")]
        lambdas = [lam for i, lam in enumerate(lambdas) if i % n_chunks == chunk_i]
    tau = lookup_tau(calib, shape)
    rows: List[Dict[str, Any]] = []
    lam0_csv = root / "variants" / name / "metrics_lambda0.csv"
    if lam0_csv.is_file() and not args.skip_lambda0_metrics:
        rows.extend(pd.read_csv(lam0_csv).to_dict(orient="records"))
    _log(
        "variant start (development -> held-out; frozen trees; no CV)",
        variant=name,
        n_lambdas=len(lambdas),
        n_jobs=n_jobs,
        n_train=data["n_train"],
        n_estimators=int(lgbm_params["n_estimators"]),
    )
    for lam in lambdas:
        raw_rho = lookup_raw_rho(calib, shape, li, lam)
        _log("fit start", variant=name, lam=lam, raw_rho=f"{raw_rho:.6g}")
        fit = fit_toy(
            X_train=data["X_train"],
            y_train=data["y_train"],
            X_eval=data["X_eval"],
            y_eval=data["y_eval"],
            df_eval=data["df_eval"],
            lgbm_params=lgbm_params,
            penalty_shape=shape,
            level_invariant=li,
            rho=raw_rho,
            verbose=bool(args.verbose),
        )
        row = metrics_row(
            shape=shape,
            li=li,
            lam=lam,
            raw_rho=raw_rho,
            tau=tau if shape == "capped_quadratic" else (fit["tau"] if np.isfinite(fit["tau"]) else float("nan")),
            fit=fit,
            n_train=data["n_train"],
            n_eval=data["n_eval"],
            shared_lambda0=False,
        )
        rows.append(row)
        save_pred(
            variant_pred_path(root, name, lam),
            fit["pred_eval"],
            {
                "surrogate_type": shape,
                "level_invariant": bool(li),
                "lambda": float(lam),
                "raw_rho": float(raw_rho),
                "shared_lambda0_fit": False,
            },
        )
        _log("fit done", variant=name, lam=lam, runtime_sec=f"{fit['runtime_sec']:.1f}", peak_rss_gb=f"{fit['peak_rss_gb']:.2f}")
    out = pd.DataFrame(rows).sort_values(["lambda"])
    dest = variant_metrics_path(root, name)
    _ensure_dir(dest.parent)
    out.to_csv(dest, index=False)
    _log("variant written", path=str(dest), n_rows=int(len(out)))
    return dest


def run_smoke(args: argparse.Namespace) -> None:
    root = _ensure_dir(Path(args.output_root) / "smoke")
    args = argparse.Namespace(**vars(args))
    args.output_root = str(root)
    if args.sample_frac is None:
        args.sample_frac = 0.03
    if args.n_estimators is None:
        args.n_estimators = 30
    args.lambdas = [0.0, 1.0]
    _log("SMOKE start", sample_frac=args.sample_frac, n_estimators=args.n_estimators)
    run_calibrate_or_probe(args, probe=False)
    for shape, li in VARIANT_SPECS:
        args.variant = variant_name(shape, li)
        args.skip_lambda0_metrics = False
        args.lambda_chunk = None
        run_fit_variant(args)
    args.output_root = str(root)
    assemble_outputs(args)
    _log("SMOKE complete (not a scientific output)", root=str(root))


def _collect_metrics(root: Path) -> pd.DataFrame:
    frames = []
    for shape, li in VARIANT_SPECS:
        path = variant_metrics_path(root, variant_name(shape, li))
        if path.is_file():
            frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError("No variant metrics found; cannot assemble.")
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["variant", "lambda"], keep="last")
    return out.sort_values(["surrogate_type", "level_invariant", "lambda"]).reset_index(drop=True)


def _collect_ratio_bins(root: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for shape, li in VARIANT_SPECS:
        name = variant_name(shape, li)
        for lam in RATIO_SHAPE_LAMBDAS:
            path = variant_pred_path(root, name, lam)
            if not path.is_file():
                continue
            pred = pd.read_parquet(path)
            sale = pred["y_true"].to_numpy(dtype=float)
            ratio = pred["y_pred"].to_numpy(dtype=float) / sale
            bins = equal_count_bins(sale, ratio, n_bins=30)
            bins["surrogate_type"] = shape
            bins["level_invariant"] = bool(li)
            bins["variant"] = name
            bins["lambda"] = float(lam)
            raw = metrics.loc[
                (metrics["variant"] == name) & np.isclose(metrics["lambda"], float(lam), atol=1e-12),
                "raw_rho",
            ]
            bins["raw_rho"] = float(raw.iloc[0]) if len(raw) else np.nan
            rows.append(bins)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _set_style() -> None:
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.0,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _lambda_x(values) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    return np.where(x <= 0.0, 0.07, x)


def plot_ratio_shape(bins: pd.DataFrame, out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    _set_style()
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.6), sharex=True, sharey=True)
    shapes = ("quadratic", "absolute", "capped_quadratic")
    titles = ("Quadratic", "Absolute", "Capped quadratic")
    row_labs = ("Fixed-level", "Level-invariant")
    cmap = plt.cm.viridis
    y_all = bins["median_ratio"].to_numpy(dtype=float)
    y_all = y_all[np.isfinite(y_all)]
    ymin, ymax = float(np.min(y_all)), float(np.max(y_all))
    pad = 0.04 * max(ymax - ymin, 0.05)
    ylim = (ymin - pad, ymax + pad)
    x_all = bins["median_sale_price"].to_numpy(dtype=float)
    xmin, xmax = float(np.min(x_all)), float(np.max(x_all))
    lams = list(RATIO_SHAPE_LAMBDAS)
    for c, shape in enumerate(shapes):
        for r, li in enumerate((False, True)):
            ax = axes[r, c]
            sub = bins.loc[(bins["surrogate_type"] == shape) & (bins["level_invariant"] == li)]
            for i, lam in enumerate(lams):
                part = sub.loc[np.isclose(sub["lambda"].astype(float), float(lam), atol=1e-12)].sort_values("bin")
                if part.empty:
                    continue
                color = cmap(0.12 + 0.8 * i / max(len(lams) - 1, 1))
                ax.plot(
                    part["median_sale_price"],
                    part["median_ratio"],
                    color=color,
                    lw=1.5,
                    marker="o",
                    ms=2.2,
                    label=rf"$\lambda$={lam:g}",
                )
            ax.axhline(1.0, color="#111827", ls=(0, (2, 2)), lw=0.8)
            ax.set_xscale("log", base=10)
            ax.set_ylim(*ylim)
            ax.set_xlim(xmin / 1.05, xmax * 1.05)
            ax.grid(True, color="#E5E7EB", lw=0.7)
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(titles[c])
            if c == 0:
                ax.set_ylabel(f"{row_labs[r]}\nValuation-to-sale ratio")
            if r == 1:
                ax.set_xlabel("Sale price")
            if r == 0 and c == 2:
                ax.legend(fontsize=7, frameon=False, loc="best")
    fig.suptitle("EXPERIMENTAL / TOY ratio-shape ablation (held-out)", fontsize=11)
    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path.with_suffix(".pdf")


def plot_mechanism(metrics: pd.DataFrame, out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    _set_style()
    fig, axes = plt.subplots(2, 3, figsize=(10.4, 6.4), sharex=True)
    shapes = ("quadratic", "absolute", "capped_quadratic")
    titles = ("Quadratic", "Absolute", "Capped quadratic")
    metric_rows = (("Beta_log", r"$\beta_{\log}$"), ("dCor_e_y", r"$\mathrm{dCor}(e,y)$"))
    styles = {False: ("-", "o", "#1D4ED8", "Fixed-level"), True: ("--", "s", "#B45309", "Level-invariant")}
    for c, shape in enumerate(shapes):
        for r, (col, ylab) in enumerate(metric_rows):
            ax = axes[r, c]
            for li, (ls, mk, color, lab) in styles.items():
                sub = metrics.loc[
                    (metrics["surrogate_type"] == shape) & (metrics["level_invariant"] == li)
                ].sort_values("lambda")
                ax.plot(
                    _lambda_x(sub["lambda"]),
                    sub[col],
                    color=color,
                    ls=ls,
                    marker=mk,
                    ms=4.0,
                    lw=1.5,
                    label=lab,
                )
            ax.set_xscale("log")
            ax.set_xticks([0.07, 0.3, 1, 3, 10, 30, 100])
            ax.set_xticklabels(["0", "0.3", "1", "3", "10", "30", "100"])
            if col == "Beta_log":
                ax.axhline(0.0, color="#111827", lw=0.8, ls=":")
            ax.grid(True, color="#E5E7EB", lw=0.7)
            ax.set_axisbelow(True)
            if r == 0:
                ax.set_title(titles[c])
            if c == 0:
                ax.set_ylabel(ylab)
            if r == 1:
                ax.set_xlabel(r"Normalized strength $\lambda$")
            if r == 0 and c == 2:
                ax.legend(frameon=False, fontsize=8)
    fig.suptitle("EXPERIMENTAL / TOY mechanism paths (held-out)", fontsize=11)
    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path.with_suffix(".pdf")


def plot_compact_metrics(metrics: pd.DataFrame, out_path: Path) -> Path:
    import matplotlib.pyplot as plt

    _set_style()
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.5), sharex=True)
    panels = (
        ("R2_price", r"$R^2_P$"),
        ("PRD", "PRD"),
        ("PRB", "PRB"),
        ("VEI", "VEI"),
        ("MKI", "MKI"),
        ("MAE_price", "MAE"),
    )
    shapes = ("quadratic", "absolute", "capped_quadratic")
    colors = {"quadratic": "#1D4ED8", "absolute": "#B45309", "capped_quadratic": "#047857"}
    for ax, (col, ylab) in zip(axes.ravel(), panels):
        for shape in shapes:
            for li, ls, mk in ((False, "-", "o"), (True, "--", "s")):
                sub = metrics.loc[
                    (metrics["surrogate_type"] == shape) & (metrics["level_invariant"] == li)
                ].sort_values("lambda")
                lab = f"{shape}{' LI' if li else ''}"
                ax.plot(
                    _lambda_x(sub["lambda"]),
                    sub[col],
                    color=colors[shape],
                    ls=ls,
                    marker=mk,
                    ms=3.5,
                    lw=1.35,
                    label=lab,
                )
        ax.set_xscale("log")
        ax.set_xticks([0.07, 1, 10, 100])
        ax.set_xticklabels(["0", "1", "10", "100"])
        ax.set_ylabel(ylab)
        ax.grid(True, color="#E5E7EB", lw=0.7)
        ax.set_axisbelow(True)
    axes[1, 0].set_xlabel(r"$\lambda$")
    axes[1, 1].set_xlabel(r"$\lambda$")
    axes[1, 2].set_xlabel(r"$\lambda$")
    axes[0, 2].legend(frameon=False, fontsize=6.5, loc="best")
    fig.suptitle("EXPERIMENTAL / TOY compact metric paths (held-out)", fontsize=11)
    fig.tight_layout()
    _ensure_dir(out_path.parent)
    fig.savefig(out_path.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out_path.with_suffix(".pdf")


def write_partial_config(
    root: Path,
    args: argparse.Namespace,
    data: Dict[str, Any],
    lgbm_params: dict,
    calib: pd.DataFrame,
    extra: Optional[dict] = None,
) -> Path:
    git = git_state(REPO)
    rho_map = {}
    for shape, li in VARIANT_SPECS:
        name = variant_name(shape, li)
        sub = calib.loc[(calib["surrogate_type"] == shape) & (calib["level_invariant"] == bool(li))]
        rho_map[name] = {
            "G_m": float(sub["G_m"].iloc[0]),
            "multiplier_Gref_over_Gm": float(sub["multiplier_Gref_over_Gm"].iloc[0]),
            "raw_rho_by_lambda": {str(float(r["lambda"])): float(r["raw_rho"]) for r in sub.to_dict(orient="records")},
        }
    cap_rows = calib.loc[calib["surrogate_type"] == "capped_quadratic"]
    payload = {
        "experiment_label": EXPERIMENT_LABEL,
        "scientific_status": "toy_mechanism_ablation_not_a_canonical_paper_method",
        **git,
        "python_path": PYTHON_PATH,
        "environment": "fairness_env",
        "versions": package_versions(),
        "data_path": str(Path(args.data_path).resolve()),
        "data_version": "CCAO/2025/training_data.parquet",
        "params_yaml": str(Path(args.params).resolve()),
        "heldout_test_mode": "pre_assessment_tail",
        "assessment_year_held_out_from_this_comparison": 2025,
        "train_n": data["n_train"],
        "eval_n": data["n_eval"],
        "train_period": [data["train_start"], data["train_end"]],
        "eval_period": [data["eval_start"], data["eval_end"]],
        "feature_count": data["n_features"],
        "categorical_count": data["n_categorical"],
        "predictor_cols": data["predictor_cols"],
        "categorical_cols": data["categorical_cols"],
        "lgbm_config_json": str(Path(args.lgbm_config_json).resolve()),
        "frozen_lgbm_params": dict(lgbm_params),
        "lgbm_params_sha256": lgbm_params_hash(lgbm_params),
        "seed": int(args.seed),
        "lambda_grid": [float(x) for x in args.lambdas],
        "cap_rule": "tau = Q_0.80(|c_i|) on the training sample; w_i = min(c_i^2, tau^2); no observation dropped",
        "realized_tau": float(cap_rows["tau"].iloc[0]) if not cap_rows.empty else None,
        "cap_quantile": CAP_QUANTILE,
        "rho_multipliers": rho_map,
        "six_objectives": objective_definitions(),
        "hessian_caveat": (
            "Level-invariant quadratic/capped: gradient of the profiled penalty is exact; "
            "LightGBM receives only the diagonal of the exact profiled Hessian "
            "W - ww^T/(sum w). The off-diagonal rank-one curvature is omitted."
        ),
        "rho_normalization_caveat": (
            "rho_m(lambda) = lambda * G_ref / G_m uses RMS fairness-only gradient at the "
            "shared rho=0 residual. This is an approximate common scale, not exact equivalence."
        ),
        "protocol": {
            "train": "full current development sample (pre_assessment_tail oldest 90%)",
            "evaluate": "current primary held-out sample only",
            "no_cv_folds": True,
            "no_rolling_origin": True,
            "no_hyperparameter_tuning": True,
            "no_early_stopping": True,
            "fixed_n_estimators": int(lgbm_params.get("n_estimators")),
            "quantities_varied": ["surrogate_type", "level_invariant", "lambda", "raw_rho"],
        },
        "sample_frac": args.sample_frac,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "partition": os.environ.get("SLURM_JOB_PARTITION"),
            "cpus_per_task": os.environ.get("SLURM_CPUS_PER_TASK"),
            "n_jobs_lightgbm": int(lgbm_params.get("n_jobs", -1)),
            "mem_on_node": os.environ.get("SLURM_MEM_PER_NODE") or os.environ.get("SLURM_MEM_PER_CPU"),
        },
    }
    if extra:
        payload.update(extra)
    dest = root / "config.json"
    write_json(dest, payload)
    return dest


def objective_definitions() -> Dict[str, Any]:
    return {
        "residual": "e_i = yhat_i - y_i (log-price residual)",
        "center": "c_i = y_i - mean(y) (identity; no logistic-quantile proxy)",
        "scaling": "LightGBM differentiates (n/2)J so rho=0 gives g=e, h=1",
        "quadratic_fixed": {
            "Psi": "(1/n) sum c_i^2 e_i^2",
            "g": "e_i + rho * c_i^2 e_i",
            "h": "1 + rho * c_i^2",
            "matches_canonical": True,
        },
        "quadratic_li": {
            "Psi": "min_a (1/n) sum c_i^2 (e_i-a)^2",
            "a_star": "sum(c^2 e) / sum(c^2)",
            "g": "e_i + rho * c_i^2 (e_i - a*)",
            "h": "1 + rho * (c_i^2 - c_i^4 / sum_j c_j^2)",
            "hessian_note": "exact gradient; diagonal of profiled Hessian only",
        },
        "absolute_fixed": {
            "Psi": "(1/n) sum |c_i| |e_i|",
            "g": "e_i + rho * (1/2) |c_i| sign(e_i)",
            "h": "1",
            "sign0": "sign(0)=0; no Huber smoothing",
        },
        "absolute_li": {
            "Psi": "min_a (1/n) sum |c_i| |e_i-a|",
            "a_star": "|c|-weighted median of e",
            "g": "e_i + rho * (1/2) |c_i| sign(e_i-a*)",
            "h": "1",
        },
        "capped_quadratic_fixed": {
            "Psi": "(1/n) sum w_i e_i^2",
            "w": "min(c_i^2, tau^2), tau=Q_0.80(|c|) on training sample",
            "g": "e_i + rho * w_i e_i",
            "h": "1 + rho * w_i",
        },
        "capped_quadratic_li": {
            "Psi": "min_a (1/n) sum w_i (e_i-a)^2",
            "a_star": "sum(w e)/sum(w)",
            "g": "e_i + rho * w_i (e_i-a*)",
            "h": "1 + rho * (w_i - w_i^2 / sum_j w_j)",
            "hessian_note": "exact gradient; diagonal of profiled Hessian only",
        },
    }


def assemble_outputs(args: argparse.Namespace) -> Dict[str, str]:
    root = Path(args.output_root)
    metrics = _collect_metrics(root)
    metrics.to_csv(root / "metrics.csv", index=False)
    compact_cols = [
        "surrogate_type",
        "level_invariant",
        "variant",
        "lambda",
        "raw_rho",
        "cap_tau",
        "R2_price",
        "MAE_price",
        "PRD",
        "PRB",
        "VEI",
        "MKI",
        "Beta_log",
        "dCor_e_y",
        "n_train",
        "n_eval",
        "runtime_sec",
    ]
    compact = metrics[[c for c in compact_cols if c in metrics.columns]].copy()
    compact.to_csv(root / "compact_metrics.csv", index=False)
    bins = _collect_ratio_bins(root, metrics)
    if not bins.empty:
        bins.to_csv(root / "ratio_bins.csv", index=False)
        fig_dir = _ensure_dir(root / "figures")
        ratio_fig = plot_ratio_shape(bins, fig_dir / "ratio_shape_evolution")
        mech_fig = plot_mechanism(metrics, fig_dir / "mechanism_vs_lambda")
        compact_fig = plot_compact_metrics(metrics, fig_dir / "compact_metric_evolution")
    else:
        ratio_fig = mech_fig = compact_fig = None
    cfg_path = root / "config.json"
    cfg = read_json(cfg_path) if cfg_path.is_file() else {}
    cfg["assembled_utc"] = pd.Timestamp.utcnow().isoformat()
    cfg["n_metric_rows"] = int(len(metrics))
    cfg["figure_paths"] = {
        "ratio_shape": None if ratio_fig is None else str(ratio_fig),
        "mechanism": None if mech_fig is None else str(mech_fig),
        "compact_metrics": None if compact_fig is None else str(compact_fig),
    }
    write_json(cfg_path, cfg)
    _log("assemble complete", metrics=str(root / "metrics.csv"))
    return {
        "metrics": str(root / "metrics.csv"),
        "ratio_shape": str(ratio_fig) if ratio_fig else "",
        "mechanism": str(mech_fig) if mech_fig else "",
        "compact": str(compact_fig) if compact_fig else "",
    }


def _parse_float_list(text: str) -> List[float]:
    return [float(x.strip()) for x in str(text).split(",") if x.strip()]


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EXPERIMENTAL / TOY 3x2 surrogate ablation.")
    p.add_argument("--mode", required=True, choices=["smoke", "probe", "calibrate", "fit-variant", "assemble"])
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    p.add_argument("--data-path", default=str(DEFAULT_DATA))
    p.add_argument("--params", default=str(DEFAULT_PARAMS))
    p.add_argument("--lgbm-config-json", default=str(DEFAULT_LGBM_CONFIG))
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--n-estimators", type=int, default=None, help="Override frozen n_estimators (smoke only).")
    p.add_argument("--lambdas", type=str, default=",".join(str(x) for x in LAMBDA_GRID))
    p.add_argument("--variant", type=str, default=None, help="fit-variant: e.g. quadratic_fixed")
    p.add_argument("--lambda-chunk", type=str, default=None, help="Optional i/n split of positive lambdas.")
    p.add_argument("--skip-lambda0-metrics", action="store_true")
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    args.lambdas = _parse_float_list(args.lambdas)
    os.environ.setdefault("MPLBACKEND", "Agg")
    _log("start", mode=args.mode, label=EXPERIMENT_LABEL, output=args.output_root)
    if args.mode == "smoke":
        run_smoke(args)
    elif args.mode == "probe":
        run_calibrate_or_probe(args, probe=True)
    elif args.mode == "calibrate":
        run_calibrate_or_probe(args, probe=False)
    elif args.mode == "fit-variant":
        if not args.variant:
            raise SystemExit("--variant is required for fit-variant")
        run_fit_variant(args)
    elif args.mode == "assemble":
        assemble_outputs(args)
    _log("done", mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
