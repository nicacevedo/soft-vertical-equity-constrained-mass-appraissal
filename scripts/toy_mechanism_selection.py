#!/usr/bin/env python3
"""EXPERIMENTAL / TOY six-path mechanism selection. Cluster-resident scientific workflow.

Writes only under output/toy_surrogate_ablation_v2/. Does not edit canonical paper methods.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import traceback
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
from canonical_experiment import git_state, lgbm_params_hash, read_json, write_json
from soft_constrained_models.toy_mechanism_objectives import (
    EXPERIMENT_LABEL,
    METHODS,
    METHOD_TITLES,
    MM_CURVATURE_DESCRIPTION,
    apply_moment_basis,
    build_training_moment_basis,
    eval_nonlinear_subspace,
    inspect_current_direct,
    k1_squared_covariance_identity,
    make_current_direct,
    method_k,
    moment_prox,
    projector_apply,
    quadratic_pointwise_prox,
    smooth_nl_metrics,
    unit_fairness_gradient,
    ToyMechanismLGB,
)
from utils.motivation_utils import compute_taxation_metrics

DEFAULT_OUTPUT = REPO / "output" / "toy_surrogate_ablation_v2"
PYTHON = "/home/nacevedo/.conda/envs/fairness_env/bin/python"
S_TARGETS: Tuple[float, ...] = (0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10)
RATIO_DISPLAY_S: Tuple[float, ...] = (1.00, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10)
S_TOL_PREF = 0.010
S_TOL_HARD = 0.015
S_PATH_STOP = 0.08
RHO_GROW = 1.7
MAX_PATH_FITS = 22
MAX_CORRECTIVE = 2
RHO_START_FRAC = 0.05


def _log(msg: str, **fields: Any) -> None:
    suffix = ""
    if fields:
        suffix = " | " + " | ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[toy_mechanism_selection] {msg}{suffix}", flush=True)


def _finite(value) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except Exception:
        return False


def atomic_csv(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def atomic_parquet(df: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def package_versions_metadata() -> Dict[str, str]:
    """Package versions via metadata only. Do not import dcor here: a cold Numba
    compile of dcor can stall preflight for tens of minutes on these nodes.
    Held-out dCor still imports dcor later, after family targets are locked.
    """
    import importlib.metadata as im

    versions = {"python": sys.version.split()[0]}
    mapping = {
        "lightgbm": "lightgbm",
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "scikit-learn",
        "dcor": "dcor",
    }
    for key, dist in mapping.items():
        try:
            versions[key] = str(im.version(dist))
        except Exception:
            versions[key] = "unknown"
    return versions


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(Path(path).read_bytes())


def sha_json(payload: Any) -> str:
    return sha_bytes(json.dumps(payload, sort_keys=True, default=str).encode())


def family_dir(root: Path, method: str) -> Path:
    return root / "families" / method


def pred_path(root: Path, method: str, tag: str) -> Path:
    return family_dir(root, method) / f"pred_{tag}.parquet"


def train_e_path(root: Path, method: str, tag: str) -> Path:
    return family_dir(root, method) / f"train_e_{tag}.npy"


def compute_nl_shape(pred: pd.DataFrame, n_bins: int = 30) -> float:
    pred = pred.copy()
    if "y_pred" not in pred.columns:
        pred["y_pred"] = np.exp(pred["y_pred_log"].to_numpy(dtype=float))
    if "y_true" not in pred.columns:
        pred["y_true"] = np.exp(pred["y_true_log"].to_numpy(dtype=float))
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
    m_b, x_b = [], []
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
    return float(np.sqrt(np.mean((m_b - np.polyval(coef, x_b)) ** 2)))


def eval_pred_frame(data: Dict[str, Any], pred_eval: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "row_id": data["df_eval"].index.to_numpy(),
            "sale_date": pd.to_datetime(data["df_eval"]["meta_sale_date"]).to_numpy(),
            "y_true_log": data["y_eval"],
            "y_pred_log": pred_eval,
            "y_true": np.exp(data["y_eval"]),
            "y_pred": np.exp(pred_eval) if np.all(np.isfinite(pred_eval)) else np.full_like(pred_eval, np.nan),
        }
    )


def training_s(beta: float, beta0: float, c_now: float, c0: float) -> float:
    s_beta = float(beta / beta0) if _finite(beta) and _finite(beta0) and abs(beta0) > 0 else float("nan")
    s_c = float(c_now / c0) if _finite(c_now) and _finite(c0) and abs(c0) > 0 else float("nan")
    if _finite(s_beta) and _finite(s_c) and abs(s_beta - s_c) > 1e-6:
        # Same target theoretically; keep beta_log definition as primary.
        pass
    return s_beta


def moment_pack(e: np.ndarray, phi: np.ndarray, m0: Optional[np.ndarray] = None) -> Dict[str, float]:
    m = (phi.T @ e) / float(e.size)
    out = {"m1": float(m[0]), "m2": float(m[1]), "m3": float(m[2])}
    if m0 is not None:
        for i, key in enumerate(("ret_m1", "ret_m2", "ret_m3")):
            out[key] = float(m[i] / m0[i]) if abs(float(m0[i])) > 0 else float("nan")
    return out


def fit_one(
    *,
    method: str,
    rho: float,
    data: Dict[str, Any],
    lgbm_params: dict,
    basis: Dict[str, Any],
    epsilon: float,
    verbose: bool = False,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    rss0 = v1._peak_rss_gb()
    status = "ok"
    error = ""
    pred_train = np.full(data["y_train"].shape, np.nan)
    pred_eval = np.full(data["y_eval"].shape, np.nan)
    model = None
    try:
        if method == "current_direct":
            model = make_current_direct(rho, lgbm_params, verbose=verbose)
        else:
            model = ToyMechanismLGB(
                method=method,
                rho=float(rho),
                lgbm_params=dict(lgbm_params),
                phi_train=basis["phi_train"],
                c_train=basis["c_train"],
                epsilon=float(epsilon),
                verbose=verbose,
                match_native_init=True,
            )
        model.fit(data["X_train"], data["y_train"])
        pred_train = np.asarray(model.predict(data["X_train"]), dtype=float).reshape(-1)
        pred_eval = np.asarray(model.predict(data["X_eval"]), dtype=float).reshape(-1)
        if (not np.all(np.isfinite(pred_train))) or (not np.all(np.isfinite(pred_eval))):
            status = "numerical_failure"
            error = "non_finite_prediction"
    except Exception as exc:
        status = "numerical_failure"
        error = f"{type(exc).__name__}: {exc}"
        model = None
    runtime = float(time.perf_counter() - t0)
    e_train = pred_train - data["y_train"]
    c_train = basis["c_train"]
    if status == "ok":
        c_now = float(np.mean(e_train * c_train))
        var_c = float(np.mean(np.square(c_train)))
        beta_tr = float(c_now / var_c) if var_c > 0.0 and np.isfinite(var_c) else float("nan")
        mse = float(np.mean(np.square(e_train)))
    else:
        c_now = float("nan")
        beta_tr = float("nan")
        mse = float("nan")
    return {
        "status": status,
        "error": error,
        "runtime_sec": runtime,
        "peak_rss_gb": max(v1._peak_rss_gb(), rss0),
        "pred_train": pred_train,
        "pred_eval": pred_eval,
        "e_train": e_train,
        "C_train": c_now,
        "Beta_log_train": beta_tr,
        "mse_train": mse,
        "best_iteration": _best_iteration(model),
        "n_estimators": int(lgbm_params.get("n_estimators", -1)),
    }


def _best_iteration(model) -> Optional[int]:
    if model is None:
        return None
    meth = getattr(model, "best_iteration", None)
    if callable(meth):
        try:
            value = meth()
            return int(value) if value is not None and int(value) > 0 else None
        except Exception:
            return None
    inner = getattr(model, "model", None)
    value = getattr(inner, "best_iteration_", None) if inner is not None else None
    try:
        iv = int(value)
    except Exception:
        return None
    return iv if iv > 0 else None


def save_fit_artifacts(root: Path, method: str, tag: str, data: Dict[str, Any], fit: Dict[str, Any]) -> None:
    fam = v1._ensure_dir(family_dir(root, method))
    if fit["status"] != "ok":
        return
    np.save(train_e_path(root, method, tag), np.asarray(fit["e_train"], dtype=np.float64))
    atomic_parquet(eval_pred_frame(data, fit["pred_eval"]), pred_path(root, method, tag))


def load_shared_state(root: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    cfg = read_json(root / "mechanism_config.json")
    packed = np.load(root / "basis.npz")
    phi = np.asarray(packed["phi_train"], dtype=np.float64)
    c = np.asarray(packed["c_train"], dtype=np.float64)
    rinv = np.asarray(packed["rinv"], dtype=np.float64)
    basis = {
        "y_mean": float(packed["y_mean"]),
        "sigma_c": float(packed["sigma_c"]),
        "rinv": rinv,
        "phi_train": phi,
        "c_train": c,
        "z_train": c / float(packed["sigma_c"]),
    }
    phi_chk = apply_moment_basis(data["y_train"], basis)
    if float(np.max(np.abs(phi_chk - phi))) > 1e-8:
        raise RuntimeError("Family job Phi rebuild disagrees with preflight basis.")
    resid = pd.read_parquet(root / "predictions" / "lambda0_train_residuals.parquet")
    e0 = resid["e0"].to_numpy(dtype=float)
    if int(e0.size) != int(data["n_train"]):
        raise RuntimeError("rho=0 residual length mismatch.")
    m0 = (phi.T @ e0) / float(e0.size)
    return {"cfg": cfg, "basis": basis, "e0": e0, "m0": m0}


def propose_rho_from_bracket(path: pd.DataFrame, sstar: float) -> Tuple[Optional[float], str]:
    ok = path.loc[path["status"].astype(str) == "ok"].copy()
    ok = ok.loc[ok["s_train"].apply(_finite)]
    ok = ok.loc[ok["s_train"].astype(float) > 0.0]
    ok = ok.loc[ok["raw_rho"].apply(_finite)]
    if ok.empty:
        return None, "no_positive_s_path"
    ok = ok.sort_values("raw_rho")
    s = ok["s_train"].astype(float).to_numpy()
    r = ok["raw_rho"].astype(float).to_numpy()
    # Closest first.
    nearest_i = int(np.argmin(np.abs(s - float(sstar))))
    if abs(s[nearest_i] - float(sstar)) <= S_TOL_PREF:
        return float(r[nearest_i]), "existing_within_pref"
    # Bracket assuming generally decreasing s in rho.
    above = np.flatnonzero(s >= float(sstar))
    below = np.flatnonzero(s <= float(sstar))
    if above.size and below.size:
        i_hi = above[np.argmin(s[above] - float(sstar))]
        i_lo = below[np.argmin(float(sstar) - s[below])]
        r_hi, r_lo = float(r[i_hi]), float(r[i_lo])
        s_hi, s_lo = float(s[i_hi]), float(s[i_lo])
        if abs(r_hi - r_lo) < 1e-18 or s_hi <= 0 or s_lo <= 0:
            return float(r[nearest_i]), "nearest_degenerate_bracket"
        # log-rho interpolation against log s when both positive.
        if min(s_hi, s_lo, sstar) > 0:
            t = (np.log(float(sstar)) - np.log(s_hi)) / (np.log(s_lo) - np.log(s_hi) + 1e-18)
            log_rho = np.log(max(r_hi, 1e-18)) + t * (np.log(max(r_lo, 1e-18)) - np.log(max(r_hi, 1e-18)))
            return float(np.exp(log_rho)), "log_rho_bracket"
        t = (float(sstar) - s_hi) / (s_lo - s_hi)
        return float(r_hi + t * (r_lo - r_hi)), "linear_rho_bracket"
    return float(r[nearest_i]), "nearest_unbracketed"


def row_from_fit(
    *,
    method: str,
    rho: float,
    tag: str,
    run_type: str,
    target_s: Optional[float],
    fit: Dict[str, Any],
    beta0: float,
    c0: float,
    m0: np.ndarray,
    phi: np.ndarray,
    mse0: float,
    note: str = "",
) -> Dict[str, Any]:
    s = training_s(fit["Beta_log_train"], beta0, fit["C_train"], c0)
    if fit["status"] == "ok" and _finite(fit["mse_train"]) and _finite(mse0) and mse0 > 0:
        if float(fit["mse_train"]) > 50.0 * float(mse0):
            fit = dict(fit)
            fit["status"] = "optimization_instability"
            fit["error"] = "train_mse_explosion"
    pack = moment_pack(fit["e_train"], phi, m0) if fit["status"] == "ok" else {k: float("nan") for k in ("m1", "m2", "m3", "ret_m1", "ret_m2", "ret_m3")}
    return {
        "experiment_label": EXPERIMENT_LABEL,
        "method": method,
        "raw_rho": float(rho),
        "run_type": run_type,
        "target_s": float(target_s) if target_s is not None else float("nan"),
        "achieved_s": s,
        "s_train": s,
        "C_train": fit["C_train"],
        "C_train_0": c0,
        "Beta_log_train": fit["Beta_log_train"],
        "Beta_log_train_0": beta0,
        "m1_train": pack.get("m1", float("nan")),
        "m2_train": pack.get("m2", float("nan")),
        "m3_train": pack.get("m3", float("nan")),
        "ret_m1_train": pack.get("ret_m1", float("nan")),
        "ret_m2_train": pack.get("ret_m2", float("nan")),
        "ret_m3_train": pack.get("ret_m3", float("nan")),
        "mse_train": fit["mse_train"],
        "runtime_sec": fit["runtime_sec"],
        "peak_rss_gb": fit["peak_rss_gb"],
        "status": fit["status"],
        "error": fit["error"],
        "pred_tag": tag,
        "note": note,
        "heldout_computed": False,
        "n_estimators": fit["n_estimators"],
    }


def fill_heldout(
    row: Dict[str, Any],
    *,
    data: Dict[str, Any],
    pred_eval: np.ndarray,
    basis: Dict[str, Any],
    y_mean_train: float,
) -> Dict[str, Any]:
    out = dict(row)
    if str(row.get("status")) != "ok" or (not np.all(np.isfinite(pred_eval))):
        return out
    _log("computing held-out metrics including dCor", method=row.get("method"), pred_tag=row.get("pred_tag"))
    raw = compute_taxation_metrics(data["y_eval"], pred_eval, scale="log", y_train=data["y_train"])
    held = v1.extract_required_metrics(raw)
    e_eval = pred_eval - data["y_eval"]
    phi_eval = apply_moment_basis(data["y_eval"], basis)
    m_eval = (phi_eval.T @ e_eval) / float(e_eval.size)
    c_eval = data["y_eval"] - float(y_mean_train)
    phi_nl = eval_nonlinear_subspace(c_eval)
    nl = smooth_nl_metrics(e_eval, phi_nl)
    pred_df = eval_pred_frame(data, pred_eval)
    out.update(held)
    out["m1_eval"] = float(m_eval[0])
    out["m2_eval"] = float(m_eval[1])
    out["m3_eval"] = float(m_eval[2])
    out["L_NL"] = float(nl["L_NL"])
    out["NL_share"] = float(nl["NL_share"])
    out["NL_shape"] = compute_nl_shape(pred_df)
    out["heldout_computed"] = True
    return out


def attach_common(row: Dict[str, Any], data: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(row)
    row["n_train"] = int(data["n_train"])
    row["n_eval"] = int(data["n_eval"])
    return row


def _csv_cell_nan(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except Exception:
        pass
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null"}:
        return True
    return False


def load_family_metrics(path: Path) -> List[Dict[str, Any]]:
    df = pd.read_csv(path)
    bool_keys = {"heldout_computed", "attained", "flag_hard_tolerance", "nonmonotone_path"}
    str_keys = {
        "experiment_label",
        "method",
        "run_type",
        "status",
        "error",
        "pred_tag",
        "note",
        "match_tol",
    }
    rows: List[Dict[str, Any]] = []
    for rec in df.to_dict(orient="records"):
        row: Dict[str, Any] = {}
        for key, value in rec.items():
            if key in bool_keys:
                if isinstance(value, str):
                    row[key] = value.strip().lower() in {"true", "1", "yes"}
                else:
                    row[key] = bool(value) if not _csv_cell_nan(value) else False
            elif key in str_keys:
                row[key] = "" if _csv_cell_nan(value) else str(value)
            elif _csv_cell_nan(value):
                row[key] = float("nan")
            else:
                row[key] = value
        rows.append(row)
    return rows


def path_resume_state(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    last_i = -1
    last_rho = None
    last_s = 1.0
    min_s = 1.0
    n_ok_path = 0
    has_reused_path = False
    for row in rows:
        status = str(row.get("status") or "")
        run_type = str(row.get("run_type") or "")
        s_now = float(row["s_train"]) if _finite(row.get("s_train")) else None
        if s_now is not None and s_now > 0:
            min_s = min(min_s, s_now)
        if run_type == "reused_path" and status == "ok":
            has_reused_path = True
        if run_type == "path" and status == "ok":
            n_ok_path += 1
            tag = str(row.get("pred_tag") or "")
            idx = n_ok_path - 1
            if tag.startswith("path_"):
                try:
                    idx = int(tag.split("_", 1)[1])
                except Exception:
                    pass
            if idx >= last_i and _finite(row.get("raw_rho")):
                last_i = idx
                last_rho = float(row["raw_rho"])
                if s_now is not None:
                    last_s = s_now
    complete = bool(has_reused_path or min_s <= S_PATH_STOP or (last_i + 1) >= MAX_PATH_FITS)
    return {
        "complete": complete,
        "next_i": last_i + 1,
        "last_rho": last_rho,
        "last_s": last_s,
        "min_s": min_s,
        "has_reused_path": has_reused_path,
        "n_ok_path": n_ok_path,
    }


def completed_match_row(
    rows: Sequence[Dict[str, Any]],
    *,
    sstar: float,
    root: Path,
    method: str,
) -> Optional[Dict[str, Any]]:
    best: Optional[Dict[str, Any]] = None
    best_err = float("inf")
    for row in rows:
        if str(row.get("status") or "") != "ok":
            continue
        if str(row.get("run_type") or "") not in {"match", "match_corrective", "match_reuse"}:
            continue
        if not _finite(row.get("target_s")) or abs(float(row["target_s"]) - float(sstar)) > 1e-9:
            continue
        if not _finite(row.get("s_train")) or float(row["s_train"]) <= 0:
            continue
        tag = str(row.get("pred_tag") or "")
        if not tag:
            continue
        if not pred_path(root, method, tag).is_file() or not train_e_path(root, method, tag).is_file():
            continue
        err = abs(float(row["s_train"]) - float(sstar))
        if err < best_err:
            best = dict(row)
            best_err = err
    return best


def run_focused_tests() -> Dict[str, Any]:
    """Run focused test_* functions with the stdlib. fairness_env has no pytest."""
    import importlib.util

    paths = [
        REPO / "tests" / "test_toy_mechanism_selection.py",
        REPO / "tests" / "test_canonical_objectives.py",
        # test_canonical_metrics.py is omitted here: it imports dcor via
        # paper_mechanism_metrics and can stall a cold Numba compile for many
        # minutes. Those metric identities already passed in the focused suite.
        # Held-out dCor is still computed after family targets are locked.
    ]
    results: List[Dict[str, Any]] = []
    n_fail = 0
    for path in paths:
        spec = importlib.util.spec_from_file_location(path.stem, path)
        if spec is None or spec.loader is None:
            n_fail += 1
            results.append({"test": path.stem, "pass": False, "error": "cannot_load_module"})
            continue
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        for name in sorted(dir(mod)):
            fn = getattr(mod, name)
            if not name.startswith("test_") or not callable(fn):
                continue
            try:
                fn()
                results.append({"test": f"{path.stem}.{name}", "pass": True})
                print(f"[preflight-test] PASS {path.stem}.{name}", flush=True)
            except Exception as exc:
                n_fail += 1
                err = f"{type(exc).__name__}: {exc}"
                results.append({"test": f"{path.stem}.{name}", "pass": False, "error": err})
                print(f"[preflight-test] FAIL {path.stem}.{name} | {err}", flush=True)
    failed = [r for r in results if not r["pass"]]
    return {
        "returncode": 0 if n_fail == 0 else 1,
        "n_ok": int(sum(1 for r in results if r["pass"])),
        "n_fail": int(n_fail),
        "failed": failed,
        "results": results,
        "runner": "stdlib_test_star_functions",
    }


def verify_reuse_hashes(cfg_old: dict, hashes_now: dict) -> Tuple[bool, List[str]]:
    checks = []
    ok = True
    pairs = [
        ("n_train", hashes_now["n_train"], 344607),
        ("n_eval", hashes_now["n_eval"], int(cfg_old.get("eval_n", -1))),
        ("feature_count", hashes_now["feature_count"], int(cfg_old.get("feature_count", -1))),
        ("n_estimators", hashes_now["n_estimators"], int((cfg_old.get("frozen_lgbm_params") or {}).get("n_estimators", -1))),
        ("seed", hashes_now["seed"], int((cfg_old.get("frozen_lgbm_params") or {}).get("random_state", -1))),
        ("heldout_test_mode", hashes_now["heldout_test_mode"], str(cfg_old.get("heldout_test_mode"))),
        ("lgbm_params_sha256", hashes_now["lgbm_params_sha256"], str(cfg_old.get("lgbm_params_sha256"))),
    ]
    for name, a, b in pairs:
        match = str(a) == str(b)
        checks.append(f"{name}: now={a} old={b} match={match}")
        ok = ok and match
    if int(hashes_now["n_train"]) != 344607 or int(hashes_now["n_eval"]) != 38290:
        ok = False
        checks.append("sample-size guard failed")
    return ok, checks


def run_preflight(args) -> None:
    root = v1._ensure_dir(Path(args.output_root))
    v1._ensure_dir(root / "logs")
    v1._ensure_dir(root / "predictions")
    v1._ensure_dir(root / "families")
    report: Dict[str, Any] = {
        "experiment_label": EXPERIMENT_LABEL,
        "status": "FAIL",
        "checks": {},
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    failures: List[str] = []

    def fail(key: str, msg: str, payload: Any = None) -> None:
        failures.append(key)
        report["checks"][key] = {"pass": False, "message": msg, "detail": payload}

    def ok(key: str, payload: Any = None) -> None:
        report["checks"][key] = {"pass": True, "detail": payload}

    pytest_info = run_focused_tests()
    if int(pytest_info["returncode"]) != 0:
        fail("pytest", "Focused theory/canonical tests failed.", pytest_info)
        write_json(root / "preflight_report.json", report)
        raise SystemExit(1)
    ok("pytest", pytest_info)

    params = v1.load_params(Path(args.params))
    n_jobs = v1._allocated_cpus()
    lgbm_params = v1.load_frozen_lgbm_params(Path(args.lgbm_config_json), n_jobs=n_jobs, n_estimators=args.n_estimators)
    data = v1.load_experiment_data(data_path=Path(args.data_path), params=params, sample_frac=args.sample_frac, seed=int(args.seed))
    _log("data ready", n_train=data["n_train"], n_eval=data["n_eval"], n_jobs=n_jobs)
    hashes_now = {
        "n_train": int(data["n_train"]),
        "n_eval": int(data["n_eval"]),
        "feature_count": int(data["n_features"]),
        "feature_list_sha256": sha_json(data["predictor_cols"]),
        "n_estimators": int(lgbm_params["n_estimators"]),
        "seed": int(lgbm_params.get("random_state", args.seed)),
        "heldout_test_mode": "pre_assessment_tail",
        "lgbm_params_sha256": lgbm_params_hash(lgbm_params),
        "params_yaml_sha256": sha_file(Path(args.params)),
        "lgbm_config_sha256": sha_file(Path(args.lgbm_config_json)),
        "train_period": [data["train_start"], data["train_end"]],
        "eval_period": [data["eval_start"], data["eval_end"]],
    }
    report["hashes"] = hashes_now
    if hashes_now["n_train"] != 344607 or hashes_now["n_eval"] != 38290:
        fail("sample_size", "Split sizes differ from the frozen paper setup.", hashes_now)
        write_json(root / "preflight_report.json", report)
        raise SystemExit(1)
    ok("sample_size", hashes_now)

    old_cfg_path = root / "config.json"
    reuse_baseline = False
    reuse_quadratic = False
    reuse_notes: List[str] = []
    if old_cfg_path.is_file():
        old_cfg = read_json(old_cfg_path)
        reuse_ok, reuse_notes = verify_reuse_hashes(old_cfg, hashes_now)
        q_metrics = family_dir(root, "quadratic") / "metrics.csv"
        q0 = root / "predictions" / "lambda0_train_residuals.parquet"
        reuse_baseline = bool(reuse_ok and q0.is_file())
        reuse_quadratic = bool(reuse_ok and q_metrics.is_file())
        if not reuse_ok:
            reuse_baseline = False
            reuse_quadratic = False
        ok("reuse_hash_compare", {"reuse_baseline": reuse_baseline, "reuse_quadratic": reuse_quadratic, "notes": reuse_notes})
    else:
        ok("reuse_hash_compare", {"reuse_baseline": False, "reuse_quadratic": False, "notes": ["no prior config.json"]})

    try:
        basis = build_training_moment_basis(data["y_train"])
        ok("basis", basis["diagnostics"])
    except Exception as exc:
        fail("basis", str(exc), traceback.format_exc())
        write_json(root / "preflight_report.json", report)
        raise SystemExit(1)

    phi = basis["phi_train"]
    rng = np.random.default_rng(2025)
    v = rng.normal(size=phi.shape[0])
    for k in (1, 2, 3):
        pv = projector_apply(v, phi[:, :k])
        p2 = projector_apply(pv, phi[:, :k])
        if float(np.max(np.abs(p2 - pv))) > 1e-6:
            fail("projector_full_data", f"P^2!=P for K={k}")
            write_json(root / "preflight_report.json", report)
            raise SystemExit(1)
    ok("projector_full_data", True)

    ident = k1_squared_covariance_identity(v, basis["c_train"], phi[:, 0], basis["sigma_c"])
    if ident["abs_err"] > 1e-6:
        fail("k1_equivalence", "squared-covariance identity failed on training directions", ident)
        write_json(root / "preflight_report.json", report)
        raise SystemExit(1)
    ok("k1_equivalence", ident)

    e_lin = 0.4 * phi[:, 0]
    geo = {}
    for k in (1, 2, 3):
        e1 = moment_prox(e_lin, phi[:, :k], 2.5)
        m1 = (phi.T @ e1) / float(e1.size)
        geo[f"k{k}_m2"] = float(m1[1])
        geo[f"k{k}_m3"] = float(m1[2])
        if abs(m1[1]) > 1e-8 or abs(m1[2]) > 1e-8:
            fail("geometry_pure_phi1", f"K={k} proximal left the phi1 span", m1.tolist())
            write_json(root / "preflight_report.json", report)
            raise SystemExit(1)
    e_q = quadratic_pointwise_prox(0.3 * basis["c_train"], basis["c_train"], 4.0)
    m_q = (phi.T @ e_q) / float(e_q.size)
    geo["quadratic_higher_order"] = float(abs(m_q[1]) + abs(m_q[2]))
    if geo["quadratic_higher_order"] <= 1e-8:
        fail("geometry_quadratic", "Quadratic prox did not create higher-order projection", m_q.tolist())
        write_json(root / "preflight_report.json", report)
        raise SystemExit(1)
    ok("geometry", geo)

    direct_info = inspect_current_direct()
    ok("current_direct_formulas", direct_info)

    # Shared rho=0
    lam0_resid = root / "predictions" / "lambda0_train_residuals.parquet"
    lam0_pred = root / "predictions" / "lambda0_shared.parquet"
    source = "fitted_here"
    if reuse_baseline and lam0_resid.is_file() and lam0_pred.is_file():
        resid = pd.read_parquet(lam0_resid)
        if int(len(resid)) == int(data["n_train"]):
            e0 = resid["e0"].to_numpy(dtype=float)
            pred_train0 = resid["y_pred_train_log"].to_numpy(dtype=float)
            source = "reused_existing_lambda0"
        else:
            reuse_baseline = False
            e0 = None
            pred_train0 = None
    else:
        e0 = None
        pred_train0 = None
    if e0 is None:
        _log("fitting shared rho=0 quadratic custom objective")
        dummy_basis = basis
        fit0 = fit_one(
            method="quadratic",
            rho=0.0,
            data=data,
            lgbm_params=lgbm_params,
            basis=dummy_basis,
            epsilon=1.0,
        )
        if fit0["status"] != "ok":
            fail("rho0", fit0["error"])
            write_json(root / "preflight_report.json", report)
            raise SystemExit(1)
        e0 = fit0["e_train"]
        pred_train0 = fit0["pred_train"]
        atomic_parquet(eval_pred_frame(data, fit0["pred_eval"]), lam0_pred)
        atomic_parquet(
            pd.DataFrame(
                {
                    "e0": e0,
                    "c": basis["c_train"],
                    "y_train_log": data["y_train"],
                    "y_pred_train_log": pred_train0,
                }
            ),
            lam0_resid,
        )
        source = "fitted_here"
    else:
        if not lam0_pred.is_file():
            fail("rho0", "reused residuals but missing eval parquet")
            write_json(root / "preflight_report.json", report)
            raise SystemExit(1)
    c0 = float(np.mean(e0 * basis["c_train"]))
    var_c = float(np.mean(np.square(basis["c_train"])))
    beta0_lin = float(c0 / var_c) if var_c > 0 else float("nan")
    epsilon = abs(beta0_lin)
    if not _finite(epsilon) or epsilon <= 0:
        fail("epsilon", "epsilon=|beta0| is not a positive finite constant", {"beta0": beta0_lin})
        write_json(root / "preflight_report.json", report)
        raise SystemExit(1)
    beta0 = float(beta0_lin)
    m0 = (phi.T @ e0) / float(e0.size)
    g_scales = {}
    for method in METHODS:
        g = unit_fairness_gradient(method=method, e=e0, c=basis["c_train"], phi=phi, epsilon=epsilon)
        g_scales[method] = {
            "G_m": float(np.sqrt(np.mean(np.square(g)))),
            "rho_start": float(RHO_START_FRAC * np.sqrt(np.mean(np.square(e0))) / max(float(np.sqrt(np.mean(np.square(g)))), 1e-18)),
        }
    ok("rho0", {"source": source, "C_train_0": c0, "Beta_log_train_0": beta0, "epsilon": epsilon, "g_scales": g_scales})

    tmp_npz = root / "basis_write.npz"
    np.savez(
        tmp_npz,
        rinv=basis["rinv"],
        y_mean=np.asarray(basis["y_mean"]),
        sigma_c=np.asarray(basis["sigma_c"]),
        phi_train=phi,
        c_train=basis["c_train"],
    )
    tmp_npz.replace(root / "basis.npz")
    write_json(root / "basis_diagnostics.json", basis["diagnostics"])
    _log("basis written; recording git and package metadata (no dcor import)")

    cfg = {
        "experiment_label": EXPERIMENT_LABEL,
        "experiment": "toy_mechanism_selection_v2",
        "methods": list(METHODS),
        "s_targets": list(S_TARGETS),
        "s_tol_pref": S_TOL_PREF,
        "s_tol_hard": S_TOL_HARD,
        "ratio_mode": "diff",
        "n_train": int(data["n_train"]),
        "n_eval": int(data["n_eval"]),
        "feature_count": int(data["n_features"]),
        "predictor_cols": list(data["predictor_cols"]),
        "frozen_lgbm_params": dict(lgbm_params),
        "hashes": hashes_now,
        "git": git_state(REPO),
        "packages": package_versions_metadata(),
        "C_train_0": c0,
        "Beta_log_train_0": beta0,
        "beta0_lin": beta0_lin,
        "epsilon": epsilon,
        "epsilon_rule": "abs(Cov_train(e0,c)/Var_train(c)) from shared rho=0 training residuals; not tuned",
        "m0_train": m0.tolist(),
        "g_scales": g_scales,
        "lambda0_source": source,
        "reuse_quadratic": bool(reuse_quadratic),
        "reuse_baseline": bool(reuse_baseline),
        "reuse_notes": reuse_notes,
        "direct": direct_info,
        "mm_curvature": MM_CURVATURE_DESCRIPTION,
        "heldout_never_used_for_rho": True,
        "no_cv": True,
        "data_path": str(Path(args.data_path).resolve()),
        "lgbm_config_json": str(Path(args.lgbm_config_json).resolve()),
    }
    write_json(root / "mechanism_config.json", cfg)

    if failures:
        report["status"] = "FAIL"
        report["failures"] = failures
        write_json(root / "preflight_report.json", report)
        raise SystemExit(1)
    report["status"] = "PASS"
    report["config_path"] = str(root / "mechanism_config.json")
    report["epsilon"] = epsilon
    report["reuse_quadratic"] = bool(reuse_quadratic)
    write_json(root / "preflight_report.json", report)
    _log("preflight PASS")


def run_fit_family(args) -> None:
    method = str(args.method)
    if method not in METHODS:
        raise SystemExit(f"Unknown method {method}")
    root = Path(args.output_root)
    pre = read_json(root / "preflight_report.json")
    if str(pre.get("status")) != "PASS":
        raise SystemExit("Refusing to run family job: preflight did not PASS.")
    params = v1.load_params(Path(args.params))
    n_jobs = v1._allocated_cpus()
    lgbm_params = v1.load_frozen_lgbm_params(Path(args.lgbm_config_json), n_jobs=n_jobs, n_estimators=args.n_estimators)
    data = v1.load_experiment_data(data_path=Path(args.data_path), params=params, sample_frac=args.sample_frac, seed=int(args.seed))
    state = load_shared_state(root, data)
    cfg = state["cfg"]
    basis = state["basis"]
    e0 = state["e0"]
    m0 = state["m0"]
    beta0 = float(cfg["Beta_log_train_0"])
    c0 = float(cfg["C_train_0"])
    epsilon = float(cfg["epsilon"])
    mse0 = float(np.mean(np.square(e0)))
    fam = v1._ensure_dir(family_dir(root, method))
    _log("family start", method=method, n_train=data["n_train"], n_estimators=lgbm_params["n_estimators"], n_jobs=n_jobs)

    rows: List[Dict[str, Any]] = []
    metrics_path = fam / "metrics.csv"
    resumed = False
    if metrics_path.is_file():
        rows = [attach_common(r, data) for r in load_family_metrics(metrics_path)]
        resumed = True
        _log("resuming family metrics", method=method, n=len(rows))

    def ensure_lambda0_row() -> None:
        if any(str(r.get("run_type") or "") == "shared_lambda0" for r in rows):
            return
        row0 = {
            "experiment_label": EXPERIMENT_LABEL,
            "method": method,
            "raw_rho": 0.0,
            "run_type": "shared_lambda0",
            "target_s": float("nan"),
            "achieved_s": 1.0,
            "s_train": 1.0,
            "C_train": c0,
            "C_train_0": c0,
            "Beta_log_train": beta0,
            "Beta_log_train_0": beta0,
            "m1_train": float(m0[0]),
            "m2_train": float(m0[1]),
            "m3_train": float(m0[2]),
            "ret_m1_train": 1.0,
            "ret_m2_train": 1.0,
            "ret_m3_train": 1.0,
            "mse_train": mse0,
            "runtime_sec": 0.0,
            "peak_rss_gb": float("nan"),
            "status": "ok",
            "error": "",
            "pred_tag": "lambda_0",
            "note": str(cfg.get("lambda0_source")),
            "heldout_computed": False,
            "n_estimators": int(lgbm_params["n_estimators"]),
        }
        rows.insert(0, attach_common(row0, data))

    ensure_lambda0_row()
    src0 = root / "predictions" / "lambda0_shared.parquet"
    if src0.is_file() and not pred_path(root, method, "lambda_0").is_file():
        shutil.copy2(src0, pred_path(root, method, "lambda_0"))
    te0 = train_e_path(root, method, "lambda_0")
    if not te0.is_file():
        np.save(te0, np.asarray(e0, dtype=np.float64))

    reuse_q = bool(cfg.get("reuse_quadratic")) and method == "quadratic"
    progress = path_resume_state(rows)
    if (not resumed) and reuse_q:
        old_q_path = family_dir(root, "quadratic") / "metrics.csv"
        if old_q_path.is_file():
            old = pd.read_csv(old_q_path)
            keep = old.loc[old["status"].astype(str) == "ok", ["raw_rho", "s_train", "C_train", "Beta_log_train", "status"]].copy()
            keep = keep.loc[keep["raw_rho"].apply(_finite) & keep["s_train"].apply(_finite)]
            keep = keep.drop_duplicates("raw_rho")
            for _, rec in keep.iterrows():
                if abs(float(rec["raw_rho"])) < 1e-18:
                    continue
                rows.append(
                    attach_common(
                        {
                            "experiment_label": EXPERIMENT_LABEL,
                            "method": method,
                            "raw_rho": float(rec["raw_rho"]),
                            "run_type": "reused_path",
                            "target_s": float("nan"),
                            "achieved_s": float(rec["s_train"]),
                            "s_train": float(rec["s_train"]),
                            "C_train": float(rec["C_train"]),
                            "C_train_0": c0,
                            "Beta_log_train": float(rec["Beta_log_train"]),
                            "Beta_log_train_0": beta0,
                            "m1_train": float("nan"),
                            "m2_train": float("nan"),
                            "m3_train": float("nan"),
                            "ret_m1_train": float("nan"),
                            "ret_m2_train": float("nan"),
                            "ret_m3_train": float("nan"),
                            "mse_train": float("nan"),
                            "runtime_sec": 0.0,
                            "peak_rss_gb": float("nan"),
                            "status": "ok",
                            "error": "",
                            "pred_tag": "",
                            "note": "reused_previous_quadratic_path_for_training_s_only",
                            "heldout_computed": False,
                            "n_estimators": int(lgbm_params["n_estimators"]),
                        },
                        data,
                    )
                )
            _log("quadratic reused previous training-s path", n=int(len(keep)))
            progress = path_resume_state(rows)

    if not progress["complete"]:
        n_path = int(progress["next_i"])
        if progress["last_rho"] is None:
            rho = float(cfg["g_scales"][method]["rho_start"])
            last_s = 1.0
        else:
            rho = float(progress["last_rho"]) * RHO_GROW
            last_s = float(progress["last_s"])
            _log(
                "resuming path",
                method=method,
                next_i=n_path,
                rho=f"{rho:.6g}",
                last_s=f"{last_s:.4g}",
            )
        while n_path < MAX_PATH_FITS:
            tag = f"path_{n_path}"
            _log("path fit", method=method, rho=f"{rho:.6g}", i=n_path)
            fit = fit_one(method=method, rho=rho, data=data, lgbm_params=lgbm_params, basis=basis, epsilon=epsilon)
            row = attach_common(
                row_from_fit(
                    method=method,
                    rho=rho,
                    tag=tag,
                    run_type="path",
                    target_s=None,
                    fit=fit,
                    beta0=beta0,
                    c0=c0,
                    m0=m0,
                    phi=basis["phi_train"],
                    mse0=mse0,
                    note="adaptive_train_s_path",
                ),
                data,
            )
            if fit["status"] == "ok":
                save_fit_artifacts(root, method, tag, data, fit)
            rows.append(row)
            atomic_csv(pd.DataFrame(rows), fam / "metrics.csv")
            n_path += 1
            if row["status"] != "ok":
                _log("path stop", reason=row["status"], error=row["error"])
                break
            last_s = float(row["s_train"]) if _finite(row["s_train"]) else last_s
            if _finite(last_s) and last_s <= S_PATH_STOP:
                break
            if _finite(last_s) and last_s <= 0.0:
                break
            rho *= RHO_GROW
            if rho > 1e8:
                break
    else:
        _log(
            "path already complete",
            method=method,
            min_s=f"{float(progress['min_s']):.4g}",
            reused=int(progress["has_reused_path"]),
        )

    path_df = pd.DataFrame(rows)
    s_vals = path_df.loc[path_df["status"] == "ok", "s_train"].map(lambda x: float(x) if _finite(x) else np.nan)
    nonmonotone = False
    s_ok = [float(x) for x in s_vals.tolist() if _finite(x)]
    if len(s_ok) >= 3:
        # After rho=0, s should tend to fall; flag any ascent larger than noise.
        diffs = np.diff(s_ok)
        if np.any(diffs > 0.03):
            nonmonotone = True
    locked: List[Dict[str, Any]] = []
    for sstar in S_TARGETS:
        sub = pd.DataFrame(rows)
        usable = sub.loc[(sub["status"] == "ok") & sub["s_train"].apply(_finite) & (sub["s_train"].astype(float) > 0.0)]
        if usable.empty:
            locked.append(
                attach_common(
                    {
                        "method": method,
                        "target_s": float(sstar),
                        "attained": False,
                        "status": "target_unattained",
                        "note": "no_positive_s_path",
                        "raw_rho": float("nan"),
                        "s_train": float("nan"),
                        "achieved_s": float("nan"),
                    },
                    data,
                )
            )
            continue
        prior = completed_match_row(rows, sstar=float(sstar), root=root, method=method)
        ds = (usable["s_train"].astype(float) - float(sstar)).abs()
        best = usable.loc[ds.idxmin()]
        need_fit = True
        rho_prop, how = propose_rho_from_bracket(usable, float(sstar))
        flag_hard = False
        chosen = None
        if prior is not None:
            chosen = dict(prior)
            chosen["target_s"] = float(sstar)
            err_prior = abs(float(chosen["s_train"]) - float(sstar))
            if err_prior <= S_TOL_PREF:
                chosen["attained"] = True
                chosen["match_tol"] = "pref_0.010"
            elif err_prior <= S_TOL_HARD:
                chosen["attained"] = True
                chosen["match_tol"] = "hard_0.015"
                chosen["flag_hard_tolerance"] = True
                flag_hard = True
            else:
                chosen["attained"] = False
                chosen["status"] = "target_unattained"
                chosen["note"] = f"closest_|ds|={err_prior:.5f}"
            need_fit = False
            _log("skip completed match", method=method, sstar=sstar, rho=f"{float(chosen.get('raw_rho', float('nan'))):.6g}", ds=f"{err_prior:.4g}")
        elif float(best["s_train"]) > 0 and float(ds.min()) <= S_TOL_PREF and str(best.get("pred_tag") or "") and pred_path(root, method, str(best["pred_tag"])).is_file() and train_e_path(root, method, str(best["pred_tag"])).is_file():
            chosen = dict(best)
            chosen["target_s"] = float(sstar)
            chosen["run_type"] = "match_reuse"
            chosen["note"] = "existing_within_pref"
            chosen["attained"] = True
            need_fit = False
        if need_fit:
            if rho_prop is None:
                locked.append(
                    attach_common(
                        {
                            "method": method,
                            "target_s": float(sstar),
                            "attained": False,
                            "status": "target_unattained",
                            "note": how,
                            "raw_rho": float("nan"),
                            "s_train": float("nan"),
                            "achieved_s": float("nan"),
                        },
                        data,
                    )
                )
                continue
            tag = f"match_s_{sstar:g}"
            _log("target fit", method=method, sstar=sstar, rho=f"{float(rho_prop):.6g}", how=how)
            fit = fit_one(method=method, rho=float(rho_prop), data=data, lgbm_params=lgbm_params, basis=basis, epsilon=epsilon)
            row = attach_common(
                row_from_fit(
                    method=method,
                    rho=float(rho_prop),
                    tag=tag,
                    run_type="match",
                    target_s=float(sstar),
                    fit=fit,
                    beta0=beta0,
                    c0=c0,
                    m0=m0,
                    phi=basis["phi_train"],
                    mse0=mse0,
                    note=how,
                ),
                data,
            )
            if fit["status"] == "ok":
                save_fit_artifacts(root, method, tag, data, fit)
            rows.append(row)
            atomic_csv(pd.DataFrame(rows), fam / "metrics.csv")
            chosen = dict(row)
            n_corr = 0
            while (
                chosen["status"] == "ok"
                and _finite(chosen["s_train"])
                and float(chosen["s_train"]) > 0
                and abs(float(chosen["s_train"]) - float(sstar)) > S_TOL_PREF
                and n_corr < MAX_CORRECTIVE
            ):
                rows_now = pd.DataFrame(rows)
                rho2, how2 = propose_rho_from_bracket(rows_now, float(sstar))
                if rho2 is None or abs(float(rho2) - float(chosen["raw_rho"])) < 1e-12:
                    break
                n_corr += 1
                tag_c = f"match_s_{sstar:g}_c{n_corr}"
                _log("corrective fit", method=method, sstar=sstar, rho=f"{float(rho2):.6g}")
                fit2 = fit_one(method=method, rho=float(rho2), data=data, lgbm_params=lgbm_params, basis=basis, epsilon=epsilon)
                row2 = attach_common(
                    row_from_fit(
                        method=method,
                        rho=float(rho2),
                        tag=tag_c,
                        run_type="match_corrective",
                        target_s=float(sstar),
                        fit=fit2,
                        beta0=beta0,
                        c0=c0,
                        m0=m0,
                        phi=basis["phi_train"],
                        mse0=mse0,
                        note=f"corrective_{how2}",
                    ),
                    data,
                )
                if fit2["status"] == "ok":
                    save_fit_artifacts(root, method, tag_c, data, fit2)
                rows.append(row2)
                atomic_csv(pd.DataFrame(rows), fam / "metrics.csv")
                if row2["status"] == "ok" and _finite(row2["s_train"]) and float(row2["s_train"]) > 0:
                    if abs(float(row2["s_train"]) - float(sstar)) <= abs(float(chosen["s_train"]) - float(sstar)):
                        chosen = dict(row2)
            if chosen["status"] != "ok" or (not _finite(chosen["s_train"])) or float(chosen["s_train"]) <= 0:
                chosen["attained"] = False
                chosen["status"] = "target_unattained"
            else:
                err = abs(float(chosen["s_train"]) - float(sstar))
                if err <= S_TOL_PREF:
                    chosen["attained"] = True
                    chosen["match_tol"] = "pref_0.010"
                elif err <= S_TOL_HARD:
                    chosen["attained"] = True
                    chosen["match_tol"] = "hard_0.015"
                    chosen["flag_hard_tolerance"] = True
                    flag_hard = True
                else:
                    chosen["attained"] = False
                    chosen["status"] = "target_unattained"
                    chosen["note"] = f"closest_|ds|={err:.5f}"
        chosen["target_s"] = float(sstar)
        chosen["flag_hard_tolerance"] = bool(flag_hard or chosen.get("flag_hard_tolerance", False))
        chosen["nonmonotone_path"] = bool(nonmonotone)
        locked.append(chosen)

    # Held-out only after targets are locked.
    locked_held = []
    for rec in locked:
        rec = dict(rec)
        tag = rec.get("pred_tag")
        p = pred_path(root, method, str(tag)) if isinstance(tag, str) and tag else None
        if rec.get("attained") and p is not None and p.is_file():
            pred = pd.read_parquet(p)
            pred_eval = pred["y_pred_log"].to_numpy(dtype=float)
            rec = fill_heldout(rec, data=data, pred_eval=pred_eval, basis=basis, y_mean_train=float(basis["y_mean"]))
        locked_held.append(rec)
    atomic_csv(pd.DataFrame(rows), fam / "metrics.csv")
    atomic_csv(pd.DataFrame(locked_held), fam / "matched.csv")
    sentinel = {
        "method": method,
        "status": "DONE",
        "n_path_rows": int(len(rows)),
        "n_matched": int(len(locked_held)),
        "n_attained": int(sum(1 for r in locked_held if r.get("attained"))),
        "nonmonotone_path": bool(nonmonotone),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
    }
    write_json(fam / "DONE.json", sentinel)
    _log("family done", method=method, attained=sentinel["n_attained"])


def _set_style():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _padded(vals) -> Tuple[float, float]:
    v = np.asarray(vals, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (-1.0, 1.0)
    lo, hi = float(np.min(v)), float(np.max(v))
    pad = 0.05 * max(hi - lo, 1e-3)
    return lo - pad, hi + pad


def plot_ratio(root: Path, matched: pd.DataFrame) -> Path:
    import matplotlib.pyplot as plt

    _set_style()
    frames = []
    base = root / "predictions" / "lambda0_shared.parquet"
    if base.is_file():
        pred = pd.read_parquet(base)
        if "y_pred" not in pred.columns:
            pred["y_pred"] = np.exp(pred["y_pred_log"])
            pred["y_true"] = np.exp(pred["y_true_log"])
        b = v1.equal_count_bins(pred["y_true"].to_numpy(), pred["y_pred"].to_numpy() / pred["y_true"].to_numpy())
        b["method"] = "baseline"
        b["target_s"] = 1.0
        frames.append(b)
    att = matched.loc[matched.get("attained", False) == True] if "attained" in matched.columns else matched
    for _, rec in att.iterrows():
        p = pred_path(root, rec["method"], str(rec.get("pred_tag")))
        if not p.is_file():
            continue
        pred = pd.read_parquet(p)
        if "y_pred" not in pred.columns:
            pred["y_pred"] = np.exp(pred["y_pred_log"])
            pred["y_true"] = np.exp(pred["y_true_log"])
        ratio = pred["y_pred"].to_numpy() / np.clip(pred["y_true"].to_numpy(), 1e-12, None)
        if not np.all(np.isfinite(ratio)):
            continue
        b = v1.equal_count_bins(pred["y_true"].to_numpy(), ratio)
        b["method"] = rec["method"]
        b["target_s"] = float(rec["target_s"])
        b["R2_price"] = rec.get("R2_price", np.nan)
        frames.append(b)
    bins = pd.concat(frames, ignore_index=True)
    atomic_csv(bins, root / "matched_ratio_bins.csv")
    display = list(RATIO_DISPLAY_S)
    plot_bins = bins.loc[
        (bins["method"] == "baseline")
        | bins["target_s"].apply(lambda s: any(np.isclose(float(s), t, atol=1e-12) for t in display))
    ].copy()
    axis_src = plot_bins.loc[plot_bins["method"] == "baseline"]
    good = att.loc[att["R2_price"].apply(_finite) & (att["R2_price"].astype(float) >= 0.0)] if "R2_price" in att.columns else att
    good_methods = set(good["method"].astype(str)) if not good.empty else set(METHODS)
    axis_src = pd.concat(
        [
            axis_src,
            plot_bins.loc[plot_bins["method"].isin(good_methods)],
        ],
        ignore_index=True,
    )
    y = axis_src["median_ratio"].to_numpy(dtype=float)
    y = y[np.isfinite(y)]
    ylim = _padded(y)
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.4), sharex=True, sharey=True)
    cmap = plt.cm.viridis
    x_all = plot_bins["median_sale_price"].to_numpy(dtype=float)
    xmin, xmax = float(np.min(x_all)), float(np.max(x_all))
    clip_rows = []
    for ax, method in zip(axes.ravel(), METHODS):
        baseb = plot_bins.loc[plot_bins["method"] == "baseline"].sort_values("bin")
        if not baseb.empty:
            ax.plot(baseb["median_sale_price"], baseb["median_ratio"], color=cmap(0.08), lw=1.5, marker="o", ms=2.0, label=r"$s=1$")
        sub = plot_bins.loc[plot_bins["method"] == method]
        curve_s = [t for t in display if abs(t - 1.0) > 1e-12]
        for i, sstar in enumerate(curve_s):
            part = sub.loc[np.isclose(sub["target_s"].astype(float), float(sstar))].sort_values("bin")
            if part.empty:
                continue
            yy = part["median_ratio"].to_numpy(dtype=float)
            yplot = np.clip(yy, ylim[0], ylim[1])
            clipped = ~np.isclose(yy, yplot, atol=1e-12)
            ax.plot(
                part["median_sale_price"],
                yplot,
                color=cmap(0.18 + 0.75 * i / max(len(curve_s) - 1, 1)),
                lw=1.4,
                marker="o",
                ms=2.0,
                label=rf"$s={sstar:.2f}$",
            )
            if np.any(clipped):
                ax.scatter(part["median_sale_price"].to_numpy()[clipped], yplot[clipped], marker="^", s=16, color="#B91C1C", zorder=5)
                clip_rows.append({"figure": "ratio", "method": method, "target_s": float(sstar), "n_clipped": int(np.sum(clipped))})
        ax.axhline(1.0, color="#111827", ls=(0, (2, 2)), lw=0.7)
        ax.set_xscale("log")
        ax.set_ylim(*ylim)
        ax.set_xlim(xmin / 1.05, xmax * 1.05)
        ax.set_title(METHOD_TITLES[method])
        ax.grid(True, color="#E5E7EB", lw=0.6)
        ax.set_axisbelow(True)
        ax.set_xlabel("Sale price")
        if method == "current_direct":
            ax.set_ylabel("Valuation-to-sale ratio")
        if method == "quadratic":
            ax.legend(fontsize=6, frameon=False, ncol=2, loc="best")
    fig.suptitle("EXPERIMENTAL / TOY matched signed-retention ratio shapes (held-out)", fontsize=11)
    fig.tight_layout()
    out = root / "figures" / "matched_ratio_shape"
    v1._ensure_dir(out.parent)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    if clip_rows:
        atomic_csv(pd.DataFrame(clip_rows), root / "plot_clipped_points.csv")
    return out.with_suffix(".pdf")


def plot_mechanism(root: Path, matched: pd.DataFrame) -> Path:
    import matplotlib.pyplot as plt

    _set_style()
    work = matched.loc[matched.get("attained", False) == True].copy() if "attained" in matched.columns else matched.copy()
    axis = work.loc[work["R2_price"].apply(_finite) & (work["R2_price"].astype(float) >= 0.0)] if "R2_price" in work.columns else work
    colors = {
        "current_direct": "#111827",
        "direct_mm_k1": "#1D4ED8",
        "moment_mm_k2": "#047857",
        "moment_mm_k3": "#6D28D9",
        "local_slope_smooth": "#B45309",
        "quadratic": "#BE123C",
    }
    panels = [
        ("Beta_log", r"held-out $\beta_{\log}$"),
        ("dCor_e_y", r"held-out $\mathrm{dCor}(e,y)$"),
        ("L_NL", r"smooth $L_{\mathrm{NL}}$"),
        ("NL_shape", r"binned $\mathrm{NL}_{\mathrm{shape}}$"),
    ]
    fig, axes = plt.subplots(4, 1, figsize=(8.6, 9.2), sharex=True)
    clip_rows = []
    for ax, (col, ylab) in zip(axes, panels):
        ylim = _padded(axis[col]) if col in axis.columns else (0.0, 1.0)
        for method in METHODS:
            sub = work.loc[work["method"] == method].sort_values("achieved_s", ascending=False)
            if sub.empty or col not in sub.columns:
                continue
            x = sub["achieved_s"].to_numpy(dtype=float)
            y = sub[col].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            x, y = x[finite], y[finite]
            yplot = np.clip(y, ylim[0], ylim[1])
            clipped = ~np.isclose(y, yplot, atol=1e-12)
            ax.plot(x, yplot, color=colors[method], marker="o", ms=4, lw=1.4, label=METHOD_TITLES[method])
            if np.any(clipped):
                ax.scatter(x[clipped], yplot[clipped], marker="^", s=22, color="#B91C1C", zorder=5)
                clip_rows.append({"figure": "mechanism", "method": method, "metric": col, "n_clipped": int(np.sum(clipped))})
        ax.set_xlim(1.05, -0.02)
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylab)
        ax.grid(True, color="#E5E7EB", lw=0.6)
        ax.set_axisbelow(True)
        if col == "Beta_log":
            ax.axhline(0.0, color="#111827", lw=0.7, ls=":")
    axes[-1].set_xlabel(r"Training signed retention $s$  ($1\rightarrow 0$)")
    axes[0].legend(frameon=False, fontsize=7, ncol=3, loc="best")
    fig.suptitle("EXPERIMENTAL / TOY matched mechanism vs training signed retention", fontsize=11)
    fig.tight_layout()
    out = root / "figures" / "mechanism_vs_s"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    if clip_rows:
        prev = root / "plot_clipped_points.csv"
        extra = pd.DataFrame(clip_rows)
        if prev.is_file():
            extra = pd.concat([pd.read_csv(prev), extra], ignore_index=True)
        atomic_csv(extra, prev)
    return out.with_suffix(".pdf")


def plot_tradeoff(root: Path, matched: pd.DataFrame) -> Path:
    import matplotlib.pyplot as plt

    _set_style()
    work = matched.loc[matched.get("attained", False) == True].copy() if "attained" in matched.columns else matched.copy()
    axis = work.loc[work["R2_price"].apply(_finite) & (work["R2_price"].astype(float) >= 0.0)] if "R2_price" in work.columns else work
    colors = {
        "current_direct": "#111827",
        "direct_mm_k1": "#1D4ED8",
        "moment_mm_k2": "#047857",
        "moment_mm_k3": "#6D28D9",
        "local_slope_smooth": "#B45309",
        "quadratic": "#BE123C",
    }
    fig, axes = plt.subplots(1, 4, figsize=(12.8, 3.4))
    panels = [
        ("R2_price", r"held-out $R^2_P$"),
        ("MAE_price", r"held-out MAE"),
        ("dCor_e_y", r"held-out $\mathrm{dCor}(e,y)$"),
        ("L_NL", r"smooth $L_{\mathrm{NL}}$"),
    ]
    beta_lim = _padded(axis["Beta_log"]) if "Beta_log" in axis.columns else (-0.2, 0.0)
    clip_rows = []
    for ax, (col, ylab) in zip(axes, panels):
        ylim = _padded(axis[col]) if col in axis.columns else (0.0, 1.0)
        for method in METHODS:
            sub = work.loc[work["method"] == method].sort_values("Beta_log")
            if sub.empty or col not in sub.columns:
                continue
            x = sub["Beta_log"].to_numpy(dtype=float)
            y = sub[col].to_numpy(dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            x, y = x[finite], y[finite]
            yplot = np.clip(y, ylim[0], ylim[1])
            xplot = np.clip(x, beta_lim[0], beta_lim[1])
            clipped = (~np.isclose(y, yplot, atol=1e-12)) | (~np.isclose(x, xplot, atol=1e-12))
            ax.plot(xplot, yplot, color=colors[method], marker="o", ms=4, lw=1.3, label=METHOD_TITLES[method])
            if np.any(clipped):
                ax.scatter(xplot[clipped], yplot[clipped], marker="^", s=22, color="#B91C1C", zorder=5)
                clip_rows.append({"figure": "tradeoff", "method": method, "metric": col, "n_clipped": int(np.sum(clipped))})
        ax.set_xlabel(r"held-out $\beta_{\log}$")
        ax.set_ylabel(ylab)
        ax.set_xlim(*beta_lim)
        ax.set_ylim(*ylim)
        ax.grid(True, color="#E5E7EB", lw=0.6)
        ax.set_axisbelow(True)
    axes[0].legend(frameon=False, fontsize=6.5, loc="best")
    fig.suptitle("EXPERIMENTAL / TOY matched tradeoffs", fontsize=11)
    fig.tight_layout()
    out = root / "figures" / "tradeoff_vs_beta"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    if clip_rows:
        prev = root / "plot_clipped_points.csv"
        extra = pd.DataFrame(clip_rows)
        if prev.is_file():
            extra = pd.concat([pd.read_csv(prev), extra], ignore_index=True)
        atomic_csv(extra, prev)
    return out.with_suffix(".pdf")


def disk_usage(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    if path.is_file():
        return path.stat().st_size
    for p in path.rglob("*"):
        if p.is_file():
            total += p.stat().st_size
    return total


def cleanup_if_pass(root: Path, matched: pd.DataFrame) -> Dict[str, Any]:
    before = disk_usage(root)
    removed: List[str] = []
    keep_tags = {("baseline", "lambda_0")}
    att = matched.loc[matched.get("attained", False) == True] if "attained" in matched.columns else matched
    for _, rec in att.iterrows():
        keep_tags.add((str(rec["method"]), str(rec.get("pred_tag"))))
        keep_tags.add((str(rec["method"]), "lambda_0"))
    for method in METHODS:
        keep_tags.add((method, "lambda_0"))
    for fam in ("huber", "absolute"):
        d = root / "families" / fam
        if not d.is_dir():
            continue
        for p in d.glob("pred_*.parquet"):
            p.unlink()
            removed.append(str(p))
        for p in d.glob("train_e_*.npy"):
            p.unlink()
            removed.append(str(p))
    for method in METHODS:
        d = family_dir(root, method)
        if not d.is_dir():
            continue
        for p in list(d.glob("pred_*.parquet")) + list(d.glob("train_e_*.npy")):
            tag = p.name.replace("pred_", "").replace("train_e_", "").replace(".parquet", "").replace(".npy", "")
            if (method, tag) not in keep_tags:
                p.unlink()
                removed.append(str(p))
    after = disk_usage(root)
    payload = {"bytes_before": before, "bytes_after": after, "n_removed": len(removed), "removed_head": removed[:40]}
    write_json(root / "cleanup_report.json", payload)
    return payload


def run_assemble(args) -> None:
    root = Path(args.output_root)
    graph_path = root / "manifests" / "mechanism_job_graph.json"
    graph = read_json(graph_path) if graph_path.is_file() else {}
    pre_path = root / "preflight_report.json"
    final: Dict[str, Any] = {
        "experiment_label": EXPERIMENT_LABEL,
        "status": "FAIL",
        "preflight_job_id": graph.get("preflight_job_id"),
        "family_job_ids": graph.get("family_job_ids"),
        "assemble_job_id": os.environ.get("SLURM_JOB_ID"),
        "git": git_state(REPO),
    }
    reasons: List[str] = []
    if not pre_path.is_file() or str(read_json(pre_path).get("status")) != "PASS":
        reasons.append("preflight_not_pass")
    sentinels = {}
    missing = []
    for method in METHODS:
        dpath = family_dir(root, method) / "DONE.json"
        mpath = family_dir(root, method) / "matched.csv"
        sentinels[method] = dpath.is_file() and mpath.is_file()
        if not sentinels[method]:
            missing.append(method)
    final["family_sentinels"] = sentinels
    if missing:
        reasons.append("missing_families:" + ",".join(missing))
    frames = []
    for method in METHODS:
        mpath = family_dir(root, method) / "matched.csv"
        if mpath.is_file():
            df = pd.read_csv(mpath)
            df["method"] = method
            frames.append(df)
    matched = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    if not matched.empty:
        # Guard: held-out must not have been used to pick rho. Family code sets heldout_computed only after lock.
        if "heldout_computed" in matched.columns and bool((matched["attained"] == True).any()) and not bool(matched.loc[matched["attained"] == True, "heldout_computed"].fillna(False).all()):
            reasons.append("heldout_metrics_missing_on_attained_rows")
        atomic_csv(matched, root / "matched_correction_metrics.csv")
        mets = []
        for method in METHODS:
            mp = family_dir(root, method) / "metrics.csv"
            if mp.is_file():
                mets.append(pd.read_csv(mp))
        if mets:
            atomic_csv(pd.concat(mets, ignore_index=True), root / "metrics.csv")
    cfg = read_json(root / "mechanism_config.json") if (root / "mechanism_config.json").is_file() else {}
    final["configuration_hash"] = sha_json(cfg.get("hashes", cfg))
    final["completed_methods"] = [m for m, ok in sentinels.items() if ok]
    if not matched.empty and "attained" in matched.columns:
        att_tab = (
            matched.groupby("method")["attained"].agg(lambda s: int(np.sum(s.astype(bool)))).to_dict()
            if "method" in matched.columns
            else {}
        )
        final["attained_counts"] = att_tab
        final["unattained"] = matched.loc[matched["attained"] != True, ["method", "target_s", "status", "note"]].to_dict(
            orient="records"
        )
    figs = {}
    if missing:
        final["reasons"] = reasons
        write_json(root / "FINAL_STATUS.json", final)
        _log("assemble FAIL", reasons=",".join(reasons))
        raise SystemExit(1)
    try:
        figs["matched_ratio_shape"] = str(plot_ratio(root, matched))
        figs["mechanism_vs_s"] = str(plot_mechanism(root, matched))
        figs["tradeoff_vs_beta"] = str(plot_tradeoff(root, matched))
    except Exception as exc:
        reasons.append(f"figure_error:{type(exc).__name__}:{exc}")
        final["reasons"] = reasons
        write_json(root / "FINAL_STATUS.json", final)
        raise
    # Internal consistency of matched s
    if "achieved_s" in matched.columns and "target_s" in matched.columns:
        att = matched.loc[matched["attained"] == True]
        if not att.empty:
            ds = (att["achieved_s"].astype(float) - att["target_s"].astype(float)).abs()
            if bool((ds > S_TOL_HARD + 1e-12).any()):
                reasons.append("attained_row_exceeds_hard_tolerance")
    if "heldout_never_used_for_rho" not in cfg:
        reasons.append("config_missing_heldout_guard")
    if reasons:
        final["status"] = "FAIL"
        final["reasons"] = reasons
        final["figure_paths"] = figs
        write_json(root / "FINAL_STATUS.json", final)
        raise SystemExit(1)
    cleanup = cleanup_if_pass(root, matched)
    final.update(
        {
            "status": "PASS",
            "figure_paths": figs,
            "output_paths": {
                "matched": str(root / "matched_correction_metrics.csv"),
                "metrics": str(root / "metrics.csv"),
                "preflight": str(root / "preflight_report.json"),
                "basis": str(root / "basis.npz"),
            },
            "cleanup": cleanup,
        }
    )
    write_json(root / "FINAL_STATUS.json", final)
    _log("assemble PASS")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EXPERIMENTAL / TOY six-path mechanism selection.")
    p.add_argument("--mode", required=True, choices=["preflight", "fit-family", "assemble"])
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    p.add_argument("--data-path", default=str(v1.DEFAULT_DATA))
    p.add_argument("--params", default=str(v1.DEFAULT_PARAMS))
    p.add_argument("--lgbm-config-json", default=str(v1.DEFAULT_LGBM_CONFIG))
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--method", type=str, default=None)
    p.add_argument("--verbose", action="store_true")
    return p


def main() -> int:
    args = build_parser().parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    _log("start", mode=args.mode, label=EXPERIMENT_LABEL)
    if args.mode == "preflight":
        run_preflight(args)
    elif args.mode == "fit-family":
        if not args.method:
            raise SystemExit("--method is required")
        run_fit_family(args)
    else:
        run_assemble(args)
    _log("done", mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
