#!/usr/bin/env python3
"""EXPERIMENTAL / TOY follow-up V3: refined QD and orthogonal QNL.

Isolated under output/toy_surrogate_followup_v3/. Does not modify V1/V2/hybrid
artifacts, paper methods, or boosting_models.py.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

import toy_hybrid_selection as hyb
import toy_mechanism_selection as six
import toy_surrogate_ablation as v1
from canonical_experiment import git_state, lgbm_params_hash, read_json, write_json
from run_temporal_cv import _load_and_split_data
from soft_constrained_models.toy_followup_metrics import (
    n3_orth_from_phi3col,
    orthonormality_diagnostics,
    reconstruct_phi_full,
)
from soft_constrained_models.toy_hybrid_objectives import ToyHybridLGB, m23_norm2
from soft_constrained_models.toy_mechanism_objectives import EXPERIMENT_LABEL, apply_moment_basis
from utils.motivation_utils import _build_time_block_bootstrap_indices, compute_taxation_metrics

V2 = REPO / "output" / "toy_surrogate_ablation_v2"
V1 = REPO / "output" / "toy_surrogate_ablation"
DEFAULT_OUTPUT = REPO / "output" / "toy_surrogate_followup_v3"
S_TOL_PREF = 0.010
S_TOL_HARD = 0.015
MAX_CORRECTIVE = 2
LAMBDA_START = 0.5
LAMBDA_GROW = 1.7
GAMMA_GRID: Tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
QD_TARGETS_BELOW = {
    0.30: (0.25, 0.225, 0.20, 0.175, 0.15),
    0.25: (0.225, 0.20, 0.175, 0.15),
}
QNL_TARGETS: Tuple[float, ...] = (0.20, 0.15, 0.10)
QD_ANCHORS: Tuple[float, ...] = (0.30, 0.25)
QNL_ANCHORS: Tuple[float, ...] = (0.25, 0.30)
BOOTSTRAP_N = 500
BOOTSTRAP_SEED = 2025
TRACE_COLS = hyb.TRACE_COLS
PRIMARY_METHODS = ("current_direct", "quadratic", "direct_mm_k1", "quadratic_direct_cap", "quadratic_nl_guardrail")
CONTEXT_METHODS = PRIMARY_METHODS + ("huber", "absolute", "moment_mm_k2", "moment_mm_k3", "local_slope_smooth")


def akey(s: float) -> str:
    return f"{float(s):.2f}"


def traces_df(traces: List[Dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(traces) if traces else pd.DataFrame(columns=list(TRACE_COLS))
    for col in TRACE_COLS:
        if col not in df.columns:
            df[col] = np.nan
    return df[list(TRACE_COLS)]


def interpolate_lambda(pairs: List[Tuple[float, float]], sstar: float) -> Tuple[Optional[float], str]:
    """Log-interpolate λ from observed (λ, s) pairs. λ=0 is allowed as a bracket endpoint."""
    ok = [
        (float(p), float(s))
        for p, s in pairs
        if six._finite(p) and six._finite(s) and float(p) >= 0 and float(s) > 0
    ]
    if not ok:
        return None, "no_pairs"
    s = np.asarray([x[1] for x in ok], dtype=float)
    p = np.asarray([x[0] for x in ok], dtype=float)
    nearest = int(np.argmin(np.abs(s - float(sstar))))
    if abs(s[nearest] - float(sstar)) <= S_TOL_PREF and float(p[nearest]) > 0:
        return float(p[nearest]), "existing_within_pref"
    above = np.flatnonzero(s >= float(sstar))
    below = np.flatnonzero(s <= float(sstar))
    eps = 1e-8
    if above.size and below.size:
        i_hi = above[int(np.argmin(s[above] - float(sstar)))]
        i_lo = below[int(np.argmin(float(sstar) - s[below]))]
        p_hi, p_lo = float(p[i_hi]), float(p[i_lo])
        s_hi, s_lo = float(s[i_hi]), float(s[i_lo])
        if abs(p_hi - p_lo) < 1e-18:
            return (float(p[nearest]) if float(p[nearest]) > 0 else None), "nearest_degenerate"
        t = (np.log(float(sstar)) - np.log(s_hi)) / (np.log(s_lo) - np.log(s_hi) + 1e-18)
        logp = np.log(p_hi + eps) + t * (np.log(p_lo + eps) - np.log(p_hi + eps))
        return float(max(np.exp(logp) - eps, eps)), "log_lambda_bracket"
    if float(p[nearest]) > 0:
        return float(p[nearest]), "nearest_unbracketed"
    return None, "nearest_is_lambda0"


def _is_attained_cell(val: Any) -> bool:
    if val is True or val == 1:
        return True
    return str(val).strip().lower() in {"true", "1", "yes"}


def _keep_plot_rows(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "attained" not in df.columns:
        return df
    a = df["attained"]
    keep = a.isna() | a.astype(str).str.lower().isin(["true", "1", "nan", ""])
    return df.loc[keep]


def _log(msg: str, **fields: Any) -> None:
    suffix = ""
    if fields:
        suffix = " | " + " | ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[toy_followup_v3] {msg}{suffix}", flush=True)


def fam_dir(root: Path, family: str) -> Path:
    return root / "families" / family


def qd_family(anchor: float) -> str:
    return f"qd_a{int(round(anchor * 100))}"


def qnl_family(anchor: float) -> str:
    return f"qnl_a{int(round(anchor * 100))}"


def tag_for(sstar: float) -> str:
    return f"lock_s_{sstar:g}"


def sentinel_path(root: Path, family: str, sstar: float) -> Path:
    return fam_dir(root, family) / f"DONE_s_{sstar:g}.json"


def load_followup_data(args) -> Dict[str, Any]:
    params = v1.load_params(Path(args.params))
    df_train, df_test, df_assess, predictor_cols, categorical_cols = _load_and_split_data(
        data_path=str(Path(args.data_path)),
        params=params,
        target_column="meta_sale_price",
        date_column="meta_sale_date",
        assessment_year=2025,
        heldout_test_mode="pre_assessment_tail",
        sample_frac=args.sample_frac,
        sample_seed=int(args.seed),
        universe_start="2016-01-01",
        pre_assessment_end="2024-12-31",
    )
    X_train, y_train = v1.prepare_xy(df_train, predictor_cols, categorical_cols, "meta_sale_price")
    X_eval, y_eval = v1.prepare_xy(df_test, predictor_cols, categorical_cols, "meta_sale_price")
    out: Dict[str, Any] = {
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
        "n_features": int(len(predictor_cols)),
        "n_categorical": int(len(categorical_cols)),
    }
    if df_assess is not None and len(df_assess):
        X_fwd, y_fwd = v1.prepare_xy(df_assess, predictor_cols, categorical_cols, "meta_sale_price")
        out.update(
            {
                "df_forward": df_assess,
                "X_forward": X_fwd,
                "y_forward": y_fwd,
                "n_forward": int(len(df_assess)),
            }
        )
    else:
        out["n_forward"] = 0
    return out


def quadratic_row(sstar: float) -> pd.Series:
    df = pd.read_csv(V2 / "families" / "quadratic" / "matched.csv")
    hit = df.loc[np.isclose(df["target_s"].astype(float), float(sstar))]
    if hit.empty:
        raise RuntimeError(f"Missing quadratic matched row at target_s={sstar}")
    return hit.iloc[0]


def attainment(s: float, sstar: float) -> Tuple[bool, str, bool]:
    if not six._finite(s) or float(s) <= 0:
        return False, "", False
    err = abs(float(s) - float(sstar))
    if err <= S_TOL_PREF:
        return True, "pref_0.010", False
    if err <= S_TOL_HARD:
        return True, "hard_0.015", True
    return False, "", False


def attach_n3(row: Dict[str, Any], e: Optional[np.ndarray], phi: np.ndarray, prefix: str) -> Dict[str, Any]:
    out = dict(row)
    if e is None or (not np.all(np.isfinite(e))):
        out[f"N3_orth_{prefix}"] = float("nan")
        out[f"N3_rel_{prefix}"] = float("nan")
        return out
    _m2, _m3, n3, n3_rel = n3_orth_from_phi3col(e, phi)
    out[f"N3_orth_{prefix}"] = n3
    out[f"N3_rel_{prefix}"] = n3_rel
    if prefix == "train":
        out["M23_train"] = n3
        out["N3_orth"] = n3
        out["N3_rel"] = n3_rel
    return out


def fit_followup(
    *,
    method: str,
    data: Dict[str, Any],
    lgbm_params: dict,
    basis: Dict[str, Any],
    d23: np.ndarray,
    alpha: float,
    lam: float,
    rho: float,
    gamma: float,
) -> Dict[str, Any]:
    """One LightGBM fit; held-out and 2025-forward predictions after the same training run."""
    t0 = time.perf_counter()
    rss0 = v1._peak_rss_gb()
    status = "ok"
    error = ""
    pred_train = np.full(data["y_train"].shape, np.nan)
    pred_eval = np.full(data["y_eval"].shape, np.nan)
    pred_fwd = np.full((int(data.get("n_forward") or 0),), np.nan)
    try:
        model = ToyHybridLGB(
            method=method,
            lgbm_params=dict(lgbm_params),
            c_train=basis["c_train"],
            phi_train=basis["phi_train"],
            d23=d23,
            alpha=alpha,
            lam=lam,
            rho=rho,
            gamma=gamma,
            match_native_init=True,
        )
        model.fit(data["X_train"], data["y_train"])
        pred_train = np.asarray(model.predict(data["X_train"]), dtype=float).reshape(-1)
        pred_eval = np.asarray(model.predict(data["X_eval"]), dtype=float).reshape(-1)
        if int(data.get("n_forward") or 0) > 0:
            pred_fwd = np.asarray(model.predict(data["X_forward"]), dtype=float).reshape(-1)
        if (not np.all(np.isfinite(pred_train))) or (not np.all(np.isfinite(pred_eval))):
            status = "numerical_failure"
            error = "non_finite_prediction"
    except Exception as exc:
        status = "numerical_failure"
        error = f"{type(exc).__name__}: {exc}"
    runtime = float(time.perf_counter() - t0)
    e_train = pred_train - data["y_train"]
    if status == "ok":
        c_now = float(np.mean(e_train * basis["c_train"]))
        var_c = float(np.mean(np.square(basis["c_train"])))
        beta_tr = float(c_now / var_c) if var_c > 0 else float("nan")
        mse = float(np.mean(np.square(e_train)))
        pack = six.moment_pack(e_train, basis["phi_train"])
        m2, m3 = float(pack["m2"]), float(pack["m3"])
    else:
        c_now = beta_tr = mse = m2 = m3 = float("nan")
        pack = {k: float("nan") for k in ("m1", "m2", "m3")}
    return {
        "status": status,
        "error": error,
        "runtime_sec": runtime,
        "peak_rss_gb": max(v1._peak_rss_gb(), rss0),
        "pred_train": pred_train,
        "pred_eval": pred_eval,
        "pred_forward": pred_fwd,
        "e_train": e_train,
        "C_train": c_now,
        "Beta_log_train": beta_tr,
        "mse_train": mse,
        "m1_train": float(pack["m1"]),
        "m2_train": m2,
        "m3_train": m3,
        "M23_train": m23_norm2(m2, m3) if six._finite(m2) else float("nan"),
        "n_estimators": int(lgbm_params.get("n_estimators", -1)),
    }


def save_lock_artifacts(
    *,
    root: Path,
    family: str,
    tag: str,
    data: Dict[str, Any],
    fit: Dict[str, Any],
) -> None:
    dest = v1._ensure_dir(fam_dir(root, family))
    np.save(dest / f"train_e_{tag}.npy", np.asarray(fit["e_train"], dtype=np.float64))
    six.atomic_parquet(six.eval_pred_frame(data, fit["pred_eval"]), dest / f"pred_{tag}.parquet")
    pf = np.asarray(fit.get("pred_forward", []), dtype=float).reshape(-1)
    n_fwd = int(data.get("n_forward") or 0)
    if n_fwd > 0 and pf.size == n_fwd and np.all(np.isfinite(pf)):
        pred_fwd = np.asarray(fit["pred_forward"], dtype=float).reshape(-1)
        fwd = pd.DataFrame(
            {
                "row_id": data["df_forward"].index.to_numpy(),
                "sale_date": pd.to_datetime(data["df_forward"]["meta_sale_date"]).to_numpy(),
                "y_true_log": data["y_forward"],
                "y_pred_log": pred_fwd,
                "y_true": np.exp(data["y_forward"]),
                "y_pred": np.exp(pred_fwd) if np.all(np.isfinite(pred_fwd)) else np.full_like(pred_fwd, np.nan),
            }
        )
        six.atomic_parquet(fwd, dest / f"pred_forward_{tag}.parquet")


def fill_eval_metrics(row: Dict[str, Any], *, data: Dict[str, Any], pred: np.ndarray, basis: Dict[str, Any], y_mean: float) -> Dict[str, Any]:
    out = six.fill_heldout(row, data=data, pred_eval=pred, basis=basis, y_mean_train=y_mean)
    e_eval = pred - data["y_eval"]
    phi_eval = apply_moment_basis(data["y_eval"], basis)
    out = attach_n3(out, e_eval, phi_eval, "eval")
    return out


def fill_forward_metrics(row: Dict[str, Any], *, data: Dict[str, Any], pred_fwd: np.ndarray, basis: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(row)
    if int(data.get("n_forward") or 0) == 0 or (not np.all(np.isfinite(pred_fwd))):
        out["forward_computed"] = False
        return out
    raw = compute_taxation_metrics(data["y_forward"], pred_fwd, scale="log", y_train=data["y_train"])
    held = v1.extract_required_metrics(raw)
    for k, val in held.items():
        out[f"{k}__forward_2025"] = val
    e_fwd = pred_fwd - data["y_forward"]
    phi_fwd = apply_moment_basis(data["y_forward"], basis)
    _m2, _m3, n3, n3_rel = n3_orth_from_phi3col(e_fwd, phi_fwd)
    out["N3_orth__forward_2025"] = n3
    out["N3_rel__forward_2025"] = n3_rel
    out["forward_computed"] = True
    out["n_forward"] = int(data["n_forward"])
    return out


def write_lock(root: Path, family: str, sstar: float, rec: Dict[str, Any], traces: List[Dict[str, Any]]) -> None:
    dest = v1._ensure_dir(fam_dir(root, family))
    six.atomic_csv(traces_df(traces), dest / f"search_trace_s_{sstar:g}.csv")
    six.atomic_csv(pd.DataFrame([rec]), dest / f"matched_s_{sstar:g}.csv")
    write_json(
        sentinel_path(root, family, sstar),
        {
            "family": family,
            "target_s": float(sstar),
            "status": "DONE",
            "attained": bool(rec.get("attained")),
            "achieved_s": rec.get("s_train"),
            "alpha": rec.get("alpha"),
            "lambda": rec.get("lambda"),
            "rho": rec.get("rho", rec.get("raw_rho")),
            "gamma": rec.get("gamma"),
            "N3_orth_train": rec.get("N3_orth_train", rec.get("M23_train")),
            "tau": rec.get("tau"),
            "provenance": rec.get("provenance"),
            "pred_tag": rec.get("pred_tag", ""),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "n_attempts": int(len(traces)),
        },
    )


def copy_hybrid_lock(
    *,
    root: Path,
    family: str,
    method: str,
    anchor: float,
    sstar: float,
    src_tag: str,
    data: Dict[str, Any],
    basis: Dict[str, Any],
    cfg: dict,
) -> None:
    src = V2 / "families" / method
    tag = tag_for(sstar)
    dest = v1._ensure_dir(fam_dir(root, family))
    if (src / f"pred_{src_tag}.parquet").is_file():
        shutil.copy2(src / f"pred_{src_tag}.parquet", dest / f"pred_{tag}.parquet")
    if (src / f"train_e_{src_tag}.npy").is_file():
        shutil.copy2(src / f"train_e_{src_tag}.npy", dest / f"train_e_{tag}.npy")
    mpath = src / f"matched_s_{sstar:g}.csv"
    rec = pd.read_csv(mpath).iloc[0].to_dict() if mpath.is_file() else {}
    rec.update(
        {
            "experiment_label": EXPERIMENT_LABEL,
            "method": method,
            "family": family,
            "anchor_s": float(anchor),
            "target_s": float(sstar),
            "provenance": f"reused_hybrid_v2:{method}:{src_tag}",
            "pred_tag": tag,
            "n_train": int(data["n_train"]),
            "n_eval": int(data["n_eval"]),
            "heldout_computed": True,
            "forward_computed": False,
            "forward_note": "reused_hybrid_lock_no_saved_booster",
        }
    )
    e_tr = np.load(dest / f"train_e_{tag}.npy")
    rec = attach_n3(rec, e_tr, basis["phi_train"], "train")
    pred = pd.read_parquet(dest / f"pred_{tag}.parquet")
    rec = attach_n3(rec, pred["y_pred_log"].to_numpy(dtype=float) - pred["y_true_log"].to_numpy(dtype=float), apply_moment_basis(data["y_eval"], basis), "eval")
    rec["tau"] = float(cfg["anchors"][akey(anchor)]["tau"])
    rec["alpha"] = float(cfg["anchors"][akey(anchor)]["rho"]) if method == "quadratic_direct_cap" else rec.get("alpha", 0.0)
    traces = []
    tpath = src / f"search_trace_s_{sstar:g}.csv"
    if tpath.is_file():
        traces = pd.read_csv(tpath).to_dict(orient="records")
        for tr in traces:
            tr["heldout_computed"] = False
    write_lock(root, family, sstar, rec, traces)
    _log("reused hybrid lock", family=family, sstar=sstar, src=src_tag)


def reuse_quadratic_as_gamma0(*, root: Path, family: str, anchor: float, sstar: float, data, basis, cfg) -> None:
    qrow = quadratic_row(sstar)
    rec = qrow.to_dict()
    rec.update(
        {
            "method": "quadratic_nl_guardrail",
            "family": family,
            "anchor_s": float(anchor),
            "gamma": 0.0,
            "lambda": 0.0,
            "rho": float(qrow["raw_rho"]),
            "provenance": f"reused_quadratic:{qrow['pred_tag']}",
            "pred_tag": str(qrow["pred_tag"]),
            "attained": True,
            "note": "gamma0_existing_quadratic_no_refit",
            "forward_computed": False,
            "forward_note": "reused_quadratic_no_saved_booster",
        }
    )
    dest = v1._ensure_dir(fam_dir(root, family))
    src_p = six.pred_path(V2, "quadratic", str(qrow["pred_tag"]))
    src_e = six.train_e_path(V2, "quadratic", str(qrow["pred_tag"]))
    tag = tag_for(sstar)
    if src_p.is_file():
        shutil.copy2(src_p, dest / f"pred_{tag}.parquet")
        rec["pred_tag"] = tag
    if src_e.is_file():
        shutil.copy2(src_e, dest / f"train_e_{tag}.npy")
        rec = attach_n3(rec, np.load(dest / f"train_e_{tag}.npy"), basis["phi_train"], "train")
    tau = float(cfg["anchors"][akey(anchor)]["tau"])
    rec["tau"] = tau
    rec["guardrail_ok"] = float(rec.get("N3_orth_train", rec.get("M23_train", np.nan))) <= 1.01 * tau if six._finite(rec.get("N3_orth_train", rec.get("M23_train"))) else False
    dummy = {
        "status": "ok",
        "C_train": float(qrow["C_train"]),
        "Beta_log_train": float(qrow["Beta_log_train"]),
        "mse_train": float(qrow["mse_train"]),
        "m1_train": float(qrow["m1_train"]),
        "m2_train": float(qrow["m2_train"]),
        "m3_train": float(qrow["m3_train"]),
        "M23_train": m23_norm2(float(qrow["m2_train"]), float(qrow["m3_train"])),
        "runtime_sec": 0.0,
    }
    traces = [hyb.trace_row("quadratic_nl_guardrail", sstar, 0, 0.0, float(qrow["raw_rho"]), 0.0, 0.0, tau, dummy, float(qrow["s_train"]), "gamma0_reused_quadratic")]
    write_lock(root, family, sstar, rec, traces)


def reuse_quadratic_as_qd_lambda0(*, root: Path, family: str, anchor: float, data, basis, cfg) -> None:
    """Reuse the frozen Quadratic row as the λ=0 endpoint of a QD path. Do not refit."""
    if sentinel_path(root, family, float(anchor)).is_file():
        return
    qrow = quadratic_row(anchor)
    rec = qrow.to_dict()
    rec.update(
        {
            "method": "quadratic_direct_cap",
            "family": family,
            "anchor_s": float(anchor),
            "target_s": float(anchor),
            "alpha": float(cfg["anchors"][akey(anchor)]["rho"]),
            "lambda": 0.0,
            "gamma": 0.0,
            "rho": 0.0,
            "tau": float(cfg["anchors"][akey(anchor)]["tau"]),
            "provenance": f"reused_quadratic_lambda0:{qrow['pred_tag']}",
            "attained": True,
            "note": "lambda0_existing_quadratic_no_refit",
            "forward_computed": False,
            "forward_note": "reused_quadratic_no_saved_booster",
        }
    )
    dest = v1._ensure_dir(fam_dir(root, family))
    src_p = six.pred_path(V2, "quadratic", str(qrow["pred_tag"]))
    src_e = six.train_e_path(V2, "quadratic", str(qrow["pred_tag"]))
    tag = tag_for(float(anchor))
    if src_p.is_file():
        shutil.copy2(src_p, dest / f"pred_{tag}.parquet")
        rec["pred_tag"] = tag
    else:
        rec["pred_tag"] = str(qrow["pred_tag"])
    if src_e.is_file():
        shutil.copy2(src_e, dest / f"train_e_{tag}.npy")
        rec = attach_n3(rec, np.load(dest / f"train_e_{tag}.npy"), basis["phi_train"], "train")
    dummy = {
        "status": "ok",
        "C_train": float(qrow["C_train"]),
        "Beta_log_train": float(qrow["Beta_log_train"]),
        "mse_train": float(qrow["mse_train"]),
        "m1_train": float(qrow["m1_train"]),
        "m2_train": float(qrow["m2_train"]),
        "m3_train": float(qrow["m3_train"]),
        "M23_train": float(cfg["anchors"][akey(anchor)]["tau"]),
        "runtime_sec": 0.0,
    }
    traces = [hyb.trace_row("quadratic_direct_cap", float(anchor), 0, float(rec["alpha"]), 0.0, 0.0, 0.0, float(rec["tau"]), dummy, float(qrow["s_train"]), "lambda0_reused_quadratic")]
    write_lock(root, family, float(anchor), rec, traces)


def fit_and_lock_qd_target(
    *,
    root: Path,
    family: str,
    anchor: float,
    sstar: float,
    data,
    lgbm_params,
    basis,
    cfg,
    d23,
    history: List[Tuple[float, float]],
    fits_store: Dict[float, Dict[str, Any]],
    warm_pairs: Optional[List[Tuple[float, float]]] = None,
) -> List[Tuple[float, float]]:
    alpha = float(cfg["anchors"][akey(anchor)]["rho"])
    tau = float(cfg["anchors"][akey(anchor)]["tau"])
    beta0 = float(cfg["Beta_log_train_0"])
    c0 = float(cfg["C_train_0"])
    traces: List[Dict[str, Any]] = []
    dummy = {
        "status": "ok",
        "C_train": cfg["anchors"][akey(anchor)]["C_train"],
        "Beta_log_train": cfg["anchors"][akey(anchor)]["Beta_log_train"],
        "mse_train": cfg["anchors"][akey(anchor)]["mse_train"],
        "m1_train": cfg["anchors"][akey(anchor)]["m1_train"],
        "m2_train": cfg["anchors"][akey(anchor)]["m2_train"],
        "m3_train": cfg["anchors"][akey(anchor)]["m3_train"],
        "M23_train": tau,
        "runtime_sec": 0.0,
    }
    traces.append(hyb.trace_row("quadratic_direct_cap", sstar, 0, alpha, 0.0, 0.0, 0.0, tau, dummy, float(cfg["anchors"][akey(anchor)]["s_train"]), "lambda0_anchor_reused"))
    n_fit = 0
    attempted: List[float] = []
    warm_pairs = list(warm_pairs or [])

    def _pairs() -> List[Tuple[float, float]]:
        has_self = any(p > 0 and p in fits_store for p, _ in history)
        return list(history) if has_self else list(history) + warm_pairs

    prop, how = interpolate_lambda(_pairs(), float(sstar))
    if prop is None or float(prop) <= 1e-12:
        prop, how = LAMBDA_START, "lambda_start_no_positive_history"
    lam = float(prop)
    while n_fit < 1 + MAX_CORRECTIVE:
        if any(abs(lam - prev) < 1e-10 for prev in attempted):
            break
        attempted.append(lam)
        existing = None
        for p_ex, s_ex in history:
            if p_ex > 0 and abs(p_ex - lam) < 1e-10 and p_ex in fits_store:
                existing = (p_ex, s_ex, fits_store[p_ex])
                break
        if existing is not None:
            lam, s, fit = existing
            n_fit += 1
            traces.append(hyb.trace_row("quadratic_direct_cap", sstar, n_fit, alpha, 0.0, lam, 0.0, tau, fit, s, how + "|reused_in_memory"))
            six.atomic_csv(traces_df(traces), fam_dir(root, family) / f"search_trace_s_{sstar:g}.csv")
            att, _, _ = attainment(s, sstar)
            if att:
                break
        else:
            _log("QD fit", anchor=anchor, sstar=sstar, lam=f"{lam:.6g}", how=how)
            fit = fit_followup(method="quadratic_direct_cap", data=data, lgbm_params=lgbm_params, basis=basis, d23=d23, alpha=alpha, lam=lam, rho=0.0, gamma=0.0)
            n_fit += 1
            s = six.training_s(fit["Beta_log_train"], beta0, fit["C_train"], c0)
            traces.append(hyb.trace_row("quadratic_direct_cap", sstar, n_fit, alpha, 0.0, lam, 0.0, tau, fit, s, how))
            six.atomic_csv(traces_df(traces), fam_dir(root, family) / f"search_trace_s_{sstar:g}.csv")
            if fit["status"] == "ok" and six._finite(s) and s > 0:
                history.append((lam, s))
                fits_store[lam] = fit
                att, _, _ = attainment(s, sstar)
                if att:
                    break
            else:
                break
        nxt, how = interpolate_lambda(_pairs(), float(sstar))
        if nxt is None or nxt <= 1e-12 or abs(float(nxt) - lam) < 1e-12:
            break
        lam = float(nxt)
    best_lam, best_s, best_fit = None, None, None
    for p, s in history:
        if p <= 0 or p not in fits_store:
            continue
        if best_s is None or abs(s - float(sstar)) < abs(best_s - float(sstar)):
            best_lam, best_s, best_fit = p, s, fits_store[p]
    att, match_tol, hard = attainment(best_s if best_s is not None else float("nan"), sstar)
    rec = {
        "experiment_label": EXPERIMENT_LABEL,
        "method": "quadratic_direct_cap",
        "family": family,
        "anchor_s": float(anchor),
        "target_s": float(sstar),
        "alpha": alpha,
        "lambda": float(best_lam) if best_lam is not None else float("nan"),
        "gamma": 0.0,
        "rho": 0.0,
        "tau": tau,
        "s_train": best_s if best_s is not None else float("nan"),
        "achieved_s": best_s if best_s is not None else float("nan"),
        "provenance": "new_v3_fit",
        "status": "ok" if att else "target_unattained",
        "attained": att,
        "match_tol": match_tol,
        "flag_hard_tolerance": hard,
        "n_estimators": int(lgbm_params.get("n_estimators", -1)),
        "n_train": int(data["n_train"]),
        "n_eval": int(data["n_eval"]),
        "note": "alpha_frozen_lambda_search_v3",
    }
    if att and best_fit is not None:
        rec.update(
            {
                "C_train": best_fit["C_train"],
                "Beta_log_train": best_fit["Beta_log_train"],
                "m1_train": best_fit["m1_train"],
                "m2_train": best_fit["m2_train"],
                "m3_train": best_fit["m3_train"],
                "mse_train": best_fit["mse_train"],
                "runtime_sec": best_fit["runtime_sec"],
                "peak_rss_gb": best_fit["peak_rss_gb"],
            }
        )
        rec = attach_n3(rec, best_fit["e_train"], basis["phi_train"], "train")
        tag = tag_for(sstar)
        rec["pred_tag"] = tag
        rec["heldout_computed"] = False
        save_lock_artifacts(root=root, family=family, tag=tag, data=data, fit=best_fit)
        rec = fill_eval_metrics(rec, data=data, pred=best_fit["pred_eval"], basis=basis, y_mean=float(basis["y_mean"]))
        if "pred_forward" in best_fit and np.all(np.isfinite(best_fit["pred_forward"])):
            rec = fill_forward_metrics(rec, data=data, pred_fwd=best_fit["pred_forward"], basis=basis)
        else:
            rec["forward_computed"] = False
            rec["forward_note"] = "forward_pred_not_saved_with_this_fit"
    else:
        rec["pred_tag"] = ""
        rec["heldout_computed"] = False
        rec["forward_computed"] = False
    write_lock(root, family, sstar, rec, traces)
    return history


def run_qd_anchor(args) -> None:
    root = Path(args.output_root)
    cfg = read_json(root / "FOLLOWUP_CONFIG.json")
    anchor = float(args.anchor_s)
    family = qd_family(anchor)
    if sentinel_path(root, family, QD_TARGETS_BELOW[anchor][-1]).is_file() and all(
        sentinel_path(root, family, s).is_file() for s in QD_TARGETS_BELOW[anchor]
    ):
        _log("QD anchor already complete", anchor=anchor)
        return
    params_ok = read_json(root / "PRECHECK.json")
    if str(params_ok.get("status")) != "PASS":
        raise SystemExit("Refusing QD: PRECHECK did not PASS.")
    n_jobs = v1._allocated_cpus()
    lgbm_params = v1.load_frozen_lgbm_params(Path(args.lgbm_config_json), n_jobs=n_jobs, n_estimators=args.n_estimators)
    data = load_followup_data(args)
    state = six.load_shared_state(V2, data)
    d23 = np.load(V2 / "hybrid_d23.npy")
    v1._ensure_dir(fam_dir(root, family))
    reuse_quadratic_as_qd_lambda0(root=root, family=family, anchor=anchor, data=data, basis=state["basis"], cfg=cfg)
    history: List[Tuple[float, float]] = [(0.0, float(cfg["anchors"][akey(anchor)]["s_train"]))]
    fits_store: Dict[float, Dict[str, Any]] = {}
    warm_pairs: List[Tuple[float, float]] = []
    for src_s in (0.20, 0.15, 0.10):
        tpath = V2 / "families" / "quadratic_direct_cap" / f"search_trace_s_{src_s:g}.csv"
        if not tpath.is_file():
            continue
        tr = pd.read_csv(tpath)
        for _, r in tr.iterrows():
            lam = float(r["lambda"])
            s = float(r["achieved_s"])
            if lam > 0 and six._finite(s) and s > 0:
                if np.isclose(anchor, 0.25):
                    history.append((lam, s))
                else:
                    warm_pairs.append((lam, s))
    history = list({(round(a, 12), b): (a, b) for a, b in history}.values())
    reuse = {
        (0.25, 0.20): "lock_s_0.2",
        (0.25, 0.15): "lock_s_0.15",
    }
    for sstar in QD_TARGETS_BELOW[anchor]:
        if sentinel_path(root, family, sstar).is_file():
            _log("skip completed QD target", anchor=anchor, sstar=sstar)
            continue
        key = (float(anchor), float(sstar))
        if key in reuse:
            copy_hybrid_lock(
                root=root,
                family=family,
                method="quadratic_direct_cap",
                anchor=anchor,
                sstar=sstar,
                src_tag=reuse[key],
                data=data,
                basis=state["basis"],
                cfg=cfg,
            )
            continue
        history = fit_and_lock_qd_target(
            root=root,
            family=family,
            anchor=anchor,
            sstar=sstar,
            data=data,
            lgbm_params=lgbm_params,
            basis=state["basis"],
            cfg=cfg,
            d23=d23,
            history=history,
            fits_store=fits_store,
            warm_pairs=warm_pairs,
        )


def run_qnl_target(args) -> None:
    root = Path(args.output_root)
    if str(read_json(root / "PRECHECK.json").get("status")) != "PASS":
        raise SystemExit("Refusing QNL: PRECHECK did not PASS.")
    anchor = float(args.anchor_s)
    sstar = float(args.target_s)
    family = qnl_family(anchor)
    if sentinel_path(root, family, sstar).is_file():
        _log("skip completed QNL target", anchor=anchor, sstar=sstar)
        return
    cfg = read_json(root / "FOLLOWUP_CONFIG.json")
    n_jobs = v1._allocated_cpus()
    lgbm_params = v1.load_frozen_lgbm_params(Path(args.lgbm_config_json), n_jobs=n_jobs, n_estimators=args.n_estimators)
    data = load_followup_data(args)
    state = six.load_shared_state(V2, data)
    basis = state["basis"]
    d23 = np.load(V2 / "hybrid_d23.npy")
    tau = float(cfg["anchors"][akey(anchor)]["tau"])
    beta0 = float(cfg["Beta_log_train_0"])
    c0 = float(cfg["C_train_0"])
    qrow = quadratic_row(sstar)
    n3_q = m23_norm2(float(qrow["m2_train"]), float(qrow["m3_train"]))
    v1._ensure_dir(fam_dir(root, family))
    if n3_q <= 1.01 * tau and abs(float(qrow["s_train"]) - float(sstar)) <= S_TOL_HARD:
        reuse_quadratic_as_gamma0(root=root, family=family, anchor=anchor, sstar=sstar, data=data, basis=basis, cfg=cfg)
        return
    if np.isclose(anchor, 0.25) and np.isclose(sstar, 0.20):
        copy_hybrid_lock(
            root=root,
            family=family,
            method="quadratic_nl_guardrail",
            anchor=anchor,
            sstar=sstar,
            src_tag="lock_s_0.2",
            data=data,
            basis=basis,
            cfg=cfg,
        )
        return
    traces: List[Dict[str, Any]] = []
    dummy = {
        "status": "ok",
        "C_train": float(qrow["C_train"]),
        "Beta_log_train": float(qrow["Beta_log_train"]),
        "mse_train": float(qrow["mse_train"]),
        "m1_train": float(qrow["m1_train"]),
        "m2_train": float(qrow["m2_train"]),
        "m3_train": float(qrow["m3_train"]),
        "M23_train": n3_q,
        "runtime_sec": 0.0,
    }
    traces.append(hyb.trace_row("quadratic_nl_guardrail", sstar, 0, 0.0, float(qrow["raw_rho"]), 0.0, 0.0, tau, dummy, float(qrow["s_train"]), "gamma0_existing_quadratic_no_refit"))
    skip_low = np.isclose(anchor, 0.25) and sstar in (0.15, 0.10)
    if skip_low:
        tpath = V2 / "families" / "quadratic_nl_guardrail" / f"search_trace_s_{sstar:g}.csv"
        if tpath.is_file():
            for _, r in pd.read_csv(tpath).iterrows():
                rec = r.to_dict()
                rec["heldout_computed"] = False
                rec["reason"] = str(rec.get("reason", "")) + "|imported_v2_trace_no_refit"
                traces.append(rec)
    gammas = [g for g in GAMMA_GRID if not (skip_low and g < 8.0 - 1e-12)]
    rho = float(qrow["raw_rho"])
    ok_fits: List[Dict[str, Any]] = []
    n_fit = 0
    for gamma in gammas:
        inner_hist: List[Tuple[float, float]] = []
        rho_try = rho
        for inner in range(1 + MAX_CORRECTIVE):
            _log("QNL fit", anchor=anchor, sstar=sstar, gamma=gamma, rho=f"{rho_try:.6g}")
            fit = fit_followup(method="quadratic_nl_guardrail", data=data, lgbm_params=lgbm_params, basis=basis, d23=d23, alpha=0.0, lam=0.0, rho=rho_try, gamma=gamma)
            n_fit += 1
            s = six.training_s(fit["Beta_log_train"], beta0, fit["C_train"], c0)
            traces.append(hyb.trace_row("quadratic_nl_guardrail", sstar, n_fit, 0.0, rho_try, 0.0, gamma, tau, fit, s, f"gamma={gamma:g}_inner_{inner}"))
            six.atomic_csv(traces_df(traces), fam_dir(root, family) / f"search_trace_s_{sstar:g}.csv")
            if fit["status"] != "ok" or not six._finite(s) or s <= 0:
                break
            inner_hist.append((rho_try, s))
            cand = {"fit": fit, "s": s, "rho": float(rho_try), "gamma": float(gamma), "M23": fit["M23_train"]}
            ok_fits.append(cand)
            rho = float(rho_try)
            att, _, _ = attainment(s, sstar)
            guard = float(fit["M23_train"]) <= 1.01 * tau
            if att and guard:
                break
            if inner == 0:
                if s > float(sstar) + S_TOL_PREF:
                    rho_try *= 1.5
                elif s < float(sstar) - S_TOL_PREF:
                    rho_try /= 1.5
                else:
                    break
            else:
                nxt, _ = hyb.interpolate_param(inner_hist, float(sstar))
                if nxt is None or abs(float(nxt) - rho_try) < 1e-12:
                    break
                rho_try = float(nxt)
        if any(attainment(c["s"], sstar)[0] and c["M23"] <= 1.01 * tau and abs(c["gamma"] - gamma) <= 1e-12 for c in ok_fits):
            break
    cap = 1.01 * tau
    feasible = [c for c in ok_fits if c["M23"] <= cap]
    pool = [c for c in feasible if attainment(c["s"], sstar)[0]] if feasible else []
    if pool:
        min_g = min(c["gamma"] for c in pool)
        pool = [c for c in pool if abs(c["gamma"] - min_g) <= 1e-12]
        best = min(pool, key=lambda c: abs(c["s"] - float(sstar)))
        att, match_tol, hard = True, attainment(best["s"], sstar)[1], attainment(best["s"], sstar)[2]
        attained = True
    else:
        best = min(ok_fits, key=lambda c: abs(c["s"] - float(sstar))) if ok_fits else None
        att, match_tol, hard = attainment(best["s"], sstar) if best else (False, "", False)
        attained = False
    rec = {
        "experiment_label": EXPERIMENT_LABEL,
        "method": "quadratic_nl_guardrail",
        "family": family,
        "anchor_s": float(anchor),
        "target_s": float(sstar),
        "alpha": 0.0,
        "lambda": 0.0,
        "tau": tau,
        "provenance": "new_v3_fit",
        "n_estimators": int(lgbm_params.get("n_estimators", -1)),
        "n_train": int(data["n_train"]),
        "n_eval": int(data["n_eval"]),
    }
    if best is None:
        rec.update({"attained": False, "status": "target_unattained", "s_train": float("nan"), "note": "no_stable_fit"})
        write_lock(root, family, sstar, rec, traces)
        return
    fit = best["fit"]
    rec.update(
        {
            "raw_rho": float(best["rho"]),
            "rho": float(best["rho"]),
            "gamma": float(best["gamma"]),
            "s_train": best["s"],
            "achieved_s": best["s"],
            "C_train": fit["C_train"],
            "Beta_log_train": fit["Beta_log_train"],
            "m1_train": fit["m1_train"],
            "m2_train": fit["m2_train"],
            "m3_train": fit["m3_train"],
            "mse_train": fit["mse_train"],
            "runtime_sec": fit["runtime_sec"],
            "peak_rss_gb": fit["peak_rss_gb"],
            "status": "ok" if attained else "target_unattained",
            "attained": attained,
            "match_tol": match_tol if att else "",
            "flag_hard_tolerance": hard,
            "note": "min_gamma_guardrail_v3" if attained else f"failed_att={att}_guard={best['M23']<=cap}_M23={best['M23']:.6g}",
        }
    )
    rec = attach_n3(rec, fit["e_train"], basis["phi_train"], "train")
    if attained:
        tag = tag_for(sstar)
        rec["pred_tag"] = tag
        save_lock_artifacts(root=root, family=family, tag=tag, data=data, fit=fit)
        rec = fill_eval_metrics(rec, data=data, pred=fit["pred_eval"], basis=basis, y_mean=float(basis["y_mean"]))
        if "pred_forward" in fit and np.all(np.isfinite(fit["pred_forward"])):
            rec = fill_forward_metrics(rec, data=data, pred_fwd=fit["pred_forward"], basis=basis)
        else:
            rec["forward_computed"] = False
            rec["forward_note"] = "forward_pred_unavailable"
    else:
        rec["pred_tag"] = ""
        rec["heldout_computed"] = False
        rec["forward_computed"] = False
    write_lock(root, family, sstar, rec, traces)


def run_preflight(args) -> None:
    root = v1._ensure_dir(Path(args.output_root))
    v1._ensure_dir(root / "logs")
    v1._ensure_dir(root / "manifests")
    report: Dict[str, Any] = {"experiment_label": EXPERIMENT_LABEL, "status": "FAIL", "checks": {}, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    failures: List[str] = []

    def fail(key: str, msg: str, payload: Any = None) -> None:
        failures.append(key)
        report["checks"][key] = {"pass": False, "message": msg, "detail": payload}

    def ok(key: str, payload: Any = None) -> None:
        report["checks"][key] = {"pass": True, "detail": payload}

    tests = hyb.run_hybrid_tests()
    import importlib.util

    tpath = REPO / "tests" / "test_toy_followup_v3.py"
    spec = importlib.util.spec_from_file_location(tpath.stem, tpath)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    extra = []
    n_fail = int(tests["n_fail"])
    for name in sorted(dir(mod)):
        fn = getattr(mod, name)
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            extra.append({"test": name, "pass": True})
            print(f"[followup-precheck] PASS {name}", flush=True)
        except Exception as exc:
            n_fail += 1
            extra.append({"test": name, "pass": False, "error": f"{type(exc).__name__}: {exc}"})
            print(f"[followup-precheck] FAIL {name} | {exc}", flush=True)
    tests["n_fail"] = n_fail
    tests["followup_tests"] = extra
    if n_fail:
        fail("pytest", "Follow-up/hybrid tests failed.", tests)
        write_json(root / "PRECHECK.json", report)
        raise SystemExit(1)
    ok("pytest", tests)

    hybrid_status = read_json(V2 / "HYBRID_FINAL_STATUS.json")
    if str(hybrid_status.get("status")) != "PASS":
        fail("hybrid_status", "Previous hybrid experiment is not PASS.")
        write_json(root / "PRECHECK.json", report)
        raise SystemExit(1)
    ok("hybrid_status", {"assemble_job_id": hybrid_status.get("assemble_job_id")})
    if str(read_json(V2 / "preflight_report.json").get("status")) != "PASS":
        fail("six_path_preflight", "Six-path preflight is not PASS.")
        write_json(root / "PRECHECK.json", report)
        raise SystemExit(1)

    n_jobs = v1._allocated_cpus()
    lgbm_params = v1.load_frozen_lgbm_params(Path(args.lgbm_config_json), n_jobs=n_jobs, n_estimators=args.n_estimators)
    data = load_followup_data(args)
    hashes_now = {
        "n_train": int(data["n_train"]),
        "n_eval": int(data["n_eval"]),
        "feature_count": int(data["n_features"]),
        "feature_list_sha256": six.sha_json(data["predictor_cols"]),
        "n_estimators": int(lgbm_params["n_estimators"]),
        "seed": int(lgbm_params.get("random_state", args.seed)),
        "heldout_test_mode": "pre_assessment_tail",
        "lgbm_params_sha256": lgbm_params_hash(lgbm_params),
        "n_forward": int(data.get("n_forward") or 0),
    }
    mech = read_json(V2 / "mechanism_config.json")
    old_h = mech.get("hashes") or {}
    hash_notes = []
    hash_ok = True
    for k in ("n_train", "n_eval", "feature_count", "n_estimators", "seed", "lgbm_params_sha256", "feature_list_sha256"):
        match = str(hashes_now.get(k)) == str(old_h.get(k))
        hash_notes.append(f"{k}: now={hashes_now.get(k)} old={old_h.get(k)} match={match}")
        hash_ok = hash_ok and match
    if hashes_now["n_train"] != 344607 or hashes_now["n_eval"] != 38290 or hashes_now["n_estimators"] != 994:
        hash_ok = False
    if not hash_ok:
        fail("benchmark_hashes", "Frozen setup mismatch; stopping.", hash_notes)
        write_json(root / "PRECHECK.json", report)
        raise SystemExit(1)
    ok("benchmark_hashes", hash_notes)

    reused = {}
    for method in ("current_direct", "direct_mm_k1", "quadratic"):
        mpath = V2 / "families" / method / "matched.csv"
        if not mpath.is_file():
            fail("benchmark_artifacts", f"Missing {method} matched.csv")
            write_json(root / "PRECHECK.json", report)
            raise SystemExit(1)
        reused[method] = {"matched_csv": str(mpath), "sha256": six.sha_file(mpath)}
    for sstar, tag in ((0.30, "match_s_0.3"), (0.25, "match_s_0.25")):
        p = six.pred_path(V2, "quadratic", tag)
        e = six.train_e_path(V2, "quadratic", tag)
        if not p.is_file() or not e.is_file():
            fail("quadratic_anchor_files", f"Missing Quadratic {tag}")
            write_json(root / "PRECHECK.json", report)
            raise SystemExit(1)
        reused[f"quadratic_{tag}"] = {"pred": str(p), "pred_sha256": six.sha_file(p), "train_e": str(e), "train_e_sha256": six.sha_file(e)}
    for method in ("quadratic_direct_cap", "quadratic_nl_guardrail"):
        for sstar, tag in ((0.20, "lock_s_0.2"), (0.15, "lock_s_0.15"), (0.10, "lock_s_0.1")):
            p = V2 / "families" / method / f"pred_{tag}.parquet"
            e = V2 / "families" / method / f"train_e_{tag}.npy"
            m = V2 / "families" / method / f"matched_s_{sstar:g}.csv"
            t = V2 / "families" / method / f"search_trace_s_{sstar:g}.csv"
            reused[f"first_hybrid_{method}_{tag}"] = {
                "pred": str(p) if p.is_file() else None,
                "pred_sha256": six.sha_file(p) if p.is_file() else None,
                "train_e_sha256": six.sha_file(e) if e.is_file() else None,
                "matched_sha256": six.sha_file(m) if m.is_file() else None,
                "trace_sha256": six.sha_file(t) if t.is_file() else None,
            }
    reused["hybrid_d23_sha256"] = six.sha_file(V2 / "hybrid_d23.npy")
    reused["hybrid_final_status"] = str(V2 / "HYBRID_FINAL_STATUS.json")
    reused["v1_metrics"] = str(V1 / "metrics.csv") if (V1 / "metrics.csv").is_file() else None
    if (V1 / "metrics.csv").is_file():
        reused["v1_metrics_sha256"] = six.sha_file(V1 / "metrics.csv")
    ok("reused_core", reused)

    packed = np.load(V2 / "basis.npz")
    phi_full = reconstruct_phi_full(packed["c_train"], float(packed["sigma_c"]), packed["rinv"])
    diag = orthonormality_diagnostics(phi_full)
    if diag["gram_offdiag_max"] > 1e-6 or abs(diag["phi2_dot_z"]) > 1e-6 or abs(diag["phi3_dot_z"]) > 1e-6:
        fail("basis_orthonormal", "Frozen 4-column basis failed orthonormality.", diag)
        write_json(root / "PRECHECK.json", report)
        raise SystemExit(1)
    ok("basis_orthonormal", diag)
    np.save(root / "phi_full_train.npy", phi_full)

    anchors = {}
    for sstar in (0.30, 0.25):
        row = quadratic_row(sstar)
        e = np.load(six.train_e_path(V2, "quadratic", str(row["pred_tag"])))
        m2, m3, n3, n3_rel = n3_orth_from_phi3col(e, np.asarray(packed["phi_train"]))
        anchors[akey(sstar)] = {
            "target_s": float(sstar),
            "rho": float(row["raw_rho"]),
            "s_train": float(row["s_train"]),
            "pred_tag": str(row["pred_tag"]),
            "m2_train": m2,
            "m3_train": m3,
            "tau": n3,
            "N3_orth": n3,
            "N3_rel": n3_rel,
            "C_train": float(row["C_train"]),
            "Beta_log_train": float(row["Beta_log_train"]),
            "m1_train": float(row["m1_train"]),
            "mse_train": float(row["mse_train"]),
            "R2_price": float(row["R2_price"]),
            "MAE_price": float(row["MAE_price"]),
            "Beta_log": float(row["Beta_log"]),
            "dCor_e_y": float(row["dCor_e_y"]),
            "NL_shape": float(row["NL_shape"]),
        }
    ok("quadratic_anchors", anchors)

    first_qnl_note = (
        "First QNL already used six-path QR orthonormal phi_2/phi_3 "
        "(phi[:,1], phi[:,2] of the 3-col training matrix). "
        "V3 reuses those locks and only runs missing anchors/targets/gamma>=8."
    )
    ok("first_qnl_basis", {"identical_orthogonal_basis": True, "note": first_qnl_note})

    cfg = {
        "experiment_label": EXPERIMENT_LABEL,
        "experiment": "toy_surrogate_followup_v3",
        "v2_root": str(V2),
        "v1_root": str(V1),
        "anchors": anchors,
        "Beta_log_train_0": float(mech["Beta_log_train_0"]),
        "C_train_0": float(mech["C_train_0"]),
        "hashes": hashes_now,
        "prior_hashes": old_h,
        "basis_diagnostics": diag,
        "git": git_state(REPO),
        "packages": six.package_versions_metadata(),
        "heldout_never_used_for_penalty": True,
        "n_forward": hashes_now["n_forward"],
        "s_tol_pref": S_TOL_PREF,
        "s_tol_hard": S_TOL_HARD,
        "gamma_grid": list(GAMMA_GRID),
    }
    write_json(root / "FOLLOWUP_CONFIG.json", cfg)
    write_json(root / "REUSED_ARTIFACTS.json", {"core": reused, "v1_exists": V1.is_dir(), "hybrid_status": str(hybrid_status.get("status"))})
    if failures:
        report["status"] = "FAIL"
        report["failures"] = failures
        write_json(root / "PRECHECK.json", report)
        raise SystemExit(1)
    report["status"] = "PASS"
    report["anchors"] = anchors
    report["n_forward"] = hashes_now["n_forward"]
    write_json(root / "PRECHECK.json", report)
    _log("PRECHECK PASS", tau25=anchors["0.25"]["tau"], tau30=anchors["0.30"]["tau"])


def _load_family_locks(root: Path, family: str) -> pd.DataFrame:
    rows = []
    d = fam_dir(root, family)
    if not d.is_dir():
        return pd.DataFrame()
    for p in sorted(d.glob("matched_s_*.csv")):
        rows.append(pd.read_csv(p))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def bootstrap_pair(pred_a: pd.DataFrame, pred_b: pd.DataFrame, y_train: np.ndarray, phi_eval: np.ndarray) -> Dict[str, Any]:
    dates = pd.to_datetime(pred_a["sale_date"])
    idx = _build_time_block_bootstrap_indices(dates, BOOTSTRAP_N, "M", BOOTSTRAP_SEED)
    ya = pred_a["y_true_log"].to_numpy(dtype=float)
    pa = pred_a["y_pred_log"].to_numpy(dtype=float)
    yb = pred_b["y_true_log"].to_numpy(dtype=float)
    pb = pred_b["y_pred_log"].to_numpy(dtype=float)
    sale_a = pred_a["y_true"].to_numpy(dtype=float)
    sale_b = pred_b["y_true"].to_numpy(dtype=float)
    diffs = {k: [] for k in ("R2_price", "MAE_price", "abs_Beta_log", "dCor_e_y", "NL_shape", "N3_rel")}
    for draw in idx:
        ma = v1.extract_required_metrics(compute_taxation_metrics(ya[draw], pa[draw], scale="log", y_train=y_train))
        mb = v1.extract_required_metrics(compute_taxation_metrics(yb[draw], pb[draw], scale="log", y_train=y_train))
        diffs["R2_price"].append(ma["R2_price"] - mb["R2_price"])
        diffs["MAE_price"].append(ma["MAE_price"] - mb["MAE_price"])
        diffs["abs_Beta_log"].append(abs(ma["Beta_log"]) - abs(mb["Beta_log"]))
        diffs["dCor_e_y"].append(ma["dCor_e_y"] - mb["dCor_e_y"])
        fa = pd.DataFrame({"y_true": sale_a[draw], "y_pred": np.exp(pa[draw]), "y_true_log": ya[draw], "y_pred_log": pa[draw]})
        fb = pd.DataFrame({"y_true": sale_b[draw], "y_pred": np.exp(pb[draw]), "y_true_log": yb[draw], "y_pred_log": pb[draw]})
        diffs["NL_shape"].append(six.compute_nl_shape(fa) - six.compute_nl_shape(fb))
        _a2, _a3, _n3a, n3ra = n3_orth_from_phi3col(pa[draw] - ya[draw], phi_eval[draw])
        _b2, _b3, _n3b, n3rb = n3_orth_from_phi3col(pb[draw] - yb[draw], phi_eval[draw])
        diffs["N3_rel"].append(n3ra - n3rb)
    out = {}
    for k, arr in diffs.items():
        arr = np.asarray(arr, dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            out[k] = {"mean": float("nan"), "p025": float("nan"), "p975": float("nan")}
            continue
        out[k] = {"mean": float(np.mean(arr)), "p025": float(np.quantile(arr, 0.025)), "p975": float(np.quantile(arr, 0.975))}
    return out


def run_assemble(args) -> None:
    root = Path(args.output_root)
    cfg = read_json(root / "FOLLOWUP_CONFIG.json") if (root / "FOLLOWUP_CONFIG.json").is_file() else {}
    pre = read_json(root / "PRECHECK.json") if (root / "PRECHECK.json").is_file() else {}
    graph = read_json(root / "manifests" / "RUN_MANIFEST.json") if (root / "manifests" / "RUN_MANIFEST.json").is_file() else {}
    final: Dict[str, Any] = {
        "experiment_label": EXPERIMENT_LABEL,
        "status": "FAIL",
        "precheck_job_id": graph.get("precheck_job_id"),
        "qd_array_job_id": graph.get("qd_array_job_id"),
        "qnl_array_job_id": graph.get("qnl_array_job_id"),
        "assemble_job_id": os.environ.get("SLURM_JOB_ID"),
        "git": git_state(REPO),
        "anchors": cfg.get("anchors"),
    }
    reasons: List[str] = []
    if str(pre.get("status")) != "PASS":
        reasons.append("precheck_not_pass")
    data = load_followup_data(args)
    state = six.load_shared_state(V2, data)
    basis = state["basis"]
    # Shape metrics for all reusable pred files.
    shape_rows = []
    master_v2 = pd.read_csv(V2 / "matched_correction_metrics.csv")
    hybrid_path = V2 / "matched_correction_metrics_hybrids.csv"
    hybrid_v2 = pd.read_csv(hybrid_path) if hybrid_path.is_file() else pd.DataFrame()
    for _, rec in master_v2.iterrows():
        method = str(rec["method"])
        tag = str(rec.get("pred_tag") or "")
        p = six.pred_path(V2, method, tag)
        row = rec.to_dict()
        row["provenance"] = "reused_v2"
        row["source_root"] = str(V2)
        row["family"] = method
        if p.is_file():
            pred = pd.read_parquet(p)
            e = pred["y_pred_log"].to_numpy(dtype=float) - pred["y_true_log"].to_numpy(dtype=float)
            phi_e = apply_moment_basis(pred["y_true_log"].to_numpy(dtype=float), basis)
            row = attach_n3(row, e, phi_e, "eval")
        te = six.train_e_path(V2, method, tag)
        if te.is_file():
            row = attach_n3(row, np.load(te), basis["phi_train"], "train")
        elif six._finite(rec.get("m2_train")) and six._finite(rec.get("m3_train")):
            n3 = m23_norm2(float(rec["m2_train"]), float(rec["m3_train"]))
            row["N3_orth_train"] = n3
            if six._finite(rec.get("mse_train")) and float(rec["mse_train"]) > 0:
                row["N3_rel_train"] = n3 / float(rec["mse_train"])
        shape_rows.append(row)
    if not hybrid_v2.empty:
        for _, rec in hybrid_v2.iterrows():
            method = str(rec["method"])
            if method not in ("quadratic_direct_cap", "quadratic_nl_guardrail"):
                continue
            row = rec.to_dict()
            row["provenance"] = "reused_first_hybrid"
            row["source_root"] = str(V2)
            row["family"] = f"first_hybrid_{method}"
            tag = str(rec.get("pred_tag") or "")
            p = six.pred_path(V2, method, tag) if tag and tag.lower() not in {"nan", ""} else Path()
            if p.is_file():
                pred = pd.read_parquet(p)
                e = pred["y_pred_log"].to_numpy(dtype=float) - pred["y_true_log"].to_numpy(dtype=float)
                phi_e = apply_moment_basis(pred["y_true_log"].to_numpy(dtype=float), basis)
                row = attach_n3(row, e, phi_e, "eval")
            te = six.train_e_path(V2, method, tag) if tag else Path()
            if te.is_file():
                row = attach_n3(row, np.load(te), basis["phi_train"], "train")
            shape_rows.append(row)
    expected = []
    for a, tgts in QD_TARGETS_BELOW.items():
        for s in tgts:
            expected.append((qd_family(a), float(s)))
    for a in QNL_ANCHORS:
        for s in QNL_TARGETS:
            expected.append((qnl_family(a), float(s)))
    for family, sstar in expected:
        if not sentinel_path(root, family, sstar).is_file():
            reasons.append(f"missing_sentinel:{family}:s={sstar:g}")
    new_frames = []
    for family in [qd_family(a) for a in QD_ANCHORS] + [qnl_family(a) for a in QNL_ANCHORS]:
        df = _load_family_locks(root, family)
        if not df.empty:
            new_frames.append(df)
    new = pd.concat(new_frames, ignore_index=True, sort=False) if new_frames else pd.DataFrame()
    if not new.empty:
        if "provenance" not in new.columns:
            new["provenance"] = "new_v3_fit"
        new["source_root"] = str(root)
        shape_rows.extend(new.to_dict(orient="records"))
    shape = pd.DataFrame(shape_rows)
    six.atomic_csv(shape, root / "shape_metrics_all_paths.csv")
    six.atomic_csv(new if not new.empty else pd.DataFrame(), root / "new_metrics.csv")
    core = shape.loc[shape["method"].isin(PRIMARY_METHODS)].copy() if "method" in shape.columns else shape
    if not core.empty and "provenance" in core.columns:
        core = core.loc[~core["provenance"].astype(str).str.contains("reused_first_hybrid", na=False)].copy()
    six.atomic_csv(core, root / "combined_primary_paths.csv")
    six.atomic_csv(shape, root / "combined_context_all_toy.csv")
    # Matched comparison vs Quadratic at 0.20/0.15/0.10
    cmp_rows = []
    qtab = master_v2.loc[master_v2["method"] == "quadratic"]
    for sstar in (0.20, 0.15, 0.10):
        qhit = qtab.loc[np.isclose(qtab["target_s"].astype(float), float(sstar))]
        if qhit.empty:
            continue
        q = qhit.iloc[0]
        qshape = shape.loc[(shape["method"] == "quadratic") & np.isclose(shape["target_s"].astype(float), float(sstar))] if "target_s" in shape.columns else pd.DataFrame()
        if qshape.empty:
            qshape = pd.DataFrame([{"N3_rel_eval": np.nan}])
        for _, rec in new.iterrows() if not new.empty else []:
            if not np.isclose(float(rec.get("target_s", np.nan)), float(sstar)):
                continue
            if str(rec.get("attained")).lower() not in {"true", "1"}:
                continue
            cmp_rows.append(
                {
                    "target_s": float(sstar),
                    "method": rec.get("method"),
                    "family": rec.get("family"),
                    "anchor_s": rec.get("anchor_s"),
                    "s_train": rec.get("s_train"),
                    "R2_price": rec.get("R2_price"),
                    "MAE_price": rec.get("MAE_price"),
                    "Beta_log": rec.get("Beta_log"),
                    "dCor_e_y": rec.get("dCor_e_y"),
                    "NL_shape": rec.get("NL_shape"),
                    "N3_orth_eval": rec.get("N3_orth_eval"),
                    "N3_rel_eval": rec.get("N3_rel_eval"),
                    "q_R2_price": q["R2_price"],
                    "q_MAE_price": q["MAE_price"],
                    "q_Beta_log": q["Beta_log"],
                    "q_dCor": q["dCor_e_y"],
                    "q_NL_shape": q["NL_shape"],
                    "q_N3_rel_eval": float(qshape["N3_rel_eval"].iloc[0]) if (not qshape.empty and "N3_rel_eval" in qshape.columns) else float("nan"),
                    "delta_R2": float(rec["R2_price"]) - float(q["R2_price"]) if six._finite(rec.get("R2_price")) else float("nan"),
                    "delta_MAE": float(rec["MAE_price"]) - float(q["MAE_price"]) if six._finite(rec.get("MAE_price")) else float("nan"),
                    "delta_abs_beta": abs(float(rec["Beta_log"])) - abs(float(q["Beta_log"])) if six._finite(rec.get("Beta_log")) else float("nan"),
                    "delta_N3_rel": float(rec["N3_rel_eval"]) - float(qshape["N3_rel_eval"].iloc[0]) if (six._finite(rec.get("N3_rel_eval")) and not qshape.empty and six._finite(qshape["N3_rel_eval"].iloc[0])) else float("nan"),
                    "noninferior_R2": (float(rec["R2_price"]) >= float(q["R2_price"]) - 0.002) if six._finite(rec.get("R2_price")) else False,
                    "noninferior_MAE": (float(rec["MAE_price"]) <= 1.01 * float(q["MAE_price"])) if six._finite(rec.get("MAE_price")) else False,
                    "noninferior_beta": (abs(float(rec["Beta_log"])) <= abs(float(q["Beta_log"])) + 0.005) if six._finite(rec.get("Beta_log")) else False,
                    "forward_computed": bool(rec.get("forward_computed")) if str(rec.get("forward_computed")).lower() not in {"nan", "none", ""} else False,
                    "provenance": rec.get("provenance"),
                }
            )
    cmp = pd.DataFrame(cmp_rows)
    six.atomic_csv(cmp, root / "matched_comparison.csv")
    boot_rows = []
    try:
        for _, rec in cmp.iterrows():
            fam = str(rec["family"])
            tag = tag_for(float(rec["target_s"]))
            p_new = fam_dir(root, fam) / f"pred_{tag}.parquet"
            qrow = qtab.loc[np.isclose(qtab["target_s"].astype(float), float(rec["target_s"]))].iloc[0]
            p_q = six.pred_path(V2, "quadratic", str(qrow["pred_tag"]))
            if not p_new.is_file() or not p_q.is_file():
                continue
            pa = pd.read_parquet(p_new)
            pb = pd.read_parquet(p_q)
            stats = bootstrap_pair(pa, pb, data["y_train"], apply_moment_basis(data["y_eval"], basis))
            for metric, st in stats.items():
                boot_rows.append(
                    {
                        "method": rec["method"],
                        "family": fam,
                        "anchor_s": rec["anchor_s"],
                        "target_s": rec["target_s"],
                        "metric": metric,
                        "diff_mean": st["mean"],
                        "diff_p025": st["p025"],
                        "diff_p975": st["p975"],
                        "note": "new_minus_quadratic_paired_month_block_bootstrap_n500",
                    }
                )
    except Exception as exc:
        reasons.append(f"bootstrap_exception:{type(exc).__name__}:{exc}")
        final["bootstrap_traceback"] = traceback.format_exc()
    six.atomic_csv(pd.DataFrame(boot_rows), root / "paired_bootstrap_differences.csv")
    # Fit status
    status_rows = []
    for family in [qd_family(a) for a in QD_ANCHORS] + [qnl_family(a) for a in QNL_ANCHORS]:
        for p in fam_dir(root, family).glob("DONE_s_*.json"):
            status_rows.append(read_json(p))
    six.atomic_csv(pd.DataFrame(status_rows), root / "fit_status.csv")
    boot = pd.DataFrame(boot_rows)
    verdict = {"quadratic_direct_cap": {}, "quadratic_nl_guardrail": {}}
    levels = (0.20, 0.15, 0.10)
    for method in ("quadratic_direct_cap", "quadratic_nl_guardrail"):
        flags = []
        notes = []
        for sstar in levels:
            part = cmp.loc[(cmp["method"] == method) & np.isclose(cmp["target_s"].astype(float), float(sstar))] if not cmp.empty else pd.DataFrame()
            ok_s = False
            detail = "no_attained_row"
            for _, r in part.iterrows() if not part.empty else []:
                ni = bool(r.get("noninferior_R2")) and bool(r.get("noninferior_MAE")) and bool(r.get("noninferior_beta"))
                bsub = boot.loc[
                    (boot["method"] == method)
                    & (boot["family"] == r["family"])
                    & np.isclose(boot["target_s"].astype(float), float(sstar))
                    & (boot["metric"] == "N3_rel")
                ] if not boot.empty else pd.DataFrame()
                shape_ci = False
                if not bsub.empty and six._finite(bsub["diff_p975"].iloc[0]):
                    shape_ci = float(bsub["diff_p975"].iloc[0]) < 0
                fwd = bool(r.get("forward_computed"))
                if ni and shape_ci and six._finite(r.get("R2_price")) and fwd:
                    ok_s = True
                    detail = f"pass_family={r['family']}"
                    break
                detail = f"ni={ni}_shape_ci={shape_ci}_forward={fwd}_family={r['family']}"
            flags.append(bool(ok_s))
            notes.append({"target_s": sstar, "pass": bool(ok_s), "detail": detail})
        adjacent = (flags[0] and flags[1]) or (flags[1] and flags[2])
        verdict[method] = {
            "targets": notes,
            "two_adjacent": bool(adjacent),
            "pass": bool(adjacent),
        }
    keep_quadratic = (not verdict["quadratic_direct_cap"]["pass"]) and (not verdict["quadratic_nl_guardrail"]["pass"])
    final["verdict"] = verdict
    final["keep_quadratic_surrogate"] = keep_quadratic
    final["stop_further_surrogate_redesign"] = keep_quadratic
    man_src = root / "manifests" / "RUN_MANIFEST.json"
    if man_src.is_file():
        shutil.copy2(man_src, root / "RUN_MANIFEST.json")
    fig_paths: Dict[str, str] = {}
    try:
        fig_paths = plot_followup(root, core, new, cfg)
    except Exception as exc:
        reasons.append(f"figure_error:{type(exc).__name__}:{exc}")
        final["figure_traceback"] = traceback.format_exc()
    final["reasons"] = reasons
    final["status"] = "PASS" if not reasons else "FAIL"
    final["figure_paths"] = {
        "unified": fig_paths,
        "main": fig_paths.get("main_paths", str(root / "figures" / "main_paths.pdf")),
        "context": fig_paths.get("context_all_toy", str(root / "figures" / "context_all_toy.pdf")),
        "matched": fig_paths.get("matched_vs_quadratic", str(root / "figures" / "matched_vs_quadratic.pdf")),
        "ratio": fig_paths.get("ratio_shape", str(root / "figures" / "ratio_shape.pdf")),
    }
    write_json(root / "FINAL_STATUS.json", final)
    if reasons:
        _log("assemble FAIL", reasons=",".join(reasons))
        raise SystemExit(1)
    _log("assemble PASS")


def plot_followup(root: Path, core: pd.DataFrame, new: pd.DataFrame, cfg: dict) -> Dict[str, str]:
    """Rebuild every comparable figure with all models overlaid. `core` may be the full shape table."""
    import toy_followup_v3_plots as plots

    shape_path = Path(root) / "shape_metrics_all_paths.csv"
    if shape_path.is_file():
        shape = pd.read_csv(shape_path)
    else:
        shape = core
    paths = plots.render_unified_figures(Path(root), shape)
    _log("unified figures written", n=len(paths))
    return paths


def run_plot(args) -> None:
    root = Path(args.output_root)
    shape_path = root / "shape_metrics_all_paths.csv"
    if not shape_path.is_file():
        raise SystemExit(f"Missing {shape_path}; run assemble first.")
    paths = plot_followup(root, pd.read_csv(shape_path), pd.DataFrame(), {})
    status_path = root / "FINAL_STATUS.json"
    if status_path.is_file():
        final = read_json(status_path)
        final["figure_paths"] = {
            "unified": paths,
            "main": paths.get("main_paths", str(root / "figures" / "main_paths.pdf")),
            "context": paths.get("context_all_toy", str(root / "figures" / "context_all_toy.pdf")),
            "matched": paths.get("matched_vs_quadratic", str(root / "figures" / "matched_vs_quadratic.pdf")),
            "ratio": paths.get("ratio_shape", str(root / "figures" / "ratio_shape.pdf")),
        }
        write_json(status_path, final)
    write_json(root / "UNIFIED_FIGURES.json", {"figure_paths": paths, "n": len(paths)})


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EXPERIMENTAL / TOY follow-up V3.")
    p.add_argument("--mode", required=True, choices=["preflight", "fit-qd-anchor", "fit-qnl-target", "assemble", "plot"])
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    p.add_argument("--data-path", default=str(v1.DEFAULT_DATA))
    p.add_argument("--params", default=str(v1.DEFAULT_PARAMS))
    p.add_argument("--lgbm-config-json", default=str(v1.DEFAULT_LGBM_CONFIG))
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--anchor-s", type=float, default=None)
    p.add_argument("--target-s", type=float, default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    _log("start", mode=args.mode, label=EXPERIMENT_LABEL)
    if args.mode == "preflight":
        run_preflight(args)
    elif args.mode == "fit-qd-anchor":
        if args.anchor_s is None:
            raise SystemExit("--anchor-s required")
        run_qd_anchor(args)
    elif args.mode == "fit-qnl-target":
        if args.anchor_s is None or args.target_s is None:
            raise SystemExit("--anchor-s and --target-s required")
        run_qnl_target(args)
    elif args.mode == "plot":
        run_plot(args)
    else:
        run_assemble(args)
    _log("done", mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
