#!/usr/bin/env python3
"""EXPERIMENTAL / TOY hybrid continuation: Quadratic+Direct and Quadratic+NL guardrail.

Cluster-resident. Reuses locked current_direct / direct_mm_k1 / quadratic.
Writes only under output/toy_surrogate_ablation_v2/.
"""

from __future__ import annotations

import argparse
import json
import os
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

import toy_surrogate_ablation as v1
import toy_mechanism_selection as six
from canonical_experiment import git_state, lgbm_params_hash, read_json, write_json
from soft_constrained_models.boosting_models import (
    canonical_direct_scaled_grad_hess,
    canonical_surrogate_scaled_grad_hess,
)
from soft_constrained_models.toy_hybrid_objectives import (
    HYBRID_METHODS,
    HYBRID_TITLES,
    QD_GRAD,
    QD_HESS_EXACT,
    QD_HESS_SUPPLIED,
    QNL_CURVATURE,
    QNL_GRAD,
    QNL_HESS_EXACT,
    QNL_HESS_SUPPLIED,
    ToyHybridLGB,
    guardrail_d23,
    m23_norm2,
    majorizer_gap,
    nl_moment_pair,
    quadratic_direct_cap_exact_hessian,
    quadratic_direct_cap_scaled_grad_hess,
    quadratic_nl_guardrail_scaled_grad_hess,
)
from soft_constrained_models.toy_mechanism_objectives import EXPERIMENT_LABEL

DEFAULT_OUTPUT = REPO / "output" / "toy_surrogate_ablation_v2"
HYBRID_TARGETS: Tuple[float, ...] = (0.20, 0.15, 0.10)
CONTEXT_S: Tuple[float, ...] = (0.30, 0.25, 0.20, 0.15, 0.10)
BENCHMARKS = ("current_direct", "direct_mm_k1", "quadratic")
ANCHOR_S = 0.25
S_TOL_PREF = 0.005
S_TOL_HARD = 0.010
M23_TOL = 1.01
MAX_FITS = 12
MAX_CORRECTIVE = 2
LAMBDA_START = 0.5
LAMBDA_GROW = 1.7
GAMMA_START = 0.5
GAMMA_GROW = 2.0
RHO_GROW = 1.5
TRACE_COLS = (
    "method",
    "target_s",
    "attempt",
    "alpha",
    "rho",
    "lambda",
    "gamma",
    "achieved_s",
    "C_train",
    "beta_train",
    "m1_train",
    "m2_train",
    "m3_train",
    "M23_train",
    "tau",
    "mse_train",
    "runtime_sec",
    "status",
    "reason",
    "heldout_computed",
)


def _log(msg: str, **fields: Any) -> None:
    suffix = ""
    if fields:
        suffix = " | " + " | ".join(f"{k}={v}" for k, v in fields.items())
    print(f"[toy_hybrid_selection] {msg}{suffix}", flush=True)


def hybrid_dir(root: Path, method: str) -> Path:
    return root / "families" / method


def target_tag(sstar: float) -> str:
    return f"lock_s_{sstar:g}"


def sentinel_path(root: Path, method: str, sstar: float) -> Path:
    return hybrid_dir(root, method) / f"DONE_s_{sstar:g}.json"


def load_matched(root: Path, method: str) -> pd.DataFrame:
    return pd.read_csv(hybrid_dir(root, method) / "matched.csv")


def quadratic_row(root: Path, sstar: float) -> pd.Series:
    df = load_matched(root, "quadratic")
    hit = df.loc[np.isclose(df["target_s"].astype(float), float(sstar))]
    if hit.empty:
        raise RuntimeError(f"Missing quadratic matched row at target_s={sstar}")
    return hit.iloc[0]


def run_hybrid_tests() -> Dict[str, Any]:
    import importlib.util

    paths = [
        REPO / "tests" / "test_toy_hybrid_objectives.py",
        REPO / "tests" / "test_canonical_objectives.py",
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
                print(f"[hybrid-preflight-test] PASS {path.stem}.{name}", flush=True)
            except Exception as exc:
                n_fail += 1
                err = f"{type(exc).__name__}: {exc}"
                results.append({"test": f"{path.stem}.{name}", "pass": False, "error": err})
                print(f"[hybrid-preflight-test] FAIL {path.stem}.{name} | {err}", flush=True)
    return {
        "returncode": 0 if n_fail == 0 else 1,
        "n_ok": int(sum(1 for r in results if r["pass"])),
        "n_fail": int(n_fail),
        "failed": [r for r in results if not r["pass"]],
        "results": results,
    }


def interpolate_param(pairs: List[Tuple[float, float]], sstar: float) -> Tuple[Optional[float], str]:
    ok = [(p, s) for p, s in pairs if six._finite(p) and six._finite(s) and float(p) > 0 and float(s) > 0]
    if not ok:
        return None, "no_positive_pairs"
    s = np.asarray([x[1] for x in ok], dtype=float)
    p = np.asarray([x[0] for x in ok], dtype=float)
    nearest = int(np.argmin(np.abs(s - float(sstar))))
    if abs(s[nearest] - float(sstar)) <= S_TOL_PREF:
        return float(p[nearest]), "existing_within_pref"
    above = np.flatnonzero(s >= float(sstar))
    below = np.flatnonzero(s <= float(sstar))
    if above.size and below.size:
        i_hi = above[np.argmin(s[above] - float(sstar))]
        i_lo = below[np.argmin(float(sstar) - s[below])]
        p_hi, p_lo = float(p[i_hi]), float(p[i_lo])
        s_hi, s_lo = float(s[i_hi]), float(s[i_lo])
        if abs(p_hi - p_lo) < 1e-18:
            return float(p[nearest]), "nearest_degenerate"
        t = (np.log(float(sstar)) - np.log(s_hi)) / (np.log(s_lo) - np.log(s_hi) + 1e-18)
        logp = np.log(max(p_hi, 1e-18)) + t * (np.log(max(p_lo, 1e-18)) - np.log(max(p_hi, 1e-18)))
        return float(np.exp(logp)), "log_param_bracket"
    return float(p[nearest]), "nearest_unbracketed"


def fit_hybrid(
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
    t0 = time.perf_counter()
    rss0 = v1._peak_rss_gb()
    status = "ok"
    error = ""
    pred_train = np.full(data["y_train"].shape, np.nan)
    pred_eval = np.full(data["y_eval"].shape, np.nan)
    model = None
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
    phi = basis["phi_train"]
    if status == "ok":
        c_now = float(np.mean(e_train * c_train))
        var_c = float(np.mean(np.square(c_train)))
        beta_tr = float(c_now / var_c) if var_c > 0 else float("nan")
        mse = float(np.mean(np.square(e_train)))
        pack = six.moment_pack(e_train, phi)
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


def trace_row(method: str, sstar: float, attempt: int, alpha: float, rho: float, lam: float, gamma: float, tau: float, fit: Dict[str, Any], s: float, reason: str) -> Dict[str, Any]:
    return {
        "method": method,
        "target_s": float(sstar),
        "attempt": int(attempt),
        "alpha": float(alpha),
        "rho": float(rho),
        "lambda": float(lam),
        "gamma": float(gamma),
        "achieved_s": float(s) if six._finite(s) else float("nan"),
        "C_train": fit.get("C_train", float("nan")),
        "beta_train": fit.get("Beta_log_train", float("nan")),
        "m1_train": fit.get("m1_train", float("nan")),
        "m2_train": fit.get("m2_train", float("nan")),
        "m3_train": fit.get("m3_train", float("nan")),
        "M23_train": fit.get("M23_train", float("nan")),
        "tau": float(tau),
        "mse_train": fit.get("mse_train", float("nan")),
        "runtime_sec": fit.get("runtime_sec", 0.0),
        "status": fit.get("status", ""),
        "reason": reason,
        "heldout_computed": False,
    }


def attainment(s: float, sstar: float) -> Tuple[bool, str, bool]:
    if not six._finite(s) or float(s) <= 0:
        return False, "", False
    err = abs(float(s) - float(sstar))
    if err <= S_TOL_PREF:
        return True, "pref_0.005", False
    if err <= S_TOL_HARD:
        return True, "hard_0.010", True
    return False, "", False


def lock_and_eval(
    *,
    root: Path,
    method: str,
    sstar: float,
    data: Dict[str, Any],
    basis: Dict[str, Any],
    cfg: dict,
    chosen: Dict[str, Any],
    fit: Optional[Dict[str, Any]],
    traces: List[Dict[str, Any]],
    reused_pred_tag: Optional[str] = None,
    reused_method: str = "quadratic",
) -> Dict[str, Any]:
    fam = v1._ensure_dir(hybrid_dir(root, method))
    tag = target_tag(sstar)
    rec = dict(chosen)
    rec["experiment_label"] = EXPERIMENT_LABEL
    rec["method"] = method
    rec["target_s"] = float(sstar)
    rec["heldout_computed"] = False
    rec["n_train"] = int(data["n_train"])
    rec["n_eval"] = int(data["n_eval"])
    if rec.get("attained") and fit is not None and fit.get("status") == "ok":
        rec["pred_tag"] = tag
        six.save_fit_artifacts(root, method, tag, data, fit)
        rec = six.fill_heldout(rec, data=data, pred_eval=fit["pred_eval"], basis=basis, y_mean_train=float(basis["y_mean"]))
    elif rec.get("attained") and reused_pred_tag:
        src_p = six.pred_path(root, reused_method, reused_pred_tag)
        src_e = six.train_e_path(root, reused_method, reused_pred_tag)
        rec["pred_tag"] = reused_pred_tag
        rec["note"] = rec.get("note", "") + "|reused_quadratic_artifact"
        if src_p.is_file():
            pred = pd.read_parquet(src_p)
            rec = six.fill_heldout(
                rec,
                data=data,
                pred_eval=pred["y_pred_log"].to_numpy(dtype=float),
                basis=basis,
                y_mean_train=float(basis["y_mean"]),
            )
        rec["reused_train_e"] = str(src_e)
    else:
        rec["attained"] = False
        rec["status"] = "target_unattained"
        rec["pred_tag"] = ""
    six.atomic_csv(pd.DataFrame(traces)[list(TRACE_COLS)], fam / f"search_trace_s_{sstar:g}.csv")
    six.atomic_csv(pd.DataFrame([rec]), fam / f"matched_s_{sstar:g}.csv")
    sentinel = {
        "method": method,
        "target_s": float(sstar),
        "status": "DONE",
        "attained": bool(rec.get("attained")),
        "achieved_s": rec.get("s_train", rec.get("achieved_s")),
        "alpha": rec.get("alpha"),
        "lambda": rec.get("lambda"),
        "rho": rec.get("rho", rec.get("raw_rho")),
        "gamma": rec.get("gamma"),
        "M23_train": rec.get("M23_train"),
        "tau": rec.get("tau"),
        "n_attempts": int(len(traces)),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_job_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "pred_tag": rec.get("pred_tag", ""),
    }
    write_json(sentinel_path(root, method, sstar), sentinel)
    _log("target locked", method=method, sstar=sstar, attained=rec.get("attained"), s=rec.get("s_train"))
    return rec


def search_qd(root: Path, sstar: float, data, lgbm_params, basis, cfg: dict, d23: np.ndarray) -> None:
    method = "quadratic_direct_cap"
    alpha = float(cfg["alpha_anchor"])
    tau = float(cfg["tau"])
    beta0 = float(cfg["Beta_log_train_0"])
    c0 = float(cfg["C_train_0"])
    q25 = cfg["quadratic_anchor"]
    traces: List[Dict[str, Any]] = []
    history: List[Tuple[float, float]] = []
    fits_store: Dict[float, Dict[str, Any]] = {}
    dummy = {
        "status": "ok",
        "C_train": q25["C_train"],
        "Beta_log_train": q25["Beta_log_train"],
        "mse_train": q25["mse_train"],
        "m1_train": q25["m1_train"],
        "m2_train": q25["m2_train"],
        "m3_train": q25["m3_train"],
        "M23_train": q25["M23_train"],
        "runtime_sec": 0.0,
    }
    s0 = float(q25["s_train"])
    traces.append(trace_row(method, sstar, 0, alpha, 0.0, 0.0, 0.0, tau, dummy, s0, "lambda0_reused_quadratic_anchor_no_fit"))
    history.append((0.0, s0))
    n_fit = 0
    lam = LAMBDA_START
    last_fit = None
    last_s = s0
    nonmonotone = False
    prev_s = s0
    while n_fit < MAX_FITS - MAX_CORRECTIVE:
        _log("QD fit", sstar=sstar, lam=f"{lam:.6g}", i=n_fit)
        fit = fit_hybrid(method=method, data=data, lgbm_params=lgbm_params, basis=basis, d23=d23, alpha=alpha, lam=lam, rho=0.0, gamma=0.0)
        n_fit += 1
        s = six.training_s(fit["Beta_log_train"], beta0, fit["C_train"], c0)
        reason = "geometric_lambda_increase"
        traces.append(trace_row(method, sstar, n_fit, alpha, 0.0, lam, 0.0, tau, fit, s, reason))
        six.atomic_csv(pd.DataFrame(traces)[list(TRACE_COLS)], hybrid_dir(root, method) / f"search_trace_s_{sstar:g}.csv")
        if fit["status"] == "ok" and six._finite(s) and s > 0:
            if six._finite(prev_s) and s > prev_s + 0.02:
                nonmonotone = True
            history.append((lam, s))
            fits_store[lam] = fit
            last_fit, last_s = fit, s
            prev_s = s
            if s <= float(sstar) + S_TOL_PREF:
                break
        else:
            break
        lam *= LAMBDA_GROW
        if lam > 1e8:
            break
    n_corr = 0
    while n_corr < MAX_CORRECTIVE and n_fit < MAX_FITS:
        prop, how = interpolate_param([(p, s) for p, s in history if p > 0], float(sstar))
        if prop is None:
            break
        if last_fit is not None and abs(float(prop) - float(list(fits_store.keys())[-1])) < 1e-12:
            break
        _log("QD corrective", sstar=sstar, lam=f"{float(prop):.6g}", how=how)
        fit = fit_hybrid(method=method, data=data, lgbm_params=lgbm_params, basis=basis, d23=d23, alpha=alpha, lam=float(prop), rho=0.0, gamma=0.0)
        n_fit += 1
        n_corr += 1
        s = six.training_s(fit["Beta_log_train"], beta0, fit["C_train"], c0)
        traces.append(trace_row(method, sstar, n_fit, alpha, 0.0, float(prop), 0.0, tau, fit, s, f"corrective_{how}"))
        six.atomic_csv(pd.DataFrame(traces)[list(TRACE_COLS)], hybrid_dir(root, method) / f"search_trace_s_{sstar:g}.csv")
        if fit["status"] == "ok" and six._finite(s) and s > 0:
            history.append((float(prop), s))
            fits_store[float(prop)] = fit
            last_fit, last_s = fit, s
    best_lam, best_s, best_fit = None, None, None
    for p, s in history:
        if p <= 0:
            continue
        if best_s is None or abs(s - float(sstar)) < abs(best_s - float(sstar)):
            best_lam, best_s, best_fit = p, s, fits_store.get(p)
    attained, match_tol, hard = attainment(best_s if best_s is not None else float("nan"), sstar)
    chosen = {
        "raw_rho": float("nan"),
        "rho": 0.0,
        "alpha": alpha,
        "lambda": float(best_lam) if best_lam is not None else float("nan"),
        "gamma": 0.0,
        "tau": tau,
        "s_train": best_s if best_s is not None else float("nan"),
        "achieved_s": best_s if best_s is not None else float("nan"),
        "C_train": best_fit["C_train"] if best_fit else float("nan"),
        "Beta_log_train": best_fit["Beta_log_train"] if best_fit else float("nan"),
        "m1_train": best_fit["m1_train"] if best_fit else float("nan"),
        "m2_train": best_fit["m2_train"] if best_fit else float("nan"),
        "m3_train": best_fit["m3_train"] if best_fit else float("nan"),
        "M23_train": best_fit["M23_train"] if best_fit else float("nan"),
        "mse_train": best_fit["mse_train"] if best_fit else float("nan"),
        "runtime_sec": best_fit["runtime_sec"] if best_fit else 0.0,
        "peak_rss_gb": best_fit["peak_rss_gb"] if best_fit else float("nan"),
        "status": "ok" if attained else "target_unattained",
        "attained": attained,
        "match_tol": match_tol,
        "flag_hard_tolerance": hard,
        "nonmonotone_path": nonmonotone,
        "note": "alpha_frozen_lambda_search",
        "n_estimators": int(lgbm_params.get("n_estimators", -1)),
    }
    lock_and_eval(root=root, method=method, sstar=sstar, data=data, basis=basis, cfg=cfg, chosen=chosen, fit=best_fit, traces=traces)


def search_qnl(root: Path, sstar: float, data, lgbm_params, basis, cfg: dict, d23: np.ndarray) -> None:
    method = "quadratic_nl_guardrail"
    tau = float(cfg["tau"])
    beta0 = float(cfg["Beta_log_train_0"])
    c0 = float(cfg["C_train_0"])
    qrow = quadratic_row(root, sstar)
    rho_q = float(qrow["raw_rho"])
    m2q, m3q = float(qrow["m2_train"]), float(qrow["m3_train"])
    m23_q = m23_norm2(m2q, m3q)
    traces: List[Dict[str, Any]] = []
    dummy = {
        "status": "ok",
        "C_train": float(qrow["C_train"]),
        "Beta_log_train": float(qrow["Beta_log_train"]),
        "mse_train": float(qrow["mse_train"]) if "mse_train" in qrow and six._finite(qrow["mse_train"]) else float("nan"),
        "m1_train": float(qrow["m1_train"]),
        "m2_train": m2q,
        "m3_train": m3q,
        "M23_train": m23_q,
        "runtime_sec": 0.0,
    }
    s_q = float(qrow["s_train"])
    traces.append(trace_row(method, sstar, 0, 0.0, rho_q, 0.0, 0.0, tau, dummy, s_q, "gamma0_existing_quadratic_no_refit"))
    if m23_q <= M23_TOL * tau and abs(s_q - float(sstar)) <= S_TOL_HARD:
        chosen = {
            "raw_rho": rho_q,
            "rho": rho_q,
            "alpha": 0.0,
            "lambda": 0.0,
            "gamma": 0.0,
            "tau": tau,
            "s_train": s_q,
            "achieved_s": s_q,
            "C_train": float(qrow["C_train"]),
            "Beta_log_train": float(qrow["Beta_log_train"]),
            "m1_train": float(qrow["m1_train"]),
            "m2_train": m2q,
            "m3_train": m3q,
            "M23_train": m23_q,
            "mse_train": dummy["mse_train"],
            "status": "ok",
            "attained": True,
            "match_tol": "pref_0.005" if abs(s_q - float(sstar)) <= S_TOL_PREF else "hard_0.010",
            "flag_hard_tolerance": abs(s_q - float(sstar)) > S_TOL_PREF,
            "note": "gamma0_already_satisfies_guardrail_reused_quadratic",
            "n_estimators": int(lgbm_params.get("n_estimators", -1)),
        }
        lock_and_eval(
            root=root,
            method=method,
            sstar=sstar,
            data=data,
            basis=basis,
            cfg=cfg,
            chosen=chosen,
            fit=None,
            traces=traces,
            reused_pred_tag=str(qrow["pred_tag"]),
        )
        return
    n_fit = 0
    gamma = GAMMA_START
    rho = rho_q
    ok_fits: List[Dict[str, Any]] = []
    last_over_gamma: Optional[float] = None

    def _record(fit: Dict[str, Any], rho_try: float, gamma_try: float, s: float) -> Dict[str, Any]:
        cand = {"fit": fit, "s": s, "rho": float(rho_try), "gamma": float(gamma_try), "M23": fit["M23_train"]}
        ok_fits.append(cand)
        return cand

    def _pick_best() -> Optional[Dict[str, Any]]:
        if not ok_fits:
            return None
        cap = M23_TOL * tau
        feasible = [x for x in ok_fits if six._finite(x["M23"]) and x["M23"] <= cap]
        pool = feasible if feasible else ok_fits
        att_pool = [x for x in pool if attainment(x["s"], sstar)[0]]
        if att_pool:
            pool = att_pool
        if feasible:
            min_g = min(float(x["gamma"]) for x in pool)
            pool = [x for x in pool if abs(float(x["gamma"]) - min_g) <= 1e-12]
        return min(pool, key=lambda x: abs(float(x["s"]) - float(sstar)))

    while n_fit < MAX_FITS:
        inner_hist: List[Tuple[float, float]] = []
        rho_try = rho
        guarded_this_gamma = False
        for inner in range(1 + MAX_CORRECTIVE):
            if n_fit >= MAX_FITS:
                break
            _log("QNL fit", sstar=sstar, gamma=f"{gamma:.6g}", rho=f"{rho_try:.6g}", i=n_fit)
            fit = fit_hybrid(method=method, data=data, lgbm_params=lgbm_params, basis=basis, d23=d23, alpha=0.0, lam=0.0, rho=rho_try, gamma=gamma)
            n_fit += 1
            s = six.training_s(fit["Beta_log_train"], beta0, fit["C_train"], c0)
            traces.append(trace_row(method, sstar, n_fit, 0.0, rho_try, 0.0, gamma, tau, fit, s, f"gamma={gamma:.4g}_inner_{inner}"))
            six.atomic_csv(pd.DataFrame(traces)[list(TRACE_COLS)], hybrid_dir(root, method) / f"search_trace_s_{sstar:g}.csv")
            if fit["status"] != "ok" or not six._finite(s) or s <= 0:
                break
            inner_hist.append((rho_try, s))
            cand = _record(fit, rho_try, gamma, s)
            rho = float(rho_try)
            att, _, _ = attainment(s, sstar)
            guard = cand["M23"] <= M23_TOL * tau
            if guard:
                guarded_this_gamma = True
            if att and guard:
                break
            if inner == 0:
                if s > float(sstar) + S_TOL_PREF:
                    rho_try *= RHO_GROW
                elif s < float(sstar) - S_TOL_PREF:
                    rho_try /= RHO_GROW
                else:
                    break
            else:
                prop, _how = interpolate_param(inner_hist, float(sstar))
                if prop is None or abs(prop - rho_try) < 1e-12:
                    break
                rho_try = float(prop)
        if guarded_this_gamma:
            if last_over_gamma is not None and last_over_gamma > 0 and n_fit < MAX_FITS:
                g_star = float(np.sqrt(last_over_gamma * gamma))
                if abs(g_star - gamma) / max(gamma, 1e-12) > 0.05:
                    _log("QNL interpolate gamma", g_star=f"{g_star:.6g}")
                    fit = fit_hybrid(method=method, data=data, lgbm_params=lgbm_params, basis=basis, d23=d23, alpha=0.0, lam=0.0, rho=rho, gamma=g_star)
                    n_fit += 1
                    s = six.training_s(fit["Beta_log_train"], beta0, fit["C_train"], c0)
                    traces.append(trace_row(method, sstar, n_fit, 0.0, rho, 0.0, g_star, tau, fit, s, "interpolated_gamma"))
                    if fit["status"] == "ok" and six._finite(s) and s > 0:
                        _record(fit, rho, g_star, s)
            while n_fit < MAX_FITS:
                if any(attainment(x["s"], sstar)[0] and x["M23"] <= M23_TOL * tau and abs(x["gamma"] - gamma) <= 1e-12 for x in ok_fits):
                    break
                prop, how = interpolate_param(inner_hist, float(sstar))
                if prop is None or (inner_hist and abs(float(prop) - float(inner_hist[-1][0])) < 1e-12):
                    break
                _log("QNL extra rho at min gamma", sstar=sstar, gamma=f"{gamma:.6g}", rho=f"{float(prop):.6g}", how=how)
                fit = fit_hybrid(method=method, data=data, lgbm_params=lgbm_params, basis=basis, d23=d23, alpha=0.0, lam=0.0, rho=float(prop), gamma=gamma)
                n_fit += 1
                s = six.training_s(fit["Beta_log_train"], beta0, fit["C_train"], c0)
                traces.append(trace_row(method, sstar, n_fit, 0.0, float(prop), 0.0, gamma, tau, fit, s, f"extra_rho_{how}"))
                six.atomic_csv(pd.DataFrame(traces)[list(TRACE_COLS)], hybrid_dir(root, method) / f"search_trace_s_{sstar:g}.csv")
                if fit["status"] != "ok" or not six._finite(s) or s <= 0:
                    break
                inner_hist.append((float(prop), s))
                _record(fit, float(prop), gamma, s)
            break
        last_over_gamma = gamma
        gamma *= GAMMA_GROW
        if gamma > 1e6:
            break
    best = _pick_best()
    if best is None:
        chosen = {"attained": False, "status": "target_unattained", "tau": tau, "gamma": float("nan"), "lambda": 0.0, "alpha": 0.0, "s_train": float("nan"), "achieved_s": float("nan"), "note": "no_stable_fit"}
        lock_and_eval(root=root, method=method, sstar=sstar, data=data, basis=basis, cfg=cfg, chosen=chosen, fit=None, traces=traces)
        return
    s = best["s"]
    att, match_tol, hard = attainment(s, sstar)
    guard = best["M23"] <= M23_TOL * tau
    attained = bool(att and guard)
    fit = best["fit"]
    chosen = {
        "raw_rho": float(best["rho"]),
        "rho": float(best["rho"]),
        "alpha": 0.0,
        "lambda": 0.0,
        "gamma": float(best["gamma"]),
        "tau": tau,
        "s_train": s,
        "achieved_s": s,
        "C_train": fit["C_train"],
        "Beta_log_train": fit["Beta_log_train"],
        "m1_train": fit["m1_train"],
        "m2_train": fit["m2_train"],
        "m3_train": fit["m3_train"],
        "M23_train": fit["M23_train"],
        "mse_train": fit["mse_train"],
        "runtime_sec": fit["runtime_sec"],
        "peak_rss_gb": fit["peak_rss_gb"],
        "status": "ok" if attained else "target_unattained",
        "attained": attained,
        "match_tol": match_tol if att else "",
        "flag_hard_tolerance": hard,
        "note": "min_gamma_guardrail" if attained else f"failed_att={att}_guard={guard}_M23={best['M23']:.6g}",
        "n_estimators": int(lgbm_params.get("n_estimators", -1)),
    }
    lock_and_eval(root=root, method=method, sstar=sstar, data=data, basis=basis, cfg=cfg, chosen=chosen, fit=fit if attained else None, traces=traces)


def run_preflight(args) -> None:
    root = v1._ensure_dir(Path(args.output_root))
    v1._ensure_dir(root / "logs")
    v1._ensure_dir(root / "families")
    report: Dict[str, Any] = {"experiment_label": EXPERIMENT_LABEL, "status": "FAIL", "checks": {}, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
    failures: List[str] = []

    def fail(key: str, msg: str, payload: Any = None) -> None:
        failures.append(key)
        report["checks"][key] = {"pass": False, "message": msg, "detail": payload}

    def ok(key: str, payload: Any = None) -> None:
        report["checks"][key] = {"pass": True, "detail": payload}

    tests = run_hybrid_tests()
    if int(tests["returncode"]) != 0:
        fail("pytest", "Hybrid/canonical objective tests failed.", tests)
        write_json(root / "hybrid_preflight_report.json", report)
        raise SystemExit(1)
    ok("pytest", tests)

    if str(read_json(root / "preflight_report.json").get("status")) != "PASS":
        fail("six_path_preflight", "Prior six-path preflight is not PASS.")
        write_json(root / "hybrid_preflight_report.json", report)
        raise SystemExit(1)

    params = v1.load_params(Path(args.params))
    n_jobs = v1._allocated_cpus()
    lgbm_params = v1.load_frozen_lgbm_params(Path(args.lgbm_config_json), n_jobs=n_jobs, n_estimators=args.n_estimators)
    data = v1.load_experiment_data(data_path=Path(args.data_path), params=params, sample_frac=args.sample_frac, seed=int(args.seed))
    _log("data ready", n_train=data["n_train"], n_eval=data["n_eval"])
    hashes_now = {
        "n_train": int(data["n_train"]),
        "n_eval": int(data["n_eval"]),
        "feature_count": int(data["n_features"]),
        "feature_list_sha256": six.sha_json(data["predictor_cols"]),
        "n_estimators": int(lgbm_params["n_estimators"]),
        "seed": int(lgbm_params.get("random_state", args.seed)),
        "heldout_test_mode": "pre_assessment_tail",
        "lgbm_params_sha256": lgbm_params_hash(lgbm_params),
    }
    mech = read_json(root / "mechanism_config.json")
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
        fail("benchmark_hashes", "Configuration/hash mismatch; refusing to reuse benchmarks.", hash_notes)
        write_json(root / "hybrid_preflight_report.json", report)
        raise SystemExit(1)
    ok("benchmark_hashes", hash_notes)

    for method in BENCHMARKS:
        mpath = hybrid_dir(root, method) / "matched.csv"
        dpath = hybrid_dir(root, method) / "DONE.json"
        if not mpath.is_file() or not dpath.is_file():
            fail("benchmark_artifacts", f"Missing locked artifacts for {method}")
            write_json(root / "hybrid_preflight_report.json", report)
            raise SystemExit(1)
    ok("benchmark_artifacts", True)

    q25 = quadratic_row(root, ANCHOR_S)
    pred = six.pred_path(root, "quadratic", str(q25["pred_tag"]))
    tre = six.train_e_path(root, "quadratic", str(q25["pred_tag"]))
    if not pred.is_file() or not tre.is_file():
        fail("quadratic_anchor_files", "Quadratic s=0.25 pred/train_e missing")
        write_json(root / "hybrid_preflight_report.json", report)
        raise SystemExit(1)
    e_anchor = np.load(tre)
    state = six.load_shared_state(root, data)
    basis = state["basis"]
    if int(e_anchor.size) != int(data["n_train"]):
        fail("quadratic_anchor_len", "Anchor residual length mismatch")
        write_json(root / "hybrid_preflight_report.json", report)
        raise SystemExit(1)
    m2_a, m3_a = nl_moment_pair(e_anchor, basis["phi_train"])
    tau = m23_norm2(m2_a, m3_a)
    alpha_anchor = float(q25["raw_rho"])
    qmet = pd.read_csv(hybrid_dir(root, "quadratic") / "metrics.csv")
    qok = qmet.loc[qmet["status"].astype(str) == "ok"].copy()
    qok = qok.loc[qok["m2_train"].apply(six._finite)]
    qok["M23"] = qok["m2_train"].astype(float) ** 2 + qok["m3_train"].astype(float) ** 2
    m23_path = qok[["s_train", "raw_rho", "M23", "m2_train", "m3_train"]].to_dict(orient="records")
    min_row = qok.loc[qok["M23"].idxmin()]
    near_min = bool(abs(float(min_row["s_train"]) - float(q25["s_train"])) <= 0.12)
    ok(
        "quadratic_anchor",
        {
            "target_s": ANCHOR_S,
            "s_train": float(q25["s_train"]),
            "alpha_anchor": alpha_anchor,
            "m2_train": m2_a,
            "m3_train": m3_a,
            "tau": tau,
            "pred_sha256": six.sha_file(pred),
            "train_e_sha256": six.sha_file(tre),
            "m23_path": m23_path,
            "training_M23_minimum_s": float(min_row["s_train"]),
            "anchor_near_training_M23_min": near_min,
            "note": "Anchor frozen at target_s=0.25; diagnostic only, not retuned.",
        },
    )
    d23 = guardrail_d23(basis["phi_train"])
    if float(np.min(d23)) < -1e-15 or not np.all(np.isfinite(d23)):
        fail("d23", "d23 not finite/nonnegative")
        write_json(root / "hybrid_preflight_report.json", report)
        raise SystemExit(1)
    rng = np.random.default_rng(2025)
    gaps = []
    for _ in range(24):
        x = rng.normal(size=d23.size)
        gaps.append(majorizer_gap(basis["phi_train"], d23, x))
        if gaps[-1] > 1e-6:
            fail("majorizer", "PSD majorizer failed on a training direction", gaps[-1])
            write_json(root / "hybrid_preflight_report.json", report)
            raise SystemExit(1)
    ok("majorizer", {"n_directions": 24, "max_gap": float(np.max(gaps)), "d23_mean": float(np.mean(d23))})

    # Formula checks on the actual anchor residual (no new full-model fit).
    c = basis["c_train"]
    g0, h0, _ = quadratic_direct_cap_scaled_grad_hess(e_anchor, c, alpha=alpha_anchor, lam=0.0)
    y_true = c
    y_pred = e_anchor + c
    gq, hq, _ = canonical_surrogate_scaled_grad_hess(y_true, y_pred, y_mean=0.0, rho=alpha_anchor)
    if float(np.max(np.abs(g0 - gq))) > 1e-8 or float(np.max(np.abs(h0 - hq))) > 1e-8:
        fail("qd_lambda0", "lambda=0 does not reproduce Quadratic derivatives")
        write_json(root / "hybrid_preflight_report.json", report)
        raise SystemExit(1)
    gA, hA, _ = quadratic_direct_cap_scaled_grad_hess(e_anchor, c, alpha=0.0, lam=1.7)
    gd, hd, _ = canonical_direct_scaled_grad_hess(y_true, y_pred, y_mean=0.0, rho=1.7)
    if float(np.max(np.abs(gA - gd))) > 1e-8 or float(np.max(np.abs(hA - hd))) > 1e-8:
        fail("qd_alpha0", "alpha=0 does not reproduce canonical Direct")
        write_json(root / "hybrid_preflight_report.json", report)
        raise SystemExit(1)
    H = quadratic_direct_cap_exact_hessian(c[:12], alpha=alpha_anchor, lam=2.0)
    Hs = 1.0 + alpha_anchor * np.square(c[:12]) + (2.0 / (2.0 * 12)) * np.square(c[:12])
    if float(np.max(np.abs(np.diag(H) - Hs))) > 1e-10:
        fail("qd_supplied_hess", "supplied Hessian is not Quadratic diag + Direct diag treatment")
        write_json(root / "hybrid_preflight_report.json", report)
        raise SystemExit(1)
    ok("qd_formulas", {"grad": QD_GRAD, "hess_supplied": QD_HESS_SUPPLIED, "hess_exact": QD_HESS_EXACT})

    gN, hN, _ = quadratic_nl_guardrail_scaled_grad_hess(e_anchor, c, basis["phi_train"], d23, rho=alpha_anchor, gamma=0.0)
    if float(np.max(np.abs(gN - gq))) > 1e-8 or float(np.max(np.abs(hN - hq))) > 1e-8:
        fail("qnl_gamma0", "gamma=0 does not reproduce Quadratic")
        write_json(root / "hybrid_preflight_report.json", report)
        raise SystemExit(1)
    if np.any(hN <= 0):
        fail("qnl_hess_pos", "non-positive Hessian at gamma=0")
        write_json(root / "hybrid_preflight_report.json", report)
        raise SystemExit(1)
    ok("qnl_formulas", {"grad": QNL_GRAD, "hess_supplied": QNL_HESS_SUPPLIED, "hess_exact": QNL_HESS_EXACT, "curvature": QNL_CURVATURE})

    q_rhos = {float(r["target_s"]): float(r["raw_rho"]) for _, r in load_matched(root, "quadratic").iterrows()}
    cfg = {
        "experiment_label": EXPERIMENT_LABEL,
        "experiment": "toy_hybrid_selection_v2",
        "alpha_anchor": alpha_anchor,
        "tau": tau,
        "m2_anchor": m2_a,
        "m3_anchor": m3_a,
        "anchor_s_train": float(q25["s_train"]),
        "anchor_pred_tag": str(q25["pred_tag"]),
        "quadratic_anchor": {
            "target_s": ANCHOR_S,
            "s_train": float(q25["s_train"]),
            "raw_rho": alpha_anchor,
            "C_train": float(q25["C_train"]),
            "Beta_log_train": float(q25["Beta_log_train"]),
            "m1_train": float(q25["m1_train"]),
            "m2_train": m2_a,
            "m3_train": m3_a,
            "M23_train": tau,
            "mse_train": float(q25["mse_train"]) if "mse_train" in q25.index and six._finite(q25["mse_train"]) else float("nan"),
            "pred_tag": str(q25["pred_tag"]),
        },
        "quadratic_target_rho": q_rhos,
        "Beta_log_train_0": float(mech["Beta_log_train_0"]),
        "C_train_0": float(mech["C_train_0"]),
        "hashes": hashes_now,
        "prior_hashes": old_h,
        "d23_mean": float(np.mean(d23)),
        "git": git_state(REPO),
        "packages": six.package_versions_metadata(),
        "heldout_never_used_for_penalty": True,
        "qd": {"grad": QD_GRAD, "hess_supplied": QD_HESS_SUPPLIED, "hess_exact": QD_HESS_EXACT},
        "qnl": {"grad": QNL_GRAD, "hess_supplied": QNL_HESS_SUPPLIED, "hess_exact": QNL_HESS_EXACT},
    }
    np.save(root / "hybrid_d23.npy", d23)
    write_json(root / "hybrid_config.json", cfg)
    if failures:
        report["status"] = "FAIL"
        report["failures"] = failures
        write_json(root / "hybrid_preflight_report.json", report)
        raise SystemExit(1)
    report["status"] = "PASS"
    report["alpha_anchor"] = alpha_anchor
    report["tau"] = tau
    report["config_path"] = str(root / "hybrid_config.json")
    write_json(root / "hybrid_preflight_report.json", report)
    _log("hybrid preflight PASS", alpha=alpha_anchor, tau=tau)


def run_target(args) -> None:
    root = Path(args.output_root)
    pre = read_json(root / "hybrid_preflight_report.json")
    if str(pre.get("status")) != "PASS":
        raise SystemExit("Refusing hybrid target: preflight did not PASS.")
    method = str(args.method)
    sstar = float(args.target_s)
    if method not in HYBRID_METHODS:
        raise SystemExit(f"Unknown hybrid method {method}")
    if not any(np.isclose(sstar, t) for t in HYBRID_TARGETS):
        raise SystemExit(f"target_s {sstar} is not in {HYBRID_TARGETS}")
    params = v1.load_params(Path(args.params))
    n_jobs = v1._allocated_cpus()
    lgbm_params = v1.load_frozen_lgbm_params(Path(args.lgbm_config_json), n_jobs=n_jobs, n_estimators=args.n_estimators)
    data = v1.load_experiment_data(data_path=Path(args.data_path), params=params, sample_frac=args.sample_frac, seed=int(args.seed))
    state = six.load_shared_state(root, data)
    cfg = read_json(root / "hybrid_config.json")
    d23 = np.load(root / "hybrid_d23.npy")
    v1._ensure_dir(hybrid_dir(root, method))
    _log("target start", method=method, sstar=sstar, n_jobs=n_jobs)
    if method == "quadratic_direct_cap":
        search_qd(root, sstar, data, lgbm_params, state["basis"], cfg, d23)
    else:
        search_qnl(root, sstar, data, lgbm_params, state["basis"], cfg, d23)


def _as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().eq("true")


def collect_hybrid_table(root: Path) -> pd.DataFrame:
    frames = []
    master = pd.read_csv(root / "matched_correction_metrics.csv")
    keep = master.loc[master["method"].isin(BENCHMARKS)].copy()
    keep = keep.loc[keep["target_s"].apply(lambda s: any(np.isclose(float(s), t) for t in CONTEXT_S))]
    frames.append(keep)
    q25 = keep.loc[(keep["method"] == "quadratic") & np.isclose(keep["target_s"].astype(float), ANCHOR_S)].copy()
    for method in HYBRID_METHODS:
        if not q25.empty:
            row = q25.iloc[0].to_dict()
            row["method"] = method
            row["alpha"] = read_json(root / "hybrid_config.json")["alpha_anchor"]
            row["lambda"] = 0.0 if method == "quadratic_direct_cap" else 0.0
            row["gamma"] = 0.0
            row["rho"] = float(q25.iloc[0]["raw_rho"]) if method == "quadratic_nl_guardrail" else 0.0
            row["note"] = "shared_quadratic_s25_anchor"
            frames.append(pd.DataFrame([row]))
        for sstar in HYBRID_TARGETS:
            p = hybrid_dir(root, method) / f"matched_s_{sstar:g}.csv"
            if p.is_file():
                frames.append(pd.read_csv(p))
    out = pd.concat(frames, ignore_index=True, sort=False)
    return out


def plot_hybrid_ratio(root: Path, matched: pd.DataFrame) -> Path:
    import matplotlib.pyplot as plt

    six._set_style()
    methods = list(BENCHMARKS) + list(HYBRID_METHODS)
    titles = {**{"current_direct": "Current Direct", "direct_mm_k1": "Direct-MM K=1", "quadratic": "Quadratic"}, **HYBRID_TITLES}
    frames = []
    att = matched.copy()
    if "attained" in att.columns:
        att = att.loc[_as_bool_series(att["attained"])]
    for _, rec in att.iterrows():
        method = str(rec["method"])
        tag = str(rec.get("pred_tag") or "")
        src_method = "quadratic" if (abs(float(rec["target_s"]) - ANCHOR_S) < 1e-12 and method in HYBRID_METHODS) else method
        p = six.pred_path(root, src_method, tag)
        if not p.is_file():
            continue
        pred = pd.read_parquet(p)
        if "y_pred" not in pred.columns:
            pred["y_pred"] = np.exp(pred["y_pred_log"])
            pred["y_true"] = np.exp(pred["y_true_log"])
        b = v1.equal_count_bins(pred["y_true"].to_numpy(), pred["y_pred"].to_numpy() / np.clip(pred["y_true"].to_numpy(), 1e-12, None))
        b["method"] = method
        b["target_s"] = float(rec["target_s"])
        frames.append(b)
    if not frames:
        return root / "figures" / "hybrid_matched_ratio_shape.pdf"
    bins = pd.concat(frames, ignore_index=True)
    six.atomic_csv(bins, root / "hybrid_matched_ratio_bins.csv")
    y = bins["median_ratio"].to_numpy(dtype=float)
    ylim = six._padded(y[np.isfinite(y)])
    fig, axes = plt.subplots(1, 5, figsize=(14.5, 3.4), sharex=True, sharey=True)
    cmap = plt.cm.viridis
    x_all = bins["median_sale_price"].to_numpy(dtype=float)
    xmin, xmax = float(np.min(x_all)), float(np.max(x_all))
    for ax, method in zip(axes, methods):
        sub = bins.loc[bins["method"] == method]
        for i, sstar in enumerate(CONTEXT_S):
            part = sub.loc[np.isclose(sub["target_s"].astype(float), float(sstar))].sort_values("bin")
            if part.empty:
                continue
            yy = np.clip(part["median_ratio"].to_numpy(dtype=float), ylim[0], ylim[1])
            ax.plot(part["median_sale_price"], yy, color=cmap(0.1 + 0.8 * i / 4), lw=1.4, marker="o", ms=2.0, label=rf"$s={sstar:.2f}$")
        ax.axhline(1.0, color="#111827", ls=(0, (2, 2)), lw=0.7)
        ax.set_xscale("log")
        ax.set_ylim(*ylim)
        ax.set_xlim(xmin / 1.05, xmax * 1.05)
        ax.set_title(titles[method], fontsize=9)
        ax.grid(True, color="#E5E7EB", lw=0.6)
        ax.set_xlabel("Sale price")
        if method == "current_direct":
            ax.set_ylabel("Valuation-to-sale ratio")
    axes[-1].legend(fontsize=6, frameon=False)
    fig.suptitle("EXPERIMENTAL / TOY hybrid matched ratio shapes", fontsize=11)
    fig.tight_layout()
    out = root / "figures" / "hybrid_matched_ratio_shape"
    v1._ensure_dir(out.parent)
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    return out.with_suffix(".pdf")


def _overlay(ax, work: pd.DataFrame, col: str, colors: dict, methods: list, titles: dict) -> None:
    axis = work.loc[work[col].apply(six._finite)] if col in work.columns else work
    ylim = six._padded(axis[col]) if col in axis.columns else (0.0, 1.0)
    for method in methods:
        sub = work.loc[work["method"] == method].sort_values("achieved_s", ascending=False)
        if sub.empty or col not in sub.columns:
            continue
        ax.plot(sub["achieved_s"], sub[col], color=colors[method], marker="o", ms=4, lw=1.3, label=titles.get(method, method))
    ax.set_ylim(*ylim)
    ax.grid(True, color="#E5E7EB", lw=0.6)
    ax.set_axisbelow(True)


def plot_hybrid_lines(root: Path, matched: pd.DataFrame) -> Dict[str, str]:
    import matplotlib.pyplot as plt

    six._set_style()
    work = matched.copy()
    if "attained" in work.columns:
        work = work.loc[_as_bool_series(work["attained"])]
    methods = list(BENCHMARKS) + list(HYBRID_METHODS)
    colors = {
        "current_direct": "#111827",
        "direct_mm_k1": "#1D4ED8",
        "quadratic": "#BE123C",
        "quadratic_direct_cap": "#0F766E",
        "quadratic_nl_guardrail": "#A16207",
    }
    titles = {**{"current_direct": "Current Direct", "direct_mm_k1": "Direct-MM K=1", "quadratic": "Quadratic"}, **HYBRID_TITLES}
    paths = {}
    fig, axes = plt.subplots(4, 1, figsize=(8.4, 9.0), sharex=True)
    for ax, col, ylab in zip(
        axes,
        ["Beta_log", "dCor_e_y", "L_NL", "NL_shape"],
        [r"held-out $\beta_{\log}$", r"held-out dCor", r"$L_{\mathrm{NL}}$", r"NL$_{\mathrm{shape}}$"],
    ):
        _overlay(ax, work, col, colors, methods, titles)
        ax.set_ylabel(ylab)
    axes[-1].set_xlabel(r"achieved signed training $s$")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("EXPERIMENTAL / TOY hybrid mechanism vs training s", fontsize=11)
    fig.tight_layout()
    out = root / "figures" / "hybrid_mechanism_vs_s"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["mechanism"] = str(out.with_suffix(".pdf"))

    work = work.copy()
    work["m23_eval"] = np.sqrt(work["m2_eval"].astype(float) ** 2 + work["m3_eval"].astype(float) ** 2)
    fig, axes = plt.subplots(4, 1, figsize=(8.4, 9.0), sharex=True)
    for ax, col, ylab in zip(axes, ["m1_eval", "m2_eval", "m3_eval", "m23_eval"], [r"$m_1$ eval", r"$m_2$ eval", r"$m_3$ eval", r"$\sqrt{m_2^2+m_3^2}$"]):
        _overlay(ax, work, col, colors, methods, titles)
        ax.set_ylabel(ylab)
    axes[-1].set_xlabel(r"achieved signed training $s$")
    axes[0].legend(frameon=False, fontsize=7, ncol=2)
    fig.suptitle("EXPERIMENTAL / TOY hybrid moment components vs training s", fontsize=11)
    fig.tight_layout()
    out = root / "figures" / "hybrid_moment_components_vs_s"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["moments"] = str(out.with_suffix(".pdf"))

    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2))
    panels = [("R2_price", r"$R^2_P$"), ("MAE_price", "MAE"), ("dCor_e_y", "dCor"), ("L_NL", r"$L_{\mathrm{NL}}$")]
    beta_lim = six._padded(work["Beta_log"]) if "Beta_log" in work.columns else (-0.2, 0.0)
    for ax, (col, ylab) in zip(axes.ravel(), panels):
        ylim = six._padded(work[col]) if col in work.columns else (0, 1)
        for method in methods:
            sub = work.loc[work["method"] == method]
            if sub.empty or col not in sub.columns:
                continue
            ax.plot(sub["Beta_log"], sub[col], color=colors[method], marker="o", ms=4, lw=1.3, label=titles[method])
        ax.set_xlabel(r"held-out $\beta_{\log}$")
        ax.set_ylabel(ylab)
        ax.set_xlim(*beta_lim)
        ax.set_ylim(*ylim)
        ax.grid(True, color="#E5E7EB", lw=0.6)
    axes[0, 0].legend(frameon=False, fontsize=6.5)
    fig.suptitle("EXPERIMENTAL / TOY hybrid tradeoffs", fontsize=11)
    fig.tight_layout()
    out = root / "figures" / "hybrid_tradeoff"
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths["tradeoff"] = str(out.with_suffix(".pdf"))
    return paths


def run_assemble(args) -> None:
    root = Path(args.output_root)
    graph = read_json(root / "manifests" / "hybrid_job_graph.json") if (root / "manifests" / "hybrid_job_graph.json").is_file() else {}
    cfg = read_json(root / "hybrid_config.json") if (root / "hybrid_config.json").is_file() else {}
    pre = read_json(root / "hybrid_preflight_report.json") if (root / "hybrid_preflight_report.json").is_file() else {}
    final: Dict[str, Any] = {
        "experiment_label": EXPERIMENT_LABEL,
        "status": "FAIL",
        "preflight_job_id": graph.get("preflight_job_id"),
        "qd_array_job_id": graph.get("qd_array_job_id"),
        "qnl_array_job_id": graph.get("qnl_array_job_id"),
        "assemble_job_id": os.environ.get("SLURM_JOB_ID"),
        "git": git_state(REPO),
        "alpha_anchor": cfg.get("alpha_anchor"),
        "tau": cfg.get("tau"),
    }
    reasons: List[str] = []
    if str(pre.get("status")) != "PASS":
        reasons.append("hybrid_preflight_not_pass")
    sentinels = {}
    locked = {}
    for method in HYBRID_METHODS:
        for sstar in HYBRID_TARGETS:
            key = f"{method}:s={sstar:g}"
            p = sentinel_path(root, method, sstar)
            sentinels[key] = p.is_file()
            if p.is_file():
                locked[key] = read_json(p)
            else:
                reasons.append("missing_sentinel:" + key)
            trace = hybrid_dir(root, method) / f"search_trace_s_{sstar:g}.csv"
            if trace.is_file():
                tr = pd.read_csv(trace)
                bad = [c for c in tr.columns if any(k in c.lower() for k in ("dcor", "r2_price", "nl_shape", "l_nl", "prd", "prb"))]
                if "heldout_computed" in tr.columns and bool(_as_bool_series(tr["heldout_computed"]).any()):
                    reasons.append(f"search_used_heldout:{key}")
                if bad:
                    reasons.append(f"search_has_heldout_cols:{key}:{','.join(bad)}")
    final["sentinels"] = sentinels
    final["locked"] = locked
    taus = {k: v.get("tau") for k, v in locked.items() if k.startswith("quadratic_nl_guardrail")}
    alphas = {k: v.get("alpha") for k, v in locked.items() if k.startswith("quadratic_direct_cap")}
    if taus and max(abs(float(t) - float(cfg.get("tau", t))) for t in taus.values() if t is not None) > 1e-12:
        reasons.append("tau_mismatch")
    if alphas and max(abs(float(a) - float(cfg.get("alpha_anchor", a))) for a in alphas.values() if a is not None) > 1e-12:
        reasons.append("alpha_mismatch")
    try:
        table = collect_hybrid_table(root)
        six.atomic_csv(table, root / "matched_correction_metrics_hybrids.csv")
        master_path = root / "matched_correction_metrics.csv"
        master = pd.read_csv(master_path)
        new_rows = table.loc[table["method"].isin(HYBRID_METHODS)].copy()
        existing = set()
        if "method" in master.columns and "target_s" in master.columns:
            for _, r in master.iterrows():
                try:
                    existing.add((str(r["method"]), round(float(r["target_s"]), 5)))
                except Exception:
                    continue
        mask = []
        for _, r in new_rows.iterrows():
            try:
                mask.append((str(r["method"]), round(float(r["target_s"]), 5)) not in existing)
            except Exception:
                mask.append(True)
        keep_new = new_rows.loc[mask] if mask else new_rows.iloc[0:0]
        if not keep_new.empty:
            combined = pd.concat([master, keep_new], ignore_index=True, sort=False)
            six.atomic_csv(combined, master_path)
        figs = {
            "hybrid_matched_ratio_shape": str(plot_hybrid_ratio(root, table)),
        }
        figs.update(plot_hybrid_lines(root, table))
        final["figure_paths"] = figs
        final["hybrids_table"] = str(root / "matched_correction_metrics_hybrids.csv")
        if "attained" in table.columns:
            hyb = table.loc[table["method"].isin(HYBRID_METHODS)]
            final["attained_counts"] = (
                hyb.groupby("method")["attained"].agg(lambda s: int(np.sum(_as_bool_series(s)))).to_dict()
                if not hyb.empty
                else {}
            )
            unatt = hyb.loc[~_as_bool_series(hyb["attained"])] if not hyb.empty else hyb
            cols = [c for c in ("method", "target_s", "status", "note", "s_train", "M23_train", "gamma", "lambda") if c in unatt.columns]
            final["unattained"] = unatt[cols].to_dict(orient="records") if cols and not unatt.empty else []
    except Exception as exc:
        reasons.append("assemble_exception:" + f"{type(exc).__name__}: {exc}")
        final["assemble_traceback"] = traceback.format_exc()
    if reasons:
        final["status"] = "FAIL"
        final["reasons"] = reasons
        write_json(root / "HYBRID_FINAL_STATUS.json", final)
        _log("assemble FAIL", reasons=",".join(reasons))
        raise SystemExit(1)
    final["status"] = "PASS"
    write_json(root / "HYBRID_FINAL_STATUS.json", final)
    _log("hybrid assemble PASS")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="EXPERIMENTAL / TOY hybrid mechanism continuation.")
    p.add_argument("--mode", required=True, choices=["preflight", "fit-target", "assemble"])
    p.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    p.add_argument("--data-path", default=str(v1.DEFAULT_DATA))
    p.add_argument("--params", default=str(v1.DEFAULT_PARAMS))
    p.add_argument("--lgbm-config-json", default=str(v1.DEFAULT_LGBM_CONFIG))
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--sample-frac", type=float, default=None)
    p.add_argument("--n-estimators", type=int, default=None)
    p.add_argument("--method", type=str, default=None)
    p.add_argument("--target-s", type=float, default=None)
    return p


def main() -> int:
    args = build_parser().parse_args()
    os.environ.setdefault("MPLBACKEND", "Agg")
    _log("start", mode=args.mode, label=EXPERIMENT_LABEL)
    if args.mode == "preflight":
        run_preflight(args)
    elif args.mode == "fit-target":
        if not args.method or args.target_s is None:
            raise SystemExit("--method and --target-s are required")
        run_target(args)
    else:
        run_assemble(args)
    _log("done", mode=args.mode)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
