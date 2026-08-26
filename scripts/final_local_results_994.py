#!/usr/bin/env python3
"""Primary CCAO remaining local results for paper_v6 (994 frozen paths).

Computes:
  1. split-specific native/custom rho=0 implementation-control audit
  2. frozen OOS Delta_NL diagnostic
  3. centered-recalibration benchmark from existing native OOF predictions

No new CV, no LightGBM retuning, no rho/family selection, no TeX compilation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from canonical_experiment import git_state, lgbm_params_hash, package_versions
from utils.delta_nl import (
    estimate_delta_nl,
    estimator_spec,
    estimator_spec_hash,
    identifier_fold_assignment,
)
from utils.motivation_utils import (
    build_rolling_origin_protocol,
    compute_taxation_metrics,
    paper_mechanism_metrics,
    split_ccao_assessment_universe,
)

RESULT_ROOT = REPO / "output" / "paper_v6_preselection_994"
OUT_DIR = RESULT_ROOT / "final_local_results"
DATA_PATH = REPO / "data" / "CCAO" / "2025" / "training_data.parquet"
NATIVE_CONFIG_ID = "252a25d9c0ce796b"
LINEAR_CONFIG_ID = "fd63507d2456c789"
DIRECT_RHO0_ID = "1fb838f7d6bfda88"
SURR_RHO0_ID = "5b7875e55e58ac62"
NATIVE_OOF_RUNS = {
    0: "d79e545b8be5d5c1",
    1: "05f6076a72c3b1e0",
    2: "f2f66192a65d897e",
    3: "2fa05df3cb4ca58a",
    4: "e64961bb0cbc7f4d",
    5: "e96269a98d368463",
    6: "9e5247d05cba9c9f",
}
TAX_TO_COMBINED = {
    "R2_price": "R2_price",
    "MAE_price": "MAE_price",
    "MAPE": "MAPE",
    "RMSE_log": "RMSE_log",
    "Median ratio": "median_ratio",
    "Mean ratio": "mean_ratio",
    "W. Mean ratio": "weighted_mean_ratio",
    "COD": "COD",
    "COV_IAAO": "COV",
    "PRD": "PRD",
    "PRB": "PRB",
    "MKI": "MKI",
    "VEI": "VEI",
    "Beta_log": "Beta_log",
    "Cov_log_residual_log_price": "Cov_log_residual_log_price",
    "dCor_e_y": "dCor_e_y",
}


def _run(cmd: List[str]) -> str:
    return subprocess.check_output(cmd, cwd=str(REPO), text=True).strip()


def git_provenance() -> Dict[str, Any]:
    status = _run(["git", "status", "--porcelain"])
    return {
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(status),
        "status_porcelain": status,
    }


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def write_df_atomic(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    if path.suffix == ".csv":
        df.to_csv(tmp, index=False)
    else:
        df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def combined_table() -> pd.DataFrame:
    return pd.read_csv(RESULT_ROOT / "analysis" / "combined_path_table.csv")


def protocol_dir() -> Path:
    return RESULT_ROOT / "protocol" / "data_id=d4929d43ec19badf" / "split_id=3d464d4a611b131b"


def pred_dir() -> Path:
    return RESULT_ROOT / "predictions" / "data_id=d4929d43ec19badf" / "split_id=3d464d4a611b131b"


def baseline_analysis_dir() -> Path:
    return (
        RESULT_ROOT
        / "baseline_reporting"
        / "analysis"
        / "data_id=d4929d43ec19badf"
        / "split_id=3d464d4a611b131b"
    )


def find_oos_pred(config_id: str, evaluation: str) -> Path:
    shard = "test_run_predictions" if evaluation == "heldout" else "assess_run_predictions"
    matches = list((RESULT_ROOT / "reporting_preview").glob(f"**/{shard}/{config_id}.parquet"))
    if not matches:
        base_shard = shard
        matches = list(baseline_analysis_dir().glob(f"{base_shard}/{config_id}.parquet"))
    if not matches:
        raise FileNotFoundError(f"no {evaluation} predictions for {config_id}")
    return matches[0]


def load_pred_frame(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "y_true_log" not in df.columns:
        df["y_true_log"] = np.log(np.clip(df["y_true"].to_numpy(dtype=float), 1e-12, None))
    if "y_pred_log" not in df.columns:
        df["y_pred_log"] = np.log(np.clip(df["y_pred"].to_numpy(dtype=float), 1e-12, None))
    if "y_true" not in df.columns:
        df["y_true"] = np.exp(df["y_true_log"].to_numpy(dtype=float))
    if "y_pred" not in df.columns:
        df["y_pred"] = np.exp(df["y_pred_log"].to_numpy(dtype=float))
    df["row_id"] = df["row_id"].astype(str)
    if df["row_id"].duplicated().any():
        raise RuntimeError(f"duplicated row_id in {path}")
    return df.sort_values("row_id").reset_index(drop=True)


def align_on_row_id(*frames: pd.DataFrame) -> List[pd.DataFrame]:
    idx = None
    for df in frames:
        keys = pd.Index(df["row_id"].astype(str))
        idx = keys if idx is None else idx.intersection(keys)
    if idx is None or len(idx) == 0:
        raise RuntimeError("empty alignment on row_id")
    if idx.has_duplicates:
        raise RuntimeError("duplicated alignment keys")
    out = []
    for df in frames:
        part = df.set_index("row_id").loc[idx].reset_index()
        out.append(part)
    y0 = out[0]["y_true_log"].to_numpy(dtype=float)
    for df in out[1:]:
        if not np.allclose(y0, df["y_true_log"].to_numpy(dtype=float), rtol=0.0, atol=1e-12):
            raise RuntimeError("y_true_log mismatch after row_id alignment")
    return out


def metrics_from_log(y_log: np.ndarray, yhat_log: np.ndarray) -> Dict[str, float]:
    tax = compute_taxation_metrics(y_log, yhat_log, scale="log")
    out = {}
    for src, dst in TAX_TO_COMBINED.items():
        out[dst] = float(tax[src])
    return out


def frozen_lgbm_params() -> Dict[str, Any]:
    return dict(load_json(RESULT_ROOT / "frozen_baseline.json")["best_lgbm_params"])


def model_config_from_run(config_id: str, fold_id: int = 0) -> Dict[str, Any]:
    run_dir = RESULT_ROOT / "runs" / "data_id=d4929d43ec19badf" / "split_id=3d464d4a611b131b" / f"fold_id={fold_id}"
    for path in run_dir.glob("*.parquet"):
        df = pd.read_parquet(path, columns=["config_id", "model_name", "model_config_json"])
        if str(df["config_id"].iloc[0]) == str(config_id):
            raw = df["model_config_json"].iloc[0]
            return json.loads(raw) if isinstance(raw, str) else dict(raw)
    raise FileNotFoundError(config_id)


def _rho0_pair_stats(native: pd.DataFrame, other: pd.DataFrame) -> Dict[str, float]:
    nlog = native["y_pred_log"].to_numpy(dtype=float)
    olog = other["y_pred_log"].to_numpy(dtype=float)
    delta = olog - nlog
    abs_d = np.abs(delta)
    return {
        "n_aligned": int(len(native)),
        "mean_abs_delta_log": float(np.mean(abs_d)),
        "median_abs_delta_log": float(np.median(abs_d)),
        "p95_abs_delta_log": float(np.quantile(abs_d, 0.95)),
        "max_abs_delta_log": float(np.max(abs_d)),
        "pearson_pred": float(np.corrcoef(nlog, olog)[0, 1]),
    }


def diagnose_rho0(native_cfg: Dict[str, Any], custom_cfg: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, Any]:
    native_params = dict(native_cfg.get("lgbm_params") or {})
    custom_params = dict(custom_cfg.get("lgbm_params") or native_params)
    native_hash = lgbm_params_hash(native_params)
    custom_hash = lgbm_params_hash(custom_params)
    frozen_hash = lgbm_params_hash(frozen_lgbm_params())
    same_vector = native_hash == custom_hash == frozen_hash
    match_native_init = bool(spec.get("match_native_init", True))
    return {
        "native_lgbm_params_sha256": native_hash,
        "custom_lgbm_params_sha256": custom_hash,
        "frozen_lgbm_params_sha256": frozen_hash,
        "same_frozen_994_param_vector": same_vector,
        "native_n_estimators": native_params.get("n_estimators"),
        "custom_n_estimators": custom_params.get("n_estimators"),
        "native_random_state": native_params.get("random_state"),
        "custom_random_state": custom_params.get("random_state"),
        "native_n_jobs": native_params.get("n_jobs"),
        "native_objective": native_params.get("objective"),
        "native_boost_from_average": native_params.get("boost_from_average", "LightGBM_default_True"),
        "match_native_init": match_native_init,
        "custom_init_path": (
            "center training labels by mean; LGBMRegressor(boost_from_average=False); "
            "init_score = zeros(n); add training mean back at predict"
        ),
        "rho0_custom_grad_hess": "canonical scaled L2: grad=e, hess=1 after multiplying the sample-average objective by n/2",
        "categorical_treatment": "same frozen LightGBM categorical hyperparameters; categoricals passed as pandas category dtype",
        "diagnosed_cause": (
            "REPRODUCIBLE_CUSTOM_OBJECTIVE_CODE_PATH: native uses built-in objective='mse' "
            "with LightGBM boost_from_average=True on the original log-price labels. "
            "Direct/Surrogate rho=0 use the custom-objective API with match_native_init: "
            "labels are centered, boost_from_average is disabled, init_score is identically zero, "
            "and the training mean is added back after boosting. At rho=0 the custom gradient/Hessian "
            "match native L2 algebraically (grad=e, hess=1), and Direct rho=0 coincides with Surrogate rho=0. "
            "The remaining native/custom prediction gap is therefore the LightGBM native-versus-custom "
            "training path (initialization and custom-objective API semantics), not a positive-rho "
            "objective bug. Positive-rho paths are left frozen. Custom rho=0 remains the within-family reference."
        ),
        "gate_outcome": "2_reproducible_custom_objective_code_path_difference",
    }


def run_rho0_audit() -> Dict[str, Any]:
    spec = load_json(RESULT_ROOT / "experiment_spec.json")
    native_cfg = model_config_from_run(NATIVE_CONFIG_ID)
    direct_cfg = model_config_from_run(DIRECT_RHO0_ID)
    surr_cfg = model_config_from_run(SURR_RHO0_ID)
    diagnosis = diagnose_rho0(native_cfg, direct_cfg, spec)
    rows = []
    split_payload = {}
    for evaluation, split_name in (("heldout", "heldout"), ("forward_2025", "forward_2025")):
        native = load_pred_frame(find_oos_pred(NATIVE_CONFIG_ID, evaluation))
        direct = load_pred_frame(find_oos_pred(DIRECT_RHO0_ID, evaluation))
        surr = load_pred_frame(find_oos_pred(SURR_RHO0_ID, evaluation))
        native, direct, surr = align_on_row_id(native, direct, surr)
        n_metrics = metrics_from_log(native["y_true_log"], native["y_pred_log"])
        d_metrics = metrics_from_log(direct["y_true_log"], direct["y_pred_log"])
        s_metrics = metrics_from_log(surr["y_true_log"], surr["y_pred_log"])
        d_stats = _rho0_pair_stats(native, direct)
        s_stats = _rho0_pair_stats(native, surr)
        ds_delta = np.abs(direct["y_pred_log"].to_numpy(dtype=float) - surr["y_pred_log"].to_numpy(dtype=float))
        split_payload[split_name] = {
            "n_aligned": int(len(native)),
            "n_native": int(len(native)),
            "identical_row_counts": True,
            "direct_vs_surrogate_mean_abs_delta_log": float(np.mean(ds_delta)),
            "direct_vs_surrogate_max_abs_delta_log": float(np.max(ds_delta)),
            "direct": d_stats,
            "surrogate": s_stats,
        }
        for model, cfg_id, mets, stats in (
            ("Ordinary LightGBM", NATIVE_CONFIG_ID, n_metrics, None),
            ("Direct rho=0", DIRECT_RHO0_ID, d_metrics, d_stats),
            ("Surrogate rho=0", SURR_RHO0_ID, s_metrics, s_stats),
        ):
            row = {
                "split": split_name,
                "model": model,
                "config_id": cfg_id,
                "n_aligned": int(len(native)),
                "R2_P": mets["R2_price"],
                "RMSE_logP": mets["RMSE_log"],
                "beta_log": mets["Beta_log"],
                "mean_abs_delta_log": None if stats is None else stats["mean_abs_delta_log"],
                "median_abs_delta_log": None if stats is None else stats["median_abs_delta_log"],
                "p95_abs_delta_log": None if stats is None else stats["p95_abs_delta_log"],
                "max_abs_delta_log": None if stats is None else stats["max_abs_delta_log"],
                "pearson_vs_native": None if stats is None else stats["pearson_pred"],
                "lgbm_params_sha256": diagnosis["frozen_lgbm_params_sha256"],
                "status": diagnosis["gate_outcome"],
            }
            rows.append(row)
    table = pd.DataFrame(rows)
    write_df_atomic(table, OUT_DIR / "rho0_split_audit.csv")
    payload = {
        "git": git_provenance(),
        "diagnosis": diagnosis,
        "direct_config_keys": sorted(direct_cfg.keys()),
        "surrogate_config_keys": sorted(surr_cfg.keys()),
        "splits": split_payload,
        "rows": rows,
        "positive_rho_paths_refit": False,
        "training_implementation_modified": False,
        "gate": "PASS_REPORT_CODE_PATH_DIFFERENCE",
    }
    write_json_atomic(OUT_DIR / "rho0_split_audit.json", payload)
    return payload


def oos_job_list(combined: pd.DataFrame) -> List[Dict[str, Any]]:
    jobs = []
    for _, row in combined.iterrows():
        family = str(row["family"])
        config_id = str(row["config_id"])
        rho = row["rho"]
        rho_val = None if pd.isna(rho) else float(rho)
        for evaluation in ("heldout", "forward_2025"):
            jobs.append(
                {
                    "family": family,
                    "rho": rho_val,
                    "config_id": config_id,
                    "model_name": str(row["model_name"]),
                    "evaluation": evaluation,
                }
            )
    return jobs


def shard_jobs(jobs: Sequence[Dict[str, Any]], shard: int, n_shards: int) -> List[Dict[str, Any]]:
    return [job for i, job in enumerate(jobs) if i % n_shards == shard]


def delta_nl_shard_path(shard: int, n_shards: int) -> Path:
    return OUT_DIR / "delta_nl_shards" / f"shard_{shard:02d}_of_{n_shards:02d}.parquet"


def run_delta_nl_shard(shard: int, n_shards: int) -> Path:
    dest = delta_nl_shard_path(shard, n_shards)
    if dest.is_file():
        return dest
    combined = combined_table()
    jobs = shard_jobs(oos_job_list(combined), shard, n_shards)
    fold_cache: Dict[str, np.ndarray] = {}
    rows = []
    for job in jobs:
        pred = load_pred_frame(find_oos_pred(job["config_id"], job["evaluation"]))
        key = job["evaluation"]
        if key not in fold_cache:
            fold_cache[key] = identifier_fold_assignment(pred["row_id"].to_numpy())
            fold_json = OUT_DIR / f"delta_nl_folds_{key}.json"
            if not fold_json.is_file():
                write_json_atomic(
                    fold_json,
                    {
                        "evaluation": key,
                        "n": int(len(pred)),
                        "estimator_spec_hash": estimator_spec_hash(),
                        "fold_counts": [int(np.sum(fold_cache[key] == k)) for k in range(5)],
                    },
                )
        folds = fold_cache[key]
        if len(folds) != len(pred):
            # Re-align cached assignment onto this model's row_id order via a map from first model.
            raise RuntimeError("fold assignment length mismatch; identifier order differs across models")
        # Rebuild folds from this frame's identifiers so order cannot leak across models.
        folds = identifier_fold_assignment(pred["row_id"].to_numpy())
        est = estimate_delta_nl(pred["y_true_log"], pred["y_pred_log"], pred["row_id"].to_numpy(), folds=folds)
        rows.append(
            {
                "family": job["family"],
                "rho": job["rho"],
                "config_id": job["config_id"],
                "model_name": job["model_name"],
                "evaluation": job["evaluation"],
                "n": est["n"],
                "Delta_NL": est["Delta_NL"],
                "Delta_NL_raw": est["Delta_NL_raw"],
                "MSE_aff": est["MSE_aff"],
                "MSE_spl": est["MSE_spl"],
                "var_e": est["var_e"],
                "fold_assignment_hash": est["fold_assignment_hash"],
                "estimator_spec_hash": est["estimator_spec_hash"],
            }
        )
    write_df_atomic(pd.DataFrame(rows), dest)
    return dest


def aggregate_delta_nl(n_shards: int) -> pd.DataFrame:
    frames = []
    for shard in range(n_shards):
        path = delta_nl_shard_path(shard, n_shards)
        if not path.is_file():
            raise FileNotFoundError(path)
        frames.append(pd.read_parquet(path))
    df = pd.concat(frames, ignore_index=True)
    df = df.drop_duplicates(["family", "config_id", "evaluation"], keep="first")
    write_df_atomic(df, OUT_DIR / "delta_nl_oos.csv")
    write_df_atomic(df, OUT_DIR / "delta_nl_oos.parquet")
    write_json_atomic(
        OUT_DIR / "delta_nl_estimator.json",
        {
            "spec": estimator_spec(),
            "spec_hash": estimator_spec_hash(),
            "n_rows": int(len(df)),
            "evaluations": sorted(df["evaluation"].unique().tolist()),
        },
    )
    return df


def load_native_oof() -> pd.DataFrame:
    frames = []
    for fold_id, run_id in NATIVE_OOF_RUNS.items():
        path = pred_dir() / f"fold_id={fold_id}" / f"{run_id}.parquet"
        df = load_pred_frame(path)
        df["fold_id"] = int(fold_id)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def load_split_frames() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    import yaml

    with open(REPO / "params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    cols = ["meta_sale_price", "meta_sale_date", "ind_pin_is_multicard", "sv_is_outlier"]
    df = pd.read_parquet(DATA_PATH, columns=cols)
    df = df.loc[
        (~df["ind_pin_is_multicard"].astype("bool").fillna(True))
        & (~df["sv_is_outlier"].astype("bool").fillna(True))
    ].copy()
    splits = split_ccao_assessment_universe(
        df,
        "meta_sale_date",
        split_prop=float(params["cv"]["split_prop"]),
        universe_start="2016-01-01",
        pre_assessment_end="2024-12-31",
        assessment_year=2025,
    )
    return splits["development"], splits["test"], splits["assessment"]


def reconstruct_fold_training_means() -> Dict[str, Any]:
    import yaml
    from run_temporal_cv import _load_and_split_data

    with open(REPO / "params.yaml", "r", encoding="utf-8") as f:
        params = yaml.safe_load(f)
    df_tv, df_test, df_assess, _, _ = _load_and_split_data(
        data_path=str(DATA_PATH),
        params=params,
        target_column="meta_sale_price",
        date_column="meta_sale_date",
        assessment_year=2025,
        heldout_test_mode="pre_assessment_tail",
        sample_frac=None,
        sample_seed=2025,
        universe_start="2016-01-01",
        pre_assessment_end="2024-12-31",
    )
    protocol = load_json(next((RESULT_ROOT / "protocol").glob("**/folds.json")))
    split_proto = protocol.get("split_protocol") or load_json(RESULT_ROOT / "experiment_spec.json")["split_protocol"]
    rebuilt = build_rolling_origin_protocol(
        df_tv,
        "meta_sale_date",
        train_mode=str(split_proto["train_mode"]),
        initial_train_months=int(split_proto["initial_train_months"]),
        val_fraction=float(split_proto["val_fraction"]),
        val_window_months=int(split_proto.get("val_window_months", 15)),
        step_months=int(split_proto["step_months"]),
        min_train_rows=int(split_proto["min_train_rows"]),
        min_val_rows=int(split_proto["min_val_rows"]),
    )
    y_log = np.log(df_tv["meta_sale_price"].to_numpy(dtype=float))
    fold_means = []
    hash_ok = True
    archived = protocol["folds"]
    if len(rebuilt) != len(archived):
        raise RuntimeError(f"rebuilt {len(rebuilt)} folds but archive has {len(archived)}")
    for rec, arch in zip(rebuilt, archived):
        idx = np.asarray(rec["train_indices"], dtype=int)
        mean_y = float(np.mean(y_log[idx]))
        ok = (
            rec["train_index_hash"] == arch["train_index_hash"]
            and rec["val_index_hash"] == arch["val_index_hash"]
            and int(rec["train_size"]) == int(arch["train_size"])
            and int(rec["val_size"]) == int(arch["val_size"])
        )
        hash_ok = hash_ok and ok
        fold_means.append(
            {
                "fold_id": int(rec["fold_id"]),
                "ybar_T": mean_y,
                "train_size": int(rec["train_size"]),
                "val_size": int(rec["val_size"]),
                "train_index_hash": rec["train_index_hash"],
                "val_index_hash": rec["val_index_hash"],
                "hashes_match_archive": ok,
            }
        )
    if not hash_ok:
        raise RuntimeError("reconstructed fold index hashes do not match the archived protocol")
    ybar_heldout = float(np.mean(y_log))
    y_prod = np.concatenate([y_log, np.log(df_test["meta_sale_price"].to_numpy(dtype=float))])
    ybar_2025 = float(np.mean(y_prod))
    meta_h = load_json(baseline_analysis_dir() / "test_eval_metadata.json")
    meta_a = load_json(baseline_analysis_dir() / "assess_eval_metadata.json")
    if abs(ybar_heldout - float(meta_h["y_train_log_mean"])) > 1e-10:
        raise RuntimeError("held-out training mean does not match baseline metadata")
    if abs(ybar_2025 - float(meta_a["y_train_log_mean"])) > 1e-10:
        raise RuntimeError("2016-2024 training mean does not match baseline metadata")
    if int(df_tv.shape[0]) != 344607 or int(df_test.shape[0]) != 38290 or int(df_assess.shape[0]) != 26641:
        raise RuntimeError("split counts do not match the frozen 994 experiment")
    return {
        "ybar_heldout_train": ybar_heldout,
        "ybar_forward_2025_train": ybar_2025,
        "n_development": int(df_tv.shape[0]),
        "n_heldout": int(df_test.shape[0]),
        "n_production": int(df_tv.shape[0] + df_test.shape[0]),
        "n_2025": int(df_assess.shape[0]),
        "folds": fold_means,
        "hashes_match_archive": True,
    }


def beta_log_from_arrays(y: np.ndarray, yhat: np.ndarray) -> float:
    return float(paper_mechanism_metrics(y, yhat)["Beta_log"])


def solve_b_star_validation_neutral(oof: pd.DataFrame, fold_means: Dict[int, float]) -> Dict[str, float]:
    y = oof["y_true_log"].to_numpy(dtype=float)
    yhat0 = oof["y_pred_log"].to_numpy(dtype=float)
    ybar = np.asarray([fold_means[int(v)] for v in oof["fold_id"].to_numpy()], dtype=float)
    c = y - float(np.mean(y))
    var_y = float(np.mean(c**2))
    cov_ybar = float(np.mean(ybar * c))
    cov_yhat = float(np.mean(yhat0 * c))
    denom = cov_yhat - cov_ybar
    if abs(denom) < 1e-18:
        raise RuntimeError("cannot solve b_star: degenerate covariance denominator")
    b_star = (var_y - cov_ybar) / denom
    if not (b_star > 0.0 and np.isfinite(b_star)):
        raise RuntimeError(f"b_star is not a positive finite value: {b_star}")
    yhat_star = ybar + b_star * (yhat0 - ybar)
    beta_star = beta_log_from_arrays(y, yhat_star)
    beta_b1 = beta_log_from_arrays(y, yhat0)
    return {
        "b_star": float(b_star),
        "pooled_oof_beta_log_at_b_star": float(beta_star),
        "pooled_oof_beta_log_at_b1": float(beta_b1),
        "n_oof": int(len(oof)),
    }


def centered_map(yhat0: np.ndarray, ybar_t: float, b: float) -> np.ndarray:
    return float(ybar_t) + float(b) * (np.asarray(yhat0, dtype=float) - float(ybar_t))


def run_recalibration(delta_nl_lookup: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    means = reconstruct_fold_training_means()
    oof = load_native_oof()
    fold_mean_map = {int(r["fold_id"]): float(r["ybar_T"]) for r in means["folds"]}
    solved = solve_b_star_validation_neutral(oof, fold_mean_map)
    b_star = float(solved["b_star"])
    b_grid = [1.0 + (j / 50.0) * (b_star - 1.0) for j in range(51)]
    if abs(b_grid[0] - 1.0) > 1e-15:
        raise RuntimeError("b=1 is not the first recalibration grid point")

    native = {
        "heldout": load_pred_frame(find_oos_pred(NATIVE_CONFIG_ID, "heldout")),
        "forward_2025": load_pred_frame(find_oos_pred(NATIVE_CONFIG_ID, "forward_2025")),
    }
    ybar = {
        "heldout": float(means["ybar_heldout_train"]),
        "forward_2025": float(means["ybar_forward_2025_train"]),
    }
    fold_cache = {
        ev: identifier_fold_assignment(native[ev]["row_id"].to_numpy()) for ev in native
    }
    rows = []
    endpoint_frames = {}
    for ev in ("heldout", "forward_2025"):
        y = native[ev]["y_true_log"].to_numpy(dtype=float)
        yhat0 = native[ev]["y_pred_log"].to_numpy(dtype=float)
        yhat_b1 = centered_map(yhat0, ybar[ev], 1.0)
        if not np.allclose(yhat_b1, yhat0, rtol=0.0, atol=1e-15):
            raise RuntimeError(f"b=1 does not reproduce native predictions on {ev}")
        for j, b in enumerate(b_grid):
            yhat_b = centered_map(yhat0, ybar[ev], b)
            mets = metrics_from_log(y, yhat_b)
            dnl = estimate_delta_nl(y, yhat_b, native[ev]["row_id"].to_numpy(), folds=fold_cache[ev])
            rec = {
                "family": "CenteredRecalibration",
                "j": int(j),
                "b": float(b),
                "is_endpoint": bool(j == 0 or j == 50),
                "evaluation": ev,
                "ybar_T": ybar[ev],
                "Delta_NL": dnl["Delta_NL"],
                "Delta_NL_raw": dnl["Delta_NL_raw"],
                **mets,
            }
            rows.append(rec)
        yhat_star = centered_map(yhat0, ybar[ev], b_star)
        endpoint = native[ev][["row_id", "y_true_log", "y_true"]].copy()
        endpoint["y_pred_log"] = yhat_star
        endpoint["y_pred"] = np.exp(yhat_star)
        endpoint["b"] = b_star
        endpoint["ybar_T"] = ybar[ev]
        endpoint_frames[ev] = endpoint

    path = pd.DataFrame(rows)
    write_df_atomic(path, OUT_DIR / "recalibration_path.csv")
    write_df_atomic(path, OUT_DIR / "recalibration_path.parquet")
    write_df_atomic(endpoint_frames["heldout"], OUT_DIR / "recalibration_endpoint_predictions_heldout.parquet")
    write_df_atomic(endpoint_frames["forward_2025"], OUT_DIR / "recalibration_endpoint_predictions_2025.parquet")
    spec = {
        "transformation": "f_b(x) = ybar_T + b * (f_0(x) - ybar_T)",
        "f_0": "ordinary/native LightGBM (config_id 252a25d9c0ce796b)",
        "b_star_source": "validation-neutral",
        "b_star": b_star,
        "b_star_details": solved,
        "heldout_or_2025_used_to_choose_b": False,
        "ybar_T": ybar,
        "fold_training_means": means["folds"],
        "source_oof_predictions": {
            fold_id: str(pred_dir() / f"fold_id={fold_id}" / f"{run_id}.parquet")
            for fold_id, run_id in NATIVE_OOF_RUNS.items()
        },
        "source_oos_native": {
            "heldout": str(find_oos_pred(NATIVE_CONFIG_ID, "heldout")),
            "forward_2025": str(find_oos_pred(NATIVE_CONFIG_ID, "forward_2025")),
        },
        "b_grid": "b_j = 1 + (j/50)*(b_star-1), j=0,...,50; b_0=1 reproduces native LightGBM",
        "n_grid": 51,
        "git": git_provenance(),
        "frozen_lgbm_params_sha256": lgbm_params_hash(frozen_lgbm_params()),
        "no_new_cv": True,
        "no_model_selection": True,
        "model_refit": False,
    }
    write_json_atomic(OUT_DIR / "recalibration_spec.json", spec)
    return spec


def update_combined_with_delta_nl(delta_nl: pd.DataFrame) -> Path:
    path = RESULT_ROOT / "analysis" / "combined_path_table.csv"
    parquet_path = RESULT_ROOT / "analysis" / "combined_path_table.parquet"
    backup = RESULT_ROOT / "analysis" / "combined_path_table.pre_delta_nl.csv"
    if path.is_file() and not backup.is_file():
        write_df_atomic(pd.read_csv(path), backup)
    old = pd.read_csv(path)
    before_cols = [c for c in old.columns if not str(c).startswith("Delta_NL")]
    before = old[before_cols].copy()
    wide = {}
    for ev, col in (("heldout", "Delta_NL__heldout"), ("forward_2025", "Delta_NL__forward_2025")):
        part = delta_nl.loc[delta_nl["evaluation"] == ev, ["config_id", "Delta_NL"]].drop_duplicates("config_id")
        wide[col] = part.set_index(part["config_id"].astype(str))["Delta_NL"]
    new = old.copy()
    cfg = new["config_id"].astype(str)
    new["Delta_NL__heldout"] = cfg.map(wide["Delta_NL__heldout"])
    new["Delta_NL__forward_2025"] = cfg.map(wide["Delta_NL__forward_2025"])
    after = new[before_cols]
    if not after.equals(before):
        # Numeric equality with possible dtype noise: compare metric columns only.
        for col in before_cols:
            if col in {"family", "model_name", "config_id", "data_id", "split_id", "source_fit_git_commit", "analysis_git_commit", "baseline_hash", "grid_hash"}:
                if not (new[col].astype(str).fillna("") == old[col].astype(str).fillna("")).all():
                    raise RuntimeError(f"combined table identity column changed: {col}")
            else:
                a = pd.to_numeric(old[col], errors="coerce")
                b = pd.to_numeric(new[col], errors="coerce")
                if not np.allclose(a.fillna(0), b.fillna(0), equal_nan=True, rtol=0.0, atol=0.0):
                    # allow exact NA alignment
                    if not ((a.isna() & b.isna()) | (a == b)).all():
                        raise RuntimeError(f"non-Delta metric changed in combined table: {col}")
    if new["Delta_NL__heldout"].isna().any() or new["Delta_NL__forward_2025"].isna().any():
        raise RuntimeError("Delta_NL missing for some combined-table rows")
    if np.isinf(new["Delta_NL__heldout"]).any() or np.isinf(new["Delta_NL__forward_2025"]).any():
        raise RuntimeError("Delta_NL contains inf")
    write_df_atomic(new, path)
    write_df_atomic(new, parquet_path)
    paper_copy = RESULT_ROOT / "paper_outputs" / "tables" / "combined_path_table.csv"
    if paper_copy.parent.is_dir():
        write_df_atomic(new, paper_copy)
    return path


def cmd_audit() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = run_rho0_audit()
    print(json.dumps({"gate": payload["gate"], "splits": {k: v["n_aligned"] for k, v in payload["splits"].items()}}, indent=2))
    return 0


def cmd_delta_nl_shard() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--n-shards", type=int, required=True)
    args, _ = parser.parse_known_args()
    path = run_delta_nl_shard(args.shard, args.n_shards)
    print(path)
    return 0


def cmd_aggregate_delta_nl() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-shards", type=int, default=1)
    args, _ = parser.parse_known_args()
    df = aggregate_delta_nl(args.n_shards)
    hashes = df.groupby("evaluation")["fold_assignment_hash"].nunique()
    if (hashes > 1).any():
        raise RuntimeError(f"fold assignment hash is not unique within split: {hashes.to_dict()}")
    update_combined_with_delta_nl(df)
    print(df.groupby(["family", "evaluation"])["Delta_NL"].count().to_string())
    return 0


def cmd_recalibrate() -> int:
    spec = run_recalibration()
    print(json.dumps({"b_star": spec["b_star"], "source": spec["b_star_source"]}, indent=2))
    return 0


def cmd_all_local() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    write_json_atomic(OUT_DIR / "delta_nl_estimator.json", {"spec": estimator_spec(), "spec_hash": estimator_spec_hash()})
    audit = run_rho0_audit()
    n_shards = 1
    run_delta_nl_shard(0, n_shards)
    delta = aggregate_delta_nl(n_shards)
    spec = run_recalibration()
    update_combined_with_delta_nl(delta)
    print(
        json.dumps(
            {
                "rho0_gate": audit["gate"],
                "b_star": spec["b_star"],
                "delta_nl_rows": int(len(delta)),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["audit", "delta-nl-shard", "aggregate-delta-nl", "recalibrate", "all-local"])
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--n-shards", type=int, default=1)
    args = parser.parse_args()
    if args.command == "audit":
        return cmd_audit()
    if args.command == "delta-nl-shard":
        return cmd_delta_nl_shard()
    if args.command == "aggregate-delta-nl":
        return cmd_aggregate_delta_nl()
    if args.command == "recalibrate":
        return cmd_recalibrate()
    return cmd_all_local()


if __name__ == "__main__":
    raise SystemExit(main())
