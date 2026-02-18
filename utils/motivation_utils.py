import numpy as np
import pandas as pd
import seaborn as sns
from pandas.tseries.frequencies import to_offset
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score, mean_absolute_percentage_error, median_absolute_error, mean_squared_error
import os
import json
import hashlib
import socket
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone
from joblib import Parallel, delayed
# from typing import Union, List
from typing import Optional, Tuple, Dict, Any, List, Callable

try:
    from sklearn.metrics import root_mean_squared_error as _root_mse
except ImportError:  # sklearn < 1.4 compatibility
    def _root_mse(y_true, y_pred):
        return np.sqrt(mean_squared_error(y_true, y_pred))



# +-------------------------------------------------+
# |          Rolling-Origin Indices Mapping         |
# +-------------------------------------------------+

def get_rolling_origin_indices(df, date_col, growth_window='9MS', val_pct=0.10):
    """
    Creates an iterable of (train_idx, val_idx) for rolling-origin resampling.
    
    :param df: The input DataFrame.
    :param date_col: Name of the column containing dates.
    :param growth_window: Frequency string for training growth (e.g., '9MS' for 9 months start).
    :param val_pct: Float representing % of sales after training to use for validation.
    :return: List of tuples [(train_idx, val_idx), ...]
    """
    # 1. Sort data chronologically to ensure time-consistency
    df = df.sort_values(date_col).reset_index(drop=False)
    
    indices_list = []
    start_date = df[date_col].min()
    end_date = df[date_col].max()
    
    # 2. Define the moving 'origin' points based on the growth window
    # We generate a range of dates starting from the first training window
    current_origin = start_date + pd.Timedelta(days=270) # Initial 9-month approx
    
    # Generate date offsets for each fold
    date_offsets = pd.date_range(start=current_origin, end=end_date, freq=growth_window)

    for origin in date_offsets:
        # Training set: everything from the very beginning up to the current origin
        train_mask = df[date_col] <= origin
        train_indices = df.index[train_mask].values
        
        if len(train_indices) == 0:
            continue
            
        # Validation set: The next 10% of records immediately following the origin
        # We find the remaining pool of data
        remaining_pool = df.index[df[date_col] > origin].values
        
        if len(remaining_pool) == 0:
            break
            
        # Take the first val_pct of that remaining pool
        val_size = int(len(df) * val_pct)
        val_indices = remaining_pool[:val_size]

        # OPTION A: The Strict Way (Drop if < 10%)
        if len(remaining_pool) < val_size:
            print("Insufficient data for a full fold. Ending CV.")
            break
        
        if len(val_indices) > 0:
            indices_list.append((train_indices, val_indices))
            
    return indices_list


def get_sliding_window_indices(df, date_col, train_duration='3YS', slide_step='9MS', val_pct=0.10):
    """
    Creates an iterable of (train_idx, val_idx) for sliding-window resampling.
    
    :param df: The input DataFrame.
    :param date_col: Name of the column containing dates.
    :param train_duration: Frequency string for the FIXED length of training (e.g., '3YS' for 3 years).
    :param slide_step: Frequency string for how much the window moves forward (e.g., '9MS').
    :param val_pct: Float representing % of total dataset size to use for validation.
    :return: List of tuples [(train_idx, val_idx), ...]
    """
    # 1. Sort data chronologically
    df = df.sort_values(date_col).reset_index(drop=False)
    
    indices_list = []
    total_start_date = df[date_col].min()
    total_end_date = df[date_col].max()
    
    # Define the first 'origin' (end of first training window)
    # We start after the first full training duration has elapsed
    initial_origin = total_start_date + to_offset(train_duration)
    
    # Generate the sequence of origins (where training ends and validation begins)
    date_offsets = pd.date_range(start=initial_origin, end=total_end_date, freq=slide_step)

    for origin in date_offsets:
        # Calculate the sliding start date for this specific window
        window_start = origin - to_offset(train_duration)
        
        # Training set: Only data between [window_start] and [origin]
        train_mask = (df[date_col] > window_start) & (df[date_col] <= origin)
        train_indices = df.index[train_mask].values
        
        # Validation set logic: The pool of data strictly after the origin
        remaining_pool = df.index[df[date_col] > origin].values
        val_size = int(len(df) * val_pct)
        
        # Strict check: if not enough data for a full validation set, we stop
        if len(remaining_pool) < val_size:
            print(f"Insufficient data after {origin.date()} for a {val_pct*100}% validation set. Ending.")
            break
            
        val_indices = remaining_pool[:val_size]
        
        if len(train_indices) > 0 and len(val_indices) > 0:
            indices_list.append((train_indices, val_indices))
            
    return indices_list


def _stable_json(data: Dict[str, Any]) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=str)


def _stable_hash(data: Dict[str, Any], hash_len: int = 16) -> str:
    payload = _stable_json(data).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:hash_len]


def _write_json_atomic(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True, default=str)
    tmp_path.replace(path)


def _safe_corr_and_fisher(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if x.size < 2 or y.size < 2:
        return np.nan, np.nan
    c = float(np.corrcoef(x, y)[0, 1])
    if not np.isfinite(c):
        return np.nan, np.nan
    c_clip = float(np.clip(c, -0.999999, 0.999999))
    return c, float(np.arctanh(c_clip))


def _build_time_block_bootstrap_indices(
    val_dates: pd.Series,
    n_bootstrap: int,
    block_freq: str,
    rng_seed: int,
) -> List[np.ndarray]:
    # Build contiguous time blocks (e.g., weeks/months), then resample blocks w/ replacement.
    block_labels = val_dates.dt.to_period(block_freq).astype(str)
    groups: Dict[str, np.ndarray] = {}
    for label in pd.unique(block_labels):
        groups[label] = np.where(block_labels == label)[0]

    if not groups:
        return []

    labels = list(groups.keys())
    target_n = val_dates.shape[0]
    rng = np.random.default_rng(rng_seed)
    out: List[np.ndarray] = []
    for _ in range(n_bootstrap):
        sampled = []
        while len(sampled) < target_n:
            selected = labels[int(rng.integers(0, len(labels)))]
            sampled.extend(groups[selected].tolist())
        out.append(np.asarray(sampled[:target_n], dtype=int))
    return out


def build_rolling_origin_protocol(
    df: pd.DataFrame,
    date_col: str,
    *,
    train_mode: str = "expanding",
    initial_train_months: int = 9,
    val_fraction: Optional[float] = None,
    val_window_months: int = 1,
    step_months: int = 1,
    min_train_rows: int = 200,
    min_val_rows: int = 100,
) -> List[Dict[str, Any]]:
    """
    Build leakage-safe rolling-origin folds using fixed *time* windows.
    Returns fold dictionaries with explicit boundaries and row indices.
    """
    if train_mode not in {"expanding", "sliding"}:
        raise ValueError("train_mode must be 'expanding' or 'sliding'.")

    dfx = df[[date_col]].copy()
    dfx = dfx.sort_values(date_col).reset_index(drop=False).rename(columns={"index": "_row_id"})
    min_date = pd.Timestamp(dfx[date_col].min()).normalize()
    max_date = pd.Timestamp(dfx[date_col].max()).normalize()

    folds: List[Dict[str, Any]] = []
    # Mode A (CCAO figure logic): fixed-size validation block, always the next contiguous rows.
    # This matches "validation set is always 10% of sales immediately following training set."
    if val_fraction is not None:
        if not (0.0 < float(val_fraction) < 1.0):
            raise ValueError("val_fraction must be in (0, 1) when provided.")

        n_total = int(dfx.shape[0])
        val_size = max(int(np.floor(n_total * float(val_fraction))), 1)

        initial_end = min_date + pd.DateOffset(months=initial_train_months)
        initial_train_mask = dfx[date_col] < initial_end
        initial_train_size = int(initial_train_mask.sum())

        if initial_train_size < min_train_rows:
            initial_train_size = min_train_rows

        fold_idx = 0
        train_end_pos = initial_train_size
        while (train_end_pos + val_size) <= n_total:
            if train_mode == "expanding":
                train_start_pos = 0
            else:
                # For sliding mode under fixed-size validation, keep train width fixed.
                train_start_pos = max(0, train_end_pos - initial_train_size)

            train_rows = dfx.iloc[train_start_pos:train_end_pos]["_row_id"].to_numpy(dtype=int)
            val_rows = dfx.iloc[train_end_pos:train_end_pos + val_size]["_row_id"].to_numpy(dtype=int)

            if train_rows.size < min_train_rows or val_rows.size < min_val_rows:
                train_end_pos += val_size
                continue

            train_start_date = dfx.iloc[train_start_pos][date_col]
            train_end_date = dfx.iloc[train_end_pos - 1][date_col]
            val_start_date = dfx.iloc[train_end_pos][date_col]
            val_end_date = dfx.iloc[train_end_pos + val_size - 1][date_col]

            folds.append(
                {
                    "fold_id": fold_idx,
                    "train_start": str(pd.Timestamp(train_start_date).date()),
                    "train_end": str(pd.Timestamp(train_end_date).date()),
                    "val_start": str(pd.Timestamp(val_start_date).date()),
                    "val_end": str(pd.Timestamp(val_end_date).date()),
                    "train_indices": train_rows.tolist(),
                    "val_indices": val_rows.tolist(),
                    "train_size": int(train_rows.size),
                    "val_size": int(val_rows.size),
                    "train_index_hash": _stable_hash({"idx": train_rows.tolist()}),
                    "val_index_hash": _stable_hash({"idx": val_rows.tolist()}),
                }
            )
            fold_idx += 1
            # Advance by one full validation block so each fold validates on the "next 10%".
            train_end_pos += val_size

        return folds

    # Mode B: fixed-time validation windows (kept for backward compatibility).
    initial_end = min_date + pd.DateOffset(months=initial_train_months)
    current_train_end = initial_end
    fold_idx = 0
    while True:
        val_start = current_train_end
        val_end = val_start + pd.DateOffset(months=val_window_months)
        if val_start > max_date:
            break

        if train_mode == "expanding":
            train_start = min_date
        else:
            train_start = val_start - pd.DateOffset(months=initial_train_months)

        train_mask = (dfx[date_col] >= train_start) & (dfx[date_col] < val_start)
        val_mask = (dfx[date_col] >= val_start) & (dfx[date_col] < val_end)
        train_rows = dfx.loc[train_mask, "_row_id"].to_numpy(dtype=int)
        val_rows = dfx.loc[val_mask, "_row_id"].to_numpy(dtype=int)

        if train_rows.size >= min_train_rows and val_rows.size >= min_val_rows:
            folds.append(
                {
                    "fold_id": fold_idx,
                    "train_start": str(pd.Timestamp(train_start).date()),
                    "train_end": str((pd.Timestamp(val_start) - pd.Timedelta(days=1)).date()),
                    "val_start": str(pd.Timestamp(val_start).date()),
                    "val_end": str((pd.Timestamp(val_end) - pd.Timedelta(days=1)).date()),
                    "train_indices": train_rows.tolist(),
                    "val_indices": val_rows.tolist(),
                    "train_size": int(train_rows.size),
                    "val_size": int(val_rows.size),
                    "train_index_hash": _stable_hash({"idx": train_rows.tolist()}),
                    "val_index_hash": _stable_hash({"idx": val_rows.tolist()}),
                }
            )
            fold_idx += 1

        current_train_end = current_train_end + pd.DateOffset(months=step_months)
        if current_train_end + pd.DateOffset(months=val_window_months) > (max_date + pd.DateOffset(days=1)):
            break

    return folds


def _compute_extended_metrics(
    y_true_log: np.ndarray,
    y_pred_log: np.ndarray,
    y_train_log: Optional[np.ndarray] = None,
    ratio_mode: str = "div",
    eps_y: float = 1e-12,
) -> Dict[str, Any]:
    base = compute_taxation_metrics(
        y_true_log,
        y_pred_log,
        scale="log",
        y_train=y_train_log if y_train_log is not None else y_true_log,
    )
    y_true = np.exp(y_true_log)
    y_pred = np.exp(y_pred_log)

    if ratio_mode == "diff":
        r = y_pred_log - y_true_log
    else:
        r = y_pred / np.maximum(np.abs(y_true), eps_y)

    corr_ry, fisher_ry = _safe_corr_and_fisher(r, y_true)
    slope = np.nan
    if y_true.size >= 2:
        try:
            slope = float(np.polyfit(y_true, r, 1)[0])
        except Exception:
            slope = np.nan

    small_y = float(np.mean(np.abs(y_true) <= eps_y * 10.0))
    quantiles = [0.05, 0.5, 0.95]
    y_q = np.quantile(y_true, quantiles)
    r_q = np.quantile(r, quantiles)
    return {
        **base,
        "fairness_ratio_mode": ratio_mode,
        "Corr(r,y)_price": corr_ry,
        "FisherZ(r,y)_price": fisher_ry,
        "Slope(r~y)_price": slope,
        "Var(r)": float(np.var(r)),
        "Var(y)": float(np.var(y_true)),
        "val_rows": int(y_true.size),
        "small_abs_y_share": small_y,
        "y_q05": float(y_q[0]),
        "y_q50": float(y_q[1]),
        "y_q95": float(y_q[2]),
        "r_q05": float(r_q[0]),
        "r_q50": float(r_q[1]),
        "r_q95": float(r_q[2]),
    }


def run_robust_rolling_origin_cv(
    df_train_validate: pd.DataFrame,
    *,
    date_col: str,
    target_col: str,
    predictor_cols: List[str],
    categorical_cols: List[str],
    model_specs: List[Dict[str, Any]],
    linear_pipeline_builder: Callable[[], Any],
    result_root: str,
    data_signature: Dict[str, Any],
    split_protocol: Dict[str, Any],
    bootstrap_protocol: Dict[str, Any],
    fairness_ratio_mode: str = "div",
    predict_store: bool = True,
    parquet_engine: str = "fastparquet",
    log_progress: bool = True,
    parallel_enabled: bool = False,
    parallel_cpu_fraction: float = 0.75,
    parallel_max_workers: Optional[int] = None,
    parallel_backend: str = "loky",
) -> Dict[str, Any]:
    """
    End-to-end robust rolling-origin CV runner with:
    - deterministic IDs
    - fold-level full metrics
    - fold-level bootstrap metrics
    - append-only parquet artifacts (one file per run_id)
    """
    output_root = Path(result_root)
    output_root.mkdir(parents=True, exist_ok=True)

    folds = build_rolling_origin_protocol(
        df_train_validate,
        date_col,
        train_mode=split_protocol.get("train_mode", "expanding"),
        initial_train_months=int(split_protocol.get("initial_train_months", 9)),
        val_window_months=int(split_protocol.get("val_window_months", 1)),
        step_months=int(split_protocol.get("step_months", 1)),
        min_train_rows=int(split_protocol.get("min_train_rows", 200)),
        min_val_rows=int(split_protocol.get("min_val_rows", 100)),
    )

    fold_signature = {
        "split_protocol": split_protocol,
        "folds": [
            {
                k: f[k]
                for k in (
                    "fold_id",
                    "train_start",
                    "train_end",
                    "val_start",
                    "val_end",
                    "train_size",
                    "val_size",
                    "train_index_hash",
                    "val_index_hash",
                )
            }
            for f in folds
        ],
        "bootstrap_protocol": bootstrap_protocol,
    }
    data_id = _stable_hash({"data_signature": data_signature})
    split_id = _stable_hash(fold_signature)

    protocol_dir = output_root / "protocol" / f"data_id={data_id}" / f"split_id={split_id}"
    protocol_dir.mkdir(parents=True, exist_ok=True)
    with (protocol_dir / "folds.json").open("w", encoding="utf-8") as f:
        json.dump(fold_signature, f, indent=2, sort_keys=True)
    if log_progress:
        print(
            f"[cv] protocol saved | data_id={data_id} split_id={split_id} "
            f"| folds={len(folds)} | models={len(model_specs)}"
        )

    bootstrap_indices_by_fold: Dict[int, List[np.ndarray]] = {}
    for fold in folds:
        fold_id = int(fold["fold_id"])
        val_idx = np.asarray(fold["val_indices"], dtype=int)
        val_dates = df_train_validate.iloc[val_idx][date_col]
        bs_indices = _build_time_block_bootstrap_indices(
            val_dates=val_dates,
            n_bootstrap=int(bootstrap_protocol.get("n_bootstrap", 100)),
            block_freq=str(bootstrap_protocol.get("block_freq", "M")),
            rng_seed=int(bootstrap_protocol.get("seed", 2025)) + fold_id,
        )
        bootstrap_indices_by_fold[fold_id] = bs_indices

        bs_dir = protocol_dir / "bootstrap_indices"
        bs_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            bs_dir / f"fold_{fold_id}.npz",
            **{f"bs_{i}": arr for i, arr in enumerate(bs_indices)},
        )

    run_records: List[Dict[str, Any]] = []
    bootstrap_records: List[Dict[str, Any]] = []
    failed_records: List[Dict[str, Any]] = []

    runs_root = output_root / "runs"
    boots_root = output_root / "bootstrap_metrics"
    preds_root = output_root / "predictions"
    status_root = output_root / "run_status"
    total_runs = max(1, len(model_specs) * len(folds))
    processed_runs = 0
    skipped_runs = 0
    completed_runs = 0
    failed_runs = 0

    pending_jobs: List[Dict[str, Any]] = []

    for spec in model_specs:
        model_name = str(spec["name"])
        model_config = dict(spec.get("config", {}))
        config_id = _stable_hash({"model_name": model_name, "config": model_config})
        if log_progress:
            print(f"[cv] model start | model={model_name} | config_id={config_id}")

        for fold in folds:
            fold_id = int(fold["fold_id"])
            processed_runs += 1
            pct = 100.0 * processed_runs / float(total_runs)
            run_id = _stable_hash(
                {
                    "data_id": data_id,
                    "split_id": split_id,
                    "fold_id": fold_id,
                    "config_id": config_id,
                }
            )
            run_dir = runs_root / f"data_id={data_id}" / f"split_id={split_id}" / f"fold_id={fold_id}"
            run_dir.mkdir(parents=True, exist_ok=True)
            run_file = run_dir / f"{run_id}.parquet"
            bs_dir = boots_root / f"data_id={data_id}" / f"split_id={split_id}" / f"fold_id={fold_id}"
            bs_file = bs_dir / f"{run_id}.parquet"
            pred_dir = preds_root / f"data_id={data_id}" / f"split_id={split_id}" / f"fold_id={fold_id}"
            pred_file = pred_dir / f"{run_id}.parquet"
            status_dir = status_root / f"data_id={data_id}" / f"split_id={split_id}" / f"fold_id={fold_id}"
            status_file = status_dir / f"{run_id}.json"

            # Skip only when completion is explicitly marked and expected artifacts exist.
            if status_file.exists():
                try:
                    status_payload = json.loads(status_file.read_text(encoding="utf-8"))
                except Exception:
                    status_payload = {}
                completed = status_payload.get("status") == "completed"
                bootstrap_expected = int(bootstrap_protocol.get("n_bootstrap", 100)) > 0
                artifacts_ok = (
                    run_file.exists()
                    and ((not bootstrap_expected) or bs_file.exists())
                    and ((not predict_store) or pred_file.exists())
                )
                if completed and artifacts_ok:
                    skipped_runs += 1
                    if log_progress:
                        print(
                            f"[cv] [{processed_runs}/{total_runs} {pct:5.1f}%] skip "
                            f"| model={model_name} fold={fold_id} run_id={run_id}"
                        )
                    continue
            pending_jobs.append(
                {
                    "spec": spec,
                    "model_name": model_name,
                    "model_config": model_config,
                    "config_id": config_id,
                    "fold": fold,
                    "fold_id": fold_id,
                    "run_id": run_id,
                    "run_file": run_file,
                    "bs_dir": bs_dir,
                    "bs_file": bs_file,
                    "pred_dir": pred_dir,
                    "pred_file": pred_file,
                    "status_file": status_file,
                }
            )

    def _execute_single_job(job: Dict[str, Any]) -> Dict[str, Any]:
        spec = job["spec"]
        model_name = job["model_name"]
        model_config = job["model_config"]
        config_id = job["config_id"]
        fold = job["fold"]
        fold_id = job["fold_id"]
        run_id = job["run_id"]
        run_file = job["run_file"]
        bs_dir = job["bs_dir"]
        bs_file = job["bs_file"]
        pred_dir = job["pred_dir"]
        pred_file = job["pred_file"]
        status_file = job["status_file"]

        if log_progress:
            print(f"[cv] run | model={model_name} fold={fold_id} run_id={run_id}")

        _write_json_atomic(
            status_file,
            {
                "status": "started",
                "run_id": run_id,
                "data_id": data_id,
                "split_id": split_id,
                "config_id": config_id,
                "model_name": model_name,
                "fold_id": fold_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            },
        )

        try:
            train_idx = np.asarray(fold["train_indices"], dtype=int)
            val_idx = np.asarray(fold["val_indices"], dtype=int)
            train_df = df_train_validate.iloc[train_idx].copy()
            val_df = df_train_validate.iloc[val_idx].copy()

            X_train = train_df[predictor_cols].copy()
            y_train = np.log(train_df[target_col].to_numpy())
            X_val = val_df[predictor_cols].copy()
            y_val = np.log(val_df[target_col].to_numpy())

            cat_cols = [c for c in categorical_cols if c in X_train.columns]
            for c in cat_cols:
                X_train[c] = X_train[c].astype("category")
                X_val[c] = X_val[c].astype("category")

            start = time.perf_counter()
            estimator = spec["factory"]()
            if bool(spec.get("requires_linear_pipeline", False)):
                linear_pipeline = linear_pipeline_builder()
                X_train_model = linear_pipeline.fit_transform(X_train, y_train)
                X_val_model = linear_pipeline.transform(X_val)
            else:
                X_train_model = X_train
                X_val_model = X_val

            estimator.fit(X_train_model, y_train)
            y_pred_val = estimator.predict(X_val_model)
            runtime_sec = float(time.perf_counter() - start)

            metrics_full = _compute_extended_metrics(
                y_true_log=y_val,
                y_pred_log=y_pred_val,
                y_train_log=y_train,
                ratio_mode=fairness_ratio_mode,
            )

            run_row = {
                "data_id": data_id,
                "split_id": split_id,
                "config_id": config_id,
                "run_id": run_id,
                "model_name": model_name,
                "fold_id": fold_id,
                "train_start": fold["train_start"],
                "train_end": fold["train_end"],
                "val_start": fold["val_start"],
                "val_end": fold["val_end"],
                "train_size": fold["train_size"],
                "val_size": fold["val_size"],
                "hostname": socket.gethostname(),
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "runtime_sec": runtime_sec,
                "model_config_json": _stable_json(model_config),
                **metrics_full,
            }
            pd.DataFrame([run_row]).to_parquet(run_file, index=False, engine=parquet_engine)

            bs_rows: List[Dict[str, Any]] = []
            bs_indices = bootstrap_indices_by_fold.get(fold_id, [])
            for b_idx, sample_idx in enumerate(bs_indices):
                y_val_bs = y_val[sample_idx]
                y_pred_bs = np.asarray(y_pred_val)[sample_idx]
                m_bs = _compute_extended_metrics(
                    y_true_log=y_val_bs,
                    y_pred_log=y_pred_bs,
                    y_train_log=y_train,
                    ratio_mode=fairness_ratio_mode,
                )
                bs_rows.append(
                    {
                        "run_id": run_id,
                        "data_id": data_id,
                        "split_id": split_id,
                        "fold_id": fold_id,
                        "config_id": config_id,
                        "bootstrap_id": b_idx,
                        "bootstrap_seed": int(bootstrap_protocol.get("seed", 2025)) + fold_id,
                        "bootstrap_block_freq": str(bootstrap_protocol.get("block_freq", "M")),
                        "bootstrap_sample_size": int(sample_idx.size),
                        **m_bs,
                    }
                )
            if bs_rows:
                bs_dir.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(bs_rows).to_parquet(bs_file, index=False, engine=parquet_engine)

            if predict_store:
                pred_dir.mkdir(parents=True, exist_ok=True)
                pred_df = pd.DataFrame(
                    {
                        "run_id": run_id,
                        "row_id": val_df.index.to_numpy(),
                        "y_true_log": y_val,
                        "y_pred_log": np.asarray(y_pred_val),
                        "y_true": np.exp(y_val),
                        "y_pred": np.exp(np.asarray(y_pred_val)),
                        "sale_date": val_df[date_col].to_numpy(),
                    }
                )
                pred_df.to_parquet(pred_file, index=False, engine=parquet_engine)

            _write_json_atomic(
                status_file,
                {
                    "status": "completed",
                    "run_id": run_id,
                    "data_id": data_id,
                    "split_id": split_id,
                    "config_id": config_id,
                    "model_name": model_name,
                    "fold_id": fold_id,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "runtime_sec": runtime_sec,
                    "artifacts": {
                        "run_file": str(run_file),
                        "bootstrap_file": str(bs_file) if bs_file.exists() else None,
                        "predictions_file": str(pred_file) if predict_store and pred_file.exists() else None,
                    },
                },
            )
            return {"status": "completed", "run_row": run_row, "bs_rows": bs_rows}
        except Exception as exc:
            failed_payload = {
                "status": "failed",
                "run_id": run_id,
                "data_id": data_id,
                "split_id": split_id,
                "config_id": config_id,
                "model_name": model_name,
                "fold_id": fold_id,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": exc.__class__.__name__,
                "error_message": str(exc),
                "traceback": traceback.format_exc(),
            }
            _write_json_atomic(status_file, failed_payload)
            return {"status": "failed", "failed_payload": failed_payload}

    cpu_total = max(1, int(os.cpu_count() or 1))
    frac = float(parallel_cpu_fraction)
    if frac <= 0.0 or frac > 1.0:
        raise ValueError("parallel_cpu_fraction must be in (0, 1].")
    workers_by_fraction = max(1, int(np.floor(cpu_total * frac)))
    workers_cap = int(parallel_max_workers) if parallel_max_workers is not None else workers_by_fraction
    if workers_cap < 1:
        workers_cap = 1
    n_workers = min(workers_by_fraction, workers_cap, max(1, len(pending_jobs)))
    if (not parallel_enabled) or (len(pending_jobs) <= 1):
        n_workers = 1

    if parallel_backend not in ("loky", "threading"):
        raise ValueError("parallel_backend must be 'loky' or 'threading'.")

    if log_progress:
        print(
            f"[cv] execution plan | pending_runs={len(pending_jobs)} skipped={skipped_runs} "
            f"| parallel_enabled={parallel_enabled} backend={parallel_backend} "
            f"| workers={n_workers}/{cpu_total}"
        )

    if n_workers == 1:
        job_results = [_execute_single_job(job) for job in pending_jobs]
    else:
        job_results = Parallel(n_jobs=n_workers, backend=parallel_backend)(
            delayed(_execute_single_job)(job) for job in pending_jobs
        )

    for res in job_results:
        if res.get("status") == "completed":
            run_records.append(res["run_row"])
            completed_runs += 1
            bs_rows = res.get("bs_rows", [])
            if bs_rows:
                bootstrap_records.extend(bs_rows)
        else:
            failed_records.append(res["failed_payload"])
            failed_runs += 1

    if log_progress:
        print(
            f"[cv] done | total={total_runs} processed={processed_runs} "
            f"completed={completed_runs} skipped={skipped_runs} failed={failed_runs}"
        )

    return {
        "data_id": data_id,
        "split_id": split_id,
        "fold_count": len(folds),
        "run_records": pd.DataFrame(run_records) if run_records else pd.DataFrame(),
        "bootstrap_records": pd.DataFrame(bootstrap_records) if bootstrap_records else pd.DataFrame(),
        "failed_records": pd.DataFrame(failed_records) if failed_records else pd.DataFrame(),
    }



# +-------------------------------------------------+
# |          PART 1: STATISTICS CALCULATION         |
# +-------------------------------------------------+

# ----------------------------
#   1.1 Accuracy Metrics
# ----------------------------

def oos_r2_score(y_test, y_pred, y_train):
    """
    Calculates Out-of-sample R2 score using the mean of the training data as the baseline.
    """
    # Transform to numpy arrays if they are Series/DataFrames
    y_train = np.asarray(y_train).flatten()
    y_test = np.asarray(y_test).flatten()
    y_pred = np.asarray(y_pred).flatten()
    
    # Calculate the Mean Squared Error of your predictions
    mse_model = np.mean((y_test - y_pred)**2)
    
    # Calculate the Mean Squared Error of the training mean applied to the test set
    # This reflects the most recent "known" information
    train_mean = np.mean(y_train)
    mse_baseline = np.mean((y_test - train_mean)**2)
    
    # Avoid division by zero if all values in y_test are identical to train_mean
    if mse_baseline == 0:
        return 0.0
        
    oos_r2 = 1 - (mse_model / mse_baseline)
    return oos_r2

# ----------------------------
#   1.2 Dispersion Metrics
# ----------------------------

# COV (IAAO Coefficient of Variation, not Covariance)
def cov_iaao(assessed, sale_price, na_rm=False):
    """
    Calculate IAAO Coefficient of Variation (COV) for sales ratios.

    IAAO 2025 exposure draft Sec. 8.1.2:
      1) ratio_i = AV_i / SP_i
      2) mean_ratio = mean(ratio)
      3) sd_ratio = sqrt( sum((ratio_i - mean_ratio)^2) / (n - 1) )
      4) COV = sd_ratio / mean_ratio

    Notes:
      - Requires at least 2 valid ratios.
      - This is NOT covariance; it's the coefficient of variation of the ratio distribution.

    Args:
        assessed: Array-like of assessed values (AV).
        sale_price: Array-like of sale prices (SP).
        na_rm: Boolean, remove NAs if True.

    Returns:
        float: COV value (unitless). np.nan if not computable.
    """
    assessed, sale_price = _ensure_arrays(assessed, sale_price)

    cleaned = _handle_na((assessed, sale_price), na_rm)
    if cleaned[0] is None:
        return np.nan
    assessed, sale_price = cleaned

    assessed = np.asarray(assessed, dtype=float)
    sale_price = np.asarray(sale_price, dtype=float)

    # keep only finite and positive values
    m = np.isfinite(assessed) & np.isfinite(sale_price) & (assessed > 0) & (sale_price > 0)
    assessed, sale_price = assessed[m], sale_price[m]
    if assessed.size < 2:
        return np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = assessed / sale_price

    ratio = ratio[np.isfinite(ratio)]
    if ratio.size < 2:
        return np.nan

    mean_ratio = np.mean(ratio)
    if mean_ratio == 0:
        return np.nan

    sd_ratio = np.std(ratio, ddof=1)  # sample SD (n-1)
    return sd_ratio / mean_ratio # I need to add x 100 


# COD (Coefficient of Dispersion)
def _ensure_arrays(a, b=None):
    """Helper to convert inputs to numpy arrays."""
    a = np.asarray(a, dtype=float)
    if b is not None:
        b = np.asarray(b, dtype=float)
        return a, b
    return a

def _handle_na(arrays, na_rm=False):
    """
    Helper to handle NA values across one or more arrays.
    Returns cleaned arrays or raises/returns nan based on na_rm.
    """
    # Stack arrays to find common NaN indices
    if isinstance(arrays, tuple):
        combined = np.column_stack(arrays)
        mask = ~np.isnan(combined).any(axis=1)
        
        if na_rm:
            return [arr[mask] for arr in arrays]
        elif not np.all(mask):
            return [None] * len(arrays) # Signal to return NaN
        return arrays
    else:
        # Single array case
        mask = ~np.isnan(arrays)
        if na_rm:
            return arrays[mask]
        elif not np.all(mask):
            return None
        return arrays

def cod(ratio, na_rm=False):
    """
    Calculate Coefficient of Dispersion (COD).
    
    COD is the average absolute percent deviation from the median ratio.
    Lower is better (indicates uniformity).
    
    Args:
        ratio: Array-like of ratios (Assessed Value / Sale Price).
        na_rm: Boolean, remove NAs if True.
        
    Returns:
        float: The COD value.
    """
    ratio = _ensure_arrays(ratio)
    
    # Handle NA
    ratio = _handle_na(ratio, na_rm)
    if ratio is None: return np.nan
    if len(ratio) == 0: return np.nan

    med_ratio = np.median(ratio)
    
    # Avoid division by zero
    if med_ratio == 0:
        return np.nan
        
    cod_val = (np.mean(np.abs(ratio - med_ratio)) / med_ratio) * 100
    return cod_val


# ----------------------------
#   1.2 Regressivity
# ----------------------------

# VEI: 2025 IAAO standard metric for Regressivity
def vei(assessed, sale_price, na_rm=False):
    """
    Calculate Vertical Equity Indicator (VEI) point estimate (IAAO 2025 exposure draft, Sec. 8.2.1).

    Steps (point estimate):
      1) ratio_i = AV_i / SP_i
      2) sample_median = median(ratio)
      3) proxy_i = 0.50 * SP_i + 0.50 * (AV_i / sample_median)
      4) sort proxy ascending; split ratios into percentile groups:
           - halves if 20 <= n <= 50
           - quartiles if 51 <= n <= 500
           - deciles if n >= 501
         (requires >=10 observations per group)
      5) VEI = 100 * (median_lastPG - median_firstPG) / sample_median

    Interpretation:
      VEI < 0: regressive tendency
      VEI > 0: progressive tendency
      Acceptable range (point estimate): -10% to +10%

    Args:
        assessed: Array-like assessed/estimated values (AV).
        sale_price: Array-like sale prices (SP).
        na_rm: Boolean, remove NAs if True.

    Returns:
        float: VEI point estimate (%). np.nan if not computable.
    """
    assessed, sale_price = _ensure_arrays(assessed, sale_price)
    cleaned = _handle_na((assessed, sale_price), na_rm)
    if cleaned[0] is None:
        return np.nan
    assessed, sale_price = cleaned

    # Need positive values for ratios/proxy
    assessed = np.asarray(assessed, dtype=float)
    sale_price = np.asarray(sale_price, dtype=float)
    m = np.isfinite(assessed) & np.isfinite(sale_price) & (assessed > 0) & (sale_price > 0)
    assessed, sale_price = assessed[m], sale_price[m]

    n = len(assessed)
    if n < 20:
        return np.nan  # IAAO grouping rule starts at 20

    # Determine number of percentile groups
    if n <= 50:
        k = 2
    elif n <= 500:
        k = 4
    else:
        k = 10

    # Ratios and sample median ratio
    ratio = assessed / sale_price
    ratio = ratio[np.isfinite(ratio)]
    if ratio.size == 0:
        return np.nan
    med = np.median(ratio)
    if not np.isfinite(med) or med == 0:
        return np.nan

    # Market value proxy (IAAO)
    proxy = 0.5 * sale_price + 0.5 * (assessed / med)

    # Sort by proxy and split into k equal groups (exact "sort then split")
    order = np.argsort(proxy, kind="mergesort")
    chunks = np.array_split(np.arange(n), k)

    # Compute median ratio in first and last proxy groups
    first_idx = order[chunks[0]]
    last_idx  = order[chunks[-1]]

    # IAAO notes: at least 10 sales per group required
    if (first_idx.size < 10) or (last_idx.size < 10):
        return np.nan

    m_first = np.median((assessed[first_idx] / sale_price[first_idx]))
    m_last  = np.median((assessed[last_idx]  / sale_price[last_idx]))

    if not (np.isfinite(m_first) and np.isfinite(m_last)):
        return np.nan

    return 100.0 * (m_last - m_first) / med

# PRD
def prd(assessed, sale_price, na_rm=False):
    """
    Calculate Price-Related Differential (PRD).
    
    Measures vertical equity (regressivity/progressivity).
    Target range: 0.98 to 1.03.
    > 1.03 indicates regressivity (low value properties over-assessed).
    
    Args:
        assessed: Array-like of assessed values.
        sale_price: Array-like of sale prices.
        na_rm: Boolean, remove NAs if True.
        
    Returns:
        float: The PRD value.
    """
    assessed, sale_price = _ensure_arrays(assessed, sale_price)
    
    cleaned = _handle_na((assessed, sale_price), na_rm)
    if cleaned[0] is None: return np.nan
    assessed, sale_price = cleaned
    
    if len(assessed) == 0: return np.nan

    # Calculate ratios
    # Use standard numpy division, handling division by zero if sale_price is 0
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = assessed / sale_price
        
    # Remove any infinite ratios generated by 0 sale price if they exist
    valid_ratios = np.isfinite(ratio)
    ratio = ratio[valid_ratios]
    assessed = assessed[valid_ratios]
    sale_price = sale_price[valid_ratios]

    mean_ratio = np.mean(ratio)
    
    # Weighted mean: sum(ratio * weight) / sum(weight)
    # Here weight is sale_price.
    # Note: R's weighted.mean(x, w) = sum(x*w)/sum(w)
    # weighted_mean = sum((av/sp) * sp) / sum(sp) = sum(av) / sum(sp)
    weighted_mean_ratio = np.sum(assessed) / np.sum(sale_price)
    
    if weighted_mean_ratio == 0:
        return np.nan
        
    prd_val = mean_ratio / weighted_mean_ratio
    return prd_val

# PRB
def prb(assessed, sale_price, na_rm=False):
    """
    Calculate Coefficient of Price-Related Bias (PRB).
    
    Measures relationship between ratios and value.
    Target range: -0.05 to 0.05.
    Positive = Progressive, Negative = Regressive.
    
    Args:
        assessed: Array-like of assessed values.
        sale_price: Array-like of sale prices.
        na_rm: Boolean, remove NAs if True.
        
    Returns:
        float: The PRB coefficient.
    """
    assessed, sale_price = _ensure_arrays(assessed, sale_price)
    
    cleaned = _handle_na((assessed, sale_price), na_rm)
    if cleaned[0] is None: return np.nan
    assessed, sale_price = cleaned
    
    if len(assessed) < 2: return np.nan # Need at least 2 points for regression

    ratio = assessed / sale_price
    med_ratio = np.median(ratio)
    
    if med_ratio == 0: return np.nan

    # LHS: Percentage difference from median
    lhs = (ratio - med_ratio) / med_ratio
    
    # RHS: Proxy for value (log base 2)
    # The formula: log2( ( (AV / Median) + SP ) / 2 )
    inner_term = ((assessed / med_ratio) + sale_price) * 0.5
    
    # Filter out non-positive values for log
    valid_idx = inner_term > 0
    if not np.any(valid_idx): return np.nan
    
    lhs = lhs[valid_idx]
    rhs_inner = inner_term[valid_idx]
    
    rhs = np.log2(rhs_inner)
    
    # Linear Regression: lhs ~ rhs
    # np.polyfit returns [slope, intercept] for deg=1
    try:
        slope, intercept = np.polyfit(rhs, lhs, 1)
        return slope
    except:
        return np.nan

# MKI (Gini / Gini version of CCAO)
def _calc_gini(assessed, sale_price):
    """Helper to calculate Gini coefficients for KI/MKI."""
    # Create DataFrame for stable sorting
    df = pd.DataFrame({'av': assessed, 'sp': sale_price})
    
    # Sort by SP ascending, then AV descending (Standard from Quintos paper)
    df = df.sort_values(by=['sp', 'av'], ascending=[True, False])
    
    assessed_sorted = df['av'].values
    sale_sorted = df['sp'].values
    n = len(assessed_sorted)
    
    # Generate sequence 1 to n
    seq = np.arange(1, n + 1)
    
    # Gini Assessed
    av_sum_prod = np.sum(assessed_sorted * seq)
    av_sum = np.sum(assessed_sorted)
    g_assessed = (2 * av_sum_prod / av_sum) - (n + 1)
    gini_assessed = g_assessed / n
    
    # Gini Sale
    sp_sum_prod = np.sum(sale_sorted * seq)
    sp_sum = np.sum(sale_sorted)
    g_sale = (2 * sp_sum_prod / sp_sum) - (n + 1)
    gini_sale = g_sale / n
    
    return gini_assessed, gini_sale

def ki(assessed, sale_price, na_rm=False):
    """
    Calculate Kakwani Index (KI).
    KI = Gini(Assessed) - Gini(Sale)
    """
    assessed, sale_price = _ensure_arrays(assessed, sale_price)
    
    cleaned = _handle_na((assessed, sale_price), na_rm)
    if cleaned[0] is None: return np.nan
    assessed, sale_price = cleaned
    
    if len(assessed) == 0: return np.nan

    g_av, g_sp = _calc_gini(assessed, sale_price)
    return g_av - g_sp

def mki(assessed, sale_price, na_rm=False):
    """
    Calculate Modified Kakwani Index (MKI).
    MKI = Gini(Assessed) / Gini(Sale)
    """
    assessed, sale_price = _ensure_arrays(assessed, sale_price)
    
    cleaned = _handle_na((assessed, sale_price), na_rm)
    if cleaned[0] is None: return np.nan
    assessed, sale_price = cleaned
    
    if len(assessed) == 0: return np.nan

    g_av, g_sp = _calc_gini(assessed, sale_price)
    
    if g_sp == 0: return np.nan
    return g_av / g_sp



# -----------------------------------
#   1.3 Main Function of Metrics
# -----------------------------------

def compute_taxation_metrics(y_real, y_pred, scale="log", y_train=None):
        if scale == "log":
            y_real_log, y_pred_log = y_real, y_pred 
            y_real, y_pred = np.exp(y_real), np.exp(y_pred)
        else:
            y_real_log, y_pred_log = np.log(y_real), np.log(y_pred)
        metrics = dict()

        # 1. Accuracy metrics
        metrics["R2"] = r2_score(y_real, y_pred)
        if y_train.all() is not None:
            metrics["OOS R2"] = oos_r2_score(y_real, y_pred, y_train)
        metrics["R2 (log)"] = r2_score(y_real_log, y_pred_log)
        metrics["RMSE"] = _root_mse(y_real, y_pred)
        metrics["MAE"] = mean_absolute_error(y_real, y_pred)
        # metrics["MdAE"] = median_absolute_error(y_real, y_pred)
        metrics["MAPE"] = mean_absolute_percentage_error(y_real, y_pred)
        metrics["MdAPE"] = 100*median_absolute_error(y_real/y_pred, y_pred/y_pred)

        # # 1.5 Loss function
        # metrics["Loss"] = 

        # 2. My metrics of interest
        from sklearn.feature_selection import mutual_info_regression
        ratios = y_pred / y_real
        metrics["Corr(r,price)"] = np.corrcoef(ratios, y_real)[0,1]
        metrics["Corr(r,logprice)"] = np.corrcoef(ratios, y_real_log)[0,1]
        metrics["Slope(r~logy)"] = np.polyfit(y_real_log, ratios, 1)[0]

        metrics["Var ratio"] = np.var(ratios)
        metrics["Median ratio"] = np.median(ratios)
        metrics["Mean ratio"] = np.mean(ratios)
        metrics["W. Mean ratio"] = np.sum(y_pred)/np.sum(y_real)

        # 3. Taxation-Domain Specific Metrics
        median_ratio = np.median(ratios)
        # metrics["COD"] = 100/median_ratio*np.mean(np.abs(ratios - median_ratio))
        metrics["COD"] = cod(ratios, na_rm=True)
        metrics["COV_IAAO"] = cov_iaao(y_pred, y_real, na_rm=True) 
        metrics["VEI"] = vei(y_pred, y_real, na_rm=True)
        # metrics["PRD"] =  np.mean(ratios) / np.sum(y_pred) * np.sum(y_real) #(ratios @ y_real) * np.sum(y_real)
        metrics["PRD"] = prd(y_pred, y_real, na_rm=True)
        # PRB: Calculate the "Proxy" value first (Average of Sale Price and "Indicated" Value)
        # proxy_vals = 0.5 * (y_pred / median_ratio + y_real)
        # median_proxy = np.median(proxy_vals)
        # y_prb = (ratios - median_ratio) / median_ratio
        # x_prb = np.log2(proxy_vals) - np.log2(median_proxy) 
        # metrics["PRB"] = np.polyfit(x_prb, y_prb, 1)[0]
        metrics["PRB"] = prb(y_pred, y_real, na_rm=True)
        # metrics["MKI"] = mki(y_pred, y_real)
        metrics["MKI"] = mki(y_pred, y_real, na_rm=True)

        # Extra Correlation Metrics (can be ignored)

        # metrics["Spearman"] = stats.spearmanr(ratios, y_real).statistic
        # metrics["Kendall"] = stats.kendalltau(ratios, y_real).statistic
        # try:
        #     metrics["MI"] = mutual_info_regression(ratios.reshape(-1, 1), y_real)[0]
        # except Exception:
        #     ratios = ratios.to_numpy()
        #     y_real = y_real.to_numpy()
        #     metrics["MI"] = mutual_info_regression(ratios.reshape(-1, 1), y_real)[0]
        # print("Computed MI!!")
        # metrics["MIC"] = simplified_mic(ratios, y_real)
        # print("Computed MIC!")
        # metrics["RDC"] = rdc(ratios, y_real) # Randomized Dependence Coefficient
        # print("Computed RDC!")
        # import dcor # distance correlation 
        # d_corr, n_trials = 0, 10
        # for _ in range(n_trials):
        #     r_sub, y_sub = np.random.choice(ratios, replace=False, size=1000), np.random.choice(y_real, replace=False, size=1000)
        #     d_corr += dcor.distance_correlation(r_sub, y_sub)
        # metrics["dCorr(r,y)"] = d_corr/n_trials
        # metrics["dCorr(r,y)"] =  dcor.u_distance_correlation_sqr(ratios, y_real)
        # print("Computed dCorr!")
        # metrics["cos(r,y)"] = (ratios @ y_real) / (np.linalg.norm(ratios) * np.linalg.norm(y_real))

        return metrics














#========================================================================
#                           OLDER HELPERS
#========================================================================




# +-------------------------------------------------+
# |        PART 2: PLOTTING & VISUALIZATION         |
# +-------------------------------------------------+

def plot_tradeoff_analysis(results_df, percentages, num_groups=3, save_dir="img/tradeoff_analysis"):
    """
    Generates and saves a comprehensive set of plots for tradeoff analysis.

    Args:
        results_df (pd.DataFrame): DataFrame containing the collected statistics for each percentage.
        percentages (np.ndarray): The array of percentage increases used in the model.
        num_groups (int): The number of groups used for analysis.
        save_dir (str): The directory where plots will be saved.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    fairness_metrics = {
        'max_abs_deviation': "Max Absolute Deviation |r_max|",
        'max_diff_deviations': "Max Deviation Difference (max(r) - min(r))",
        'fairness_max_abs_diff_group_mae': "Max Abs Diff of Group MAEs"
    }
    
    accuracy_metrics = {'rmse': 'RMSE', 'r2': 'R^2'}

    # --- Plot Type 1 & 3: Dual-Axis Evolution and Direct Trade-off ---
    for acc_key, acc_label in accuracy_metrics.items():
        for fair_key, fair_label in fairness_metrics.items():
            
            # --- Dual-Axis Plot ---
            fig, ax1 = plt.subplots(figsize=(10, 6))
            ax1.plot(percentages, results_df[f'{acc_key}'], "--x", color='tab:red', label=acc_label)
            ax1.set_xlabel("Model Constraint: % Increase")
            ax1.set_ylabel(acc_label, color='tab:red')
            ax1.tick_params(axis='y', labelcolor='tab:red')
            
            ax2 = ax1.twinx()
            ax2.plot(percentages, results_df[f'{fair_key}'], ":x", color='tab:blue', label=fair_label)
            ax2.set_ylabel(fair_label, color='tab:blue')
            ax2.tick_params(axis='y', labelcolor='tab:blue')
            
            fig.tight_layout(rect=[0, 0, 0.9, 1])
            plt.title(f"{acc_label} vs. {fair_label} Evolution")
            plt.savefig(f"{save_dir}/{acc_key}_vs_{fair_key}_evolution.png", dpi=300)
            plt.close(fig)

            # --- Direct Trade-off Scatter Plot ---
            plt.figure(figsize=(8, 6))
            sc = plt.scatter(
                results_df[f'{fair_key}'], results_df[f'{acc_key}'],
                c=percentages, cmap="viridis", s=80, edgecolor="k"
            )
            plt.colorbar(sc, label="Model Constraint: % Increase")
            plt.xlabel(fair_label)
            plt.ylabel(acc_label)
            plt.title(f"Trade-off: {acc_label} vs. {fair_label}")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(f"{save_dir}/{acc_key}_vs_{fair_key}_tradeoff.png", dpi=300)
            plt.close()

    # --- Plot Type 2: Group Metric Evolution ---
    group_metrics_to_plot = ['rmse', 'r2', 'mae', 'max_deviation', 'min_deviation']

    for metric in group_metrics_to_plot:
        plt.figure(figsize=(10, 6))
        for i in range(num_groups):
            col_name = f'group_{i}_{metric}'
            avg_price_val = results_df[f"group_{i}_avg_price"].iloc[0]
            samples = results_df[f"group_{i}_samples"].iloc[0]

            avg_label = f" (avg={avg_price_val:.4f})" if avg_price_val is not None else ""
            plt.plot(percentages, results_df[col_name], "--o", label=f'Group {i}{avg_label}|{samples}', alpha=0.5)
        
        plt.xlabel("Model Constraint: % Increase")
        plt.ylabel(f"Group-wise {metric.upper()}")
        plt.title(f"Evolution of Group-wise {metric.upper()}")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/group_evolution_{metric}.png", dpi=300)
        plt.close()
        
    print(f"All analysis plots saved to '{save_dir}' directory.")


# relationship metrics (extra to kendall/pearson/corr)
from scipy.stats import rankdata
def rdc(x, y, k=20, s=1/6.0, n=1, regularizer=1e-5):
    """
    Computes the Randomized Dependence Coefficient with Regularization.
    
    Args:
        x, y: 1D or 2D numpy arrays.
        k: Number of random non-linear features.
        s: Scale of random weights.
        n: Number of repetitions.
        regularizer: Ridge penalty (epsilon) to prevent overfitting on noise.
    """
    if n > 1:
        values = []
        for i in range(n):
            values.append(rdc(x, y, k, s, 1, regularizer))
        return np.median(values)

    # 1. Reshape and Rank Transform (Copula)
    if len(x.shape) == 1: x = x.reshape(-1, 1)
    if len(y.shape) == 1: y = y.reshape(-1, 1)
    
    # Transform to uniform [0,1]
    x = rankdata(x, axis=0) / x.shape[0]
    y = rankdata(y, axis=0) / y.shape[0]

    # 2. Random Non-Linear Projection
    # W ~ Normal(0, s)
    Rx = np.random.normal(0, s, (x.shape[1], k))
    Ry = np.random.normal(0, s, (y.shape[1], k))
    bx = np.random.uniform(0, 2*np.pi, k)
    by = np.random.uniform(0, 2*np.pi, k)

    X_feat = np.sin(x @ Rx + bx)
    Y_feat = np.sin(y @ Ry + by)

    # 3. Center the features
    X_feat -= X_feat.mean(axis=0)
    Y_feat -= Y_feat.mean(axis=0)

    # 4. Regularized CCA (The Fix)
    # Instead of raw SVD, we use the covariance matrices with a ridge penalty.
    # Metric = Max Eigenvalue of (Cxx^-1/2 * Cxy * Cyy^-1/2)
    
    # Compute Covariance Matrices
    # Note: We use N-1 normalization for unbiased estimator
    N = X_feat.shape[0]
    Cxx = (X_feat.T @ X_feat) / (N - 1) + regularizer * np.eye(k)
    Cyy = (Y_feat.T @ Y_feat) / (N - 1) + regularizer * np.eye(k)
    Cxy = (X_feat.T @ Y_feat) / (N - 1)
    
    # Compute Inverse Square Roots (using Cholesky or Eigendecomposition)
    # inv(sqrt(Cxx))
    def power_inverse(Matrix):
        # M = U S U^T -> M^-1/2 = U S^-1/2 U^T
        vals, vecs = np.linalg.eigh(Matrix)
        # vals can be slightly negative due to numerics, clip at 0
        vals = np.maximum(vals, 1e-15)
        return vecs @ np.diag(1.0 / np.sqrt(vals)) @ vecs.T

    Cxx_inv_sqrt = power_inverse(Cxx)
    Cyy_inv_sqrt = power_inverse(Cyy)
    
    # The Omega matrix
    Omega = Cxx_inv_sqrt @ Cxy @ Cyy_inv_sqrt
    
    # Singular values of Omega are the canonical correlations
    # The RDC is the largest one
    correlations = np.linalg.svd(Omega, compute_uv=False)
    
    return correlations[0]


from sklearn.metrics import mutual_info_score

def simplified_mic(x, y, alpha=0.6):
    """
    Computes an approximation of the Maximal Information Coefficient (MIC).
    
    Parameters:
    x, y : arrays, shape (n_samples,)
    alpha : float, exponent for max grid size (B = N^alpha)
    """
    n = len(x)
    # 1. Calculate max number of cells (B) according to paper
    B = n ** alpha
    
    max_score = 0.0
    
    # 2. Heuristic: Search grid sizes (nx, ny)
    # We loop through different x-bin counts and y-bin counts
    # Optimization: We limit bins to roughly sqrt(B) to keep total cells < B
    max_bins = int(np.ceil(B**0.5))
    
    # Pre-sort for efficient binning logic if needed, 
    # but for simplicity we use histogram counting here.
    
    for nx in range(2, max_bins + 1):
        for ny in range(2, max_bins + 1):
            
            # Constraint from paper: Total cells nx * ny < B
            if nx * ny > B:
                continue
                
            # 3. Create Grids
            # True MIC optimizes bin edges (dynamic programming). 
            # Simplified MIC uses Equi-width or Equi-frequency bins.
            # We use histogram2d (Equi-width) for speed/simplicity.
            # (Ideally, you want equi-frequency, but that is slower in pure python loops)
            
            c_xy = np.histogram2d(x, y, bins=[nx, ny])[0]
            
            # 4. Compute Mutual Information on this grid
            # mi_score calculates I(X;Y)
            mi = mutual_info_score(None, None, contingency=c_xy)
            
            # 5. Normalize: MIC_norm = I(X;Y) / log(min(nx, ny))
            denom = np.log(min(nx, ny))
            
            # Handle 0 division or log(1)
            if denom > 0:
                score = mi / denom
                if score > max_score:
                    max_score = score
                    
    return max_score



################################################################################################
# CCAO's package translation to python
# original source (.R code): https://github.com/ccao-data/assessr/blob/master/R/formulas.R
################################################################################################ 


# =============================================================================
# IAAO 2025 Exposure Draft guidance (Table 7 + glossary + VEI/PRB sections)
# =============================================================================
# NOTE: These are *guidance ranges* used in ratio studies. They are not a substitute for
# office policy, sale validation, time adjustments, or statistical significance testing.

# Valuation level guidance (often referenced as "acceptable level")
IAAO_LEVEL_RANGE: Tuple[float, float] = (0.90, 1.10)

# PRD interpretive guidance
IAAO_PRD_RANGE: Tuple[float, float] = (0.98, 1.03)

# PRB interpretive guidance (glossary guidance)
IAAO_PRB_RANGE: Tuple[float, float] = (-0.05, 0.05)
IAAO_PRB_UNACCEPTABLE_RANGE: Tuple[float, float] = (-0.10, 0.10)

# VEI acceptable point-estimate range
IAAO_VEI_RANGE: Tuple[float, float] = (-10.0, 10.0)  # percent

# COD guidance ranges (Table 7). Choose the closest match to your use-case.
IAAO_COD_RANGES: Dict[str, Tuple[float, float]] = {
    "Residential Improved": (5.0, 15.0),
    "Newer/Homogeneous Residential": (5.0, 10.0),
    "Rural/Seasonal/Mobile Homes": (5.0, 20.0),
    "Multi-Family": (5.0, 20.0),
    "Rural Jurisdictions": (5.0, 25.0),
    "Commercial/Industrial": (5.0, 25.0),
    "Commercial/Industrial Condominiums": (5.0, 15.0),
    "Less Active/Small Sample": (5.0, 30.0),
    "Vacant Land": (5.0, 20.0),
    "Vacant Land Less Active/Small Sample": (5.0, 25.0),
    "Agricultural Land": (5.0, 25.0),
}


# =============================================================================
# Pretty printing helpers
# =============================================================================

def _fmt_num(x: Any, digits: int = 4, comma: bool = False) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "nan"
    if isinstance(x, (int, np.integer)):
        return f"{int(x):,}" if comma else str(int(x))
    if comma:
        return f"{float(x):,.{digits}f}"
    return f"{float(x):.{digits}f}"


def _fmt_range(rng: Optional[Tuple[float, float]], digits: int = 2, pct: bool = False) -> str:
    if rng is None:
        return "—"
    lo, hi = rng
    if pct:
        return f"[{lo:.{digits}f}, {hi:.{digits}f}]%"
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def _in_range(x: float, rng: Tuple[float, float]) -> bool:
    return (x >= rng[0]) and (x <= rng[1])


def _print_metric_table(rows, title: str):
    """rows: list of (metric, value_str, iaao_expected_str, interpretation_str)"""
    col_names = ("Metric", "Value", "IAAO expected", "Interpretation")
    cols = list(zip(*([col_names] + rows)))
    widths = [max(len(str(v)) for v in col) for col in cols]

    def _line(ch: str = "-"):
        return ch * (sum(widths) + 3 * (len(widths) - 1))

    print(_line("="))
    print(title)
    print(_line("="))
    header = "  ".join(
        [
            f"{col_names[i]:<{widths[i]}}" if i == 0 else f"{col_names[i]:>{widths[i]}}"
            for i in range(len(widths))
        ]
    )
    print(header)
    print(_line("-"))
    for r in rows:
        metric, value, expected, interp = r
        line = "  ".join(
            [
                f"{metric:<{widths[0]}}",
                f"{value:>{widths[1]}}",
                f"{expected:>{widths[2]}}",
                f"{interp:<{widths[3]}}",
            ]
        )
        print(line)
    print(_line("="))


def _print_df_table(df: pd.DataFrame, title: str, float_digits: int = 4):
    """ASCII pretty print for small DataFrames (3–10 models)."""
    if df is None or df.empty:
        print(title)
        print("(empty)")
        return

    disp = df.copy()

    def _format_cell(v):
        if v is None or (isinstance(v, float) and not np.isfinite(v)):
            return "nan"
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}"
        if isinstance(v, float):
            return f"{v:.{float_digits}f}"
        return str(v)

    disp = disp.applymap(_format_cell)

    headers = list(disp.columns)
    rows = disp.values.tolist()

    widths = [len(h) for h in headers]
    for row in rows:
        for j, cell in enumerate(row):
            widths[j] = max(widths[j], len(cell))

    def _line(ch="-"):
        return ch * (sum(widths) + 3 * (len(widths) - 1))

    print(_line("="))
    print(title)
    print(_line("="))
    print("  ".join([f"{headers[j]:<{widths[j]}}" for j in range(len(headers))]))
    print(_line("-"))
    for row in rows:
        print("  ".join([f"{row[j]:<{widths[j]}}" for j in range(len(headers))]))
    print(_line("="))


# =============================================================================
# Flagging and interpretation helpers
# =============================================================================

def _flag_from_range(x: float, rng: Tuple[float, float], *, warn_margin: float = 0.02) -> str:
    """PASS if inside range. WARN if within a small absolute margin outside. FAIL otherwise.

    warn_margin is *not* IAAO policy; it's a readability heuristic for model comparison.
    """
    if not np.isfinite(x):
        return "NA"
    lo, hi = rng
    if lo <= x <= hi:
        return "PASS"
    if (lo - warn_margin) <= x <= (hi + warn_margin):
        return "WARN"
    return "FAIL"


def _interp_prd(prd_val: float) -> str:
    if not np.isfinite(prd_val):
        return "—"
    if prd_val > IAAO_PRD_RANGE[1]:
        return "Regressive tendency"
    if prd_val < IAAO_PRD_RANGE[0]:
        return "Progressive tendency"
    return "Within guidance"


def _interp_prb(prb_val: float) -> str:
    if not np.isfinite(prb_val):
        return "—"
    if prb_val < 0:
        return "Regressive (ratios fall as value rises)"
    if prb_val > 0:
        return "Progressive (ratios rise as value rises)"
    return "No bias"


def _interp_vei(vei: float) -> str:
    if not np.isfinite(vei):
        return "—"
    if _in_range(vei, IAAO_VEI_RANGE):
        return "Acceptable (pt est.)"
    return "Regressive" if vei < 0 else "Progressive"


def _equity_verdict(level_flag: str, uniformity_flag: str, vertical_flag: str) -> str:
    # Conservative: if any FAIL then overall FAIL; if any WARN then WARN; else PASS.
    flags = [level_flag, uniformity_flag, vertical_flag]
    if "FAIL" in flags:
        return "FAIL"
    if "WARN" in flags:
        return "WARN"
    if "PASS" in flags:
        return "PASS"
    return "NA"


# =============================================================================
# IAAO-style PRB (Coefficient of Price-Related Bias)
# =============================================================================

def iaao_prb(assessed, sale_price, na_rm: bool = False) -> float:
    """IAAO-style PRB, following the Appendix-D style construction.

    Steps (IAAO):
      1) Ratio ASR = AV / SP
      2) Median ratio Med
      3) Value proxy: Value = 0.5*SP + 0.5*(AV/Med)
      4) Indep var: ln(Value)/ln(2)   (each +1 corresponds to a 100% increase)
      5) Dep var: (ASR - Med)/Med
      6) PRB = slope in OLS regression: Dep = a + PRB * Indep

    Returns:
      PRB (float). Negative => regressive; Positive => progressive.
    """
    av = np.asarray(assessed, dtype=float).reshape(-1)
    sp = np.asarray(sale_price, dtype=float).reshape(-1)

    if av.shape != sp.shape:
        raise ValueError(f"assessed and sale_price must have same shape. Got {av.shape} vs {sp.shape}.")

    mask = np.isfinite(av) & np.isfinite(sp) & (av > 0) & (sp > 0)
    av = av[mask]
    sp = sp[mask]

    if av.size < 3:
        return np.nan

    with np.errstate(divide="ignore", invalid="ignore"):
        r = av / sp

    r = r[np.isfinite(r)]
    if r.size < 3:
        return np.nan

    med = float(np.median(r))
    if not np.isfinite(med) or med == 0:
        return np.nan

    # Value proxy (equal weight SP and AV/Med)
    value = 0.5 * sp + 0.5 * (av / med)
    mask2 = np.isfinite(value) & (value > 0)
    value = value[mask2]
    r2 = r[mask2]

    if value.size < 3:
        return np.nan

    x = np.log(value) / float(np.log(2.0))
    y = (r2 - med) / med

    mask3 = np.isfinite(x) & np.isfinite(y)
    x = x[mask3]
    y = y[mask3]
    if x.size < 3:
        return np.nan

    # OLS slope with intercept: slope = cov(x,y) / var(x)
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)
    denom = float(np.sum(x_centered ** 2))
    if denom == 0:
        return np.nan

    slope = float(np.sum(x_centered * y_centered) / denom)
    return slope


# =============================================================================
# Main report function
# =============================================================================

def analyze_financial_performance(
    y_true,
    y_pred,
    n_quantiles: int = 4,
    show_plots: bool = True,
    vei_n_groups: Optional[int] = None,
    return_df: bool = False,
    iaao_property_class: str = "Residential Improved",
    verbosity: int = 2,
):
    """Compute a compact accuracy + ratio-study style diagnostic report.

    Inputs must be on the **price scale** and positive.

    verbosity:
      0 -> compute and return stats only (no printing)
      1 -> print executive summary + scorecard-style tables (no quantile breakdown)
      2 -> full report (tables + quantile breakdown; plots controlled by show_plots)
    """

    # ----------------------------
    # 0) Input hygiene
    # ----------------------------
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if y_true.shape != y_pred.shape:
        raise ValueError(f"y_true and y_pred must have same shape. Got {y_true.shape} vs {y_pred.shape}.")

    mask = np.isfinite(y_true) & np.isfinite(y_pred) & (y_true > 0) & (y_pred > 0)
    dropped = int(y_true.size - mask.sum())
    if dropped > 0 and verbosity >= 1:
        print(f"[WARN] Dropping {dropped} rows with non-finite or non-positive Actual/Predicted values.")

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    n = int(y_true.size)
    if n == 0:
        raise ValueError("No valid observations after filtering (need positive, finite Actual and Predicted).")

    ratio = y_pred / y_true
    log_ratio = np.log(y_pred) - np.log(y_true)  # log(pred/actual) (kept for optional future use)

    df = pd.DataFrame(
        {
            "Actual": y_true,
            "Predicted": y_pred,
            "Error": y_pred - y_true,
            "Abs_Error": np.abs(y_pred - y_true),
            "Ratio": ratio,
            "LogRatio": log_ratio,
        }
    )

    # ----------------------------
    # 1) Global metrics
    # ----------------------------
    # Accuracy (price scale)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    ape = np.abs((y_pred - y_true) / y_true)
    mape = float(np.mean(ape) * 100.0)
    mdape = float(np.median(ape) * 100.0)

    rmse = float(np.sqrt(np.mean((y_pred - y_true) ** 2)))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    bias = float(np.mean(y_pred - y_true))

    # Ratio-study metrics
    median_ratio = float(np.median(ratio))
    mean_ratio = float(np.mean(ratio))
    weighted_mean_ratio = float(np.sum(y_pred) / np.sum(y_true))

    avg_abs_dev_from_median = float(np.mean(np.abs(ratio - median_ratio)))
    cod = 100.0 * avg_abs_dev_from_median / median_ratio if median_ratio != 0 else np.nan

    prd_val = mean_ratio / weighted_mean_ratio if weighted_mean_ratio != 0 else np.nan

    prb_val = iaao_prb(assessed=y_pred, sale_price=y_true)

    # Dispersion helpers
    std_ratio = float(np.std(ratio, ddof=1)) if n > 1 else 0.0
    cov_ratio = (std_ratio / mean_ratio) if mean_ratio != 0 else np.nan
    q25, q75 = np.percentile(ratio, [25, 75])
    iqr_ratio = float(q75 - q25)

    # ----------------------------
    # 2) VEI (point estimate + profile)
    # ----------------------------
    # IAAO 2025 exposure draft Sec 8.2.1:
    #   proxy_i = 0.5*SP_i + 0.5*(AV_i / sample_median_ratio)
    #   sort proxy asc; split ratios into halves/quartiles/deciles based on n
    #   VEI = 100 * (median_lastPG - median_firstPG) / sample_median_ratio

    if vei_n_groups is None:
        if n < 20:
            k_vei = None
        elif n <= 50:
            k_vei = 2
        elif n <= 500:
            k_vei = 4
        else:
            k_vei = 10
    else:
        k_vei = int(vei_n_groups)
        if k_vei <= 1:
            k_vei = None

    vei = np.nan
    vei_groups_used = 0
    vei_first_median = np.nan
    vei_last_median = np.nan
    vei_pg_medians: List[float] = []
    vei_pg_counts: List[int] = []

    if (k_vei is not None) and np.isfinite(median_ratio) and (median_ratio != 0):
        proxy = 0.5 * y_true + 0.5 * (y_pred / median_ratio)
        df["_MV_Proxy"] = proxy

        # exact "sort then split" (stable)
        order = np.argsort(proxy, kind="mergesort")
        chunks = np.array_split(np.arange(n), k_vei)

        pg_id = np.empty(n, dtype=int)
        for g, chunk in enumerate(chunks):
            pg_id[order[chunk]] = g

        df["_VEI_PG"] = pd.Categorical(pg_id + 1, categories=list(range(1, k_vei + 1)), ordered=True)

        pg_stats = df.groupby("_VEI_PG", observed=False)["Ratio"].agg(["median", "size"])
        vei_pg_medians = pg_stats["median"].to_list()
        vei_pg_counts = pg_stats["size"].astype(int).to_list()
        vei_groups_used = int(len(vei_pg_medians))

        # IAAO note: need >=10 sales per group
        if (vei_groups_used >= 2) and all(c >= 10 for c in vei_pg_counts):
            vei_first_median = float(vei_pg_medians[0])
            vei_last_median = float(vei_pg_medians[-1])
            vei = 100.0 * (vei_last_median - vei_first_median) / float(median_ratio)
        elif verbosity >= 1:
            print(
                f"[WARN] VEI not computed because some VEI groups have <10 observations. "
                f"Group sizes: {vei_pg_counts}"
            )

    # ----------------------------
    # 3) Quantile breakdown (by Actual) – diagnostic stratification
    # ----------------------------
    quantile_stats = None
    if n_quantiles and n_quantiles >= 2:
        try:
            q = pd.qcut(df["Actual"], q=n_quantiles, duplicates="drop")
        except ValueError:
            q = pd.Series([pd.Interval(-np.inf, np.inf)] * len(df), index=df.index)

        df["Quantile"] = pd.Categorical(q.cat.codes + 1, ordered=True) if hasattr(q, "cat") else 1

        def compute_group_metrics(group: pd.DataFrame) -> pd.Series:
            r = group["Ratio"].to_numpy(dtype=float)
            med_r = float(np.median(r))
            mean_r = float(np.mean(r))
            w_mean_r = float(group["Predicted"].sum() / group["Actual"].sum())

            aad = float(np.mean(np.abs(r - med_r)))
            group_cod = (aad / med_r) * 100.0 if med_r != 0 else np.nan
            group_prd = mean_r / w_mean_r if w_mean_r != 0 else np.nan
            group_prb = iaao_prb(assessed=group["Predicted"].to_numpy(), sale_price=group["Actual"].to_numpy())

            mape_g = float(np.mean(np.abs((group["Predicted"] - group["Actual"]) / group["Actual"])) * 100.0)

            return pd.Series(
                {
                    "Count": len(group),
                    "Min ($)": float(group["Actual"].min()),
                    "Max ($)": float(group["Actual"].max()),
                    "Mean Error ($)": float(group["Error"].mean()),
                    "MAPE (%)": mape_g,
                    "Mean Ratio": mean_r,
                    "Median Ratio": med_r,
                    "COD": group_cod,
                    "PRD": group_prd,
                    "PRB": group_prb,
                }
            )

        quantile_stats = df.groupby("Quantile", observed=False).apply(compute_group_metrics, include_groups=False)

    # ----------------------------
    # 4) Pack results
    # ----------------------------
    global_stats = {
        "Count": n,
        "Mean Price ($)": float(df["Actual"].mean()),
        "R2": float(r2),
        "MAE ($)": mae,
        "RMSE ($)": rmse,
        "MAPE (%)": mape,
        "MdAPE (%)": mdape,
        "Bias ($)": bias,
        "Median Ratio": median_ratio,
        "Mean Ratio": mean_ratio,
        "Weighted Mean Ratio": weighted_mean_ratio,
        "COD": cod,
        "Std(Ratio)": std_ratio,
        "COV(Ratio)": cov_ratio,
        "IQR(Ratio)": iqr_ratio,
        "PRD": prd_val,
        "PRB": prb_val,
        "VEI": vei,
        "VEI Groups Used": vei_groups_used,
        "VEI First PG Median": vei_first_median,
        "VEI Last PG Median": vei_last_median,
        "VEI k (requested)": (k_vei if k_vei is not None else np.nan),
        "VEI PG Medians": vei_pg_medians,
        "VEI PG Counts": vei_pg_counts,
        # optional diagnostics kept for downstream analysis
        "MAE(LogRatio)": float(np.mean(np.abs(log_ratio))),
        "RMSE(LogRatio)": float(np.sqrt(np.mean(log_ratio ** 2))),
        "Bias(LogRatio)": float(np.mean(log_ratio)),
        "Std(LogRatio)": float(np.std(log_ratio, ddof=1)) if n > 1 else 0.0,
    }

    # ----------------------------
    # 5) Printing (executive summary + tables)
    # ----------------------------
    if verbosity >= 1:
        cod_range = IAAO_COD_RANGES.get(iaao_property_class)

        level_flag = _flag_from_range(median_ratio, IAAO_LEVEL_RANGE)
        wmr_flag = _flag_from_range(weighted_mean_ratio, IAAO_LEVEL_RANGE)
        # conservative: take worse of the two for "level"
        if "FAIL" in (level_flag, wmr_flag):
            level_overall = "FAIL"
        elif "WARN" in (level_flag, wmr_flag):
            level_overall = "WARN"
        else:
            level_overall = "PASS"

        uniformity_flag = "NA"
        if cod_range is not None and np.isfinite(cod):
            uniformity_flag = _flag_from_range(cod, cod_range, warn_margin=1.0)
        elif np.isfinite(cod):
            uniformity_flag = "PASS" if cod < 15 else "WARN"  # fallback heuristic

        prd_flag = _flag_from_range(prd_val, IAAO_PRD_RANGE)
        prb_flag = _flag_from_range(prb_val, IAAO_PRB_RANGE, warn_margin=0.01)
        vei_flag = _flag_from_range(vei, IAAO_VEI_RANGE, warn_margin=2.0) if np.isfinite(vei) else "NA"

        # vertical equity: worst among PRD/PRB/VEI (ignoring NA)
        vertical_flags = [prd_flag, prb_flag] + ([vei_flag] if vei_flag != "NA" else [])
        if "FAIL" in vertical_flags:
            vertical_flag = "FAIL"
        elif "WARN" in vertical_flags:
            vertical_flag = "WARN"
        else:
            vertical_flag = "PASS"

        verdict = _equity_verdict(level_overall, uniformity_flag, vertical_flag)

        # Executive bullets
        print("=" * 99)
        print("EXECUTIVE SUMMARY")
        print("=" * 99)
        print(
            f"Accuracy: R2={_fmt_num(r2,4)}, RMSE={_fmt_num(rmse,2,comma=True)}, MAE={_fmt_num(mae,2,comma=True)}, MAPE={_fmt_num(mape,2)}%"
        )
        print(
            f"Level: Median={_fmt_num(median_ratio,4)} ({level_flag}), WMR={_fmt_num(weighted_mean_ratio,4)} ({wmr_flag}) | "
            f"Uniformity: COD={_fmt_num(cod,2)} ({uniformity_flag}) | "
            f"Vertical: PRD={_fmt_num(prd_val,4)} ({prd_flag}, {_interp_prd(prd_val)}), "
            f"PRB={_fmt_num(prb_val,4)} ({prb_flag}, {_interp_prb(prb_val)}), "
            f"VEI={_fmt_num(vei,2)} ({vei_flag}, {_interp_vei(vei)})"
        )
        print(f"Overall equity verdict (heuristic): {verdict}")

        # A) Accuracy table (PRICE SCALE ONLY)
        acc_rows = [
            ("Count", _fmt_num(n, digits=0, comma=True), "—", "Valid observations"),
            ("Mean Price ($)", _fmt_num(global_stats["Mean Price ($)"], digits=2, comma=True), "—", "Scale of target"),
            ("R2", _fmt_num(r2, digits=4), "—", "Closer to 1 is better"),
            ("MAE ($)", _fmt_num(mae, digits=2, comma=True), "—", "Lower is better"),
            ("RMSE ($)", _fmt_num(rmse, digits=2, comma=True), "—", "Lower is better"),
            ("MAPE (%)", _fmt_num(mape, digits=2), "—", "Lower is better"),
            ("MdAPE (%)", _fmt_num(mdape, digits=2), "—", "Robust to outliers"),
            ("Bias ($)", _fmt_num(bias, digits=2, comma=True), "—", "Near 0 is better"),
        ]
        _print_metric_table(acc_rows, title="MODEL ACCURACY (PRICE SCALE)")

        # B) IAAO-style table
        iaao_rows = [
            ("Median Ratio", _fmt_num(median_ratio, 4), _fmt_range(IAAO_LEVEL_RANGE, 2), f"{level_flag}: level"),
            ("Weighted Mean Ratio", _fmt_num(weighted_mean_ratio, 4), _fmt_range(IAAO_LEVEL_RANGE, 2), f"{wmr_flag}: level"),
            (
                f"COD (%) [{iaao_property_class}]",
                _fmt_num(cod, 2),
                _fmt_range(cod_range, 1, pct=False) if cod_range else "(see Table 7)",
                f"{uniformity_flag}: uniformity",
            ),
            ("PRD", _fmt_num(prd_val, 4), _fmt_range(IAAO_PRD_RANGE, 2), f"{prd_flag}: {_interp_prd(prd_val)}"),
            ("PRB", _fmt_num(prb_val, 4), _fmt_range(IAAO_PRB_RANGE, 2), f"{prb_flag}: {_interp_prb(prb_val)}"),
            ("VEI (%)", _fmt_num(vei, 2), _fmt_range(IAAO_VEI_RANGE, 0, pct=True), f"{vei_flag}: {_interp_vei(vei)}"),
            ("COV(Ratio)", _fmt_num(cov_ratio, 4), "—", "Lower is tighter (relative disp.)"),
            ("IQR(Ratio)", _fmt_num(iqr_ratio, 4), "—", "Smaller = tighter ratios"),
        ]
        _print_metric_table(iaao_rows, title="IAAO-STYLE RATIO STUDY METRICS (Pred/Actual as AV/SP)")

        if np.isfinite(vei):
            print(
                f"VEI details: k={k_vei}, groups_used={vei_groups_used}, "
                f"firstPG_median={_fmt_num(vei_first_median, 4)}, lastPG_median={_fmt_num(vei_last_median, 4)}"
            )

        # C) Quantile table (only if full verbosity)
        if verbosity >= 2 and quantile_stats is not None:
            print("" + "=" * 75)
            print(f"DIAGNOSTICS BY PRICE QUANTILE (by Actual; n_quantiles={n_quantiles})")
            print("=" * 75)
            with pd.option_context(
                "display.max_columns", None,
                "display.width", 1200,
                "display.float_format", lambda x: f"{x:,.3f}",
            ):
                print(quantile_stats)

    # ----------------------------
    # 6) Plots
    # ----------------------------
    if show_plots:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))

        sns.boxplot(data=df, x="Quantile", y="Ratio", ax=axes[0])
        axes[0].axhline(1.0, color="red", linestyle="--", linewidth=2)
        axes[0].set_title(f"Pred/Actual Ratio by Actual-Quantile (q={n_quantiles})")
        axes[0].set_ylabel("Ratio (Predicted / Actual)")

        sns.scatterplot(data=df, x="Actual", y="Predicted", hue="Quantile", alpha=0.35, ax=axes[1])
        max_limit = float(max(df["Actual"].max(), df["Predicted"].max()))
        axes[1].plot([0, max_limit], [0, max_limit], "r--", linewidth=2)
        axes[1].set_title("Predicted vs Actual")
        axes[1].set_xlabel("Actual Price")
        axes[1].set_ylabel("Predicted Price")
        axes[1].legend(bbox_to_anchor=(1.05, 1), loc="upper left")

        plt.tight_layout()
        plt.show()

        # VEI profile plot (helps convince non-technical audiences)
        if isinstance(global_stats.get("VEI PG Medians"), list) and len(global_stats["VEI PG Medians"]) >= 2:
            plt.figure(figsize=(8, 4))
            xs = np.arange(1, len(global_stats["VEI PG Medians"]) + 1)
            plt.plot(xs, global_stats["VEI PG Medians"], marker="o")
            plt.axhline(global_stats["Median Ratio"], linestyle="--", linewidth=2)
            plt.title("VEI Percentile-Group Medians (sorted by IAAO market value proxy)")
            plt.xlabel("Percentile Group (low proxy → high proxy)")
            plt.ylabel("Median Ratio (AV/SP)")
            plt.tight_layout()
            plt.show()

    if return_df:
        return global_stats, quantile_stats, df

    return global_stats, quantile_stats


# =============================================================================
# Multi-model comparison helpers (3–5 models)
# =============================================================================

def model_label(model: Any) -> str:
    """Best-effort concise label for sklearn / custom wrappers."""
    cls = model.__class__.__name__

    bits = [cls]
    if hasattr(model, "rho"):
        bits.append(f"rho={getattr(model,'rho')}")
    if hasattr(model, "ratio_mode"):
        bits.append(f"mode={getattr(model,'ratio_mode')}")
    if hasattr(model, "rho_cov"):
        bits.append(f"rho_cov={getattr(model,'rho_cov')}")
    if hasattr(model, "rho_disp"):
        bits.append(f"rho_disp={getattr(model,'rho_disp')}")
    if hasattr(model, "n_bins"):
        bits.append(f"bins={getattr(model,'n_bins')}")

    return "(" + ", ".join(bits) + ")" if len(bits) > 1 else cls


def build_scorecard_records(
    label: str,
    global_stats: Dict[str, Any],
    *,
    mode: str = "",
) -> Dict[str, Any]:
    """Normalize global_stats into a flat record for scorecard comparison."""
    rec = {
        "Model": label,
        "Mode": mode,
        # Accuracy (price scale)
        "R2": float(global_stats.get("R2", np.nan)),
        "RMSE": float(global_stats.get("RMSE ($)", np.nan)),
        "MAE": float(global_stats.get("MAE ($)", np.nan)),
        "MAPE": float(global_stats.get("MAPE (%)", np.nan)),
        # Level / equity
        "Median": float(global_stats.get("Median Ratio", np.nan)),
        "WMR": float(global_stats.get("Weighted Mean Ratio", np.nan)),
        "COD": float(global_stats.get("COD", np.nan)),
        "COV": float(global_stats.get("COV(Ratio)", np.nan)),
        "PRD": float(global_stats.get("PRD", np.nan)),
        "PRB": float(global_stats.get("PRB", np.nan)),
        "VEI": float(global_stats.get("VEI", np.nan)),
    }
    return rec


def make_comparison_scorecard(
    records: List[Dict[str, Any]],
    *,
    baseline_model: Optional[str] = None,
    iaao_property_class: str = "Residential Improved",
) -> pd.DataFrame:
    """Create a compact comparison table for 3–5 models, including deltas vs baseline."""
    df = pd.DataFrame(records).copy()
    if df.empty:
        return df

    # Determine baseline
    if baseline_model is None:
        baseline_idx = 0
    else:
        matches = df.index[df["Model"].astype(str) == str(baseline_model)].tolist()
        baseline_idx = matches[0] if matches else 0

    base = df.loc[baseline_idx]

    # Deltas vs baseline
    df["ΔR2"] = df["R2"] - float(base["R2"])
    df["ΔRMSE"] = df["RMSE"] - float(base["RMSE"])
    df["ΔMAE"] = df["MAE"] - float(base["MAE"])
    df["ΔMAPE"] = df["MAPE"] - float(base["MAPE"])

    df["ΔCOD"] = df["COD"] - float(base["COD"])
    df["ΔPRD"] = df["PRD"] - float(base["PRD"])
    df["ΔPRB"] = df["PRB"] - float(base["PRB"])
    df["ΔVEI"] = df["VEI"] - float(base["VEI"])
    df["ΔCOV"] = df["COV"] - float(base["COV"])

    # Flags
    cod_rng = IAAO_COD_RANGES.get(iaao_property_class)
    df["LevelFlag"] = df["Median"].apply(lambda x: _flag_from_range(float(x), IAAO_LEVEL_RANGE))
    df["WMRFlag"] = df["WMR"].apply(lambda x: _flag_from_range(float(x), IAAO_LEVEL_RANGE))

    df["UniformityFlag"] = df["COD"].apply(
        lambda x: _flag_from_range(float(x), cod_rng, warn_margin=1.0)
        if cod_rng
        else ("PASS" if float(x) < 15 else "WARN")
    )

    df["PRDFlag"] = df["PRD"].apply(lambda x: _flag_from_range(float(x), IAAO_PRD_RANGE))
    df["PRBFlag"] = df["PRB"].apply(lambda x: _flag_from_range(float(x), IAAO_PRB_RANGE, warn_margin=0.01))
    df["VEIFlag"] = df["VEI"].apply(
        lambda x: _flag_from_range(float(x), IAAO_VEI_RANGE, warn_margin=2.0) if np.isfinite(float(x)) else "NA"
    )

    # Overall verdict
    def _level_overall(row):
        if "FAIL" in (row["LevelFlag"], row["WMRFlag"]):
            return "FAIL"
        if "WARN" in (row["LevelFlag"], row["WMRFlag"]):
            return "WARN"
        if "PASS" in (row["LevelFlag"], row["WMRFlag"]):
            return "PASS"
        return "NA"

    def _vertical_overall(row):
        flags = [row["PRDFlag"], row["PRBFlag"]]
        if row["VEIFlag"] != "NA":
            flags.append(row["VEIFlag"])
        if "FAIL" in flags:
            return "FAIL"
        if "WARN" in flags:
            return "WARN"
        if "PASS" in flags:
            return "PASS"
        return "NA"

    df["Level"] = df.apply(_level_overall, axis=1)
    df["Vertical"] = df.apply(_vertical_overall, axis=1)
    df["EquityVerdict"] = df.apply(lambda r: _equity_verdict(r["Level"], r["UniformityFlag"], r["Vertical"]), axis=1)

    cols = [
        "Model",
        "Mode",
        # Accuracy (price)
        "R2",
        "RMSE",
        "MAE",
        "MAPE",
        "ΔR2",
        "ΔRMSE",
        "ΔMAE",
        "ΔMAPE",
        # Equity
        "Median",
        "WMR",
        "COD",
        "COV",
        "PRD",
        "PRB",
        "VEI",
        "ΔCOD",
        "ΔCOV",
        "ΔPRD",
        "ΔPRB",
        "ΔVEI",
        # Scorecard flags
        "Level",
        "UniformityFlag",
        "Vertical",
        "EquityVerdict",
    ]
    cols = [c for c in cols if c in df.columns]
    return df[cols].copy()


def print_comparison_scorecard(
    df_score: pd.DataFrame,
    title: str = "MODEL COMPARISON SCORECARD",
    *,
    transpose: bool = True,
    model_name_maxlen: int = 55,
    float_digits: int = 4,
):
    """Pretty-print the scorecard for 3–5 models + legend.

    If transpose=True (default), prints *models as columns* and *metrics as rows*,
    which is usually easier to read when the scorecard has many metrics.
    """
    if df_score is None or df_score.empty:
        print(title)
        print("(empty)")
        return

    disp = df_score.copy()

    # Build a compact model key (Model + optional Mode)
    if "Mode" in disp.columns:
        mode_suffix = disp["Mode"].astype(str).apply(lambda m: f" | {m}" if (m and m != "nan") else "")
        disp["_ModelKey"] = disp["Model"].astype(str) + mode_suffix
    else:
        disp["_ModelKey"] = disp["Model"].astype(str)

    disp["_ModelKey"] = disp["_ModelKey"].astype(str).str.slice(0, model_name_maxlen)

    if transpose:
        cols_drop = [c for c in ["Model", "Mode"] if c in disp.columns]
        wide = disp.drop(columns=cols_drop, errors="ignore").set_index("_ModelKey").T
        wide.insert(0, "Metric", wide.index.astype(str))
        wide = wide.reset_index(drop=True)
        _print_df_table(wide, title=title, float_digits=float_digits)
    else:
        disp["Model"] = disp["_ModelKey"]
        disp = disp.drop(columns=["_ModelKey"], errors="ignore")
        _print_df_table(disp, title=title, float_digits=float_digits)

    # Legend / bullets (office-friendly)
    print("\nLEGEND (how to read Level / UniformityFlag / Vertical / EquityVerdict)")
    print(
        "- Level: compares Median Ratio and Weighted Mean Ratio (WMR) to IAAO guidance "
        f"{_fmt_range(IAAO_LEVEL_RANGE,2)}. If either is out-of-range, Level worsens."
    )
    print(
        "- UniformityFlag: compares COD to the IAAO Table-7 guidance for the chosen property class. "
        "Lower COD means tighter appraisal uniformity."
    )
    print("- Vertical: summarizes vertical equity using PRD, PRB, and (if available) VEI.")
    print("  • PRD compares mean ratio vs weighted mean ratio (target ~1).")
    print(
        "  • PRB is regression-based; it estimates the % change in ratios when values double/halve. "
        f"IAAO guidance ~{_fmt_range(IAAO_PRB_RANGE,2)} (|PRB|>0.10 typically unacceptable)."
    )
    print("  • VEI is percentile-group based; within ±10% is acceptable as a point estimate.")
    print(
        "- PASS/WARN/FAIL: PASS = within IAAO guidance; WARN = close-to-boundary (readability heuristic); "
        "FAIL = clearly out-of-range."
    )
    print("- EquityVerdict: conservative roll-up: FAIL if any component FAILs; else WARN if any WARN; else PASS.")

# =============================================================================
# Optional: print baseline + finalists in full detail (for stakeholders)
# =============================================================================

def print_baseline_and_finalists_reports(
    y_true,
    pred_by_model: Dict[str, np.ndarray],
    *,
    baseline_key: str,
    finalist_keys: List[str],
    iaao_property_class: str = "Residential Improved",
    n_quantiles: int = 4,
    show_plots: bool = False,
):
    """Print detailed reports for baseline and selected finalist(s) using the same template."""
    keys = [baseline_key] + [k for k in finalist_keys if k != baseline_key]
    for k in keys:
        if k not in pred_by_model:
            print(f"[WARN] '{k}' not in pred_by_model keys; skipping.")
            continue
        print("" + "=" * 100)
        print(f"DETAILED REPORT: {k}")
        print("=" * 100)
        analyze_financial_performance(
            y_true,
            pred_by_model[k],
            n_quantiles=n_quantiles,
            show_plots=show_plots,
            iaao_property_class=iaao_property_class,
            verbosity=2,
        )
