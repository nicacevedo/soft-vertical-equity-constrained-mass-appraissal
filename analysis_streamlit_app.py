#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd

from utils.motivation_utils import IAAO_PRB_RANGE, IAAO_PRD_RANGE, IAAO_VEI_RANGE


TIDY_REQUIRED_COLUMNS = [
    "data_id",
    "split_id",
    "analysis_run_id",
    "split",
    "grain",
    "artifact",
    "model_name",
    "config_id",
    "rho",
    "cvar_keep",
    "metric",
    "stat",
    "value",
    "fold_id",
    "bootstrap_id",
    "cv_scheme",
    "bootstrap_spec",
    "code_version",
]

POINTS_META_COLUMNS = [
    "data_id",
    "split_id",
    "analysis_run_id",
    "split",
    "grain",
    "artifact",
    "model_name",
    "config_id",
    "rho",
    "cvar_keep",
    "stat",
    "fold_id",
    "bootstrap_id",
    "cv_scheme",
    "bootstrap_spec",
    "code_version",
]


def _debug_log(*, run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
    payload = {
        "sessionId": "c06ffa",
        "runId": str(run_id),
        "hypothesisId": str(hypothesis_id),
        "location": str(location),
        "message": str(message),
        "data": data,
        "timestamp": int(time.time() * 1000),
    }
    try:
        with open(
            "/home/nacevedo/RA/soft-vertical-equity-constrained-mass-appraissal/.cursor/debug-c06ffa.log",
            "a",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(payload, default=str) + "\n")
    except Exception:
        pass


def _compute_2d_pareto_mask(
    x: np.ndarray,
    y: np.ndarray,
    *,
    minimize_x: bool = True,
    minimize_y: bool = False,
) -> np.ndarray:
    x = np.asarray(x, dtype=float).reshape(-1)
    y = np.asarray(y, dtype=float).reshape(-1)
    n = x.shape[0]
    if n == 0:
        return np.zeros(0, dtype=bool)
    mask = np.ones(n, dtype=bool)
    for i in range(n):
        if not mask[i]:
            continue
        xi, yi = x[i], y[i]
        no_worse_x = (x <= xi) if minimize_x else (x >= xi)
        strictly_better_x = (x < xi) if minimize_x else (x > xi)
        no_worse_y = (y <= yi) if minimize_y else (y >= yi)
        strictly_better_y = (y < yi) if minimize_y else (y > yi)
        dominates_i = no_worse_x & no_worse_y & (strictly_better_x | strictly_better_y)
        dominates_i[i] = False
        if bool(np.any(dominates_i)):
            mask[i] = False
    return mask


def _parse_csv_list(value: str) -> List[str]:
    return [str(x).strip() for x in str(value).split(",") if str(x).strip()]


def _candidate_metrics(df: pd.DataFrame) -> List[str]:
    out: List[str] = []
    reserved = set(POINTS_META_COLUMNS)
    for c in df.columns:
        if c in reserved:
            continue
        if str(c).startswith("pareto_2d_"):
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            out.append(str(c))
    return out


def _validate_schema(tidy: pd.DataFrame, points: pd.DataFrame) -> None:
    missing_tidy = [c for c in TIDY_REQUIRED_COLUMNS if c not in tidy.columns]
    if missing_tidy:
        raise ValueError(f"analysis_tidy.parquet is missing required columns: {missing_tidy}")
    missing_points_meta = [c for c in POINTS_META_COLUMNS if c not in points.columns]
    if missing_points_meta:
        raise ValueError(f"analysis_points.parquet is missing required metadata columns: {missing_points_meta}")


def _coerce_dtypes(tidy: pd.DataFrame, points: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    tidy_out = tidy.copy()
    points_out = points.copy()

    for c in [
        "data_id",
        "split_id",
        "analysis_run_id",
        "split",
        "grain",
        "artifact",
        "model_name",
        "config_id",
        "metric",
        "stat",
        "cv_scheme",
        "bootstrap_spec",
        "code_version",
    ]:
        if c in tidy_out.columns:
            tidy_out[c] = tidy_out[c].astype("string")
    for c in ["rho", "cvar_keep", "value"]:
        if c in tidy_out.columns:
            tidy_out[c] = pd.to_numeric(tidy_out[c], errors="coerce")
    for c in ["fold_id", "bootstrap_id"]:
        if c in tidy_out.columns:
            tidy_out[c] = pd.to_numeric(tidy_out[c], errors="coerce").astype("Int64")

    for c in [
        "data_id",
        "split_id",
        "analysis_run_id",
        "split",
        "grain",
        "artifact",
        "model_name",
        "config_id",
        "stat",
        "cv_scheme",
        "bootstrap_spec",
        "code_version",
    ]:
        if c in points_out.columns:
            points_out[c] = points_out[c].astype("string")
    for c in ["rho", "cvar_keep"]:
        if c in points_out.columns:
            points_out[c] = pd.to_numeric(points_out[c], errors="coerce")
    for c in ["fold_id", "bootstrap_id"]:
        if c in points_out.columns:
            points_out[c] = pd.to_numeric(points_out[c], errors="coerce").astype("Int64")

    return tidy_out, points_out


def _default_pairs(points: pd.DataFrame) -> List[Tuple[str, str]]:
    preferred = [
        ("abs_PRD_dev", "R2"),
        ("abs_PRB_dev", "R2"),
        ("abs_VEI_dev", "R2"),
        ("abs_Corr_r_price", "R2"),
        ("abs_PRD_dev", "OOS R2"),
        ("abs_PRB_dev", "OOS R2"),
        ("abs_VEI_dev", "OOS R2"),
        ("abs_Corr_r_price", "OOS R2"),
    ]
    cols = set(points.columns)
    return [(x, y) for (x, y) in preferred if x in cols and y in cols]


def _apply_filters(
    df: pd.DataFrame,
    *,
    split_values: Sequence[str],
    grain_values: Sequence[str],
    stat_values: Sequence[str],
    include_models: Sequence[str],
    exclude_models: Sequence[str],
) -> pd.DataFrame:
    out = df.copy()
    if split_values:
        out = out[out["split"].astype(str).isin([str(x) for x in split_values])]
    if grain_values:
        out = out[out["grain"].astype(str).isin([str(x) for x in grain_values])]
    if stat_values:
        out = out[out["stat"].astype(str).isin([str(x) for x in stat_values])]
    if include_models:
        inc = {str(x) for x in include_models}
        out = out[out["model_name"].astype(str).isin(inc)]
    if exclude_models:
        exc = {str(x) for x in exclude_models}
        out = out[~out["model_name"].astype(str).isin(exc)]
    return out


def _add_filtered_pareto_columns(df: pd.DataFrame, pairs: Sequence[Tuple[str, str]]) -> pd.DataFrame:
    out = df.copy()
    gcols = [c for c in ["split", "grain", "artifact", "stat", "fold_id"] if c in out.columns]
    for xcol, ycol in pairs:
        col = f"pareto_2d_filtered__{xcol}__{ycol}"
        out[col] = False
        if xcol not in out.columns or ycol not in out.columns:
            continue
        for _, g in out.groupby(gcols, dropna=False):
            xv = pd.to_numeric(g[xcol], errors="coerce").to_numpy(dtype=float)
            yv = pd.to_numeric(g[ycol], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(xv) & np.isfinite(yv)
            if int(np.sum(ok)) < 2:
                continue
            m = _compute_2d_pareto_mask(xv[ok], yv[ok], minimize_x=True, minimize_y=False)
            idx = g.index.to_numpy()[ok]
            out.loc[idx, col] = m
    return out


def _sample_rows(df: pd.DataFrame, *, max_rows: int, seed: int) -> pd.DataFrame:
    if int(max_rows) <= 0 or df.shape[0] <= int(max_rows):
        return df
    return df.sample(n=int(max_rows), random_state=int(seed), replace=False)


def _sanitize_for_wandb_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert pandas nullable dtypes (e.g., Int64 with pd.NA) into
    W&B Table compatible Python/NumPy scalars.
    """
    out = df.copy()

    # Nullable integer ids are common in tidy/points and trigger NAType errors in W&B.
    for c in ["fold_id", "bootstrap_id"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce").astype(float)

    # Replace pandas NA/NaT with plain None for W&B serialization.
    out = out.astype(object).where(pd.notna(out), None)
    return out


def _metric_objective_band(metric_col: str) -> Tuple[Optional[Tuple[float, float]], Optional[float]]:
    m = str(metric_col).replace("_mean", "")
    if m == "PRD":
        lo, hi = float(IAAO_PRD_RANGE[0]), float(IAAO_PRD_RANGE[1])
        return (min(lo, hi), max(lo, hi)), 1.0
    if m == "PRB":
        lo, hi = float(IAAO_PRB_RANGE[0]), float(IAAO_PRB_RANGE[1])
        return (min(lo, hi), max(lo, hi)), 0.0
    if m == "VEI":
        lo, hi = float(IAAO_VEI_RANGE[0]), float(IAAO_VEI_RANGE[1])
        return (min(lo, hi), max(lo, hi)), 0.0
    if m == "abs_PRD_dev":
        lo = abs(float(IAAO_PRD_RANGE[0]) - 1.0)
        hi = abs(float(IAAO_PRD_RANGE[1]) - 1.0)
        return (0.0, max(lo, hi)), 0.0
    if m == "abs_PRB_dev":
        hi = max(abs(float(IAAO_PRB_RANGE[0])), abs(float(IAAO_PRB_RANGE[1])))
        return (0.0, hi), 0.0
    if m == "abs_VEI_dev":
        hi = max(abs(float(IAAO_VEI_RANGE[0])), abs(float(IAAO_VEI_RANGE[1])))
        return (0.0, hi), 0.0
    return None, None


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = str(hex_color).strip()
    if h.startswith("#"):
        h = h[1:]
    if len(h) != 6:
        return f"rgba(31,119,180,{float(np.clip(alpha, 0.05, 1.0)):.4f})"
    r = int(h[0:2], 16)
    g = int(h[2:4], 16)
    b = int(h[4:6], 16)
    return f"rgba({r},{g},{b},{float(np.clip(alpha, 0.05, 1.0)):.4f})"


def _rho_alpha_series(df: pd.DataFrame, *, baseline_mask: np.ndarray) -> np.ndarray:
    n = int(df.shape[0])
    alpha = np.full(n, 0.55, dtype=float)
    alpha[baseline_mask] = 0.95
    if "rho" not in df.columns:
        return alpha
    rho_vals = pd.to_numeric(df["rho"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(rho_vals) & (~baseline_mask)
    if np.any(finite):
        lrho = np.log10(np.maximum(rho_vals[finite], 1e-12))
        lo, hi = float(np.nanmin(lrho)), float(np.nanmax(lrho))
        if hi > lo:
            norm = (lrho - lo) / (hi - lo)
        else:
            norm = np.ones_like(lrho)
        alpha[finite] = 0.25 + 0.70 * norm
    return alpha


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Upload analysis parquet outputs to Weights & Biases.")
    p.add_argument("--result-root", type=str, default="./output/robust_rolling_origin_cv/analysis")
    p.add_argument("--data-id", type=str, required=True)
    p.add_argument("--split-id", type=str, required=True)
    p.add_argument("--wandb-project", type=str, default="vertical-equity-analysis")
    p.add_argument("--wandb-entity", type=str, default=None)
    p.add_argument("--wandb-run-name", type=str, default=None)
    p.add_argument("--wandb-job-type", type=str, default="analysis_dashboard")
    p.add_argument("--wandb-group", type=str, default=None)
    p.add_argument("--wandb-tags", type=str, default="analysis,tradeoff")
    p.add_argument("--max-table-rows", type=int, default=60000)
    p.add_argument("--sample-seed", type=int, default=2025)
    p.add_argument("--filter-split", type=str, default="validation,test")
    p.add_argument("--filter-grain", type=str, default="config_agg,fold")
    p.add_argument("--filter-stat", type=str, default="mean,raw")
    p.add_argument("--include-model-families", type=str, default="")
    p.add_argument("--exclude-model-families", type=str, default="")
    p.add_argument("--compute-pareto-filtered", action="store_true", default=True)
    p.add_argument("--no-compute-pareto-filtered", dest="compute_pareto_filtered", action="store_false")
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    debug_run_id = f"{args.data_id}:{args.split_id}"
    # #region agent log
    _debug_log(
        run_id=debug_run_id,
        hypothesis_id="H0",
        location="analysis_streamlit_app.py:main:start",
        message="analysis_streamlit_app start",
        data={"result_root": str(args.result_root), "data_id": str(args.data_id), "split_id": str(args.split_id)},
    )
    # #endregion
    try:
        import wandb
    except Exception as exc:
        raise RuntimeError("wandb is not installed. Install it with `pip install wandb`.") from exc

    analysis_dir = Path(args.result_root) / f"data_id={args.data_id}" / f"split_id={args.split_id}"
    tidy_path = analysis_dir / "analysis_tidy.parquet"
    points_path = analysis_dir / "analysis_points.parquet"
    if not tidy_path.exists() or not points_path.exists():
        raise FileNotFoundError(
            f"Missing parquet files in {analysis_dir}. Run analyze_results.py first "
            f"(expected {tidy_path.name} and {points_path.name})."
        )

    # #region agent log
    _debug_log(
        run_id=debug_run_id,
        hypothesis_id="H5",
        location="analysis_streamlit_app.py:main:parquet_paths",
        message="parquet file existence and size",
        data={
            "tidy_exists": bool(tidy_path.exists()),
            "points_exists": bool(points_path.exists()),
            "tidy_size": int(os.path.getsize(tidy_path)) if tidy_path.exists() else -1,
            "points_size": int(os.path.getsize(points_path)) if points_path.exists() else -1,
        },
    )
    # #endregion
    try:
        tidy = pd.read_parquet(tidy_path)
        # #region agent log
        _debug_log(
            run_id=debug_run_id,
            hypothesis_id="H5",
            location="analysis_streamlit_app.py:main:read_tidy_default_ok",
            message="read tidy parquet with default engine",
            data={"rows": int(tidy.shape[0]), "cols": int(tidy.shape[1])},
        )
        # #endregion
    except Exception:
        # #region agent log
        _debug_log(
            run_id=debug_run_id,
            hypothesis_id="H5",
            location="analysis_streamlit_app.py:main:read_tidy_default_error",
            message="failed reading tidy parquet with default engine",
            data={"error": traceback.format_exc()},
        )
        # #endregion
        raise

    try:
        points = pd.read_parquet(points_path)
        # #region agent log
        _debug_log(
            run_id=debug_run_id,
            hypothesis_id="H5",
            location="analysis_streamlit_app.py:main:read_points_default_ok",
            message="read points parquet with default engine",
            data={"rows": int(points.shape[0]), "cols": int(points.shape[1])},
        )
        # #endregion
    except Exception:
        # #region agent log
        _debug_log(
            run_id=debug_run_id,
            hypothesis_id="H5",
            location="analysis_streamlit_app.py:main:read_points_default_error",
            message="failed reading points parquet with default engine",
            data={"error": traceback.format_exc()},
        )
        # #endregion
        try:
            p_fast = pd.read_parquet(points_path, engine="fastparquet")
            # #region agent log
            _debug_log(
                run_id=debug_run_id,
                hypothesis_id="H5",
                location="analysis_streamlit_app.py:main:read_points_fastparquet_ok",
                message="read points parquet with fastparquet engine",
                data={"rows": int(p_fast.shape[0]), "cols": int(p_fast.shape[1])},
            )
            # #endregion
        except Exception:
            # #region agent log
            _debug_log(
                run_id=debug_run_id,
                hypothesis_id="H5",
                location="analysis_streamlit_app.py:main:read_points_fastparquet_error",
                message="failed reading points parquet with fastparquet engine",
                data={"error": traceback.format_exc()},
            )
            # #endregion
        raise
    _validate_schema(tidy, points)
    tidy, points = _coerce_dtypes(tidy, points)

    split_values = _parse_csv_list(args.filter_split)
    grain_values = _parse_csv_list(args.filter_grain)
    stat_values = _parse_csv_list(args.filter_stat)
    include_models = _parse_csv_list(args.include_model_families)
    exclude_models = _parse_csv_list(args.exclude_model_families)

    points_filtered = _apply_filters(
        points,
        split_values=split_values,
        grain_values=grain_values,
        stat_values=stat_values,
        include_models=include_models,
        exclude_models=exclude_models,
    )
    pairs = _default_pairs(points_filtered)
    # #region agent log
    _debug_log(
        run_id=debug_run_id,
        hypothesis_id="H2",
        location="analysis_streamlit_app.py:main:after_filters",
        message="filter and pair summary",
        data={
            "rows_points": int(points.shape[0]),
            "rows_points_filtered": int(points_filtered.shape[0]),
            "pairs_count": int(len(pairs)),
            "pairs": [f"{x}__{y}" for (x, y) in pairs],
            "splits_filtered": sorted(points_filtered["split"].dropna().astype(str).unique().tolist()) if "split" in points_filtered.columns else [],
        },
    )
    # #endregion
    if bool(args.compute_pareto_filtered) and pairs:
        points_filtered = _add_filtered_pareto_columns(points_filtered, pairs=pairs)

    run_name = args.wandb_run_name or f"analysis-{args.data_id}-{args.split_id}"
    tags = _parse_csv_list(args.wandb_tags)
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        job_type=args.wandb_job_type,
        group=args.wandb_group or f"data_id={args.data_id}",
        tags=tags,
        config={
            "data_id": str(args.data_id),
            "split_id": str(args.split_id),
            "analysis_dir": str(analysis_dir),
            "filter_split": split_values,
            "filter_grain": grain_values,
            "filter_stat": stat_values,
            "include_model_families": include_models,
            "exclude_model_families": exclude_models,
            "compute_pareto_filtered": bool(args.compute_pareto_filtered),
            "max_table_rows": int(args.max_table_rows),
            "sample_seed": int(args.sample_seed),
            "schema_version": "analysis-v1",
        },
    )
    try:
        artifact = wandb.Artifact(
            name=f"analysis-{args.data_id}-{args.split_id}",
            type="analysis-dataset",
            metadata={
                "data_id": str(args.data_id),
                "split_id": str(args.split_id),
                "rows_tidy": int(tidy.shape[0]),
                "rows_points": int(points.shape[0]),
                "rows_points_filtered": int(points_filtered.shape[0]),
            },
        )
        artifact.add_file(str(tidy_path))
        artifact.add_file(str(points_path))
        run.log_artifact(artifact)

        tidy_upload = _sanitize_for_wandb_table(
            _sample_rows(tidy, max_rows=int(args.max_table_rows), seed=int(args.sample_seed))
        )
        points_upload = _sanitize_for_wandb_table(
            _sample_rows(points_filtered, max_rows=int(args.max_table_rows), seed=int(args.sample_seed))
        )
        run.log(
            {
                "tables/analysis_tidy": wandb.Table(dataframe=tidy_upload),
                "tables/analysis_points_filtered": wandb.Table(dataframe=points_upload),
            }
        )

        run.summary["rows_tidy"] = int(tidy.shape[0])
        run.summary["rows_points"] = int(points.shape[0])
        run.summary["rows_points_filtered"] = int(points_filtered.shape[0])
        run.summary["model_families_filtered"] = int(points_filtered["model_name"].astype(str).nunique()) if "model_name" in points_filtered.columns else 0

        try:
            # #region agent log
            _debug_log(
                run_id=debug_run_id,
                hypothesis_id="H1",
                location="analysis_streamlit_app.py:main:plotly_try_entry",
                message="entering plotly logging block",
                data={"pairs_count": int(len(pairs)), "rows_points_filtered": int(points_filtered.shape[0])},
            )
            # #endregion
            import plotly.graph_objects as go  # type: ignore
            from plotly.colors import qualitative as qcolors  # type: ignore

            for split_name in sorted(points_filtered["split"].dropna().astype(str).unique().tolist()):
                dsplit = points_filtered[points_filtered["split"].astype(str) == split_name].copy()
                if dsplit.empty:
                    continue
                # #region agent log
                _debug_log(
                    run_id=debug_run_id,
                    hypothesis_id="H2",
                    location="analysis_streamlit_app.py:main:split_loop",
                    message="processing split for plotly",
                    data={"split": str(split_name), "rows_split": int(dsplit.shape[0]), "pairs_count": int(len(pairs))},
                )
                # #endregion
                for xcol, ycol in pairs:
                    if xcol not in dsplit.columns or ycol not in dsplit.columns:
                        continue
                    dd = dsplit[
                        np.isfinite(pd.to_numeric(dsplit[xcol], errors="coerce"))
                        & np.isfinite(pd.to_numeric(dsplit[ycol], errors="coerce"))
                    ].copy()
                    if dd.empty:
                        continue
                    # #region agent log
                    _debug_log(
                        run_id=debug_run_id,
                        hypothesis_id="H4",
                        location="analysis_streamlit_app.py:main:before_plotly_scatter",
                        message="plot candidate dataframe diagnostics",
                        data={
                            "split": str(split_name),
                            "xcol": str(xcol),
                            "ycol": str(ycol),
                            "rows_dd": int(dd.shape[0]),
                            "dd_dtypes": {str(k): str(v) for k, v in dd.dtypes.items() if str(k) in {"fold_id", "bootstrap_id", "rho", "cvar_keep", "model_name", str(xcol), str(ycol)}},
                            "na_counts": {str(k): int(dd[str(k)].isna().sum()) for k in [xcol, ycol, "fold_id", "bootstrap_id"] if str(k) in dd.columns},
                        },
                    )
                    # #endregion
                    sym_col = f"pareto_2d_filtered__{xcol}__{ycol}"
                    xarr = pd.to_numeric(dd[xcol], errors="coerce").to_numpy(dtype=float)
                    yarr = pd.to_numeric(dd[ycol], errors="coerce").to_numpy(dtype=float)
                    pareto = _compute_2d_pareto_mask(xarr, yarr, minimize_x=True, minimize_y=False)
                    mnames = dd["model_name"].astype(str).to_numpy()
                    baseline_models = {"LinearRegression", "LGBMRegressor", "LR", "LGBM"}
                    is_baseline = np.array([m in baseline_models for m in mnames], dtype=bool)
                    alpha = _rho_alpha_series(dd, baseline_mask=is_baseline)

                    uniq_models = sorted(set(mnames.tolist()))
                    palette = qcolors.Plotly
                    cmap = {m: palette[i % len(palette)] for i, m in enumerate(uniq_models)}
                    fig = go.Figure()
                    hover_tmpl = (
                        "model=%{customdata[0]}<br>"
                        "config_id=%{customdata[1]}<br>"
                        "rho=%{customdata[2]}<br>"
                        "cvar_keep=%{customdata[3]}<br>"
                        "split=%{customdata[4]}<br>"
                        "grain=%{customdata[5]}<br>"
                        "artifact=%{customdata[6]}<br>"
                        "stat=%{customdata[7]}<br>"
                        "fold_id=%{customdata[8]}<br>"
                        "bootstrap_id=%{customdata[9]}<br>"
                        f"{xcol}=%{{x}}<br>{ycol}=%{{y}}<extra></extra>"
                    )
                    for model in uniq_models:
                        mm = (mnames == model)
                        if int(np.sum(mm)) == 0:
                            continue
                        x_m = xarr[mm]
                        y_m = yarr[mm]
                        a_m = alpha[mm]
                        symbol = "star" if model in baseline_models else "circle"
                        size = 13 if model in baseline_models else 8
                        color_hex = cmap.get(model, "#1f77b4")
                        colors = [_hex_to_rgba(color_hex, float(a)) for a in a_m.tolist()]
                        sub = dd.loc[mm, :]
                        custom = np.column_stack(
                            [
                                sub["model_name"].astype(str).to_numpy(),
                                sub["config_id"].astype(str).to_numpy() if "config_id" in sub.columns else np.array([""] * sub.shape[0], dtype=object),
                                pd.to_numeric(sub["rho"], errors="coerce").to_numpy(dtype=float) if "rho" in sub.columns else np.full(sub.shape[0], np.nan),
                                pd.to_numeric(sub["cvar_keep"], errors="coerce").to_numpy(dtype=float) if "cvar_keep" in sub.columns else np.full(sub.shape[0], np.nan),
                                sub["split"].astype(str).to_numpy() if "split" in sub.columns else np.array([""] * sub.shape[0], dtype=object),
                                sub["grain"].astype(str).to_numpy() if "grain" in sub.columns else np.array([""] * sub.shape[0], dtype=object),
                                sub["artifact"].astype(str).to_numpy() if "artifact" in sub.columns else np.array([""] * sub.shape[0], dtype=object),
                                sub["stat"].astype(str).to_numpy() if "stat" in sub.columns else np.array([""] * sub.shape[0], dtype=object),
                                pd.to_numeric(sub["fold_id"], errors="coerce").to_numpy(dtype=float) if "fold_id" in sub.columns else np.full(sub.shape[0], np.nan),
                                pd.to_numeric(sub["bootstrap_id"], errors="coerce").to_numpy(dtype=float) if "bootstrap_id" in sub.columns else np.full(sub.shape[0], np.nan),
                            ]
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=x_m,
                                y=y_m,
                                mode="markers",
                                name=str(model),
                                customdata=custom,
                                hovertemplate=hover_tmpl,
                                marker={
                                    "symbol": symbol,
                                    "size": size,
                                    "color": colors,
                                    "line": {"width": 0.0},
                                },
                            )
                        )
                    # Pareto emphasis ring (same core logic as local plots).
                    if int(np.sum(pareto)) > 0:
                        fig.add_trace(
                            go.Scatter(
                                x=xarr[pareto],
                                y=yarr[pareto],
                                mode="markers",
                                name="pareto (2D)",
                                marker={
                                    "symbol": "circle",
                                    "size": 16,
                                    "line": {"color": "black", "width": 2},
                                    "color": "rgba(0,0,0,0)",
                                },
                                hovertemplate=f"pareto point<br>{xcol}=%{{x}}<br>{ycol}=%{{y}}<extra></extra>",
                            )
                        )
                    # Objective line and transparent band on fairness axis.
                    band, target = _metric_objective_band(xcol)
                    if band is not None:
                        fig.add_vrect(
                            x0=float(band[0]),
                            x1=float(band[1]),
                            fillcolor="limegreen",
                            opacity=0.18,
                            line_width=0,
                            layer="below",
                            annotation_text="IAAO band",
                            annotation_position="top left",
                        )
                    if target is not None:
                        fig.add_vline(
                            x=float(target),
                            line_color="forestgreen",
                            line_width=1.5,
                            line_dash="dash",
                        )
                    fig.update_layout(
                        title=f"{split_name}: {xcol} vs {ycol}",
                        xaxis_title=xcol,
                        yaxis_title=ycol,
                        template="plotly_white",
                    )
                    run.log({f"plotly/{split_name}/{xcol}__{ycol}": fig})
                    # #region agent log
                    _debug_log(
                        run_id=debug_run_id,
                        hypothesis_id="H1",
                        location="analysis_streamlit_app.py:main:plot_logged",
                        message="logged plotly figure",
                        data={"key": f"plotly/{split_name}/{xcol}__{ycol}", "rows_dd": int(dd.shape[0])},
                    )
                    # #endregion
        except Exception:
            # #region agent log
            _debug_log(
                run_id=debug_run_id,
                hypothesis_id="H1",
                location="analysis_streamlit_app.py:main:plotly_try_exception",
                message="plotly logging block exception",
                data={"error": traceback.format_exc()},
            )
            # #endregion
    finally:
        run.finish()


if __name__ == "__main__":
    main()
