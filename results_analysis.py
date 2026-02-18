from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D

from utils.motivation_utils import IAAO_PRB_RANGE, IAAO_PRD_RANGE, IAAO_VEI_RANGE, _compute_extended_metrics


def _safe_read_json(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _latest_protocol(result_root: Path) -> Tuple[str, str, Path]:
    protocol_root = result_root / "protocol"
    candidates = list(protocol_root.glob("data_id=*/split_id=*/folds.json"))
    if not candidates:
        raise FileNotFoundError(f"No protocol files found under: {protocol_root}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    data_id = latest.parent.parent.name.split("=", 1)[1]
    split_id = latest.parent.name.split("=", 1)[1]
    return data_id, split_id, latest


def _normalize_series(s: pd.Series) -> pd.Series:
    x = pd.to_numeric(s, errors="coerce")
    mn, mx = np.nanmin(x.values), np.nanmax(x.values)
    if not np.isfinite(mn) or not np.isfinite(mx) or abs(mx - mn) < 1e-15:
        return pd.Series(np.full(x.shape, 0.5), index=x.index)
    return (x - mn) / (mx - mn)


def _as_better_high(x_norm: pd.Series, direction: str) -> pd.Series:
    if direction == "max":
        return x_norm
    if direction == "min":
        return 1.0 - x_norm
    raise ValueError("direction must be 'max' or 'min'")


def _is_pareto_efficient(values: np.ndarray, maximize_mask: np.ndarray) -> np.ndarray:
    """
    Pareto efficient mask for mixed max/min objectives.
    Internally converts to minimization by negating maximize dimensions.
    """
    if values.size == 0:
        return np.array([], dtype=bool)
    v = values.copy().astype(float)
    v[:, maximize_mask] *= -1.0
    n = v.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        # In minimization space, i dominates j when v[i] <= v[j] on all objectives
        # and strictly better on at least one.
        dominated_by_i = np.all(v[i] <= v, axis=1) & np.any(v[i] < v, axis=1)
        is_efficient[dominated_by_i] = False
        is_efficient[i] = True
    return is_efficient


def load_cv_artifacts(
    result_root: str = "./output/robust_rolling_origin_cv",
    data_id: Optional[str] = None,
    split_id: Optional[str] = None,
) -> Dict[str, Any]:
    root = Path(result_root)
    if data_id is None or split_id is None:
        d_id, s_id, protocol_file = _latest_protocol(root)
        data_id = data_id or d_id
        split_id = split_id or s_id
    else:
        protocol_file = root / "protocol" / f"data_id={data_id}" / f"split_id={split_id}" / "folds.json"

    if not protocol_file.exists():
        raise FileNotFoundError(f"Protocol file not found: {protocol_file}")

    protocol = _safe_read_json(protocol_file)
    status_files = sorted(
        (root / "run_status" / f"data_id={data_id}" / f"split_id={split_id}").glob("fold_id=*/*.json")
    )
    status_rows = []
    run_files = []
    bootstrap_files = []

    for sf in status_files:
        payload = _safe_read_json(sf)
        if not payload:
            continue
        payload["_status_file"] = str(sf)
        status_rows.append(payload)
        if payload.get("status") == "completed":
            arts = payload.get("artifacts", {})
            run_file = arts.get("run_file")
            bs_file = arts.get("bootstrap_file")
            if run_file:
                run_path = Path(run_file)
                if not run_path.is_absolute():
                    run_path = Path(".") / run_file
                if run_path.exists():
                    run_files.append(run_path)
            if bs_file:
                bs_path = Path(bs_file)
                if not bs_path.is_absolute():
                    bs_path = Path(".") / bs_file
                if bs_path.exists():
                    bootstrap_files.append(bs_path)

    status_df = pd.DataFrame(status_rows)

    run_frames = []
    for p in sorted(set(run_files)):
        try:
            run_frames.append(pd.read_parquet(p))
        except Exception:
            continue
    runs_df = pd.concat(run_frames, ignore_index=True) if run_frames else pd.DataFrame()
    if not runs_df.empty and "run_id" in runs_df.columns:
        runs_df = runs_df.drop_duplicates(subset=["run_id"], keep="last")

    bs_frames = []
    for p in sorted(set(bootstrap_files)):
        try:
            bs_frames.append(pd.read_parquet(p))
        except Exception:
            continue
    bootstrap_df = pd.concat(bs_frames, ignore_index=True) if bs_frames else pd.DataFrame()
    if not bootstrap_df.empty and {"run_id", "bootstrap_id"}.issubset(set(bootstrap_df.columns)):
        bootstrap_df = bootstrap_df.drop_duplicates(subset=["run_id", "bootstrap_id"], keep="last")

    return {
        "data_id": data_id,
        "split_id": split_id,
        "protocol_file": str(protocol_file),
        "protocol": protocol,
        "status_df": status_df,
        "runs_df": runs_df,
        "bootstrap_df": bootstrap_df,
    }


def summarize_by_config(runs_df: pd.DataFrame) -> pd.DataFrame:
    if runs_df.empty:
        return pd.DataFrame()

    id_cols = [c for c in ["data_id", "split_id", "config_id", "model_name", "model_config_json"] if c in runs_df.columns]
    num_cols = []
    for c in runs_df.columns:
        if c in id_cols:
            continue
        if pd.api.types.is_numeric_dtype(runs_df[c]):
            num_cols.append(c)

    agg = {c: ["mean", "std", "median"] for c in num_cols}
    summary = runs_df.groupby(id_cols, dropna=False).agg(agg)
    summary.columns = [f"{c}_{s}" for c, s in summary.columns]
    summary = summary.reset_index()
    summary["fold_count"] = runs_df.groupby(id_cols, dropna=False).size().values

    # Derived vertical-equity stats in "distance to ideal" form (lower is better)
    if "PRB_mean" in summary.columns:
        summary["abs_PRB_mean"] = np.abs(summary["PRB_mean"])
    if "PRD_mean" in summary.columns:
        summary["abs_PRD_dev_mean"] = np.abs(summary["PRD_mean"] - 1.0)
    if "VEI_mean" in summary.columns:
        summary["abs_VEI_mean"] = np.abs(summary["VEI_mean"])
    if "Corr(r,y)_price_mean" in summary.columns:
        summary["abs_Corr_r_y_price_mean"] = np.abs(summary["Corr(r,y)_price_mean"])
    return summary


def compute_pareto_front(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return summary_df.copy()

    acc_col = "OOS R2_mean" if "OOS R2_mean" in summary_df.columns else "R2_mean"
    objectives = [(acc_col, "max")]
    for c in ["abs_PRB_mean", "abs_PRD_dev_mean", "abs_VEI_mean", "abs_Corr_r_y_price_mean"]:
        if c in summary_df.columns:
            objectives.append((c, "min"))

    obj_cols = [c for c, _ in objectives]
    maximize_mask = np.array([d == "max" for _, d in objectives], dtype=bool)
    values = summary_df[obj_cols].to_numpy(dtype=float)
    ok = np.all(np.isfinite(values), axis=1)
    mask = np.zeros(summary_df.shape[0], dtype=bool)
    if np.any(ok):
        mask_ok = _is_pareto_efficient(values[ok], maximize_mask=maximize_mask)
        mask[np.where(ok)[0]] = mask_ok

    out = summary_df.copy()
    out["is_pareto"] = mask
    return out


def build_shortlist(
    summary_with_pareto: pd.DataFrame,
    top_k: int = 10,
    fairness_norm_threshold: float = 0.35,
) -> Dict[str, Any]:
    if summary_with_pareto.empty:
        return {"shortlists": {}}

    df = summary_with_pareto.copy()
    acc_col = "OOS R2_mean" if "OOS R2_mean" in df.columns else "R2_mean"
    fairness_cols = [c for c in ["abs_PRB_mean", "abs_PRD_dev_mean", "abs_VEI_mean", "abs_Corr_r_y_price_mean"] if c in df.columns]

    df["_acc_norm"] = _normalize_series(df[acc_col])
    for c in fairness_cols:
        df[f"_{c}_norm"] = _normalize_series(df[c])
        df[f"_{c}_good"] = _as_better_high(df[f"_{c}_norm"], "min")

    # 1) Utopian distance on normalized objective space.
    dist_terms = [(1.0 - df["_acc_norm"]) ** 2]
    for c in fairness_cols:
        dist_terms.append(df[f"_{c}_norm"] ** 2)
    df["_utopian_dist"] = np.sqrt(np.sum(dist_terms, axis=0))

    # 2) Max accuracy under normalized fairness constraints.
    if fairness_cols:
        fairness_ok = np.ones(df.shape[0], dtype=bool)
        for c in fairness_cols:
            fairness_ok &= (df[f"_{c}_norm"] <= float(fairness_norm_threshold))
        df["_fairness_ok"] = fairness_ok
    else:
        df["_fairness_ok"] = True

    # 3) Geometric utility (sum of logs proxy).
    eps = 1e-8
    utility_terms = [np.log(eps + df["_acc_norm"])]
    for c in fairness_cols:
        utility_terms.append(np.log(eps + df[f"_{c}_good"]))
    df["_log_utility"] = np.sum(utility_terms, axis=0)

    keep_cols = [c for c in ["config_id", "model_name", acc_col, "is_pareto"] if c in df.columns]
    keep_cols += fairness_cols

    shortlist_utopian = df.sort_values("_utopian_dist", ascending=True).head(top_k)
    shortlist_constrained = df[df["_fairness_ok"]].sort_values("_acc_norm", ascending=False).head(top_k)
    shortlist_utility = df.sort_values("_log_utility", ascending=False).head(top_k)

    return {
        "meta": {
            "top_k": int(top_k),
            "accuracy_col": acc_col,
            "fairness_cols": fairness_cols,
            "fairness_norm_threshold": float(fairness_norm_threshold),
        },
        "shortlists": {
            "utopian_distance": shortlist_utopian[keep_cols + ["_utopian_dist"]].to_dict(orient="records"),
            "max_accuracy_subject_to_fairness": shortlist_constrained[keep_cols + ["_acc_norm"]].to_dict(orient="records"),
            "max_log_utility": shortlist_utility[keep_cols + ["_log_utility"]].to_dict(orient="records"),
        },
    }


def _extract_rho(model_name: str) -> Optional[float]:
    # examples: LGBSmoothPenalty_rho_0.1  or ...rho_2154.434...
    m = re.search(r"_rho_([\-+0-9.eE]+)$", str(model_name))
    if m is None:
        m = re.search(r"_rho_([\-+0-9.eE]+)", str(model_name))
    if m is None:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _family_name(model_name: str) -> str:
    s = str(model_name)
    return re.sub(r"_rho_[\-+0-9.eE]+", "", s)


def _safe_to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return np.nan


def _base_family_name(model_name: Any) -> str:
    s = str(model_name)
    if "_rho_" in s:
        return _family_name(s)
    return s


def _build_family_color_map(families: List[str]) -> Dict[str, Tuple[float, float, float]]:
    fams = sorted({str(f) for f in families if str(f)})
    if not fams:
        return {}
    # Curated 20-color palette with high visual separation.
    palette_hex = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173",
        "#3182bd", "#31a354", "#756bb1", "#636363", "#e6550d",
    ]
    palette = [mcolors.to_rgb(h) for h in palette_hex]
    return {fam: palette[i % len(palette)] for i, fam in enumerate(fams)}


def _build_family_rho_intensity(df: pd.DataFrame) -> Dict[Tuple[str, float], float]:
    """
    Map (family, rho) -> intensity in [0.35, 1.0].
    Baselines (rho missing) are intentionally excluded.
    """
    mapping: Dict[Tuple[str, float], float] = {}
    if df.empty or "model_family" not in df.columns or "rho" not in df.columns:
        return mapping

    for fam, sub in df.groupby("model_family", dropna=False):
        rho_vals = sorted({float(x) for x in pd.to_numeric(sub["rho"], errors="coerce").dropna().tolist()})
        if not rho_vals:
            continue
        if len(rho_vals) == 1:
            mapping[(str(fam), rho_vals[0])] = 1.0
            continue
        ints = np.linspace(0.35, 1.0, num=len(rho_vals))
        for r, intensity in zip(rho_vals, ints):
            mapping[(str(fam), float(r))] = float(intensity)
    return mapping


def _shade_color(base_rgb: Tuple[float, float, float], intensity: float) -> Tuple[float, float, float]:
    """
    Blend base color with white. intensity=1 keeps base color,
    lower values become lighter for visual rho progression.
    """
    i = float(np.clip(intensity, 0.0, 1.0))
    b = np.array(base_rgb, dtype=float)
    white = np.array([1.0, 1.0, 1.0], dtype=float)
    out = white * (1.0 - i) + b * i
    return tuple(np.clip(out, 0.0, 1.0))


def _reference_for_metric(metric_name: str) -> Optional[Tuple[float, float, float]]:
    """
    Returns (ideal, lo, hi) reference for fairness metrics.
    """
    if metric_name in ("PRD_mean", "PRD"):
        lo, hi = IAAO_PRD_RANGE
        return 1.0, float(lo), float(hi)
    if metric_name == "abs_PRD_dev_mean":
        max_dev = max(abs(IAAO_PRD_RANGE[0] - 1.0), abs(IAAO_PRD_RANGE[1] - 1.0))
        return 0.0, 0.0, float(max_dev)
    if metric_name in ("PRB_mean", "PRB"):
        lo, hi = IAAO_PRB_RANGE
        return 0.0, float(lo), float(hi)
    if metric_name == "abs_PRB_mean":
        max_abs = max(abs(IAAO_PRB_RANGE[0]), abs(IAAO_PRB_RANGE[1]))
        return 0.0, 0.0, float(max_abs)
    if metric_name in ("VEI_mean", "VEI"):
        lo, hi = IAAO_VEI_RANGE
        return 0.0, float(lo), float(hi)
    if metric_name == "abs_VEI_mean":
        max_abs = max(abs(IAAO_VEI_RANGE[0]), abs(IAAO_VEI_RANGE[1]))
        return 0.0, 0.0, float(max_abs)
    return None


def _add_fairness_reference_x(ax: plt.Axes, fairness_col: str) -> None:
    ref = _reference_for_metric(fairness_col)
    if ref is None:
        return
    ideal, lo, hi = ref
    ax.axvline(ideal, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="_nolegend_")
    ax.axvspan(lo, hi, color="#2ca02c", alpha=0.16, label="_nolegend_")


def _add_fairness_reference_y(ax: plt.Axes, fairness_col: str) -> None:
    ref = _reference_for_metric(fairness_col)
    if ref is None:
        return
    ideal, lo, hi = ref
    ax.axhline(ideal, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="_nolegend_")
    ax.axhspan(lo, hi, color="#2ca02c", alpha=0.16, label="_nolegend_")


def _fairness_objective_text(fairness_col: str) -> str:
    ref = _reference_for_metric(fairness_col)
    if ref is None:
        return "fairness objective"
    ideal = ref[0]
    if fairness_col.startswith("abs_"):
        return f"fairness objective: {fairness_col}={ideal:.0f}"
    return f"fairness objective: {fairness_col}={ideal:g}"


def _select_plot_space(df: pd.DataFrame, top_k_plot: Optional[int]) -> pd.DataFrame:
    if df.empty:
        return df
    if top_k_plot is None or int(top_k_plot) <= 0:
        return df.copy()
    acc_col = "OOS R2_mean" if "OOS R2_mean" in df.columns else ("R2_mean" if "R2_mean" in df.columns else None)
    if acc_col is None:
        return df.head(int(top_k_plot)).copy()
    return df.nlargest(int(top_k_plot), acc_col).copy()


def _prepare_plot_dataframe(summary_with_pareto: pd.DataFrame) -> pd.DataFrame:
    if summary_with_pareto.empty:
        return summary_with_pareto.copy()
    df = summary_with_pareto.copy()
    if "rho" not in df.columns:
        df["rho"] = df["model_name"].map(_extract_rho)
    else:
        df["rho"] = pd.to_numeric(df["rho"], errors="coerce")
    if "model_family" not in df.columns:
        df["model_family"] = df["model_name"].map(_base_family_name)
    else:
        df["model_family"] = df["model_family"].astype(str)
    return df


def _read_test_metrics_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.DataFrame()


def _resolve_test_metrics_path(result_root: str, out_dir: Path, test_metrics_path: Optional[str]) -> Optional[Path]:
    if test_metrics_path:
        p = Path(test_metrics_path)
        return p if p.exists() else None

    root = Path(result_root)
    candidates = [
        out_dir / "test_metrics.csv",
        out_dir / "test_metrics.parquet",
        root / "test_metrics.csv",
        root / "test_metrics.parquet",
        root.parent.parent / "tmp" / "rho_sweep_metrics_test.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _prepare_test_summary(test_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert a single-run test table to the same schema style used by
    validation summary plots (e.g., OOS R2_mean, abs_PRD_dev_mean).
    """
    if test_df.empty:
        return pd.DataFrame()

    df = test_df.copy()
    if "model_name" not in df.columns:
        if "model_repr" in df.columns:
            df["model_name"] = df["model_repr"].astype(str)
        else:
            return pd.DataFrame()

    # Keep one row per model/config identity for plotting.
    dedup_cols = [c for c in ["model_name", "model_repr", "rho"] if c in df.columns]
    if dedup_cols:
        df = df.drop_duplicates(subset=dedup_cols, keep="last").copy()

    out = pd.DataFrame()
    out["model_name"] = df["model_name"].astype(str)
    # Keep config_id if present so we can align with stacking weights.
    if "config_id" in df.columns:
        out["config_id"] = df["config_id"].astype(str)
    else:
        out["config_id"] = np.arange(df.shape[0]).astype(str)
    if "rho" in df.columns:
        out["rho"] = pd.to_numeric(df["rho"], errors="coerce")

    # Keep both accuracy metrics when available so test plots mirror validation.
    if "OOS R2" in df.columns:
        out["OOS R2_mean"] = pd.to_numeric(df["OOS R2"], errors="coerce")
    if "R2" in df.columns:
        out["R2_mean"] = pd.to_numeric(df["R2"], errors="coerce")

    if "PRB" in df.columns:
        out["PRB_mean"] = pd.to_numeric(df["PRB"], errors="coerce")
        out["abs_PRB_mean"] = np.abs(out["PRB_mean"])
    if "PRD" in df.columns:
        out["PRD_mean"] = pd.to_numeric(df["PRD"], errors="coerce")
        out["abs_PRD_dev_mean"] = np.abs(out["PRD_mean"] - 1.0)
    if "VEI" in df.columns:
        out["VEI_mean"] = pd.to_numeric(df["VEI"], errors="coerce")
        out["abs_VEI_mean"] = np.abs(out["VEI_mean"])
    # Prefer the CV-consistent "Corr(r,y)_price" if available; else fall back to Corr(r,price).
    if "Corr(r,y)_price" in df.columns:
        out["Corr(r,y)_price_mean"] = pd.to_numeric(df["Corr(r,y)_price"], errors="coerce")
        out["abs_Corr_r_y_price_mean"] = np.abs(out["Corr(r,y)_price_mean"])
    elif "Corr(r,price)" in df.columns:
        out["Corr(r,y)_price_mean"] = pd.to_numeric(df["Corr(r,price)"], errors="coerce")
        out["abs_Corr_r_y_price_mean"] = np.abs(out["Corr(r,y)_price_mean"])

    out["model_family"] = out["model_name"].map(_base_family_name)
    return out


def _pick_validation_single_run_fold_id(runs_df: pd.DataFrame) -> Optional[int]:
    """
    Choose a single validation fold to act as a "single-run" analogue to test.

    We pick the "hardest" fold: the fold with the lowest *mean* accuracy across
    all configs (preferring OOS R2 if available, else R2).
    """
    if runs_df.empty or "fold_id" not in runs_df.columns:
        return None
    acc_col = "OOS R2" if "OOS R2" in runs_df.columns else ("R2" if "R2" in runs_df.columns else None)
    if acc_col is None:
        return None
    tmp = runs_df[["fold_id", acc_col]].copy()
    tmp[acc_col] = pd.to_numeric(tmp[acc_col], errors="coerce")
    grp = tmp.groupby("fold_id", dropna=False)[acc_col].mean()
    grp = grp.dropna()
    if grp.empty:
        return None
    # Lowest mean accuracy = hardest fold.
    try:
        return int(grp.sort_values(ascending=True).index.tolist()[0])
    except Exception:
        return None


def _prepare_validation_single_run_df(runs_df: pd.DataFrame, fold_id: Optional[int]) -> pd.DataFrame:
    """
    Validation "single-run" rows for ONE fold, normalized to plotting schema.
    """
    if runs_df.empty:
        return pd.DataFrame()
    df = runs_df.copy()
    if fold_id is not None and "fold_id" in df.columns:
        df = df[pd.to_numeric(df["fold_id"], errors="coerce") == int(fold_id)].copy()
    if df.empty:
        return pd.DataFrame()
    if "PRB" in df.columns:
        df["abs_PRB_mean"] = np.abs(pd.to_numeric(df["PRB"], errors="coerce"))
    if "PRD" in df.columns:
        prd = pd.to_numeric(df["PRD"], errors="coerce")
        df["abs_PRD_dev_mean"] = np.abs(prd - 1.0)
    if "VEI" in df.columns:
        df["abs_VEI_mean"] = np.abs(pd.to_numeric(df["VEI"], errors="coerce"))
    if "Corr(r,y)_price" in df.columns:
        df["abs_Corr_r_y_price_mean"] = np.abs(pd.to_numeric(df["Corr(r,y)_price"], errors="coerce"))
    if "OOS R2" in df.columns and "OOS R2_mean" not in df.columns:
        df["OOS R2_mean"] = pd.to_numeric(df["OOS R2"], errors="coerce")
    if "R2" in df.columns and "R2_mean" not in df.columns:
        df["R2_mean"] = pd.to_numeric(df["R2"], errors="coerce")
    if "model_name" in df.columns:
        df["rho"] = df["model_name"].map(_extract_rho)
        df["model_family"] = df["model_name"].map(_base_family_name)
    return df


def _load_stacking_solution_overlay(out_dir: Path) -> Dict[str, Any]:
    """
    Reads stacking_pf_opt artifacts (if present) and returns:
      - avg: dict with ensemble average metrics
      - folds: DataFrame with ensemble fold metrics
    """
    stack_dir = out_dir / "stacking_pf_opt"
    fold_path = stack_dir / "fold_ensemble_metrics.csv"
    if not fold_path.exists():
        return {}
    try:
        fold_df = pd.read_csv(fold_path)
    except Exception:
        return {}
    if fold_df.empty:
        return {}

    # Normalize expected naming to plot metric names.
    rename = {}
    if "OOS_R2_ensemble" in fold_df.columns:
        rename["OOS_R2_ensemble"] = "OOS R2_mean"
    if "R2_ensemble" in fold_df.columns:
        rename["R2_ensemble"] = "R2_mean"
    if "PRD_ensemble" in fold_df.columns:
        rename["PRD_ensemble"] = "PRD_mean"
    if "PRB_ensemble" in fold_df.columns:
        rename["PRB_ensemble"] = "PRB_mean"
    if "VEI_ensemble" in fold_df.columns:
        rename["VEI_ensemble"] = "VEI_mean"
    fold_df = fold_df.rename(columns=rename)

    if "PRD_mean" in fold_df.columns:
        fold_df["abs_PRD_dev_mean"] = np.abs(pd.to_numeric(fold_df["PRD_mean"], errors="coerce") - 1.0)
    if "PRB_mean" in fold_df.columns:
        fold_df["abs_PRB_mean"] = np.abs(pd.to_numeric(fold_df["PRB_mean"], errors="coerce"))
    if "VEI_mean" in fold_df.columns:
        fold_df["abs_VEI_mean"] = np.abs(pd.to_numeric(fold_df["VEI_mean"], errors="coerce"))
    # Corr may not exist in optimizer fold outputs.
    if "abs_Corr_r_y_price_mean" not in fold_df.columns:
        fold_df["abs_Corr_r_y_price_mean"] = np.nan

    avg = {}
    for c in ["OOS R2_mean", "R2_mean", "abs_PRD_dev_mean", "abs_PRB_mean", "abs_VEI_mean", "abs_Corr_r_y_price_mean"]:
        if c in fold_df.columns:
            avg[c] = float(pd.to_numeric(fold_df[c], errors="coerce").mean())

    return {"avg": avg, "folds": fold_df}


def _compute_test_stacking_overlay(
    out_dir: Path,
    protocol: Dict[str, Any],
    block_freq: str = "Q",
) -> Dict[str, Any]:
    """
    Build true test-set stacking metrics from per-row predictions + optimal weights.

    Required artifacts:
      - stacking_pf_opt/weights.csv
      - test_predictions.parquet
    """
    stack_dir = out_dir / "stacking_pf_opt"
    weights_path = stack_dir / "weights.csv"
    preds_path = out_dir / "test_predictions.parquet"
    if (not weights_path.exists()) or (not preds_path.exists()):
        return {}

    try:
        wdf = pd.read_csv(weights_path)
        pdf = pd.read_parquet(preds_path)
    except Exception:
        return {}
    if wdf.empty or pdf.empty:
        return {}
    if "model_id" not in wdf.columns or "weight" not in wdf.columns:
        return {}
    required_pred_cols = {"config_id", "model_name", "row_id", "y_true", "y_pred", "sale_date"}
    if not required_pred_cols.issubset(set(pdf.columns)):
        return {}

    w = wdf.copy()
    # model_id schema from optimize_stacking_pf: "config_id | model_name"
    split_parts = w["model_id"].astype(str).str.split(" | ", n=1, regex=False)
    w["config_id"] = split_parts.str[0].astype(str)
    w["model_name"] = split_parts.str[1].astype(str)
    w["weight"] = pd.to_numeric(w["weight"], errors="coerce")
    w = w[np.isfinite(w["weight"]) & (w["weight"] > 0)].copy()
    if w.empty:
        return {}

    merged = pdf.merge(w[["config_id", "model_name", "weight"]], on=["config_id", "model_name"], how="inner")
    if merged.empty:
        return {}

    merged["y_pred"] = pd.to_numeric(merged["y_pred"], errors="coerce")
    merged["y_true"] = pd.to_numeric(merged["y_true"], errors="coerce")
    merged["weight"] = pd.to_numeric(merged["weight"], errors="coerce")
    merged["sale_date"] = pd.to_datetime(merged["sale_date"], errors="coerce")
    merged = merged.dropna(subset=["row_id", "y_true", "y_pred", "weight", "sale_date"]).copy()
    if merged.empty:
        return {}

    merged["wyp"] = merged["weight"].to_numpy(dtype=float) * merged["y_pred"].to_numpy(dtype=float)
    pred_sum = merged.groupby("row_id", dropna=False)["wyp"].sum().rename("y_pred").reset_index()
    meta = (
        merged.sort_values(["row_id"])
        .groupby("row_id", dropna=False)
        .agg({"y_true": "first", "sale_date": "first"})
        .reset_index()
    )
    blended = meta.merge(pred_sum, on="row_id", how="inner")
    blended = blended[np.isfinite(blended["y_true"]) & np.isfinite(blended["y_pred"]) & (blended["y_true"] > 0) & (blended["y_pred"] > 0)]
    if blended.empty:
        return {}

    train_log_mean = np.nan
    if "y_train_log_mean" in merged.columns:
        train_log_mean = pd.to_numeric(merged["y_train_log_mean"], errors="coerce").dropna().mean()
    y_train_stub = np.array([float(train_log_mean)]) if np.isfinite(train_log_mean) else None

    y_true_log = np.log(blended["y_true"].to_numpy(dtype=float))
    y_pred_log = np.log(blended["y_pred"].to_numpy(dtype=float))
    test_metrics = _compute_extended_metrics(
        y_true_log=y_true_log,
        y_pred_log=y_pred_log,
        y_train_log=(y_train_stub if y_train_stub is not None else y_true_log),
        ratio_mode="diff",
    )
    avg_overlay = {
        "OOS R2_mean": float(test_metrics["OOS R2"]) if "OOS R2" in test_metrics else np.nan,
        "R2_mean": float(test_metrics["R2"]) if "R2" in test_metrics else np.nan,
        "abs_PRD_dev_mean": float(abs(float(test_metrics["PRD"]) - 1.0)) if "PRD" in test_metrics else np.nan,
        "abs_PRB_mean": float(abs(float(test_metrics["PRB"]))) if "PRB" in test_metrics else np.nan,
        "abs_VEI_mean": float(abs(float(test_metrics["VEI"]))) if "VEI" in test_metrics else np.nan,
        "abs_Corr_r_y_price_mean": float(abs(float(test_metrics["Corr(r,y)_price"]))) if "Corr(r,y)_price" in test_metrics else np.nan,
    }

    freq = str(block_freq).strip() if block_freq else "Q"
    try:
        periods = pd.to_datetime(blended["sale_date"], errors="coerce").dt.to_period(freq)
    except Exception:
        periods = pd.to_datetime(blended["sale_date"], errors="coerce").dt.to_period("Q")
        freq = "Q"
    block_rows = []
    for blk in sorted([b for b in periods.dropna().unique()]):
        idx = (periods == blk).to_numpy()
        if int(np.sum(idx)) < 10:
            continue
        y_t = y_true_log[idx]
        y_p = y_pred_log[idx]
        m_blk = _compute_extended_metrics(
            y_true_log=y_t,
            y_pred_log=y_p,
            y_train_log=(y_train_stub if y_train_stub is not None else y_true_log),
            ratio_mode="diff",
        )
        block_rows.append(
            {
                "block": str(blk),
                "n_rows": int(np.sum(idx)),
                "OOS R2": float(m_blk["OOS R2"]) if "OOS R2" in m_blk else np.nan,
                "R2": float(m_blk["R2"]) if "R2" in m_blk else np.nan,
                "PRD": float(m_blk["PRD"]) if "PRD" in m_blk else np.nan,
                "PRB": float(m_blk["PRB"]) if "PRB" in m_blk else np.nan,
                "VEI": float(m_blk["VEI"]) if "VEI" in m_blk else np.nan,
                "Corr(r,y)_price": float(m_blk["Corr(r,y)_price"]) if "Corr(r,y)_price" in m_blk else np.nan,
                "abs_PRD_dev_mean": float(abs(float(m_blk["PRD"]) - 1.0)) if "PRD" in m_blk else np.nan,
                "abs_PRB_mean": float(abs(float(m_blk["PRB"]))) if "PRB" in m_blk else np.nan,
                "abs_VEI_mean": float(abs(float(m_blk["VEI"]))) if "VEI" in m_blk else np.nan,
                "abs_Corr_r_y_price_mean": float(abs(float(m_blk["Corr(r,y)_price"]))) if "Corr(r,y)_price" in m_blk else np.nan,
            }
        )
    block_df = pd.DataFrame(block_rows)

    worst_overlay = {}
    if not block_df.empty:
        sort_col = "OOS R2" if "OOS R2" in block_df.columns and block_df["OOS R2"].notna().any() else "R2"
        if sort_col in block_df.columns:
            worst = block_df.sort_values(sort_col, ascending=True).iloc[0]
            worst_overlay = {
                "OOS R2_mean": float(worst.get("OOS R2", np.nan)),
                "R2_mean": float(worst.get("R2", np.nan)),
                "abs_PRD_dev_mean": float(worst.get("abs_PRD_dev_mean", np.nan)),
                "abs_PRB_mean": float(worst.get("abs_PRB_mean", np.nan)),
                "abs_VEI_mean": float(worst.get("abs_VEI_mean", np.nan)),
                "abs_Corr_r_y_price_mean": float(worst.get("abs_Corr_r_y_price_mean", np.nan)),
            }

    stack_dir.mkdir(parents=True, exist_ok=True)
    avg_out = pd.DataFrame(
        [
            {
                "data_id": None,
                "split_id": None,
                "kind": "test_avg",
                "block_freq": freq,
                **avg_overlay,
            }
        ]
    )
    avg_out.to_csv(stack_dir / "test_ensemble_metrics.csv", index=False)
    if not block_df.empty:
        block_df.to_csv(stack_dir / "test_ensemble_block_metrics.csv", index=False)
    if worst_overlay:
        pd.DataFrame([{"kind": "test_worst_block", "block_freq": freq, **worst_overlay}]).to_csv(
            stack_dir / "test_ensemble_worst_block_metrics.csv", index=False
        )

    return {
        "avg": avg_overlay,
        "worst": worst_overlay,
        "block_metrics": block_df,
        "block_freq": freq,
    }


def _compute_2d_pareto_mask(df: pd.DataFrame, acc_col: str, fairness_col: str) -> np.ndarray:
    """
    Compute Pareto mask using ONLY the 2D objectives shown in a tradeoff plot:
    maximize accuracy, minimize fairness distance metric.
    """
    if df.empty or acc_col not in df.columns or fairness_col not in df.columns:
        return np.zeros(df.shape[0], dtype=bool)

    vals = df[[acc_col, fairness_col]].to_numpy(dtype=float)
    ok = np.all(np.isfinite(vals), axis=1)
    mask = np.zeros(df.shape[0], dtype=bool)
    if np.any(ok):
        # acc: max, fairness: min
        mask_ok = _is_pareto_efficient(vals[ok], maximize_mask=np.array([True, False], dtype=bool))
        mask[np.where(ok)[0]] = mask_ok
    return mask


def _verify_test_metrics_match_protocol(
    test_df: pd.DataFrame,
    data_id: str,
    split_id: str,
    protocol: Dict[str, Any],
    strict: bool,
) -> Tuple[bool, str]:
    """
    Safeguard against accidental leakage/mismatch:
    - requires split_name='test'
    - if present, requires data_id/split_id match current analysis
    - if date metadata present, requires test_max_sale_date year <= 2023
      and strictly after the latest validation end date in protocol folds.
    In strict mode, missing required metadata fails closed.
    """
    if test_df.empty:
        return False, "test metrics file is empty"

    if "split_name" not in test_df.columns:
        if strict:
            return False, "missing split_name column in test metrics (strict safeguard)"
    else:
        vals = {str(v).strip().lower() for v in test_df["split_name"].dropna().unique().tolist()}
        if vals and vals != {"test"}:
            return False, f"split_name must be only 'test', got: {sorted(vals)}"

    if "data_id" in test_df.columns:
        ids = {str(v).strip() for v in test_df["data_id"].dropna().unique().tolist()}
        if ids and ids != {str(data_id)}:
            return False, f"data_id mismatch: expected {data_id}, got {sorted(ids)}"
    elif strict:
        return False, "missing data_id column in test metrics (strict safeguard)"

    if "split_id" in test_df.columns:
        ids = {str(v).strip() for v in test_df["split_id"].dropna().unique().tolist()}
        if ids and ids != {str(split_id)}:
            return False, f"split_id mismatch: expected {split_id}, got {sorted(ids)}"
    elif strict:
        return False, "missing split_id column in test metrics (strict safeguard)"

    folds = protocol.get("folds", []) if isinstance(protocol, dict) else []
    max_val_end = None
    try:
        if folds:
            max_val_end = max(pd.to_datetime(f.get("val_end")) for f in folds if f.get("val_end"))
    except Exception:
        max_val_end = None

    if "test_max_sale_date" in test_df.columns:
        max_test_dt = pd.to_datetime(test_df["test_max_sale_date"], errors="coerce").max()
        if pd.notna(max_test_dt) and int(max_test_dt.year) > 2023:
            return False, f"test_max_sale_date year must be <= 2023, got {max_test_dt.date()}"
    elif strict:
        return False, "missing test_max_sale_date column in test metrics (strict safeguard)"

    if "test_min_sale_date" in test_df.columns and max_val_end is not None:
        min_test_dt = pd.to_datetime(test_df["test_min_sale_date"], errors="coerce").min()
        if pd.notna(min_test_dt) and pd.notna(max_val_end) and not (min_test_dt > max_val_end):
            return False, (
                f"test_min_sale_date must be after protocol max val_end ({max_val_end.date()}), "
                f"got {min_test_dt.date()}"
            )
    elif strict and max_val_end is not None:
        return False, "missing test_min_sale_date column in test metrics (strict safeguard)"

    return True, "ok"


def plot_analysis(
    summary_with_pareto: pd.DataFrame,
    out_dir: Path,
    split_label: str,
    plot_top_k: Optional[int] = None,
    optimized_overlay: Optional[Dict[str, Any]] = None,
) -> None:
    if summary_with_pareto.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)

    df_plot = _prepare_plot_dataframe(_select_plot_space(summary_with_pareto, top_k_plot=plot_top_k)).reset_index(drop=True)
    acc_cols = [c for c in ["OOS R2_mean", "R2_mean"] if c in df_plot.columns]
    if not acc_cols:
        return
    fairness_cols = [c for c in ["abs_PRB_mean", "abs_PRD_dev_mean", "abs_VEI_mean", "abs_Corr_r_y_price_mean"] if c in df_plot.columns]
    color_by_family = _build_family_color_map(df_plot["model_family"].dropna().astype(str).tolist())
    intensity_by_family_rho = _build_family_rho_intensity(df_plot)

    # Tradeoff scatter with model-family colors, rho intensity, and Pareto outline.
    for acc_col in acc_cols:
        for fcol in fairness_cols:
            fig, ax = plt.subplots(figsize=(8, 6))
            x = pd.to_numeric(df_plot[fcol], errors="coerce")
            y = pd.to_numeric(df_plot[acc_col], errors="coerce")
            mask_p = _compute_2d_pareto_mask(df_plot, acc_col=acc_col, fairness_col=fcol)

            for i, row in df_plot.iterrows():
                xv, yv = _safe_to_float(x.get(i, np.nan)), _safe_to_float(y.get(i, np.nan))
                if not np.isfinite(xv) or not np.isfinite(yv):
                    continue
                fam = str(row.get("model_family", row.get("model_name", "model")))
                rho_v = _safe_to_float(row.get("rho", np.nan))
                is_baseline = not np.isfinite(rho_v)
                base_color = color_by_family.get(fam, (0.3, 0.3, 0.3))
                if is_baseline:
                    point_color = mcolors.to_rgb(base_color)
                    marker = "*"
                    size = 130
                else:
                    intensity = intensity_by_family_rho.get((fam, float(rho_v)), 0.85)
                    point_color = _shade_color(base_color, intensity)
                    marker = "o"
                    size = 55
                ax.scatter(
                    [xv],
                    [yv],
                    marker=marker,
                    s=size,
                    c=[point_color],
                    alpha=0.92,
                    edgecolors="white",
                    linewidths=0.6,
                )
                if mask_p[i]:
                    ax.scatter(
                        [xv],
                        [yv],
                        marker="o",
                        s=size + 70,
                        facecolors="none",
                        edgecolors="black",
                        linewidths=1.3,
                        alpha=0.95,
                    )

            # Overlay optimized ensemble average point (if available).
            drew_opt_avg = False
            if optimized_overlay and isinstance(optimized_overlay.get("avg"), dict):
                avg_row = optimized_overlay["avg"]
                ov_x = _safe_to_float(avg_row.get(fcol, np.nan))
                ov_y = _safe_to_float(avg_row.get(acc_col, np.nan))
                if np.isfinite(ov_x) and np.isfinite(ov_y):
                    ax.scatter(
                        [ov_x],
                        [ov_y],
                        marker="D",
                        s=130,
                        c=["#ff1493"],
                        edgecolors="black",
                        linewidths=1.1,
                        alpha=0.95,
                    )
                    drew_opt_avg = True
            drew_opt_worst = False
            if optimized_overlay and isinstance(optimized_overlay.get("worst"), dict):
                worst_row = optimized_overlay["worst"]
                ov_x = _safe_to_float(worst_row.get(fcol, np.nan))
                ov_y = _safe_to_float(worst_row.get(acc_col, np.nan))
                if np.isfinite(ov_x) and np.isfinite(ov_y):
                    ax.scatter(
                        [ov_x],
                        [ov_y],
                        marker="^",
                        s=120,
                        c=["#ff8c00"],
                        edgecolors="black",
                        linewidths=1.0,
                        alpha=0.95,
                    )
                    drew_opt_worst = True

            # Overlay optimized fold-level points on single-run style plots.
            drew_opt_folds = False
            if "single_run" in split_label.lower() and optimized_overlay and isinstance(optimized_overlay.get("folds"), pd.DataFrame):
                ov_fold = optimized_overlay["folds"]
                single_fold_id = optimized_overlay.get("single_fold_id", None)
                if single_fold_id is not None and "fold_id" in ov_fold.columns:
                    ov_fold = ov_fold[pd.to_numeric(ov_fold["fold_id"], errors="coerce") == int(single_fold_id)].copy()
                if (fcol in ov_fold.columns) and (acc_col in ov_fold.columns):
                    ov_fx = pd.to_numeric(ov_fold[fcol], errors="coerce")
                    ov_fy = pd.to_numeric(ov_fold[acc_col], errors="coerce")
                    ok = np.isfinite(ov_fx.to_numpy()) & np.isfinite(ov_fy.to_numpy())
                    if np.any(ok):
                        ax.scatter(
                            ov_fx.to_numpy()[ok],
                            ov_fy.to_numpy()[ok],
                            marker="X",
                            s=85,
                            c="#8a2be2",
                            edgecolors="black",
                            linewidths=0.8,
                            alpha=0.9,
                        )
                        drew_opt_folds = True

            _add_fairness_reference_x(ax, fairness_col=fcol)
            ref_x = _reference_for_metric(fcol)
            if ref_x is not None:
                ideal_x = float(ref_x[0])
                y_bottom = float(np.nanmin(y.values)) if np.any(np.isfinite(y.values)) else 0.0
                y_top = float(np.nanmax(y.values)) if np.any(np.isfinite(y.values)) else 1.0
                y_text = y_bottom + 0.02 * (y_top - y_bottom if y_top > y_bottom else 1.0)
                ax.text(
                    ideal_x,
                    y_text,
                    _fairness_objective_text(fcol),
                    fontsize=7.5,
                    color="black",
                    alpha=0.85,
                    ha="left",
                    va="bottom",
                )
            ax.set_xlabel(fcol)
            ax.set_ylabel(acc_col)
            topk_note = f"top_k={int(plot_top_k)}" if (plot_top_k is not None and int(plot_top_k) > 0) else "full space"
            ax.set_title(f"Tradeoff: {acc_col} vs {fcol}", fontsize=11, pad=14)
            ax.text(
                0.5,
                1.01,
                f"{split_label} | {topk_note}",
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=8,
                alpha=0.85,
            )
            ax.grid(True, alpha=0.3)

            baseline_families = {
                str(fam)
                for fam, sub in df_plot.groupby("model_family", dropna=False)
                if np.any(~np.isfinite(pd.to_numeric(sub["rho"], errors="coerce")))
            }
            legend_handles: List[Line2D] = []
            for fam, base_color in sorted(color_by_family.items(), key=lambda x: x[0]):
                fam_marker = "*" if str(fam) in baseline_families else "o"
                fam_size = 10 if fam_marker == "*" else 8
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker=fam_marker,
                        color="none",
                        markerfacecolor=base_color,
                        markeredgecolor="white",
                        markeredgewidth=0.5,
                        markersize=fam_size,
                        label=fam,
                    )
                )
            legend_handles.append(Line2D([0], [0], color="black", linestyle="-", linewidth=1.0, label=""))
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="*",
                    color="none",
                    markerfacecolor="#666666",
                    markeredgecolor="white",
                    markeredgewidth=0.5,
                    markersize=11,
                    label="Baselines",
                )
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="black",
                    markerfacecolor="none",
                    linestyle="None",
                    markersize=8,
                    label="pareto 2D (outline)",
                )
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color="#2ca02c",
                    linestyle="-",
                    linewidth=6.0,
                    alpha=0.28,
                    label="IAAO acceptable band",
                )
            )
            if drew_opt_avg:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="D",
                        color="none",
                        markerfacecolor="#ff1493",
                        markeredgecolor="black",
                        markeredgewidth=1.0,
                        markersize=8,
                        label="stacking optimum (avg)",
                    )
                )
            if drew_opt_worst:
                legend_handles.append(
                    Line2D(
                        [0],
                        [0],
                        marker="^",
                        color="none",
                        markerfacecolor="#ff8c00",
                        markeredgecolor="black",
                        markeredgewidth=1.0,
                        markersize=8,
                        label="stacking optimum (worst block)",
                    )
                )
            if drew_opt_folds and "single_run" in split_label.lower():
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            marker="X",
                            color="none",
                            markerfacecolor="#8a2be2",
                            markeredgecolor="black",
                            markeredgewidth=0.8,
                            markersize=7,
                            label="stacking optimum (folds)",
                        )
                    )
            ax.legend(handles=legend_handles, loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"tradeoff_{acc_col.replace(' ', '_')}_vs_{fcol}.png", dpi=220)
            plt.close(fig)

    # Paired rho-evolution: accuracy (left axis) + fairness metric (right axis).
    evo = df_plot[np.isfinite(pd.to_numeric(df_plot["rho"], errors="coerce"))].copy()
    baselines = df_plot[~np.isfinite(pd.to_numeric(df_plot["rho"], errors="coerce"))].copy()
    if not evo.empty:
        for acc_col in acc_cols:
            for fairness_metric in fairness_cols:
                fig, ax_left = plt.subplots(figsize=(9.5, 6))
                ax_right = ax_left.twinx()
                legend_handles: List[Line2D] = []
                for fam, sub in evo.groupby("model_family"):
                    sub = sub.sort_values("rho")
                    color = color_by_family.get(str(fam), (0.3, 0.3, 0.3))
                    ax_left.plot(
                        sub["rho"],
                        sub[acc_col],
                        color=color,
                        marker="o",
                        linestyle="-",
                        linewidth=1.8,
                        alpha=0.9,
                    )
                    ax_right.plot(
                        sub["rho"],
                        sub[fairness_metric],
                        color=color,
                        marker="s",
                        linestyle="--",
                        linewidth=1.5,
                        alpha=0.9,
                    )
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            color=color,
                            linestyle="-",
                            marker="o",
                            markersize=5,
                            linewidth=1.8,
                            label=str(fam),
                        )
                    )

                x_min = float(np.nanmin(pd.to_numeric(evo["rho"], errors="coerce").values))
                x_max = float(np.nanmax(pd.to_numeric(evo["rho"], errors="coerce").values))
                for _, brow in baselines.iterrows():
                    fam = str(brow.get("model_family", brow.get("model_name", "baseline")))
                    color = color_by_family.get(fam, (0.3, 0.3, 0.3))
                    y_acc = _safe_to_float(brow.get(acc_col, np.nan))
                    y_fair = _safe_to_float(brow.get(fairness_metric, np.nan))
                    if np.isfinite(y_acc):
                        ax_left.plot(
                            [x_min, x_max],
                            [y_acc, y_acc],
                            color=color,
                            linestyle="-",
                            linewidth=1.8,
                            marker="*",
                            markersize=7,
                            alpha=0.95,
                        )
                    if np.isfinite(y_fair):
                        ax_right.plot(
                            [x_min, x_max],
                            [y_fair, y_fair],
                            color=color,
                            linestyle="--",
                            linewidth=1.5,
                            marker="P",
                            markersize=6,
                            alpha=0.9,
                        )
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            color=color,
                            linestyle="-",
                            marker="*",
                            markersize=7,
                            linewidth=1.6,
                            label=f"{fam} baseline ({acc_col})",
                        )
                    )
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            color=color,
                            linestyle="--",
                            marker="P",
                            markersize=6,
                            linewidth=1.5,
                            label=f"{fam} baseline ({fairness_metric})",
                        )
                    )

                # Optimized ensemble average reference in rho-evolution panels.
                if optimized_overlay and isinstance(optimized_overlay.get("avg"), dict):
                    avg_row = optimized_overlay["avg"]
                    ov_acc = _safe_to_float(avg_row.get(acc_col, np.nan))
                    ov_fair = _safe_to_float(avg_row.get(fairness_metric, np.nan))
                    if np.isfinite(ov_acc):
                        ax_left.plot(
                            [x_min, x_max],
                            [ov_acc, ov_acc],
                            color="#ff1493",
                            linestyle="-.",
                            linewidth=1.8,
                            marker="D",
                            markersize=6,
                            alpha=0.95,
                        )
                    if np.isfinite(ov_fair):
                        ax_right.plot(
                            [x_min, x_max],
                            [ov_fair, ov_fair],
                            color="#ff1493",
                            linestyle="-.",
                            linewidth=1.6,
                            marker="D",
                            markersize=5,
                            alpha=0.9,
                        )
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            color="#ff1493",
                            linestyle="-.",
                            marker="D",
                            markersize=6,
                            linewidth=1.7,
                            label="stacking optimum (avg)",
                        )
                    )
                if optimized_overlay and isinstance(optimized_overlay.get("worst"), dict):
                    worst_row = optimized_overlay["worst"]
                    ov_acc_worst = _safe_to_float(worst_row.get(acc_col, np.nan))
                    ov_fair_worst = _safe_to_float(worst_row.get(fairness_metric, np.nan))
                    if np.isfinite(ov_acc_worst):
                        ax_left.plot(
                            [x_min, x_max],
                            [ov_acc_worst, ov_acc_worst],
                            color="#ff8c00",
                            linestyle=":",
                            linewidth=1.8,
                            marker="^",
                            markersize=6,
                            alpha=0.95,
                        )
                    if np.isfinite(ov_fair_worst):
                        ax_right.plot(
                            [x_min, x_max],
                            [ov_fair_worst, ov_fair_worst],
                            color="#ff8c00",
                            linestyle=":",
                            linewidth=1.6,
                            marker="^",
                            markersize=5,
                            alpha=0.9,
                        )
                    legend_handles.append(
                        Line2D(
                            [0],
                            [0],
                            color="#ff8c00",
                            linestyle=":",
                            marker="^",
                            markersize=6,
                            linewidth=1.7,
                            label="stacking optimum (worst block)",
                        )
                    )

                _add_fairness_reference_y(ax_right, fairness_col=fairness_metric)
                ax_left.set_xscale("log")
                ax_left.set_xlabel("rho (log scale)")
                ax_left.set_ylabel(acc_col)
                ax_right.set_ylabel(fairness_metric)
                ax_left.set_title(
                    f"Rho evolution: {acc_col} (left, solid) + {fairness_metric} (right, dashed)",
                    fontsize=11,
                    pad=14,
                )
                ax_left.text(
                    0.5,
                    1.01,
                    split_label,
                    transform=ax_left.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    alpha=0.85,
                )
                ax_left.grid(True, alpha=0.3)
                legend_handles.append(
                    Line2D([0], [0], color="black", linestyle="--", linewidth=1.0, label="fairness ideal / IAAO band")
                )
                ax_left.legend(handles=legend_handles, loc="best", fontsize=8)
                fig.tight_layout()
                fig.savefig(
                    out_dir / f"rho_evolution_{acc_col.replace(' ', '_')}_and_{fairness_metric}.png",
                    dpi=220,
                )
                plt.close(fig)


def run_results_analysis(
    result_root: str = "./output/robust_rolling_origin_cv",
    data_id: Optional[str] = None,
    split_id: Optional[str] = None,
    top_k: int = 10,
    fairness_norm_threshold: float = 0.35,
    save_shortlist: bool = True,
    test_metrics_path: Optional[str] = None,
    plot_top_k: Optional[int] = None,
    strict_test_safeguard: bool = True,
) -> Dict[str, Any]:
    loaded = load_cv_artifacts(result_root=result_root, data_id=data_id, split_id=split_id)
    data_id = loaded["data_id"]
    split_id = loaded["split_id"]
    protocol = loaded.get("protocol", {})
    runs_df = loaded["runs_df"]
    bs_df = loaded["bootstrap_df"]
    status_df = loaded["status_df"]

    out_dir = Path(result_root) / "analysis" / f"data_id={data_id}" / f"split_id={split_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Save/read fold-level metrics artifacts.
    runs_path = out_dir / "run_metrics_fold_level.parquet"
    status_path = out_dir / "run_status_table.parquet"
    bs_path = out_dir / "bootstrap_metrics_fold_level.parquet"
    if not runs_df.empty:
        runs_df.to_parquet(runs_path, index=False)
    if not status_df.empty:
        status_df.to_parquet(status_path, index=False)
    if not bs_df.empty:
        bs_df.to_parquet(bs_path, index=False)

    # 2) summarize -> 3) pareto -> 4) shortlist + plots.
    summary_df = summarize_by_config(runs_df=runs_df)
    summary_path = out_dir / "summary_by_config.parquet"
    summary_csv = out_dir / "summary_by_config.csv"
    if not summary_df.empty:
        summary_df.to_parquet(summary_path, index=False)
        summary_df.to_csv(summary_csv, index=False)

    pareto_df = compute_pareto_front(summary_df)
    pareto_path = out_dir / "pareto_front.parquet"
    pareto_csv = out_dir / "pareto_front.csv"
    if not pareto_df.empty:
        pareto_df[pareto_df["is_pareto"]].to_parquet(pareto_path, index=False)
        pareto_df[pareto_df["is_pareto"]].to_csv(pareto_csv, index=False)

    shortlist = build_shortlist(
        summary_with_pareto=pareto_df,
        top_k=top_k,
        fairness_norm_threshold=fairness_norm_threshold,
    )
    if save_shortlist:
        with (out_dir / "shortlist.json").open("w", encoding="utf-8") as f:
            json.dump(shortlist, f, indent=2)

    split_label = f"data={data_id} split={split_id}"
    plots_root = out_dir / "plots"
    (plots_root / "validation").mkdir(parents=True, exist_ok=True)
    (plots_root / "validation_single_run").mkdir(parents=True, exist_ok=True)
    (plots_root / "test").mkdir(parents=True, exist_ok=True)
    optimized_overlay = _load_stacking_solution_overlay(out_dir=out_dir)

    single_run_fold_id = _pick_validation_single_run_fold_id(runs_df=runs_df)
    if optimized_overlay and isinstance(optimized_overlay, dict):
        optimized_overlay["single_fold_id"] = single_run_fold_id
    plot_analysis(
        summary_with_pareto=pareto_df,
        out_dir=plots_root / "validation",
        split_label=f"{split_label} | validation",
        plot_top_k=plot_top_k,
        optimized_overlay=optimized_overlay if optimized_overlay else None,
    )
    validation_single_run_df = _prepare_validation_single_run_df(runs_df=runs_df, fold_id=single_run_fold_id)
    if not validation_single_run_df.empty:
        plot_analysis(
            summary_with_pareto=validation_single_run_df,
            out_dir=plots_root / "validation_single_run",
            split_label=f"{split_label} | validation_single_run fold={single_run_fold_id}",
            plot_top_k=plot_top_k,
            optimized_overlay=optimized_overlay if optimized_overlay else None,
        )

    test_rows = 0
    resolved_test_path = _resolve_test_metrics_path(
        result_root=result_root,
        out_dir=out_dir,
        test_metrics_path=test_metrics_path,
    )
    if resolved_test_path is not None:
        test_df = _read_test_metrics_table(resolved_test_path)
        ok, msg = _verify_test_metrics_match_protocol(
            test_df=test_df,
            data_id=str(data_id),
            split_id=str(split_id),
            protocol=protocol if isinstance(protocol, dict) else {},
            strict=bool(strict_test_safeguard),
        )
        if not ok:
            raise ValueError(f"Test metrics safeguard failed: {msg}")

        test_summary_df = _prepare_test_summary(test_df)
        if not test_summary_df.empty:
            # Build true stacking-on-test overlay from per-row predictions + weights.
            # Use a quarterly notion of "worst block" for readability/stability.
            test_block_freq = "Q"
            optimized_test_overlay = _compute_test_stacking_overlay(
                out_dir=out_dir,
                protocol=protocol if isinstance(protocol, dict) else {},
                block_freq=test_block_freq,
            )
            test_summary_df.to_csv(out_dir / "test_summary_by_config.csv", index=False)
            test_pareto_df = compute_pareto_front(test_summary_df)
            if not test_pareto_df.empty and "is_pareto" in test_pareto_df.columns:
                test_pareto_df[test_pareto_df["is_pareto"]].to_csv(out_dir / "test_pareto_front.csv", index=False)
            plot_analysis(
                summary_with_pareto=test_pareto_df,
                out_dir=plots_root / "test",
                split_label=f"{split_label} | test",
                plot_top_k=plot_top_k,
                optimized_overlay=optimized_test_overlay if optimized_test_overlay else None,
            )
            test_rows = int(test_summary_df.shape[0])

    return {
        "data_id": data_id,
        "split_id": split_id,
        "analysis_dir": str(out_dir),
        "n_status_rows": int(status_df.shape[0]),
        "n_completed_runs": int(runs_df.shape[0]),
        "n_bootstrap_rows": int(bs_df.shape[0]),
        "n_summary_configs": int(summary_df.shape[0]) if not summary_df.empty else 0,
        "n_pareto": int(pareto_df["is_pareto"].sum()) if ("is_pareto" in pareto_df.columns) else 0,
        "n_test_rows": int(test_rows),
        "test_metrics_path_used": (str(resolved_test_path) if resolved_test_path is not None else None),
        "strict_test_safeguard": bool(strict_test_safeguard),
    }


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Read-only analysis over robust rolling-origin CV artifacts.")
    p.add_argument("--result-root", type=str, default="./output/robust_rolling_origin_cv")
    p.add_argument("--data-id", type=str, default=None)
    p.add_argument("--split-id", type=str, default=None)
    p.add_argument("--top-k", type=int, default=10)
    p.add_argument("--fairness-norm-threshold", type=float, default=0.35)
    p.add_argument("--no-shortlist", action="store_true", help="Do not write shortlist.json")
    p.add_argument(
        "--test-metrics-path",
        type=str,
        default=None,
        help="Optional CSV/Parquet with single-run test metrics to generate mirrored test plots.",
    )
    p.add_argument(
        "--plot-top-k",
        type=int,
        default=None,
        help="Optional: only plot the top-k configs by accuracy. Leave unset to plot full space.",
    )
    p.add_argument(
        "--no-strict-test-safeguard",
        action="store_true",
        help=(
            "Disable strict verification that test metrics metadata matches the "
            "current data_id/split_id and CCAO-style date boundaries."
        ),
    )
    return p


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    out = run_results_analysis(
        result_root=args.result_root,
        data_id=args.data_id,
        split_id=args.split_id,
        top_k=args.top_k,
        fairness_norm_threshold=args.fairness_norm_threshold,
        save_shortlist=not args.no_shortlist,
        test_metrics_path=args.test_metrics_path,
        plot_top_k=args.plot_top_k,
        strict_test_safeguard=not args.no_strict_test_safeguard,
    )
    print("=" * 90)
    print("RESULTS ANALYSIS COMPLETED")
    print("=" * 90)
    print(
        f"data_id={out['data_id']} | split_id={out['split_id']} | "
        f"completed_runs={out['n_completed_runs']} | summary_configs={out['n_summary_configs']} | "
        f"pareto={out['n_pareto']} | test_rows={out['n_test_rows']}"
    )
    if out.get("test_metrics_path_used"):
        print(f"test_metrics_path={out['test_metrics_path_used']}")
    print(f"strict_test_safeguard={out['strict_test_safeguard']}")
    print(f"analysis_dir={out['analysis_dir']}")
