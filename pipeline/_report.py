"""
Three-Model Comparison report — interactive HTML helpers.

This module builds the report compared in
``pipeline/Three-Model Comparison.html`` (the reference Quarto document):

  - Overview (run id, spec label, description per model)
  - Metric Table — Bootstrap tabset (Overall / City / North / South),
    best-of-3 in green, worst-of-3 in red.
  - Per-Metric Comparisons — one Plotly grouped bar chart per metric
    (R², RMSE, MAE, MAPE, MdAPE, Median Ratio, COD, PRD, PRB, MKI, VEI),
    with bars for each scope and traces per model.
  - Median Ratio Decile Curves — Overall (Plotly) + Per-Triad tabset.
    The IAAO target ratio band is shaded at 0.95–1.05 with a parity line at 1.0.
  - Decile Ratio Curves with Quartiles — same layout, with median ± IQR
    ribbons.
  - **Geography maps (tabbed per model)** — (1) **All 38 Cook County political townships**
    from official County GIS (layer 43), with mean % error colored where the test split has sales;
    (2) **PUMA** choropleths for finer geography. Triad outlines use a fixed North / City / South
    assignment aligned with assessor practice.

We delegate metric arithmetic to the same ``utils.motivation_utils`` helpers
used by the rest of the codebase so numbers agree with ``test_metrics.csv``
and ``selected_models_evaluation.csv``.
"""

from __future__ import annotations

import json
import sys
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from . import _geo_maps

from utils.motivation_utils import (
    IAAO_COD_RANGES,
    IAAO_LEVEL_RANGE,
    IAAO_PRB_RANGE,
    IAAO_PRD_RANGE,
    IAAO_VEI_RANGE,
    cod,
    mki,
    prb,
    prd,
    vei,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


# The decile target band used by the reference report (tighter than
# IAAO_LEVEL_RANGE which is [0.90, 1.10]).
IAAO_TARGET_BAND: Tuple[float, float] = (0.95, 1.05)


# Metrics shown in the metric table (and as panels). Order matters.
METRIC_ROWS: Sequence[str] = (
    "N",
    "R²",
    "RMSE",
    "MAE",
    "MAPE %",
    "MdAPE %",
    "Med Ratio",
    "COD",
    "PRD",
    "PRB",
    "MKI",
    "VEI",
)


# Mapping from display label -> internal computed key.
_METRIC_KEY: Dict[str, str] = {
    "N": "n_obs",
    "R²": "R2",
    "RMSE": "RMSE",
    "MAE": "MAE",
    "MAPE %": "MAPE_pct",
    "MdAPE %": "MdAPE_pct",
    "Med Ratio": "Median ratio",
    "COD": "COD",
    "PRD": "PRD",
    "PRB": "PRB",
    "MKI": "MKI",
    "VEI": "VEI",
}


# Direction used for best-of-3 / worst-of-3 highlighting:
#   "max"      → higher is better (R²)
#   "min"      → lower is better (RMSE, MAE, MAPE, MdAPE, COD)
#   "target_1" → closer to 1.0 is better (Median Ratio, PRD, MKI)
#   "target_0" → closer to 0.0 is better (PRB, VEI)
#   "none"     → never highlighted (N)
_METRIC_DIRECTION: Dict[str, str] = {
    "N": "none",
    "R²": "max",
    "RMSE": "min",
    "MAE": "min",
    "MAPE %": "min",
    "MdAPE %": "min",
    "Med Ratio": "target_1",
    "COD": "min",
    "PRD": "target_1",
    "PRB": "target_0",
    "MKI": "target_1",
    "VEI": "target_0",
}


# Numeric formatting per metric (digits, comma, percent).
_METRIC_FMT: Dict[str, Tuple[int, bool, bool]] = {
    "N": (0, True, False),
    "R²": (3, False, False),
    "RMSE": (0, True, False),
    "MAE": (0, True, False),
    "MAPE %": (3, False, False),
    "MdAPE %": (3, False, False),
    "Med Ratio": (3, False, False),
    "COD": (3, False, False),
    "PRD": (3, False, False),
    "PRB": (3, False, False),
    "MKI": (3, False, False),
    "VEI": (3, False, False),
}


# Scopes shown in tabsets and panels.
SCOPE_ORDER: Tuple[str, ...] = ("Overall", "City", "North", "South")
TRIAD_ONLY_ORDER: Tuple[str, ...] = ("City", "North", "South")


# Plotly-friendly model palette (color-blind safe).
_MODEL_PALETTE: Tuple[str, ...] = (
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
)


# ---------------------------------------------------------------------------
# Selected-model resolution
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSelection:
    label: str
    rule: str
    config_id: str
    model_name: str
    model_family: str
    model_config_json: str = ""
    description: str = ""


_RULE_DESCRIPTION: Dict[str, str] = {
    "linear_baseline": (
        "Untuned LinearRegression baseline trained on the same predictors as the "
        "LightGBM models. Useful as a non-flexible reference."
    ),
    "ccao_min_rmse": (
        "Configuration with the minimum mean fold RMSE across the LightGBM-flavored "
        "candidate pool. Mirrors the Cook County AVM ``select_best(lgbm_search, "
        "metric = 'rmse')`` selection. No fairness constraints."
    ),
    "nash": (
        "Penalized-model winner via **Nash-style product of utilities** (same construction "
        "as ``simple_model_selection.py::_select_nash_candidate``): transform fold-mean "
        "RMSE to ``1/RMSE``, R² to ``1+R²``, and map PRD / PRB / VEI through IAAO-band "
        "positive utilities, then maximize ``\\sum \\log u_j`` (**no across-candidate "
        "min–max normalization**). Pool defaults to ``LGBCovPenalty`` only."
    ),
    "utopia": "Legacy selection key in older ``selected_models.json`` files (treated as Nash).",
    "reference_baseline": "User-supplied reference configuration.",
}


def _config_choice(test_df: pd.DataFrame, *, family: str) -> Optional[Dict[str, Any]]:
    sub = test_df.loc[test_df["model_name"].astype(str) == family]
    if sub.empty:
        return None
    sub = sub.copy()
    sub["RMSE"] = pd.to_numeric(sub["RMSE"], errors="coerce")
    sub = sub.sort_values(by=["RMSE", "config_id"], ascending=[True, True])
    return sub.iloc[0].to_dict()


def resolve_three_models(
    *,
    selected_models_json: Dict[str, Any],
    test_metrics_df: pd.DataFrame,
    reference_config_id: Optional[str] = None,
) -> List[ModelSelection]:
    out: List[ModelSelection] = []

    if reference_config_id:
        ref_row = test_metrics_df.loc[
            test_metrics_df["config_id"].astype(str) == str(reference_config_id)
        ]
        if ref_row.empty:
            raise ValueError(f"reference_config_id {reference_config_id!r} not in test_metrics.csv")
        ref = ref_row.iloc[0].to_dict()
        out.append(
            ModelSelection(
                label="Reference baseline",
                rule="reference_baseline",
                config_id=str(ref["config_id"]),
                model_name=str(ref["model_name"]),
                model_family=str(ref["model_name"]).split("[", 1)[0],
                model_config_json=str(ref.get("model_config_json", "")),
                description=_RULE_DESCRIPTION["reference_baseline"],
            )
        )
    else:
        chosen = _config_choice(test_metrics_df, family="LinearRegression")
        label = "Linear baseline"
        if chosen is None:
            chosen = _config_choice(test_metrics_df, family="LGBMRegressor")
            label = "LightGBM baseline"
        if chosen is None:
            raise RuntimeError(
                "No baseline candidate found (need LinearRegression or LGBMRegressor in test_metrics.csv)."
            )
        out.append(
            ModelSelection(
                label=label,
                rule="linear_baseline",
                config_id=str(chosen["config_id"]),
                model_name=str(chosen["model_name"]),
                model_family=str(chosen["model_name"]).split("[", 1)[0],
                model_config_json=str(chosen.get("model_config_json", "")),
                description=_RULE_DESCRIPTION["linear_baseline"],
            )
        )

    sels = (selected_models_json or {}).get("selections", {}) or {}
    nash_sel = sels.get("nash") or sels.get("utopia")
    for base_rule, label in (("ccao_min_rmse", "CCAO min RMSE"), ("nash", "Nash equilibrium")):
        if base_rule == "nash":
            sel = nash_sel
        else:
            sel = sels.get(base_rule)
        if not sel:
            raise RuntimeError(
                f"selected_models.json is missing the '{base_rule}' selection "
                f"(for Nash, provide `nash` or legacy `utopia`). Run pipeline/02_assess.py first."
            )
        eff_rule = base_rule
        if base_rule == "nash" and "utopia" in sels and "nash" not in sels:
            eff_rule = "utopia"  # description fallback only
        out.append(
            ModelSelection(
                label=label,
                rule=eff_rule,
                config_id=str(sel["config_id"]),
                model_name=str(sel["model_name"]),
                model_family=str(sel.get("model_family", "")),
                model_config_json=str(sel.get("model_config_json", "")),
                description=_RULE_DESCRIPTION.get("nash", "")
                if base_rule == "nash"
                else _RULE_DESCRIPTION.get(base_rule, ""),
            )
        )

    seen, deduped = set(), []
    for m in out:
        if m.config_id in seen:
            continue
        seen.add(m.config_id)
        deduped.append(m)
    return deduped


# ---------------------------------------------------------------------------
# Predictions + geography join
# ---------------------------------------------------------------------------


def load_predictions_with_geography(
    *,
    result_root: Path,
    data_id: str,
    split_id: str,
    config_ids: Sequence[str],
    training_data_path: Path,
) -> pd.DataFrame:
    pred_path = (
        result_root / "analysis" / f"data_id={data_id}" / f"split_id={split_id}"
        / "test_predictions.parquet"
    )
    if not pred_path.is_file():
        raise FileNotFoundError(f"test_predictions.parquet not found: {pred_path}")
    cols = ["config_id", "model_name", "row_id", "sale_date", "y_true", "y_pred"]
    preds = pd.read_parquet(pred_path, columns=cols)
    preds["config_id"] = preds["config_id"].astype(str)
    preds = preds.loc[preds["config_id"].isin({str(c) for c in config_ids})].copy()
    preds["sale_date"] = pd.to_datetime(preds["sale_date"])
    # ``y_true`` is stored as ``exp(log(meta_sale_price))`` which introduces
    # ~1e-10 float noise; round to two decimals so the equality join with the
    # untransformed ``meta_sale_price`` succeeds for every test row.
    preds["_price_key"] = preds["y_true"].astype(float).round(2)

    geo_cols_want = [
        "meta_sale_date",
        "meta_sale_price",
        "meta_triad_name",
        "meta_township_name",
        "meta_class",
        "loc_latitude",
        "loc_longitude",
        "loc_census_puma_geoid",
    ]
    if not training_data_path.is_file():
        raise FileNotFoundError(f"training_data.parquet not found: {training_data_path}")
    import fastparquet

    pf = fastparquet.ParquetFile(str(training_data_path))
    present = set(pf.columns)
    geo_cols = [c for c in geo_cols_want if c in present]
    geo = pd.read_parquet(training_data_path, columns=geo_cols)
    for c in geo_cols_want:
        if c not in geo.columns:
            geo[c] = np.nan
    geo = geo.rename(columns={"meta_sale_date": "sale_date"})
    geo["sale_date"] = pd.to_datetime(geo["sale_date"])
    geo["_price_key"] = pd.to_numeric(geo["meta_sale_price"], errors="coerce").round(2)
    agg_spec: Dict[str, Tuple[str, str]] = {
        "meta_triad_name": ("meta_triad_name", "first"),
        "meta_township_name": ("meta_township_name", "first"),
        "meta_class": ("meta_class", "first"),
        "loc_latitude": ("loc_latitude", "first"),
        "loc_longitude": ("loc_longitude", "first"),
    }
    if "loc_census_puma_geoid" in geo.columns:
        agg_spec["loc_census_puma_geoid"] = ("loc_census_puma_geoid", "first")
    geo = (
        geo.dropna(subset=["sale_date", "_price_key"])
        .groupby(["sale_date", "_price_key"], as_index=False)
        .agg(**agg_spec)
    )
    merged = preds.merge(geo, on=["sale_date", "_price_key"], how="left")
    merged = merged.drop(columns=["_price_key"])
    merged["meta_triad_name"] = merged["meta_triad_name"].fillna("Unknown")
    merged = merged.loc[
        np.isfinite(merged["y_true"].astype(float))
        & np.isfinite(merged["y_pred"].astype(float))
        & (merged["y_true"].astype(float) > 0.0)
    ].copy()
    return merged


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _safe(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v if np.isfinite(v) else float("nan")


def _accuracy_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size == 0:
        return {k: float("nan") for k in ("R2", "MAE", "RMSE", "MAPE_pct", "MdAPE_pct")}
    err = y_pred - y_true
    abs_err = np.abs(err)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(abs_err))
    ape_pct = abs_err / np.maximum(np.abs(y_true), 1e-12) * 100.0
    mape_pct = float(np.mean(ape_pct))
    mdape_pct = float(np.median(ape_pct))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {"R2": r2, "MAE": mae, "RMSE": rmse, "MAPE_pct": mape_pct, "MdAPE_pct": mdape_pct}


def _ratio_study_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    if y_true.size < 2:
        return {k: float("nan") for k in ("Median ratio", "W. Mean ratio", "COD", "PRD", "PRB", "VEI", "MKI")}
    ratio = y_pred / y_true
    median_ratio = float(np.median(ratio))
    wmean_ratio = float(np.sum(y_pred) / np.sum(y_true)) if np.sum(y_true) > 0 else float("nan")
    return {
        "Median ratio": median_ratio,
        "W. Mean ratio": wmean_ratio,
        "COD": _safe(cod(ratio, na_rm=True)),
        "PRD": _safe(prd(y_pred, y_true, na_rm=True)),
        "PRB": _safe(prb(y_pred, y_true, na_rm=True)),
        "VEI": _safe(vei(y_pred, y_true, na_rm=True)),
        "MKI": _safe(mki(y_pred, y_true, na_rm=True)),
    }


def compute_scoped_metrics(
    df: pd.DataFrame,
    *,
    models: Sequence[ModelSelection],
    scope_label: str,
    n_min: int = 30,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for m in models:
        sub = df.loc[df["config_id"] == m.config_id]
        y_true = sub["y_true"].to_numpy(dtype=float)
        y_pred = sub["y_pred"].to_numpy(dtype=float)
        n = int(y_true.size)
        row: Dict[str, Any] = {
            "scope": scope_label,
            "model_label": m.label,
            "model_rule": m.rule,
            "config_id": m.config_id,
            "model_name": m.model_name,
            "n_obs": n,
        }
        if n < n_min:
            row["note"] = f"insufficient observations (<{n_min})"
            rows.append(row)
            continue
        row["mean_price"] = float(np.mean(y_true))
        row["median_price"] = float(np.median(y_true))
        row.update(_accuracy_metrics(y_true, y_pred))
        row.update(_ratio_study_metrics(y_true, y_pred))
        rows.append(row)
    return pd.DataFrame(rows)


def build_metrics_table(
    *,
    df: pd.DataFrame,
    models: Sequence[ModelSelection],
    triad_order: Sequence[str] = SCOPE_ORDER,
) -> pd.DataFrame:
    out: List[pd.DataFrame] = [compute_scoped_metrics(df, models=models, scope_label="Overall")]
    for triad in triad_order:
        if triad == "Overall":
            continue
        sub = df.loc[df["meta_triad_name"].astype(str) == triad]
        out.append(compute_scoped_metrics(sub, models=models, scope_label=triad))
    return pd.concat(out, ignore_index=True)


# ---------------------------------------------------------------------------
# Decile median + IQR table
# ---------------------------------------------------------------------------


def _ntile(values: np.ndarray, n: int) -> np.ndarray:
    out = np.full(values.shape, np.nan, dtype=float)
    finite = np.isfinite(values)
    if not np.any(finite):
        return out
    order = np.argsort(values[finite], kind="stable")
    rank = np.empty_like(order)
    rank[order] = np.arange(order.size)
    bins = np.floor(rank * n / max(order.size, 1)).astype(int) + 1
    bins = np.clip(bins, 1, n)
    out[finite] = bins
    return out


def compute_decile_curve(
    df: pd.DataFrame,
    *,
    models: Sequence[ModelSelection],
    n_deciles: int = 10,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for m in models:
        sub = df.loc[df["config_id"] == m.config_id].copy()
        if sub.empty:
            continue
        sub["decile"] = _ntile(sub["y_true"].to_numpy(dtype=float), n_deciles)
        sub = sub.dropna(subset=["decile"])
        sub["decile"] = sub["decile"].astype(int)
        sub["ratio"] = sub["y_pred"].astype(float) / sub["y_true"].astype(float)
        for decile, sub2 in sub.groupby("decile", sort=True):
            ratios = sub2["ratio"].to_numpy(dtype=float)
            ratios = ratios[np.isfinite(ratios)]
            if ratios.size == 0:
                continue
            rows.append(
                {
                    "model_label": m.label,
                    "model_rule": m.rule,
                    "config_id": m.config_id,
                    "decile": int(decile),
                    "n_obs": int(ratios.size),
                    "median_ratio": float(np.median(ratios)),
                    "mean_ratio": float(np.mean(ratios)),
                    "q25": float(np.quantile(ratios, 0.25)),
                    "q75": float(np.quantile(ratios, 0.75)),
                    "lower_price": float(np.min(sub2["y_true"])),
                    "upper_price": float(np.max(sub2["y_true"])),
                }
            )
    return pd.DataFrame(rows)


def build_decile_table(
    *,
    df: pd.DataFrame,
    models: Sequence[ModelSelection],
    n_deciles: int = 10,
    triad_order: Sequence[str] = SCOPE_ORDER,
) -> pd.DataFrame:
    parts: List[pd.DataFrame] = []
    for triad in triad_order:
        sub = df if triad == "Overall" else df.loc[df["meta_triad_name"].astype(str) == triad]
        deciles = compute_decile_curve(sub, models=models, n_deciles=n_deciles)
        if deciles.empty:
            continue
        deciles.insert(0, "scope", triad)
        parts.append(deciles)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# ---------------------------------------------------------------------------
# Best-of-3 / worst-of-3 highlighting helpers
# ---------------------------------------------------------------------------


def _ranking_score(value: float, direction: str) -> float:
    if not np.isfinite(value):
        return float("inf")
    if direction == "max":
        return -value
    if direction == "min":
        return value
    if direction == "target_1":
        return abs(value - 1.0)
    if direction == "target_0":
        return abs(value)
    return float("inf")


def _highlight_color(score: float, *, best: float, worst: float) -> str:
    if not np.isfinite(score):
        return ""
    if abs(score - best) < 1e-12:
        return "#c8e6c9"  # Material green 100 (same as reference)
    if abs(score - worst) < 1e-12 and abs(worst - best) > 1e-12:
        return "#ffcdd2"  # Material red 100
    return ""


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------


def _fmt_value(label: str, value: Any) -> str:
    if value is None:
        return ""
    try:
        v = float(value)
    except (TypeError, ValueError):
        return ""
    if not np.isfinite(v):
        return ""
    digits, comma, _ = _METRIC_FMT.get(label, (3, False, False))
    if label == "N":
        return f"{int(round(v)):,}"
    if comma:
        return f"{v:,.{digits}f}"
    return f"{v:.{digits}f}"


# ---------------------------------------------------------------------------
# HTML — metric table tabset
# ---------------------------------------------------------------------------


def _row_score_table(metrics_df: pd.DataFrame, *, scope: str, models: Sequence[ModelSelection]) -> List[Dict[str, Any]]:
    sub = metrics_df.loc[metrics_df["scope"] == scope].copy()
    by_cfg = {str(r["config_id"]): r for _, r in sub.iterrows()}
    table_rows: List[Dict[str, Any]] = []
    for label in METRIC_ROWS:
        key = _METRIC_KEY[label]
        direction = _METRIC_DIRECTION.get(label, "none")
        cells: List[Dict[str, Any]] = []
        scores: List[float] = []
        for m in models:
            row = by_cfg.get(m.config_id, {})
            value = row.get(key, np.nan) if row is not None else np.nan
            cells.append({"value": value, "score": _ranking_score(_safe(value), direction)})
            scores.append(_ranking_score(_safe(value), direction))
        finite_scores = [s for s in scores if np.isfinite(s)]
        best = min(finite_scores) if finite_scores else float("nan")
        worst = max(finite_scores) if finite_scores else float("nan")
        for c in cells:
            color = "" if direction == "none" else _highlight_color(c["score"], best=best, worst=worst)
            c["color"] = color
        table_rows.append({"label": label, "direction": direction, "cells": cells})
    return table_rows


def _metric_table_html(metrics_df: pd.DataFrame, *, scope: str, models: Sequence[ModelSelection]) -> str:
    rows = _row_score_table(metrics_df, scope=scope, models=models)
    headers = "<tr><th class='metric'>Metric</th>" + "".join(
        f"<th class='model'>{m.label}<br><span class='small mono'>{m.model_name}</span></th>" for m in models
    ) + "</tr>"
    body_rows: List[str] = []
    for row in rows:
        cells_html = [f"<td class='metric'>{row['label']}</td>"]
        for c in row["cells"]:
            txt = _fmt_value(row["label"], c["value"])
            color = c["color"]
            if color:
                cells_html.append(
                    f"<td class='right'><span style='background-color:{color};padding:2px 6px;border-radius:3px;display:inline-block;font-weight:600'>{txt}</span></td>"
                )
            else:
                cells_html.append(f"<td class='right'>{txt}</td>")
        body_rows.append("<tr>" + "".join(cells_html) + "</tr>")
    return (
        "<table class='metrics-table table table-striped table-bordered'>"
        f"<thead>{headers}</thead><tbody>{''.join(body_rows)}</tbody>"
        "</table>"
    )


def _build_tabset(
    *,
    tabset_id: str,
    tabs: Sequence[Tuple[str, str]],  # (label, content_html)
) -> str:
    nav_items: List[str] = []
    pane_items: List[str] = []
    for i, (label, content) in enumerate(tabs):
        active = " active" if i == 0 else ""
        selected = "true" if i == 0 else "false"
        tab_id = f"{tabset_id}-{i}"
        nav_items.append(
            f'<li class="nav-item" role="presentation">'
            f'<button class="nav-link{active}" id="{tab_id}-tab" data-bs-toggle="tab" data-bs-target="#{tab_id}" '
            f'type="button" role="tab" aria-controls="{tab_id}" aria-selected="{selected}">{label}</button>'
            f"</li>"
        )
        pane_items.append(
            f'<div class="tab-pane fade show{active}" id="{tab_id}" role="tabpanel" '
            f'aria-labelledby="{tab_id}-tab">{content}</div>'
        )
    return (
        f'<ul class="nav nav-tabs" id="{tabset_id}-nav" role="tablist">{"".join(nav_items)}</ul>'
        f'<div class="tab-content" id="{tabset_id}-content">{"".join(pane_items)}</div>'
    )


# ---------------------------------------------------------------------------
# Plotly figures
# ---------------------------------------------------------------------------


def _plotly_to_div(fig, *, include_plotlyjs: bool = False, div_id: Optional[str] = None) -> str:
    import plotly.io as pio  # local import: keeps non-report code import-light

    return pio.to_html(
        fig,
        include_plotlyjs=("cdn" if include_plotlyjs else False),
        full_html=False,
        div_id=div_id,
        config={"responsive": True, "displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )


# ---------------------------------------------------------------------------
# Geography mean % error maps (official Census geometry + PUMA detail)
# ---------------------------------------------------------------------------

def build_puma_mean_pct_error(
    df: pd.DataFrame,
    *,
    config_id: str,
) -> pd.DataFrame:
    """Mean percentage error ``100 * (y_pred/y_true - 1)`` by Census PUMA (GEOID)."""
    sub = df.loc[df["config_id"].astype(str) == str(config_id)].copy()
    if "loc_census_puma_geoid" not in sub.columns:
        return pd.DataFrame(
            columns=["puma_id", "mean_pct_error", "n_obs", "std_pct_error"]
        )
    sub = sub.dropna(subset=["loc_census_puma_geoid"])
    sub["puma_id"] = sub["loc_census_puma_geoid"].astype(str).str.replace(r"\.0$", "", regex=True)
    sub = sub.loc[sub["puma_id"].str.len() > 0].copy()
    if sub.empty:
        return pd.DataFrame(columns=["puma_id", "mean_pct_error", "n_obs", "std_pct_error"])
    sub["pct_err"] = (sub["y_pred"].astype(float) / sub["y_true"].astype(float) - 1.0) * 100.0
    g = sub.groupby("puma_id")["pct_err"].agg(["mean", "count", "std"]).reset_index()
    g.columns = ["puma_id", "mean_pct_error", "n_obs", "std_pct_error"]
    return g


def _puma_labels_from_geojson(puma_src: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for feat in puma_src.get("features") or []:
        props = feat.get("properties") or {}
        gid = str(props.get("GEOID", "")).strip()
        if not gid:
            continue
        out[gid] = str(props.get("BASENAME") or props.get("NAME") or gid)
    return out


def build_township_mean_pct_error(
    df: pd.DataFrame,
    *,
    config_id: str,
) -> pd.DataFrame:
    """Mean percentage error ``100 * (y_pred/y_true - 1)`` by assessor township."""
    sub = df.loc[df["config_id"].astype(str) == str(config_id)].copy()
    sub = sub.dropna(subset=["meta_township_name"])
    sub["meta_township_name"] = sub["meta_township_name"].astype(str)
    sub = sub.loc[~sub["meta_township_name"].isin(["Unknown", "nan"])].copy()
    sub["pct_err"] = (sub["y_pred"].astype(float) / sub["y_true"].astype(float) - 1.0) * 100.0
    g = sub.groupby("meta_township_name")["pct_err"].agg(["mean", "count", "std"]).reset_index()
    g.columns = ["meta_township_name", "mean_pct_error", "n_obs", "std_pct_error"]
    return g


def _symmetric_zmax(values: Sequence[float], *, floor: float = 4.0, ceiling: float = 22.0) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return floor
    p = float(np.nanpercentile(np.abs(arr), 96))
    return float(np.clip(max(p, floor), floor, ceiling))


def _map_colorbar_kwargs(*, title: str = "Mean % error") -> Dict[str, Any]:
    """Place colorbar right of the map with padding so it does not collide with the triad legend."""
    return dict(
        title=dict(text=title, side="right", font=dict(size=11)),
        tickformat=".0f",
        x=1.05,
        xanchor="left",
        xpad=24,
        len=0.75,
        thickness=18,
        outlinewidth=0,
        y=0.51,
        yanchor="middle",
    )


def _map_outer_layout() -> Dict[str, Any]:
    return dict(
        margin=dict(l=8, r=118, t=52, b=98),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.11,
            x=0.5,
            xanchor="center",
            font=dict(size=11),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="#cccccc",
            borderwidth=1,
            itemsizing="constant",
        ),
    )


def render_spatial_error_maps_html(
    *,
    models: Sequence[ModelSelection],
    df: pd.DataFrame,
    z_floor: float = 4.0,
    z_ceiling: float = 22.0,
) -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    Tabbed choropleths per model: official township geography, then PUMA-level detail.
    Triad boundaries are scatter line overlays; fills use a blue / green / red band scale.
    """
    import plotly.graph_objects as go

    band = _geo_maps.IAAO_MEAN_PCT_ERROR_BAND

    try:
        township_geojson = _geo_maps.load_or_fetch_cook_political_township_geojson()
        puma_src = _geo_maps.load_or_fetch_cook_puma_geojson()
    except Exception as exc:
        raise RuntimeError(
            "Could not load Cook County GIS boundaries (cache under data/geo/ or network). "
            f"Details: {exc}"
        ) from exc

    puma_labels_map = _puma_labels_from_geojson(puma_src)
    puma_feat_coll = _geo_maps.build_puma_official_geojson(puma_src)

    triad_traces = _geo_maps.triad_outline_traces(
        township_geojson=township_geojson,
        township_to_triad=_geo_maps.cook_political_township_triad_by_gis_name(),
    )

    feats_all = sorted(
        township_geojson.get("features") or [],
        key=lambda f: str((f.get("properties") or {}).get("twn", "")),
    )
    gis_township_order = [str(f.get("properties", {}).get("twn")) for f in feats_all]
    puma_feats_all = puma_feat_coll.get("features") or []
    puma_id_set = {str(f.get("properties", {}).get("puma_id")) for f in puma_feats_all}

    has_puma_col = (
        "loc_census_puma_geoid" in df.columns and bool(df["loc_census_puma_geoid"].notna().any())
    )

    all_means: List[float] = []
    tw_rows: List[Dict[str, Any]] = []
    pu_rows: List[Dict[str, Any]] = []

    for m in models:
        tdf = build_township_mean_pct_error(df, config_id=m.config_id)
        all_means.extend(tdf["mean_pct_error"].astype(float).tolist())
        for _, r in tdf.iterrows():
            tw_rows.append(
                {
                    "geo_level": "township",
                    "model_label": m.label,
                    "config_id": m.config_id,
                    "cook_county_gis_name": _geo_maps.ccao_meta_township_to_gis_name(r["meta_township_name"]),
                    "meta_township_name": r["meta_township_name"],
                    "mean_pct_error": r["mean_pct_error"],
                    "std_pct_error": r["std_pct_error"],
                    "n_obs": int(r["n_obs"]),
                }
            )
        if has_puma_col:
            pdf = build_puma_mean_pct_error(df, config_id=m.config_id)
            all_means.extend(pdf["mean_pct_error"].astype(float).tolist())

    zmax = _symmetric_zmax(all_means, floor=z_floor, ceiling=z_ceiling)
    colorscale = _geo_maps.mean_pct_error_tri_colorscale(zmax=zmax, band=band)

    for m in models:
        if not has_puma_col:
            continue
        pdf = build_puma_mean_pct_error(df, config_id=m.config_id)
        for _, r in pdf.iterrows():
            gid = str(r["puma_id"])
            pu_rows.append(
                {
                    "geo_level": "puma",
                    "model_label": m.label,
                    "config_id": m.config_id,
                    "puma_geoid": gid,
                    "puma_label": puma_labels_map.get(gid, gid),
                    "mean_pct_error": r["mean_pct_error"],
                    "std_pct_error": r["std_pct_error"],
                    "n_obs": int(r["n_obs"]),
                }
            )

    tw_export = pd.DataFrame(tw_rows)
    pu_export = pd.DataFrame(pu_rows)

    tw_tabs: List[Tuple[str, str]] = []
    gj_full: Dict[str, Any] = {"type": "FeatureCollection", "features": feats_all}

    for i, m in enumerate(models):
        tdf = build_township_mean_pct_error(df, config_id=m.config_id)
        stats_html = _geo_maps.map_summary_stats(tdf, acceptable_band=band)
        mean_by_gis: Dict[str, Any] = {}
        for _, row in tdf.iterrows():
            gk = _geo_maps.ccao_meta_township_to_gis_name(str(row["meta_township_name"]))
            mean_by_gis[gk] = row

        locs = list(gis_township_order)
        z = []
        txt: List[str] = []
        for twn in locs:
            if twn in mean_by_gis:
                row = mean_by_gis[twn]
                sig = row["std_pct_error"]
                sig_s = f"{float(sig):.2f}%" if pd.notna(sig) and np.isfinite(sig) else "n/a"
                z.append(float(row["mean_pct_error"]))
                txt.append(
                    f"{row['meta_township_name']}<br>mean % error: {row['mean_pct_error']:.2f}%<br>"
                    f"σ={sig_s}<br>n={int(row['n_obs'])}"
                )
            else:
                z.append(float("nan"))
                lbl = _geo_maps.gis_name_to_display_label(twn)
                txt.append(f"{lbl}<br>No test observations in this township")

        if tdf.empty:
            stats_html = (
                stats_html
                + "<p class='note'>No township-level test rows for this model; map shows boundaries only.</p>"
            )

        fig = go.Figure()
        fig.add_trace(
            go.Choroplethmapbox(
                geojson=gj_full,
                locations=locs,
                z=z,
                text=txt,
                featureidkey="properties.twn",
                colorscale=colorscale,
                zmin=-zmax,
                zmax=zmax,
                marker_line_width=0.75,
                marker_line_color="#333",
                colorbar=_map_colorbar_kwargs(),
                hoverinfo="text",
                showlegend=False,
            )
        )
        for tr in triad_traces:
            fig.add_trace(
                go.Scattermapbox(**{k: v for k, v in tr.items() if k != "type"})
            )
        fig.update_layout(
            title=dict(
                text=f"{m.label} — all 38 political townships (test)",
                font=dict(size=14),
            ),
            mapbox_style="white-bg",
            mapbox_zoom=8.55,
            mapbox_center={"lat": 41.90, "lon": -87.75},
            height=520,
            font=dict(size=12),
            **_map_outer_layout(),
        )
        tw_tabs.append((m.label, stats_html + _plotly_to_div(fig, div_id=f"township-map-{i}")))

    expl_township = (
        "<p class='note'><strong>Township boundaries</strong> are the 38 <strong>Cook County GIS Political "
        "Township</strong> polygons (MapServer layer 43, same basis as "
        "<a href='https://maps.cookcountyil.gov/cookviewer/'>CookViewer</a> and published assessor maps). "
        "Every township is drawn; <strong>mean % error</strong> is colored only where the held-out test split "
        "has sales in that township. Areas with no test observations remain uncolored. "
        "Lake Michigan is excluded from official land boundaries. "
        f"Shared diverging scale (±{zmax:.1f}% cap): "
        f"<span style='color:rgb(30,64,175)'>blue</span> under, "
        f"<span style='color:rgb(22,163,74)'>green</span> within ±{band:.0f}%, "
        "<span style='color:rgb(185,28,28)'>red</span> over. "
        "<strong>City</strong> triad outline is on top of North/South (heavier line).</p>"
    )

    tw_section = (
        "<h2 id='township-maps'>Township mean % error — Cook County political boundaries (all 38)</h2>\n"
        + expl_township
        + _build_tabset(tabset_id="township-error-maps-tab", tabs=tw_tabs)
    )

    if not has_puma_col:
        pu_section = (
            "<h2 id='puma-maps'>PUMA mean % error — finer geography</h2>\n"
            "<p class='note'>PUMA-level maps need <code>loc_census_puma_geoid</code> from the training-data join.</p>"
        )
        return tw_section + "\n" + pu_section, tw_export, pd.DataFrame()

    expl_puma = (
        "<p class='note'><strong>PUMA maps.</strong> One polygon per 2020 <strong>Public Use Microdata Area</strong> "
        "(Cook-nesting PUMAs, ~36 regions). Same color scale and triad overlays as the township maps above. "
        "No minimum-sample filter: every PUMA with at least one test sale is shown.</p>"
    )

    pu_tabs: List[Tuple[str, str]] = []
    for i, m in enumerate(models):
        pdf = build_puma_mean_pct_error(df, config_id=m.config_id)
        pdf = pdf.copy()
        pdf["puma_label"] = pdf["puma_id"].map(lambda x: puma_labels_map.get(str(x), str(x)))
        stats_html = _geo_maps.map_summary_stats(
            pdf,
            acceptable_band=band,
            label_column="puma_label",
            region_word="PUMAs",
        )
        ok = pdf["puma_id"].astype(str).isin(puma_id_set)
        pdf = pdf.loc[ok].copy()
        if pdf.empty:
            pu_tabs.append(
                (
                    m.label,
                    stats_html
                    + "<p class='note'>No test sales fell in a PUMA with a known boundary for this model.</p>",
                )
            )
            continue

        locs = pdf["puma_id"].astype(str).tolist()
        z = pdf["mean_pct_error"].astype(float).tolist()
        txt = []
        for _, row in pdf.iterrows():
            sig = row["std_pct_error"]
            sig_s = f"{float(sig):.2f}%" if pd.notna(sig) and np.isfinite(sig) else "n/a"
            txt.append(
                f"{row['puma_label']} ({row['puma_id']})<br>mean % error: {row['mean_pct_error']:.2f}%<br>"
                f"σ={sig_s}<br>n={int(row['n_obs'])}"
            )
        sub_features = [f for f in puma_feats_all if str(f.get("properties", {}).get("puma_id")) in set(locs)]
        gj_pu: Dict[str, Any] = {"type": "FeatureCollection", "features": sub_features}

        figp = go.Figure()
        figp.add_trace(
            go.Choroplethmapbox(
                geojson=gj_pu,
                locations=locs,
                z=z,
                text=txt,
                featureidkey="properties.puma_id",
                colorscale=colorscale,
                zmin=-zmax,
                zmax=zmax,
                marker_line_width=0.65,
                marker_line_color="#333",
                colorbar=_map_colorbar_kwargs(),
                hoverinfo="text",
                showlegend=False,
            )
        )
        for tr in triad_traces:
            figp.add_trace(
                go.Scattermapbox(**{k: v for k, v in tr.items() if k != "type"})
            )
        figp.update_layout(
            title=dict(text=f"{m.label} — PUMA (test)", font=dict(size=14)),
            mapbox_style="white-bg",
            mapbox_zoom=8.85,
            mapbox_center={"lat": 41.90, "lon": -87.75},
            height=520,
            font=dict(size=12),
            **_map_outer_layout(),
        )
        pu_tabs.append((m.label, stats_html + _plotly_to_div(figp, div_id=f"puma-map-{i}")))

    pu_section = (
        "<h2 id='puma-maps'>PUMA mean % error — finer geography</h2>\n"
        + expl_puma
        + _build_tabset(tabset_id="puma-error-maps-tab", tabs=pu_tabs)
    )

    combined = tw_section + "\n" + pu_section
    return combined, tw_export, pu_export


def _per_metric_bar(metrics_df: pd.DataFrame, *, label: str, models: Sequence[ModelSelection]) -> Any:
    import plotly.graph_objects as go

    key = _METRIC_KEY[label]
    fig = go.Figure()
    for i, m in enumerate(models):
        ys = []
        texts = []
        for scope in SCOPE_ORDER:
            row = metrics_df.loc[
                (metrics_df["scope"] == scope) & (metrics_df["config_id"] == m.config_id)
            ]
            v = float(row.iloc[0][key]) if not row.empty and key in row.columns else np.nan
            ys.append(v if np.isfinite(v) else None)
            texts.append(_fmt_value(label, v))
        fig.add_trace(
            go.Bar(
                name=m.label,
                x=list(SCOPE_ORDER),
                y=ys,
                text=texts,
                textposition="outside",
                marker_color=_MODEL_PALETTE[i % len(_MODEL_PALETTE)],
                hovertemplate=f"<b>{m.label}</b><br>scope=%{{x}}<br>{label}=%{{text}}<extra></extra>",
            )
        )
    if label == "PRD":
        fig.add_hrect(y0=IAAO_PRD_RANGE[0], y1=IAAO_PRD_RANGE[1], fillcolor="lightgreen", opacity=0.18, line_width=0)
    if label == "PRB":
        fig.add_hrect(y0=IAAO_PRB_RANGE[0], y1=IAAO_PRB_RANGE[1], fillcolor="lightgreen", opacity=0.18, line_width=0)
    if label == "VEI":
        fig.add_hrect(y0=IAAO_VEI_RANGE[0], y1=IAAO_VEI_RANGE[1], fillcolor="lightgreen", opacity=0.18, line_width=0)
    if label == "Med Ratio":
        fig.add_hrect(y0=IAAO_LEVEL_RANGE[0], y1=IAAO_LEVEL_RANGE[1], fillcolor="lightgreen", opacity=0.18, line_width=0)
        fig.add_hline(y=1.0, line_dash="dot", line_color="gray")
    if label == "PRD" or label == "MKI":
        fig.add_hline(y=1.0, line_dash="dot", line_color="gray")
    if label in ("PRB", "VEI"):
        fig.add_hline(y=0.0, line_dash="dot", line_color="gray")
    fig.update_layout(
        barmode="group",
        title=label,
        xaxis_title="Scope",
        yaxis_title=label,
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=50, b=80),
        height=380,
        template="simple_white",
    )
    return fig


def _decile_median_fig(
    decile_df: pd.DataFrame,
    *,
    scope: str,
    models: Sequence[ModelSelection],
) -> Any:
    import plotly.graph_objects as go

    sub = decile_df.loc[decile_df["scope"] == scope]
    fig = go.Figure()
    fig.add_hrect(
        y0=IAAO_TARGET_BAND[0], y1=IAAO_TARGET_BAND[1],
        fillcolor="lightgreen", opacity=0.25, line_width=0,
        annotation_text=f"IAAO target [{IAAO_TARGET_BAND[0]}, {IAAO_TARGET_BAND[1]}]",
        annotation_position="bottom right",
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray")
    for i, m in enumerate(models):
        msub = sub.loc[sub["config_id"] == m.config_id].sort_values("decile")
        if msub.empty:
            continue
        fig.add_trace(
            go.Scatter(
                x=msub["decile"],
                y=msub["median_ratio"],
                mode="lines+markers",
                name=m.label,
                line=dict(width=2.2, color=_MODEL_PALETTE[i % len(_MODEL_PALETTE)]),
                marker=dict(size=7),
                hovertemplate=(
                    f"<b>{m.label}</b><br>"
                    "decile=%{x}<br>"
                    "median ratio=%{y:.3f}<br>"
                    "n=%{customdata[0]:,}<br>"
                    "price=$%{customdata[1]:,.0f}–$%{customdata[2]:,.0f}<extra></extra>"
                ),
                customdata=msub[["n_obs", "lower_price", "upper_price"]].to_numpy(),
            )
        )
    fig.update_layout(
        title=f"Decile Median Ratio — {scope}",
        xaxis_title="Sale price decile (1 = cheapest)",
        yaxis_title="Median ratio (y_pred / y_true)",
        xaxis=dict(dtick=1),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=60, b=80),
        height=420,
        template="simple_white",
    )
    return fig


def _decile_quartile_fig(
    decile_df: pd.DataFrame,
    *,
    scope: str,
    models: Sequence[ModelSelection],
) -> Any:
    import plotly.graph_objects as go

    sub = decile_df.loc[decile_df["scope"] == scope]
    fig = go.Figure()
    fig.add_hrect(
        y0=IAAO_TARGET_BAND[0], y1=IAAO_TARGET_BAND[1],
        fillcolor="lightgreen", opacity=0.25, line_width=0,
        annotation_text=f"IAAO target [{IAAO_TARGET_BAND[0]}, {IAAO_TARGET_BAND[1]}]",
        annotation_position="bottom right",
    )
    fig.add_hline(y=1.0, line_dash="dot", line_color="gray")
    for i, m in enumerate(models):
        msub = sub.loc[sub["config_id"] == m.config_id].sort_values("decile")
        if msub.empty:
            continue
        color = _MODEL_PALETTE[i % len(_MODEL_PALETTE)]
        rgba = _hex_to_rgba(color, 0.18)
        # IQR ribbon: q75 then reverse q25
        fig.add_trace(
            go.Scatter(
                x=list(msub["decile"]) + list(msub["decile"][::-1]),
                y=list(msub["q75"]) + list(msub["q25"][::-1]),
                fill="toself",
                fillcolor=rgba,
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
                name=f"{m.label} IQR",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=msub["decile"],
                y=msub["median_ratio"],
                mode="lines+markers",
                name=m.label,
                line=dict(width=2.2, color=color),
                marker=dict(size=7),
                hovertemplate=(
                    f"<b>{m.label}</b><br>"
                    "decile=%{x}<br>"
                    "median=%{y:.3f}<br>"
                    "Q25=%{customdata[0]:.3f} · Q75=%{customdata[1]:.3f}<br>"
                    "n=%{customdata[2]:,}<extra></extra>"
                ),
                customdata=msub[["q25", "q75", "n_obs"]].to_numpy(),
            )
        )
    fig.update_layout(
        title=f"Decile Median Ratio ± IQR — {scope}",
        xaxis_title="Sale price decile (1 = cheapest)",
        yaxis_title="Sales ratio",
        xaxis=dict(dtick=1),
        legend=dict(orientation="h", yanchor="top", y=-0.18, xanchor="center", x=0.5),
        margin=dict(l=40, r=20, t=60, b=80),
        height=440,
        template="simple_white",
    )
    return fig


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    if len(h) != 6:
        return f"rgba(120, 120, 120, {alpha})"
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r}, {g}, {b}, {alpha})"


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------


_CSS = """
:root { color-scheme: light; }
body { font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif; color: #1d1d1d; max-width: 1240px; margin: 28px auto; padding: 0 24px; line-height: 1.5; }
h1 { font-size: 26px; margin-bottom: 4px; }
h2 { font-size: 21px; margin-top: 38px; border-bottom: 1px solid #e3e3e3; padding-bottom: 6px; }
h3 { font-size: 17px; margin-top: 22px; }
.small { font-size: 12px; }
.mono { font-family: 'Menlo', 'Consolas', monospace; }
.note { color: #444; margin: 8px 0 16px; }
.kvs td:first-child { color: #666; padding-right: 8px; }
.summary { background: #f7f7fa; border: 1px solid #e3e3e3; padding: 12px 16px; border-radius: 6px; margin: 16px 0; }
table.metrics-table { width: 100%; font-size: 13px; }
table.metrics-table th, table.metrics-table td { padding: 6px 10px; vertical-align: middle; }
table.metrics-table th.metric, table.metrics-table td.metric { width: 18%; font-weight: 600; }
table.metrics-table th.model { font-weight: 600; text-align: left; }
table.metrics-table td.right { text-align: right; font-variant-numeric: tabular-nums; }
.nav-tabs { margin-top: 12px; }
.nav-tabs .nav-link { color: #1d1d1d; }
.nav-tabs .nav-link.active { font-weight: 600; }
.tab-content { padding: 16px 0; }
"""

try:
    from plotly.offline import get_plotlyjs_version

    _PLOTLYJS_VERSION = get_plotlyjs_version()
except Exception:  # pragma: no cover — offline import always works in normal envs
    _PLOTLYJS_VERSION = "3.3.1"

_PLOTLYJS_CDN = f"https://cdn.plot.ly/plotly-{_PLOTLYJS_VERSION}.min.js"


def render_html_report(
    *,
    title: str,
    subtitle: str,
    metadata_kv: Dict[str, str],
    models: Sequence[ModelSelection],
    metrics_df: pd.DataFrame,
    decile_df: pd.DataFrame,
    predictions_geography_df: Optional[pd.DataFrame] = None,
) -> Tuple[str, Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    # Overview
    rows = "".join(
        f"<tr><td class='mono small'>{m.config_id[:12]}</td>"
        f"<td><strong>{m.label}</strong><br><span class='small mono'>rule={m.rule}</span></td>"
        f"<td>{m.description}</td></tr>"
        for m in models
    )
    overview_html = (
        "<table class='table table-striped table-bordered'>"
        "<thead><tr><th>Run ID (config_id)</th><th>Spec Label</th><th>Description</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )

    # Metric Table tabset
    metric_tabs: List[Tuple[str, str]] = []
    for scope in SCOPE_ORDER:
        metric_tabs.append((scope, _metric_table_html(metrics_df, scope=scope, models=models)))
    metric_table_html = _build_tabset(tabset_id="metric-tab", tabs=metric_tabs)

    # Per-Metric panels (one figure per metric except N)
    panel_blocks: List[str] = []
    for label in METRIC_ROWS:
        if label == "N":
            continue
        fig = _per_metric_bar(metrics_df, label=label, models=models)
        panel_blocks.append(
            f'<h3 id="{_anchor(label)}">{label}</h3>'
            + _plotly_to_div(fig, div_id=f"per-metric-{_anchor(label)}")
        )

    # Decile median ratio — Overall + per-triad tabset
    decile_overall_fig = _decile_median_fig(decile_df, scope="Overall", models=models)
    decile_per_triad_tabs: List[Tuple[str, str]] = []
    for scope in TRIAD_ONLY_ORDER:
        fig = _decile_median_fig(decile_df, scope=scope, models=models)
        decile_per_triad_tabs.append(
            (scope, _plotly_to_div(fig, div_id=f"decile-median-{scope.lower()}"))
        )
    decile_section = (
        "<h3>Overall</h3>"
        + _plotly_to_div(decile_overall_fig, div_id="decile-median-overall")
        + "<h3>Per-Triad Individual Plots</h3>"
        + _build_tabset(tabset_id="decile-median-tab", tabs=decile_per_triad_tabs)
    )

    # Decile median ± IQR — Overall + per-triad tabset
    iqr_overall_fig = _decile_quartile_fig(decile_df, scope="Overall", models=models)
    iqr_per_triad_tabs: List[Tuple[str, str]] = []
    for scope in TRIAD_ONLY_ORDER:
        fig = _decile_quartile_fig(decile_df, scope=scope, models=models)
        iqr_per_triad_tabs.append(
            (scope, _plotly_to_div(fig, div_id=f"decile-iqr-{scope.lower()}"))
        )
    iqr_section = (
        "<h3>Overall (Interactive)</h3>"
        + _plotly_to_div(iqr_overall_fig, div_id="decile-iqr-overall")
        + "<h3>Per-Triad Individual Plots</h3>"
        + _build_tabset(tabset_id="decile-iqr-tab", tabs=iqr_per_triad_tabs)
    )

    township_export: Optional[pd.DataFrame] = None
    puma_export: Optional[pd.DataFrame] = None
    maps_section = ""
    if predictions_geography_df is not None and not predictions_geography_df.empty:
        frag, township_export, puma_export = render_spatial_error_maps_html(
            models=models,
            df=predictions_geography_df,
        )
        maps_section = frag

    # Metadata block
    meta_rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in metadata_kv.items())

    body = textwrap.dedent(
        f"""
        <!doctype html>
        <html lang='en'>
          <head>
            <meta charset='utf-8'>
            <meta name='viewport' content='width=device-width, initial-scale=1'>
            <title>{title}</title>
            <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/css/bootstrap.min.css">
            <style>{_CSS}</style>
            <script src="{_PLOTLYJS_CDN}" charset="utf-8"></script>
          </head>
          <body>
            <h1>{title}</h1>
            <p class='note'>{subtitle}</p>

            <div class='summary'>
              <table class='kvs small'>{meta_rows}</table>
            </div>

            <h2 id='overview'>Overview</h2>
            <p>This report compares three residential AVM model configurations on the held-out test split.
            <code>meta_sale_price</code> is the truth column; <code>y_pred</code> is the test prediction.
            All metrics and decile ratio curves are computed both overall (pooled across all Cook County
            triads) and separately by <code>meta_triad_name</code>.</p>
            {overview_html}

            <h2 id='metric-table'>Metric Table</h2>
            <p>The best value among the three models is highlighted in
              <span style='background:#c8e6c9;padding:2px 6px;border-radius:3px'>green</span>;
              the worst in
              <span style='background:#ffcdd2;padding:2px 6px;border-radius:3px'>red</span>.
              Direction conventions: R² → higher better; RMSE / MAE / MAPE / MdAPE / COD → lower better;
              Median Ratio / PRD / MKI → closer to 1 better; PRB / VEI → closer to 0 better. <em>N</em> is the
              row count (no highlight applied).</p>
            {metric_table_html}

            <h2 id='per-metric-comparisons'>Per-Metric Comparisons</h2>
            <p>Each panel compares the three models on a single metric — overall and per triad.</p>
            {''.join(panel_blocks)}

            <h2 id='decile-median'>Median Ratio Decile Curves</h2>
            <p>Deciles are computed within each (model × scope) by ranking <code>meta_sale_price</code>
              into 10 equal-count bins. The shaded green band is the IAAO target ratio band
              [{IAAO_TARGET_BAND[0]}, {IAAO_TARGET_BAND[1]}]; the dotted gray line is parity (1.0).</p>
            {decile_section}

            <h2 id='decile-iqr'>Decile Ratio Curves with Quartiles</h2>
            <p>Same decile binning, with the median surrounded by an IQR (25th–75th percentile)
              ribbon per model.</p>
            {iqr_section}

            {maps_section}

            <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.3/dist/js/bootstrap.bundle.min.js"></script>
            <script>
            (function () {{
              function resizePlotlyGraphs() {{
                if (!window.Plotly) return;
                document.querySelectorAll('.plotly-graph-div').forEach(function (el) {{
                  try {{
                    Plotly.Plots.resize(el);
                  }} catch (e) {{}}
                }});
              }}
              document.addEventListener('DOMContentLoaded', function () {{
                document.querySelectorAll('[data-bs-toggle="tab"]').forEach(function (tab) {{
                  tab.addEventListener('shown.bs.tab', resizePlotlyGraphs);
                }});
                window.addEventListener('resize', resizePlotlyGraphs);
              }});
            }})();
            </script>
          </body>
        </html>
        """
    ).strip()
    return body, township_export, puma_export


def _anchor(label: str) -> str:
    out = label.lower().replace("²", "2").replace(" ", "-").replace("%", "pct").replace(".", "")
    return out.strip("-") or "metric"
