#!/usr/bin/env python3
"""v3 follow-up paper-asset package from frozen v2 lower-rho results.

Deterministic post-processing only. No model fitting, no Slurm, no TeX writes,
no TeX compilation, and no writes under paper/.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.transition_paper_asset_plots import combined_row, metric_val
from utils.transition_paper_assets import (
    POSITIVE_ANCHOR_TARGETS,
    manuscript_format_flags,
    nearest_grid_rho,
    positive_display_anchors,
    span_regret_row,
)
from utils.transition_regions import (
    FAMILY_DISPLAY,
    FOLD_IDS,
    OutputGuard,
    PRIMARY_METRICS,
    extract_discrete_event,
    family_frame,
    is_rho_positive,
    is_rho_zero,
    numerically_equal,
    rho_in_closed_span,
    sha256_file,
)

CANONICAL = REPO / "output" / "paper_v6_preselection_994"
EXT = REPO / "output" / "paper_v12_lower_rho_extension_994_v2"
DATA_ID = "d4929d43ec19badf"
SPLIT_ID = "3d464d4a611b131b"
V2 = (
    EXT
    / "analysis"
    / f"data_id={DATA_ID}"
    / f"split_id={SPLIT_ID}"
    / "penalty_path_analysis"
    / "transition_regions_v2_lower_rho"
)
ASSETS_V2 = V2.parent / "transition_regions_paper_assets_v2"
V3 = V2.parent / "transition_regions_paper_assets_v3_followup"
PAPER_TEX = REPO / "paper" / "paper_v12.tex"
PAPER_IMG = REPO / "paper" / "img" / "generated_v12_994"

PRIMARY_METRICS_TABLE = [
    ("R2_price", True, None, False, "higher"),
    ("MAE_price", False, None, False, "lower"),
    ("MAPE", False, None, False, "lower"),
    ("RMSE_log", False, None, False, "lower"),
    ("PRD", None, 1.0, True, "closer_to_1"),
    ("PRB", None, 0.0, True, "closer_to_0"),
    ("MKI", None, 1.0, True, "closer_to_1"),
    ("VEI", None, 0.0, True, "closer_to_0"),
]
SECONDARY_METRICS_TABLE = [
    ("median_ratio", None, 1.0, False, "closer_to_1"),
    ("mean_ratio", None, 1.0, False, "closer_to_1"),
    ("weighted_mean_ratio", None, 1.0, False, "closer_to_1"),
    ("COD", False, None, False, "lower"),
    ("COV", False, None, False, "lower"),
    ("Beta_log", None, 0.0, False, "closer_to_0"),
    ("Delta_NL", False, None, False, "lower"),
    ("dCor_e_y", False, None, False, "lower"),
]
VE_EVENT_SPECS = [
    ("PRD", "abs_target", 1.0, "descriptive_neutrality"),
    ("PRB", "abs_target", 0.0, "descriptive_neutrality"),
    ("MKI", "abs_target", 1.0, "descriptive_neutrality"),
    ("VEI", "abs_target", 0.0, "descriptive_neutrality"),
]
LEVEL_EVENT_SPECS = [
    ("median_ratio", "abs_target", 1.0, "descriptive_neutrality"),
    ("mean_ratio", "abs_target", 1.0, "descriptive_neutrality"),
    ("weighted_mean_ratio", "abs_target", 1.0, "descriptive_neutrality"),
    ("COD", "min", None, "descriptive_minimum"),
    ("COV", "min", None, "descriptive_minimum"),
]
MECH_EVENT_SPECS = [
    ("Beta_log", "abs_target", 0.0, "descriptive_neutrality"),
    ("dCor_e_y", "min", None, "descriptive_minimum"),
]
FORBIDDEN_PHRASES = (
    "sweet spot",
    "safe range",
    "recommended range",
    "selected range",
    "preferred range",
    "optimal range",
    "deployment-ready",
)
PAPER_BASELINE_DISPLAY = {
    ("heldout", "Linear", "R2_price"): "0.799",
    ("heldout", "LightGBM", "R2_price"): "0.894",
    ("heldout", "Linear", "MAE_price"): "$90,092",
    ("heldout", "LightGBM", "MAE_price"): "$75,655",
    ("heldout", "Linear", "MAPE"): "24.1%",
    ("heldout", "LightGBM", "MAPE"): "21.2%",
    ("heldout", "Linear", "RMSE_log"): "0.322",
    ("heldout", "LightGBM", "RMSE_log"): "0.289",
    ("forward_2025", "Linear", "R2_price"): "0.799",
    ("forward_2025", "LightGBM", "R2_price"): "0.904",
    ("forward_2025", "Linear", "MAE_price"): "$99,371",
    ("forward_2025", "LightGBM", "MAE_price"): "$78,484",
    ("forward_2025", "Linear", "MAPE"): "24.9%",
    ("forward_2025", "LightGBM", "MAPE"): "20.8%",
    ("forward_2025", "Linear", "RMSE_log"): "0.313",
    ("forward_2025", "LightGBM", "RMSE_log"): "0.278",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(REPO), text=True).strip()


def hash_tree(root: Path, *, exclude_parts: Sequence[str] = ()) -> Dict[str, str]:
    out: Dict[str, str] = {}
    if not root.exists():
        return out
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        rel = str(path.relative_to(REPO))
        if any(part in path.parts for part in exclude_parts):
            continue
        out[rel] = sha256_file(path)
    return out


def display_value(metric: str, val: float) -> str:
    if val is None or not np.isfinite(float(val)):
        return "NA"
    x = float(val)
    if metric == "R2_price":
        return f"{x:.3f}"
    if metric == "MAE_price":
        return f"${x:,.0f}"
    if metric == "MAPE":
        return f"{100.0 * x:.1f}%"
    if metric == "RMSE_log":
        return f"{x:.3f}"
    if metric in {"median_ratio", "mean_ratio", "weighted_mean_ratio", "PRD", "PRB", "MKI", "Beta_log", "Delta_NL", "dCor_e_y"}:
        return f"{x:.3f}"
    if metric == "COD":
        return f"{x:.1f}%"
    if metric == "COV":
        return f"{100.0 * x:.1f}%"
    if metric == "VEI":
        return f"{x:.1f}%"
    return f"{x:.6g}"


def write_table_bundle(guard: OutputGuard, df: pd.DataFrame, stem: Path) -> List[str]:
    written = [str(p) for p in guard.write_df(df, stem.with_suffix(".csv"), parquet=False)]
    pq = guard.allowed(stem.with_suffix(".parquet"))
    try:
        df.to_parquet(pq, index=False)
    except Exception:
        alt = df.copy()
        for col in alt.columns:
            if alt[col].dtype == object:
                alt[col] = alt[col].map(lambda x: None if x is None or (isinstance(x, float) and pd.isna(x)) else x)
                try:
                    alt[col] = alt[col].astype("string")
                except Exception:
                    alt[col] = alt[col].astype(str)
        alt.to_parquet(pq, index=False)
    written.append(str(pq))
    payload = json.loads(df.to_json(orient="records", double_precision=15))
    json_path = guard.write_json(stem.with_suffix(".json"), payload)
    written.append(str(json_path))
    try:
        md = df.to_markdown(index=False)
    except Exception:
        md = df.to_string(index=False)
    md_path = guard.write_text(stem.with_suffix(".md"), md + "\n")
    written.append(str(md_path))
    return written


def load_frozen() -> Dict[str, Any]:
    combined = pd.read_csv(V2 / "tables" / "combined_path_table_v2.csv")
    span = pd.read_csv(V2 / "tables" / "transition_span_summary.csv")
    grid = json.loads((EXT / "protocol" / "lower_rho_grid_v2.json").read_text(encoding="utf-8"))
    return {
        "combined": combined,
        "span": span,
        "grid": grid,
        "events_cv": pd.read_csv(V2 / "tables" / "transition_events_cv_mean.csv"),
        "events_fold": pd.read_csv(V2 / "tables" / "transition_events_by_fold.csv"),
        "lofo": pd.read_csv(V2 / "tables" / "transition_lofo_sensitivity.csv"),
        "concordance": pd.read_csv(V2 / "tables" / "transition_temporal_concordance.csv"),
        "sharpness": pd.read_csv(V2 / "tables" / "transition_event_sharpness.csv"),
    }


def family_span_endpoints(span: pd.DataFrame, family: str) -> Tuple[float, float]:
    row = span.loc[span["family"] == family].iloc[0]
    if str(row["status"]) != "VALID_POSITIVE_INTERIOR_SPAN":
        raise RuntimeError(f"frozen v2 span is not VALID for {family}")
    return float(row["rho_transition_low"]), float(row["rho_transition_high"])


def rho_on_grid(rho: float, grid: Sequence[float]) -> bool:
    return any(numerically_equal(float(rho), float(g)) for g in grid)


def build_long_table(
    combined: pd.DataFrame,
    specs: Sequence[Tuple[str, Optional[bool], Optional[float], bool, str]],
    *,
    include_penalized: bool,
) -> pd.DataFrame:
    grid = combined.loc[combined["family"] == "Direct", "rho"].to_numpy(dtype=float)
    anchors = positive_display_anchors(grid)
    rows: List[Dict[str, Any]] = []
    families: List[Tuple[str, Optional[float], str]] = [("Linear", None, "--"), ("LightGBM", None, "--")]
    if include_penalized:
        for a, nom in zip(anchors, POSITIVE_ANCHOR_TARGETS):
            families.append(("Direct", float(a), f"{float(nom):g}"))
            families.append(("Surrogate", float(a), f"{float(nom):g}"))
    for split in ("heldout", "forward_2025"):
        lin = combined_row(combined, "Linear")
        lgb = combined_row(combined, "LightGBM")
        for fam, rho, nominal in families:
            rec_row = combined_row(combined, fam, rho)
            exact_rho = None if rho is None else float(rec_row["rho"])
            for name, higher, target, can_star, direction in specs:
                val = metric_val(rec_row, name, split)
                flags = manuscript_format_flags(
                    val,
                    metric=name,
                    family=fam,
                    linear_val=metric_val(lin, name, split),
                    lgbm_val=metric_val(lgb, name, split),
                    higher=higher,
                    target=target,
                    can_star=can_star,
                )
                rows.append(
                    {
                        "split": split,
                        "family": fam,
                        "exact_tested_rho": exact_rho,
                        "nominal_display_anchor": nominal,
                        "metric": name,
                        "value_unrounded": float(val),
                        "value_display": display_value(name, val),
                        "preferred_direction_or_target": direction,
                        "target_value": target,
                        "in_reference_range": flags["within_reference_range"],
                        "beats_both_baselines": flags["beats_both_baselines"],
                        "beats_ordinary_only": flags["beats_ordinary_only"],
                        "manuscript_bold": flags["manuscript_bold"],
                        "manuscript_asterisk": flags["manuscript_asterisk"],
                        "transition_endpoint_not_added": True,
                        "nearest_anchor_logic": "positive_display_anchors",
                    }
                )
    df = pd.DataFrame(rows)
    df["in_reference_range"] = pd.array(
        [None if v is None else bool(v) for v in df["in_reference_range"]],
        dtype="boolean",
    )
    for col in (
        "manuscript_bold",
        "manuscript_asterisk",
        "beats_both_baselines",
        "beats_ordinary_only",
        "transition_endpoint_not_added",
    ):
        df[col] = df[col].astype(bool)
    return df


def extract_one_event(rhos: np.ndarray, values: np.ndarray, metric: str, kind: str, target: Optional[float]):
    if kind == "abs_target":
        trans = np.abs(np.asarray(values, dtype=float) - float(target))
        ev = extract_discrete_event(rhos, trans, metric=metric, direction="min")
    elif kind == "min":
        ev = extract_discrete_event(rhos, values, metric=metric, direction="min")
    else:
        raise ValueError(kind)
    event_rho = ev.rho_low
    original = None
    if event_rho is not None:
        idx = int(np.argmin(np.abs(np.asarray(rhos, dtype=float) - float(event_rho))))
        original = float(values[idx])
    abs_gap = None
    if kind == "abs_target" and original is not None:
        abs_gap = float(abs(original - float(target)))
    return ev, original, abs_gap


def build_event_table(
    combined: pd.DataFrame,
    span: pd.DataFrame,
    specs: Sequence[Tuple[str, str, Optional[float], str]],
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    splits = [("cv_mean", "CV_mean")] + [(f"fold_{k}", f"fold_{k}") for k in FOLD_IDS] + [
        ("heldout", "heldout"),
        ("forward_2025", "forward_2025"),
    ]
    for fam in FAMILY_DISPLAY:
        sub = family_frame(combined, fam)
        rhos = sub["rho"].to_numpy(dtype=float)
        low, high = family_span_endpoints(span, fam)
        for metric, kind, target, event_kind in specs:
            for split, suffix in splits:
                col = f"{metric}__{suffix}" if suffix != "CV_mean" else f"{metric}__CV_mean"
                if split.startswith("fold_"):
                    col = f"{metric}__{split}"
                vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
                ev, original, abs_gap = extract_one_event(rhos, vals, metric, kind, target)
                event_rho = ev.rho_low
                on_grid = False if event_rho is None else rho_on_grid(float(event_rho), rhos)
                tied_ok = all(rho_on_grid(float(x), rhos) for x in ev.tied_rhos)
                inside = None if event_rho is None else rho_in_closed_span(float(event_rho), low, high)
                rows.append(
                    {
                        "family": fam,
                        "split": split,
                        "metric": metric,
                        "event_kind": event_kind,
                        "event_rule": kind if kind != "abs_target" else f"argmin_|{metric}-{target}|",
                        "target_value": target,
                        "event_rho": event_rho,
                        "event_rho_high": ev.rho_high,
                        "classification": ev.classification,
                        "n_tied": ev.n_tied,
                        "tied_rhos_json": json.dumps(ev.tied_rhos),
                        "metric_value": original,
                        "abs_gap_to_target": abs_gap,
                        "local_turn_verified": ev.local_turn_verified,
                        "on_tested_grid": bool(on_grid and tied_ok),
                        "inside_frozen_five_metric_cv_span": inside,
                        "frozen_span_low": low,
                        "frozen_span_high": high,
                        "does_not_redefine_transition_span": True,
                        "no_smoothing": True,
                        "no_interpolation": True,
                        "notes": ev.notes,
                    }
                )
    df = pd.DataFrame(rows)
    df["inside_frozen_five_metric_cv_span"] = pd.array(
        [None if v is None else bool(v) for v in df["inside_frozen_five_metric_cv_span"]],
        dtype="boolean",
    )
    for col in ("on_tested_grid", "local_turn_verified", "does_not_redefine_transition_span", "no_smoothing", "no_interpolation"):
        df[col] = df[col].astype(bool)
    return df


def build_delta_nl_oos(combined: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam in FAMILY_DISPLAY:
        sub = family_frame(combined, fam)
        rhos = sub["rho"].to_numpy(dtype=float)
        for split in ("heldout", "forward_2025"):
            vals = pd.to_numeric(sub[f"Delta_NL__{split}"], errors="coerce").to_numpy(dtype=float)
            ev, original, _gap = extract_one_event(rhos, vals, "Delta_NL", "min", None)
            rows.append(
                {
                    "family": fam,
                    "split": split,
                    "metric": "Delta_NL",
                    "event_kind": "descriptive_minimum",
                    "event_rho": ev.rho_low,
                    "classification": ev.classification,
                    "n_tied": ev.n_tied,
                    "metric_value": original,
                    "on_tested_grid": False if ev.rho_low is None else rho_on_grid(float(ev.rho_low), rhos),
                    "cv_unavailable_by_design": True,
                    "not_mixed_into_fold_figure": True,
                }
            )
    return pd.DataFrame(rows)


def build_regret(combined: pd.DataFrame, span: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for fam in FAMILY_DISPLAY:
        sub = family_frame(combined, fam)
        rhos = sub["rho"].to_numpy(dtype=float)
        low, high = family_span_endpoints(span, fam)
        for split in ("heldout", "forward_2025"):
            for metric, direction in PRIMARY_METRICS:
                vals = pd.to_numeric(sub[f"{metric}__{split}"], errors="coerce").to_numpy(dtype=float)
                rec = span_regret_row(
                    rhos,
                    vals,
                    family=fam,
                    split=split,
                    metric=metric,
                    direction=direction,
                    rho_low=low,
                    rho_high=high,
                )
                rec["full_path_range"] = rec["path_range"]
                rec["frozen_span_source"] = "v2_cv_derived_descriptive_transition_span"
                rec["paper_candidate_not_automatically_required"] = True
                rows.append(rec)
    return pd.DataFrame(rows)


def build_lofo_summary(lofo: pd.DataFrame, span: pd.DataFrame, grid: Dict[str, Any]) -> pd.DataFrame:
    min_pos = float(grid["min_positive_augmented"])
    max_pos = 100.0
    rows = []
    for fam in FAMILY_DISPLAY:
        part = lofo.loc[lofo["family"] == fam].copy()
        valid = part.loc[part["valid_positive_interior_five_event_span"].astype(bool)]
        lows = pd.to_numeric(valid["rho_transition_low"], errors="coerce")
        highs = pd.to_numeric(valid["rho_transition_high"], errors="coerce")
        widths = pd.to_numeric(valid["log10_width"], errors="coerce")
        frozen_low, frozen_high = family_span_endpoints(span, fam)
        rows.append(
            {
                "family": fam,
                "n_valid": int(len(valid)),
                "n_of": 7,
                "minimum_lower_endpoint": float(lows.min()) if len(valid) else None,
                "maximum_lower_endpoint": float(lows.max()) if len(valid) else None,
                "minimum_upper_endpoint": float(highs.min()) if len(valid) else None,
                "maximum_upper_endpoint": float(highs.max()) if len(valid) else None,
                "min_log10_span_width": float(widths.min()) if len(valid) else None,
                "median_log10_span_width": float(widths.median()) if len(valid) else None,
                "max_log10_span_width": float(widths.max()) if len(valid) else None,
                "any_lower_equals_first_positive_grid": bool(any(numerically_equal(float(x), min_pos) for x in lows)),
                "any_upper_equals_last_positive_grid": bool(any(numerically_equal(float(x), max_pos) for x in highs)),
                "n_lower_equals_first_positive_grid": int(sum(numerically_equal(float(x), min_pos) for x in lows)),
                "n_upper_equals_last_positive_grid": int(sum(numerically_equal(float(x), max_pos) for x in highs)),
                "frozen_v2_span_low": frozen_low,
                "frozen_v2_span_high": frozen_high,
                "source_artifact": "transition_lofo_sensitivity.csv",
            }
        )
    return pd.DataFrame(rows)


def build_sharpness_summary(sharp: pd.DataFrame) -> pd.DataFrame:
    primary = [m for m, _d in PRIMARY_METRICS]
    sub = sharp.loc[(sharp["split"] == "cv_mean") & (sharp["metric"].isin(primary))].copy()
    if sub.empty:
        # some v2 artifacts use no split filter beyond family/metric for cv_mean rows mixed with folds
        sub = sharp.loc[sharp["metric"].isin(primary) & (sharp["split"].astype(str) == "cv_mean")].copy()
    sub["best_vs_second_gap_over_range"] = pd.to_numeric(sub["best_vs_second_gap_over_range"], errors="coerce")
    sub["rank_by_normalized_best_vs_second_gap"] = sub.groupby("family")["best_vs_second_gap_over_range"].rank(
        ascending=False, method="min"
    )
    n_per = sub.groupby("family")["metric"].transform("count")
    sub["comparatively_sharp_rank_description"] = np.where(
        sub["rank_by_normalized_best_vs_second_gap"] == 1,
        "highest_normalized_best_vs_second_gap_in_family",
        np.where(
            sub["rank_by_normalized_best_vs_second_gap"] == n_per,
            "lowest_normalized_best_vs_second_gap_in_family",
            "intermediate",
        ),
    )
    keep = [
        "family",
        "split",
        "metric",
        "classification",
        "n_tied",
        "local_turn_verified",
        "optimum_rho",
        "best_vs_second_gap",
        "best_vs_second_gap_over_range",
        "lower_neighbor_gap_over_range",
        "higher_neighbor_gap_over_range",
        "rank_by_normalized_best_vs_second_gap",
        "comparatively_sharp_rank_description",
    ]
    # column names in v2 sharpness
    rename = {}
    if "n_tied" not in sub.columns and "n_tied_best" in sub.columns:
        sub["n_tied"] = sub["n_tied_best"]
    if "classification" not in sub.columns:
        # event_sharpness may use different field
        pass
    available = [c for c in keep if c in sub.columns]
    out = sub[available].copy()
    out["diagnostic_only"] = True
    out["does_not_define_a_new_rho_range"] = True
    return out.sort_values(["family", "rank_by_normalized_best_vs_second_gap", "metric"]).reset_index(drop=True)


def paper_value_crosscheck(combined: pd.DataFrame) -> List[str]:
    problems = []
    for (split, fam, metric), expected in PAPER_BASELINE_DISPLAY.items():
        row = combined_row(combined, fam)
        got = display_value(metric, metric_val(row, metric, split))
        if got != expected:
            problems.append(f"paper baseline mismatch {split} {fam} {metric}: got {got} expected {expected}")
    return problems


def extract_paper_printed_anchor_checks(combined: pd.DataFrame) -> List[str]:
    """Match already-printed (non-source) primary/secondary cells under display rounding."""
    problems = []
    grid = combined.loc[combined["family"] == "Direct", "rho"].to_numpy(dtype=float)
    anchors = positive_display_anchors(grid)
    printed = {
        ("heldout", "Direct", anchors[0], "R2_price"): "0.894",
        ("heldout", "Direct", anchors[0], "MAE_price"): "$76,235",
        ("heldout", "Direct", anchors[1], "R2_price"): "0.899",
        ("heldout", "Direct", anchors[1], "MAE_price"): "$74,485",
        ("heldout", "Surrogate", anchors[0], "R2_price"): "0.894",
        ("heldout", "Surrogate", anchors[3], "R2_price"): "0.889",
        ("forward_2025", "Direct", anchors[1], "R2_price"): "0.910",
        ("forward_2025", "Direct", anchors[1], "MAE_price"): "$77,139",
        ("heldout", "Linear", None, "median_ratio"): "0.969",
        ("heldout", "LightGBM", None, "median_ratio"): "0.929",
        ("heldout", "Linear", None, "COD"): "24.7%",
        ("heldout", "LightGBM", None, "COD"): "21.6%",
        ("heldout", "Direct", anchors[0], "Beta_log"): "-0.147",
        ("heldout", "Direct", anchors[3], "dCor_e_y"): "0.265",
        ("heldout", "Surrogate", anchors[2], "PRD"): "1.024",
        ("forward_2025", "Surrogate", anchors[2], "PRB"): "0.000",
    }
    for (split, fam, rho, metric), expected in printed.items():
        row = combined_row(combined, fam, rho)
        got = display_value(metric, metric_val(row, metric, split))
        if got != expected:
            problems.append(f"printed-cell mismatch {split} {fam} rho={rho} {metric}: got {got} expected {expected}")
    return problems


def source_placeholder_rows(primary: pd.DataFrame, secondary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mape_rmse = primary.loc[
        primary["family"].isin(["Direct", "Surrogate"]) & primary["metric"].isin(["MAPE", "RMSE_log"])
    ]
    for _, r in mape_rmse.iterrows():
        rows.append(
            {
                "table": "tab:path_anchor_summary",
                "split": r["split"],
                "family": r["family"],
                "nominal_display_anchor": r["nominal_display_anchor"],
                "exact_tested_rho": r["exact_tested_rho"],
                "metric": r["metric"],
                "value_unrounded": r["value_unrounded"],
                "value_display": r["value_display"],
                "manuscript_bold": r["manuscript_bold"],
                "manuscript_asterisk": r["manuscript_asterisk"],
                "replaces_paper_source_placeholder": True,
            }
        )
    sec_miss = secondary.loc[
        secondary["family"].isin(["Direct", "Surrogate"])
        & secondary["metric"].isin(["median_ratio", "mean_ratio", "weighted_mean_ratio", "COD", "COV"])
    ]
    for _, r in sec_miss.iterrows():
        rows.append(
            {
                "table": "tab:path_anchor_complementary",
                "split": r["split"],
                "family": r["family"],
                "nominal_display_anchor": r["nominal_display_anchor"],
                "exact_tested_rho": r["exact_tested_rho"],
                "metric": r["metric"],
                "value_unrounded": r["value_unrounded"],
                "value_display": r["value_display"],
                "manuscript_bold": r["manuscript_bold"],
                "manuscript_asterisk": r["manuscript_asterisk"],
                "replaces_paper_source_placeholder": True,
            }
        )
    return pd.DataFrame(rows)


def search_forbidden(root: Path) -> List[str]:
    hits = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".pdf", ".png", ".parquet"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for phrase in FORBIDDEN_PHRASES:
            if phrase in text:
                hits.append(f"{path.relative_to(REPO)}: {phrase}")
    return hits


def figure_qa_row(
    name: str,
    *,
    panel_count: int,
    x_metrics: Sequence[str],
    y_metrics: Sequence[str],
    grid_present: bool,
    neutral_refs_present: Sequence[str],
    span_fill: bool,
    span_boundaries: bool,
    span_source: str,
    input_sha: str,
    location: str,
) -> Dict[str, Any]:
    return {
        "figure_name": name,
        "panel_count": panel_count,
        "x_metrics": list(x_metrics),
        "y_metrics": list(y_metrics),
        "grid_present": grid_present,
        "neutral_refs_present": list(neutral_refs_present),
        "span_fill_present": span_fill,
        "span_boundaries_present": span_boundaries,
        "span_source": span_source,
        "input_table_sha256": input_sha,
        "location": location,
        "pdf_and_png": True,
    }


def main() -> int:
    problems: List[str] = []
    created: List[str] = []
    branch = git("branch", "--show-current")
    if branch != "testing":
        raise RuntimeError(f"expected testing branch, got {branch}")
    head = git("rev-parse", "HEAD")
    dirty = git("status", "--short")
    tex_before = sha256_file(PAPER_TEX)

    v3 = V3
    v3.mkdir(parents=True, exist_ok=True)
    guard = OutputGuard(v3, REPO)
    for sub in ("tables", "figures/main_candidate", "figures/appendix_candidate", "figures/diagnostic", "qa", "provenance"):
        guard.ensure_subdir(sub)

    frozen_files = {
        "paper/paper_v12.tex": tex_before,
        str((V2 / "tables" / "combined_path_table_v2.csv").relative_to(REPO)): sha256_file(V2 / "tables" / "combined_path_table_v2.csv"),
        str((V2 / "tables" / "transition_span_summary.csv").relative_to(REPO)): sha256_file(V2 / "tables" / "transition_span_summary.csv"),
        str((V2 / "tables" / "transition_events_cv_mean.csv").relative_to(REPO)): sha256_file(V2 / "tables" / "transition_events_cv_mean.csv"),
        str((V2 / "tables" / "transition_events_by_fold.csv").relative_to(REPO)): sha256_file(V2 / "tables" / "transition_events_by_fold.csv"),
        str((V2 / "tables" / "transition_lofo_sensitivity.csv").relative_to(REPO)): sha256_file(V2 / "tables" / "transition_lofo_sensitivity.csv"),
        str((V2 / "tables" / "transition_temporal_concordance.csv").relative_to(REPO)): sha256_file(V2 / "tables" / "transition_temporal_concordance.csv"),
        str((V2 / "tables" / "transition_event_sharpness.csv").relative_to(REPO)): sha256_file(V2 / "tables" / "transition_event_sharpness.csv"),
        str((EXT / "protocol" / "lower_rho_grid_v2.json").relative_to(REPO)): sha256_file(EXT / "protocol" / "lower_rho_grid_v2.json"),
        str((EXT / "qa" / "FINAL_STATUS.json").relative_to(REPO)): sha256_file(EXT / "qa" / "FINAL_STATUS.json"),
        str((V2 / "qa" / "MERGE_STATUS.json").relative_to(REPO)): sha256_file(V2 / "qa" / "MERGE_STATUS.json"),
        str((V2 / "qa" / "CV_TRANSITION_FREEZE_STATUS.json").relative_to(REPO)): sha256_file(V2 / "qa" / "CV_TRANSITION_FREEZE_STATUS.json"),
        str((CANONICAL / "analysis" / "combined_path_table.csv").relative_to(REPO)): sha256_file(CANONICAL / "analysis" / "combined_path_table.csv"),
        str((CANONICAL / "paper_outputs" / "tables" / "combined_path_table.csv").relative_to(REPO)): sha256_file(
            CANONICAL / "paper_outputs" / "tables" / "combined_path_table.csv"
        ),
    }
    v2_tree = hash_tree(V2)
    v2_assets = hash_tree(ASSETS_V2)
    paper_img = hash_tree(PAPER_IMG)
    preflight = {
        "utc": utc_now(),
        "git_branch": branch,
        "git_head": head,
        "git_status_short": dirty.splitlines(),
        "paper_v12_tex_sha256": tex_before,
        "note": "Working-tree paper/paper_v12.tex is the freeze baseline (user-updated, uncommitted).",
        "frozen_named_files": frozen_files,
        "v2_analysis_tree": v2_tree,
        "v2_paper_assets_tree": v2_assets,
        "paper_img_generated_v12_994": paper_img,
        "no_model_fitting": True,
        "no_slurm": True,
        "no_tex_write": True,
        "no_tex_compile": True,
    }
    created.append(str(guard.write_json(v3 / "provenance" / "preflight.json", preflight)))

    data = load_frozen()
    combined = data["combined"]
    span = data["span"]
    grid = data["grid"]
    min_pos = float(grid["min_positive_augmented"])
    q = float(grid["q"])

    if len(combined) != 168:
        problems.append(f"combined row count {len(combined)} != 168")
    for fam in FAMILY_DISPLAY:
        n_pos = int(sum(is_rho_positive(float(x)) for x in combined.loc[combined["family"] == fam, "rho"]))
        if n_pos != 82:
            problems.append(f"{fam} positive rho count {n_pos} != 82")
    dmin = float(np.min([float(x) for x in combined.loc[combined["family"] == "Direct", "rho"] if is_rho_positive(float(x))]))
    if not numerically_equal(dmin, min_pos):
        problems.append(f"smallest positive rho {dmin} != frozen {min_pos}")
    dlow, dhigh = family_span_endpoints(span, "Direct")
    slow, shigh = family_span_endpoints(span, "Surrogate")
    if not numerically_equal(dlow, 0.04941713361323837) or not numerically_equal(dhigh, 1.0985411419875584):
        problems.append(f"Direct span drifted: {[dlow, dhigh]}")
    if not numerically_equal(slow, 0.002222996482526201) or not numerically_equal(shigh, 0.954095476349994):
        problems.append(f"Surrogate span drifted: {[slow, shigh]}")

    problems.extend(paper_value_crosscheck(combined))
    problems.extend(extract_paper_printed_anchor_checks(combined))

    primary_b = build_long_table(combined, PRIMARY_METRICS_TABLE, include_penalized=False)
    secondary_b = build_long_table(combined, SECONDARY_METRICS_TABLE, include_penalized=False)
    primary_r = build_long_table(combined, PRIMARY_METRICS_TABLE, include_penalized=True)
    secondary_r = build_long_table(combined, SECONDARY_METRICS_TABLE, include_penalized=True)
    for name, df in (
        ("baseline_primary", primary_b),
        ("baseline_secondary", secondary_b),
        ("representative_rho_primary", primary_r),
        ("representative_rho_secondary", secondary_r),
    ):
        created.extend(write_table_bundle(guard, df, v3 / "tables" / name))
        if df["value_unrounded"].isna().any() or not np.isfinite(pd.to_numeric(df["value_unrounded"], errors="coerce")).all():
            problems.append(f"nonfinite values in {name}")

    placeholders = source_placeholder_rows(primary_r, secondary_r)
    created.extend(write_table_bundle(guard, placeholders, v3 / "tables" / "paper_source_placeholder_replacements"))

    ve_events = build_event_table(combined, span, VE_EVENT_SPECS)
    lu_events = build_event_table(combined, span, LEVEL_EVENT_SPECS)
    mech_events = build_event_table(combined, span, MECH_EVENT_SPECS)
    dnl_oos = build_delta_nl_oos(combined)
    for name, df in (
        ("vertical_equity_event_locations", ve_events),
        ("level_uniformity_event_locations", lu_events),
        ("mechanism_event_locations", mech_events),
        ("delta_nl_oos_only_event_locations", dnl_oos),
    ):
        created.extend(write_table_bundle(guard, df, v3 / "tables" / name))
        if "on_tested_grid" in df.columns and not bool(df["on_tested_grid"].all()):
            problems.append(f"event rho not on tested grid in {name}")

    regret = build_regret(combined, span)
    if len(regret) != 20:
        problems.append(f"span regret rows {len(regret)} != 20")
    created.extend(write_table_bundle(guard, regret, v3 / "tables" / "transition_oos_span_regret_v3"))

    lofo_sum = build_lofo_summary(data["lofo"], span, grid)
    created.extend(write_table_bundle(guard, lofo_sum, v3 / "tables" / "transition_lofo_endpoint_summary_v3"))
    for fam in FAMILY_DISPLAY:
        n = int(lofo_sum.loc[lofo_sum["family"] == fam, "n_valid"].iloc[0])
        if n != 7:
            problems.append(f"{fam} LOFO valid {n}/7")

    sharp = build_sharpness_summary(data["sharpness"])
    created.extend(write_table_bundle(guard, sharp, v3 / "tables" / "event_sharpness_summary_v3"))

    combined_sha = frozen_files[str((V2 / "tables" / "combined_path_table_v2.csv").relative_to(REPO))]
    span_src = "frozen_v2_cv_derived_descriptive_transition_span"

    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import utils.paper_v12_lower_rho_plots as P

    main_dir = v3 / "figures" / "main_candidate"
    app_dir = v3 / "figures" / "appendix_candidate"
    diag_dir = v3 / "figures" / "diagnostic"
    qa_mean: List[Dict[str, Any]] = []
    fig_qa: List[Dict[str, Any]] = []
    kw = dict(v3=True)

    def _saved(paths: List[str]) -> None:
        created.extend(paths)

    _saved(P.plot_mechanism(plt, combined, span, min_pos, q, main_dir / "mechanism_vs_rho", **kw))
    fig_qa.append(
        figure_qa_row(
            "mechanism_vs_rho",
            panel_count=6,
            x_metrics=["rho"],
            y_metrics=["Beta_log", "Delta_NL", "dCor_e_y"],
            grid_present=True,
            neutral_refs_present=["Beta_log=0"],
            span_fill=True,
            span_boundaries=True,
            span_source=span_src,
            input_sha=combined_sha,
            location="main_candidate",
        )
    )
    _saved(P.plot_main_tradeoff(plt, combined, main_dir / "accuracy_equity_trajectories_inprocessing_only", **kw))
    fig_qa.append(
        figure_qa_row(
            "accuracy_equity_trajectories_inprocessing_only",
            panel_count=8,
            x_metrics=["PRD", "PRB", "MKI", "VEI"],
            y_metrics=["R2_price"],
            grid_present=True,
            neutral_refs_present=["PRD=1", "PRB=0", "MKI=1", "VEI=0"],
            span_fill=False,
            span_boundaries=False,
            span_source="none_tradeoff_plane",
            input_sha=combined_sha,
            location="main_candidate",
        )
    )
    _saved(P.plot_ratio_shape(plt, CANONICAL, combined, main_dir / "ratio_shape_evolution", **kw))
    fig_qa.append(
        figure_qa_row(
            "ratio_shape_evolution",
            panel_count=4,
            x_metrics=["sale_price"],
            y_metrics=["median_ratio"],
            grid_present=True,
            neutral_refs_present=["ratio=1", "ratio=0.9", "ratio=1.1"],
            span_fill=False,
            span_boundaries=False,
            span_source="none_ratio_shape_vs_sale_price",
            input_sha=combined_sha,
            location="main_candidate",
        )
    )
    _saved(P.plot_predictive(plt, combined, span, min_pos, q, app_dir / "predictive_metric_paths", **kw))
    fig_qa.append(
        figure_qa_row(
            "predictive_metric_paths",
            panel_count=8,
            x_metrics=["rho"],
            y_metrics=["R2_price", "MAE_price", "MAPE", "RMSE_log"],
            grid_present=True,
            neutral_refs_present=[],
            span_fill=True,
            span_boundaries=True,
            span_source=span_src,
            input_sha=combined_sha,
            location="appendix_candidate",
        )
    )
    _saved(P.plot_level_uniformity(plt, combined, span, min_pos, q, app_dir / "level_uniformity_paths", **kw))
    fig_qa.append(
        figure_qa_row(
            "level_uniformity_paths",
            panel_count=10,
            x_metrics=["rho"],
            y_metrics=["median_ratio", "mean_ratio", "weighted_mean_ratio", "COD", "COV"],
            grid_present=True,
            neutral_refs_present=["median_ratio=1", "mean_ratio=1", "weighted_mean_ratio=1"],
            span_fill=True,
            span_boundaries=True,
            span_source=span_src,
            input_sha=combined_sha,
            location="appendix_candidate",
        )
    )
    _saved(P.plot_vertical_equity(plt, combined, span, min_pos, q, app_dir / "vertical_equity_metric_paths", **kw))
    fig_qa.append(
        figure_qa_row(
            "vertical_equity_metric_paths",
            panel_count=8,
            x_metrics=["rho"],
            y_metrics=["PRD", "PRB", "MKI", "VEI"],
            grid_present=True,
            neutral_refs_present=["PRD=1", "PRB=0", "MKI=1", "VEI=0"],
            span_fill=True,
            span_boundaries=True,
            span_source=span_src,
            input_sha=combined_sha,
            location="appendix_candidate",
        )
    )
    _saved(
        P.plot_cv_group(
            plt,
            combined,
            span,
            (("R2_price", r"$R^2_P$"), ("MAE_price", r"MAE$_P$"), ("MAPE", r"MAPE$_P$ (\%)"), ("RMSE_log", r"RMSE$_{\log P}$")),
            min_pos,
            q,
            app_dir / "cv_predictive_metric_paths",
            qa_mean,
            **kw,
        )
    )
    fig_qa.append(
        figure_qa_row(
            "cv_predictive_metric_paths",
            panel_count=8,
            x_metrics=["rho"],
            y_metrics=["R2_price", "MAE_price", "MAPE", "RMSE_log"],
            grid_present=True,
            neutral_refs_present=[],
            span_fill=True,
            span_boundaries=True,
            span_source=span_src,
            input_sha=combined_sha,
            location="appendix_candidate",
        )
    )
    _saved(
        P.plot_cv_group(
            plt,
            combined,
            span,
            (("median_ratio", "Median ratio"), ("mean_ratio", "Mean ratio"), ("weighted_mean_ratio", "W. mean ratio"), ("COD", "COD"), ("COV", "COV (\%)")),
            min_pos,
            q,
            app_dir / "cv_level_uniformity_paths",
            qa_mean,
            **kw,
        )
    )
    fig_qa.append(
        figure_qa_row(
            "cv_level_uniformity_paths",
            panel_count=10,
            x_metrics=["rho"],
            y_metrics=["median_ratio", "mean_ratio", "weighted_mean_ratio", "COD", "COV"],
            grid_present=True,
            neutral_refs_present=["median_ratio=1", "mean_ratio=1", "weighted_mean_ratio=1"],
            span_fill=True,
            span_boundaries=True,
            span_source=span_src,
            input_sha=combined_sha,
            location="appendix_candidate",
        )
    )
    _saved(
        P.plot_cv_group(
            plt,
            combined,
            span,
            (("PRD", "PRD"), ("PRB", "PRB"), ("MKI", "MKI"), ("VEI", "VEI")),
            min_pos,
            q,
            app_dir / "cv_vertical_equity_metric_paths",
            qa_mean,
            **kw,
        )
    )
    fig_qa.append(
        figure_qa_row(
            "cv_vertical_equity_metric_paths",
            panel_count=8,
            x_metrics=["rho"],
            y_metrics=["PRD", "PRB", "MKI", "VEI"],
            grid_present=True,
            neutral_refs_present=["PRD=1", "PRB=0", "MKI=1", "VEI=0"],
            span_fill=True,
            span_boundaries=True,
            span_source=span_src,
            input_sha=combined_sha,
            location="appendix_candidate",
        )
    )
    _saved(
        P.plot_cv_group(
            plt,
            combined,
            span,
            (("Beta_log", r"$\beta_{\log}$"), ("dCor_e_y", r"dCor$(e,y)$")),
            min_pos,
            q,
            app_dir / "cv_mechanism_metric_paths",
            qa_mean,
            **kw,
        )
    )
    fig_qa.append(
        figure_qa_row(
            "cv_mechanism_metric_paths",
            panel_count=4,
            x_metrics=["rho"],
            y_metrics=["Beta_log", "dCor_e_y"],
            grid_present=True,
            neutral_refs_present=["Beta_log=0"],
            span_fill=True,
            span_boundaries=True,
            span_source=span_src,
            input_sha=combined_sha,
            location="appendix_candidate",
        )
    )
    _saved(P.plot_ratio_shape_span_only(plt, CANONICAL, combined, span, app_dir / "ratio_shape_cv_transition_span_only", **kw))
    fig_qa.append(
        figure_qa_row(
            "ratio_shape_cv_transition_span_only",
            panel_count=4,
            x_metrics=["sale_price"],
            y_metrics=["median_ratio"],
            grid_present=True,
            neutral_refs_present=["ratio=1", "ratio=0.9", "ratio=1.1"],
            span_fill=False,
            span_boundaries=False,
            span_source="none_ratio_shape_vs_sale_price_subset_of_span_rhos_only",
            input_sha=combined_sha,
            location="appendix_candidate",
        )
    )
    equity_x = (("PRD", "PRD", (0.98, 1.03)), ("PRB", "PRB", (-0.05, 0.05)), ("MKI", "MKI", (0.95, 1.05)), ("VEI", "VEI", (-10.0, 10.0)))
    _saved(P.plot_tradeoff_atlas(plt, combined, equity_x, app_dir / "tradeoff_equity_vs_accuracy_heldout", **kw))
    _saved(P.plot_tradeoff_atlas(plt, combined, equity_x, app_dir / "tradeoff_equity_vs_accuracy_2025", **kw))
    for nm in ("tradeoff_equity_vs_accuracy_heldout", "tradeoff_equity_vs_accuracy_2025"):
        fig_qa.append(
            figure_qa_row(
                nm,
                panel_count=16,
                x_metrics=["PRD", "PRB", "MKI", "VEI"],
                y_metrics=["R2_price", "MAE_price", "MAPE", "RMSE_log"],
                grid_present=True,
                neutral_refs_present=["PRD=1", "PRB=0", "MKI=1", "VEI=0"],
                span_fill=False,
                span_boundaries=False,
                span_source="none_tradeoff_plane",
                input_sha=combined_sha,
                location="appendix_candidate",
            )
        )
    mech_x = (("Beta_log", r"$\beta_{\log}$", None), ("Delta_NL", r"$\Delta_{NL}$", None), ("dCor_e_y", "dCor", None))
    _saved(
        P.plot_tradeoff_atlas(
            plt,
            combined,
            mech_x,
            app_dir / "tradeoff_mechanism_vs_accuracy_heldout",
            zero_x={"Beta_log"},
            no_zero={"Delta_NL", "dCor_e_y"},
            **kw,
        )
    )
    _saved(
        P.plot_tradeoff_atlas(
            plt,
            combined,
            mech_x,
            app_dir / "tradeoff_mechanism_vs_accuracy_2025",
            zero_x={"Beta_log"},
            no_zero={"Delta_NL", "dCor_e_y"},
            **kw,
        )
    )
    for nm in ("tradeoff_mechanism_vs_accuracy_heldout", "tradeoff_mechanism_vs_accuracy_2025"):
        fig_qa.append(
            figure_qa_row(
                nm,
                panel_count=12,
                x_metrics=["Beta_log", "Delta_NL", "dCor_e_y"],
                y_metrics=["R2_price", "MAE_price", "MAPE", "RMSE_log"],
                grid_present=True,
                neutral_refs_present=["Beta_log=0"],
                span_fill=False,
                span_boundaries=False,
                span_source="none_tradeoff_plane",
                input_sha=combined_sha,
                location="appendix_candidate",
            )
        )
    tables = {
        "transition_events_cv_mean.csv": data["events_cv"],
        "transition_events_by_fold.csv": data["events_fold"],
        "transition_temporal_concordance.csv": data["concordance"],
        "transition_span_summary.csv": span,
        "transition_lofo_sensitivity.csv": data["lofo"],
    }
    _saved(P.plot_event_locations(plt, combined, tables, min_pos, q, app_dir / "paper_transition_event_locations", **kw))
    fig_qa.append(
        figure_qa_row(
            "paper_transition_event_locations",
            panel_count=2,
            x_metrics=["rho"],
            y_metrics=[m for m, _ in PRIMARY_METRICS],
            grid_present=True,
            neutral_refs_present=[],
            span_fill=True,
            span_boundaries=True,
            span_source=span_src,
            input_sha=combined_sha,
            location="appendix_candidate",
        )
    )
    _saved(
        P.plot_descriptive_event_locations(
            plt,
            ve_events,
            span,
            min_pos,
            q,
            app_dir / "vertical_equity_event_locations",
            labels={
                "PRD": r"$|\mathrm{PRD}-1|$ min",
                "PRB": r"$|\mathrm{PRB}|$ min",
                "MKI": r"$|\mathrm{MKI}-1|$ min",
                "VEI": r"$|\mathrm{VEI}|$ min",
            },
            metric_order=["PRD", "PRB", "MKI", "VEI"],
            note="Descriptive neutrality events. Gray is the frozen five-metric CV-derived descriptive transition span.",
        )
    )
    fig_qa.append(
        figure_qa_row(
            "vertical_equity_event_locations",
            panel_count=2,
            x_metrics=["rho"],
            y_metrics=["PRD", "PRB", "MKI", "VEI"],
            grid_present=True,
            neutral_refs_present=[],
            span_fill=True,
            span_boundaries=True,
            span_source=span_src,
            input_sha=combined_sha,
            location="appendix_candidate",
        )
    )
    _saved(
        P.plot_descriptive_event_locations(
            plt,
            lu_events,
            span,
            min_pos,
            q,
            app_dir / "level_uniformity_event_locations",
            labels={
                "median_ratio": r"$|$median$-1|$ min",
                "mean_ratio": r"$|$mean$-1|$ min",
                "weighted_mean_ratio": r"$|$w.mean$-1|$ min",
                "COD": "COD min",
                "COV": "COV min",
            },
            metric_order=["median_ratio", "mean_ratio", "weighted_mean_ratio", "COD", "COV"],
            note="Descriptive neutrality/minimum events. Gray is the frozen five-metric CV-derived descriptive transition span.",
        )
    )
    fig_qa.append(
        figure_qa_row(
            "level_uniformity_event_locations",
            panel_count=2,
            x_metrics=["rho"],
            y_metrics=["median_ratio", "mean_ratio", "weighted_mean_ratio", "COD", "COV"],
            grid_present=True,
            neutral_refs_present=[],
            span_fill=True,
            span_boundaries=True,
            span_source=span_src,
            input_sha=combined_sha,
            location="appendix_candidate",
        )
    )
    _saved(
        P.plot_descriptive_event_locations(
            plt,
            mech_events,
            span,
            min_pos,
            q,
            app_dir / "mechanism_event_locations",
            labels={"Beta_log": r"$|\beta_{\log}|$ min", "dCor_e_y": r"dCor min"},
            metric_order=["Beta_log", "dCor_e_y"],
            note="Descriptive mechanism events. No CV Delta_NL. Gray is the frozen five-metric CV-derived descriptive transition span.",
        )
    )
    fig_qa.append(
        figure_qa_row(
            "mechanism_event_locations",
            panel_count=2,
            x_metrics=["rho"],
            y_metrics=["Beta_log", "dCor_e_y"],
            grid_present=True,
            neutral_refs_present=[],
            span_fill=True,
            span_boundaries=True,
            span_source=span_src,
            input_sha=combined_sha,
            location="appendix_candidate",
        )
    )
    _saved(P.plot_main_tradeoff(plt, combined, diag_dir / "accuracy_equity_no_linear_preview", v3=True, omit_linear=True))
    fig_qa.append(
        figure_qa_row(
            "accuracy_equity_no_linear_preview",
            panel_count=8,
            x_metrics=["PRD", "PRB", "MKI", "VEI"],
            y_metrics=["R2_price"],
            grid_present=True,
            neutral_refs_present=["PRD=1", "PRB=0", "MKI=1", "VEI=0"],
            span_fill=False,
            span_boundaries=False,
            span_source="none_tradeoff_plane",
            input_sha=combined_sha,
            location="diagnostic",
        )
    )

    if any(not x.get("ok") for x in qa_mean):
        problems.append("CV mean vs fold-average mismatch")

    created.append(str(guard.write_json(v3 / "qa" / "figure_qa_manifest.json", fig_qa)))
    created.append(str(guard.write_json(v3 / "qa" / "cv_mean_fold_checks.json", qa_mean)))

    # Immutability re-hash
    tex_after = sha256_file(PAPER_TEX)
    if tex_after != tex_before:
        problems.append("paper/paper_v12.tex HASH CHANGED")
    for rel, expected in frozen_files.items():
        got = sha256_file(REPO / rel)
        if got != expected:
            problems.append(f"frozen hash changed: {rel}")
    v2_tree_after = hash_tree(V2)
    if v2_tree_after != v2_tree:
        problems.append("v2 analysis tree hash changed")
    v2_assets_after = hash_tree(ASSETS_V2)
    if v2_assets_after != v2_assets:
        problems.append("v2 paper-asset tree hash changed")
    paper_img_after = hash_tree(PAPER_IMG)
    if paper_img_after != paper_img:
        problems.append("paper/img/generated_v12_994 hash changed")

    tex_created = [str(p) for p in v3.rglob("*.tex")]
    if tex_created:
        problems.append(f".tex files created under v3: {tex_created}")
    forbidden = search_forbidden(v3)
    if forbidden:
        problems.append(f"forbidden phrases: {forbidden}")

    classification = {
        "baseline_primary": "PAPER-READY",
        "baseline_secondary": "PAPER-READY",
        "representative_rho_primary": "PAPER-READY",
        "representative_rho_secondary": "PAPER-READY",
        "paper_source_placeholder_replacements": "PAPER-READY",
        "mechanism_vs_rho": "PAPER-CANDIDATE",
        "accuracy_equity_trajectories_inprocessing_only": "PAPER-CANDIDATE",
        "ratio_shape_evolution": "PAPER-CANDIDATE",
        "predictive_metric_paths": "PAPER-CANDIDATE",
        "level_uniformity_paths": "PAPER-CANDIDATE",
        "vertical_equity_metric_paths": "PAPER-CANDIDATE",
        "cv_predictive_metric_paths": "PAPER-CANDIDATE",
        "cv_level_uniformity_paths": "PAPER-CANDIDATE",
        "cv_vertical_equity_metric_paths": "PAPER-CANDIDATE",
        "cv_mechanism_metric_paths": "PAPER-CANDIDATE",
        "ratio_shape_cv_transition_span_only": "PAPER-CANDIDATE",
        "tradeoff_equity_vs_accuracy_heldout": "PAPER-CANDIDATE",
        "tradeoff_equity_vs_accuracy_2025": "PAPER-CANDIDATE",
        "tradeoff_mechanism_vs_accuracy_heldout": "PAPER-CANDIDATE",
        "tradeoff_mechanism_vs_accuracy_2025": "PAPER-CANDIDATE",
        "paper_transition_event_locations": "PAPER-CANDIDATE",
        "vertical_equity_event_locations": "PAPER-CANDIDATE",
        "level_uniformity_event_locations": "PAPER-CANDIDATE",
        "mechanism_event_locations": "PAPER-CANDIDATE",
        "transition_oos_span_regret_v3": "PAPER-CANDIDATE",
        "transition_lofo_endpoint_summary_v3": "PAPER-CANDIDATE",
        "event_sharpness_summary_v3": "DIAGNOSTIC-ONLY",
        "delta_nl_oos_only_event_locations": "DIAGNOSTIC-ONLY",
        "accuracy_equity_no_linear_preview": "DIAGNOSTIC-ONLY",
    }

    status = "PASS" if not problems else "FAIL"
    final = {
        "status": status,
        "utc": utc_now(),
        "git_branch": branch,
        "git_head": head,
        "git_status_short": dirty.splitlines(),
        "paper_v12_tex_sha256_before": tex_before,
        "paper_v12_tex_sha256_after": tex_after,
        "paper_v12_tex_unchanged": tex_after == tex_before,
        "v1_hashes_unchanged": True if not any("canonical" in p.lower() or "v1" in p.lower() or "paper_v6" in p for p in problems) else False,
        "v2_hashes_unchanged": v2_tree_after == v2_tree and v2_assets_after == v2_assets,
        "no_model_fitting": True,
        "no_slurm": True,
        "no_tex_file_created_or_modified": not tex_created and tex_after == tex_before,
        "no_tex_compilation": True,
        "v3_root": str(v3.relative_to(REPO)),
        "n_created": len(created),
        "created_files": created,
        "classification": classification,
        "problems": problems,
        "direct_span": [dlow, dhigh],
        "surrogate_span": [slow, shigh],
        "statement": "NO PAPER FILES WERE POPULATED. NO LATEX/TEX COMPILATION WAS PERFORMED.",
    }
    created.append(str(guard.write_json(v3 / "qa" / "FINAL_STATUS.json", final)))
    created.append(str(guard.write_json(v3 / "provenance" / "postflight.json", {"tex_after": tex_after, "problems": problems})))
    print(json.dumps({"status": status, "n_problems": len(problems), "problems": problems[:20], "v3": str(v3)}, indent=2))
    return 0 if not problems else 1


if __name__ == "__main__":
    rc = 1
    try:
        rc = main()
    except Exception:
        import traceback

        traceback.print_exc()
        rc = 1
    finally:
        os._exit(rc)
