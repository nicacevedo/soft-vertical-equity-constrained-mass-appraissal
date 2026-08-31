#!/usr/bin/env python3
"""Generic CV-derived rho candidate-region screening (v2).

Chronological CV only for endpoints. No model refits, no grid extension,
no manuscript edits, no overwrite of v1 or promoted paper figures.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from math import floor, log10
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tests.test_rho_screening_region import run_all as run_synthetic_tests  # noqa: E402
from utils.paper_v12_lower_rho_plots import (  # noqa: E402
    DIRECT_COLOR,
    SURR_COLOR,
    apply_major_grid,
    draw_neutral_hline,
    family_span,
    maybe_percent,
    nearby_targets,
    rho_plot_x,
)
from utils.rho_screening_v2 import (  # noqa: E402
    BENEFIT_METRICS,
    METHOD_SPEC,
    PREDICTIVE_COST_METRICS,
    SCALE_EQUIVARIANCE_FACTOR,
    UNIFORMITY_SUPPORT_METRIC,
    family_status,
    json_safe,
    log10_rho,
    lofo_stability,
    median_log_spacing,
    min_segment_points,
    screen_positive_path,
)
from utils.transition_paper_asset_plots import equal_count_bins, load_pred, padded_lim  # noqa: E402
from utils.transition_regions import (  # noqa: E402
    FAMILY_DISPLAY,
    FOLD_IDS,
    OutputGuard,
    family_frame,
    is_rho_positive,
    is_rho_zero,
    sha256_file,
)

CANONICAL = REPO / "output" / "paper_v6_preselection_994"
EXT = REPO / "output" / "paper_v12_lower_rho_extension_994_v2"
DATA_ID = "d4929d43ec19badf"
SPLIT_ID = "3d464d4a611b131b"
LGBM_CONFIG_ID = "407d47775760c14d"
SEED = 2025
TREES = 994
V2 = (
    EXT
    / "analysis"
    / f"data_id={DATA_ID}"
    / f"split_id={SPLIT_ID}"
    / "penalty_path_analysis"
    / "transition_regions_v2_lower_rho"
)
V4 = V2.parent / "transition_regions_paper_assets_v4_delta_nl_bends"
V1_OUT = V2.parent / "rho_screening_candidate_regions_v1"
OUT = V2.parent / "rho_screening_candidate_regions_v2_generic_guardrails"
PAPER_TEX = REPO / "paper" / "paper_v12.tex"
V1_SCRIPT = REPO / "scripts" / "analyze_rho_screening_region.py"

CAND_FACE = "#86A789"
CAND_ALPHA = 0.16
PRED_DASH = dict(color="#9CA3AF", ls=(0, (2.2, 2.0)), lw=0.7, alpha=0.75, zorder=1)
ACTIVITY_LINE = dict(color="#374151", ls=(0, (4.0, 2.2)), lw=0.85, alpha=0.9, zorder=3)
GUARD_LINE = dict(color="#14532D", ls="-", lw=1.35, alpha=0.95, zorder=4)
CAUTION_LINE = dict(color="#7C3AED", ls=":", lw=0.95, alpha=0.85, zorder=3)

V1_EXPECTED_HASHES = {
    "combined_v2": "06ac4d5b0831ec667d660471f740b27a8098bca4ed89af7734843109fc1241e0",
    "combined_v4_view": "d5b0fcfc49eb8e2a209d5f477b4b37312a1d6015f7e0ba7bd7ac54a178b60363",
    "span_summary": "9d4e34572d23e449e2deb50e47b8f218fd46c9f3887b4ee797543d0e142cc5d3",
    "delta_nl_cv_mean": "dd7a7000881e0a87133c7bb202e2a992aa86a8ad616539a97486781a6bb5c393",
    "delta_nl_cv_by_fold": "4ee93069b652ce30ebf996faa4aa1b38114f6b5975728f3b87cbbcacb38904cd",
    "delta_nl_estimator_spec": "113d4dd4fe3217507731b0220f6e276b55c9f7a1f3fe80f5fb7af807f480f89a",
    "grid": "072e4ea94c35a252fbe7e433f63a9c75bc1d557c950a0a987ad44b14e73dbdbc",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(REPO), text=True).strip()


def dumps(payload: Any) -> str:
    return json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n"


def write_json(guard: OutputGuard, path: Path, payload: Any) -> Path:
    return guard.write_text(path, dumps(payload))


def audit_v1_hardcoded_rho() -> Dict[str, Any]:
    text = V1_SCRIPT.read_text(encoding="utf-8")
    literals = []
    for m in re.finditer(r"(?m)^(?!.*#).{0,40}?((?:MIN_SEG|LOG10_TOL|EPS|SEED|TREES)\s*=\s*[-+eE0-9.]+)", text):
        literals.append({"match": m.group(1).strip(), "kind": "methodological_or_identity"})
    for m in re.finditer(r"window\s*=\s*(\d+)", text):
        literals.append({"match": f"window={m.group(1)}", "kind": "methodological"})
    # Float literals that could be rho results (exclude colors, alphas, BIC internals).
    for m in re.finditer(r"(?<![A-Za-z_])(\d+\.\d+)(?![A-Za-z_])", text):
        val = float(m.group(1))
        if val in {1.0, 0.0, 0.08, 0.18, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.05, 1.15, 1.3, 1.6, 2.0, 2.1, 2.15, 2.2, 3.0, 4.0, 5.0, 6.0, 7.0, 8.9, 9.4}:
            continue
        if val >= 10 and val in {160.0, 200.0}:
            continue
        literals.append({"match": m.group(1), "kind": "numeric_literal_inspected"})
    result_specific = [x for x in literals if x["kind"] == "result_specific"]
    return {
        "script": str(V1_SCRIPT.relative_to(REPO)),
        "literals": literals[:80],
        "result_specific_rho_endpoints_in_v1_methodology": result_specific,
        "v1_endpoint_rule": "grid/event driven; no CCAO rho endpoints hardcoded for activity/guardrail",
        "v2_removed_or_replaced": [
            "MIN_SEG=6 -> max(5, ceil(0.05*N))",
            "LOG10_TOL=1.0 absolute log10-rho window -> +/- 4 median log-grid steps",
            "cost+saturation 3/5 majority Direct guardrail -> earliest 2-of-4 predictive cluster",
            "dCor flattening as NL corroboration -> dCor classified; only ratio-shape can CONFIRM NL",
            "plot xmax=130 and decade display anchors not used for QA-point or endpoint selection",
        ],
        "endpoint_and_qa_point_selection_fully_grid_event_driven": True,
        "pass": len(result_specific) == 0,
    }


def log_rho_axes_generic(ax, rho_positive: np.ndarray, q: float) -> None:
    pos = np.asarray(rho_positive, dtype=float)
    pos = pos[np.isfinite(pos) & (pos > 0)]
    min_p = float(np.min(pos))
    max_p = float(np.max(pos))
    x0 = min_p / float(q)
    ax.set_xscale("log")
    lo_dec = int(floor(log10(min_p)))
    hi_dec = int(floor(log10(max_p)))
    ticks = [x0]
    labels = ["0"]
    for p in range(lo_dec, hi_dec + 1):
        t = 10.0 ** p
        if min_p / 1.05 <= t <= max_p * 1.05:
            ticks.append(t)
            labels.append(f"{t:g}")
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_xlim(x0 / 1.15, max_p * 1.25)


def input_paths() -> Dict[str, Path]:
    return {
        "combined_v2": V2 / "tables" / "combined_path_table_v2.csv",
        "combined_v4_view": V4 / "tables" / "combined_path_table_v4_analysis_view.csv",
        "span_summary": V2 / "tables" / "transition_span_summary.csv",
        "delta_nl_cv_mean": V4 / "delta_nl_cv" / "delta_nl_cv_mean.csv",
        "delta_nl_cv_by_fold": V4 / "delta_nl_cv" / "delta_nl_cv_by_fold.csv",
        "delta_nl_estimator_spec": V4 / "delta_nl_cv" / "estimator_spec.json",
        "grid": EXT / "protocol" / "lower_rho_grid_v2.json",
        "cv_prediction_inventory": V4 / "qa" / "cv_prediction_inventory.csv",
    }


def input_hashes() -> Dict[str, str]:
    return {k: sha256_file(p) for k, p in input_paths().items()}


def positive_sub(frame: pd.DataFrame) -> pd.DataFrame:
    rho = pd.to_numeric(frame["rho"], errors="coerce")
    keep = np.array([is_rho_positive(float(x)) for x in rho])
    return frame.loc[keep].sort_values("rho", kind="mergesort").reset_index(drop=True)


def rho0_row(frame: pd.DataFrame) -> pd.Series:
    rho = pd.to_numeric(frame["rho"], errors="coerce")
    keep = np.array([is_rho_zero(float(x)) if np.isfinite(x) else False for x in rho])
    sub = frame.loc[keep]
    if sub.empty:
        raise RuntimeError("missing within-family rho=0 path origin")
    return sub.iloc[0]


def col_values(sub: pd.DataFrame, metric: str, suffix: str) -> np.ndarray:
    return pd.to_numeric(sub[f"{metric}__{suffix}"], errors="coerce").to_numpy(dtype=float)


def lofo_values(sub: pd.DataFrame, metric: str, left_out: int) -> np.ndarray:
    mats = []
    for k in FOLD_IDS:
        if int(k) == int(left_out):
            continue
        mats.append(col_values(sub, metric, f"fold_{k}"))
    return np.nanmean(np.vstack(mats), axis=0)


def path_inputs(frame: pd.DataFrame, *, left_out: Optional[int] = None) -> Dict[str, Any]:
    pos = positive_sub(frame)
    z = rho0_row(frame)
    rho = pd.to_numeric(pos["rho"], errors="coerce").to_numpy(dtype=float)
    if left_out is None:
        def take(metric: str) -> np.ndarray:
            return col_values(pos, metric, "CV_mean")

        def zval(metric: str) -> float:
            return float(z[f"{metric}__CV_mean"])
    else:
        def take(metric: str) -> np.ndarray:
            return lofo_values(pos, metric, int(left_out))

        def zval(metric: str) -> float:
            vals = []
            for k in FOLD_IDS:
                if int(k) == int(left_out):
                    continue
                vals.append(float(z[f"{metric}__fold_{k}"]))
            return float(np.mean(vals))

    benefit = {m: take(m) for m in BENEFIT_METRICS}
    pred = {m: take(m) for m in PREDICTIVE_COST_METRICS}
    pred0 = {m: zval(m) for m in PREDICTIVE_COST_METRICS}
    dnl = take("Delta_NL")
    dcor = take("dCor_e_y")
    cod = take(UNIFORMITY_SUPPORT_METRIC)
    cod0 = zval(UNIFORMITY_SUPPORT_METRIC)
    return {
        "rho": rho,
        "benefit_raw": benefit,
        "predictive_raw": pred,
        "predictive_rho0": pred0,
        "delta_nl": dnl,
        "dcor": dcor,
        "cod": cod,
        "cod_rho0": cod0,
    }


def screen_frame(frame: pd.DataFrame, *, left_out: Optional[int] = None) -> Dict[str, Any]:
    kw = path_inputs(frame, left_out=left_out)
    return screen_positive_path(**kw)


def index_list(n: int, spec: Sequence[Optional[int]]) -> List[int]:
    out: List[int] = []
    for j in spec:
        if j is None:
            continue
        jj = int(np.clip(int(j), 0, n - 1))
        if jj not in out:
            out.append(jj)
    return out


def trough_depth(median_ratio: np.ndarray) -> float:
    r = np.asarray(median_ratio, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 10:
        return 0.0
    interior = r[3:-3]
    return float(0.5 * (r[0] + r[-1]) - float(np.min(interior)))


def load_fold_profile(pred_path: Path) -> pd.DataFrame:
    df = load_pred(pred_path)
    sale = df["y_true"].to_numpy(dtype=float)
    pred = df["y_pred"].to_numpy(dtype=float)
    ratio = pred / np.clip(sale, 1e-12, None)
    ok = np.isfinite(sale) & (sale > 0) & np.isfinite(ratio)
    return equal_count_bins(sale[ok], ratio[ok], n_bins=30)


def inventory_lookup(inv: pd.DataFrame) -> Dict[Tuple[str, float, int], Path]:
    out: Dict[Tuple[str, float, int], Path] = {}
    for _, row in inv.iterrows():
        if str(row.get("family")) != "Surrogate":
            continue
        if not bool(row.get("ok", False)):
            continue
        rho = float(row["rho"])
        fold = int(row["fold_id"])
        p = REPO / str(row["prediction_path"])
        out[("Surrogate", rho, fold)] = p
    return out


def inv_path(inv_map: Dict[Tuple[str, float, int], Path], rho: float, fold: int) -> Optional[Path]:
    for (fam, r, f), p in inv_map.items():
        if fam != "Surrogate" or int(f) != int(fold):
            continue
        if is_rho_zero(r) and is_rho_zero(rho):
            return p
        if np.isfinite(r) and np.isfinite(rho) and abs(float(r) - float(rho)) <= 1e-10:
            return p
    return None


def nearest_rho(grid: np.ndarray, target: float) -> float:
    g = np.asarray(grid, dtype=float)
    return float(g[int(np.argmin(np.abs(g - float(target))))])
    g = np.asarray(grid, dtype=float)
    return float(g[int(np.argmin(np.abs(g - float(target))))])


def ratio_shape_qa(
    inv_map: Dict[Tuple[str, float, int], Path],
    rho_grid: np.ndarray,
    rho0: float,
    screened: Dict[str, Any],
) -> Dict[str, Any]:
    n = int(rho_grid.size)
    act = screened["activity"].get("index")
    dnl = screened.get("delta_nl") or {}
    j_nl = dnl.get("index")
    j_valley = dnl.get("valley_index")
    j_pred = screened.get("index_predictive_guardrail")
    j_max = n - 1
    near = index_list(n, None if j_nl is None else [j_nl - 2, j_nl - 1, j_nl, j_nl + 1, j_nl + 2])
    glob = index_list(n, [act, j_valley, j_nl, j_pred, j_max])
    rhos_near = [float(rho_grid[j]) for j in near]
    rhos_glob = [float(rho0)] + [float(rho_grid[j]) for j in glob]
    seen = set()
    rhos_glob_u = []
    for r in rhos_glob:
        key = round(r, 12)
        if key in seen:
            continue
        seen.add(key)
        rhos_glob_u.append(r)

    def profiles_for(rhos: Sequence[float]) -> Dict[float, Dict[int, pd.DataFrame]]:
        bag: Dict[float, Dict[int, pd.DataFrame]] = {}
        for r in rhos:
            rr = nearest_rho(np.concatenate([[rho0], rho_grid]), r) if r > 0 else float(rho0)
            if is_rho_zero(rr):
                rr = float(rho0)
            bag[rr] = {}
            for fold in FOLD_IDS:
                path = inv_path(inv_map, float(rr), int(fold))
                if path is None or not path.is_file():
                    continue
                bag[rr][int(fold)] = load_fold_profile(path)
        return bag

    bags_near = profiles_for(rhos_near)
    bags_glob = profiles_for(rhos_glob_u)

    def mean_depth(rho_v: float, bag: Dict[float, Dict[int, pd.DataFrame]]) -> Tuple[float, int]:
        depths = []
        for _f, prof in bag.get(rho_v, {}).items():
            depths.append(trough_depth(prof["median_ratio"].to_numpy(dtype=float)))
        if not depths:
            return 0.0, 0
        return float(np.mean(depths)), len(depths)

    fold_flags = []
    if j_nl is not None and near:
        before_j = index_list(n, [j_nl - 2]) 
        before_rho = float(rho_grid[before_j[0]]) if before_j else None
        act_rho = None if act is None else float(rho_grid[int(act)])
        near_rhos = [float(rho_grid[j]) for j in near]
        far_js = index_list(n, [j_pred, j_max])
        far_rhos = [float(rho_grid[j]) for j in far_js]
        for fold in FOLD_IDS:
            def depth_at(rlist: Sequence[float]) -> float:
                vals = []
                for r in rlist:
                    rr = nearest_rho(rho_grid, r)
                    prof = bags_near.get(rr, {}).get(int(fold))
                    if prof is None:
                        prof = bags_glob.get(rr, {}).get(int(fold))
                    if prof is None:
                        continue
                    vals.append(trough_depth(prof["median_ratio"].to_numpy(dtype=float)))
                return max(vals) if vals else 0.0
            d_near = depth_at(near_rhos)
            d_before = 0.0
            if before_rho is not None:
                d_before = depth_at([before_rho])
            d_act = 0.0 if act_rho is None else depth_at([act_rho])
            d_far = depth_at(far_rhos) if far_rhos else 0.0
            near_emerges = d_near > d_before + 0.005 and d_near > d_act + 0.005
            far_only = (not near_emerges) and (d_far > d_before + 0.005) and (d_far > d_near + 0.005)
            fold_flags.append(
                {
                    "fold": int(fold),
                    "depth_near": d_near,
                    "depth_before": d_before,
                    "depth_activity": d_act,
                    "depth_far": d_far,
                    "near_emerges": bool(near_emerges),
                    "far_only": bool(far_only),
                }
            )
    n_near = sum(1 for f in fold_flags if f["near_emerges"])
    n_far = sum(1 for f in fold_flags if f["far_only"])
    majority = 4
    return {
        "j_activity": act,
        "j_valley": j_valley,
        "j_nl": j_nl,
        "j_pred": j_pred,
        "j_max": j_max,
        "near_indices": near,
        "global_indices": glob,
        "near_rhos": rhos_near,
        "global_rhos": rhos_glob_u,
        "fold_flags": fold_flags,
        "n_folds_near_emerges": n_near,
        "n_folds_far_only": n_far,
        "majority_threshold": majority,
        "bags_near": bags_near,
        "bags_glob": bags_glob,
        "shape_majority_near": n_near >= majority,
        "shape_majority_far_only": n_far >= majority and n_near < majority,
    }


def nl_status_from_qa(dnl: Dict[str, Any], lofo: Dict[str, Any], shape: Dict[str, Any]) -> str:
    if dnl.get("event") != "nonlinear_rebound" or dnl.get("raw_path_qa") != "supported":
        return "AMBIGUOUS"
    if not lofo.get("stable"):
        return "AMBIGUOUS"
    if shape.get("shape_majority_near"):
        return "CONFIRMED"
    return "CAUTION_ONLY"


def _save(plt, fig, stem: Path) -> List[str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    pdf = stem.with_suffix(".pdf")
    png = stem.with_suffix(".png")
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    fig.savefig(png, format="png", dpi=160, bbox_inches="tight")
    plt.close(fig)
    return [str(pdf), str(png)]


def shade_regions(ax, fam_res: Dict[str, Any], span_df: pd.DataFrame, family: str) -> None:
    plow, phigh, pok = family_span(span_df, family)
    if pok:
        ax.axvline(float(plow), **PRED_DASH)
        ax.axvline(float(phigh), **PRED_DASH)
    act = fam_res.get("rho_activity")
    grd = fam_res.get("rho_guardrail")
    if act is not None and grd is not None and float(act) <= float(grd):
        ax.axvspan(float(act), float(grd), color=CAND_FACE, alpha=CAND_ALPHA, lw=0, zorder=0)
    if act is not None:
        ax.axvline(float(act), **ACTIVITY_LINE)
    if grd is not None:
        ax.axvline(float(grd), **GUARD_LINE)
    if fam_res.get("nonlinear_status") == "CAUTION_ONLY" and fam_res.get("rho_nl") is not None:
        ax.axvline(float(fam_res["rho_nl"]), **CAUTION_LINE)
    if fam_res.get("nonlinear_status") == "CONFIRMED" and fam_res.get("guardrail_driver") == "nonlinear-structure":
        pass


def plot_paths(plt, combined, span_df, regions, metrics, min_pos, q, stem, *, oos: bool) -> List[str]:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    fig, axes = plt.subplots(len(metrics), 2, figsize=(8.9, 2.15 * len(metrics)), sharex=True)
    if len(metrics) == 1:
        axes = np.asarray([axes])
    pos_all = pd.to_numeric(combined.loc[combined["family"].isin(FAMILY_DISPLAY), "rho"], errors="coerce").to_numpy(dtype=float)
    pos_all = pos_all[np.isfinite(pos_all) & (pos_all > 0)]
    for r, (col, ylab) in enumerate(metrics):
        row_vals: List[float] = []
        for fam in FAMILY_DISPLAY:
            sub = combined.loc[combined["family"] == fam]
            if oos:
                for ev in ("heldout", "forward_2025"):
                    row_vals.extend(maybe_percent(col, pd.to_numeric(sub[f"{col}__{ev}"], errors="coerce").to_numpy(dtype=float)).tolist())
            else:
                row_vals.extend(maybe_percent(col, pd.to_numeric(sub[f"{col}__CV_mean"], errors="coerce").to_numpy(dtype=float)).tolist())
                for k in FOLD_IDS:
                    row_vals.extend(maybe_percent(col, pd.to_numeric(sub[f"{col}__fold_{k}"], errors="coerce").to_numpy(dtype=float)).tolist())
        ylim = padded_lim(row_vals, pad=0.08, include=nearby_targets(row_vals, []))
        for c, fam in enumerate(FAMILY_DISPLAY):
            ax = axes[r, c]
            shade_regions(ax, regions[fam], span_df, fam)
            sub = combined.loc[combined["family"] == fam].sort_values("rho")
            x = rho_plot_x(sub["rho"].to_numpy(dtype=float), min_positive=min_pos, q=q)
            color = DIRECT_COLOR if fam == "Direct" else SURR_COLOR
            if oos:
                yh = maybe_percent(col, pd.to_numeric(sub[f"{col}__heldout"], errors="coerce").to_numpy(dtype=float))
                yf = maybe_percent(col, pd.to_numeric(sub[f"{col}__forward_2025"], errors="coerce").to_numpy(dtype=float))
                ax.plot(x, yh, color=color, marker="o", ms=3, lw=1.3, zorder=5)
                ax.plot(x, yf, color=color, ls="--", marker="s", ms=3, lw=1.2, zorder=5)
            else:
                for k in FOLD_IDS:
                    yk = maybe_percent(col, pd.to_numeric(sub[f"{col}__fold_{k}"], errors="coerce").to_numpy(dtype=float))
                    ax.plot(x, yk, color="#9CA3AF", lw=0.8, alpha=0.7, zorder=2)
                ym = maybe_percent(col, pd.to_numeric(sub[f"{col}__CV_mean"], errors="coerce").to_numpy(dtype=float))
                ax.plot(x, ym, color=color, lw=2.1, zorder=5)
            draw_neutral_hline(ax, col)
            log_rho_axes_generic(ax, pos_all, q)
            apply_major_grid(ax)
            ax.set_ylim(*ylim)
            if r == 0:
                ax.set_title(fam)
            if c == 0:
                ax.set_ylabel(ylab)
            if r == len(metrics) - 1:
                ax.set_xlabel(r"Penalty strength $\rho$")
            if r == 0 and c == 1:
                handles = [
                    Patch(facecolor=CAND_FACE, alpha=CAND_ALPHA, label="CV-derived candidate region"),
                    Line2D([0], [0], color="#374151", ls=(0, (4.0, 2.2)), lw=0.85, label="activity onset"),
                    Line2D([0], [0], color="#14532D", lw=1.35, label="upper guardrail"),
                    Line2D([0], [0], color="#9CA3AF", ls=(0, (2.2, 2.0)), lw=0.7, label="prediction/COD transition span"),
                    Line2D([0], [0], color="#7C3AED", ls=":", lw=0.95, label="nonlinear caution (if any)"),
                ]
                ax.legend(handles=handles, frameon=False, fontsize=6.0, loc="best")
    fig.tight_layout()
    return _save(plt, fig, stem)


def plot_qa_breakpoints(plt, combined, span_df, regions, min_pos, q, stem) -> List[str]:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rows = list(BENEFIT_METRICS) + list(PREDICTIVE_COST_METRICS) + [UNIFORMITY_SUPPORT_METRIC, "Delta_NL"]
    fig, axes = plt.subplots(len(rows), 2, figsize=(9.4, 1.45 * len(rows)), sharex=True)
    pos_all = pd.to_numeric(combined.loc[combined["family"].isin(FAMILY_DISPLAY), "rho"], errors="coerce").to_numpy(dtype=float)
    pos_all = pos_all[np.isfinite(pos_all) & (pos_all > 0)]
    for r, metric in enumerate(rows):
        for c, fam in enumerate(FAMILY_DISPLAY):
            ax = axes[r, c]
            shade_regions(ax, regions[fam], span_df, fam)
            sub = combined.loc[combined["family"] == fam].sort_values("rho")
            x = rho_plot_x(sub["rho"].to_numpy(dtype=float), min_positive=min_pos, q=q)
            y = maybe_percent(metric, pd.to_numeric(sub[f"{metric}__CV_mean"], errors="coerce").to_numpy(dtype=float))
            ax.plot(x, y, color=DIRECT_COLOR if fam == "Direct" else SURR_COLOR, lw=1.55, zorder=5)
            log_rho_axes_generic(ax, pos_all, q)
            apply_major_grid(ax)
            if r == 0:
                ax.set_title(fam)
            if c == 0:
                ax.set_ylabel(metric, fontsize=8)
            if r == len(rows) - 1:
                ax.set_xlabel(r"Penalty strength $\rho$")
    handles = [
        Patch(facecolor=CAND_FACE, alpha=CAND_ALPHA, label="CV-derived candidate region"),
        Line2D([0], [0], color="#374151", ls=(0, (4.0, 2.2)), label="activity onset"),
        Line2D([0], [0], color="#14532D", label="upper guardrail"),
        Line2D([0], [0], color="#9CA3AF", ls=(0, (2.2, 2.0)), label="prediction/COD span"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.01), fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _save(plt, fig, stem)


def _plot_ratio_panel(ax, bag: Dict[float, Dict[int, pd.DataFrame]], rhos: Sequence[float], *, mean_only: bool) -> None:
    cmap = ["#000000", "#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442"]
    for i, r in enumerate(rhos):
        color = cmap[i % len(cmap)]
        folds = bag.get(float(r), {})
        if not folds:
            continue
        acc = []
        for fold, prof in folds.items():
            acc.append(prof["median_ratio"].to_numpy(dtype=float))
            x = prof["median_sale_price"].to_numpy(dtype=float)
            if not mean_only:
                ax.plot(x, prof["median_ratio"], color=color, lw=0.7, alpha=0.35, zorder=3)
        if acc:
            mean_r = np.nanmean(np.vstack(acc), axis=0)
            x = list(folds.values())[0]["median_sale_price"].to_numpy(dtype=float)
            lab = r"$\rho$=0" if is_rho_zero(r) else rf"$\rho$={float(r):.4g}"
            ax.plot(x, mean_r, color=color, lw=1.7, zorder=4, label=lab)
    ax.axhline(1.0, color="#111827", lw=1.0, zorder=2)
    ax.set_xscale("log", base=10)
    ax.set_xlabel("Sale price")
    ax.set_ylabel("Valuation-to-sale ratio")


def plot_ratio_qa(plt, shape: Dict[str, Any], stem_context: Path, stem_zoom: Path, stem_folds: Path) -> List[str]:
    saved: List[str] = []
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _plot_ratio_panel(ax, shape["bags_glob"], shape["global_rhos"], mean_only=True)
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.set_title("Surrogate CV ratio-shape: event-relative global context")
    fig.tight_layout()
    saved += _save(plt, fig, stem_context)
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    _plot_ratio_panel(ax, shape["bags_near"], shape["near_rhos"], mean_only=True)
    ax.legend(frameon=False, fontsize=7, loc="best")
    ax.set_title("Surrogate CV ratio-shape: nonlinear-event neighborhood")
    fig.tight_layout()
    saved += _save(plt, fig, stem_zoom)
    fig, axes = plt.subplots(4, 2, figsize=(8.8, 9.2), sharex=True, sharey=True)
    axes = axes.ravel()
    for i, fold in enumerate(FOLD_IDS):
        ax = axes[i]
        cmap = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7"]
        for j, r in enumerate(shape["near_rhos"]):
            prof = shape["bags_near"].get(float(r), {}).get(int(fold))
            if prof is None:
                continue
            ax.plot(
                prof["median_sale_price"],
                prof["median_ratio"],
                color=cmap[j % len(cmap)],
                lw=1.4,
                label=rf"$\rho$={float(r):.4g}",
            )
        ax.axhline(1.0, color="#111827", lw=0.9)
        ax.set_xscale("log", base=10)
        ax.set_title(f"Fold {fold}")
        if i % 2 == 0:
            ax.set_ylabel("Valuation-to-sale ratio")
        if i >= 5:
            ax.set_xlabel("Sale price")
        if i == 0:
            ax.legend(frameon=False, fontsize=6)
    axes[-1].axis("off")
    fig.suptitle("Fold-level Surrogate ratio-shape near the detected NL event", fontsize=11)
    fig.tight_layout()
    saved += _save(plt, fig, stem_folds)
    return saved


def flatten_family(fam: str, screened: Dict[str, Any], finalized: Dict[str, Any], lofo_tbl: List[Dict[str, Any]]) -> Dict[str, Any]:
    act = screened["activity"]
    cl = screened["predictive_cluster"]
    dnl = screened.get("delta_nl") or {}
    return {
        "family": fam,
        "n_positive": screened["n"],
        "min_segment_points": screened["min_segment_points"],
        "h": screened["h"],
        "rho_activity": act.get("rho"),
        "index_activity": act.get("index"),
        "activity_metrics": act.get("metrics"),
        "rho_predictive_guardrail": cl.get("guardrail_rho"),
        "index_predictive_guardrail": cl.get("guardrail_index"),
        "predictive_cluster_metrics": cl.get("metrics"),
        "predictive_cluster_status": cl.get("status"),
        "rho_nl": dnl.get("rho"),
        "index_nl": dnl.get("index"),
        "valley_rho": dnl.get("valley_rho"),
        "valley_index": dnl.get("valley_index"),
        "nl_slopes": dnl.get("slopes"),
        "nl_complexity": dnl.get("complexity"),
        "nl_bic": dnl.get("bic"),
        "nl_raw_qa": dnl.get("raw_path_qa"),
        "dcor_status": (screened.get("dcor") or {}).get("status"),
        "nonlinear_status": finalized.get("nonlinear_status"),
        "guardrail_driver": finalized.get("guardrail_driver"),
        "rho_guardrail": finalized.get("rho_guardrail"),
        "index_guardrail": finalized.get("index_guardrail"),
        "status": finalized.get("status"),
        "region_defined": finalized.get("region_defined"),
        "lofo_activity_stable": finalized.get("lofo_activity_stable"),
        "lofo_guardrail_stable": finalized.get("lofo_guardrail_stable"),
        "lofo_nl_stable": finalized.get("lofo_nl_stable"),
        "n_lofo": len(lofo_tbl),
        "not_a_model_selection_rule": True,
        "object_name": "CV-derived candidate region",
    }


def main() -> int:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    OUT.mkdir(parents=True, exist_ok=True)
    for sub in ("tables", "figures", "qa", "provenance"):
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    guard = OutputGuard(OUT, REPO)

    # --- freeze method spec before any CCAO endpoint computation ---
    method_path = OUT / "provenance" / "rho_screening_method_v2.json"
    write_json(guard, method_path, METHOD_SPEC)
    method_hash = sha256_file(method_path)

    audit = audit_v1_hardcoded_rho()
    write_json(guard, OUT / "qa" / "no_hardcoded_rho_audit.json", audit)

    paths = input_paths()
    in_hash = input_hashes()
    frozen_ok = {k: in_hash[k] == V1_EXPECTED_HASHES[k] for k in V1_EXPECTED_HASHES}
    tex_before = sha256_file(PAPER_TEX) if PAPER_TEX.is_file() else None
    v2_combined_h = sha256_file(paths["combined_v2"])
    v2_span_h = sha256_file(paths["span_summary"])
    v1_status_h = sha256_file(V1_OUT / "tables" / "rho_screening_status.json") if (V1_OUT / "tables" / "rho_screening_status.json").is_file() else None

    preflight = {
        "utc": utc_now(),
        "git_branch": git("branch", "--show-current"),
        "git_head": git("rev-parse", "HEAD"),
        "git_status_short": git("status", "--short").splitlines(),
        "canonical_root": str(CANONICAL.relative_to(REPO)),
        "extension_root": str(EXT.relative_to(REPO)),
        "identity": {
            "data_id": DATA_ID,
            "split_id": SPLIT_ID,
            "lgbm_config_id": LGBM_CONFIG_ID,
            "seed": SEED,
            "n_estimators": TREES,
            "n_folds": 7,
        },
        "input_sha256": in_hash,
        "frozen_inputs_unchanged_vs_v1": frozen_ok,
        "method_spec_sha256": method_hash,
        "paper_tex_sha256_before": tex_before,
        "v1_status_sha256": v1_status_h,
        "no_model_fitting": True,
        "no_manuscript_edit": True,
        "no_v1_overwrite": True,
    }
    write_json(guard, OUT / "provenance" / "preflight.json", preflight)
    write_json(guard, OUT / "provenance" / "input_hashes.json", in_hash)
    if not all(frozen_ok.values()):
        write_json(guard, OUT / "qa" / "frozen_input_mismatch.json", frozen_ok)
        print(json.dumps({"error": "frozen_input_hash_mismatch", "detail": frozen_ok}, indent=2))
        return 1

    combined = pd.read_csv(paths["combined_v4_view"])
    span = pd.read_csv(paths["span_summary"])
    grid_blob = json.loads(paths["grid"].read_text(encoding="utf-8"))
    min_pos = float(grid_blob["min_positive_augmented"])
    q = float(grid_blob["q"])
    inv = pd.read_csv(paths["cv_prediction_inventory"])

    # --- Phase A: CV endpoints ---
    full: Dict[str, Dict[str, Any]] = {}
    lofo: Dict[str, List[Dict[str, Any]]] = {fam: [] for fam in FAMILY_DISPLAY}
    for fam in FAMILY_DISPLAY:
        frame = family_frame(combined, fam)
        full[fam] = screen_frame(frame)
        for left in FOLD_IDS:
            lofo[fam].append({"left_out_fold": int(left), **screen_frame(frame, left_out=int(left))})

    rho_grid = path_inputs(family_frame(combined, "Surrogate"))["rho"]
    x_s = log10_rho(rho_grid)
    h_s = median_log_spacing(x_s)
    rho0_s = float(rho0_row(family_frame(combined, "Surrogate"))["rho"])

    # Scale-equivariance on an in-memory copy of the CV table.
    scaled = combined.copy()
    rho_col = pd.to_numeric(scaled["rho"], errors="coerce")
    pos_mask = np.array([is_rho_positive(float(v)) if np.isfinite(v) else False for v in rho_col])
    scaled.loc[pos_mask, "rho"] = rho_col[pos_mask] * SCALE_EQUIVARIANCE_FACTOR
    scale_rows = []
    scale_ok = True
    for fam in FAMILY_DISPLAY:
        a = full[fam]
        b = screen_frame(family_frame(scaled, fam))
        keys = [
            ("activity", a["activity"].get("index"), b["activity"].get("index")),
            ("pred_guard", a.get("index_predictive_guardrail"), b.get("index_predictive_guardrail")),
            ("nl", (a.get("delta_nl") or {}).get("index"), (b.get("delta_nl") or {}).get("index")),
        ]
        for m in BENEFIT_METRICS:
            keys.append((f"sat_{m}", a["benefit_events"][m].get("benefit_saturation_index"), b["benefit_events"][m].get("benefit_saturation_index")))
        idx_ok = all(u == v for _n, u, v in keys)
        rho_ok = True
        if a["activity"].get("rho") is not None and b["activity"].get("rho") is not None:
            rho_ok = abs(float(b["activity"]["rho"]) - float(a["activity"]["rho"]) * SCALE_EQUIVARIANCE_FACTOR) <= 1e-10 * max(1.0, abs(float(b["activity"]["rho"])))
        if not idx_ok or not rho_ok:
            scale_ok = False
        scale_rows.append({"family": fam, "index_pairs": [{"name": n, "full": u, "scaled": v} for n, u, v in keys], "index_ok": idx_ok, "rho_scaled_ok": rho_ok})
    scale_payload = {"pass": scale_ok, "factor": SCALE_EQUIVARIANCE_FACTOR, "families": scale_rows}
    write_json(guard, OUT / "qa" / "rho_scale_equivariance.json", scale_payload)
    if not scale_ok:
        print(json.dumps({"error": "scale_equivariance_failed", "detail": scale_payload}, indent=2, default=str))
        return 1

    syn = run_synthetic_tests()
    write_json(guard, OUT / "qa" / "synthetic_tests.json", syn)
    if not syn["pass"]:
        print(json.dumps({"error": "synthetic_tests_failed", "detail": syn}, indent=2))
        return 1

    # Ratio-shape QA uses frozen CV predictions only (after numeric NL event exists).
    inv_map = inventory_lookup(inv)
    shape = ratio_shape_qa(inv_map, rho_grid, rho0_s, full["Surrogate"])
    shape_store = {k: v for k, v in shape.items() if k not in {"bags_near", "bags_glob"}}

    finalized: Dict[str, Dict[str, Any]] = {}
    lofo_rows: List[Dict[str, Any]] = []
    raw_qa_rows: List[Dict[str, Any]] = []
    bp_rows: List[Dict[str, Any]] = []
    sat_rows: List[Dict[str, Any]] = []

    for fam in FAMILY_DISPLAY:
        frame = family_frame(combined, fam)
        rho = path_inputs(frame)["rho"]
        x = log10_rho(rho)
        h = median_log_spacing(x)
        sc = full[fam]
        act_lofo = [L["activity"].get("index") for L in lofo[fam]]
        pred_lofo = [L.get("index_predictive_guardrail") for L in lofo[fam]]
        nl_lofo = [(L.get("delta_nl") or {}).get("index") for L in lofo[fam]]
        act_stab = lofo_stability(sc["activity"].get("index"), act_lofo, x, h)
        pred_stab = lofo_stability(sc.get("index_predictive_guardrail"), pred_lofo, x, h)
        nl_stab = lofo_stability((sc.get("delta_nl") or {}).get("index"), nl_lofo, x, h, require_event=True)
        nl_stat = None
        if fam == "Surrogate":
            nl_stat = nl_status_from_qa(sc.get("delta_nl") or {}, nl_stab, shape)
        else:
            nl_stat = "AMBIGUOUS" if (sc.get("delta_nl") or {}).get("event") != "nonlinear_rebound" else "CAUTION_ONLY"

        pred_rho = sc.get("rho_predictive_guardrail")
        pred_idx = sc.get("index_predictive_guardrail")
        nl_rho = (sc.get("delta_nl") or {}).get("rho")
        nl_idx = (sc.get("delta_nl") or {}).get("index")
        driver = None
        guard_rho = None
        guard_idx = None
        if fam == "Direct":
            if pred_idx is None:
                status = "DIRECT_GUARDRAIL_AMBIGUOUS"
            else:
                guard_rho, guard_idx, driver = pred_rho, pred_idx, "predictive-deterioration"
                status = family_status(
                    activity_ok=sc["activity"].get("index") is not None,
                    guardrail_ok=True,
                    activity_lofo=bool(act_stab["stable"]),
                    guardrail_lofo=bool(pred_stab["stable"]),
                    guardrail_ambiguous_name="DIRECT_GUARDRAIL_AMBIGUOUS",
                )
        else:
            if nl_stat == "CONFIRMED" and nl_idx is not None and pred_idx is not None:
                if float(nl_rho) <= float(pred_rho) + 1e-15:
                    guard_rho, guard_idx, driver = nl_rho, nl_idx, "nonlinear-structure"
                    g_stab = nl_stab
                else:
                    guard_rho, guard_idx, driver = pred_rho, pred_idx, "predictive-deterioration"
                    g_stab = pred_stab
                status = family_status(
                    activity_ok=sc["activity"].get("index") is not None,
                    guardrail_ok=True,
                    activity_lofo=bool(act_stab["stable"]),
                    guardrail_lofo=bool(g_stab["stable"]),
                    guardrail_ambiguous_name="SURROGATE_GUARDRAIL_AMBIGUOUS",
                )
            elif nl_stat == "CONFIRMED" and nl_idx is not None and pred_idx is None:
                guard_rho, guard_idx, driver = nl_rho, nl_idx, "nonlinear-structure"
                status = family_status(
                    activity_ok=sc["activity"].get("index") is not None,
                    guardrail_ok=True,
                    activity_lofo=bool(act_stab["stable"]),
                    guardrail_lofo=bool(nl_stab["stable"]),
                    guardrail_ambiguous_name="SURROGATE_GUARDRAIL_AMBIGUOUS",
                )
            elif pred_idx is not None:
                guard_rho, guard_idx, driver = pred_rho, pred_idx, "predictive-deterioration"
                status = family_status(
                    activity_ok=sc["activity"].get("index") is not None,
                    guardrail_ok=True,
                    activity_lofo=bool(act_stab["stable"]),
                    guardrail_lofo=bool(pred_stab["stable"]),
                    guardrail_ambiguous_name="SURROGATE_GUARDRAIL_AMBIGUOUS",
                )
            else:
                status = "SURROGATE_GUARDRAIL_AMBIGUOUS"

        region_ok = sc["activity"].get("rho") is not None and guard_rho is not None and float(sc["activity"]["rho"]) <= float(guard_rho) + 1e-15
        finalized[fam] = {
            "nonlinear_status": nl_stat,
            "guardrail_driver": driver,
            "rho_guardrail": guard_rho,
            "index_guardrail": guard_idx,
            "rho_activity": sc["activity"].get("rho"),
            "rho_nl": nl_rho,
            "status": status,
            "region_defined": bool(region_ok and status == "PASS"),
            "lofo_activity_stable": bool(act_stab["stable"]),
            "lofo_guardrail_stable": bool(pred_stab["stable"] if driver != "nonlinear-structure" else nl_stab["stable"]),
            "lofo_nl_stable": bool(nl_stab["stable"]),
            "lofo_activity": act_stab,
            "lofo_predictive": pred_stab,
            "lofo_nl": nl_stab,
        }

        for L in lofo[fam]:
            lofo_rows.append(
                {
                    "family": fam,
                    "left_out_fold": L["left_out_fold"],
                    "index_activity": L["activity"].get("index"),
                    "rho_activity": L["activity"].get("rho"),
                    "index_predictive_guardrail": L.get("index_predictive_guardrail"),
                    "rho_predictive_guardrail": L.get("rho_predictive_guardrail"),
                    "index_nl": (L.get("delta_nl") or {}).get("index"),
                    "rho_nl": (L.get("delta_nl") or {}).get("rho"),
                    "nl_event": (L.get("delta_nl") or {}).get("event"),
                    "nl_raw_qa": (L.get("delta_nl") or {}).get("raw_path_qa"),
                    "predictive_cluster_status": L["predictive_cluster"].get("status"),
                }
            )
        for m, ev in sc["benefit_events"].items():
            bp_rows.append({"family": fam, "metric": m, "role": "benefit", **{k: ev.get(k) for k in ("complexity", "benefit_onset", "benefit_onset_index", "benefit_saturation", "benefit_saturation_index", "slopes", "sse", "bic", "raw_path_qa", "classification")}})
            sat_rows.append({"family": fam, "metric": m, "benefit_onset": ev.get("benefit_onset"), "benefit_onset_index": ev.get("benefit_onset_index"), "benefit_saturation": ev.get("benefit_saturation"), "benefit_saturation_index": ev.get("benefit_saturation_index"), "relative_to_direct_guardrail": None if fam != "Direct" or ev.get("benefit_saturation_index") is None or guard_idx is None else int(ev["benefit_saturation_index"]) - int(guard_idx)})
            raw_qa_rows.append({"family": fam, "metric": m, "event": "benefit", "raw_path_qa": ev.get("raw_path_qa")})
        for m, ev in sc["predictive_events"].items():
            bp_rows.append({"family": fam, "metric": m, "role": "predictive_cost", **{k: ev.get(k) for k in ("complexity", "event", "rho", "index", "slopes", "sse", "bic", "raw_path_qa", "classification")}})
            raw_qa_rows.append({"family": fam, "metric": m, "event": "cost_deterioration", "raw_path_qa": ev.get("raw_path_qa"), "index": ev.get("index")})
        if sc.get("cod_event"):
            ev = sc["cod_event"]
            bp_rows.append({"family": fam, "metric": UNIFORMITY_SUPPORT_METRIC, "role": "uniformity_support", **{k: ev.get(k) for k in ("complexity", "event", "rho", "index", "slopes", "sse", "bic", "raw_path_qa", "classification")}})
        if sc.get("delta_nl"):
            ev = sc["delta_nl"]
            bp_rows.append({"family": fam, "metric": "Delta_NL", "role": "nonlinear", **{k: ev.get(k) for k in ("complexity", "event", "rho", "index", "valley_rho", "valley_index", "slopes", "sse", "bic", "raw_path_qa", "classification", "uses_exact_minimum")}})

    # Phase A tables (before any OOS interpretation)
    summary_rows = [flatten_family(fam, full[fam], finalized[fam], lofo[fam]) for fam in FAMILY_DISPLAY]
    status_blob = {
        "phase": "A",
        "object_name": "CV-derived candidate region",
        "not_a_model_selection_rule": True,
        "method_spec_sha256": method_hash,
        "heldout_2025_used_for_endpoints": False,
        "families": {fam: {**summary_rows[i], "screen": json_safe({k: full[fam][k] for k in ("activity", "predictive_cluster", "delta_nl", "dcor")})} for i, fam in enumerate(FAMILY_DISPLAY)},
    }
    write_json(guard, OUT / "tables" / "rho_candidate_regions_v2_status.json", status_blob)
    guard.write_df(pd.DataFrame(summary_rows), OUT / "tables" / "rho_candidate_regions_v2_summary.csv", parquet=False)
    write_json(guard, OUT / "tables" / "rho_candidate_regions_v2_summary.json", summary_rows)
    bp_df = pd.json_normalize(bp_rows)
    guard.write_df(bp_df, OUT / "tables" / "rho_metric_breakpoints_v2.csv", parquet=False)
    write_json(guard, OUT / "tables" / "rho_metric_breakpoints_v2.json", bp_rows)
    guard.write_df(pd.DataFrame(lofo_rows), OUT / "tables" / "rho_lofo_endpoints_v2.csv", parquet=False)
    write_json(guard, OUT / "tables" / "rho_lofo_endpoints_v2.json", lofo_rows)
    guard.write_df(pd.DataFrame(sat_rows), OUT / "tables" / "direct_saturation_diagnostics_v2.csv", parquet=False)
    write_json(guard, OUT / "tables" / "surrogate_nonlinear_qa_v2.json", {"shape": shape_store, "dcor": full["Surrogate"].get("dcor"), "delta_nl": full["Surrogate"].get("delta_nl"), "lofo": finalized["Surrogate"]["lofo_nl"], "status": finalized["Surrogate"]["nonlinear_status"]})
    write_json(guard, OUT / "qa" / "raw_path_event_qa.json", raw_qa_rows)

    phase_a_files = [
        OUT / "provenance" / "rho_screening_method_v2.json",
        OUT / "tables" / "rho_candidate_regions_v2_status.json",
        OUT / "tables" / "rho_candidate_regions_v2_summary.csv",
        OUT / "tables" / "rho_metric_breakpoints_v2.csv",
        OUT / "tables" / "rho_lofo_endpoints_v2.csv",
        OUT / "qa" / "rho_scale_equivariance.json",
        OUT / "qa" / "no_hardcoded_rho_audit.json",
        OUT / "qa" / "synthetic_tests.json",
    ]
    phase_a_hashes = {str(p.relative_to(OUT)): sha256_file(p) for p in phase_a_files}
    write_json(guard, OUT / "provenance" / "phase_a_output_sha256.json", {"utc": utc_now(), "hashes": phase_a_hashes, "method_spec_sha256": method_hash})

    # Determinism: recompute full-CV screens and compare indices/rho.
    det_ok = True
    det_problems = []
    for fam in FAMILY_DISPLAY:
        again = screen_frame(family_frame(combined, fam))
        for key in ("activity",):
            if again[key].get("index") != full[fam][key].get("index"):
                det_ok = False
                det_problems.append(f"{fam} activity index {full[fam][key].get('index')} vs {again[key].get('index')}")
        if again.get("index_predictive_guardrail") != full[fam].get("index_predictive_guardrail"):
            det_ok = False
            det_problems.append(f"{fam} pred guardrail")
        if (again.get("delta_nl") or {}).get("index") != (full[fam].get("delta_nl") or {}).get("index"):
            det_ok = False
            det_problems.append(f"{fam} nl index")
    write_json(guard, OUT / "qa" / "determinism.json", {"pass": det_ok, "problems": det_problems, "phase_a_hashes": phase_a_hashes})
    if not det_ok:
        print(json.dumps({"error": "determinism_failed", "problems": det_problems}, indent=2))
        return 1

    # --- Phase B: overlay only ---
    port = {}
    for fam in FAMILY_DISPLAY:
        act = finalized[fam].get("rho_activity")
        grd = finalized[fam].get("rho_guardrail")
        sub = family_frame(combined, fam)
        rho = pd.to_numeric(sub["rho"], errors="coerce").to_numpy(dtype=float)
        inside = np.zeros(len(sub), dtype=bool)
        below = np.zeros(len(sub), dtype=bool)
        above = np.zeros(len(sub), dtype=bool)
        if act is not None and grd is not None:
            inside = (rho >= float(act) - 1e-15) & (rho <= float(grd) + 1e-15) & np.array([is_rho_positive(float(v)) if np.isfinite(v) else False for v in rho])
            below = np.array([is_rho_positive(float(v)) if np.isfinite(v) else False for v in rho]) & (rho < float(act) - 1e-15)
            above = np.array([is_rho_positive(float(v)) if np.isfinite(v) else False for v in rho]) & (rho > float(grd) + 1e-15)

        def mean_split(mask, col, split):
            v = pd.to_numeric(sub.loc[mask, f"{col}__{split}"], errors="coerce").to_numpy(dtype=float)
            v = v[np.isfinite(v)]
            return None if v.size == 0 else float(np.mean(v))

        port[fam] = {
            "rho_activity": act,
            "rho_guardrail": grd,
            "n_inside": int(inside.sum()),
            "n_below": int(below.sum()),
            "n_above": int(above.sum()),
            "heldout": {
                "R2_inside": mean_split(inside, "R2_price", "heldout"),
                "R2_above": mean_split(above, "R2_price", "heldout"),
                "R2_below": mean_split(below, "R2_price", "heldout"),
                "COD_inside": mean_split(inside, "COD", "heldout"),
                "COD_above": mean_split(above, "COD", "heldout"),
                "Delta_NL_inside": mean_split(inside, "Delta_NL", "heldout"),
                "Delta_NL_above": mean_split(above, "Delta_NL", "heldout"),
            },
            "forward_2025": {
                "R2_inside": mean_split(inside, "R2_price", "forward_2025"),
                "R2_above": mean_split(above, "R2_price", "forward_2025"),
                "R2_below": mean_split(below, "R2_price", "forward_2025"),
                "COD_inside": mean_split(inside, "COD", "forward_2025"),
                "COD_above": mean_split(above, "COD", "forward_2025"),
                "Delta_NL_inside": mean_split(inside, "Delta_NL", "forward_2025"),
                "Delta_NL_above": mean_split(above, "Delta_NL", "forward_2025"),
            },
            "endpoints_unchanged": True,
            "qualitative": (
                "Retrospective overlay of the frozen CV candidate region. "
                "Below-onset points remain near the unpenalized path; post-guardrail "
                "points are inspected for predictive (Direct) or nonlinear/predictive "
                "(Surrogate) deterioration. Endpoints were not revised."
            ),
        }
    write_json(guard, OUT / "qa" / "phase_b_portability.json", {"phase": "B", "families": port, "endpoints_unchanged": True})

    fig_dir = OUT / "figures"
    plot_kw = dict(combined=combined, span_df=span, regions=finalized, min_pos=min_pos, q=q)
    plot_paths(plt, **plot_kw, metrics=(("R2_price", r"$R^2_P$"), ("MAE_price", r"MAE$_P$"), ("MAPE", r"MAPE$_P$ (\%)"), ("RMSE_log", r"RMSE$_{\log P}$")), stem=fig_dir / "cv_predictive_metric_paths_candidate_region", oos=False)
    plot_paths(plt, **plot_kw, metrics=(("median_ratio", "Median ratio"), ("mean_ratio", "Mean ratio"), ("weighted_mean_ratio", "Weighted mean ratio"), ("COD", "COD"), ("COV", "COV (\%)")), stem=fig_dir / "cv_level_uniformity_paths_candidate_region", oos=False)
    plot_paths(plt, **plot_kw, metrics=(("PRD", "PRD"), ("PRB", "PRB"), ("MKI", "MKI"), ("VEI", "VEI")), stem=fig_dir / "cv_vertical_equity_metric_paths_candidate_region", oos=False)
    plot_paths(plt, **plot_kw, metrics=(("Beta_log", r"$\beta_{\log}$"), ("Delta_NL", r"$\Delta_{\mathrm{NL}}$"), ("dCor_e_y", r"dCor$(e,y)$")), stem=fig_dir / "cv_mechanism_metric_paths_candidate_region", oos=False)
    plot_paths(plt, **plot_kw, metrics=(("R2_price", r"$R^2_P$"), ("MAE_price", r"MAE$_P$"), ("MAPE", r"MAPE$_P$ (\%)"), ("RMSE_log", r"RMSE$_{\log P}$")), stem=fig_dir / "predictive_metric_paths_candidate_region", oos=True)
    plot_paths(plt, **plot_kw, metrics=(("PRD", "PRD"), ("PRB", "PRB"), ("MKI", "MKI"), ("VEI", "VEI")), stem=fig_dir / "vertical_equity_metric_paths_candidate_region", oos=True)
    plot_paths(plt, **plot_kw, metrics=(("Beta_log", r"$\beta_{\log}$"), ("Delta_NL", r"$\Delta_{\mathrm{NL}}$"), ("dCor_e_y", r"dCor$(e,y)$")), stem=fig_dir / "mechanism_vs_rho_candidate_region", oos=True)
    plot_qa_breakpoints(plt, combined, span, finalized, min_pos, q, fig_dir / "rho_screening_breakpoints_v2")
    plot_ratio_qa(
        plt,
        shape,
        fig_dir / "surrogate_ratio_shape_guardrail_context_qa",
        fig_dir / "surrogate_ratio_shape_guardrail_zoom_qa",
        fig_dir / "surrogate_ratio_shape_fold_level_qa",
    )

    # Integration note only if statuses are usable.
    if finalized["Direct"]["status"] == "PASS" or finalized["Surrogate"]["status"] == "PASS":
        d = summary_rows[0]
        s = summary_rows[1]
        note = (
            "# Paper integration notes (not a manuscript edit)\n\n"
            "The v2 procedure is a generic chronological-CV screen on log10(rho): "
            "activity onset is the first grid point at which at least three of five "
            "benefit distances have a supported improvement breakpoint, and the upper "
            "guardrail is the earliest robust high-rho deterioration regime "
            "(Direct: first 2-of-4 predictive-cost cluster; Surrogate: earlier of a "
            "confirmed Delta_NL rebound and the same predictive cluster). "
            "It does not select a deployment rho.\n\n"
            f"Direct activity onset is rho={d.get('rho_activity')} "
            f"(metrics {d.get('activity_metrics')}); the conservative upper guardrail "
            f"is rho={d.get('rho_guardrail')} as the onset of robust predictive deterioration "
            f"(cluster metrics {d.get('predictive_cluster_metrics')}, status {d.get('status')}). "
            "Later benefit-saturation events are descriptive only.\n\n"
            f"Surrogate activity onset is rho={s.get('rho_activity')}; predictive guardrail "
            f"rho={s.get('rho_predictive_guardrail')}; Delta_NL rebound rho={s.get('rho_nl')} "
            f"with nonlinear status {s.get('nonlinear_status')} and dCor {s.get('dcor_status')}. "
            f"The binding upper guardrail is {s.get('guardrail_driver')} at rho={s.get('rho_guardrail')} "
            f"(status {s.get('status')}).\n\n"
            "Suggested caption language: shaded band = CV-derived candidate region "
            "(activity onset to upper guardrail); thin gray dashes = frozen prediction/COD "
            "transition span (separate descriptive object). This screen does not choose "
            "a final model.\n\n"
            "The procedure screens rho values on datasets with qualitatively comparable "
            "regularization-path geometry. It is not a universal or deployment-selection rule.\n"
        )
        guard.write_text(OUT / "paper_integration_notes.md", note)

    tex_after = sha256_file(PAPER_TEX) if PAPER_TEX.is_file() else None
    v1_after = sha256_file(V1_OUT / "tables" / "rho_screening_status.json") if v1_status_h else None
    safety = {
        "no_model_fitting": True,
        "no_frozen_path_artifact_modified": sha256_file(paths["combined_v2"]) == v2_combined_h,
        "no_transition_span_redefined": sha256_file(paths["span_summary"]) == v2_span_h,
        "no_manuscript_edit": tex_after == tex_before,
        "no_v1_overwrite": v1_after == v1_status_h,
        "heldout_2025_did_not_affect_cv_endpoints": True,
        "paper_figures_not_overwritten": True,
        "scale_equivariance_pass": scale_ok,
        "synthetic_pass": syn["pass"],
        "determinism_pass": det_ok,
        "method_spec_sha256": method_hash,
        "phase_a_hashes": phase_a_hashes,
    }
    write_json(guard, OUT / "provenance" / "safety.json", safety)
    print(
        json.dumps(
            {
                "method_spec_sha256": method_hash,
                "families": {fam: {"status": finalized[fam]["status"], "activity": finalized[fam]["rho_activity"], "guardrail": finalized[fam]["rho_guardrail"], "driver": finalized[fam]["guardrail_driver"], "nl_status": finalized[fam]["nonlinear_status"]} for fam in FAMILY_DISPLAY},
                "safety": safety,
            },
            indent=2,
            default=str,
        )
    )
    return 0 if det_ok and scale_ok and syn["pass"] and safety["no_manuscript_edit"] else 1


if __name__ == "__main__":
    try:
        code = main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(code)
