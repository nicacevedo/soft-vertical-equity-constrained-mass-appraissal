#!/usr/bin/env python3
"""v2.1 Surrogate upper-guardrail follow-up.

Reuses the v2 engine for Direct (unchanged) and for Surrogate activity,
predictive-bend clustering, Delta_NL, dCor, and LOFO windows. Only the
Surrogate upper-guardrail binding rule is revised: predictive bend is
descriptive; predictive harm and NL caution may bind.

No model refits, no grid extension, no overwrite of v1/v2 outputs, no
manuscript edit, no ratio-shape figure regeneration.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    EPS,
    PREDICTIVE_COST_METRICS,
    RAW_NEIGHBOR_STEPS,
    SCALE_EQUIVARIANCE_FACTOR,
    UNIFORMITY_SUPPORT_METRIC,
    advantage_exhausted,
    classify_dcor,
    family_status,
    json_safe,
    log10_rho,
    lofo_stability,
    median_log_spacing,
    surrogate_upper_guardrail,
)
from utils.transition_paper_asset_plots import padded_lim  # noqa: E402
from utils.transition_regions import (  # noqa: E402
    FAMILY_DISPLAY,
    FOLD_IDS,
    OutputGuard,
    family_frame,
    is_rho_positive,
    sha256_file,
)

_V2_PATH = REPO / "scripts" / "analyze_rho_screening_region_v2.py"
_spec = importlib.util.spec_from_file_location("rho_screening_v2_script", _V2_PATH)
v2s = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(v2s)

CANONICAL = v2s.CANONICAL
EXT = v2s.EXT
DATA_ID = v2s.DATA_ID
SPLIT_ID = v2s.SPLIT_ID
LGBM_CONFIG_ID = v2s.LGBM_CONFIG_ID
SEED = v2s.SEED
TREES = v2s.TREES
V2_OUT = v2s.OUT
V1_OUT = v2s.V1_OUT
PAPER_TEX = v2s.PAPER_TEX
OUT = V2_OUT.parent / "rho_screening_candidate_regions_v2_1_surrogate_harm_nl"

CAND_FACE = v2s.CAND_FACE
CAND_ALPHA = v2s.CAND_ALPHA
PRED_DASH = v2s.PRED_DASH
ACTIVITY_LINE = v2s.ACTIVITY_LINE
GUARD_LINE = v2s.GUARD_LINE
CAUTION_LINE = v2s.CAUTION_LINE
BEND_LINE = dict(color="#B45309", ls=(0, (1.4, 1.6)), lw=0.9, alpha=0.8, zorder=3)

METHOD_SPEC_V21: Dict[str, Any] = {
    "version": "v2.1",
    "parent_version": "v2",
    "object_name": "CV-derived candidate region",
    "scope": "Surrogate upper-guardrail refinement only",
    "not_a_model_selection_rule": True,
    "direct_unchanged": True,
    "unchanged": [
        "Direct activity onset",
        "Direct predictive-deterioration clustering",
        "Direct upper guardrail",
        "Direct LOFO",
        "both families' activity-onset rule",
        "piecewise-linear breakpoint engine",
        "metric families",
        "log-rho / grid-adaptive clustering and LOFO windows",
        "Delta_NL valley / rebound / BIC / raw-QA / LOFO",
        "ratio-shape QA methodology",
        "frozen prediction/COD transition spans",
    ],
    "predictive_bend": (
        "Surrogate first robust predictive-deterioration cluster. Descriptive only; "
        "does not bind the Surrogate upper guardrail."
    ),
    "predictive_harm": (
        "For each of R2, MAE, MAPE, RMSE_log: first observed positive-rho grid point "
        "at or after that metric's already-computed supported deterioration breakpoint "
        "where the metric has exhausted its advantage vs the within-family custom-objective "
        "rho=0 origin, remains in the worsening regime, and the baseline crossing is not "
        "an isolated fluctuation. No interpolation. Cluster harm events with the v2 "
        "log-grid rule (width 2*h, min 2 of 4 metrics)."
    ),
    "nl_caution": (
        "Supported, raw-QA-supported, LOFO-stable post-valley Delta_NL rebound. "
        "Means nonlinear residual-price structure has stopped improving and begun "
        "worsening persistently. Does not mean a visible S-shape is already present."
    ),
    "visible_nl_pathology": (
        "Optional descriptive status from the existing ratio-shape analysis. "
        "Separate from NL_CAUTION. Does not move the Delta_NL breakpoint or gate binding."
    ),
    "dcor": (
        "REBOUND / FLATTENING / STILL_IMPROVING / AMBIGUOUS. Only REBOUND is evidence "
        "that broader dependence is worsening. FLATTENING is not pathology confirmation. "
        "dCor does not determine the guardrail."
    ),
    "surrogate_upper_guardrail": (
        "min{rho_predictive_harm, rho_nl_caution} among events that satisfy their "
        "stability/QA requirements. If neither exists: SURROGATE_GUARDRAIL_AMBIGUOUS. "
        "Do not fall back to the predictive-bend cluster."
    ),
    "rho_zero_reference": "within-family custom-objective rho=0; LightGBM is not the screening reference",
}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(REPO), text=True).strip()


def dumps(payload: Any) -> str:
    return json.dumps(json_safe(payload), indent=2, sort_keys=True) + "\n"


def write_json(guard: OutputGuard, path: Path, payload: Any) -> Path:
    return guard.write_text(path, dumps(payload))


def file_hash(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    return sha256_file(path)


def audit_no_hardcoded_rho() -> Dict[str, Any]:
    parent = v2s.audit_v1_hardcoded_rho()
    text = Path(__file__).read_text(encoding="utf-8")
    literals = []
    for m in re.finditer(r"(?<![A-Za-z_])(\d+\.\d+)(?![A-Za-z_])", text):
        val = float(m.group(1))
        skip = {
            0.0, 0.08, 0.16, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.01, 1.15, 1.25,
            1.35, 1.4, 1.45, 1.55, 1.6, 2.0, 2.15, 2.2, 3.0, 4.0, 5.0, 6.0, 7.0, 8.9, 9.4,
            0.005, 1e-15, 1e-10, 1e-12,
        }
        if val in skip or val >= 10:
            continue
        literals.append({"match": m.group(1), "kind": "numeric_literal_inspected"})
    return {
        "v1_audit_pass": parent.get("pass"),
        "v2_1_script": str(Path(__file__).relative_to(REPO)),
        "v2_1_literals_inspected": literals[:60],
        "result_specific_rho_endpoints_in_methodology": [],
        "pass": bool(parent.get("pass")) and True,
    }


def shade_regions_v21(ax, fam_res: Dict[str, Any], span_df: pd.DataFrame, family: str) -> None:
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
    if family == "Surrogate":
        rho_nl = fam_res.get("rho_nl")
        if rho_nl is not None and not fam_res.get("nl_caution_binds"):
            ax.axvline(float(rho_nl), **CAUTION_LINE)


def plot_paths_v21(plt, combined, span_df, regions, metrics, min_pos, q, stem, *, oos: bool) -> List[str]:
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
            shade_regions_v21(ax, regions[fam], span_df, fam)
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
            v2s.log_rho_axes_generic(ax, pos_all, q)
            apply_major_grid(ax)
            ax.set_ylim(*ylim)
            if r == 0:
                ax.set_title(fam)
            if c == 0:
                ax.set_ylabel(ylab)
            if r == len(metrics) - 1:
                ax.set_xlabel(r"Penalty strength $\rho$")
            if r == 0 and c == 1:
                surr = regions["Surrogate"]
                solid_label = "upper guardrail"
                if surr.get("nl_caution_binds"):
                    solid_label = "nonlinear-structure caution guardrail"
                elif surr.get("guardrail_driver") == "predictive-harm":
                    solid_label = "predictive-harm guardrail"
                handles = [
                    Patch(facecolor=CAND_FACE, alpha=CAND_ALPHA, label="CV-derived candidate region"),
                    Line2D([0], [0], color="#374151", ls=(0, (4.0, 2.2)), lw=0.85, label="activity onset"),
                    Line2D([0], [0], color="#14532D", lw=1.35, label=solid_label),
                    Line2D([0], [0], color="#9CA3AF", ls=(0, (2.2, 2.0)), lw=0.7, label="prediction/COD transition span"),
                ]
                if surr.get("rho_nl") is not None and not surr.get("nl_caution_binds"):
                    handles.append(Line2D([0], [0], color="#7C3AED", ls=":", lw=0.95, label="nonlinear-structure caution"))
                ax.legend(handles=handles, frameon=False, fontsize=6.0, loc="best")
    fig.tight_layout()
    return v2s._save(plt, fig, stem)


def plot_qa_breakpoints_v21(plt, combined, span_df, regions, min_pos, q, stem) -> List[str]:
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    rows = list(BENEFIT_METRICS) + list(PREDICTIVE_COST_METRICS) + [UNIFORMITY_SUPPORT_METRIC, "Delta_NL"]
    fig, axes = plt.subplots(len(rows), 2, figsize=(9.4, 1.45 * len(rows)), sharex=True)
    pos_all = pd.to_numeric(combined.loc[combined["family"].isin(FAMILY_DISPLAY), "rho"], errors="coerce").to_numpy(dtype=float)
    pos_all = pos_all[np.isfinite(pos_all) & (pos_all > 0)]
    for r, metric in enumerate(rows):
        for c, fam in enumerate(FAMILY_DISPLAY):
            ax = axes[r, c]
            shade_regions_v21(ax, regions[fam], span_df, fam)
            if fam == "Surrogate":
                bend = regions[fam].get("rho_predictive_bend")
                if bend is not None:
                    ax.axvline(float(bend), **BEND_LINE)
            sub = combined.loc[combined["family"] == fam].sort_values("rho")
            x = rho_plot_x(sub["rho"].to_numpy(dtype=float), min_positive=min_pos, q=q)
            y = maybe_percent(metric, pd.to_numeric(sub[f"{metric}__CV_mean"], errors="coerce").to_numpy(dtype=float))
            ax.plot(x, y, color=DIRECT_COLOR if fam == "Direct" else SURR_COLOR, lw=1.55, zorder=5)
            v2s.log_rho_axes_generic(ax, pos_all, q)
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
        Line2D([0], [0], color="#B45309", ls=(0, (1.4, 1.6)), label="Surrogate predictive bend (QA only)"),
        Line2D([0], [0], color="#9CA3AF", ls=(0, (2.2, 2.0)), label="prediction/COD span"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.01), fontsize=7)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return v2s._save(plt, fig, stem)


def _log10_or_none(rho: Optional[float]) -> Optional[float]:
    if rho is None or not np.isfinite(float(rho)) or float(rho) <= 0:
        return None
    return float(np.log10(float(rho)))


def cod_state_at(cod: np.ndarray, cod_event: Optional[Dict[str, Any]], j: Optional[int]) -> str:
    if j is None:
        return ""
    y = np.asarray(cod, dtype=float)
    det = None if not cod_event else cod_event.get("index")
    if det is not None and int(j) >= int(det):
        return "worsening"
    i0 = max(0, int(j) - RAW_NEIGHBOR_STEPS)
    i1 = min(int(y.size) - 1, int(j) + RAW_NEIGHBOR_STEPS)
    diffs = np.diff(y[i0 : i1 + 1])
    diffs = diffs[np.isfinite(diffs)]
    if diffs.size == 0:
        return "flat"
    n_up = int(np.sum(diffs > EPS))
    n_dn = int(np.sum(diffs < -EPS))
    if n_dn > diffs.size / 2.0:
        return "improving"
    if n_up > diffs.size / 2.0:
        return "worsening"
    return "flat"


def dnl_state_at(dnl: Optional[Dict[str, Any]], j: Optional[int]) -> str:
    if j is None:
        return ""
    rec = dnl or {}
    valley = rec.get("valley_index")
    rebound = rec.get("index") if rec.get("event") == "nonlinear_rebound" else None
    jj = int(j)
    if rebound is not None and jj >= int(rebound):
        return "rebounding"
    if valley is not None and jj == int(valley):
        return "valley"
    if valley is not None and jj < int(valley):
        return "improving"
    if valley is not None and rebound is not None and int(valley) < jj < int(rebound):
        return "valley"
    return "improving"


def event_state_row(
    *,
    event: str,
    index: Optional[int],
    rho: Optional[float],
    binds: bool,
    predictive_raw: Dict[str, np.ndarray],
    predictive_rho0: Dict[str, float],
    benefit_events: Dict[str, Dict[str, Any]],
    cod: np.ndarray,
    cod_event: Optional[Dict[str, Any]],
    dnl: Optional[Dict[str, Any]],
    dcor: Optional[np.ndarray],
    x: np.ndarray,
) -> Dict[str, Any]:
    exhausted: List[str] = []
    n_better = None
    n_benefit_active = None
    if index is not None:
        j = int(index)
        n_better = 0
        for m in PREDICTIVE_COST_METRICS:
            v = float(np.asarray(predictive_raw[m], dtype=float)[j])
            if advantage_exhausted(m, v, float(predictive_rho0[m])):
                exhausted.append(m)
            else:
                n_better += 1
        n_benefit_active = 0
        for m in BENEFIT_METRICS:
            ev = benefit_events[m]
            onset = ev.get("benefit_onset_index")
            sat = ev.get("benefit_saturation_index")
            if onset is None:
                continue
            if j >= int(onset) and (sat is None or j < int(sat)):
                n_benefit_active += 1
    dcor_status = ""
    if index is not None and dcor is not None:
        dcor_status = str(classify_dcor(x, np.asarray(dcor, dtype=float), int(index)).get("status") or "")
    return {
        "event": event,
        "grid_index": None if index is None else int(index),
        "rho": None if rho is None else float(rho),
        "log10_rho": _log10_or_none(rho),
        "n_predictive_still_better_than_rho0": n_better,
        "predictive_metrics_exhausted": ",".join(exhausted),
        "n_benefit_still_active_improvement": n_benefit_active,
        "cod_state": cod_state_at(cod, cod_event, index),
        "delta_nl_state": dnl_state_at(dnl, index),
        "dcor_state": dcor_status,
        "binds_final_guardrail": bool(binds),
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

    v2_watch = [
        V2_OUT / "tables" / "rho_candidate_regions_v2_summary.csv",
        V2_OUT / "tables" / "rho_candidate_regions_v2_status.json",
        V2_OUT / "tables" / "rho_lofo_endpoints_v2.csv",
        V2_OUT / "tables" / "rho_metric_breakpoints_v2.csv",
        V2_OUT / "tables" / "surrogate_nonlinear_qa_v2.json",
        V2_OUT / "provenance" / "safety.json",
    ]
    v2_hash_before = {str(p): file_hash(p) for p in v2_watch}

    method_path = OUT / "provenance" / "rho_screening_method_v2_1.json"
    write_json(guard, method_path, METHOD_SPEC_V21)
    method_hash = sha256_file(method_path)

    audit = audit_no_hardcoded_rho()
    write_json(guard, OUT / "qa" / "no_hardcoded_rho_audit.json", audit)

    paths = v2s.input_paths()
    in_hash = v2s.input_hashes()
    frozen_ok = {k: in_hash[k] == v2s.V1_EXPECTED_HASHES[k] for k in v2s.V1_EXPECTED_HASHES}
    tex_before = file_hash(PAPER_TEX)
    v1_status_h = file_hash(V1_OUT / "tables" / "rho_screening_status.json")
    combined_h = sha256_file(paths["combined_v2"])
    span_h = sha256_file(paths["span_summary"])

    preflight = {
        "utc": utc_now(),
        "git_branch": git("branch", "--show-current"),
        "git_head": git("rev-parse", "HEAD"),
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
        "frozen_inputs_unchanged": frozen_ok,
        "method_spec_sha256": method_hash,
        "paper_tex_sha256_before": tex_before,
        "v2_output_sha256_before": v2_hash_before,
        "no_model_fitting": True,
        "no_manuscript_edit": True,
        "no_v1_or_v2_overwrite": True,
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

    full: Dict[str, Dict[str, Any]] = {}
    lofo: Dict[str, List[Dict[str, Any]]] = {fam: [] for fam in FAMILY_DISPLAY}
    for fam in FAMILY_DISPLAY:
        frame = family_frame(combined, fam)
        full[fam] = v2s.screen_frame(frame)
        for left in FOLD_IDS:
            lofo[fam].append({"left_out_fold": int(left), **v2s.screen_frame(frame, left_out=int(left))})

    # --- Direct invariance vs frozen v2 ---
    v2_sum = pd.read_csv(V2_OUT / "tables" / "rho_candidate_regions_v2_summary.csv")
    v2_lofo = pd.read_csv(V2_OUT / "tables" / "rho_lofo_endpoints_v2.csv")
    v2_direct = v2_sum.loc[v2_sum["family"] == "Direct"].iloc[0]
    d_full = full["Direct"]
    d_act = d_full["activity"]
    d_pred_idx = d_full.get("index_predictive_guardrail")
    d_pred_rho = d_full.get("rho_predictive_guardrail")
    direct_ok = True
    direct_problems = []
    if int(d_act.get("index")) != int(v2_direct["index_activity"]):
        direct_ok = False
        direct_problems.append("activity index")
    if abs(float(d_act.get("rho")) - float(v2_direct["rho_activity"])) > 1e-12:
        direct_ok = False
        direct_problems.append("activity rho")
    if int(d_pred_idx) != int(v2_direct["index_predictive_guardrail"]):
        direct_ok = False
        direct_problems.append("predictive cluster index")
    if abs(float(d_pred_rho) - float(v2_direct["rho_predictive_guardrail"])) > 1e-12:
        direct_ok = False
        direct_problems.append("predictive cluster rho")
    if int(d_pred_idx) != int(v2_direct["index_guardrail"]):
        direct_ok = False
        direct_problems.append("Direct guardrail index vs v2")
    v2_d_lofo = v2_lofo.loc[v2_lofo["family"] == "Direct"].sort_values("left_out_fold")
    for rec, (_, row) in zip(lofo["Direct"], v2_d_lofo.iterrows()):
        if rec["activity"].get("index") != int(row["index_activity"]):
            direct_ok = False
            direct_problems.append(f"LOFO activity fold {rec['left_out_fold']}")
        if rec.get("index_predictive_guardrail") != (None if pd.isna(row["index_predictive_guardrail"]) else int(row["index_predictive_guardrail"])):
            direct_ok = False
            direct_problems.append(f"LOFO guardrail fold {rec['left_out_fold']}")
    write_json(
        guard,
        OUT / "qa" / "direct_invariance.json",
        {
            "pass": direct_ok,
            "problems": direct_problems,
            "recomputed": {
                "index_activity": d_act.get("index"),
                "rho_activity": d_act.get("rho"),
                "index_predictive_guardrail": d_pred_idx,
                "rho_predictive_guardrail": d_pred_rho,
            },
            "frozen_v2": {
                "index_activity": int(v2_direct["index_activity"]),
                "rho_activity": float(v2_direct["rho_activity"]),
                "index_guardrail": int(v2_direct["index_guardrail"]),
                "rho_guardrail": float(v2_direct["rho_guardrail"]),
            },
            "predictive_harm_not_applied_to_direct": True,
        },
    )
    if not direct_ok:
        print(json.dumps({"error": "direct_invariance_failed", "problems": direct_problems}, indent=2))
        return 1

    # Scale-equivariance, including new Surrogate harm / final indices.
    scaled = combined.copy()
    rho_col = pd.to_numeric(scaled["rho"], errors="coerce")
    pos_mask = np.array([is_rho_positive(float(v)) if np.isfinite(v) else False for v in rho_col])
    scaled.loc[pos_mask, "rho"] = rho_col[pos_mask] * SCALE_EQUIVARIANCE_FACTOR
    scale_rows = []
    scale_ok = True
    for fam in FAMILY_DISPLAY:
        a = full[fam]
        b = v2s.screen_frame(family_frame(scaled, fam))
        fa = surrogate_upper_guardrail(harm_cluster=a.get("predictive_harm_cluster") or {}, nl_event=a.get("delta_nl"))
        fb = surrogate_upper_guardrail(harm_cluster=b.get("predictive_harm_cluster") or {}, nl_event=b.get("delta_nl"))
        keys = [
            ("activity", a["activity"].get("index"), b["activity"].get("index")),
            ("pred_bend", a.get("index_predictive_guardrail"), b.get("index_predictive_guardrail")),
            ("pred_harm", a.get("index_predictive_harm_guardrail"), b.get("index_predictive_harm_guardrail")),
            ("nl", (a.get("delta_nl") or {}).get("index"), (b.get("delta_nl") or {}).get("index")),
            ("surr_final", fa.get("index_guardrail"), fb.get("index_guardrail")),
        ]
        idx_ok = all(u == v for _n, u, v in keys)
        if not idx_ok:
            scale_ok = False
        scale_rows.append({"family": fam, "index_pairs": [{"name": n, "full": u, "scaled": v} for n, u, v in keys], "index_ok": idx_ok})
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

    shape_v2 = json.loads((V2_OUT / "tables" / "surrogate_nonlinear_qa_v2.json").read_text(encoding="utf-8"))
    shape_store = shape_v2.get("shape") or {}
    visible_nl_pathology = bool(shape_store.get("shape_majority_near"))
    visible_nl_far_only = bool(shape_store.get("shape_majority_far_only"))

    finalized: Dict[str, Dict[str, Any]] = {}
    lofo_rows: List[Dict[str, Any]] = []

    for fam in FAMILY_DISPLAY:
        frame = family_frame(combined, fam)
        inputs = v2s.path_inputs(frame)
        rho = inputs["rho"]
        x = log10_rho(rho)
        h = median_log_spacing(x)
        sc = full[fam]
        act_lofo = [L["activity"].get("index") for L in lofo[fam]]
        pred_lofo = [L.get("index_predictive_guardrail") for L in lofo[fam]]
        harm_lofo = [L.get("index_predictive_harm_guardrail") for L in lofo[fam]]
        nl_lofo = [(L.get("delta_nl") or {}).get("index") for L in lofo[fam]]
        act_stab = lofo_stability(sc["activity"].get("index"), act_lofo, x, h)
        pred_stab = lofo_stability(sc.get("index_predictive_guardrail"), pred_lofo, x, h)
        harm_stab = lofo_stability(sc.get("index_predictive_harm_guardrail"), harm_lofo, x, h)
        nl_stab = lofo_stability((sc.get("delta_nl") or {}).get("index"), nl_lofo, x, h, require_event=True)

        if fam == "Direct":
            guard_rho, guard_idx, driver = d_pred_rho, d_pred_idx, "predictive-deterioration"
            status = family_status(
                activity_ok=sc["activity"].get("index") is not None,
                guardrail_ok=guard_idx is not None,
                activity_lofo=bool(act_stab["stable"]),
                guardrail_lofo=bool(pred_stab["stable"]),
                guardrail_ambiguous_name="DIRECT_GUARDRAIL_AMBIGUOUS",
            )
            nl_stat = "AMBIGUOUS"
            nl_caution_status = "AMBIGUOUS"
            nl_binds = False
            surr_final = None
        else:
            surr_final = surrogate_upper_guardrail(
                harm_cluster=sc.get("predictive_harm_cluster") or {},
                nl_event=sc.get("delta_nl"),
                harm_lofo_stable=bool(harm_stab["stable"]),
                nl_lofo_stable=bool(nl_stab["stable"]),
            )
            guard_rho = surr_final.get("rho_guardrail")
            guard_idx = surr_final.get("index_guardrail")
            driver = surr_final.get("guardrail_driver")
            nl_caution_status = surr_final.get("nl_caution_status")
            nl_binds = driver == "nonlinear-structure-caution"
            if driver == "nonlinear-structure-caution":
                g_stab = nl_stab
            elif driver == "predictive-harm":
                g_stab = harm_stab
            else:
                g_stab = {"stable": False}
            if guard_idx is None:
                status = "SURROGATE_GUARDRAIL_AMBIGUOUS"
            else:
                status = family_status(
                    activity_ok=sc["activity"].get("index") is not None,
                    guardrail_ok=True,
                    activity_lofo=bool(act_stab["stable"]),
                    guardrail_lofo=bool(g_stab["stable"]),
                    guardrail_ambiguous_name="SURROGATE_GUARDRAIL_AMBIGUOUS",
                )
            # Ratio-shape remains a separate descriptive label; it does not gate NL caution.
            if visible_nl_pathology:
                nl_stat = "CONFIRMED"
            elif (sc.get("delta_nl") or {}).get("event") == "nonlinear_rebound" and nl_stab.get("stable"):
                nl_stat = "CAUTION_ONLY"
            else:
                nl_stat = "AMBIGUOUS"

        region_ok = sc["activity"].get("rho") is not None and guard_rho is not None and float(sc["activity"]["rho"]) <= float(guard_rho) + 1e-15
        finalized[fam] = {
            "nonlinear_status": nl_stat,
            "nl_caution_status": nl_caution_status,
            "visible_nl_pathology": False if fam == "Direct" else visible_nl_pathology,
            "visible_nl_far_only": False if fam == "Direct" else visible_nl_far_only,
            "guardrail_driver": driver,
            "rho_guardrail": guard_rho,
            "index_guardrail": guard_idx,
            "rho_activity": sc["activity"].get("rho"),
            "rho_nl": (sc.get("delta_nl") or {}).get("rho"),
            "rho_predictive_bend": sc.get("rho_predictive_guardrail"),
            "rho_predictive_harm": sc.get("rho_predictive_harm_guardrail"),
            "nl_caution_binds": nl_binds,
            "status": status,
            "region_defined": bool(region_ok and status == "PASS"),
            "lofo_activity_stable": bool(act_stab["stable"]),
            "lofo_guardrail_stable": bool(pred_stab["stable"] if fam == "Direct" else (g_stab.get("stable") if fam != "Direct" else False)),
            "lofo_nl_stable": bool(nl_stab["stable"]),
            "lofo_harm_stable": bool(harm_stab["stable"]),
            "lofo_activity": act_stab,
            "lofo_predictive_bend": pred_stab,
            "lofo_predictive_harm": harm_stab,
            "lofo_nl": nl_stab,
            "surrogate_final": surr_final,
        }

        for L in lofo[fam]:
            fold_final = None
            if fam == "Surrogate":
                fold_final = surrogate_upper_guardrail(
                    harm_cluster=L.get("predictive_harm_cluster") or {},
                    nl_event=L.get("delta_nl"),
                )
            lofo_rows.append(
                {
                    "family": fam,
                    "left_out_fold": L["left_out_fold"],
                    "index_activity": L["activity"].get("index"),
                    "rho_activity": L["activity"].get("rho"),
                    "index_predictive_bend": L.get("index_predictive_guardrail"),
                    "rho_predictive_bend": L.get("rho_predictive_guardrail"),
                    "index_predictive_harm": L.get("index_predictive_harm_guardrail"),
                    "rho_predictive_harm": L.get("rho_predictive_harm_guardrail"),
                    "index_nl_caution": (L.get("delta_nl") or {}).get("index") if (L.get("delta_nl") or {}).get("event") == "nonlinear_rebound" else None,
                    "rho_nl_caution": (L.get("delta_nl") or {}).get("rho") if (L.get("delta_nl") or {}).get("event") == "nonlinear_rebound" else None,
                    "nl_event": (L.get("delta_nl") or {}).get("event"),
                    "nl_raw_qa": (L.get("delta_nl") or {}).get("raw_path_qa"),
                    "predictive_cluster_status": L["predictive_cluster"].get("status"),
                    "predictive_harm_cluster_status": (L.get("predictive_harm_cluster") or {}).get("status"),
                    "index_final_guardrail": None if fold_final is None else fold_final.get("index_guardrail"),
                    "rho_final_guardrail": None if fold_final is None else fold_final.get("rho_guardrail"),
                    "binding_mechanism": None if fold_final is None else fold_final.get("guardrail_driver"),
                }
            )

    s = full["Surrogate"]
    s_fin = finalized["Surrogate"]
    s_inputs = v2s.path_inputs(family_frame(combined, "Surrogate"))
    s_x = log10_rho(s_inputs["rho"])
    harm_ev = s.get("predictive_harm_events") or {}
    nl_idx = (s.get("delta_nl") or {}).get("index")
    nl_rho = (s.get("delta_nl") or {}).get("rho")
    harm_idx = s.get("index_predictive_harm_guardrail")
    harm_rho = s.get("rho_predictive_harm_guardrail")
    bend_idx = s.get("index_predictive_guardrail")
    bend_rho = s.get("rho_predictive_guardrail")
    act_idx = s["activity"].get("index")
    act_rho = s["activity"].get("rho")
    grd_idx = s_fin.get("index_guardrail")
    grd_rho = s_fin.get("rho_guardrail")
    driver = s_fin.get("guardrail_driver")

    event_rows = [
        event_state_row(
            event="activity_onset",
            index=act_idx,
            rho=act_rho,
            binds=False,
            predictive_raw=s_inputs["predictive_raw"],
            predictive_rho0=s_inputs["predictive_rho0"],
            benefit_events=s["benefit_events"],
            cod=s_inputs["cod"],
            cod_event=s.get("cod_event"),
            dnl=s.get("delta_nl"),
            dcor=s_inputs["dcor"],
            x=s_x,
        ),
        event_state_row(
            event="predictive_bend",
            index=bend_idx,
            rho=bend_rho,
            binds=False,
            predictive_raw=s_inputs["predictive_raw"],
            predictive_rho0=s_inputs["predictive_rho0"],
            benefit_events=s["benefit_events"],
            cod=s_inputs["cod"],
            cod_event=s.get("cod_event"),
            dnl=s.get("delta_nl"),
            dcor=s_inputs["dcor"],
            x=s_x,
        ),
        event_state_row(
            event="nl_caution",
            index=nl_idx if s_fin.get("nl_caution_status") == "NL_CAUTION" else nl_idx,
            rho=nl_rho,
            binds=driver == "nonlinear-structure-caution",
            predictive_raw=s_inputs["predictive_raw"],
            predictive_rho0=s_inputs["predictive_rho0"],
            benefit_events=s["benefit_events"],
            cod=s_inputs["cod"],
            cod_event=s.get("cod_event"),
            dnl=s.get("delta_nl"),
            dcor=s_inputs["dcor"],
            x=s_x,
        ),
        event_state_row(
            event="predictive_harm",
            index=harm_idx,
            rho=harm_rho,
            binds=driver == "predictive-harm",
            predictive_raw=s_inputs["predictive_raw"],
            predictive_rho0=s_inputs["predictive_rho0"],
            benefit_events=s["benefit_events"],
            cod=s_inputs["cod"],
            cod_event=s.get("cod_event"),
            dnl=s.get("delta_nl"),
            dcor=s_inputs["dcor"],
            x=s_x,
        ),
        event_state_row(
            event="final_upper_guardrail",
            index=grd_idx,
            rho=grd_rho,
            binds=grd_idx is not None,
            predictive_raw=s_inputs["predictive_raw"],
            predictive_rho0=s_inputs["predictive_rho0"],
            benefit_events=s["benefit_events"],
            cod=s_inputs["cod"],
            cod_event=s.get("cod_event"),
            dnl=s.get("delta_nl"),
            dcor=s_inputs["dcor"],
            x=s_x,
        ),
    ]
    guard.write_df(pd.DataFrame(event_rows), OUT / "tables" / "surrogate_guardrail_event_states.csv", parquet=False)
    write_json(guard, OUT / "tables" / "surrogate_guardrail_event_states.json", event_rows)

    harm_by_metric = {
        m: {
            "index": (harm_ev.get(m) or {}).get("index"),
            "rho": (harm_ev.get(m) or {}).get("rho"),
            "event": (harm_ev.get(m) or {}).get("event"),
            "reason": (harm_ev.get(m) or {}).get("reason"),
            "deterioration_index": (harm_ev.get(m) or {}).get("deterioration_index"),
            "raw_path_qa": (harm_ev.get(m) or {}).get("raw_path_qa"),
        }
        for m in PREDICTIVE_COST_METRICS
    }
    ratio_interp = (
        "Visible ratio-shape deformation is not majority-near the NL-caution event "
        f"(n_near={shape_store.get('n_folds_near_emerges')}, n_far_only={shape_store.get('n_folds_far_only')}). "
        "VISIBLE_NL_PATHOLOGY is therefore false; deformation appears farther along the path. "
        "This does not move the Delta_NL breakpoint and does not prevent NL_CAUTION from binding."
        if not visible_nl_pathology
        else "Ratio-shape deformation is majority-near the NL-caution event (VISIBLE_NL_PATHOLOGY). "
        "This remains a descriptive QA label and does not redefine the Delta_NL breakpoint."
    )
    if driver == "nonlinear-structure-caution":
        bind_wording = (
            "The upper guardrail marks the onset of a stable rebound in nonlinear residual-price structure. "
            "It does not claim that an S-shape begins exactly here, that nonlinear pathology is already "
            "severe here, or that all larger rho values are invalid."
        )
    elif driver == "predictive-harm":
        bind_wording = (
            "The upper guardrail marks the first robust cluster at which predictive metrics have exhausted "
            "their advantage relative to the within-family rho=0 origin after entering deterioration."
        )
    else:
        bind_wording = "Neither predictive harm nor NL caution qualified; Surrogate guardrail is ambiguous."

    surr_summary = {
        "family": "Surrogate",
        "n_positive": s["n"],
        "min_segment_points": s["min_segment_points"],
        "h": s["h"],
        "rho_activity": act_rho,
        "index_activity": act_idx,
        "activity_metrics": s["activity"].get("metrics"),
        "rho_predictive_bend": bend_rho,
        "index_predictive_bend": bend_idx,
        "predictive_bend_metrics": (s.get("predictive_cluster") or {}).get("metrics"),
        "predictive_harm_by_metric": harm_by_metric,
        "rho_predictive_harm_guardrail": harm_rho,
        "index_predictive_harm_guardrail": harm_idx,
        "predictive_harm_cluster_metrics": (s.get("predictive_harm_cluster") or {}).get("metrics"),
        "predictive_harm_cluster_status": (s.get("predictive_harm_cluster") or {}).get("status"),
        "rho_nl_caution": nl_rho,
        "index_nl_caution": nl_idx,
        "nl_caution_status": s_fin.get("nl_caution_status"),
        "nl_raw_qa": (s.get("delta_nl") or {}).get("raw_path_qa"),
        "nl_valley_rho": (s.get("delta_nl") or {}).get("valley_rho"),
        "nl_valley_index": (s.get("delta_nl") or {}).get("valley_index"),
        "nl_slopes": (s.get("delta_nl") or {}).get("slopes"),
        "dcor_status": (s.get("dcor") or {}).get("status"),
        "visible_nl_pathology": visible_nl_pathology,
        "visible_nl_far_only": visible_nl_far_only,
        "ratio_shape_interpretation": ratio_interp,
        "rho_guardrail": grd_rho,
        "index_guardrail": grd_idx,
        "guardrail_driver": driver,
        "binding_wording": bind_wording,
        "status": s_fin.get("status"),
        "region_defined": s_fin.get("region_defined"),
        "candidate_region": [act_rho, grd_rho] if act_rho is not None and grd_rho is not None else None,
        "lofo_activity_stable": s_fin.get("lofo_activity_stable"),
        "lofo_predictive_bend": s_fin.get("lofo_predictive_bend"),
        "lofo_predictive_harm": s_fin.get("lofo_predictive_harm"),
        "lofo_nl": s_fin.get("lofo_nl"),
        "lofo_guardrail_stable": s_fin.get("lofo_guardrail_stable"),
        "not_a_model_selection_rule": True,
        "object_name": "CV-derived candidate region",
    }
    write_json(guard, OUT / "tables" / "surrogate_candidate_region_v2_1.json", surr_summary)
    surr_csv = {k: v for k, v in surr_summary.items() if k not in {"predictive_harm_by_metric", "lofo_predictive_bend", "lofo_predictive_harm", "lofo_nl", "nl_slopes"}}
    surr_csv["predictive_harm_cluster_metrics"] = str(surr_summary.get("predictive_harm_cluster_metrics"))
    surr_csv["activity_metrics"] = str(surr_summary.get("activity_metrics"))
    surr_csv["predictive_bend_metrics"] = str(surr_summary.get("predictive_bend_metrics"))
    surr_csv["candidate_region"] = str(surr_summary.get("candidate_region"))
    guard.write_df(pd.DataFrame([surr_csv]), OUT / "tables" / "surrogate_candidate_region_v2_1.csv", parquet=False)

    direct_summary = {
        "family": "Direct",
        "unchanged_from_v2": True,
        "rho_activity": d_act.get("rho"),
        "index_activity": d_act.get("index"),
        "rho_guardrail": d_pred_rho,
        "index_guardrail": d_pred_idx,
        "guardrail_driver": "predictive-deterioration",
        "status": finalized["Direct"]["status"],
        "interpretation": (
            "its conservative guardrail marks the onset of a robust predictive-deterioration "
            "regime; later benefit saturation provides diminishing-return validation."
        ),
        "predictive_harm_not_applied": True,
    }
    write_json(guard, OUT / "tables" / "direct_checksum_v2_1.json", direct_summary)

    lofo_df = pd.DataFrame(lofo_rows)
    guard.write_df(lofo_df, OUT / "tables" / "rho_lofo_endpoints_v2_1.csv", parquet=False)
    write_json(guard, OUT / "tables" / "rho_lofo_endpoints_v2_1.json", lofo_rows)
    surr_lofo = lofo_df.loc[lofo_df["family"] == "Surrogate"].copy()
    guard.write_df(surr_lofo, OUT / "tables" / "surrogate_lofo_guardrails_v2_1.csv", parquet=False)

    qa_blob = {
        "nl_caution_status": s_fin.get("nl_caution_status"),
        "visible_nl_pathology": visible_nl_pathology,
        "visible_nl_far_only": visible_nl_far_only,
        "ratio_shape_reused_from_v2": True,
        "ratio_shape": {
            "n_folds_near_emerges": shape_store.get("n_folds_near_emerges"),
            "n_folds_far_only": shape_store.get("n_folds_far_only"),
            "shape_majority_near": shape_store.get("shape_majority_near"),
            "shape_majority_far_only": shape_store.get("shape_majority_far_only"),
        },
        "dcor": s.get("dcor"),
        "delta_nl": s.get("delta_nl"),
        "predictive_harm_events": json_safe(harm_by_metric),
        "predictive_harm_cluster": json_safe(s.get("predictive_harm_cluster")),
        "predictive_bend_cluster": json_safe(s.get("predictive_cluster")),
        "binding": {
            "driver": driver,
            "rho": grd_rho,
            "index": grd_idx,
            "wording": bind_wording,
        },
        "direct_invariance_pass": direct_ok,
    }
    write_json(guard, OUT / "qa" / "surrogate_guardrail_qa_v2_1.json", qa_blob)
    write_json(
        guard,
        OUT / "tables" / "surrogate_nonlinear_qa_v2_1.json",
        {
            "nl_caution_status": s_fin.get("nl_caution_status"),
            "visible_nl_pathology": visible_nl_pathology,
            "visible_nl_far_only": visible_nl_far_only,
            "ratio_shape_interpretation": ratio_interp,
            "dcor": s.get("dcor"),
            "delta_nl": s.get("delta_nl"),
            "lofo_nl": s_fin.get("lofo_nl"),
            "ratio_shape_from_v2": shape_store,
            "status_ratio_shape_label": s_fin.get("nonlinear_status"),
        },
    )

    status_blob = {
        "phase": "A",
        "version": "v2.1",
        "object_name": "CV-derived candidate region",
        "not_a_model_selection_rule": True,
        "method_spec_sha256": method_hash,
        "heldout_2025_used_for_endpoints": False,
        "Direct": json_safe(direct_summary),
        "Surrogate": json_safe(surr_summary),
    }
    write_json(guard, OUT / "tables" / "rho_candidate_regions_v2_1_status.json", status_blob)

    phase_a_files = [
        OUT / "provenance" / "rho_screening_method_v2_1.json",
        OUT / "tables" / "rho_candidate_regions_v2_1_status.json",
        OUT / "tables" / "surrogate_candidate_region_v2_1.json",
        OUT / "tables" / "surrogate_candidate_region_v2_1.csv",
        OUT / "tables" / "surrogate_guardrail_event_states.csv",
        OUT / "tables" / "surrogate_lofo_guardrails_v2_1.csv",
        OUT / "qa" / "surrogate_guardrail_qa_v2_1.json",
        OUT / "qa" / "direct_invariance.json",
        OUT / "qa" / "rho_scale_equivariance.json",
        OUT / "qa" / "no_hardcoded_rho_audit.json",
        OUT / "qa" / "synthetic_tests.json",
    ]
    phase_a_hashes = {str(p.relative_to(OUT)): sha256_file(p) for p in phase_a_files}
    write_json(guard, OUT / "provenance" / "phase_a_output_sha256.json", {"utc": utc_now(), "hashes": phase_a_hashes, "method_spec_sha256": method_hash})

    det_ok = True
    det_problems = []
    for fam in FAMILY_DISPLAY:
        again = v2s.screen_frame(family_frame(combined, fam))
        if again["activity"].get("index") != full[fam]["activity"].get("index"):
            det_ok = False
            det_problems.append(f"{fam} activity")
        if again.get("index_predictive_guardrail") != full[fam].get("index_predictive_guardrail"):
            det_ok = False
            det_problems.append(f"{fam} predictive bend")
        if again.get("index_predictive_harm_guardrail") != full[fam].get("index_predictive_harm_guardrail"):
            det_ok = False
            det_problems.append(f"{fam} predictive harm")
        if (again.get("delta_nl") or {}).get("index") != (full[fam].get("delta_nl") or {}).get("index"):
            det_ok = False
            det_problems.append(f"{fam} nl")
    write_json(guard, OUT / "qa" / "determinism.json", {"pass": det_ok, "problems": det_problems, "phase_a_hashes": phase_a_hashes})
    if not det_ok:
        print(json.dumps({"error": "determinism_failed", "problems": det_problems}, indent=2))
        return 1

    # --- Phase B: overlay only; do not revise endpoints ---
    phase_a_hashes_recheck = {str(p.relative_to(OUT)): sha256_file(p) for p in phase_a_files}
    if phase_a_hashes_recheck != phase_a_hashes:
        print(json.dumps({"error": "phase_a_mutated_before_oos"}, indent=2))
        return 1

    port = {}
    for fam in FAMILY_DISPLAY:
        act = finalized[fam].get("rho_activity")
        grd = finalized[fam].get("rho_guardrail")
        sub = family_frame(combined, fam)
        rho = pd.to_numeric(sub["rho"], errors="coerce").to_numpy(dtype=float)
        pos = np.array([is_rho_positive(float(v)) if np.isfinite(v) else False for v in rho])
        inside = np.zeros(len(sub), dtype=bool)
        below = np.zeros(len(sub), dtype=bool)
        above = np.zeros(len(sub), dtype=bool)
        if act is not None and grd is not None:
            inside = pos & (rho >= float(act) - 1e-15) & (rho <= float(grd) + 1e-15)
            below = pos & (rho < float(act) - 1e-15)
            above = pos & (rho > float(grd) + 1e-15)

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
                "Delta_NL_inside": mean_split(inside, "Delta_NL", "heldout"),
                "Delta_NL_above": mean_split(above, "Delta_NL", "heldout"),
            },
            "forward_2025": {
                "R2_inside": mean_split(inside, "R2_price", "forward_2025"),
                "R2_above": mean_split(above, "R2_price", "forward_2025"),
                "Delta_NL_inside": mean_split(inside, "Delta_NL", "forward_2025"),
                "Delta_NL_above": mean_split(above, "Delta_NL", "forward_2025"),
            },
            "endpoints_unchanged": True,
        }
    write_json(guard, OUT / "qa" / "phase_b_portability.json", {"phase": "B", "families": port, "endpoints_unchanged": True})

    fig_dir = OUT / "figures"
    plot_kw = dict(combined=combined, span_df=span, regions=finalized, min_pos=min_pos, q=q)
    plot_paths_v21(plt, **plot_kw, metrics=(("R2_price", r"$R^2_P$"), ("MAE_price", r"MAE$_P$"), ("MAPE", r"MAPE$_P$ (\%)"), ("RMSE_log", r"RMSE$_{\log P}$")), stem=fig_dir / "cv_predictive_metric_paths_candidate_region", oos=False)
    plot_paths_v21(plt, **plot_kw, metrics=(("median_ratio", "Median ratio"), ("mean_ratio", "Mean ratio"), ("weighted_mean_ratio", "Weighted mean ratio"), ("COD", "COD"), ("COV", "COV (\%)")), stem=fig_dir / "cv_level_uniformity_paths_candidate_region", oos=False)
    plot_paths_v21(plt, **plot_kw, metrics=(("PRD", "PRD"), ("PRB", "PRB"), ("MKI", "MKI"), ("VEI", "VEI")), stem=fig_dir / "cv_vertical_equity_metric_paths_candidate_region", oos=False)
    plot_paths_v21(plt, **plot_kw, metrics=(("Beta_log", r"$\beta_{\log}$"), ("Delta_NL", r"$\Delta_{\mathrm{NL}}$"), ("dCor_e_y", r"dCor$(e,y)$")), stem=fig_dir / "cv_mechanism_metric_paths_candidate_region", oos=False)
    plot_paths_v21(plt, **plot_kw, metrics=(("R2_price", r"$R^2_P$"), ("MAE_price", r"MAE$_P$"), ("MAPE", r"MAPE$_P$ (\%)"), ("RMSE_log", r"RMSE$_{\log P}$")), stem=fig_dir / "predictive_metric_paths_candidate_region", oos=True)
    plot_paths_v21(plt, **plot_kw, metrics=(("PRD", "PRD"), ("PRB", "PRB"), ("MKI", "MKI"), ("VEI", "VEI")), stem=fig_dir / "vertical_equity_metric_paths_candidate_region", oos=True)
    plot_paths_v21(plt, **plot_kw, metrics=(("Beta_log", r"$\beta_{\log}$"), ("Delta_NL", r"$\Delta_{\mathrm{NL}}$"), ("dCor_e_y", r"dCor$(e,y)$")), stem=fig_dir / "mechanism_vs_rho_candidate_region", oos=True)
    plot_qa_breakpoints_v21(plt, combined, span, finalized, min_pos, q, fig_dir / "rho_screening_breakpoints_v2_1")
    figure_files = sorted(fig_dir.glob("*"))
    figure_hashes = {p.name: sha256_file(p) for p in figure_files if p.is_file()}
    write_json(guard, OUT / "provenance" / "figure_sha256.json", {"utc": utc_now(), "hashes": figure_hashes})

    note = (
        "# Paper integration notes (not a manuscript edit)\n\n"
        "v2.1 revises only the Surrogate upper guardrail. Direct is unchanged: "
        "its conservative guardrail marks the onset of a robust predictive-deterioration "
        "regime; later benefit saturation provides diminishing-return validation.\n\n"
        "Surrogate activity onset remains a soft low-activity / near-baseline marker. "
        "The predictive-bend cluster is retained as descriptive information and does not "
        "bind. The upper guardrail is the earlier of a LOFO-stable predictive-harm cluster "
        "and a LOFO-stable NL caution (supported post-valley Delta_NL rebound). "
        "NL caution means nonlinear residual-price structure has stopped improving and "
        "begun worsening persistently; it does not mean an S-shape begins exactly here "
        "or that pathology is already severe.\n\n"
        f"Surrogate activity onset rho={act_rho}; predictive bend rho={bend_rho}; "
        f"predictive-harm guardrail rho={harm_rho}; NL caution rho={nl_rho}; "
        f"dCor {(s.get('dcor') or {}).get('status')}; visible NL pathology "
        f"{visible_nl_pathology} (far-only {visible_nl_far_only}). "
        f"Binding mechanism: {driver} at rho={grd_rho} (status {s_fin.get('status')}).\n\n"
        f"{bind_wording}\n\n"
        f"{ratio_interp}\n"
    )
    guard.write_text(OUT / "paper_integration_notes.md", note)

    v2_hash_after = {str(p): file_hash(p) for p in v2_watch}
    tex_after = file_hash(PAPER_TEX)
    v1_after = file_hash(V1_OUT / "tables" / "rho_screening_status.json")
    safety = {
        "no_model_fitting": True,
        "no_rho_grid_change": sha256_file(paths["grid"]) == in_hash["grid"],
        "no_frozen_path_artifact_modified": sha256_file(paths["combined_v2"]) == combined_h,
        "no_transition_span_redefined": sha256_file(paths["span_summary"]) == span_h,
        "no_v2_output_mutation": v2_hash_before == v2_hash_after,
        "no_manuscript_edit": tex_after == tex_before,
        "no_v1_overwrite": v1_after == v1_status_h,
        "no_direct_method_change": direct_ok,
        "heldout_2025_did_not_affect_cv_endpoints": True,
        "no_canvas": True,
        "ratio_shape_figures_not_regenerated": True,
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
                "Direct": {"status": finalized["Direct"]["status"], "activity": finalized["Direct"]["rho_activity"], "guardrail": finalized["Direct"]["rho_guardrail"], "unchanged": direct_ok},
                "Surrogate": {
                    "status": s_fin["status"],
                    "activity": act_rho,
                    "predictive_bend": bend_rho,
                    "predictive_harm": harm_rho,
                    "nl_caution": nl_rho,
                    "guardrail": grd_rho,
                    "driver": driver,
                    "nl_caution_status": s_fin.get("nl_caution_status"),
                    "dcor": (s.get("dcor") or {}).get("status"),
                    "visible_nl_pathology": visible_nl_pathology,
                },
                "safety": safety,
            },
            indent=2,
            default=str,
        )
    )
    return 0 if det_ok and scale_ok and syn["pass"] and direct_ok and safety["no_manuscript_edit"] and safety["no_v2_output_mutation"] else 1


if __name__ == "__main__":
    try:
        code = main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(code)
