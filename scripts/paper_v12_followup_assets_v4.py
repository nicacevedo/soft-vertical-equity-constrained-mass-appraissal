#!/usr/bin/env python3
"""v4 decade-anchor, CV Delta_NL, and post-hoc equity/mechanism bend diagnostics.

Deterministic post-processing of frozen 994-tree predictions only.
NO LightGBM fitting, NO new rho, NO retuning, NO TeX compilation.
Does not mutate v1/v2/v3 analysis artifacts or combined_path_table_v2.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.paper_v12_followup_assets_v3 import (  # noqa: E402
    PRIMARY_METRICS_TABLE,
    SECONDARY_METRICS_TABLE,
    display_value,
    extract_one_event,
    write_table_bundle,
)
from utils.delta_nl import estimate_delta_nl_from_frame, estimator_spec, estimator_spec_hash  # noqa: E402
from utils.transition_paper_asset_plots import combined_row, metric_val  # noqa: E402
from utils.transition_paper_assets import (  # noqa: E402
    DECADE_DISPLAY_ANCHOR_TARGETS,
    decade_nominal_to_tested,
    decade_positive_display_anchors,
    manuscript_format_flags,
)
from utils.transition_regions import (  # noqa: E402
    FAMILY_DISPLAY,
    FOLD_IDS,
    OutputGuard,
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
ASSETS_V2 = V2.parent / "transition_regions_paper_assets_v2"
V3 = V2.parent / "transition_regions_paper_assets_v3_followup"
V4 = V2.parent / "transition_regions_paper_assets_v4_delta_nl_bends"
PAPER_TEX = REPO / "paper" / "paper_v12.tex"
PAPER_IMG = REPO / "paper" / "img" / "generated_v12_994"

FROZEN_DIRECT_SPAN = (0.04941713361323837, 1.0985411419875584)
FROZEN_SURR_SPAN = (0.002222996482526201, 0.954095476349994)
EXPECTED_DECADE_TESTED = {
    0.01: 0.010481131341546875,
    0.1: 0.1,
    1.0: 0.9540954763499939,
    10.0: 10.481131341546853,
    100.0: 100.0,
}
CORE_BEND_METRICS = (
    ("PRD", 1.0, "abs_target"),
    ("PRB", 0.0, "abs_target"),
    ("MKI", 1.0, "abs_target"),
    ("VEI", 0.0, "abs_target"),
    ("Beta_log", 0.0, "abs_target"),
    ("dCor_e_y", None, "identity"),
)
LEVEL_CONTEXT_METRICS = ("median_ratio", "mean_ratio", "weighted_mean_ratio", "COD", "COV")
FORBIDDEN_PHRASES = (
    "sweet spot",
    "safe range",
    "recommended range",
    "preferred range",
    "selected range",
    "optimal range",
    "operationally safe",
    "deployment-ready",
)
MECH_EVENT_SPECS_V4 = [
    ("Beta_log", "abs_target", 0.0, "descriptive_neutrality"),
    ("Delta_NL", "min", None, "descriptive_minimum"),
    ("dCor_e_y", "min", None, "descriptive_minimum"),
]


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
        if any(part in path.parts for part in exclude_parts):
            continue
        out[str(path.relative_to(REPO))] = sha256_file(path)
    return out


def frozen_named_files() -> Dict[str, Path]:
    return {
        "paper/paper_v12.tex": PAPER_TEX,
        "combined_path_table_v2.csv": V2 / "tables" / "combined_path_table_v2.csv",
        "lower_rho_grid_v2.json": EXT / "protocol" / "lower_rho_grid_v2.json",
        "transition_span_summary.csv": V2 / "tables" / "transition_span_summary.csv",
        "transition_events_cv_mean.csv": V2 / "tables" / "transition_events_cv_mean.csv",
        "transition_events_by_fold.csv": V2 / "tables" / "transition_events_by_fold.csv",
        "transition_lofo_sensitivity.csv": V2 / "tables" / "transition_lofo_sensitivity.csv",
        "transition_temporal_concordance.csv": V2 / "tables" / "transition_temporal_concordance.csv",
        "transition_oos_span_regret_v3.csv": V3 / "tables" / "transition_oos_span_regret_v3.csv",
        "event_sharpness_summary_v3.csv": V3 / "tables" / "event_sharpness_summary_v3.csv",
        "utils/delta_nl.py": REPO / "utils" / "delta_nl.py",
    }


def family_span_endpoints(span: pd.DataFrame, family: str) -> Tuple[float, float]:
    row = span.loc[span["family"] == family].iloc[0]
    if str(row["status"]) != "VALID_POSITIVE_INTERIOR_SPAN":
        raise RuntimeError(f"frozen v2 span is not VALID for {family}")
    return float(row["rho_transition_low"]), float(row["rho_transition_high"])


def load_frozen() -> Dict[str, Any]:
    combined = pd.read_csv(V2 / "tables" / "combined_path_table_v2.csv")
    span = pd.read_csv(V2 / "tables" / "transition_span_summary.csv")
    grid = json.loads((EXT / "protocol" / "lower_rho_grid_v2.json").read_text(encoding="utf-8"))
    return {"combined": combined, "span": span, "grid": grid}


def tex_display(metric: str, val: float, flags: Dict[str, Any]) -> str:
    s = display_value(metric, val).replace("$", r"\$").replace("%", r"\%")
    if flags["manuscript_bold"]:
        s = r"\textbf{" + s + "}"
    if flags["manuscript_asterisk"]:
        s = s + r"\textsuperscript{*}"
    return s


def build_decade_tables(combined: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    grid = combined.loc[combined["family"] == "Direct", "rho"].to_numpy(dtype=float)
    anchors = decade_positive_display_anchors(grid)
    if len(anchors) != 5:
        raise RuntimeError(f"expected 5 decade anchors, got {anchors}")
    rows_p: List[Dict[str, Any]] = []
    rows_s: List[Dict[str, Any]] = []
    for split in ("heldout", "forward_2025"):
        lin = combined_row(combined, "Linear")
        lgb = combined_row(combined, "LightGBM")
        for fam in FAMILY_DISPLAY:
            for a, nom in zip(anchors, DECADE_DISPLAY_ANCHOR_TARGETS):
                rec_row = combined_row(combined, fam, float(a))
                exact = float(rec_row["rho"])
                for specs, sink in ((PRIMARY_METRICS_TABLE, rows_p), (SECONDARY_METRICS_TABLE, rows_s)):
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
                        sink.append(
                            {
                                "split": split,
                                "family": fam,
                                "exact_tested_rho": exact,
                                "nominal_display_anchor": float(nom),
                                "metric": name,
                                "value_unrounded": float(val),
                                "value_display": display_value(name, val),
                                "value_tex": tex_display(name, val, flags),
                                "preferred_direction_or_target": direction,
                                "target_value": target,
                                "in_reference_range": flags["within_reference_range"],
                                "beats_both_baselines": flags["beats_both_baselines"],
                                "beats_ordinary_only": flags["beats_ordinary_only"],
                                "manuscript_bold": flags["manuscript_bold"],
                                "manuscript_asterisk": flags["manuscript_asterisk"],
                                "nearest_anchor_logic": "decade_positive_display_anchors",
                                "config_id": str(rec_row["config_id"]),
                            }
                        )
    def _finish(rows: List[Dict[str, Any]]) -> pd.DataFrame:
        df = pd.DataFrame(rows)
        df["in_reference_range"] = pd.array(
            [None if v is None else bool(v) for v in df["in_reference_range"]],
            dtype="boolean",
        )
        for col in ("manuscript_bold", "manuscript_asterisk", "beats_both_baselines", "beats_ordinary_only"):
            df[col] = df[col].astype(bool)
        return df

    return _finish(rows_p), _finish(rows_s)


def disk_fold_id(paper_fold: int) -> int:
    """Paper fold labels are 1..7; prediction directories use 0..6."""
    return int(paper_fold) - 1


def resolve_pred_path(rel_or_abs: Optional[str], run_id: str, paper_fold: int) -> Optional[Path]:
    disk_fold = disk_fold_id(paper_fold)
    candidates: List[Path] = []
    if rel_or_abs:
        p = Path(rel_or_abs)
        candidates.append(p if p.is_absolute() else REPO / p)
    for root in (CANONICAL, EXT):
        candidates.append(
            root
            / "predictions"
            / f"data_id={DATA_ID}"
            / f"split_id={SPLIT_ID}"
            / f"fold_id={disk_fold}"
            / f"{run_id}.parquet"
        )
    for p in candidates:
        if p.is_file():
            return p
    return None


def load_run_status_index() -> Dict[Tuple[str, int], Dict[str, Any]]:
    index: Dict[Tuple[str, int], Dict[str, Any]] = {}
    for root in (CANONICAL, EXT):
        base = root / "run_status" / f"data_id={DATA_ID}" / f"split_id={SPLIT_ID}"
        for fold in range(7):
            d = base / f"fold_id={fold}"
            if not d.is_dir():
                continue
            for path in d.glob("*.json"):
                blob = json.loads(path.read_text(encoding="utf-8"))
                cid = str(blob["config_id"])
                fid = int(blob["fold_id"])
                rec = {
                    "root": str(root),
                    "run_id": str(blob["run_id"]),
                    "config_id": cid,
                    "fold_id": fid,
                    "status": blob.get("status"),
                    "model_name": blob.get("model_name"),
                    "predictions_file": (blob.get("artifacts") or {}).get("predictions_file"),
                    "run_status_path": str(path.relative_to(REPO)),
                }
                key = (cid, fid)
                prev = index.get(key)
                if prev is None or (prev.get("status") != "completed" and rec["status"] == "completed"):
                    index[key] = rec
    return index


def sha256_pred_frame(df: pd.DataFrame) -> str:
    cols = ["row_id", "y_true_log", "y_pred_log"]
    payload = df[cols].sort_values("row_id", kind="mergesort").to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def inventory_cv_predictions(combined: pd.DataFrame) -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    index = load_run_status_index()
    rows: List[Dict[str, Any]] = []
    missing: List[Dict[str, Any]] = []
    sub = combined.loc[combined["family"].isin(FAMILY_DISPLAY)].copy()
    for _, rec in sub.iterrows():
        fam = str(rec["family"])
        rho = float(rec["rho"])
        cid = str(rec["config_id"])
        required = is_rho_positive(rho)
        for fold in FOLD_IDS:
            st = index.get((cid, disk_fold_id(int(fold))))
            pred_path = None
            pred_ok = False
            notes = []
            n_rows = None
            n_unique = None
            n_finite = None
            ytrue_hash = None
            pred_hash = None
            ytrue_ok = None
            unique_ok = None
            if st is None:
                notes.append("missing_run_status")
            else:
                pred_path = resolve_pred_path(st.get("predictions_file"), st["run_id"], int(fold))
                if pred_path is None:
                    notes.append("missing_prediction_parquet")
                else:
                    try:
                        df = pd.read_parquet(pred_path, columns=["row_id", "y_true_log", "y_pred_log"])
                    except Exception as exc:
                        notes.append(f"read_error:{type(exc).__name__}")
                    else:
                        n_rows = int(len(df))
                        n_unique = int(pd.Series(df["row_id"]).nunique())
                        unique_ok = n_unique == n_rows
                        y = pd.to_numeric(df["y_true_log"], errors="coerce").to_numpy(dtype=float)
                        p = pd.to_numeric(df["y_pred_log"], errors="coerce").to_numpy(dtype=float)
                        n_finite = int(np.isfinite(y).sum())
                        finite_ok = bool(np.all(np.isfinite(y)) and np.all(np.isfinite(p)) and y.size == p.size)
                        ytrue_ok = bool(np.all(np.isfinite(y)))
                        ytrue_hash = hashlib.sha256(np.ascontiguousarray(y).tobytes()).hexdigest()
                        pred_hash = hashlib.sha256(
                            np.ascontiguousarray(p).tobytes()
                            + b"|"
                            + hashlib.sha256("|".join(map(str, df["row_id"].tolist())).encode("utf-8")).digest()
                        ).hexdigest()
                        pred_ok = bool(unique_ok and finite_ok)
                        if not unique_ok:
                            notes.append("duplicate_row_id")
                        if not finite_ok:
                            notes.append("nonfinite_predictions_or_y")
            row = {
                "family": fam,
                "rho": rho,
                "config_id": cid,
                "fold_id": int(fold),
                "disk_fold_id": disk_fold_id(int(fold)),
                "required_positive_rho": bool(required),
                "rho_zero": bool(is_rho_zero(rho)),
                "run_id": None if st is None else st["run_id"],
                "run_status": None if st is None else st.get("status"),
                "prediction_path": None if pred_path is None else str(pred_path.relative_to(REPO)),
                "n_rows": n_rows,
                "n_unique_row_id": n_unique,
                "unique_row_id": unique_ok,
                "y_true_finite": ytrue_ok,
                "n_finite_y_true": n_finite,
                "y_true_hash": ytrue_hash,
                "prediction_sha256": pred_hash,
                "ok": bool(pred_ok),
                "notes": "|".join(notes),
            }
            rows.append(row)
            if required and not pred_ok:
                missing.append(row)
        if len(rows) % 140 == 0:
            print(f"inventory progress n_rows={len(rows)} missing={len(missing)}", flush=True)
    return pd.DataFrame(rows), missing


def _delta_nl_job(job: Dict[str, Any]) -> Dict[str, Any]:
    path = REPO / job["prediction_path"]
    df = pd.read_parquet(path, columns=["row_id", "y_true_log", "y_pred_log"])
    out = estimate_delta_nl_from_frame(df)
    return {
        "family": job["family"],
        "rho": float(job["rho"]),
        "config_id": job["config_id"],
        "fold_id": int(job["fold_id"]),
        "run_id": job["run_id"],
        "n": int(out["n"]),
        "Delta_NL": float(out["Delta_NL"]),
        "Delta_NL_raw": float(out["Delta_NL_raw"]),
        "MSE_aff": float(out["MSE_aff"]),
        "MSE_spl": float(out["MSE_spl"]),
        "var_e": float(out["var_e"]),
        "fold_assignment_hash": out["fold_assignment_hash"],
        "estimator_spec_hash": out["estimator_spec_hash"],
        "prediction_sha256": job["prediction_sha256"],
        "prediction_path": job["prediction_path"],
    }


def target_distance(values: np.ndarray, kind: str, target: Optional[float]) -> np.ndarray:
    v = np.asarray(values, dtype=float)
    if kind == "identity":
        return v
    if target is None:
        raise ValueError("abs_target requires a target")
    return np.abs(v - float(target))


def ols_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
    A = np.column_stack([x, np.ones_like(x)])
    coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    pred = A @ coef
    sse = float(np.sum((y - pred) ** 2))
    return float(coef[0]), float(coef[1]), sse


def primary_two_segment_bend(rho: np.ndarray, dist: np.ndarray) -> Dict[str, Any]:
    r = np.asarray(rho, dtype=float)
    d = np.asarray(dist, dtype=float)
    mask = np.isfinite(r) & np.isfinite(d) & np.array([is_rho_positive(float(x)) for x in r])
    r, d = r[mask], d[mask]
    order = np.argsort(r, kind="mergesort")
    r, d = r[order], d[order]
    base = {
        "rho_bend": None,
        "pre_slope": None,
        "post_slope": None,
        "slope_ratio": None,
        "SSE_pre": None,
        "SSE_post": None,
        "SSE_total": None,
        "branch_start": None,
        "branch_end": None,
        "classification": "INVALID_NO_DISCERNIBLE_DIMINISHING_RETURN_BEND",
        "n_candidate_breaks": 0,
        "n_admissible_breaks": 0,
        "n_branch": 0,
        "imin": None,
    }
    if r.size < 8:
        base["n_branch"] = int(r.size)
        if r.size:
            base["branch_start"] = float(r[0])
            base["branch_end"] = float(r[-1])
        return base
    imin = int(np.argmin(d))
    r_b, d_b = r[: imin + 1], d[: imin + 1]
    base["branch_start"] = float(r_b[0])
    base["branch_end"] = float(r_b[-1])
    base["n_branch"] = int(r_b.size)
    base["imin"] = imin
    if r_b.size < 8:
        return base
    x = np.log10(r_b)
    n = int(x.size)
    n_cand = 0
    n_adm = 0
    best: Optional[Dict[str, Any]] = None
    for k in range(3, n - 3):
        n_cand += 1
        pre_s, _, sse_pre = ols_line(x[: k + 1], d_b[: k + 1])
        post_s, _, sse_post = ols_line(x[k:], d_b[k:])
        if not (pre_s < 0.0):
            continue
        if not (post_s <= 0.0):
            continue
        if not (abs(post_s) < abs(pre_s)):
            continue
        n_adm += 1
        sse_tot = sse_pre + sse_post
        rec = {
            "rho_bend": float(r_b[k]),
            "pre_slope": float(pre_s),
            "post_slope": float(post_s),
            "slope_ratio": float(abs(post_s) / abs(pre_s)) if abs(pre_s) > 0 else np.nan,
            "SSE_pre": float(sse_pre),
            "SSE_post": float(sse_post),
            "SSE_total": float(sse_tot),
            "k": int(k),
        }
        if best is None:
            best = rec
        else:
            better_sse = sse_tot < float(best["SSE_total"]) - 1e-15
            tie = abs(sse_tot - float(best["SSE_total"])) <= 1e-15
            if better_sse or (tie and rec["rho_bend"] < float(best["rho_bend"])):
                best = rec
    base["n_candidate_breaks"] = n_cand
    base["n_admissible_breaks"] = n_adm
    if best is None:
        return base
    base.update(best)
    base["classification"] = "VALID_POSITIVE_INTERIOR_BEND"
    return base


def curvature_sign_sensitivity(rho: np.ndarray, dist: np.ndarray) -> Dict[str, Any]:
    r = np.asarray(rho, dtype=float)
    d = np.asarray(dist, dtype=float)
    mask = np.isfinite(r) & np.isfinite(d) & np.array([is_rho_positive(float(x)) for x in r])
    r, d = r[mask], d[mask]
    order = np.argsort(r, kind="mergesort")
    r, d = r[order], d[order]
    out = {
        "has_persistent_sign_reversal": False,
        "reversal_rho": None,
        "new_sign": None,
        "n_second_diff": 0,
        "branch_end": None,
    }
    if r.size < 5:
        return out
    imin = int(np.argmin(d))
    r_b, d_b = r[: imin + 1], d[: imin + 1]
    out["branch_end"] = float(r_b[-1])
    if r_b.size < 5:
        return out
    x = np.log10(r_b)
    d1 = np.diff(d_b) / np.diff(x)
    d2 = d_b[2:] - 2.0 * d_b[1:-1] + d_b[:-2]
    out["n_second_diff"] = int(d2.size)
    signs = np.sign(d2)
    prev = 0.0
    for i, s in enumerate(signs):
        if s == 0:
            continue
        if prev != 0.0 and s != prev:
            nxt = signs[i : i + 4]
            if nxt.size >= 4 and np.all(nxt == s):
                out["has_persistent_sign_reversal"] = True
                out["reversal_rho"] = float(r_b[i + 1])
                out["new_sign"] = float(s)
                out["first_diff_at_reversal"] = float(d1[i])
                return out
        prev = float(s)
    return out


def metric_path(frame: pd.DataFrame, metric: str, suffix: str) -> Tuple[np.ndarray, np.ndarray]:
    sub = frame.sort_values("rho")
    rho = pd.to_numeric(sub["rho"], errors="coerce").to_numpy(dtype=float)
    vals = pd.to_numeric(sub[f"{metric}__{suffix}"], errors="coerce").to_numpy(dtype=float)
    return rho, vals


def lofo_suffix_values(frame: pd.DataFrame, metric: str, left_out: int) -> np.ndarray:
    mats = []
    for k in FOLD_IDS:
        if int(k) == int(left_out):
            continue
        mats.append(pd.to_numeric(frame[f"{metric}__fold_{k}"], errors="coerce").to_numpy(dtype=float))
    return np.nanmean(np.vstack(mats), axis=0)


def locate_relative_to_span(rho: Optional[float], low: Optional[float], high: Optional[float]) -> Optional[str]:
    if rho is None or low is None or high is None:
        return None
    if float(rho) < float(low) - 1e-15:
        return "below"
    if float(rho) > float(high) + 1e-15:
        return "above"
    return "inside"


def make_dirs(guard: OutputGuard) -> Dict[str, Path]:
    names = (
        "tables",
        "figures/main_candidate",
        "figures/appendix_candidate",
        "figures/diagnostic",
        "delta_nl_cv",
        "bend_analysis",
        "qa",
        "provenance",
    )
    return {n: guard.ensure_subdir(n) for n in names}


def search_forbidden(root: Path) -> List[str]:
    hits = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".pdf", ".png", ".parquet"}:
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except Exception:
            continue
        for phrase in FORBIDDEN_PHRASES:
            if phrase in text:
                hits.append(f"{path.relative_to(REPO)}:{phrase}")
    return hits


def figure_qa_row(**kwargs: Any) -> Dict[str, Any]:
    return kwargs


def run_preflight(guard: OutputGuard, problems: List[str]) -> Dict[str, Any]:
    frozen = {k: sha256_file(p) for k, p in frozen_named_files().items()}
    img_hash = hash_tree(PAPER_IMG)
    v2_hash = hash_tree(V2)
    v2_assets = hash_tree(ASSETS_V2)
    v3_hash = hash_tree(V3)
    delta_hash = sha256_file(REPO / "utils" / "delta_nl.py")
    payload = {
        "utc": utc_now(),
        "git_branch": git("branch", "--show-current"),
        "git_head": git("rev-parse", "HEAD"),
        "git_status_short": git("status", "--short").splitlines(),
        "frozen_named_files": frozen,
        "paper_img_generated_v12_994": img_hash,
        "v2_analysis_tree": v2_hash,
        "v2_paper_assets_tree": v2_assets,
        "v3_followup_tree": v3_hash,
        "utils_delta_nl_sha256": delta_hash,
        "no_model_fitting": True,
        "no_new_rho": True,
        "no_retuning": True,
        "no_change_to_frozen_prediction_cod_spans": True,
        "no_tex_compilation": True,
        "identity": {
            "data_id": DATA_ID,
            "split_id": SPLIT_ID,
            "seed": SEED,
            "trees": TREES,
            "folds": 7,
        },
    }
    guard.write_json(V4 / "provenance" / "preflight.json", payload)
    data = load_frozen()
    combined = data["combined"]
    span = data["span"]
    if len(combined) != 168:
        problems.append(f"combined row count {len(combined)} != 168")
    dlow, dhigh = family_span_endpoints(span, "Direct")
    slow, shigh = family_span_endpoints(span, "Surrogate")
    if not numerically_equal(dlow, FROZEN_DIRECT_SPAN[0]) or not numerically_equal(dhigh, FROZEN_DIRECT_SPAN[1]):
        problems.append(f"Direct span drifted: {[dlow, dhigh]}")
    if not numerically_equal(slow, FROZEN_SURR_SPAN[0]) or not numerically_equal(shigh, FROZEN_SURR_SPAN[1]):
        problems.append(f"Surrogate span drifted: {[slow, shigh]}")
    return {"preflight": payload, "data": data, "frozen": frozen, "img_hash": img_hash, "v2_hash": v2_hash, "v3_hash": v3_hash}


def run_decade(guard: OutputGuard, combined: pd.DataFrame, problems: List[str]) -> Dict[str, Any]:
    grid_pos = [float(x) for x in json.loads((EXT / "protocol" / "lower_rho_grid_v2.json").read_text(encoding="utf-8"))["augmented_positive_rhos"]]
    mapping_rows = []
    for rec in decade_nominal_to_tested(grid_pos):
        nom = float(rec["nominal_rho"])
        tested = float(rec["tested_rho"])
        expected = EXPECTED_DECADE_TESTED[nom]
        if not numerically_equal(tested, expected, atol=1e-12, rtol=1e-12):
            problems.append(f"decade mapping unexpected for {nom}: tested={tested} expected={expected}")
        mapping_rows.append(
            {
                **rec,
                "expected_tested_rho": expected,
                "mapping_ok": bool(numerically_equal(tested, expected, atol=1e-12, rtol=1e-12)),
                "family_scope": "shared_positive_grid",
                "provenance_hash": sha256_file(EXT / "protocol" / "lower_rho_grid_v2.json"),
                "grid_source": str((EXT / "protocol" / "lower_rho_grid_v2.json").relative_to(REPO)),
            }
        )
    map_df = pd.DataFrame(mapping_rows)
    written = write_table_bundle(guard, map_df, V4 / "tables" / "decade_anchor_mapping")
    primary, secondary = build_decade_tables(combined)
    written += write_table_bundle(guard, primary, V4 / "tables" / "decade_representative_rho_primary")
    written += write_table_bundle(guard, secondary, V4 / "tables" / "decade_representative_rho_secondary")
    return {"mapping": map_df, "primary": primary, "secondary": secondary, "written": written}


def run_inventory(guard: OutputGuard, combined: pd.DataFrame) -> Dict[str, Any]:
    inv, missing = inventory_cv_predictions(combined)
    guard.write_df(inv, V4 / "qa" / "cv_prediction_inventory.csv", parquet=False)
    payload = {
        "expected_positive": 2 * 82 * 7,
        "found_ok_positive": int(inv.loc[inv["required_positive_rho"] & inv["ok"]].shape[0]),
        "missing_positive": missing,
        "n_missing_positive": len(missing),
        "n_inventory_rows": int(len(inv)),
        "complete": len(missing) == 0,
    }
    guard.write_json(V4 / "qa" / "cv_prediction_inventory.json", payload)
    return {"inventory": inv, "missing": missing, "complete": len(missing) == 0, "summary": payload}


def run_delta_nl(guard: OutputGuard, inv: pd.DataFrame, workers: int) -> Dict[str, Any]:
    jobs = []
    for _, row in inv.iterrows():
        if not bool(row["required_positive_rho"]):
            continue
        if not bool(row["ok"]):
            continue
        jobs.append(
            {
                "family": row["family"],
                "rho": float(row["rho"]),
                "config_id": row["config_id"],
                "fold_id": int(row["fold_id"]),
                "run_id": row["run_id"],
                "prediction_path": row["prediction_path"],
                "prediction_sha256": row["prediction_sha256"],
            }
        )
    if not jobs:
        raise RuntimeError("no complete positive-rho CV prediction jobs for Delta_NL")
    t0 = time.time()
    smoke = _delta_nl_job(jobs[0])
    smoke_s = time.time() - t0
    guard.write_json(V4 / "delta_nl_cv" / "smoke_test.json", {"seconds": smoke_s, "job": jobs[0], "result": smoke})
    results = [None] * len(jobs)
    results[0] = smoke
    remaining = list(enumerate(jobs))[1:]
    done = 1
    if remaining:
        with ProcessPoolExecutor(max_workers=max(1, int(workers))) as ex:
            futs = {ex.submit(_delta_nl_job, job): i for i, job in remaining}
            for fut in as_completed(futs):
                i = futs[fut]
                results[i] = fut.result()
                done += 1
                if done % 50 == 0 or done == len(jobs):
                    print(f"delta_nl progress {done}/{len(jobs)}", flush=True)
    by_fold = pd.DataFrame(results)
    spec_h = estimator_spec_hash()
    if not (by_fold["estimator_spec_hash"] == spec_h).all():
        raise RuntimeError("estimator_spec_hash mismatch in CV Delta_NL rows")
    mean_rows = []
    for (fam, rho, cid), g in by_fold.groupby(["family", "rho", "config_id"], sort=True):
        if len(g) != 7:
            raise RuntimeError(f"expected 7 folds for {fam} rho={rho}, got {len(g)}")
        vals = g.sort_values("fold_id")["Delta_NL"].to_numpy(dtype=float)
        mean_rows.append(
            {
                "family": fam,
                "rho": float(rho),
                "config_id": cid,
                "n_folds": 7,
                "Delta_NL__CV_mean": float(np.mean(vals)),
                "Delta_NL__CV_sd": float(np.std(vals, ddof=1)),
                "estimator_spec_hash": spec_h,
            }
        )
    mean_df = pd.DataFrame(mean_rows)
    write_table_bundle(guard, by_fold, V4 / "delta_nl_cv" / "delta_nl_cv_by_fold")
    write_table_bundle(guard, mean_df, V4 / "delta_nl_cv" / "delta_nl_cv_mean")
    guard.write_json(V4 / "delta_nl_cv" / "estimator_spec.json", {"spec": estimator_spec(), "hash": spec_h})
    guard.write_json(
        V4 / "delta_nl_cv" / "provenance.json",
        {
            "utc": utc_now(),
            "n_jobs": len(jobs),
            "smoke_seconds": smoke_s,
            "workers": workers,
            "estimator_spec_hash": spec_h,
            "no_model_fitting": True,
            "no_concatenation_of_chronological_folds": True,
        },
    )
    return {"by_fold": by_fold, "mean": mean_df, "estimator_spec_hash": spec_h, "smoke_seconds": smoke_s}


def join_v4_view(guard: OutputGuard, combined: pd.DataFrame, dnl_fold: pd.DataFrame, dnl_mean: pd.DataFrame) -> pd.DataFrame:
    out = combined.copy()
    for k in FOLD_IDS:
        out[f"Delta_NL__fold_{k}"] = np.nan
    out["Delta_NL__CV_mean"] = np.nan
    out["Delta_NL__CV_sd"] = np.nan
    for _, row in dnl_fold.iterrows():
        mask = (out["family"] == row["family"]) & np.isclose(pd.to_numeric(out["rho"], errors="coerce"), float(row["rho"]), atol=1e-12)
        out.loc[mask, f"Delta_NL__fold_{int(row['fold_id'])}"] = float(row["Delta_NL"])
    for _, row in dnl_mean.iterrows():
        mask = (out["family"] == row["family"]) & np.isclose(pd.to_numeric(out["rho"], errors="coerce"), float(row["rho"]), atol=1e-12)
        out.loc[mask, "Delta_NL__CV_mean"] = float(row["Delta_NL__CV_mean"])
        out.loc[mask, "Delta_NL__CV_sd"] = float(row["Delta_NL__CV_sd"])
    for fam in FAMILY_DISPLAY:
        m = out["family"] == fam
        pos = np.array([is_rho_positive(float(x)) for x in out.loc[m, "rho"]])
        for k in FOLD_IDS:
            if not np.isfinite(pd.to_numeric(out.loc[m, f"Delta_NL__fold_{k}"], errors="coerce").to_numpy()[pos]).all():
                raise RuntimeError(f"v4 join missing Delta_NL fold {k} for {fam}")
        if not np.isfinite(pd.to_numeric(out.loc[m, "Delta_NL__CV_mean"], errors="coerce").to_numpy()[pos]).all():
            raise RuntimeError(f"v4 join missing Delta_NL CV mean for {fam}")
    guard.write_df(out, V4 / "tables" / "combined_path_table_v4_analysis_view.csv", parquet=True)
    guard.write_json(
        V4 / "qa" / "v2_byte_identity.json",
        {
            "combined_path_table_v2_sha256": sha256_file(V2 / "tables" / "combined_path_table_v2.csv"),
            "note": "v2 table was not mutated; v4 is a derived join only",
        },
    )
    return out


def build_mech_events(combined: pd.DataFrame, span: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    splits = [("cv_mean", "CV_mean")] + [(f"fold_{k}", f"fold_{k}") for k in FOLD_IDS] + [
        ("heldout", "heldout"),
        ("forward_2025", "forward_2025"),
    ]
    for fam in FAMILY_DISPLAY:
        sub = family_frame(combined, fam)
        rhos = sub["rho"].to_numpy(dtype=float)
        low, high = family_span_endpoints(span, fam)
        for metric, kind, target, event_kind in MECH_EVENT_SPECS_V4:
            for split, suffix in splits:
                col = f"{metric}__{split}" if split.startswith("fold_") else f"{metric}__{suffix}"
                if col not in sub.columns:
                    continue
                vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
                mask = np.isfinite(vals)
                ev, original, abs_gap = extract_one_event(rhos[mask], vals[mask], metric, kind, target)
                event_rho = ev.rho_low
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
                        "classification": ev.classification,
                        "n_tied": ev.n_tied,
                        "metric_value": original,
                        "abs_gap_to_target": abs_gap,
                        "inside_frozen_five_metric_cv_span": inside,
                        "does_not_redefine_transition_span": True,
                    }
                )
    return pd.DataFrame(rows)


def run_bends(guard: OutputGuard, combined: pd.DataFrame, span: pd.DataFrame) -> Dict[str, Any]:
    mean_rows: List[Dict[str, Any]] = []
    fold_rows: List[Dict[str, Any]] = []
    lofo_rows: List[Dict[str, Any]] = []
    curv_rows: List[Dict[str, Any]] = []
    span_rows: List[Dict[str, Any]] = []
    for fam in FAMILY_DISPLAY:
        frame = family_frame(combined, fam).sort_values("rho").reset_index(drop=True)
        # CV mean
        cv_bends = []
        for metric, target, kind in CORE_BEND_METRICS:
            rho, vals = metric_path(frame, metric, "CV_mean")
            dist = target_distance(vals, kind, target)
            ev = primary_two_segment_bend(rho, dist)
            ev.update({"family": fam, "metric": metric, "path": "cv_mean", "target": target, "distance_kind": kind})
            mean_rows.append(ev)
            cv_bends.append(ev)
            curv = curvature_sign_sensitivity(rho, dist)
            curv.update({"family": fam, "metric": metric, "path": "cv_mean"})
            curv_rows.append(curv)
        for k in FOLD_IDS:
            fb = []
            for metric, target, kind in CORE_BEND_METRICS:
                rho, vals = metric_path(frame, metric, f"fold_{k}")
                dist = target_distance(vals, kind, target)
                ev = primary_two_segment_bend(rho, dist)
                ev.update({"family": fam, "metric": metric, "path": f"fold_{k}", "fold_id": int(k), "target": target, "distance_kind": kind})
                fold_rows.append(ev)
                fb.append(ev)
                curv = curvature_sign_sensitivity(rho, dist)
                curv.update({"family": fam, "metric": metric, "path": f"fold_{k}", "fold_id": int(k)})
                curv_rows.append(curv)
        lofo_ok_all = True
        lofo_spans = []
        for left in FOLD_IDS:
            lb = []
            for metric, target, kind in CORE_BEND_METRICS:
                rho = pd.to_numeric(frame["rho"], errors="coerce").to_numpy(dtype=float)
                vals = lofo_suffix_values(frame, metric, int(left))
                dist = target_distance(vals, kind, target)
                ev = primary_two_segment_bend(rho, dist)
                ev.update(
                    {
                        "family": fam,
                        "metric": metric,
                        "path": f"lofo_leave_{left}",
                        "left_out_fold": int(left),
                        "target": target,
                        "distance_kind": kind,
                    }
                )
                lofo_rows.append(ev)
                lb.append(ev)
            valid = [e for e in lb if e["classification"] == "VALID_POSITIVE_INTERIOR_BEND" and e["rho_bend"] is not None]
            ok = len(valid) == 6
            if not ok:
                lofo_ok_all = False
                lofo_spans.append({"left_out_fold": int(left), "valid": False, "rho_low": None, "rho_high": None})
            else:
                rhos = [float(e["rho_bend"]) for e in valid]
                lofo_spans.append(
                    {
                        "left_out_fold": int(left),
                        "valid": True,
                        "rho_low": float(min(rhos)),
                        "rho_high": float(max(rhos)),
                    }
                )
        valid_cv = [e for e in cv_bends if e["classification"] == "VALID_POSITIVE_INTERIOR_BEND" and e["rho_bend"] is not None]
        cv_ok = len(valid_cv) == 6
        cand_low = float(min(float(e["rho_bend"]) for e in valid_cv)) if cv_ok else None
        cand_high = float(max(float(e["rho_bend"]) for e in valid_cv)) if cv_ok else None
        # Delta_NL sensitivity
        rho, dnl = metric_path(frame, "Delta_NL", "CV_mean")
        mask = np.isfinite(dnl) & np.array([is_rho_positive(float(x)) for x in rho])
        dnl_min_ev, dnl_min_val, _ = extract_one_event(rho[mask], dnl[mask], "Delta_NL", "min", None)
        dnl_bend = primary_two_segment_bend(rho, dnl)
        span_rows.append(
            {
                "family": fam,
                "cv_mean_all_six_valid": bool(cv_ok),
                "lofo_all_seven_valid": bool(lofo_ok_all),
                "rho_bend_low": cand_low,
                "rho_bend_high": cand_high,
                "status": "VALID" if cv_ok else "INVALID",
                "valid": bool(cv_ok),
                "n_lofo_valid_spans": int(sum(1 for x in lofo_spans if x["valid"])),
                "Delta_NL_min_rho": dnl_min_ev.rho_low,
                "Delta_NL_min_value": dnl_min_val,
                "Delta_NL_bend_rho": dnl_bend.get("rho_bend"),
                "Delta_NL_bend_classification": dnl_bend.get("classification"),
                "Delta_NL_min_vs_candidate": locate_relative_to_span(dnl_min_ev.rho_low, cand_low, cand_high),
                "Delta_NL_bend_vs_candidate": locate_relative_to_span(dnl_bend.get("rho_bend"), cand_low, cand_high),
                "frozen_pred_cod_low": family_span_endpoints(span, fam)[0],
                "frozen_pred_cod_high": family_span_endpoints(span, fam)[1],
                "lofo_spans_json": json.dumps(lofo_spans),
            }
        )
    mean_df = pd.DataFrame(mean_rows)
    fold_df = pd.DataFrame(fold_rows)
    lofo_df = pd.DataFrame(lofo_rows)
    curv_df = pd.DataFrame(curv_rows)
    span_df = pd.DataFrame(span_rows)
    write_table_bundle(guard, mean_df, V4 / "bend_analysis" / "equity_mechanism_bend_events_cv_mean")
    write_table_bundle(guard, fold_df, V4 / "bend_analysis" / "equity_mechanism_bend_events_by_fold")
    write_table_bundle(guard, lofo_df, V4 / "bend_analysis" / "equity_mechanism_bend_lofo")
    write_table_bundle(guard, span_df, V4 / "bend_analysis" / "equity_mechanism_bend_span_summary")
    write_table_bundle(guard, curv_df, V4 / "bend_analysis" / "curvature_sign_sensitivity")

    reasons = []
    pass_ok = True
    for _, row in span_df.iterrows():
        if not bool(row["cv_mean_all_six_valid"]):
            pass_ok = False
            reasons.append(f"{row['family']}: not all six CV-mean bends valid")
        if not bool(row["lofo_all_seven_valid"]):
            pass_ok = False
            reasons.append(f"{row['family']}: LOFO 7/7 six-metric common spans failed ({row['n_lofo_valid_spans']}/7)")
    status = {
        "FINAL_BEND_STATUS": "PASS" if pass_ok else "FAIL",
        "reason": "; ".join(reasons) if reasons else "all six CV-mean bends valid and all 7 LOFO means produce valid six-metric common bend spans",
        "post_hoc": True,
        "does_not_select_rho": True,
        "does_not_redefine_prediction_cod_span": True,
        "introduced_after_oos_inspection": True,
        "families": span_df.to_dict(orient="records"),
    }
    guard.write_json(V4 / "bend_analysis" / "FINAL_BEND_STATUS.json", status)

    ctx_rows = []
    if any(bool(x) for x in span_df["cv_mean_all_six_valid"]):
        for fam in FAMILY_DISPLAY:
            srow = span_df.loc[span_df["family"] == fam].iloc[0]
            if not bool(srow["cv_mean_all_six_valid"]):
                continue
            frame = family_frame(combined, fam)
            for endpoint, rho in (("low", float(srow["rho_bend_low"])), ("high", float(srow["rho_bend_high"]))):
                rec = combined_row(combined, fam, rho)
                for split, suffix in [("cv_mean", "CV_mean"), ("heldout", "heldout"), ("forward_2025", "forward_2025")] + [
                    (f"fold_{k}", f"fold_{k}") for k in FOLD_IDS
                ]:
                    rowc: Dict[str, Any] = {
                        "family": fam,
                        "endpoint": endpoint,
                        "rho": rho,
                        "split": split,
                        "descriptive_context_only": True,
                        "does_not_define_a_third_band": True,
                    }
                    for met in LEVEL_CONTEXT_METRICS:
                        col = f"{met}__{suffix}"
                        rowc[met] = float(rec[col]) if col in rec.index and pd.notna(rec[col]) else None
                    ctx_rows.append(rowc)
        if ctx_rows:
            write_table_bundle(guard, pd.DataFrame(ctx_rows), V4 / "bend_analysis" / "bend_span_level_uniformity_context")
    return {
        "status": status,
        "mean": mean_df,
        "fold": fold_df,
        "lofo": lofo_df,
        "span": span_df,
        "curv": curv_df,
        "pass": pass_ok,
    }


def plot_bend_audit(plt, mean_df, fold_df, lofo_df, span_df, pred_span, min_pos, q, stem, two_band: bool) -> List[str]:
    import utils.paper_v12_lower_rho_plots as P
    from matplotlib.lines import Line2D

    labels = {
        "PRD": "PRD",
        "PRB": "PRB",
        "MKI": "MKI",
        "VEI": "VEI",
        "Beta_log": r"$\beta_{\log}$",
        "dCor_e_y": r"dCor$(e,y)$",
    }
    metric_order = [m for m, _, _ in CORE_BEND_METRICS]
    ymap = {m: i for i, m in enumerate(reversed(metric_order))}
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 5.1), sharey=True)
    cls = "PAPER-CANDIDATE" if two_band else "DIAGNOSTIC-ONLY"
    note = (
        "Diminishing-return bends of target-distance paths on log10 rho; post-hoc CV diagnostic; "
        "distinct from prediction/COD turning events and exact neutrality events; does not select rho."
    )
    for ax, fam in zip(axes, FAMILY_DISPLAY):
        color = P.DIRECT_COLOR if fam == "Direct" else P.SURR_COLOR
        low, high, ok = P.family_span(pred_span, fam)
        P.shade_spans_for_path(ax, low, high, ok, v3=True, two_band=two_band, bend_df=span_df, family=fam)
        part_m = mean_df.loc[mean_df["family"] == fam]
        part_f = fold_df.loc[fold_df["family"] == fam]
        part_l = lofo_df.loc[lofo_df["family"] == fam]
        for metric in metric_order:
            y = ymap[metric]
            folds = part_f.loc[part_f["metric"] == metric]
            xf = P.rho_plot_x(pd.to_numeric(folds["rho_bend"], errors="coerce").to_numpy(dtype=float), min_positive=min_pos, q=q)
            ax.scatter(xf, np.full_like(xf, y, dtype=float), s=11, color=color, alpha=0.35, zorder=4)
            lof = part_l.loc[part_l["metric"] == metric]
            xl = P.rho_plot_x(pd.to_numeric(lof["rho_bend"], errors="coerce").to_numpy(dtype=float), min_positive=min_pos, q=q)
            ax.scatter(xl, np.full_like(xl, y - 0.18, dtype=float), s=16, marker="x", color="#6B7280", zorder=5)
            cv = part_m.loc[part_m["metric"] == metric]
            if not cv.empty and pd.notna(cv.iloc[0]["rho_bend"]):
                ax.scatter(
                    P.rho_plot_x([float(cv.iloc[0]["rho_bend"])], min_positive=min_pos, q=q)[0],
                    y,
                    s=70,
                    marker="o",
                    color=color,
                    zorder=6,
                    edgecolors="white",
                    linewidths=0.6,
                )
        P.log_rho_axes(ax, min_positive=min_pos, q=q)
        P.apply_major_grid(ax)
        ax.set_ylim(-1.05, float(len(metric_order)) - 0.25)
        ax.set_yticks(list(ymap.values()))
        ax.set_yticklabels([labels[m] for m in reversed(metric_order)])
        ax.set_title(fam)
        ax.set_xlabel(r"Penalty strength $\rho$")
        ax.text(0.02, 0.98, f"{cls}", transform=ax.transAxes, fontsize=6.4, va="top", color="#111827")
        ax.text(0.02, 0.02, note, transform=ax.transAxes, fontsize=5.4, va="bottom", color="#374151")
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1D4ED8", ms=8, label="CV-mean bend"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#1D4ED8", ms=4, alpha=0.4, label="Fold bends"),
        Line2D([0], [0], marker="x", color="#6B7280", ms=6, lw=0, label="LOFO-mean bends"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return P._save(plt, fig, stem)


def run_figures(
    guard: OutputGuard,
    combined: pd.DataFrame,
    span: pd.DataFrame,
    grid: Dict[str, Any],
    mech_events: pd.DataFrame,
    bend: Optional[Dict[str, Any]],
    dnl_ok: bool,
) -> List[Dict[str, Any]]:
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import utils.paper_v12_lower_rho_plots as P

    min_pos = float(grid["min_positive_augmented"])
    q = float(grid["q"])
    two_band = bool(bend and bend["pass"])
    bend_span = None if bend is None else bend["span"]
    fig_qa: List[Dict[str, Any]] = []
    main = V4 / "figures" / "main_candidate"
    app = V4 / "figures" / "appendix_candidate"
    diag = V4 / "figures" / "diagnostic"
    extra = (EXT,)
    kw = dict(v3=True, decade=True, extra_roots=extra)
    P.plot_ratio_shape(plt, CANONICAL, combined, main / "ratio_shape_evolution", **kw)
    fig_qa.append(figure_qa_row(name="ratio_shape_evolution", location="main_candidate", span_fill=False, decade=True))
    P.plot_ratio_shape_span_only(plt, CANONICAL, combined, span, app / "ratio_shape_cv_transition_span_only", **kw)
    fig_qa.append(figure_qa_row(name="ratio_shape_cv_transition_span_only", location="appendix_candidate", span_fill=False, decade=True))

    path_kw = dict(v3=True, two_band=two_band, bend_span_df=bend_span)
    if dnl_ok:
        qa_mean: List[Dict[str, Any]] = []
        P.plot_cv_group(
            plt,
            combined,
            span,
            (
                ("Beta_log", r"$\beta_{\log}$"),
                ("Delta_NL", r"$\Delta_{\mathrm{NL}}$"),
                ("dCor_e_y", r"$\mathrm{dCor}(e,y)$"),
            ),
            min_pos,
            q,
            app / "cv_mechanism_metric_paths",
            qa_mean,
            **path_kw,
        )
        fig_qa.append(figure_qa_row(name="cv_mechanism_metric_paths", location="appendix_candidate", two_band=two_band, cv_mean_ok=all(x.get("ok") for x in qa_mean)))
        guard.write_json(V4 / "qa" / "cv_mean_fold_checks.json", qa_mean)
        note = (
            "Descriptive mechanism events. Distinct from the frozen prediction/COD transition span. Does not select rho."
        )
        P.plot_descriptive_event_locations(
            plt,
            mech_events,
            span,
            min_pos,
            q,
            app / "mechanism_event_locations",
            labels={"Beta_log": r"$\beta_{\log}$", "Delta_NL": r"$\Delta_{\mathrm{NL}}$", "dCor_e_y": r"dCor$(e,y)$"},
            metric_order=["Beta_log", "Delta_NL", "dCor_e_y"],
            note=note,
            two_band=two_band,
            bend_span_df=bend_span,
        )
        fig_qa.append(figure_qa_row(name="mechanism_event_locations", location="appendix_candidate", two_band=two_band))
        P.plot_mechanism(plt, combined, span, min_pos, q, main / "mechanism_vs_rho", **path_kw)
        fig_qa.append(figure_qa_row(name="mechanism_vs_rho", location="main_candidate", two_band=two_band))

    if two_band:
        P.plot_predictive(plt, combined, span, min_pos, q, app / "predictive_metric_paths", **path_kw)
        P.plot_level_uniformity(plt, combined, span, min_pos, q, app / "level_uniformity_paths", **path_kw)
        P.plot_vertical_equity(plt, combined, span, min_pos, q, app / "vertical_equity_metric_paths", **path_kw)
        qa2: List[Dict[str, Any]] = []
        P.plot_cv_group(plt, combined, span, (("R2_price", r"$R^2_P$"), ("MAE_price", r"MAE$_P$"), ("MAPE", r"MAPE$_P$ (\%)"), ("RMSE_log", r"RMSE$_{\log P}$")), min_pos, q, app / "cv_predictive_metric_paths", qa2, **path_kw)
        qa3: List[Dict[str, Any]] = []
        P.plot_cv_group(plt, combined, span, (("median_ratio", "Median ratio"), ("mean_ratio", "Mean ratio"), ("weighted_mean_ratio", "Weighted mean ratio"), ("COD", "COD"), ("COV", "COV (\%)")), min_pos, q, app / "cv_level_uniformity_paths", qa3, **path_kw)
        qa4: List[Dict[str, Any]] = []
        P.plot_cv_group(plt, combined, span, (("PRD", "PRD"), ("PRB", "PRB"), ("MKI", "MKI"), ("VEI", "VEI")), min_pos, q, app / "cv_vertical_equity_metric_paths", qa4, **path_kw)
        fig_qa.extend(
            [
                figure_qa_row(name="predictive_metric_paths", two_band=True),
                figure_qa_row(name="level_uniformity_paths", two_band=True),
                figure_qa_row(name="vertical_equity_metric_paths", two_band=True),
                figure_qa_row(name="cv_predictive_metric_paths", two_band=True),
                figure_qa_row(name="cv_level_uniformity_paths", two_band=True),
                figure_qa_row(name="cv_vertical_equity_metric_paths", two_band=True),
            ]
        )
        # vertical equity events already exist from v3; regenerate with two bands if file of events exists
        ve_path = V3 / "tables" / "vertical_equity_event_locations.csv"
        if not ve_path.is_file():
            ve_path = V3 / "tables" / "descriptive_vertical_equity_events.csv"
        # optional; skip if missing

    if bend is not None:
        plot_bend_audit(
            plt,
            bend["mean"],
            bend["fold"],
            bend["lofo"],
            bend["span"],
            span,
            min_pos,
            q,
            (app if two_band else diag) / "equity_mechanism_bend_event_locations",
            two_band=two_band,
        )
        fig_qa.append(
            figure_qa_row(
                name="equity_mechanism_bend_event_locations",
                classification="PAPER-CANDIDATE" if two_band else "DIAGNOSTIC-ONLY",
                two_band=two_band,
            )
        )
    guard.write_json(V4 / "qa" / "figure_qa_manifest.json", fig_qa)
    return fig_qa


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["all", "preflight", "decade", "inventory", "delta_nl", "bend", "figures"])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--reuse-delta-nl", action="store_true", help="Reuse existing v4 CV Delta_NL tables if present")
    args = parser.parse_args()
    problems: List[str] = []
    guard = OutputGuard(V4, REPO)
    make_dirs(guard)
    pre = run_preflight(guard, problems)
    data = pre["data"]
    combined = data["combined"]
    span = data["span"]
    grid = data["grid"]
    if args.phase == "preflight":
        guard.write_json(V4 / "qa" / "problems.json", problems)
        print(json.dumps({"phase": "preflight", "problems": problems, "head": pre["preflight"]["git_head"]}, indent=2))
        return 0 if not problems else 1

    decade = run_decade(guard, combined, problems)
    if args.phase == "decade":
        guard.write_json(V4 / "qa" / "problems.json", problems)
        print(decade["mapping"].to_string(index=False))
        return 0 if not problems else 1

    inv = run_inventory(guard, combined)
    print(json.dumps(inv["summary"], indent=2, default=str)[:4000])
    dnl_ok = bool(inv["complete"])
    dnl = None
    combined_v4 = combined
    bend = None
    mech = None
    if not dnl_ok:
        problems.append("CV prediction inventory incomplete; stopping Delta_NL/bend")
        guard.write_json(V4 / "qa" / "MISSING_CV_PREDICTIONS.json", inv["summary"])
    if args.phase == "inventory":
        guard.write_json(V4 / "qa" / "problems.json", problems)
        return 0 if dnl_ok and not problems else 2

    if dnl_ok and args.phase in {"all", "delta_nl", "bend", "figures"}:
        mean_path = V4 / "delta_nl_cv" / "delta_nl_cv_mean.csv"
        fold_path = V4 / "delta_nl_cv" / "delta_nl_cv_by_fold.csv"
        if args.reuse_delta_nl and mean_path.is_file() and fold_path.is_file():
            dnl = {
                "by_fold": pd.read_csv(fold_path),
                "mean": pd.read_csv(mean_path),
                "estimator_spec_hash": estimator_spec_hash(),
                "smoke_seconds": None,
            }
            print("reusing existing CV Delta_NL tables", flush=True)
        else:
            dnl = run_delta_nl(guard, inv["inventory"], args.workers)
        combined_v4 = join_v4_view(guard, combined, dnl["by_fold"], dnl["mean"])
        if args.phase == "delta_nl":
            guard.write_json(V4 / "qa" / "problems.json", problems)
            print("Delta_NL done", dnl["estimator_spec_hash"], "smoke_s", dnl["smoke_seconds"])
            return 0 if not problems else 1

    if dnl_ok and args.phase in {"all", "bend", "figures"}:
        mech = build_mech_events(combined_v4, span)
        write_table_bundle(guard, mech, V4 / "tables" / "mechanism_event_locations_v4")
        bend = run_bends(guard, combined_v4, span)
        if args.phase == "bend":
            print(json.dumps(bend["status"], indent=2, default=str)[:5000])
            guard.write_json(V4 / "qa" / "problems.json", problems)
            return 0 if not problems else 1

    if args.phase in {"all", "figures"}:
        # decade figures always; Delta_NL/bend figures if available
        view = combined_v4 if dnl_ok else combined
        run_figures(guard, view, span, grid, mech if mech is not None else pd.DataFrame(), bend, dnl_ok)

    forbidden = search_forbidden(V4)
    if forbidden:
        problems.append(f"forbidden phrases: {forbidden}")
    # immutability
    for k, p in frozen_named_files().items():
        if k == "paper/paper_v12.tex":
            continue
        got = sha256_file(p)
        if got != pre["frozen"][k]:
            problems.append(f"frozen hash changed: {k}")
    if hash_tree(V2) != pre["v2_hash"]:
        problems.append("v2 analysis tree hash changed")
    if hash_tree(V3) != pre["v3_hash"]:
        problems.append("v3 followup tree hash changed")
    tex_created = [str(p) for p in V4.rglob("*.tex")]
    if tex_created:
        problems.append(f".tex files created under v4: {tex_created}")
    status = {
        "utc": utc_now(),
        "problems": problems,
        "inventory_complete": dnl_ok,
        "FINAL_BEND_STATUS": None if bend is None else bend["status"]["FINAL_BEND_STATUS"],
        "no_model_fitting": True,
        "no_new_rho": True,
        "no_retuning": True,
        "no_tex_compilation": True,
    }
    guard.write_json(V4 / "qa" / "FINAL_STATUS.json", status)
    print(json.dumps(status, indent=2, default=str)[:6000])
    return 0 if not problems else 1


if __name__ == "__main__":
    try:
        code = main()
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
    os._exit(code)
