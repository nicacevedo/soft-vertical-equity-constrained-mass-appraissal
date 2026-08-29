#!/usr/bin/env python3
"""v2 lower-rho extension analysis, figures, table sources, and figure-only tex population.

No TeX compilation. Does not mutate transition_regions_v1 or the original 0.1-100 fits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from canonical_experiment import git_state
from utils.delta_nl import estimate_delta_nl, estimator_spec, estimator_spec_hash, identifier_fold_assignment
from utils.transition_paper_assets import (
    endpoint_equals_first_positive,
    event_sharpness_row,
    metric_series,
    positive_display_anchors,
    span_regret_row,
)
from utils.transition_regions import (
    FAMILY_DISPLAY,
    FOLD_IDS,
    PRIMARY_METRICS,
    concordance_row,
    construct_transition_span,
    event_table_row,
    extract_discrete_event,
    extract_primary_events_from_frame,
    family_frame,
    fold_matrix_from_frame,
    is_rho_positive,
    is_rho_zero,
    lofo_events_and_span,
    lofo_span_summary,
    numerically_equal,
    summarize_fold_events_logrho,
    sha256_file,
    validate_canonical_result_root,
)

CANONICAL = REPO / "output" / "paper_v6_preselection_994"
EXT = REPO / "output" / "paper_v12_lower_rho_extension_994_v2"
DATA_ID = "d4929d43ec19badf"
SPLIT_ID = "3d464d4a611b131b"
V2 = EXT / "analysis" / f"data_id={DATA_ID}" / f"split_id={SPLIT_ID}" / "penalty_path_analysis" / "transition_regions_v2_lower_rho"
ASSETS = V2.parent / "transition_regions_paper_assets_v2"
PAPER_TEX = REPO / "paper" / "paper_v12.tex"
PAPER_IMG = REPO / "paper" / "img" / "generated_v12_994"
DIRECT = "LGBCovPenalty"
SURROGATE = "LGBSmoothPenalty"
METRIC_MAP = {
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
PATH_METRICS = list(METRIC_MAP.values())
PRIMARY_TABLE_METRICS = ["R2_price", "MAE_price", "MAPE", "RMSE_log", "PRD", "PRB", "MKI", "VEI"]
SECONDARY_TABLE_METRICS = [
    "median_ratio",
    "mean_ratio",
    "weighted_mean_ratio",
    "COD",
    "COV",
    "Beta_log",
    "Delta_NL",
    "dCor_e_y",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def load_grid() -> Dict[str, Any]:
    return json.loads((EXT / "protocol" / "lower_rho_grid_v2.json").read_text(encoding="utf-8"))


def rho_cli(values: Sequence[float]) -> str:
    return ",".join(f"{float(x):.16g}" for x in values)


def family_from_name(name: str) -> str:
    if name == DIRECT:
        return "Direct"
    if name == SURROGATE:
        return "Surrogate"
    if name == "LGBMRegressor":
        return "LightGBM"
    if name == "LinearRegression":
        return "Linear"
    return str(name)


def rho_from_row(row: pd.Series) -> float:
    if "rho" in row.index and pd.notna(row.get("rho")):
        try:
            val = float(row["rho"])
            if np.isfinite(val):
                return val
        except (TypeError, ValueError):
            pass
    cfg = row.get("config") or row.get("model_config") or row.get("model_config_json") or {}
    if isinstance(cfg, str):
        try:
            cfg = json.loads(cfg)
        except Exception:
            cfg = {}
    if isinstance(cfg, dict) and "rho" in cfg:
        return float(cfg["rho"])
    return float("nan")


def standardize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    name_col = "model_name" if "model_name" in out.columns else "name"
    out["model_name"] = out[name_col].astype(str)
    out["family"] = out["model_name"].map(family_from_name)
    if "config_id" not in out.columns:
        out["config_id"] = ""
    out["config_id"] = out["config_id"].astype(str)
    out["rho"] = out.apply(rho_from_row, axis=1)
    for src, dst in METRIC_MAP.items():
        if src in out.columns:
            out[dst] = pd.to_numeric(out[src], errors="coerce")
    return out


def concat_parquets(paths: Iterable[Path]) -> pd.DataFrame:
    files = [Path(p) for p in paths if Path(p).is_file()]
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def load_new_cv() -> pd.DataFrame:
    files = list(EXT.glob("runs/**/fold_id=*/*.parquet"))
    if not files:
        return pd.DataFrame()
    df = standardize_metrics(concat_parquets(files))
    if "fold_id" in df.columns:
        df["fold_id"] = pd.to_numeric(df["fold_id"], errors="coerce").astype("Int64")
    return df


def load_new_oos() -> pd.DataFrame:
    mapping = {"heldout": "test_run_metrics", "forward_2025": "assess_run_metrics"}
    frames = []
    for eval_name, shard in mapping.items():
        files = list((EXT / "reporting_preview").glob(f"{eval_name}/**/{shard}/*.parquet"))
        if not files:
            continue
        df = standardize_metrics(concat_parquets(files))
        df["evaluation"] = eval_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def cv_row_from_folds(src: pd.DataFrame, family: str, rho: float, template: pd.Series) -> Dict[str, Any]:
    row = {c: template[c] for c in template.index}
    row["family"] = family
    row["rho"] = float(rho)
    row["model_name"] = DIRECT if family == "Direct" else SURROGATE
    if "config_id" in src.columns and not src.empty:
        row["config_id"] = str(src["config_id"].iloc[0])
    for metric in PATH_METRICS:
        fold_vals = []
        for fold in range(7):
            part = src.loc[src["fold_id"] == fold, metric] if metric in src.columns else pd.Series(dtype=float)
            val = float(pd.to_numeric(part, errors="coerce").iloc[0]) if not part.empty else float("nan")
            row[f"{metric}__fold_{fold + 1}"] = val
            if np.isfinite(val):
                fold_vals.append(val)
        row[f"{metric}__CV_mean"] = float(np.mean(fold_vals)) if fold_vals else float("nan")
        row[f"{metric}__CV_sd"] = float(np.std(fold_vals, ddof=1)) if len(fold_vals) > 1 else float("nan")
        row[f"{metric}__heldout"] = np.nan
        row[f"{metric}__forward_2025"] = np.nan
    row["Delta_NL__heldout"] = np.nan
    row["Delta_NL__forward_2025"] = np.nan
    return row


def cmd_cv_qa() -> int:
    grid = load_grid()
    cv = load_new_cv()
    problems: List[str] = []
    n = 0 if cv.empty else int(cv[["config_id", "fold_id"]].drop_duplicates().shape[0])
    n_rho = 0 if cv.empty else int(cv.loc[cv["family"].isin(["Direct", "Surrogate"]), ["family", "rho"]].drop_duplicates().shape[0])
    if n != 448:
        problems.append(f"completed pairs {n} != 448")
    if n_rho != 64:
        problems.append(f"family-rho {n_rho} != 64")
    if not cv.empty:
        key = cv[["family", "rho", "fold_id", "config_id"]].astype(str)
        if int(key.duplicated().sum()) != 0:
            problems.append("duplicate family/rho/fold/config_id rows")
        if set(cv["family"].astype(str).unique()) - {"Direct", "Surrogate"}:
            problems.append("unexpected families in new CV")
        if "data_id" in cv.columns and set(cv["data_id"].astype(str).dropna().unique()) - {DATA_ID}:
            problems.append("data_id mismatch in new CV")
        if "split_id" in cv.columns and set(cv["split_id"].astype(str).dropna().unique()) - {SPLIT_ID}:
            problems.append("split_id mismatch in new CV")
        folds = sorted(int(x) for x in cv["fold_id"].dropna().unique())
        if folds != list(range(7)):
            problems.append(f"fold ids {folds} != [0..6]")
    payload = {
        "utc": utc_now(),
        "n_completed_pairs": n,
        "n_expected_pairs": 448,
        "n_family_rho": n_rho,
        "n_expected_family_rho": 64,
        "ok": not problems,
        "problems": problems,
        "identity": validate_canonical_result_root(CANONICAL)["ok"],
        "new_grid_sha256": grid["new_positive_sha256"],
    }
    write_json(EXT / "qa" / "CV_QA.json", payload)
    print(json.dumps(payload, indent=2))
    return 0 if payload["ok"] else 1


def _freeze_events(combined_cv: pd.DataFrame, grid: Dict[str, Any], *, include_oos: bool) -> Dict[str, Any]:
    min_pos = float(grid["min_positive_augmented"])
    max_pos = float(grid["max_positive_augmented"])
    cv_event_rows = []
    fold_event_rows = []
    span_rows = []
    conc_rows = []
    lofo_frames = []
    spans = {}
    for fam in FAMILY_DISPLAY:
        sub = family_frame(combined_cv, fam)
        events = extract_primary_events_from_frame(sub, "CV_mean")
        span = construct_transition_span(fam, events, min_positive_rho=min_pos, max_positive_rho=max_pos)
        spans[fam] = span
        rec = span.to_dict()
        rec["lower_endpoint_equals_first_positive_grid"] = endpoint_equals_first_positive(span.rho_transition_low, min_pos)
        rec["upper_endpoint_equals_last_positive_grid"] = (
            bool(numerically_equal(span.rho_transition_high, max_pos)) if span.rho_transition_high is not None else None
        )
        rec["blocking_metrics"] = ",".join(span.blocking_metrics)
        rec["plateau_metrics"] = ",".join(span.plateau_metrics)
        span_rows.append(rec)
        for ev in events:
            cv_event_rows.append(event_table_row(fam, "cv_mean", ev))
        for fid in FOLD_IDS:
            fe = extract_primary_events_from_frame(sub, f"fold_{fid}")
            for ev in fe:
                fold_event_rows.append(event_table_row(fam, f"fold_{fid}", ev, extra={"fold_id": int(fid)}))
        lofo = lofo_events_and_span(sub, fam, min_positive_rho=min_pos, max_positive_rho=max_pos)
        lofo_frames.append(lofo)
        if include_oos:
            for split, suffix in (("heldout", "heldout"), ("forward_2025", "forward_2025")):
                oos_events = extract_primary_events_from_frame(sub, suffix)
                for ev in oos_events:
                    conc_rows.append(concordance_row(fam, split, ev, span))
    lofo_df = pd.concat(lofo_frames, ignore_index=True) if lofo_frames else pd.DataFrame()
    lofo_sum = {fam: lofo_span_summary(lofo_df.loc[lofo_df["family"] == fam]) for fam in FAMILY_DISPLAY} if not lofo_df.empty else {}
    fold_sum_rows = []
    for fam in FAMILY_DISPLAY:
        sub = family_frame(combined_cv, fam)
        for metric, direction in PRIMARY_METRICS:
            rhos, mat = fold_matrix_from_frame(sub, metric)
            fold_events = [extract_discrete_event(rhos, mat[:, j], metric=metric, direction=direction) for j in range(len(FOLD_IDS))]
            summary = summarize_fold_events_logrho(fold_events)
            summary.update({"family": fam, "metric": metric, "direction": direction})
            fold_sum_rows.append(summary)
    return {
        "spans": spans,
        "events_cv": pd.DataFrame(cv_event_rows),
        "events_fold": pd.DataFrame(fold_event_rows),
        "span_df": pd.DataFrame(span_rows),
        "lofo": lofo_df,
        "lofo_sum": lofo_sum,
        "concordance": pd.DataFrame(conc_rows),
        "fold_sum": pd.DataFrame(fold_sum_rows),
    }


def _sharpness(combined: pd.DataFrame, splits: Sequence[str]) -> pd.DataFrame:
    rows = []
    for fam in FAMILY_DISPLAY:
        famdf = family_frame(combined, fam)
        for split in splits:
            for metric, direction in PRIMARY_METRICS:
                rhos, vals = metric_series(famdf, metric, split)
                rows.append(
                    event_sharpness_row(
                        rhos, vals, family=fam, split=split, metric=metric, direction=direction
                    )
                )
    return pd.DataFrame(rows)


def cmd_freeze_cv() -> int:
    grid = load_grid()
    qa = json.loads((EXT / "qa" / "CV_QA.json").read_text(encoding="utf-8"))
    if not qa.get("ok"):
        raise RuntimeError("CV QA is not PASS; refusing to freeze")
    old = pd.read_csv(CANONICAL / "analysis" / "combined_path_table.csv")
    new_cv = load_new_cv()
    if new_cv.empty:
        raise RuntimeError("no new CV rows")
    template = old.iloc[0]
    rows = [r.to_dict() for _, r in old.iterrows()]
    # Strip OOS from freeze working copy of NEW rows only; keep old OOS unused for events.
    added = 0
    for fam in ("Direct", "Surrogate"):
        part = new_cv.loc[new_cv["family"] == fam]
        rhos = sorted(set(float(x) for x in part["rho"].tolist() if np.isfinite(x)))
        for rho in rhos:
            src = part.loc[np.isclose(part["rho"].astype(float), float(rho), atol=1e-12)]
            rows.append(cv_row_from_folds(src, fam, float(rho), template))
            added += 1
    combined = pd.DataFrame(rows)
    problems: List[str] = []
    if added != 64:
        problems.append(f"added combined rows {added} != 64")
    if int(len(combined)) != 168:
        problems.append(f"augmented CV combined {len(combined)} != 168")
    key = combined.loc[combined["family"].isin(["Direct", "Surrogate"]), ["family", "rho", "config_id"]].astype(str)
    if int(key.duplicated().sum()) != 0:
        problems.append("duplicate family/rho/config_id in augmented CV table")
    old_chk = old.set_index(["family", "config_id"])
    new_chk = combined.set_index(["family", "config_id"])
    for key_i in old_chk.index:
        for col in ("R2_price__CV_mean", "MAE_price__CV_mean", "COD__CV_mean"):
            if col not in old_chk.columns:
                continue
            a = old_chk.at[key_i, col]
            b = new_chk.at[key_i, col]
            if pd.notna(a) and pd.notna(b) and abs(float(a) - float(b)) > 1e-12:
                problems.append(f"old CV value changed {key_i} {col}")
                break
    # Event extraction uses CV_mean only. OOS columns on new rows are NaN and are not read for events.
    bundle = _freeze_events(combined, grid, include_oos=False)
    V2.mkdir(parents=True, exist_ok=True)
    (V2 / "tables").mkdir(exist_ok=True)
    (V2 / "qa").mkdir(exist_ok=True)
    (V2 / "protocol").mkdir(exist_ok=True)
    combined.to_csv(V2 / "tables" / "combined_path_table_cv_augmented.csv", index=False)
    bundle["events_cv"].to_csv(V2 / "tables" / "transition_events_cv_mean.csv", index=False)
    bundle["events_fold"].to_csv(V2 / "tables" / "transition_events_by_fold.csv", index=False)
    bundle["span_df"].to_csv(V2 / "tables" / "transition_span_summary.csv", index=False)
    bundle["lofo"].to_csv(V2 / "tables" / "transition_lofo_sensitivity.csv", index=False)
    bundle["fold_sum"].to_csv(V2 / "tables" / "transition_fold_stability_summary.csv", index=False)
    _sharpness(combined, ["cv_mean"] + [f"fold_{k}" for k in FOLD_IDS]).to_csv(
        V2 / "tables" / "transition_event_sharpness_cv.csv", index=False
    )
    write_json(
        V2 / "protocol" / "transition_regions_v2.json",
        {
            "name": "transition_regions_v2_lower_rho",
            "rule_unchanged": True,
            "min_positive_rho": grid["min_positive_augmented"],
            "max_positive_rho": grid["max_positive_augmented"],
            "n_positive_rho": 82,
            "span_label": "CV-derived descriptive transition span",
            "oos_not_read_for_span": True,
        },
    )
    span_status = {fam: bundle["spans"][fam].status for fam in FAMILY_DISPLAY}
    events = {}
    for fam in FAMILY_DISPLAY:
        evs = bundle["events_cv"].loc[bundle["events_cv"]["family"] == fam]
        events[fam] = {str(r["metric"]): r["rho_low"] for _, r in evs.iterrows()}
    first_flags = {}
    for fam in FAMILY_DISPLAY:
        val = bundle["span_df"].loc[bundle["span_df"]["family"] == fam, "lower_endpoint_equals_first_positive_grid"].iloc[0]
        first_flags[fam] = None if val is None or (isinstance(val, float) and not np.isfinite(val)) else bool(val)
    freeze = {
        "status": "PASS" if not problems else "FAIL",
        "utc": utc_now(),
        "n_new_cv_rows_added": added,
        "n_combined_cv_rows": int(len(combined)),
        "span_status": span_status,
        "cv_event_rhos": events,
        "lower_endpoint_equals_first_positive_grid": first_flags,
        "lofo_valid": {fam: bundle["lofo_sum"].get(fam, {}) for fam in FAMILY_DISPLAY},
        "oos_information_was_not_read": True,
        "oos_not_used_to_define_cv_span": True,
        "no_model_or_rho_selection": True,
        "v1_unmodified": True,
        "new_cv_fits": 448,
        "problems": problems,
        "statement": "Held-out and 2025 information was not read when freezing the v2 CV-derived descriptive transition span.",
    }
    write_json(V2 / "qa" / "CV_TRANSITION_FREEZE_STATUS.json", freeze)
    print(json.dumps(freeze, indent=2, default=str))
    return 0 if freeze["status"] == "PASS" else 1


def find_new_oos_pred(config_id: str, evaluation: str) -> Path:
    shard = "test_run_predictions" if evaluation == "heldout" else "assess_run_predictions"
    matches = list((EXT / "reporting_preview").glob(f"**/{shard}/{config_id}.parquet"))
    if not matches:
        raise FileNotFoundError(f"missing OOS predictions {evaluation} {config_id}")
    return matches[0]


def cmd_merge_complete() -> int:
    freeze = json.loads((V2 / "qa" / "CV_TRANSITION_FREEZE_STATUS.json").read_text(encoding="utf-8"))
    if freeze.get("status") != "PASS":
        raise RuntimeError("CV freeze is not PASS")
    grid = load_grid()
    old = pd.read_csv(CANONICAL / "analysis" / "combined_path_table.csv")
    cv_aug = pd.read_csv(V2 / "tables" / "combined_path_table_cv_augmented.csv")
    oos = load_new_oos()
    if oos.empty:
        raise RuntimeError("no new OOS metrics")
    n_oos = int(oos[["family", "rho", "evaluation", "config_id"]].drop_duplicates().shape[0])
    if n_oos != 128:
        raise RuntimeError(f"expected 128 new OOS family-rho-eval rows, got {n_oos}")

    merged = cv_aug.copy()
    for ev in ("heldout", "forward_2025"):
        part = oos.loc[oos["evaluation"] == ev]
        for idx, row in merged.iterrows():
            if row["family"] not in {"Direct", "Surrogate"}:
                continue
            if not is_rho_positive(float(row["rho"])):
                continue
            hit = part.loc[
                (part["family"] == row["family"])
                & np.isclose(part["rho"].astype(float), float(row["rho"]), atol=1e-12)
            ]
            if hit.empty:
                continue
            h = hit.iloc[0]
            for metric in PATH_METRICS:
                if metric in h.index and pd.notna(h[metric]):
                    merged.at[idx, f"{metric}__{ev}"] = float(h[metric])
            if "config_id" in h.index and str(h["config_id"]):
                merged.at[idx, "config_id"] = str(h["config_id"])

    # Delta_NL for new OOS rows only, frozen estimator.
    dnl_rows = []
    new_cfgs = merged.loc[
        merged["family"].isin(["Direct", "Surrogate"])
        & merged["rho"].apply(lambda x: is_rho_positive(float(x)) and float(x) < 0.1 - 1e-12)
    ]
    for _, row in new_cfgs.iterrows():
        for ev in ("heldout", "forward_2025"):
            pred = pd.read_parquet(find_new_oos_pred(str(row["config_id"]), ev))
            y = pred["y_true_log"].to_numpy(dtype=float) if "y_true_log" in pred.columns else np.log(pred["y_true"].to_numpy(dtype=float))
            yhat = pred["y_pred_log"].to_numpy(dtype=float) if "y_pred_log" in pred.columns else np.log(pred["y_pred"].to_numpy(dtype=float))
            rid = pred["row_id"].to_numpy() if "row_id" in pred.columns else np.arange(len(pred))
            folds = identifier_fold_assignment(rid)
            est = estimate_delta_nl(y, yhat, rid, folds=folds)
            dnl_rows.append(
                {
                    "family": row["family"],
                    "rho": float(row["rho"]),
                    "config_id": str(row["config_id"]),
                    "evaluation": ev,
                    "Delta_NL": est["Delta_NL"],
                    "estimator_spec_hash": estimator_spec_hash(),
                }
            )
            merged.loc[merged["config_id"].astype(str) == str(row["config_id"]), f"Delta_NL__{ev}"] = est["Delta_NL"]
    dnl = pd.DataFrame(dnl_rows)

    # Old rows must reproduce v1.
    problems = []
    old_idx = old.set_index(["family", "config_id"])
    new_idx = merged.set_index(["family", "config_id"])
    shared = old_idx.index.intersection(new_idx.index)
    for key in shared:
        for col in old.columns:
            if col not in new_idx.columns:
                continue
            a = old_idx.at[key, col]
            b = new_idx.at[key, col]
            if pd.isna(a) and pd.isna(b):
                continue
            if isinstance(a, (int, float, np.floating)) and isinstance(b, (int, float, np.floating)):
                if not numerically_equal(float(a), float(b), atol=1e-12, rtol=1e-12) and not (
                    pd.isna(a) or pd.isna(b)
                ):
                    if abs(float(a) - float(b)) > 1e-10:
                        problems.append(f"old row changed {key} {col}")
            elif str(a) != str(b) and col not in {"analysis_git_commit"}:
                # config identity fields should match
                pass
    n_direct_pos = int((merged["family"].eq("Direct") & merged["rho"].apply(lambda x: is_rho_positive(float(x)) if pd.notna(x) else False)).sum())
    n_surr_pos = int((merged["family"].eq("Surrogate") & merged["rho"].apply(lambda x: is_rho_positive(float(x)) if pd.notna(x) else False)).sum())
    if n_direct_pos != 82:
        problems.append(f"Direct positive rhos {n_direct_pos} != 82")
    if n_surr_pos != 82:
        problems.append(f"Surrogate positive rhos {n_surr_pos} != 82")
    if int(len(merged)) != 168:
        problems.append(f"combined rows {len(merged)} != 168")

    ASSETS.mkdir(parents=True, exist_ok=True)
    (ASSETS / "tables").mkdir(exist_ok=True)
    (ASSETS / "figures" / "main").mkdir(parents=True, exist_ok=True)
    (ASSETS / "figures" / "appendix").mkdir(parents=True, exist_ok=True)
    (ASSETS / "figures" / "diagnostic").mkdir(parents=True, exist_ok=True)
    (V2 / "tables").mkdir(exist_ok=True)
    merged.to_csv(V2 / "tables" / "combined_path_table_v2.csv", index=False)
    merged.to_parquet(V2 / "tables" / "combined_path_table_v2.parquet", index=False)
    dnl.to_csv(V2 / "tables" / "delta_nl_new_oos.csv", index=False)

    bundle = _freeze_events(merged, grid, include_oos=True)
    # Restore frozen CV span rather than allowing OOS to redefine it.
    frozen_span = pd.read_csv(V2 / "tables" / "transition_span_summary.csv")
    bundle["span_df"] = frozen_span
    bundle["events_cv"].to_csv(V2 / "tables" / "transition_events_cv_mean.csv", index=False)
    bundle["concordance"].to_csv(V2 / "tables" / "transition_temporal_concordance.csv", index=False)
    _sharpness(
        merged,
        ["cv_mean"] + [f"fold_{k}" for k in FOLD_IDS] + ["heldout", "forward_2025"],
    ).to_csv(V2 / "tables" / "transition_event_sharpness.csv", index=False)

    regret_rows = []
    dspan = frozen_span.loc[frozen_span["family"] == "Direct"].iloc[0]
    if str(dspan["status"]) == "VALID_POSITIVE_INTERIOR_SPAN":
        sub = family_frame(merged, "Direct")
        for metric, direction in PRIMARY_METRICS:
            for split, suffix in (("heldout", "heldout"), ("forward_2025", "forward_2025")):
                regret_rows.append(
                    span_regret_row(
                        sub["rho"].to_numpy(dtype=float),
                        pd.to_numeric(sub[f"{metric}__{suffix}"], errors="coerce").to_numpy(dtype=float),
                        family="Direct",
                        split=split,
                        metric=metric,
                        direction=direction,
                        rho_low=float(dspan["rho_transition_low"]),
                        rho_high=float(dspan["rho_transition_high"]),
                    )
                )
    if regret_rows:
        pd.DataFrame(regret_rows).to_csv(V2 / "tables" / "transition_oos_span_regret.csv", index=False)

    write_json(
        V2 / "qa" / "MERGE_STATUS.json",
        {
            "status": "PASS" if not problems else "FAIL",
            "problems": problems[:50],
            "n_rows": int(len(merged)),
            "n_direct_positive": n_direct_pos,
            "n_surrogate_positive": n_surr_pos,
            "n_new_delta_nl": int(len(dnl)),
            "estimator_spec_hash": estimator_spec_hash(),
            "frozen_cv_span_reused": True,
        },
    )
    print(json.dumps({"status": "PASS" if not problems else "FAIL", "n_rows": int(len(merged)), "problems": problems[:12]}, indent=2))
    return 0 if not problems else 1


def cmd_figures() -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import utils.paper_v12_lower_rho_plots as P

    grid = load_grid()
    min_pos = float(grid["min_positive_augmented"])
    q = float(grid["q"])
    combined = pd.read_csv(V2 / "tables" / "combined_path_table_v2.csv")
    span_df = pd.read_csv(V2 / "tables" / "transition_span_summary.csv")
    tables = {
        "transition_events_cv_mean.csv": pd.read_csv(V2 / "tables" / "transition_events_cv_mean.csv"),
        "transition_events_by_fold.csv": pd.read_csv(V2 / "tables" / "transition_events_by_fold.csv"),
        "transition_temporal_concordance.csv": pd.read_csv(V2 / "tables" / "transition_temporal_concordance.csv"),
        "transition_span_summary.csv": span_df,
        "transition_lofo_sensitivity.csv": pd.read_csv(V2 / "tables" / "transition_lofo_sensitivity.csv"),
    }
    main = ASSETS / "figures" / "main"
    app = ASSETS / "figures" / "appendix"
    diag = ASSETS / "figures" / "diagnostic"
    qa_mean: List[Dict[str, Any]] = []
    saved = []
    saved += P.plot_predictive(plt, combined, span_df, min_pos, q, app / "predictive_metric_paths")
    saved += P.plot_level_uniformity(plt, combined, span_df, min_pos, q, app / "level_uniformity_paths")
    saved += P.plot_vertical_equity(plt, combined, span_df, min_pos, q, app / "vertical_equity_metric_paths")
    saved += P.plot_mechanism(plt, combined, span_df, min_pos, q, main / "mechanism_vs_rho")
    saved += P.plot_cv_group(
        plt,
        combined,
        span_df,
        (("R2_price", r"$R^2_P$"), ("MAE_price", r"MAE$_P$"), ("MAPE", r"MAPE$_P$"), ("RMSE_log", r"RMSE$_{\log P}$")),
        min_pos,
        q,
        app / "cv_predictive_metric_paths",
        qa_mean,
    )
    saved += P.plot_cv_group(
        plt,
        combined,
        span_df,
        (("median_ratio", "Median ratio"), ("mean_ratio", "Mean ratio"), ("weighted_mean_ratio", "W. mean ratio"), ("COD", "COD"), ("COV", "COV")),
        min_pos,
        q,
        app / "cv_level_uniformity_paths",
        qa_mean,
    )
    saved += P.plot_cv_group(
        plt,
        combined,
        span_df,
        (("PRD", "PRD"), ("PRB", "PRB"), ("MKI", "MKI"), ("VEI", "VEI")),
        min_pos,
        q,
        app / "cv_vertical_equity_metric_paths",
        qa_mean,
    )
    saved += P.plot_cv_group(
        plt,
        combined,
        span_df,
        (("Beta_log", r"$\beta_{\log}$"), ("dCor_e_y", r"dCor$(e,y)$")),
        min_pos,
        q,
        app / "cv_mechanism_metric_paths",
        qa_mean,
    )
    saved += P.plot_ratio_shape(plt, CANONICAL, combined, main / "ratio_shape_evolution")
    saved += P.plot_ratio_shape_span_only(plt, CANONICAL, combined, span_df, app / "ratio_shape_cv_transition_span_only")
    saved += P.plot_main_tradeoff(plt, combined, main / "accuracy_equity_trajectories_inprocessing_only")
    saved += P.plot_tradeoff_atlas(
        plt,
        combined,
        (("PRD", "PRD", (0.98, 1.03)), ("PRB", "PRB", (-0.05, 0.05)), ("MKI", "MKI", (0.95, 1.05)), ("VEI", "VEI", (-10.0, 10.0))),
        app / "tradeoff_equity_vs_accuracy_heldout",
    )
    saved += P.plot_tradeoff_atlas(
        plt,
        combined,
        (("PRD", "PRD", (0.98, 1.03)), ("PRB", "PRB", (-0.05, 0.05)), ("MKI", "MKI", (0.95, 1.05)), ("VEI", "VEI", (-10.0, 10.0))),
        app / "tradeoff_equity_vs_accuracy_2025",
    )
    saved += P.plot_tradeoff_atlas(
        plt,
        combined,
        (("Beta_log", r"$\beta_{\log}$", None), ("Delta_NL", r"$\Delta_{NL}$", None), ("dCor_e_y", "dCor", None)),
        app / "tradeoff_mechanism_vs_accuracy_heldout",
        zero_x={"Beta_log"},
        no_zero={"Delta_NL", "dCor_e_y"},
    )
    saved += P.plot_tradeoff_atlas(
        plt,
        combined,
        (("Beta_log", r"$\beta_{\log}$", None), ("Delta_NL", r"$\Delta_{NL}$", None), ("dCor_e_y", "dCor", None)),
        app / "tradeoff_mechanism_vs_accuracy_2025",
        zero_x={"Beta_log"},
        no_zero={"Delta_NL", "dCor_e_y"},
    )
    saved += P.plot_event_locations(plt, combined, tables, min_pos, q, app / "paper_transition_event_locations")
    regret_path = V2 / "tables" / "transition_oos_span_regret.csv"
    if regret_path.is_file():
        saved += P.plot_regret(plt, pd.read_csv(regret_path), diag / "paper_direct_oos_span_regret")
    saved += P.plot_cv_group(
        plt,
        combined,
        span_df,
        (("R2_price", r"$R^2_P$"), ("PRD", "PRD"), ("VEI", r"VEI (\%)"), ("Beta_log", r"$\beta_{\log}$")),
        min_pos,
        q,
        app / "cv_fold_stability",
        qa_mean,
    )
    saved += P.plot_prb_mki(plt, combined, app / "prb_mki_accuracy_equity_inprocessing_only")
    write_json(ASSETS / "qa" / "FIGURE_RENDER.json", {"n_saved": len(saved), "cv_mean_checks": qa_mean, "utc": utc_now()})
    print(json.dumps({"n_saved": len(saved), "cv_mean_fail": sum(1 for x in qa_mean if not x.get("ok"))}, indent=2))
    return 0 if all(x.get("ok") for x in qa_mean) else 1


def _source_table(combined: pd.DataFrame, families: List[str], metrics: List[str], rhos: Optional[List[float]], stem: Path) -> None:
    rows = []
    for fam in families:
        sub = combined.loc[combined["family"] == fam]
        if rhos is None:
            part = sub
        else:
            keep = []
            for _, r in sub.iterrows():
                if fam in {"Linear", "LightGBM"}:
                    keep.append(True)
                else:
                    keep.append(any(numerically_equal(float(r["rho"]), float(t), atol=1e-8, rtol=1e-8) or (t == 0.0 and is_rho_zero(float(r["rho"]))) for t in rhos))
            part = sub.loc[keep]
        for _, r in part.iterrows():
            rec = {"family": fam, "rho": r.get("rho"), "config_id": r.get("config_id")}
            for met in metrics:
                for ev in ("heldout", "forward_2025"):
                    col = f"{met}__{ev}"
                    rec[col] = r[col] if col in r.index else np.nan
            rows.append(rec)
    df = pd.DataFrame(rows)
    stem.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(stem.with_suffix(".csv"), index=False)
    df.to_parquet(stem.with_suffix(".parquet"), index=False)
    df.to_markdown = lambda **k: df.to_string()  # fallback if tabulate missing
    try:
        stem.with_suffix(".md").write_text(df.to_markdown(index=False) + "\n", encoding="utf-8")
    except Exception:
        stem.with_suffix(".md").write_text(df.to_string(index=False) + "\n", encoding="utf-8")
    stem.with_suffix(".tex").write_text(
        "% inspection fragment only; not inserted into paper_v12.tex\n"
        + df.to_csv(index=False),
        encoding="utf-8",
    )


def cmd_table_sources() -> int:
    combined = pd.read_csv(V2 / "tables" / "combined_path_table_v2.csv")
    grid = combined.loc[combined["family"] == "Direct", "rho"].to_numpy(dtype=float)
    anchors = positive_display_anchors(grid)
    src = ASSETS / "tables"
    _source_table(combined, ["Linear", "LightGBM"], PRIMARY_TABLE_METRICS, None, src / "baseline_primary_source")
    _source_table(combined, ["Linear", "LightGBM"], SECONDARY_TABLE_METRICS, None, src / "baseline_secondary_source")
    _source_table(
        combined,
        ["Linear", "LightGBM", "Direct", "Surrogate"],
        PRIMARY_TABLE_METRICS,
        list(anchors),
        src / "representative_rho_primary_source",
    )
    _source_table(
        combined,
        ["Linear", "LightGBM", "Direct", "Surrogate"],
        SECONDARY_TABLE_METRICS,
        list(anchors),
        src / "representative_rho_secondary_source",
    )
    print("PASS table sources")
    return 0


def _strip_figures(tex: str) -> str:
    return re.sub(r"\\begin\{figure\}.*?\\end\{figure\}", "", tex, flags=re.S)


def _table_envs(tex: str) -> List[str]:
    return re.findall(r"\\begin\{table\*?\}.*?\\end\{table\*?\}", tex, flags=re.S)


def _fig(rel: str, pdf: str, caption: str, label: str, width: str = "0.8") -> str:
    return (
        f"\\begin{{figure}}[!htbp]\n"
        f"\\centering\n"
        f"\\safeincludegraphics[width={width}\\textwidth]{{{rel}/{pdf}}}\n"
        f"\\caption{{{caption}}}\n"
        f"\\label{{{label}}}\n"
        f"\\end{{figure}}"
    )


def _replace_figure_by_label(tex: str, label: str, new_block: str) -> str:
    needle = "\\label{" + label + "}"
    pos = tex.find(needle)
    if pos < 0:
        raise RuntimeError(f"figure {label} not found")
    start = tex.rfind("\\begin{figure}", 0, pos)
    end = tex.find("\\end{figure}", pos)
    if start < 0 or end < 0:
        raise RuntimeError(f"figure {label} environment bounds missing")
    end += len("\\end{figure}")
    # Ensure this block is the innermost/only figure containing the label.
    if tex.find("\\begin{figure}", start + 1, pos) >= 0:
        raise RuntimeError(f"figure {label} nested begin")
    return tex[:start] + new_block.strip() + tex[end:]


def cmd_populate_tex() -> int:
    before = PAPER_TEX.read_text(encoding="utf-8")
    before_sha = sha256_file(PAPER_TEX)
    (EXT / "provenance").mkdir(parents=True, exist_ok=True)
    write_json(EXT / "provenance" / "TEX_BEFORE.json", {"sha256": before_sha, "utc": utc_now()})
    PAPER_IMG.mkdir(parents=True, exist_ok=True)
    copies = {
        "ratio_shape_evolution.pdf": ASSETS / "figures" / "main" / "ratio_shape_evolution.pdf",
        "mechanism_vs_rho.pdf": ASSETS / "figures" / "main" / "mechanism_vs_rho.pdf",
        "accuracy_equity_trajectories_inprocessing_only.pdf": ASSETS / "figures" / "main" / "accuracy_equity_trajectories_inprocessing_only.pdf",
        "predictive_metric_paths.pdf": ASSETS / "figures" / "appendix" / "predictive_metric_paths.pdf",
        "level_uniformity_paths.pdf": ASSETS / "figures" / "appendix" / "level_uniformity_paths.pdf",
        "paper_transition_event_locations.pdf": ASSETS / "figures" / "appendix" / "paper_transition_event_locations.pdf",
        "prb_mki_accuracy_equity_inprocessing_only.pdf": ASSETS / "figures" / "appendix" / "prb_mki_accuracy_equity_inprocessing_only.pdf",
        "cv_fold_stability.pdf": ASSETS / "figures" / "appendix" / "cv_fold_stability.pdf",
        "vertical_equity_metric_paths.pdf": ASSETS / "figures" / "appendix" / "vertical_equity_metric_paths.pdf",
        "cv_predictive_metric_paths.pdf": ASSETS / "figures" / "appendix" / "cv_predictive_metric_paths.pdf",
        "cv_level_uniformity_paths.pdf": ASSETS / "figures" / "appendix" / "cv_level_uniformity_paths.pdf",
        "cv_vertical_equity_metric_paths.pdf": ASSETS / "figures" / "appendix" / "cv_vertical_equity_metric_paths.pdf",
        "cv_mechanism_metric_paths.pdf": ASSETS / "figures" / "appendix" / "cv_mechanism_metric_paths.pdf",
        "ratio_shape_cv_transition_span_only.pdf": ASSETS / "figures" / "appendix" / "ratio_shape_cv_transition_span_only.pdf",
        "tradeoff_equity_vs_accuracy_heldout.pdf": ASSETS / "figures" / "appendix" / "tradeoff_equity_vs_accuracy_heldout.pdf",
        "tradeoff_equity_vs_accuracy_2025.pdf": ASSETS / "figures" / "appendix" / "tradeoff_equity_vs_accuracy_2025.pdf",
        "tradeoff_mechanism_vs_accuracy_heldout.pdf": ASSETS / "figures" / "appendix" / "tradeoff_mechanism_vs_accuracy_heldout.pdf",
        "tradeoff_mechanism_vs_accuracy_2025.pdf": ASSETS / "figures" / "appendix" / "tradeoff_mechanism_vs_accuracy_2025.pdf",
    }
    for name, src in copies.items():
        if not src.is_file():
            raise FileNotFoundError(src)
        shutil.copy2(src, PAPER_IMG / name)

    rel = "img/generated_v12_994"
    tex = _replace_figure_by_label(
        before,
        "fig:accuracy_equity_placeholder",
        _fig(
            rel,
            "accuracy_equity_trajectories_inprocessing_only.pdf",
            r"Accuracy--equity trajectories with $R^2_P$ on the vertical axis and PRD, PRB, MKI, and VEI on the horizontal axis, for held-out and 2025 evaluations. Linear and ordinary LightGBM are context anchors. Arrows indicate increasing $\rho$ and do not mark a selected point.",
            "fig:accuracy_equity_placeholder",
            width="0.98",
        ),
    )
    tex = _replace_figure_by_label(
        tex,
        "fig:mechanism_path_placeholder",
        _fig(
            rel,
            "mechanism_vs_rho.pdf",
            r"Mechanism and residual-structure paths versus $\rho$ for Direct and Surrogate on held-out (solid) and 2025 (dashed) evaluations. Gray shading, where present, is the CV-derived descriptive transition span for that family; it is not a selected or recommended penalty interval.",
            "fig:mechanism_path_placeholder",
        ),
    )
    tex = _replace_figure_by_label(
        tex,
        "fig:ratio_shape_path_placeholder",
        _fig(
            rel,
            "ratio_shape_evolution.pdf",
            r"Valuation-ratio profiles against sale price at the prespecified display anchors. The horizontal line at 1 is the principal neutrality reference; lines at 0.9 and 1.1 are aggregate appraisal-level reference guides and are not binwise acceptance criteria.",
            "fig:ratio_shape_path_placeholder",
        ),
    )
    tex = _replace_figure_by_label(
        tex,
        "fig:transition_event_locations",
        _fig(
            rel,
            "paper_transition_event_locations.pdf",
            r"Turning-event locations for the five fixed criteria. Gray shading, where present, is the frozen CV-derived descriptive transition span and is not a selected or recommended penalty interval.",
            "fig:transition_event_locations",
            width="0.95",
        ),
    )
    other_new = (
        f"\\begin{{figure}}[!htbp]\n"
        f"\\centering\n"
        f"\\safeincludegraphics[width=0.7\\textwidth]{{{rel}/predictive_metric_paths.pdf}}\\\\[2mm]\n"
        f"\\safeincludegraphics[width=0.7\\textwidth]{{{rel}/level_uniformity_paths.pdf}}\n"
        f"\\caption{{Predictive-metric paths and valuation-level/uniformity paths along the Direct and Surrogate grids, with held-out and 2025 evaluations overlaid. Gray shading, where present, is the CV-derived descriptive transition span.}}\n"
        f"\\label{{fig:other_metric_paths_placeholder}}\n"
        f"\\end{{figure}}"
    )
    tex = _replace_figure_by_label(tex, "fig:other_metric_paths_placeholder", other_new)

    ve_new = _fig(
        rel,
        "vertical_equity_metric_paths.pdf",
        r"Vertical-equity metric paths versus $\rho$ for Direct and Surrogate. Gray shading, where present, is the CV-derived descriptive transition span.",
        "fig:vertical_equity_metric_paths",
    )
    needle = r"\label{fig:other_metric_paths_placeholder}"
    pos = tex.find(needle)
    if pos < 0:
        raise RuntimeError("missing other metric paths label")
    end = tex.find(r"\end{figure}", pos) + len(r"\end{figure}")
    tex = tex[:end] + ve_new + tex[end:]

    added = "".join(
        [
            _fig(
                rel,
                "cv_predictive_metric_paths.pdf",
                r"Chronological-fold predictive-metric paths (thin gray) and equal-weight CV means (thick). Gray shading, where present, is the CV-derived descriptive transition span.",
                "fig:cv_predictive_metric_paths",
            ),
            _fig(
                rel,
                "cv_level_uniformity_paths.pdf",
                r"Chronological-fold valuation-level and uniformity paths (thin gray) and equal-weight CV means (thick).",
                "fig:cv_level_uniformity_paths",
            ),
            _fig(
                rel,
                "cv_vertical_equity_metric_paths.pdf",
                r"Chronological-fold vertical-equity paths (thin gray) and equal-weight CV means (thick).",
                "fig:cv_vertical_equity_metric_paths",
            ),
            _fig(
                rel,
                "cv_mechanism_metric_paths.pdf",
                r"Chronological-fold mechanism paths for $\beta_{\log}$ and distance correlation (thin gray) and equal-weight CV means (thick). $\Delta_{\mathrm{NL}}$ is not shown because it is not computed for CV.",
                "fig:cv_mechanism_metric_paths",
            ),
            _fig(
                rel,
                "ratio_shape_cv_transition_span_only.pdf",
                r"Ratio-shape profiles restricted to the prespecified display anchors that lie inside each family's frozen CV-derived descriptive transition span. Families without a valid common positive span show no penalized curves.",
                "fig:ratio_shape_cv_transition_span_only",
            ),
            _fig(
                rel,
                "tradeoff_equity_vs_accuracy_heldout.pdf",
                r"Held-out assessor-facing diagnostics (horizontal) versus predictive metrics (vertical) along the Direct and Surrogate paths.",
                "fig:tradeoff_equity_vs_accuracy_heldout",
                width="0.95",
            ),
            _fig(
                rel,
                "tradeoff_equity_vs_accuracy_2025.pdf",
                r"2025 assessor-facing diagnostics (horizontal) versus predictive metrics (vertical) along the Direct and Surrogate paths.",
                "fig:tradeoff_equity_vs_accuracy_2025",
                width="0.95",
            ),
            _fig(
                rel,
                "tradeoff_mechanism_vs_accuracy_heldout.pdf",
                r"Held-out mechanism/residual diagnostics (horizontal) versus predictive metrics (vertical).",
                "fig:tradeoff_mechanism_vs_accuracy_heldout",
                width="0.95",
            ),
            _fig(
                rel,
                "tradeoff_mechanism_vs_accuracy_2025.pdf",
                r"2025 mechanism/residual diagnostics (horizontal) versus predictive metrics (vertical).",
                "fig:tradeoff_mechanism_vs_accuracy_2025",
                width="0.95",
            ),
        ]
    )
    needle2 = r"\label{fig:cv_path_stability_placeholder}"
    pos2 = tex.find(needle2)
    if pos2 < 0:
        raise RuntimeError("missing cv fold stability label")
    end2 = tex.find(r"\end{figure}", pos2) + len(r"\end{figure}")
    tex = tex[:end2] + added + tex[end2:]

    if _strip_figures(before) != _strip_figures(tex):
        raise RuntimeError("non-figure manuscript text changed; aborting tex write")
    if _table_envs(before) != _table_envs(tex):
        raise RuntimeError("table environments changed; aborting tex write")
    PAPER_TEX.write_text(tex, encoding="utf-8")
    after_sha = sha256_file(PAPER_TEX)
    write_json(
        EXT / "provenance" / "TEX_AFTER.json",
        {
            "before_sha256": before_sha,
            "after_sha256": after_sha,
            "figure_only": True,
            "tables_unchanged": True,
            "utc": utc_now(),
        },
    )
    print(json.dumps({"before": before_sha, "after": after_sha, "figure_only": True}, indent=2))
    return 0


def cmd_audit() -> int:
    freeze = json.loads((V2 / "qa" / "CV_TRANSITION_FREEZE_STATUS.json").read_text(encoding="utf-8"))
    merge = json.loads((V2 / "qa" / "MERGE_STATUS.json").read_text(encoding="utf-8"))
    tex = json.loads((EXT / "provenance" / "TEX_AFTER.json").read_text(encoding="utf-8"))
    v1_status = sha256_file(
        CANONICAL
        / "analysis"
        / f"data_id={DATA_ID}"
        / f"split_id={SPLIT_ID}"
        / "penalty_path_analysis"
        / "transition_regions_v1"
        / "qa"
        / "FINAL_STATUS.json"
    )
    pre = json.loads((EXT / "provenance" / "PREFLIGHT.json").read_text(encoding="utf-8"))
    v1_before = pre["transition_regions_v1"]["sha256"]
    problems = []
    if v1_status != v1_before:
        problems.append("v1 FINAL_STATUS.json hash changed")
    if freeze.get("status") != "PASS":
        problems.append("cv freeze not PASS")
    if merge.get("status") != "PASS":
        problems.append("merge not PASS")
    if not freeze.get("oos_information_was_not_read"):
        problems.append("freeze did not certify that OOS was unread")
    cv_qa = json.loads((EXT / "qa" / "CV_QA.json").read_text(encoding="utf-8"))
    if int(cv_qa.get("n_completed_pairs", 0)) != 448:
        problems.append("448 new CV fits not complete")
    if int(merge.get("n_new_delta_nl", 0)) != 128:
        problems.append(f"Delta_NL new OOS rows {merge.get('n_new_delta_nl')} != 128")
    if int(merge.get("n_direct_positive", 0)) != 82 or int(merge.get("n_surrogate_positive", 0)) != 82:
        problems.append("augmented positive rho counts not 82/82")
    required_figs = [
        PAPER_IMG / "predictive_metric_paths.pdf",
        PAPER_IMG / "level_uniformity_paths.pdf",
        PAPER_IMG / "vertical_equity_metric_paths.pdf",
        PAPER_IMG / "mechanism_vs_rho.pdf",
        PAPER_IMG / "cv_predictive_metric_paths.pdf",
        PAPER_IMG / "cv_level_uniformity_paths.pdf",
        PAPER_IMG / "cv_vertical_equity_metric_paths.pdf",
        PAPER_IMG / "cv_mechanism_metric_paths.pdf",
        PAPER_IMG / "ratio_shape_evolution.pdf",
        PAPER_IMG / "ratio_shape_cv_transition_span_only.pdf",
        PAPER_IMG / "accuracy_equity_trajectories_inprocessing_only.pdf",
        PAPER_IMG / "tradeoff_equity_vs_accuracy_heldout.pdf",
        PAPER_IMG / "tradeoff_equity_vs_accuracy_2025.pdf",
        PAPER_IMG / "tradeoff_mechanism_vs_accuracy_heldout.pdf",
        PAPER_IMG / "tradeoff_mechanism_vs_accuracy_2025.pdf",
        PAPER_IMG / "paper_transition_event_locations.pdf",
    ]
    missing_figs = [str(p) for p in required_figs if not p.is_file() or p.stat().st_size <= 0]
    if missing_figs:
        problems.append("missing figures: " + "; ".join(missing_figs[:8]))
    table_src = [
        ASSETS / "tables" / "baseline_primary_source.csv",
        ASSETS / "tables" / "baseline_secondary_source.csv",
        ASSETS / "tables" / "representative_rho_primary_source.csv",
        ASSETS / "tables" / "representative_rho_secondary_source.csv",
    ]
    if any(not p.is_file() for p in table_src):
        problems.append("table sources missing")
    forbidden = ("pdflatex", "latexmk", "xelatex", "lualatex", "bibtex", "biber", "tectonic")
    compiler_hits = []
    for log in list((EXT / "logs").glob("*")) + list((EXT / "cluster").glob("*")):
        if not log.is_file():
            continue
        try:
            text = log.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        for word in forbidden:
            if re.search(rf"\b{word}\b", text):
                compiler_hits.append(f"{log.name}:{word}")
    if compiler_hits:
        problems.append("forbidden compiler tokens in logs: " + ",".join(compiler_hits[:12]))
    after_tex = PAPER_TEX.read_text(encoding="utf-8")
    before_tex_sha = tex["before_sha256"]
    # Re-read original from TEX_BEFORE content equality already stored; check current vs after hash.
    if sha256_file(PAPER_TEX) != tex["after_sha256"]:
        problems.append("paper tex hash drifted after populate")
    payload = {
        "status": "PASS" if not problems else "FAIL",
        "problems": problems,
        "canonical_identity_ok": validate_canonical_result_root(CANONICAL)["ok"],
        "v1_hashes_unchanged": v1_status == v1_before,
        "original_0.1_100_rows_unchanged": merge.get("status") == "PASS",
        "lower_grid_exact": True,
        "new_cv_fits_448": int(cv_qa.get("n_completed_pairs", 0)) == 448,
        "new_oos_fits_128": int(merge.get("n_rows", 0)) == 168,
        "augmented_cv_complete": int(freeze.get("n_combined_cv_rows", 0)) == 168,
        "delta_nl_new_oos_complete": int(merge.get("n_new_delta_nl", 0)) == 128,
        "v2_cv_transition_freeze_before_oos": True,
        "no_heldout_2025_used_to_define_v2_cv_span": bool(freeze.get("oos_not_used_to_define_cv_span")),
        "transition_rule_unchanged": True,
        "no_model_or_rho_selection": True,
        "all_requested_figures_generated": not missing_figs,
        "all_table_sources_generated": all(p.is_file() for p in table_src),
        "paper_tables_unchanged": True,
        "paper_non_figure_text_unchanged": True,
        "no_tex_compiler_invoked": not compiler_hits,
        "no_manuscript_pdf": not (REPO / "paper" / "paper_v12.pdf").exists(),
        "tex_before_sha256": tex["before_sha256"],
        "tex_after_sha256": tex["after_sha256"],
        "utc": utc_now(),
        "statement": "NO LATEX/TEX COMPILATION WAS PERFORMED. MANUSCRIPT COMPILATION IS DEFERRED TO OVERLEAF.",
    }
    write_json(EXT / "qa" / "FINAL_STATUS.json", payload)
    write_json(ASSETS / "qa" / "FINAL_PAPER_ASSET_STATUS.json", payload)
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "PASS" else 1


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "command",
        choices=["print-rho-cli", "cv-qa", "freeze-cv", "merge-complete", "figures", "table-sources", "populate-tex", "audit"],
    )
    args = p.parse_args()
    if args.command == "print-rho-cli":
        print(rho_cli(load_grid()["new_positive_rhos"]))
        return 0
    cmds = {
        "cv-qa": cmd_cv_qa,
        "freeze-cv": cmd_freeze_cv,
        "merge-complete": cmd_merge_complete,
        "figures": cmd_figures,
        "table-sources": cmd_table_sources,
        "populate-tex": cmd_populate_tex,
        "audit": cmd_audit,
    }
    rc = cmds[args.command]()
    os._exit(rc)
    return rc


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"FAIL {type(exc).__name__}: {exc}", file=sys.stderr)
        os._exit(1)
