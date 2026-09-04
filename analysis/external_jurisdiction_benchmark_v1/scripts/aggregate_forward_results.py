#!/usr/bin/env python3
"""Aggregate frozen 2025 path metrics, CV comparison, candidate-region
validation, and completeness. Does not write or move candidate regions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from analysis.external_jurisdiction_benchmark_v1.scripts.forward_common import (  # noqa: E402
    ANALYSIS, CANONICAL_METRIC_MAP, DIRECT_COMMON_INTERVAL, IDEALS, OUTPUT,
    add_derived, canonicalize_metrics, frozen_anchor_points, frozen_grid_rho_tilde,
    verify_forward_freeze,
)
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import ALL_KEYS  # noqa: E402

PARTIAL = ANALYSIS / "forward_2025" / "metrics" / "partial"
METRICS = ANALYSIS / "forward_2025" / "metrics"
TABLES = ANALYSIS / "forward_2025" / "tables"
AUDITS = ANALYSIS / "forward_2025" / "audits"
CV = ANALYSIS / "cv"

CORE = [
    "R2_price", "R2_log", "NMSE", "RMSE_log", "MAE", "MAPE",
    "median_ratio", "mean_ratio", "weighted_mean_ratio", "COD", "COV",
    "PRD", "PRB", "MKI", "VEI", "beta_log", "Delta_NL", "dCor",
    "Delta_NMSE", "Delta_R2_price", "A_beta",
    "I_PRD", "I_PRB", "I_MKI", "I_VEI", "I_beta_log",
    "Delta_I_PRD", "Delta_I_PRB", "Delta_I_MKI", "Delta_I_VEI", "Delta_I_beta_log",
]


def _near(a, b) -> bool:
    if pd.isna(a) or pd.isna(b):
        return False
    return bool(np.isclose(float(a), float(b), rtol=0.0, atol=1e-10))


def load_cv_family(key: str, family: str) -> pd.DataFrame:
    path = CV / f"{key}_{family}_normalized_cv_path_summary.csv"
    df = pd.read_csv(path)
    if "fit_status" in df.columns:
        df = df.loc[df["fit_status"].astype(str).eq("OK") | df["fit_status"].isna()].copy()
    metric_keys = list(dict.fromkeys(CANONICAL_METRIC_MAP.values()))
    rows = []
    for _, r in df.iterrows():
        raw = r.to_dict()
        canon = canonicalize_metrics(raw)
        rows.append({
            "county_key": key, "family": family, "fold": r.get("fold"),
            "validation_year": r.get("validation_year"),
            "rho_tilde": float(r["rho_tilde"]),
            **canon,
        })
    out = pd.DataFrame(rows)
    base_parts = []
    for fold, g in out.groupby("fold"):
        g = g.sort_values("rho_tilde")
        b = g.iloc[0].to_dict()
        for _, row in g.iterrows():
            rec = row.to_dict()
            rec.update(add_derived(
                {k: rec.get(k) for k in metric_keys},
                {k: b.get(k) for k in metric_keys},
            ))
            rec["evaluation_layer"] = "CV_FOLD"
            base_parts.append(rec)
    return pd.DataFrame(base_parts)


def interpolate_cv_row(csub: pd.DataFrame, rho: float) -> pd.Series | None:
    hit = match_rho(csub, rho)
    if hit is not None:
        return hit
    if not len(csub) or rho <= 0:
        return None
    sub = csub.sort_values("rho_tilde")
    x = sub["rho_tilde"].astype(float).to_numpy()
    if rho < x.min() or rho > x.max():
        return None
    rec = {"rho_tilde": float(rho), "interpolated_from_cv_grid": True}
    log_r = np.log10(rho)
    log_x = np.log10(np.clip(x, 1e-12, None))
    for col in sub.columns:
        if col in ("county_key", "family", "rho_tilde") or not col.endswith(("_cv_mean", "_cv_sd", "_cv_min", "_cv_max")):
            if col in ("county_key", "family"):
                rec[col] = sub.iloc[0][col]
            continue
        y = pd.to_numeric(sub[col], errors="coerce").to_numpy()
        if np.all(~np.isfinite(y)):
            rec[col] = np.nan
            continue
        rec[col] = float(np.interp(log_r, log_x, y))
    rec["n_folds"] = sub["n_folds"].iloc[0] if "n_folds" in sub.columns else np.nan
    return pd.Series(rec)


def cv_summary(cv_df: pd.DataFrame) -> pd.DataFrame:
    metrics = [c for c in CORE if c in cv_df.columns]
    rows = []
    for (key, fam, rho), g in cv_df.groupby(["county_key", "family", "rho_tilde"]):
        rec = {"county_key": key, "family": fam, "rho_tilde": float(rho), "n_folds": int(len(g))}
        for m in metrics:
            s = pd.to_numeric(g[m], errors="coerce")
            rec[f"{m}_cv_mean"] = float(s.mean()) if s.notna().any() else np.nan
            rec[f"{m}_cv_sd"] = float(s.std(ddof=1)) if s.notna().sum() > 1 else np.nan
            rec[f"{m}_cv_min"] = float(s.min()) if s.notna().any() else np.nan
            rec[f"{m}_cv_max"] = float(s.max()) if s.notna().any() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def load_forward_partial() -> pd.DataFrame:
    files = sorted(PARTIAL.glob("*_forward_path.csv"))
    if len(files) != 18:
        raise RuntimeError(f"expected 18 forward partial CSVs, found {len(files)}: {[p.name for p in files]}")
    return pd.concat([pd.read_csv(p) for p in files], ignore_index=True)


def completeness(fwd: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in ALL_KEYS:
        for family in ("direct", "surrogate"):
            sub = fwd.loc[(fwd.county_key == key) & (fwd.family == family)]
            grid = frozen_grid_rho_tilde(family)
            grid_rows = sub.loc[sub.on_frozen_grid == True]  # noqa: E712
            n_ok = int((grid_rows.fit_status.astype(str) == "OK").sum()) if len(grid_rows) else 0
            rows.append({
                "county_key": key, "family": family,
                "n_forward_rows": int(len(sub)),
                "n_grid_rows": int(len(grid_rows)),
                "expected_grid": int(len(grid)),
                "n_grid_ok": n_ok,
                "n_eval": int(sub["n_eval"].dropna().iloc[0]) if len(sub) and sub["n_eval"].notna().any() else np.nan,
                "n_train": int(sub["n_train"].dropna().iloc[0]) if len(sub) and sub["n_train"].notna().any() else np.nan,
                "Var_train_y": float(sub["Var_train_y"].dropna().iloc[0]) if len(sub) and sub["Var_train_y"].notna().any() else np.nan,
                "complete": bool(len(grid_rows) == len(grid)),
            })
    return pd.DataFrame(rows)


def match_rho(frame: pd.DataFrame, rho: float) -> pd.Series | None:
    if frame is None or not len(frame):
        return None
    hits = frame.loc[np.isclose(frame["rho_tilde"].astype(float), float(rho), rtol=0.0, atol=1e-10)]
    if not len(hits):
        return None
    return hits.iloc[0]


def build_anchor_table(fwd: pd.DataFrame, cv_mean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for key in ALL_KEYS:
        for family in ("direct", "surrogate"):
            fsub = fwd.loc[(fwd.county_key == key) & (fwd.family == family)]
            csub = cv_mean.loc[(cv_mean.county_key == key) & (cv_mean.family == family)]
            for p in frozen_anchor_points(key, family):
                fr = match_rho(fsub, p["rho_tilde"])
                cr = interpolate_cv_row(csub, p["rho_tilde"])
                rec = {
                    "county_key": key, "family": family, "role": p["role"],
                    "rho_tilde": float(p["rho_tilde"]),
                    "target_A_beta": p.get("target_A_beta"),
                }
                for m in CORE:
                    rec[f"{m}_2025"] = float(fr[m]) if fr is not None and m in fr.index and pd.notna(fr[m]) else np.nan
                    rec[f"{m}_cv_mean"] = float(cr[f"{m}_cv_mean"]) if cr is not None and f"{m}_cv_mean" in cr.index and pd.notna(cr[f"{m}_cv_mean"]) else np.nan
                    rec[f"{m}_2025_minus_cv"] = rec[f"{m}_2025"] - rec[f"{m}_cv_mean"]
                rec["fit_status_2025"] = None if fr is None else fr.get("fit_status")
                rec["present_in_2025"] = fr is not None
                rec["present_in_cv"] = cr is not None
                rows.append(rec)
    return pd.DataFrame(rows)


def candidate_validation(fwd: pd.DataFrame, cv_mean: pd.DataFrame) -> pd.DataFrame:
    cand = pd.read_csv(ANALYSIS / "candidate_regions" / "candidate_regions.csv")
    rows = []
    for _, crow in cand.iterrows():
        key, family = crow["county_key"], crow["family"]
        status = str(crow["status"])
        fsub = fwd.loc[(fwd.county_key == key) & (fwd.family == family)]
        csub = cv_mean.loc[(cv_mean.county_key == key) & (cv_mean.family == family)]
        points = [("activity", crow["activity_rho_tilde"]), ("guardrail", crow["guardrail_rho_tilde"])]
        for p in frozen_anchor_points(key, family):
            if p["role"] in ("A_beta_0.25", "A_beta_0.5", "direct_common_lo", "direct_common_hi"):
                points.append((p["role"], p["rho_tilde"]))
        seen = set()
        for role, rho in points:
            if pd.isna(rho):
                continue
            tag = (role, round(float(rho), 12))
            if tag in seen:
                continue
            seen.add(tag)
            fr = match_rho(fsub, float(rho))
            cr = interpolate_cv_row(csub, float(rho))
            rec = {
                "county_key": key, "family": family, "cv_status": status,
                "role": role, "rho_tilde": float(rho),
                "protocol_valid_region": status == "CANDIDATE_REGION",
            }
            for m in ("NMSE", "Delta_NMSE", "R2_price", "PRD", "PRB", "MKI", "VEI", "beta_log",
                      "A_beta", "Delta_NL", "dCor", "COD", "median_ratio"):
                rec[f"{m}_cv"] = np.nan if cr is None or f"{m}_cv_mean" not in cr.index else float(cr[f"{m}_cv_mean"])
                rec[f"{m}_2025"] = np.nan if fr is None or m not in fr.index else float(fr[m])
            rec["predictive_cost_2025"] = rec.get("Delta_NMSE_2025")
            rec["vertical_equity_benefit_2025"] = rec.get("Delta_I_PRB_2025") if False else (
                abs(rec["PRB_2025"]) - abs(rec["PRB_cv"]) if np.isfinite(rec.get("PRB_2025", np.nan)) and np.isfinite(rec.get("PRB_cv", np.nan)) else np.nan
            )
            rec["fit_status_2025"] = None if fr is None else fr.get("fit_status")
            rec["forward_failure"] = bool(fr is None or str(fr.get("fit_status")) != "OK")
            rows.append(rec)
    return pd.DataFrame(rows)


def common_band_rows(fwd: pd.DataFrame, cv_mean: pd.DataFrame) -> pd.DataFrame:
    rows = []
    stable = [k for k in ALL_KEYS if k != "allegheny"]
    for key in ALL_KEYS:
        fsub = fwd.loc[(fwd.county_key == key) & (fwd.family == "direct")]
        csub = cv_mean.loc[(cv_mean.county_key == key) & (cv_mean.family == "direct")]
        for role, rho in (("direct_common_lo", DIRECT_COMMON_INTERVAL[0]),
                          ("direct_common_hi", DIRECT_COMMON_INTERVAL[1])):
            fr = match_rho(fsub, rho)
            cr = match_rho(csub, rho)
            rec = {
                "county_key": key, "role": role, "rho_tilde": rho,
                "allegheny_sensitivity_only": key == "allegheny",
                "in_protocol_valid_set": key in stable,
            }
            for m in ("NMSE", "Delta_NMSE", "R2_price", "PRB", "beta_log", "A_beta", "Delta_NL", "dCor"):
                rec[f"{m}_cv"] = np.nan if cr is None or f"{m}_cv_mean" not in cr.index else float(cr[f"{m}_cv_mean"])
                rec[f"{m}_2025"] = np.nan if fr is None or m not in fr.index else float(fr[m])
            rec["fit_status_2025"] = None if fr is None else fr.get("fit_status")
            rec["practically_useful_2025"] = bool(
                fr is not None and str(fr.get("fit_status")) == "OK"
                and pd.notna(fr.get("Delta_NMSE")) and float(fr.get("A_beta") or np.nan) >= 0
            )
            rows.append(rec)
    return pd.DataFrame(rows)


def comparison_table(fwd: pd.DataFrame, cv_mean: pd.DataFrame) -> pd.DataFrame:
    grid_fwd = fwd.loc[fwd.on_frozen_grid == True].copy()  # noqa: E712
    rows = []
    for _, fr in grid_fwd.iterrows():
        cr = match_rho(cv_mean.loc[(cv_mean.county_key == fr.county_key) & (cv_mean.family == fr.family)],
                       fr.rho_tilde)
        rec = {
            "county_key": fr.county_key, "family": fr.family, "rho_tilde": float(fr.rho_tilde),
            "fit_status_2025": fr.get("fit_status"),
        }
        for m in CORE:
            rec[f"{m}_2025"] = fr[m] if m in fr.index else np.nan
            rec[f"{m}_cv_mean"] = np.nan if cr is None or f"{m}_cv_mean" not in cr.index else cr[f"{m}_cv_mean"]
            rec[f"{m}_cv_sd"] = np.nan if cr is None or f"{m}_cv_sd" not in cr.index else cr[f"{m}_cv_sd"]
            rec[f"{m}_cv_min"] = np.nan if cr is None or f"{m}_cv_min" not in cr.index else cr[f"{m}_cv_min"]
            rec[f"{m}_cv_max"] = np.nan if cr is None or f"{m}_cv_max" not in cr.index else cr[f"{m}_cv_max"]
            rec[f"{m}_2025_minus_cv"] = rec[f"{m}_2025"] - rec[f"{m}_cv_mean"]
        rows.append(rec)
    return pd.DataFrame(rows)


def main() -> int:
    verify_forward_freeze()
    METRICS.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    AUDITS.mkdir(parents=True, exist_ok=True)

    fwd = load_forward_partial()
    grid = fwd.loc[fwd.on_frozen_grid == True].copy()  # noqa: E712
    grid.to_csv(METRICS / "forward_2025_path_metrics.csv", index=False)

    cv_parts = [load_cv_family(k, f) for k in ALL_KEYS for f in ("direct", "surrogate")]
    cv_df = pd.concat(cv_parts, ignore_index=True)
    cv_mean = cv_summary(cv_df)
    cv_mean.to_csv(METRICS / "pooled_oof_path_metrics.csv", index=False)

    comparison_table(fwd, cv_mean).to_csv(METRICS / "cv_forward_path_comparison.csv", index=False)
    anchors = build_anchor_table(fwd, cv_mean)
    anchors.to_csv(METRICS / "forward_anchor_metrics.csv", index=False)

    completeness(fwd).to_csv(AUDITS / "forward_fit_completeness.csv", index=False)

    baseline = anchors.loc[anchors.role.isin(["baseline_rho0", "native_lgbm_baseline"])].copy()
    TABLES.mkdir(parents=True, exist_ok=True)
    baseline.to_csv(TABLES / "forward_baseline_summary.csv", index=False)
    candidate_validation(fwd, cv_mean).to_csv(TABLES / "forward_candidate_validation.csv", index=False)
    mech = anchors.loc[anchors.role.str.startswith("A_beta_")].copy()
    mech.to_csv(TABLES / "mechanism_anchor_forward.csv", index=False)
    common_band_rows(fwd, cv_mean).to_csv(TABLES / "direct_common_band_forward.csv", index=False)

    # Compact cross-jurisdiction summary at frozen 25/50 anchors + baseline.
    want = {"baseline_rho0", "native_lgbm_baseline", "activity", "guardrail", "A_beta_0.25", "A_beta_0.5"}
    summary = anchors.loc[anchors.role.isin(want)].copy()
    summary.to_csv(TABLES / "cross_jurisdiction_forward_summary.csv", index=False)
    print(json.dumps({
        "n_forward_rows": int(len(fwd)),
        "n_grid_rows": int(len(grid)),
        "n_anchor_rows": int(len(anchors)),
        "completeness_ok": bool(pd.read_csv(AUDITS / "forward_fit_completeness.csv")["complete"].all()),
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
