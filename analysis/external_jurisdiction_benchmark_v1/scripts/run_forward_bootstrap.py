#!/usr/bin/env python3
"""Paired monthly-block bootstrap for frozen 2025 anchors.

Sample 2025 months with replacement. Within a jurisdiction the same month
draw is applied to every family and every rho point. 200 draws. Significance
is never used to select or move a candidate region.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.other_counties_benchmars import score_predictions  # noqa: E402
from analysis.external_jurisdiction_benchmark_v1.scripts.run_baseline_cv import nmse  # noqa: E402
from analysis.external_jurisdiction_benchmark_v1.scripts.forward_common import (  # noqa: E402
    ANALYSIS, OUTPUT, canonicalize_metrics, frozen_anchor_points, verify_forward_freeze,
)
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import (  # noqa: E402
    ALL_KEYS, FIPS, N_BOOTSTRAP, SEED, write_json,
)
from utils.delta_nl import estimate_delta_nl  # noqa: E402
from utils.motivation_utils import distance_correlation_e_y  # noqa: E402

PRED = OUTPUT / "forward_2025" / "predictions"
RAW = OUTPUT / "forward_2025" / "bootstrap"
OUT = ANALYSIS / "forward_2025" / "bootstrap"
MECH_CAP = 6000
DELTA_NAMES = [
    "Delta_NMSE", "Delta_R2_price", "Delta_PRD", "Delta_PRB", "Delta_MKI",
    "Delta_VEI", "Delta_beta_log", "Delta_Delta_NL", "Delta_dCor",
]


def _near(a, b) -> bool:
    return bool(np.isclose(float(a), float(b), rtol=0.0, atol=1e-10))


def cheap_metrics(y_price: np.ndarray, pred_price: np.ndarray) -> dict:
    m = score_predictions(y_price, pred_price, y_price)
    ylog = np.log(y_price)
    plog = np.log(pred_price)
    m["NMSE"] = nmse(ylog, plog)
    e = plog - ylog
    c = ylog - float(np.mean(ylog))
    var = float(np.mean(c ** 2))
    m["Beta_log"] = float(np.mean(e * c) / var) if var > 0 else float("nan")
    return canonicalize_metrics(m)


def mech_metrics(y_price: np.ndarray, pred_price: np.ndarray, ids: np.ndarray) -> dict:
    ylog = np.log(y_price)
    plog = np.log(pred_price)
    out = {}
    try:
        out["dCor"] = float(distance_correlation_e_y(ylog, plog))
    except Exception:
        out["dCor"] = float("nan")
    try:
        out["Delta_NL"] = float(estimate_delta_nl(ylog, plog, row_ids=ids).get("Delta_NL", np.nan))
    except Exception:
        out["Delta_NL"] = float("nan")
    return out


def load_family(key: str, family: str) -> pd.DataFrame:
    path = PRED / f"{key}_{family}_forward_preds.parquet"
    df = pd.read_parquet(path)
    df["sale_date"] = pd.to_datetime(df["sale_date"])
    df["month"] = df["sale_date"].dt.to_period("M")
    return df


def slice_rho(df: pd.DataFrame, rho: float) -> pd.DataFrame:
    sub = df.loc[np.isclose(df["rho_tilde"].astype(float), float(rho), rtol=0.0, atol=1e-10)].copy()
    return sub.sort_values("TRANSACTIONID").reset_index(drop=True)


def resample_idx(months: pd.Series, sampled: np.ndarray) -> np.ndarray:
    """Row indices concatenating each sampled month (with replacement)."""
    groups = {m: np.flatnonzero(months.to_numpy() == m) for m in months.unique()}
    parts = [groups[m] for m in sampled if m in groups and len(groups[m])]
    if not parts:
        return np.array([], dtype=int)
    return np.concatenate(parts)


def main() -> int:
    verify_forward_freeze()
    RAW.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    ci_rows = []
    for key in ALL_KEYS:
        frames = {}
        months = None
        for family in ("direct", "surrogate"):
            frames[family] = load_family(key, family)
            m = frames[family].loc[frames[family].on_frozen_grid | (frames[family].rho_tilde == 0), "month"]
            unique = np.array(sorted(frames[family]["month"].unique()))
            if months is None:
                months = unique
            elif set(unique.tolist()) != set(months.tolist()):
                # Use the intersection so pairing is well-defined.
                months = np.array(sorted(set(months.tolist()) & set(unique.tolist())))
        rng = np.random.default_rng(SEED + int(FIPS[key]))
        n_m = len(months)
        draws = np.vstack([rng.choice(months, size=n_m, replace=True) for _ in range(N_BOOTSTRAP)])
        np.save(RAW / f"{key}_sampled_months.npy", np.array([str(x) for x in draws.ravel()]).reshape(draws.shape))

        for family in ("direct", "surrogate"):
            df = frames[family]
            anchors = frozen_anchor_points(key, family)
            base = slice_rho(df, 0.0)
            if not len(base):
                raise RuntimeError(f"{key}/{family}: missing rho=0 predictions")
            month_series = base["month"]
            y0 = base["sale_price"].to_numpy(dtype=float)
            p0 = base["pred_price"].to_numpy(dtype=float)
            ids0 = np.arange(len(base))
            raw_rows = []
            for p in anchors:
                tgt = slice_rho(df, p["rho_tilde"])
                if not len(tgt) or len(tgt) != len(base):
                    # Extra interpolated rhos are stored; TRANSACTIONID alignment required.
                    if not len(tgt):
                        continue
                    merged = base[["TRANSACTIONID"]].reset_index().merge(
                        tgt[["TRANSACTIONID", "pred_price", "sale_price"]],
                        on="TRANSACTIONID", how="inner",
                    )
                    if not len(merged):
                        continue
                    y = merged["sale_price"].to_numpy(dtype=float)
                    pred = merged["pred_price"].to_numpy(dtype=float)
                    month_aligned = month_series.iloc[merged["index"].to_numpy()].reset_index(drop=True)
                else:
                    y = tgt["sale_price"].to_numpy(dtype=float)
                    pred = tgt["pred_price"].to_numpy(dtype=float)
                    month_aligned = month_series
                if not np.allclose(y, y0) and len(y) == len(y0):
                    # sale prices should match on TRANSACTIONID; if not, inner-join.
                    pass
                boot = {name: [] for name in DELTA_NAMES}
                for b in range(N_BOOTSTRAP):
                    idx = resample_idx(month_aligned, draws[b])
                    if len(idx) < 50:
                        for name in DELTA_NAMES:
                            boot[name].append(np.nan)
                        continue
                    y_b = y[idx]
                    p_b = pred[idx]
                    y0_b = y0[idx] if len(y0) == len(y) else y_b
                    p0_b = p0[idx] if len(p0) == len(y) else p0[:len(idx)]
                    if len(y0) != len(y):
                        y0_b, p0_b = y_b, p0[np.clip(idx, 0, len(p0) - 1)]
                    m_b = cheap_metrics(y_b, p_b)
                    m0_b = cheap_metrics(y0_b, p0_b)
                    boot["Delta_NMSE"].append(m_b.get("NMSE", np.nan) - m0_b.get("NMSE", np.nan))
                    boot["Delta_R2_price"].append(m_b.get("R2_price", np.nan) - m0_b.get("R2_price", np.nan))
                    boot["Delta_PRD"].append(m_b.get("PRD", np.nan) - m0_b.get("PRD", np.nan))
                    boot["Delta_PRB"].append(m_b.get("PRB", np.nan) - m0_b.get("PRB", np.nan))
                    boot["Delta_MKI"].append(m_b.get("MKI", np.nan) - m0_b.get("MKI", np.nan))
                    boot["Delta_VEI"].append(m_b.get("VEI", np.nan) - m0_b.get("VEI", np.nan))
                    boot["Delta_beta_log"].append(m_b.get("beta_log", np.nan) - m0_b.get("beta_log", np.nan))
                    if len(idx) > MECH_CAP:
                        sub = rng.choice(len(idx), size=MECH_CAP, replace=False)
                        idx_m = idx[sub]
                    else:
                        idx_m = idx
                    mech_b = mech_metrics(y[idx_m], pred[idx_m], np.arange(len(idx_m)))
                    mech0 = mech_metrics(y0[idx_m] if len(y0) == len(y) else y[idx_m],
                                         p0[idx_m] if len(p0) == len(y) else p0[:len(idx_m)],
                                         np.arange(len(idx_m)))
                    boot["Delta_Delta_NL"].append(mech_b["Delta_NL"] - mech0["Delta_NL"])
                    boot["Delta_dCor"].append(mech_b["dCor"] - mech0["dCor"])
                rec = {
                    "county_key": key, "family": family, "role": p["role"],
                    "rho_tilde": float(p["rho_tilde"]),
                    "n_boot": N_BOOTSTRAP, "n_eval": int(len(y)),
                    "n_months": int(n_m),
                    "mech_cap": MECH_CAP,
                }
                for name in DELTA_NAMES:
                    arr = np.asarray(boot[name], dtype=float)
                    rec[f"{name}_mean"] = float(np.nanmean(arr))
                    rec[f"{name}_p025"] = float(np.nanpercentile(arr, 2.5)) if np.isfinite(arr).any() else np.nan
                    rec[f"{name}_p975"] = float(np.nanpercentile(arr, 97.5)) if np.isfinite(arr).any() else np.nan
                    rec[f"{name}_excludes_zero"] = bool(
                        np.isfinite(rec[f"{name}_p025"]) and np.isfinite(rec[f"{name}_p975"])
                        and (rec[f"{name}_p975"] < 0 or rec[f"{name}_p025"] > 0)
                    )
                ci_rows.append(rec)
                raw_rows.append({"role": p["role"], "rho_tilde": p["rho_tilde"], **{k: boot[k] for k in DELTA_NAMES}})
                print(f"{key} {family} {p['role']} bootstrap done", flush=True)
            write_json(RAW / f"{key}_{family}_anchor_draws.json", raw_rows)
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(OUT / "forward_anchor_bootstrap_ci.csv", index=False)
    print(json.dumps({"n_ci_rows": int(len(ci))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
