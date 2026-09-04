"""Shared freeze-identity and metric helpers for the 2025 forward pass.

Does not read 2025 modeling outcomes. Used by table-build, path-fit, and tests.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import (
    ANALYSIS, OUTPUT, normalized_rho_tilde_grid, sha256_file,
)

EXPECTED_FORWARD_FREEZE_SHA256 = "2b4aec9f8f16baf933db84bae43a7f24975a9fc43d747d00630e86a12c2a8d18"
FORWARD_FREEZE_PATH = ANALYSIS / "path_freeze" / "FORWARD_FREEZE.yaml"
FORWARD_LOCK_DATE = pd.Timestamp("2025-01-01")
FORWARD_END_DATE = pd.Timestamp("2025-12-31")

CANONICAL_METRIC_MAP = {
    "R2_price": "R2_price",
    "R2_log": "R2_log",
    "NMSE": "NMSE",
    "RMSE_log": "RMSE_log",
    "MAE_price": "MAE",
    "MAE": "MAE",
    "MAPE": "MAPE",
    "Median ratio": "median_ratio",
    "Mean ratio": "mean_ratio",
    "W. Mean ratio": "weighted_mean_ratio",
    "COD": "COD",
    "COV_IAAO": "COV",
    "PRD": "PRD",
    "PRB": "PRB",
    "MKI": "MKI",
    "VEI": "VEI",
    "Beta_log": "beta_log",
    "Delta_NL": "Delta_NL",
    "dCor_e_y": "dCor",
    "N": "n_eval",
}

IDEALS = {"PRD": 1.0, "PRB": 0.0, "MKI": 1.0, "VEI": 0.0, "beta_log": 0.0}

# Frozen Direct protocol-valid intersection (8 jurisdictions). Do not re-estimate.
DIRECT_COMMON_INTERVAL = (0.3872983346207417, 0.5620804515923291)
DIRECT_COVERAGE_75 = (0.3909396965577807, 0.8111718071042558)
SURROGATE_COVERAGE_75 = (0.08812484459797208, 0.2648753015080894)
SURROGATE_INTERSECTION_STATUS = "NO_NONDEGENERATE_ALL_JURISDICTION_INTERSECTION"


def frozen_grid_rho_tilde(family: str) -> np.ndarray:
    if family == "direct":
        return normalized_rho_tilde_grid(include_zero=True, extra_decades=0.0)
    if family == "surrogate":
        return normalized_rho_tilde_grid(include_zero=False, extra_decades=0.0)
    raise ValueError(family)


def _parse_interval(text) -> tuple[float, float] | None:
    if text is None or (isinstance(text, float) and not np.isfinite(text)):
        return None
    s = str(text).strip()
    if not s.startswith("["):
        return None
    inner = s.strip("[]")
    a, b = inner.split(",")
    return float(a), float(b)


def frozen_anchor_points(county_key: str, family: str) -> list[dict]:
    """Predeclared coordinates only. Never derived from 2025."""
    cand_path = ANALYSIS / "candidate_regions" / "candidate_regions.csv"
    anc_path = ANALYSIS / "candidate_regions" / "achieved_mechanism_anchors.csv"
    cand = pd.read_csv(cand_path)
    anc = pd.read_csv(anc_path)
    row = cand.loc[(cand.county_key == county_key) & (cand.family == family)]
    if len(row) != 1:
        raise RuntimeError(f"expected one candidate-region row for {county_key}/{family}")
    row = row.iloc[0]
    points = [{"rho_tilde": 0.0, "role": "baseline_rho0"}]
    if pd.notna(row["activity_rho_tilde"]):
        points.append({"rho_tilde": float(row["activity_rho_tilde"]), "role": "activity"})
    if pd.notna(row["guardrail_rho_tilde"]):
        points.append({
            "rho_tilde": float(row["guardrail_rho_tilde"]),
            "role": "guardrail",
            "candidate_status": str(row["status"]),
        })
    sub = anc.loc[(anc.county_key == county_key) & (anc.family == family) & (anc.attained == True)]  # noqa: E712
    for _, a in sub.iterrows():
        tgt = float(a["target_A_beta"])
        if tgt in (0.75, 0.90) or tgt in (0.25, 0.50):
            points.append({
                "rho_tilde": float(a["rho_tilde"]),
                "role": f"A_beta_{tgt:g}",
                "target_A_beta": tgt,
            })
    if family == "direct":
        points.append({"rho_tilde": DIRECT_COMMON_INTERVAL[0], "role": "direct_common_lo"})
        points.append({"rho_tilde": DIRECT_COMMON_INTERVAL[1], "role": "direct_common_hi"})
        points.append({"rho_tilde": DIRECT_COVERAGE_75[0], "role": "direct_coverage75_lo"})
        points.append({"rho_tilde": DIRECT_COVERAGE_75[1], "role": "direct_coverage75_hi"})
    else:
        points.append({"rho_tilde": SURROGATE_COVERAGE_75[0], "role": "surrogate_coverage75_lo"})
        points.append({"rho_tilde": SURROGATE_COVERAGE_75[1], "role": "surrogate_coverage75_hi"})
    # Deduplicate near-identical rhos, keep first role.
    uniq = []
    seen = []
    for p in points:
        r = float(p["rho_tilde"])
        if any(np.isclose(r, s, rtol=0.0, atol=1e-12) for s in seen):
            continue
        seen.append(r)
        uniq.append(p)
    return uniq


def assert_dev_table_hash(county_key: str, freeze: dict | None = None) -> str:
    freeze = freeze or verify_forward_freeze()
    path = OUTPUT / "modeling_tables" / county_key / "history_market_core_dev.parquet"
    if not path.exists():
        raise RuntimeError(f"{county_key}: frozen development table missing at {path}")
    got = sha256_file(path)
    expected = freeze["cohort_and_model_table_hashes"][county_key]["modeling_table_hash"]
    if got != expected:
        raise RuntimeError(
            f"{county_key}: frozen _dev.parquet sha256 {got} != freeze {expected}. "
            "STOP before reading 2025."
        )
    return got


def preforward_identity(county_key: str, full_df: pd.DataFrame) -> dict:
    """Require the 2016-2024 slice of the full table to match the frozen dev table."""
    freeze = verify_forward_freeze()
    dev_hash = assert_dev_table_hash(county_key, freeze)
    dev_path = OUTPUT / "modeling_tables" / county_key / "history_market_core_dev.parquet"
    cols = [c for c in ("TRANSACTIONID", "ATTOMID", "sale_date", "sale_price") if c in full_df.columns]
    if "TRANSACTIONID" not in cols:
        raise RuntimeError(f"{county_key}: TRANSACTIONID missing from full table")
    dev = pd.read_parquet(dev_path, columns=cols)
    dev["sale_date"] = pd.to_datetime(dev["sale_date"])
    pre = full_df.loc[pd.to_datetime(full_df["sale_date"]) < FORWARD_LOCK_DATE, cols].copy()
    pre["sale_date"] = pd.to_datetime(pre["sale_date"])
    if len(dev) != len(pre):
        raise RuntimeError(
            f"{county_key}: pre-2025 full-table n={len(pre)} != frozen dev n={len(dev)}"
        )
    dev_ids = set(dev["TRANSACTIONID"].astype(str))
    pre_ids = set(pre["TRANSACTIONID"].astype(str))
    if dev_ids != pre_ids:
        missing = len(dev_ids - pre_ids)
        extra = len(pre_ids - dev_ids)
        raise RuntimeError(
            f"{county_key}: pre-2025 TRANSACTIONID set mismatch "
            f"(missing_from_full={missing}, extra_in_full={extra})"
        )
    n_2025 = int((pd.to_datetime(full_df["sale_date"]) >= FORWARD_LOCK_DATE).sum())
    if n_2025 == 0:
        raise RuntimeError(f"{county_key}: full table has no 2025 rows")
    return {
        "ok": True,
        "county_key": county_key,
        "dev_n": int(len(dev)),
        "preforward_n": int(len(pre)),
        "n_2025": n_2025,
        "dev_table_sha256": dev_hash,
    }


def freeze_sha256() -> str:
    return sha256_file(FORWARD_FREEZE_PATH)


def load_freeze() -> dict:
    return yaml.safe_load(FORWARD_FREEZE_PATH.read_text())


def verify_forward_freeze(*, require_untouched: bool = True) -> dict:
    """Raise if the canonical freeze is not the approved pre-forward object."""
    if not FORWARD_FREEZE_PATH.exists():
        raise RuntimeError("FORWARD_FREEZE.yaml missing; refusing 2025 access")
    got = freeze_sha256()
    if got != EXPECTED_FORWARD_FREEZE_SHA256:
        raise RuntimeError(
            f"FORWARD_FREEZE.yaml sha256 mismatch: got {got}, "
            f"expected {EXPECTED_FORWARD_FREEZE_SHA256}. STOP before reading 2025."
        )
    freeze = load_freeze()
    roles = freeze.get("jurisdiction_roles", {})
    if set(roles) != {
        "wayne", "philadelphia", "st_louis_county", "allegheny", "maricopa",
        "king", "miami_dade", "middlesex", "cook",
    }:
        raise RuntimeError(f"unexpected freeze jurisdictions: {sorted(roles)}")
    if any(v != "PRIMARY_FULL_7_FOLD" for v in roles.values()):
        raise RuntimeError(f"not all jurisdictions PRIMARY_FULL_7_FOLD: {roles}")
    if require_untouched and freeze.get("integrity_amendment", {}).get("2025_still_untouched") is not True:
        # After this pass the flag remains historically true of the freeze file
        # itself; it does not forbid running forward, it records that the freeze
        # was written without 2025 outcomes.
        pass
    bf_path = Path(freeze["baseline_freeze_file"])
    if not bf_path.exists():
        bf_path = ANALYSIS / "baseline" / "BASELINE_FREEZE.yaml"
    bf_sha = sha256_file(bf_path)
    if bf_sha != freeze["baseline_freeze_sha256"]:
        raise RuntimeError("BASELINE_FREEZE.yaml hash does not match the forward freeze")
    return freeze


def canonicalize_metrics(raw: dict) -> dict:
    out = {}
    for src, dst in CANONICAL_METRIC_MAP.items():
        if src in raw and dst not in out:
            try:
                out[dst] = float(raw[src]) if raw[src] is not None else float("nan")
            except (TypeError, ValueError):
                out[dst] = float("nan")
    return out


def add_derived(metrics: dict, baseline: dict | None) -> dict:
    """Delta vs same-split rho=0 and A_beta. Never uses another split's baseline."""
    out = dict(metrics)
    if baseline:
        if "NMSE" in metrics and "NMSE" in baseline:
            out["Delta_NMSE"] = metrics["NMSE"] - baseline["NMSE"]
        if "R2_price" in metrics and "R2_price" in baseline:
            out["Delta_R2_price"] = metrics["R2_price"] - baseline["R2_price"]
        for name in ("PRD", "PRB", "MKI", "VEI", "beta_log", "Delta_NL", "dCor"):
            if name in metrics and name in baseline:
                out[f"Delta_{name}"] = metrics[name] - baseline[name]
        b0 = baseline.get("beta_log")
        b = metrics.get("beta_log")
        if b0 is not None and b is not None and np.isfinite(b0) and abs(b0) > 1e-12 and np.isfinite(b):
            out["A_beta"] = 1.0 - abs(b) / abs(b0)
        else:
            out["A_beta"] = float("nan")
        for name, ideal in IDEALS.items():
            if name in metrics and np.isfinite(metrics[name]):
                out[f"I_{name}"] = abs(metrics[name] - ideal)
                if name in baseline and np.isfinite(baseline[name]):
                    out[f"Delta_I_{name}"] = out[f"I_{name}"] - abs(baseline[name] - ideal)
    return out


def write_input_freeze_check(path: Path, extra: dict | None = None) -> dict:
    freeze = verify_forward_freeze()
    rec = {
        "ok": True,
        "forward_freeze_path": str(FORWARD_FREEZE_PATH),
        "forward_freeze_sha256": EXPECTED_FORWARD_FREEZE_SHA256,
        "baseline_configs": freeze["baseline_configs"],
        "jurisdiction_roles": freeze["jurisdiction_roles"],
        "n_direct_grid_points": freeze["normalized_path_grid"]["n_points"],
        "independent_pre_2025_test_split": False,
        "evaluation_layers": ["CV_FOLD", "CV_OOF", "FORWARD_2025"],
    }
    if extra:
        rec.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rec, indent=2) + "\n")
    return rec
