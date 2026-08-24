"""PASS/FAIL checks for the independent OOS paper-v6 smoke. No selection."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from canonical_experiment import lgbm_params_hash

ALLOWED = set(['LGBCovPenalty', 'LGBMRegressor', 'LGBSmoothPenalty', 'LinearRegression'])
PARITY_ATOL = 1e-3
PARITY_RTOL = 1e-3


def _fail(msg: str) -> None:
    print(f"FAIL {msg}")
    raise SystemExit(1)


def _load_metric_shards(root: Path) -> pd.DataFrame:
    files = list(root.glob("analysis/**/test_run_metrics/*.parquet"))
    files += list(root.glob("analysis/**/assess_run_metrics/*.parquet"))
    if not files:
        files = list(root.glob("**/*run_metrics/*.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def _rho(row: pd.Series) -> float:
    if "rho" in row.index and pd.notna(row.get("rho")):
        try:
            return float(row["rho"])
        except (TypeError, ValueError):
            pass
    raw = row.get("model_config_json")
    if isinstance(raw, str) and raw.strip():
        try:
            return float(json.loads(raw).get("rho", np.nan))
        except Exception:
            return np.nan
    return np.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", required=True)
    parser.add_argument("--section2-config", required=True)
    args = parser.parse_args()
    root = Path(args.result_root)
    cfg = json.loads(Path(args.section2_config).read_text(encoding="utf-8"))
    freeze_path = root / "frozen_baseline.json"
    if not freeze_path.is_file():
        freeze_path = root / "frozen_baseline.json"
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    params = dict(freeze.get("best_lgbm_params", freeze.get("lgbm_params", {})))
    expected = cfg.get("lgbm_params_sha256") or lgbm_params_hash(cfg.get("lgbm_params", {}))
    got = freeze.get("lgbm_params_sha256") or lgbm_params_hash(params)
    if got != expected:
        _fail(f"Section-2 hash mismatch freeze={got} config={expected}")
    if int(params.get("n_estimators", -1)) != 500:
        _fail(f"n_estimators={params.get('n_estimators')} is not 500")
    print("PASS freeze hash and n_estimators=500")

    spec_path = root / "experiment_spec.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec.get("include_cvar_models") or spec.get("include_logistic_proxy"):
        _fail("unexpected CVaR/logistic family in smoke spec")
    if spec.get("early_stopping_rounds") not in (None, "None"):
        _fail(f"early_stopping_rounds={spec.get('early_stopping_rounds')}")
    print("PASS spec: no CVaR/logistic, early stopping off")

    import dcor

    print(f"PASS dcor import version={dcor.__version__}")

    df = _load_metric_shards(root)
    if df.empty:
        _fail("no OOS metric shards found")
    name_col = "model_name" if "model_name" in df.columns else "name"
    names = sorted(df[name_col].astype(str).unique().tolist())
    extra = [n for n in names if n not in ALLOWED]
    if extra:
        _fail(f"unwanted families: {extra}")
    print(f"PASS families={names}")

    df = df.copy()
    df["rho"] = df.apply(_rho, axis=1)
    metric_cols = [c for c in ("R2_price", "RMSE_log", "PRD", "VEI", "Beta_log", "dCor_e_y") if c in df.columns]
    for col in metric_cols:
        if not np.isfinite(pd.to_numeric(df[col], errors="coerce")).all():
            _fail(f"non-finite {col}")
    print("PASS finite headline metrics")

    zero = df.loc[np.isclose(df["rho"].fillna(-1.0), 0.0)]
    direct0 = zero.loc[zero[name_col].astype(str) == "LGBCovPenalty"]
    smooth0 = zero.loc[zero[name_col].astype(str) == "LGBSmoothPenalty"]
    native = df.loc[df[name_col].astype(str) == "LGBMRegressor"]
    if direct0.empty or smooth0.empty:
        _fail("missing direct or surrogate rho=0")
    key = "R2_price"
    dmean = float(direct0[key].mean())
    smean = float(smooth0[key].mean())
    if abs(dmean - smean) > max(PARITY_ATOL, PARITY_RTOL * max(abs(dmean), abs(smean), 1e-12)):
        _fail(f"direct0 vs surrogate0 {key}: {dmean:.6f} vs {smean:.6f}")
    print(f"PASS direct0 ≈ surrogate0 {key}")
    if not native.empty:
        nmean = float(native[key].mean())
        if abs(nmean - dmean) > 0.03:
            _fail(f"native vs custom0 {key}: {nmean:.6f} vs {dmean:.6f}")
        print("PASS native ≈ custom0")
    print("PASS OOS smoke QA (no rho selected)")


if __name__ == "__main__":
    main()
