"""PASS/FAIL checks for the paper-v6 preselection smoke. No rho/model selection."""

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

ALLOWED_FAMILIES = {
    "LinearRegression",
    "LGBMRegressor",
    "LGBCovPenalty",
    "LGBSmoothPenalty",
}
BANNED_TOKENS = ("CVaR", "Logistic", "Group", "Stack", "div")
PARITY_ATOL = 1e-4
PARITY_RTOL = 1e-4


def _fail(msg: str) -> None:
    print(f"FAIL {msg}")
    raise SystemExit(1)


def _load_run_tables(root: Path) -> pd.DataFrame:
    files = list(root.glob("runs/**/fold_id=*/*.parquet"))
    if not files:
        files = list(root.glob("runs/**/*.parquet"))
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
            cfg = json.loads(raw)
            if "rho" in cfg:
                return float(cfg["rho"])
        except Exception:
            return np.nan
    return np.nan


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--result-root", type=str, required=True)
    p.add_argument("--section2-config", type=str, required=True)
    args = p.parse_args()
    root = Path(args.result_root)
    cfg = json.loads(Path(args.section2_config).read_text(encoding="utf-8"))
    freeze = json.loads((root / "frozen_baseline.json").read_text(encoding="utf-8"))
    params = dict(freeze.get("best_lgbm_params", freeze.get("lgbm_params", {})))
    expected_hash = cfg["lgbm_params_sha256"]
    got_hash = freeze.get("lgbm_params_sha256") or lgbm_params_hash(params)
    if got_hash != expected_hash:
        _fail(f"Section-2 hash mismatch freeze={got_hash} config={expected_hash}")
    if int(params.get("n_estimators", -1)) != 500:
        _fail(f"n_estimators={params.get('n_estimators')} is not the Section-2 500-tree vector")
    print("PASS freeze hash matches Section-2 config")
    print("PASS n_estimators=500")

    spec = json.loads((root / "experiment_spec.json").read_text(encoding="utf-8"))
    if spec.get("include_cvar_models"):
        _fail("CVaR models were included in the smoke spec")
    if spec.get("include_logistic_proxy"):
        _fail("logistic proxy was included in the smoke spec")
    if spec.get("early_stopping_rounds") not in (None, "None"):
        _fail(f"early_stopping_rounds={spec.get('early_stopping_rounds')}")
    rhos = [float(x) for x in spec.get("cov_rhos", [])]
    if 0.0 not in rhos:
        _fail("rho=0 missing from cov_rhos")
    print("PASS spec: identity/diff only, early stopping disabled, rho=0 present")
    print(f"INFO stage={spec.get('stage')} n_smooth={len(spec.get('smooth_rhos', []))} n_cov={len(spec.get('cov_rhos', []))}")

    try:
        import dcor

        print(f"PASS dcor import version={dcor.__version__}")
        rng = np.random.default_rng(0)
        a = rng.normal(size=40)
        b = 0.4 * a + rng.normal(scale=0.2, size=40)
        fast = float(dcor.distance_correlation(a, b, method="auto"))
        naive = float(dcor.distance_correlation(a, b, method="naive"))
        if not np.isfinite(fast) or abs(fast - naive) > 1e-10:
            _fail("dcor fast/naive mismatch or non-finite")
        print("PASS dcor fast matches naive")
    except Exception as exc:
        _fail(f"dcor failed on this node: {exc}")

    df = _load_run_tables(root)
    if df.empty:
        _fail("no smoke run parquet files found")
    name_col = "model_name" if "model_name" in df.columns else "name"
    names = sorted(df[name_col].astype(str).unique().tolist())
    extra = [n for n in names if n not in ALLOWED_FAMILIES]
    if extra:
        _fail(f"unwanted families: {extra}")
    banned = [n for n in names if any(tok in n for tok in BANNED_TOKENS)]
    if banned:
        _fail(f"banned family tokens: {banned}")
    print(f"PASS families={names}")

    df = df.copy()
    df["rho"] = df.apply(_rho, axis=1)
    metric_keys = [c for c in df.columns if any(k in c.lower() for k in ("r2", "rmse", "mae", "prd", "prb", "vei", "mki", "beta", "dcor"))]
    bad = [c for c in metric_keys if not np.isfinite(pd.to_numeric(df[c], errors="coerce")).all()]
    if bad:
        _fail(f"non-finite metrics: {bad[:8]}")
    print("PASS inspected metric columns finite")

    zero = df.loc[np.isclose(df["rho"].fillna(-1.0), 0.0)]
    direct0 = zero.loc[zero[name_col].astype(str) == "LGBCovPenalty"]
    smooth0 = zero.loc[zero[name_col].astype(str) == "LGBSmoothPenalty"]
    native = df.loc[df[name_col].astype(str) == "LGBMRegressor"]
    if direct0.empty or smooth0.empty:
        _fail("missing direct or surrogate rho=0 rows")
    key = "R2_price" if "R2_price" in df.columns else ("R2" if "R2" in df.columns else None)
    if key:
        dmean = float(direct0[key].mean())
        smean = float(smooth0[key].mean())
        if abs(dmean - smean) > max(PARITY_ATOL, PARITY_RTOL * max(abs(dmean), abs(smean), 1e-12)):
            _fail(f"direct-zero vs surrogate-zero {key} gap: {dmean:.6f} vs {smean:.6f}")
        print(f"PASS direct-zero ≈ surrogate-zero {key}")
        if not native.empty:
            nmean = float(native[key].mean())
            if abs(nmean - dmean) > 0.02:
                _fail(f"native vs custom-zero {key} degraded: {nmean:.6f} vs {dmean:.6f}")
            print(f"PASS native/custom-zero {key} near parity")
    print("PASS smoke QA finished (no rho selected)")


if __name__ == "__main__":
    main()
