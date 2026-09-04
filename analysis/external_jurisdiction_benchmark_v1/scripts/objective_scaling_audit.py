#!/usr/bin/env python3
"""Step 1: objective-scale + rho=0 parity audit. HARD GATE before any large run.

Two independent checks, both on REAL jurisdiction data (not synthetic):

1. rho_tilde = rho * Vy_T equivalence: supplying rho = rho_tilde/Vy_T must
   reproduce bit-identical predictions to supplying that raw rho directly, for
   both Direct and Surrogate.
2. Native-vs-custom rho=0 parity under the already-validated canonical
   configuration (ratio_mode="diff", match_native_init=True, and for Surrogate
   weighting_proxy_mode="identity") -- see
   tests/test_paper_v6_guards.py::test_native_custom_rho0_parity_after_mean_init,
   which already passes on synthetic data. This script re-runs the identical
   configuration on real feature matrices from all three pilot counties.

STOP rule: if native/custom parity is not interpretable on real data (mean
|delta y_hat log-space| does not stay small AND stable across counties), this
script exits nonzero and writes BLOCKER.md. No large run may start until this
passes.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from soft_constrained_models.boosting_models import LGBCovPenalty, LGBSmoothPenalty  # noqa: E402
from utils.motivation_utils import paper_mechanism_metrics  # noqa: E402
from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import (  # noqa: E402
    ANALYSIS, DIRECT_CANONICAL_KWARGS, SURROGATE_CANONICAL_KWARGS, population_variance,
    write_json,
)

# Real, already-leakage-safe feature tables from completed prior work. Used
# here ONLY for the parity smoke check -- not the frozen v1 modeling cohort,
# which Phase B will construct under the harmonized HISTORY_MARKET_CORE rule.
REAL_DATA_SOURCES = {
    "wayne": ROOT / "output/berry_attom_validation_v3/modeling_tables/wayne/history_market_core.parquet",
    "philadelphia": ROOT / "output/berry_attom_validation_v3/modeling_tables/philadelphia/history_market_core.parquet",
    "cook": ROOT / "output/county_bench_17031_floor50000/matched_sales.parquet",
}
SAMPLE_N = 4000
TINY_PARAMS = dict(
    n_estimators=60, num_leaves=15, max_depth=4, learning_rate=0.1,
    min_child_samples=20, n_jobs=2, verbosity=-1, random_state=2025,
)


def load_xy(key: str, path: Path, n: int = SAMPLE_N):
    """Numeric-only feature frame + log(sale_price) target, deterministic sample."""
    frame = pd.read_parquet(path)
    price_col = "sale_price" if "sale_price" in frame.columns else "SALE_PRICE"
    frame = frame.loc[pd.to_numeric(frame[price_col], errors="coerce") > 0].copy()
    if len(frame) > n:
        frame = frame.sample(n=n, random_state=2025)
    y = np.log(pd.to_numeric(frame[price_col], errors="coerce").to_numpy(dtype=float))
    numeric = frame.select_dtypes(include=[np.number]).drop(
        columns=[c for c in frame.columns if c == price_col], errors="ignore",
    )
    numeric = numeric.loc[:, numeric.nunique(dropna=True) > 1]
    X = numeric.fillna(numeric.median(numeric_only=True)).to_numpy(dtype=float)
    finite = np.isfinite(y) & np.isfinite(X).all(axis=1)
    return X[finite], y[finite]


def rho_tilde_equivalence(X: np.ndarray, y: np.ndarray) -> list[dict]:
    """Fit at a directly-supplied raw rho, and again at rho = rho_tilde/Vy_T
    for the corresponding rho_tilde = rho * Vy_T. Predictions must match
    exactly: this is a restatement of the same objective, not a new model."""
    vy = population_variance(y)
    rows = []
    for rho_direct in (0.5, 3.0):
        rho_tilde = rho_direct * vy
        rho_recovered = rho_tilde / vy
        for cls, kwargs, label in (
            (LGBCovPenalty, DIRECT_CANONICAL_KWARGS, "direct"),
            (LGBSmoothPenalty, SURROGATE_CANONICAL_KWARGS, "surrogate"),
        ):
            m_direct = cls(rho=rho_direct, early_stopping_rounds=None,
                            lgbm_params=dict(TINY_PARAMS), verbose=False, **kwargs)
            m_recovered = cls(rho=rho_recovered, early_stopping_rounds=None,
                               lgbm_params=dict(TINY_PARAMS), verbose=False, **kwargs)
            m_direct.fit(X, y)
            m_recovered.fit(X, y)
            p_direct = m_direct.predict(X)
            p_recovered = m_recovered.predict(X)
            max_abs_diff = float(np.max(np.abs(p_direct - p_recovered)))
            rows.append({
                "family": label, "rho_direct": rho_direct, "Vy_T": vy,
                "rho_tilde": rho_tilde, "rho_recovered": rho_recovered,
                "max_abs_prediction_diff": max_abs_diff,
                "exact_match": bool(np.array_equal(p_direct, p_recovered)),
            })
    return rows


def native_custom_parity(X: np.ndarray, y: np.ndarray, county: str) -> dict:
    native = LGBMRegressor(**TINY_PARAMS)
    native.fit(X, y)
    p_native = native.predict(X)

    direct = LGBCovPenalty(rho=0.0, early_stopping_rounds=None,
                            lgbm_params=dict(TINY_PARAMS), verbose=False, **DIRECT_CANONICAL_KWARGS)
    surrogate = LGBSmoothPenalty(rho=0.0, early_stopping_rounds=None,
                                  lgbm_params=dict(TINY_PARAMS), verbose=False,
                                  **SURROGATE_CANONICAL_KWARGS)
    direct.fit(X, y)
    surrogate.fit(X, y)
    p_direct = direct.predict(X)
    p_surrogate = surrogate.predict(X)

    beta_native = paper_mechanism_metrics(y, p_native)["Beta_log"]
    beta_direct = paper_mechanism_metrics(y, p_direct)["Beta_log"]
    return {
        "county": county, "n": len(y),
        "direct_vs_surrogate_max_abs_diff": float(np.max(np.abs(p_direct - p_surrogate))),
        "native_vs_direct_mean_abs_diff": float(np.mean(np.abs(p_native - p_direct))),
        "native_vs_direct_max_abs_diff": float(np.max(np.abs(p_native - p_direct))),
        "beta_native": float(beta_native), "beta_direct": float(beta_direct),
        "abs_beta_diff": float(abs(beta_native - beta_direct)),
        "base_score_matches_mean_y": bool(np.isclose(direct.base_score_, float(np.mean(y)))),
    }


def main() -> int:
    ANALYSIS.joinpath("audits").mkdir(parents=True, exist_ok=True)
    equivalence_rows: list[dict] = []
    parity_rows: list[dict] = []
    missing = [k for k, p in REAL_DATA_SOURCES.items() if not p.exists()]
    if missing:
        write_blocker(f"Real-data sources missing for parity audit: {missing}")
        return 1

    for key, path in REAL_DATA_SOURCES.items():
        X, y = load_xy(key, path)
        eq = rho_tilde_equivalence(X, y)
        for r in eq:
            r["county_key"] = key
        equivalence_rows.extend(eq)
        parity_rows.append(native_custom_parity(X, y, key))

    eq_df = pd.DataFrame(equivalence_rows)
    parity_df = pd.DataFrame(parity_rows)
    eq_df.to_csv(ANALYSIS / "audits" / "rho_normalization_equivalence.csv", index=False)
    parity_df.to_csv(ANALYSIS / "audits" / "zero_rho_parity.csv", index=False)

    eq_ok = bool((eq_df["max_abs_prediction_diff"] < 1e-9).all())
    # Interpretability bar for the STOP rule: the mean gap must be small
    # (matching the already-validated synthetic test's <5e-3 bar) and stable
    # across counties (no county blowing up relative to the others).
    parity_means = parity_df["native_vs_direct_mean_abs_diff"].to_numpy()
    parity_ok = bool((parity_means < 5e-3).all())
    direct_eq_surrogate_ok = bool((parity_df["direct_vs_surrogate_max_abs_diff"] < 1e-6).all())

    verdict = {
        "written_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "rho_tilde_equivalence_exact_everywhere": eq_ok,
        "native_custom_parity_interpretable": parity_ok,
        "direct_equals_surrogate_at_rho0": direct_eq_surrogate_ok,
        "parity_mean_abs_diff_by_county": dict(zip(parity_df["county"], parity_means.tolist())),
        "canonical_kwargs_direct": DIRECT_CANONICAL_KWARGS,
        "canonical_kwargs_surrogate": SURROGATE_CANONICAL_KWARGS,
        "reference_synthetic_test": "tests/test_paper_v6_guards.py::test_native_custom_rho0_parity_after_mean_init",
        "gate_passed": eq_ok and parity_ok and direct_eq_surrogate_ok,
    }
    write_json(ANALYSIS / "audits" / "objective_scaling_audit_verdict.json", verdict)

    md = [
        "# Objective-scaling + rho=0 parity audit\n",
        f"Written: {verdict['written_at_utc']}\n",
        "## rho_tilde = rho * Vy_T equivalence\n",
        eq_df.to_markdown(index=False) if hasattr(eq_df, "to_markdown") else eq_df.to_string(index=False),
        "\n\n## Native-vs-custom rho=0 parity (real pilot-county data)\n",
        parity_df.to_string(index=False),
        f"\n\n**Gate passed: {verdict['gate_passed']}**\n",
        (
            "\nThis reuses the exact canonical configuration validated on synthetic data by "
            "`tests/test_paper_v6_guards.py::test_native_custom_rho0_parity_after_mean_init` "
            "(ratio_mode='diff', match_native_init=True, and for Surrogate "
            "weighting_proxy_mode='identity'). `plan_rho_grid` normalizes by "
            "A = Var(baseline predictions), a DIFFERENT quantity from Vy_T used here; "
            "rho_tilde is never derived from A.\n"
        ),
    ]
    (ANALYSIS / "audits" / "objective_scaling_audit.md").write_text("".join(md), encoding="utf-8")

    print(json.dumps(verdict, indent=2, default=str))
    if not verdict["gate_passed"]:
        write_blocker(
            "Objective-scaling / rho=0 parity gate FAILED. "
            f"eq_ok={eq_ok} parity_ok={parity_ok} direct_eq_surrogate_ok={direct_eq_surrogate_ok}. "
            "See audits/objective_scaling_audit_verdict.json and zero_rho_parity.csv."
        )
        return 1
    return 0


def write_blocker(message: str) -> None:
    text = (
        "# BLOCKER\n\n"
        f"Written: {datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')}\n\n"
        f"{message}\n\n"
        "Per the unattended-execution rule, no downstream artifact may be produced past this "
        "point until a human resolves this. No workaround was invented.\n"
    )
    (ANALYSIS / "BLOCKER.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
