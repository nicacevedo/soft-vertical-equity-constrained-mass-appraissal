#!/usr/bin/env python3
"""Fast synthetic smoke test for the LGBCovPenalty comparison arm."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.other_counties_benchmars import (  # noqa: E402
    compare_penalized_models, load_lgbm_configs, parse_shrinkage_targets,
)

rng = np.random.default_rng(7)
# This is intentionally large enough for two held-out geographic groups to
# clear the production local-equity minimum (30 sales), but small enough for the
# six-point exact capped-neighbor search to stay a fast smoke test.
n = 360
dates = pd.date_range("2020-01-01", periods=n, freq="6h")
x1 = rng.normal(size=n)
x2 = rng.normal(size=n)
group = pd.Categorical(rng.choice(list("abcde"), size=n))
log_price = 12.0 + 0.7 * x1 + 0.3 * x2 + 0.25 * group.codes + rng.normal(scale=0.30, size=n)

data = pd.DataFrame({
    "sale_date": dates,
    "sale_price": np.exp(log_price),
    "AREABUILDING": np.exp(6 + 0.5 * x1),
    "AREALOTSF": np.exp(7 + 0.4 * x2),
    "YEARBUILT": (1950 + 60 * rng.random(n)).round(),
    "TAXYEARASSESSED": dates.year - 1,
    "PROPERTYUSESTANDARDIZED": group.astype(str),
    "ROOMSCOUNT": rng.integers(3, 10, size=n),
    "BEDROOMSCOUNT": rng.integers(1, 6, size=n),
    "BATHCOUNT": rng.integers(1, 4, size=n),
    "BATHPARTIALCOUNT": rng.integers(0, 2, size=n),
    "tax_assessor_latitude": 41.88 + rng.normal(scale=0.003, size=n),
    "tax_assessor_longitude": -87.63 + rng.normal(scale=0.003, size=n),
    "LATITUDE": 41.88 + rng.normal(scale=0.003, size=n),
    "LONGITUDE": -87.63 + rng.normal(scale=0.003, size=n),
    "tax_assessor_geoid": np.where(np.arange(n) % 2, "17031000100", "17031000200"),
})

configs = load_lgbm_configs(ROOT / "best_lgbm_baseline_configs.yaml", "cv_top2_r2", threads=1)
for params in configs.values():
    params.update(n_estimators=20, num_leaves=31, max_depth=6, learning_rate=0.2)

split = int(0.8 * n)
validation_split = int(0.9 * split)
target_log = np.log(data["sale_price"])
candidates = pd.DataFrame([
    {"feature_set": "ccao_core_acs", "target_scale": "log", "lgbm_config": "cv_top2_r2",
     "R2": 0.9, "R2 (log)": 0.9, "MAPE": 12.0, "COD": 10.0, "PRD": 1.02, "PRB": -0.02},
    {"feature_set": "ccao_core_acs", "target_scale": "raw", "lgbm_config": "cv_top2_r2",
     "R2": 0.8, "R2 (log)": 0.8, "MAPE": 15.0, "COD": 14.0, "PRD": 1.07, "PRB": -0.07},
])

targets = parse_shrinkage_targets("0.25,0.5,0.75")
print("shrinkage targets:", targets)

result = compare_penalized_models(
    data, configs=configs, candidates=candidates, split=split, validation_split=validation_split,
    target_log=target_log,
    shrinkage_targets=targets, county_fips="99999", n_bootstrap=8,
    bootstrap_block_freq="M", seed=1, early_stopping_rounds=None,
)

print("\n--- rho plan ---")
print(result["rho_plan"][[
    "rho", "rho_source", "requested_covariance_reduction",
    "delta_mse_log_frac_of_baseline", "realized_covariance_reduction_test",
]].to_string(index=False))

print("\n--- test metrics ---")
test = result["metrics"].query("split == 'test'")
print(test[["model", "rho", "R2 (log)", "COD", "PRD", "PRB", "Cov(e,logprice)"]].to_string(index=False))

print("\n--- bootstrap summary head ---")
print(result["bootstrap_summary"].query("metric == 'PRD'").to_string(index=False))

print("\n--- predictions columns ---")
print(list(result["predictions"].columns))

manifest = result["comparison_selection"]["model_manifest"]
test_keys = set(result["metrics"].loc[result["metrics"]["split"].eq("test"), "model_key"])
assert test_keys == set(manifest["expected_model_keys"])
assert result["predictions"].shape[0] == n
assert result["bootstrap_summary"]["mean"].notna().all()
assert set(result["local_equity_summary"]["model_key"]) == {
    "lgbm_baseline",
    result["comparison_selection"]["selected_neighbor_model_key"],
    result["comparison_selection"]["selected_neighbor_penalty_model_key"],
}
print("\nOK")
