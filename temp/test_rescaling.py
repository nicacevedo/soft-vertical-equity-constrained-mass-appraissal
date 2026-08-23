"""Is the covariance penalty equivalent to rescaling the baseline's log predictions?

Sweeps a single slope on the baseline log prediction and traces the accuracy/equity
frontier it produces, for comparison with the frontier the penalty traces.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_county_penalty_dashboard import BASELINE_KEY, load_run
from scripts.other_counties_benchmars import score_predictions

pd.set_option("display.width", 300)

for folder in ["output/county_bench_17031_floor50000", "output/county_bench_42003_floor50000"]:
    run = load_run(Path(folder))
    train, test = run.train_predictions, run.predictions
    column = f"predicted_sale_price__{BASELINE_KEY}"
    train_log_pred = np.log(train[column].to_numpy())
    test_log_pred = np.log(test[column].to_numpy())
    centre = train_log_pred.mean()
    level = np.log(train["sale_price"].to_numpy()).mean()
    rows = []
    for slope in [1.0, 1.02, 1.05, 1.08, 1.11, 1.14, 1.18, 1.22, 1.26, 1.30]:
        corrected = np.exp(level + slope * (test_log_pred - centre))
        scored = score_predictions(
            test["sale_price"].to_numpy(), corrected, train["sale_price"].to_numpy()
        )
        rows.append({
            "slope": slope,
            "R2 (log)": scored["R2 (log)"],
            "rmse_pct": 100 * (scored["RMSE (log)"] / run.baseline["RMSE (log)"] - 1),
            "COD": scored["COD"],
            "PRD": scored["PRD"],
            "PRB": scored["PRB"],
            "MKI": scored["MKI"],
        })
    print("==", run.label, " baseline PRB", round(run.baseline["PRB"], 4))
    print(pd.DataFrame(rows).to_string(index=False))
