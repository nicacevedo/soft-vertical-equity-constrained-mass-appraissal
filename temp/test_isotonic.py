"""Can a price-varying correction beat both the penalty and the linear rescaling?

Fits an isotonic regression of log price on the baseline log prediction over the
training block, then blends it with the identity by a weight w so the strength of the
correction can be traced out the same way rho and the rescaling slope are.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.build_county_penalty_dashboard import BASELINE_KEY, load_run
from scripts.other_counties_benchmars import score_predictions

pd.set_option("display.width", 300)

for folder in [
    "output/county_bench_17031_floor50000",
    "output/county_bench_42003_floor50000",
    "output/county_bench_04013_floor50000",
    "output/county_bench_53033_floor50000",
]:
    run = load_run(Path(folder))
    train, test = run.train_predictions, run.predictions
    column = f"predicted_sale_price__{BASELINE_KEY}"
    train_log_pred = np.log(train[column].to_numpy())
    train_log_y = np.log(train["sale_price"].to_numpy())
    test_log_pred = np.log(test[column].to_numpy())
    iso = IsotonicRegression(out_of_bounds="clip").fit(train_log_pred, train_log_y)
    mapped = iso.predict(test_log_pred)
    rows = []
    for weight in [0.0, 0.25, 0.5, 0.75, 1.0]:
        corrected = np.exp((1 - weight) * test_log_pred + weight * mapped)
        scored = score_predictions(
            test["sale_price"].to_numpy(), corrected, train["sale_price"].to_numpy()
        )
        rows.append({
            "weight": weight,
            "R2 (log)": scored["R2 (log)"],
            "rmse_pct": 100 * (scored["RMSE (log)"] / run.baseline["RMSE (log)"] - 1),
            "COD": scored["COD"],
            "PRD": scored["PRD"],
            "PRB": scored["PRB"],
            "MKI": scored["MKI"],
        })
    print("==", run.label, "| baseline PRD", round(run.baseline["PRD"], 4),
          "PRB", round(run.baseline["PRB"], 4), "COD", round(run.baseline["COD"], 2))
    print(pd.DataFrame(rows).to_string(index=False))
