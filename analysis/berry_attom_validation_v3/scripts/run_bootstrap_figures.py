#!/usr/bin/env python3
"""Shared 200 monthly-block bootstrap across baseline/Direct/Surrogate within a county."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.other_counties_benchmars import score_predictions  # noqa: E402
from utils.motivation_utils import _build_time_block_bootstrap_indices, paper_mechanism_metrics
from analysis.berry_attom_validation_v3.scripts.v3_common import (  # noqa: E402
    ANALYSIS, COUNTIES, N_BOOTSTRAP, OUTPUT, SEED, chronological_splits,
)

FIG = ANALYSIS / "figures"
METRIC_KEYS = [
    "R2", "RMSE_log", "MAE", "MAPE", "COD", "PRD", "PRB", "MKI", "VEI",
    "Beta_log", "Delta_NL", "dCor", "level",
]


def _score(y, pred, train_price) -> dict:
    m = score_predictions(y, pred, train_price)
    m.update(paper_mechanism_metrics(np.log(y), np.log(pred)))
    return m


def bootstrap_one(y, pred, train_price, indices) -> pd.DataFrame:
    rows = []
    for idx in indices:
        rows.append(_score(y[idx], pred[idx], train_price))
    return pd.DataFrame(rows)


def decile_profile(y, pred) -> pd.Series:
    ratio = pred / y
    q = pd.qcut(y, 10, labels=False, duplicates="drop")
    return pd.Series({int(i) + 1: float(np.median(ratio[q == i])) for i in np.unique(q)})


def summarize(draws: pd.DataFrame) -> pd.DataFrame:
    numeric = draws.select_dtypes(include="number")
    return pd.DataFrame({
        "metric": numeric.columns,
        "mean": numeric.mean().to_numpy(),
        "std": numeric.std(ddof=1).to_numpy(),
        "ci_2_5": numeric.quantile(0.025).to_numpy(),
        "ci_97_5": numeric.quantile(0.975).to_numpy(),
    })


def county_work(key: str) -> None:
    held = OUTPUT / "final_models" / key / "heldout_predictions.parquet"
    table = OUTPUT / "modeling_tables" / key / "history_market_core.parquet"
    if not held.exists() or not table.exists():
        print("skip", key, "missing held-out or table", flush=True)
        return
    data = pd.read_parquet(table).sort_values("sale_date").reset_index(drop=True)
    split, _ = chronological_splits(len(data))
    test = data.iloc[split:].reset_index(drop=True)
    preds = pd.read_parquet(held).reset_index(drop=True)
    y = preds["y"].to_numpy() if "y" in preds else test["sale_price"].to_numpy()
    dates = pd.to_datetime(preds["sale_date"] if "sale_date" in preds else test["sale_date"])
    train_price = data.sale_price.iloc[:split].to_numpy()
    indices = _build_time_block_bootstrap_indices(dates, N_BOOTSTRAP, "M", SEED)
    np.save(OUTPUT / "final_models" / key / "bootstrap_indices.npy", np.stack(indices))
    methods = {"baseline_lgbm": preds["lgbm"].to_numpy()}
    if "lr" in preds:
        methods["baseline_lr"] = preds["lr"].to_numpy()
    mt = OUTPUT / "method_transfer" / key
    if mt.exists():
        for p in sorted(mt.glob("*_heldout.parquet")):
            methods[p.stem] = pd.read_parquet(p)["pred"].to_numpy()
    ana = ANALYSIS / "final_baselines" / key
    ana.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    profiles = []
    summaries = []
    for name, pred in methods.items():
        if len(pred) != len(y):
            print("length mismatch", key, name, len(pred), len(y), flush=True)
            continue
        draws = bootstrap_one(y, pred, train_price, indices)
        draws.to_csv(ana / f"bootstrap_draws_{name}.csv", index=False)
        summ = summarize(draws).assign(method=name, county_key=key)
        summaries.append(summ)
        prof = decile_profile(y, pred)
        profiles.append(pd.DataFrame({
            "county_key": key, "method": name, "decile": prof.index, "median_ratio": prof.values,
        }))
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(ana / "bootstrap_ci_all_methods.csv", index=False)
    if profiles:
        prof_df = pd.concat(profiles, ignore_index=True)
        prof_df.to_csv(ana / "decile_valuation_ratio_profiles.csv", index=False)
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        for method, g in prof_df.groupby("method"):
            ax.plot(g["decile"], g["median_ratio"], marker="o", label=method)
        ax.axhline(1.0, color="gray", lw=0.8)
        ax.set_xlabel("sale-price decile")
        ax.set_ylabel("median valuation / sale")
        ax.set_title(f"{key} held-out ratio profiles")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(FIG / f"{key}_heldout_ratio_by_decile.pdf")
        plt.close(fig)
    print("bootstrap done", key, "n_methods", len(methods), flush=True)


def main() -> int:
    for c in COUNTIES:
        county_work(c["key"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
