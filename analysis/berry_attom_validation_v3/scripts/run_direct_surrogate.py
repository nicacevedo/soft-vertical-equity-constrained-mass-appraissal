#!/usr/bin/env python3
"""Direct LGBCovPenalty[diff] and Surrogate LGBSmoothPenalty.

Runs only if freeze authorizes. Rho from pretest only.
Surrogate: first contiguous low-rho branch; no global sort+interp.

Surrogate pass 3. Pass 1 used a fixed 1e-6..1e2 grid whose ceiling sat below
the Direct 97% anchor, plus a branch detector with no noise floor. Pass 2 fixed
the grid but kept a hard-coded 0.01 floor that sits below the measured low-rho
noise envelope in all three counties (0.017 / 0.028 / 0.042), so a wiggle could
still cut a branch. Pass 3 estimates the floor from each curve's own inactive
tail. Every defect is diagnosable from pretest artifacts alone; none was found
by looking at test metrics, and rho is still never chosen from test. Superseded
outputs are kept as surrogate_pass{1,2}_*.csv.
See panel_freeze/SURROGATE_RECALIBRATION_LOG.md.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMRegressor

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.other_counties_benchmars import (  # noqa: E402
    load_lgbm_configs, plan_rho_grid, score_predictions,
)
from soft_constrained_models.boosting_models import LGBCovPenalty, LGBSmoothPenalty
from utils.delta_nl import estimate_delta_nl
from utils.motivation_utils import paper_mechanism_metrics
from analysis.berry_attom_validation_v3.scripts.run_prefreeze_baselines import v3_features  # noqa: E402
from analysis.berry_attom_validation_v3.scripts.v3_common import (  # noqa: E402
    ANALYSIS, LGBM_CONFIG_PATH, OUTPUT, chronological_splits, first_branch_calibrate,
    surrogate_rho_grid, write_json,
)

TARGETS = (0.10, 0.25, 0.50, 0.67, 0.80, 0.90, 0.97)
FREEZE = ANALYSIS / "panel_freeze" / "final_panel_freeze_v3.yaml"


def enrich(actual, predicted, train_actual) -> dict:
    m = score_predictions(actual, predicted, train_actual)
    m.update(paper_mechanism_metrics(np.log(actual), np.log(predicted)))
    try:
        m["Delta_NL"] = float(estimate_delta_nl(
            np.log(actual), np.log(predicted), row_ids=np.arange(len(actual))
        ).get("Delta_NL", np.nan))
    except Exception as exc:
        m["Delta_NL"] = np.nan
        m["Delta_NL_error"] = str(exc)
    ratio = predicted / actual
    q = pd.qcut(actual, 10, labels=False, duplicates="drop")
    m["ratio_by_decile"] = {str(int(i)): float(np.median(ratio[q == i])) for i in np.unique(q)}
    return m


def first_order_reduction(base_cov: float, new_cov: float) -> float:
    if not np.isfinite(base_cov) or abs(base_cov) < 1e-18:
        return float("nan")
    return float(1.0 - new_cov / base_cov)


SURROGATE_PASS = 3
PASS_FILES = (
    "surrogate_rho_first_branch.csv", "surrogate_branch_curve.csv",
    "surrogate_heldout.csv", "surrogate_full_grid_curve.csv",
    "surrogate_fit_errors.json",
)


def current_pass_on_disk(ana: Path) -> int:
    """Which calibration pass produced the output currently sitting here."""
    held = ana / "surrogate_heldout.csv"
    if not held.exists():
        return 0
    try:
        frame = pd.read_csv(held)
    except Exception:
        return 1
    if "surrogate_pass" not in frame.columns or not len(frame):
        return 1
    return int(frame["surrogate_pass"].iloc[0])


def preserve_prior_pass(ana: Path, out_dir: Path) -> list[str]:
    """Snapshot the superseded Surrogate outputs before overwriting them.

    Every pass is evidence, not scrap: the defects each one exposed are the
    documented reason for the next. Snapshots are written under the pass number
    that produced them and are never overwritten, so a later run cannot
    retroactively relabel its own numbers as an earlier pass.

    Superseded held-out prediction parquets are *moved* into a
    ``superseded_pass<N>/`` subdirectory rather than left in place. The
    bootstrap stage globs ``*_heldout.parquet`` to discover methods, so leaving
    them would silently bootstrap two or three calibration passes side by side
    as if they were distinct methods.
    """
    prior = current_pass_on_disk(ana)
    if prior <= 0 or prior >= SURROGATE_PASS:
        return []
    saved = []
    for name in PASS_FILES:
        src = ana / name
        dst = ana / name.replace("surrogate_", f"surrogate_pass{prior}_", 1)
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)
            saved.append(dst.name)
    stale = out_dir / f"superseded_pass{prior}"
    for src in sorted(out_dir.glob("surrogate_rho_*_heldout.parquet")):
        stale.mkdir(parents=True, exist_ok=True)
        dst = stale / src.name
        if not dst.exists():
            shutil.move(str(src), str(dst))
            saved.append(f"{stale.name}/{dst.name}")
    return saved


def _skip(county_key: str, path: str, reason: str) -> int:
    out = ANALYSIS / "method_transfer" / county_key
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / f"{path}_skip.json", {"county_key": county_key, "path": path, "reason": reason})
    print(json.dumps({"county_key": county_key, "path": path, "skipped": True, "reason": reason}))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--county-key", required=True)
    parser.add_argument("--path", choices=["direct", "surrogate"], required=True)
    parser.add_argument("--lgbm-threads", type=int, default=8)
    args = parser.parse_args()
    if not FREEZE.exists():
        return _skip(args.county_key, args.path, "no freeze file")
    freeze = yaml.safe_load(FREEZE.read_text())
    if not freeze.get("direct_surrogate_authorized"):
        return _skip(args.county_key, args.path, "freeze did not authorize Direct/Surrogate")
    passing = set(freeze.get("passing_model_transfer_units") or [])
    if args.county_key not in passing:
        return _skip(args.county_key, args.path, f"{args.county_key} not a passing MODEL_TRANSFER unit")
    held = OUTPUT / "final_models" / args.county_key / "heldout_predictions.parquet"
    if not held.exists():
        raise SystemExit("held-out baseline predictions missing; run final baselines first")

    pre = json.loads((ANALYSIS / "baselines_pre_freeze" / args.county_key / "run_meta.json").read_text())
    data = pd.read_parquet(OUTPUT / "modeling_tables" / args.county_key / "history_market_core.parquet")
    data = data.sort_values("sale_date").reset_index(drop=True)
    split, validation_split = chronological_splits(len(data))
    y_log = np.log(data["sale_price"].astype(float))
    features, cats = v3_features(data, split, True)
    params = load_lgbm_configs(
        LGBM_CONFIG_PATH, "test_best_r2,cv_top1_r2,cv_top2_r2", args.lgbm_threads,
    )[pre["selected_lgbm_config"]]
    base = LGBMRegressor(**params)
    base.fit(features.iloc[:split], y_log.iloc[:split], categorical_feature=cats)
    f0 = base.predict(features.iloc[:split])
    out_dir = OUTPUT / "method_transfer" / args.county_key
    out_dir.mkdir(parents=True, exist_ok=True)
    ana = ANALYSIS / "method_transfer" / args.county_key
    ana.mkdir(parents=True, exist_ok=True)

    if args.path == "direct":
        plan, _, _ = plan_rho_grid(
            y_log.iloc[:split].to_numpy(), f0, county_fips=args.county_key,
            config_key=pre["selected_lgbm_config"],
            shrinkage_targets=list(TARGETS), include_anchors=False,
        )
        plan.to_csv(ana / "direct_rho_from_pretest.csv", index=False)
        rows = []
        test = data.iloc[split:]
        train_price = data.sale_price.iloc[:split].to_numpy()
        for _, r in plan.iterrows():
            rho = float(r["rho"])
            model = LGBCovPenalty(
                rho=rho, ratio_mode="diff", early_stopping_rounds=None,
                zero_grad_tol=1e-12, lgbm_params=params, verbose=False,
            )
            model.fit(features.iloc[:split], y_log.iloc[:split])
            pred = np.exp(model.predict(features.iloc[split:]))
            mets = enrich(test.sale_price.to_numpy(), pred, train_price)
            rows.append({
                "rho": rho,
                "requested_reduction": r.get("requested_covariance_reduction"),
                **{k: v for k, v in mets.items() if k != "ratio_by_decile"},
                "ratio_by_decile": json.dumps(mets.get("ratio_by_decile")),
            })
            pd.DataFrame({
                "sale_date": test.sale_date.astype(str).to_numpy(),
                "y": test.sale_price.to_numpy(),
                "pred": pred,
            }).to_parquet(out_dir / f"direct_rho_{rho:.6g}_heldout.parquet", index=False)
        pd.DataFrame(rows).to_csv(ana / "direct_heldout.csv", index=False)
        return 0

    # Grid ceiling comes from this county's own Direct rho mapping, which is
    # itself pretest-only. A fixed 1e2 ceiling sat below the Direct 97% anchor.
    direct_plan = pd.read_csv(ana / "direct_rho_from_pretest.csv")
    grid = surrogate_rho_grid(float(direct_plan["rho"].max()))
    moved = preserve_prior_pass(ana, out_dir)
    if moved:
        print(json.dumps({"preserved_prior_pass": moved}), flush=True)
    val_y = data.sale_price.iloc[validation_split:split]
    val_train = data.sale_price.iloc[:validation_split]
    # Calibrate first-order reduction from a development-only fit, never the
    # pretest refit -- the validation block must stay out of the fit it scores.
    base_dev = LGBMRegressor(**params)
    base_dev.fit(
        features.iloc[:validation_split], y_log.iloc[:validation_split],
        categorical_feature=cats,
    )
    base_val = np.exp(base_dev.predict(features.iloc[validation_split:split]))
    base_cov = score_predictions(val_y, base_val, val_train).get("Cov(e,logprice)", np.nan)
    achieved = []
    fit_errors = []
    for rho in grid:
        try:
            model = LGBSmoothPenalty(
                rho=float(rho), ratio_mode="diff", early_stopping_rounds=None,
                lgbm_params=params, verbose=False,
            )
            model.fit(features.iloc[:validation_split], y_log.iloc[:validation_split])
            pred = np.exp(model.predict(features.iloc[validation_split:split]))
            cov = score_predictions(val_y, pred, val_train).get("Cov(e,logprice)", np.nan)
            achieved.append(first_order_reduction(base_cov, cov))
        except Exception as exc:
            achieved.append(np.nan)
            fit_errors.append({"rho": float(rho), "error": f"{type(exc).__name__}: {exc}"})
    write_json(ana / "surrogate_fit_errors.json", {
        "county_key": args.county_key,
        "base_cov_validation": float(base_cov),
        "grid_min": float(grid.min()),
        "grid_max": float(grid.max()),
        "grid_points": int(len(grid)),
        "max_direct_rho": float(direct_plan["rho"].max()),
        "n_fit_errors": len(fit_errors),
        "fit_errors": fit_errors,
    })
    frozen, branch = first_branch_calibrate(grid, np.array(achieved, dtype=float))
    pd.DataFrame({"rho": grid, "achieved_reduction": achieved}).to_csv(
        ana / "surrogate_full_grid_curve.csv", index=False,
    )
    frozen.to_csv(ana / "surrogate_rho_first_branch.csv", index=False)
    branch.to_csv(ana / "surrogate_branch_curve.csv", index=False)
    test = data.iloc[split:]
    train_price = data.sale_price.iloc[:split].to_numpy()
    rows = []
    for _, r in frozen.iterrows():
        if r["status"] == "UNATTAINED" or not np.isfinite(r["rho"]):
            rows.append({**r.to_dict(), "heldout_status": "UNATTAINED"})
            continue
        model = LGBSmoothPenalty(
            rho=float(r["rho"]), ratio_mode="diff", early_stopping_rounds=None,
            lgbm_params=params, verbose=False,
        )
        model.fit(features.iloc[:split], y_log.iloc[:split])
        pred = np.exp(model.predict(features.iloc[split:]))
        mets = enrich(test.sale_price.to_numpy(), pred, train_price)
        rows.append({
            **r.to_dict(),
            **{k: v for k, v in mets.items() if k != "ratio_by_decile"},
            "ratio_by_decile": json.dumps(mets.get("ratio_by_decile")),
            "heldout_status": "evaluated",
        })
        pd.DataFrame({
            "sale_date": test.sale_date.astype(str).to_numpy(),
            "y": test.sale_price.to_numpy(),
            "pred": pred,
        }).to_parquet(out_dir / f"surrogate_rho_{float(r['rho']):.6g}_heldout.parquet", index=False)
    out = pd.DataFrame(rows)
    out.insert(0, "surrogate_pass", SURROGATE_PASS)
    out.to_csv(ana / "surrogate_heldout.csv", index=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
