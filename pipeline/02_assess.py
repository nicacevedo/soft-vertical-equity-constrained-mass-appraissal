#!/usr/bin/env python3
"""
Assess (CCAO ``02-assess`` analog).

Cook County applies the trained model to the full assessment universe and runs
multicard / land / rounding adjustments. This research codebase does not
replicate office-side adjustments; the closest analogue is **post-CV model
selection**.

This stage runs selection rules over the existing CV
artifacts and writes their winners for later stages to consume:

1. ``linear_regression`` — fixed untuned linear baseline; no hyperparameter or
   rho selection is performed.

2. ``lgbm_min_rmse`` — pick the LGBM configuration with the minimum mean fold
   RMSE (or MSE if ``--accuracy-metric MSE``). Mirrors the Cook County AVM
   ``select_best(lgbm_search, metric = params$cv$best_metric)`` logic in
   ``01-train.R``: pure validation-error minimization, no fairness
   constraints.

3. ``cov_penalty_min_mse`` — best ``LGBCovPenalty`` rho within family.

4. ``smooth_identity_min_mse`` — best ``LGBSmoothPenalty`` rho with
   ``weighting_proxy_mode="identity"``.

5. ``smooth_logistic_min_mse`` — best ``LGBSmoothPenalty`` rho with
   ``weighting_proxy_mode="logistic_quantile"``.

Outputs (under ``analysis/data_id=…/split_id=…/selected/``):

- ``selected_models.json`` — winners + per-fold metrics + held-out test metrics
- ``selected_models.csv``  — flat one-row-per-selection summary table

The pipeline context file (``pipeline/pipeline_last_context.json``) is
updated with the ``config_id`` winners so subsequent stages can read them
without re-loading the runs parquet.

Usage::

  python pipeline/02_assess.py
  python pipeline/02_assess.py --accuracy-metric MSE
  python pipeline/02_assess.py --nash-families LGBCovPenalty
  python pipeline/02_assess.py --penalized-selection-mode nash_only
  python pipeline/02_assess.py --constraint-metrics PRD,COD
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pipeline._helpers import (
    DEFAULT_RESULT_ROOT,
    parse_data_split_ids,
    read_context,
    write_context,
)
from pipeline._selection import (
    DEFAULT_ACCURACY_METRIC,
    DEFAULT_CCAO_FAMILIES,
    DEFAULT_CONSTRAINT_METRICS,
    DEFAULT_NASH_FAMILIES,
    DEFAULT_PENALIZED_SELECTION_MODE,
    DEFAULT_UTOPIA_FAMILIES,
    run_selection,
)


def _split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [tok.strip() for tok in str(value).split(",") if tok.strip()]


def main() -> None:
    p = argparse.ArgumentParser(description="Assess stage — model selection over CV artifacts.")
    p.add_argument("--result-root", type=str, default=str(DEFAULT_RESULT_ROOT))
    p.add_argument("--data-id", type=str, default=None)
    p.add_argument("--split-id", type=str, default=None)
    p.add_argument("--no-context", action="store_true")
    p.add_argument(
        "--accuracy-metric",
        type=str,
        default=DEFAULT_ACCURACY_METRIC,
        choices=["RMSE", "MSE", "MAE", "R2"],
        help="Metric used for the LGBM winner. Penalized mse_only mode uses MSE; Nash mode uses RMSE.",
    )
    p.add_argument(
        "--constraint-metrics",
        type=str,
        default=",".join(DEFAULT_CONSTRAINT_METRICS),
        help="Comma-separated fairness metrics recorded for selected models (default: PRD). Nash mode is fixed to PRD.",
    )
    p.add_argument(
        "--ccao-families",
        type=str,
        default=",".join(DEFAULT_CCAO_FAMILIES),
        help=(
            "Family pool for the CCAO-style minimum-RMSE selector. The default is "
            "strictly LGBMRegressor so the baseline LightGBM winner is selected "
            "separately from penalized families."
        ),
    )
    p.add_argument(
        "--nash-families",
        type=str,
        default=",".join(DEFAULT_NASH_FAMILIES),
        help="Comma-separated model families for the CovPenalty selector (default: LGBCovPenalty).",
    )
    p.add_argument(
        "--penalized-selection-mode",
        type=str,
        choices=["mse_only", "nash_only"],
        default=DEFAULT_PENALIZED_SELECTION_MODE,
        help=(
            "Selection rule for the penalized families. "
            "'mse_only' picks the minimum mean fold MSE within family; "
            "'nash_only' uses the RMSE+PRD Nash utility."
        ),
    )
    p.add_argument(
        "--utopia-families",
        type=str,
        default=None,
        help="Deprecated. Same as --nash-families if set.",
    )
    args = p.parse_args()

    result_root = Path(args.result_root).resolve()
    data_id, split_id = parse_data_split_ids(
        data_id=args.data_id,
        split_id=args.split_id,
        result_root=result_root,
        prefer_context=not args.no_context,
    )

    ccao_families = _split_csv(args.ccao_families) or None
    if args.utopia_families is not None:
        nash_families = _split_csv(args.utopia_families) or None
    else:
        nash_families = _split_csv(args.nash_families) or None

    payload = run_selection(
        result_root=result_root,
        data_id=data_id,
        split_id=split_id,
        accuracy_metric=str(args.accuracy_metric),
        constraint_metrics=_split_csv(args.constraint_metrics) or list(DEFAULT_CONSTRAINT_METRICS),
        ccao_families=ccao_families,
        utopia_families=nash_families,
        nash_families=nash_families,
        penalized_selection_mode=str(args.penalized_selection_mode),
    )

    print("=" * 70)
    print("ASSESS — model selection")
    print("=" * 70)
    print(f"  data_id={data_id}  split_id={split_id}")
    print(f"  penalized_selection_mode={payload.get('penalized_selection_mode')}")
    for rule, pool in payload["candidate_pools"].items():
        print(
            f"  pool[{rule}]: {pool['n_configs']} configs across {pool['n_folds']} folds "
            f"(families: {pool['families']})"
        )
    for rule, sel in payload["selections"].items():
        acc_key = f"cv_{payload['accuracy_metric']}_mean"
        if acc_key not in sel:
            acc_key = f"cv_{payload.get('penalized_accuracy_metric', payload['accuracy_metric'])}_mean"
        acc_val = sel.get(acc_key)
        print(f"  [{rule}] config_id={sel['config_id']}  model_name={sel['model_name']}")
        if acc_val is None:
            print(f"           {acc_key}=NA", end="")
        else:
            print(f"           {acc_key}={float(acc_val):.6g}", end="")
        if sel.get("selector_label"):
            print(f"  selector={sel.get('selector_label')}", end="")
        if "nash_log_utility" in sel:
            print(f"  nash_log_utility={sel.get('nash_log_utility', float('nan')):.6g}")
        else:
            print()
    print(f"  → JSON: {payload['json_path']}")
    print(f"  → CSV : {payload['csv_path']}")

    ctx = dict(read_context())
    ctx.update(
        {
            "stage": "assess",
            "data_id": data_id,
            "split_id": split_id,
            "result_root": str(result_root),
            "selected_models_json": payload["json_path"],
            "selected_models_csv": payload["csv_path"],
            "selected_config_ids": {
                rule: sel["config_id"] for rule, sel in payload["selections"].items()
            },
        }
    )
    write_context(ctx)


if __name__ == "__main__":
    main()
