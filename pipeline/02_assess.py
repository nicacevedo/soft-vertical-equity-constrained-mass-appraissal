#!/usr/bin/env python3
"""
Assess (CCAO ``02-assess`` analog).

Cook County applies the trained model to the full assessment universe and runs
multicard / land / rounding adjustments. This research codebase does not
replicate office-side adjustments; the closest analogue is **post-CV model
selection**.

This stage runs selection rules over the existing CV
artifacts and writes their winners for later stages to consume:

1. ``ccao_min_rmse`` — pick the configuration with the minimum mean fold
   RMSE (or MSE if ``--accuracy-metric MSE``). Mirrors the Cook County AVM
   ``select_best(lgbm_search, metric = params$cv$best_metric)`` logic in
   ``01-train.R``: pure validation-error minimization, no fairness
   constraints.

2. ``nash`` — **Nash equilibrium** (product-of-utilities / log-sum) selection
   over the same positive utility transforms as ``simple_model_selection.py``:
   RMSE → ``1/RMSE``, R² → ``1+R²``, and IAAO-band utilities for PRD / PRB / VEI
   (**no min–max normalization across candidates**). Default pool:
   ``LGBCovPenalty`` (``--nash-families``).

3. ``smooth_penalty_nash`` — the ``LGBSmoothPenalty`` winner under the Nash
   product-of-utilities selector.

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
  python pipeline/02_assess.py --constraint-metrics PRD,PRB,VEI,COD
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
        help="Metric used for the CCAO-style winner and the Nash product (accuracy axis).",
    )
    p.add_argument(
        "--constraint-metrics",
        type=str,
        default=",".join(DEFAULT_CONSTRAINT_METRICS),
        help="Comma-separated fairness metrics for the Nash utilities (default: PRD,PRB,VEI).",
    )
    p.add_argument(
        "--ccao-families",
        type=str,
        default=",".join(DEFAULT_CCAO_FAMILIES),
        help=(
            "Family pool for the CCAO-style minimum-RMSE selector. CCAO tunes "
            "LightGBM, so the default mirrors that with the LightGBM-flavored "
            "families. Pass an empty string '' to consider every family."
        ),
    )
    p.add_argument(
        "--nash-families",
        type=str,
        default=",".join(DEFAULT_NASH_FAMILIES),
        help="Comma-separated model families for the Nash selector (default: LGBCovPenalty).",
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
    )

    print("=" * 70)
    print("ASSESS — model selection")
    print("=" * 70)
    print(f"  data_id={data_id}  split_id={split_id}")
    for rule, pool in payload["candidate_pools"].items():
        print(
            f"  pool[{rule}]: {pool['n_configs']} configs across {pool['n_folds']} folds "
            f"(families: {pool['families']})"
        )
    for rule, sel in payload["selections"].items():
        acc_key = f"cv_{payload['accuracy_metric']}_mean"
        print(f"  [{rule}] config_id={sel['config_id']}  model_name={sel['model_name']}")
        print(f"           cv_{payload['accuracy_metric']}_mean={sel.get(acc_key):.6g}", end="")
        if "nash" in str(rule):
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
