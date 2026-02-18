"""
Entry point alias for the stacking weight optimization.

This file exists to make the pipeline easier to discover:
  - `optimize_stacking_weights.py` solves the convex stacking problem.
  - The legacy entry point `optimize_stacking_pf.py` is still supported.
"""

from __future__ import annotations

from optimize_stacking_pf import _build_arg_parser, run_stacking_pf_optimization


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    out = run_stacking_pf_optimization(
        result_root=args.result_root,
        data_id=args.data_id,
        split_id=args.split_id,
        accuracy_metric=args.accuracy_metric,
        objective_mode=args.objective_mode,
        max_models=args.max_models,
    )
    print("=" * 90)
    print("STACKING WEIGHTS OPTIMIZATION COMPLETED")
    print("=" * 90)
    print(f"weights_csv={out['weights_csv']}")
    print(f"fold_metrics_csv={out['fold_metrics_csv']}")
    print(f"summary_json={out['summary_json']}")

