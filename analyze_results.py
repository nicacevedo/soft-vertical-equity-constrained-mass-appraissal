"""
Entry point alias for results analysis + plotting.

This file exists to make the pipeline easier to discover:
  - `analyze_results.py` generates tables and plots from saved artifacts.
  - The legacy entry point `results_analysis.py` is still supported.
"""

from __future__ import annotations

from results_analysis import _build_arg_parser, run_results_analysis


if __name__ == "__main__":
    args = _build_arg_parser().parse_args()
    out = run_results_analysis(
        result_root=args.result_root,
        data_id=args.data_id,
        split_id=args.split_id,
        top_k=args.top_k,
        fairness_norm_threshold=args.fairness_norm_threshold,
        save_shortlist=(not args.no_shortlist),
        test_metrics_path=args.test_metrics_path,
        plot_top_k=args.plot_top_k,
        strict_test_safeguard=(not args.no_strict_test_safeguard),
    )
    print("=" * 90)
    print("RESULTS ANALYSIS COMPLETED")
    print("=" * 90)
    print(f"data_id={out['data_id']} | split_id={out['split_id']}")
    print(f"analysis_dir={out['analysis_dir']}")
    print(f"n_completed_runs={out['n_completed_runs']} | n_summary_configs={out['n_summary_configs']} | n_pareto={out['n_pareto']}")
    if out.get("test_metrics_path_used"):
        print(f"test_metrics_path={out['test_metrics_path_used']}")

