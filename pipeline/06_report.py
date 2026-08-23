#!/usr/bin/env python3
"""
Five-model comparison report (CCAO ``performance.qmd``-style analog).

Cook County renders a Quarto HTML report (``reports/performance/performance.qmd``)
from the parquet artifacts produced by ``pipeline/03-evaluate.R``. The report
is organized around a single model, but its content — overall ratio-study
metrics, geography breakouts, decile-by-price ratio curves — is exactly what
a selected-model comparison report needs. Cook County itself does the
cross-model comparison in a separate Tableau dashboard; this stage produces
the equivalent in plain HTML, in pure Python, using the artifacts already
produced by stages 02 and 03.

The report compares these models by default:

  1. **Linear baseline** — untuned ``LinearRegression``.
  2. **LGBM baseline** — best ``LGBMRegressor`` hyperparameters by validation error.
  3. **CovPenalty** — best ``LGBCovPenalty`` rho by validation error.
  4. **SmoothPenalty identity** — best ``LGBSmoothPenalty`` rho with
     ``weighting_proxy_mode="identity"``.
  5. **SmoothPenalty logistic-quantile** — best ``LGBSmoothPenalty`` rho with
     ``weighting_proxy_mode="logistic_quantile"``.

Outputs (under ``analysis/data_id=…/split_id=…/selected/report/``):

- ``five_model_comparison.html``     — full report (single HTML file)
- ``five_model_metrics.csv``         — flat metrics table (model × scope)
- ``five_model_decile.csv``          — long table (model × scope × decile)
- ``five_model_township_error.csv`` — median assessment-ratio error by township
- ``five_model_tract_error.csv``    — median assessment-ratio error by Census tract

Usage::

  python pipeline/06_report.py
  python pipeline/06_report.py --reference-config-id 4a382ebb24aa1108
  python pipeline/06_report.py --training-data data/CCAO/2025/training_data.parquet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

import pandas as pd

from pipeline._helpers import (
    DEFAULT_RESULT_ROOT,
    parse_data_split_ids,
    read_context,
    write_context,
)
from pipeline._report import (
    build_decile_table,
    build_metrics_table,
    load_predictions_with_geography,
    render_html_report,
    resolve_five_models,
)


_DEFAULT_TRAINING_DATA = Path("data/CCAO/2025/training_data.parquet")
_TRIAD_ORDER = ("Overall", "City", "North", "South")


def _read_selected_models(analysis_dir: Path) -> Dict[str, Any]:
    json_path = analysis_dir / "selected" / "selected_models.json"
    if not json_path.is_file():
        raise FileNotFoundError(
            f"selected_models.json not found at {json_path}. Run pipeline/02_assess.py first."
        )
    return json.loads(json_path.read_text(encoding="utf-8"))


def _selection_split_windows(*, result_root: Path, data_id: str, split_id: str) -> Dict[str, str]:
    protocol_path = result_root / "protocol" / f"data_id={data_id}" / f"split_id={split_id}" / "folds.json"
    if not protocol_path.is_file():
        return {}
    try:
        payload = json.loads(protocol_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    folds = payload.get("folds") or []
    if not isinstance(folds, list) or not folds:
        return {}

    def _window(start_key: str, end_key: str) -> str:
        starts = sorted(str(f.get(start_key, "")).strip() for f in folds if f.get(start_key))
        ends = sorted(str(f.get(end_key, "")).strip() for f in folds if f.get(end_key))
        if not starts or not ends:
            return ""
        return f"{starts[0]} to {ends[-1]}"

    windows: Dict[str, str] = {}
    train_window = _window("train_start", "train_end")
    validation_window = _window("val_start", "val_end")
    if train_window:
        windows["model_selection_train_window"] = train_window
    if validation_window:
        windows["model_selection_validation_window"] = validation_window
    return windows


def _held_out_test_window(df: pd.DataFrame) -> str:
    if df.empty or "sale_date" not in df.columns:
        return ""
    sale_dates = pd.to_datetime(df["sale_date"], errors="coerce").dropna()
    if sale_dates.empty:
        return ""
    return f"{sale_dates.min().date().isoformat()} to {sale_dates.max().date().isoformat()}"


def _report_subtitle(selected_models_json: Dict[str, Any]) -> str:
    penalized_mode = str(selected_models_json.get("penalized_selection_mode", "mse_only")).strip().lower()
    if penalized_mode == "nash_only":
        penalized_text = (
            "(3) the Nash-selected CovPenalty model, (4) the Nash-selected SmoothPenalty "
            "identity model, and (5) the Nash-selected SmoothPenalty logistic-quantile model."
        )
    else:
        penalized_text = (
            "(3) the family-specific minimum mean fold MSE CovPenalty model, "
            "(4) the family-specific minimum mean fold MSE SmoothPenalty identity model, and "
            "(5) the family-specific minimum mean fold MSE SmoothPenalty logistic-quantile model."
        )
    return (
        "Held-out test ratio-study comparison of (1) the untuned linear baseline, "
        "(2) the tuned LGBM baseline, "
        f"{penalized_text} Metrics are reported overall and broken out by Cook County "
        "triad (City / North / South). Township and Census tract maps summarize "
        "geographic median assessment-ratio error on fixed polygons in pipeline/geo_data."
    )


def main() -> None:
    p = argparse.ArgumentParser(description="Report stage — selected-model comparison HTML report.")
    p.add_argument("--result-root", type=str, default=str(DEFAULT_RESULT_ROOT))
    p.add_argument("--data-id", type=str, default=None)
    p.add_argument("--split-id", type=str, default=None)
    p.add_argument("--no-context", action="store_true")
    p.add_argument(
        "--reference-config-id",
        type=str,
        default=None,
        help=(
            "Override the linear reference model with a specific config_id from test_metrics.csv. "
            "Default: use the selected LinearRegression baseline."
        ),
    )
    p.add_argument(
        "--training-data",
        type=str,
        default=str(_DEFAULT_TRAINING_DATA),
        help="Path to training_data.parquet (used to attach triad / township / class).",
    )
    p.add_argument("--n-deciles", type=int, default=10)
    p.add_argument("--n-min", type=int, default=30, help="Minimum observations needed per (model × scope).")
    args = p.parse_args()

    result_root = Path(args.result_root).resolve()
    data_id, split_id = parse_data_split_ids(
        data_id=args.data_id,
        split_id=args.split_id,
        result_root=result_root,
        prefer_context=not args.no_context,
    )
    analysis_dir = result_root / "analysis" / f"data_id={data_id}" / f"split_id={split_id}"
    test_metrics_path = analysis_dir / "test_metrics.csv"
    if not test_metrics_path.is_file():
        raise FileNotFoundError(f"test_metrics.csv not found at {test_metrics_path}")
    test_metrics_df = pd.read_csv(test_metrics_path)
    test_metrics_df["config_id"] = test_metrics_df["config_id"].astype(str)

    selected = _read_selected_models(analysis_dir)
    models = resolve_five_models(
        selected_models_json=selected,
        test_metrics_df=test_metrics_df,
        reference_config_id=args.reference_config_id,
    )

    config_ids = [m.config_id for m in models]
    training_data = Path(args.training_data)
    if not training_data.is_absolute():
        training_data = (_REPO / training_data).resolve()
    df = load_predictions_with_geography(
        result_root=result_root,
        data_id=data_id,
        split_id=split_id,
        config_ids=config_ids,
        training_data_path=training_data,
    )

    out_dir = analysis_dir / "selected" / "report"
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_df = build_metrics_table(df=df, models=models, triad_order=_TRIAD_ORDER)
    decile_df = build_decile_table(df=df, models=models, n_deciles=args.n_deciles, triad_order=_TRIAD_ORDER)
    validation_summary_path = analysis_dir / "selected" / "selected_models_evaluation.csv"
    validation_df = pd.read_csv(validation_summary_path) if validation_summary_path.is_file() else None

    metrics_csv = out_dir / "five_model_metrics.csv"
    decile_csv = out_dir / "five_model_decile.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    decile_df.to_csv(decile_csv, index=False)

    triad_counts = (
        df.groupby("meta_triad_name", dropna=False)["row_id"].nunique().to_dict()
        if not df.empty
        else {}
    )
    split_windows = _selection_split_windows(result_root=result_root, data_id=data_id, split_id=split_id)
    held_out_test_window = _held_out_test_window(df)
    metadata_kv: Dict[str, str] = {
        "data_id": data_id,
        "split_id": split_id,
        **split_windows,
        "result_root": str(result_root),
        "training_data": str(training_data),
        "n_test_rows_total": f"{int(df['row_id'].nunique()) if not df.empty else 0:,}",
        "n_test_rows_overall (joined)": f"{int(df.shape[0] // max(len(models), 1)):,}",
        "triad_counts": ", ".join(
            f"{k}: {v:,}" for k, v in sorted(triad_counts.items(), key=lambda kv: str(kv[0]))
        ),
        "ratio_error_bands": "township maps ±5%; Census tract maps and decile ratio curves ±10%",
        "n_deciles": str(args.n_deciles),
        "min_obs_per_scope": str(args.n_min),
    }
    if held_out_test_window:
        metadata_kv["held_out_test_window"] = held_out_test_window

    html, township_df, tract_df = render_html_report(
        title="Five-Model Comparison",
        subtitle=_report_subtitle(selected),
        metadata_kv=metadata_kv,
        models=models,
        metrics_df=metrics_df,
        decile_df=decile_df,
        validation_df=validation_df,
        predictions_geography_df=df,
    )
    html_path = out_dir / "five_model_comparison.html"
    html_path.write_text(html, encoding="utf-8")

    township_csv: Optional[Path] = None
    tract_csv: Optional[Path] = None
    if township_df is not None and not township_df.empty:
        township_csv = out_dir / "five_model_township_error.csv"
        township_df.to_csv(township_csv, index=False)
    if tract_df is not None and not tract_df.empty:
        tract_csv = out_dir / "five_model_tract_error.csv"
        tract_df.to_csv(tract_csv, index=False)

    print("=" * 70)
    print("REPORT — selected-model comparison")
    print("=" * 70)
    print(f"  data_id={data_id}  split_id={split_id}")
    print("  models compared:")
    for m in models:
        print(f"    [{m.label}] {m.model_name}  config_id={m.config_id}")
    print(f"  → HTML : {html_path}")
    print(f"  → CSV  : {metrics_csv}")
    print(f"  → CSV  : {decile_csv}")
    if township_csv is not None:
        print(f"  → CSV  : {township_csv}")
    if tract_csv is not None:
        print(f"  → CSV  : {tract_csv}")

    ctx = dict(read_context())
    ctx.update(
        {
            "stage": "report",
            "data_id": data_id,
            "split_id": split_id,
            "result_root": str(result_root),
            "report_html": str(html_path),
            "report_metrics_csv": str(metrics_csv),
            "report_decile_csv": str(decile_csv),
            "report_township_csv": str(township_csv) if township_csv else "",
            "report_tract_csv": str(tract_csv) if tract_csv else "",
        }
    )
    write_context(ctx)


if __name__ == "__main__":
    main()
