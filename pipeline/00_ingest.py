#!/usr/bin/env python3
"""
Ingest (CCAO ``00-ingest`` analog).

Upstream Cook County AVM pulls sales and characteristics from Athena, applies
sales-validation flags, and writes canonical parquet inputs.

This project already ships a frozen training extract at:

  data/CCAO/2025/training_data.parquet

This stage only **validates** that the file exists and prints a compact summary
so later stages fail fast with a clear message.

Usage::

  python pipeline/00_ingest.py
  python pipeline/00_ingest.py --data-path /path/to/training_data.parquet
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pipeline._helpers import DEFAULT_TRAINING_PARQUET, repo_root


def _read_parquet_sample(path: Path, *, max_rows: int) -> tuple[pd.DataFrame, int]:
    """Return (sample_df, total_rows) without materializing the full table."""
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise ImportError("Install pyarrow for efficient parquet ingest sampling.") from exc

    pf = pq.ParquetFile(path)
    total = int(pf.metadata.num_rows)
    take = max(1, min(max_rows, total))
    batches = []
    seen = 0
    for batch in pf.iter_batches(batch_size=min(take, 65_536)):
        batches.append(batch)
        seen += batch.num_rows
        if seen >= take:
            break
    if not batches:
        return pd.DataFrame(), total
    table = pa.Table.from_batches(batches).slice(0, take)
    return table.to_pandas(), total


def _dtypes_summary(df: pd.DataFrame, *, max_cols: int) -> None:
    print("\nColumn dtypes (first columns):")
    for i, (col, dt) in enumerate(df.dtypes.items()):
        if i >= max_cols:
            print(f"  ... ({df.shape[1] - max_cols} more columns)")
            break
        print(f"  {col}: {dt}")


def run_ingest(*, data_path: Path, sample_rows: int, list_columns: int) -> Dict[str, Any]:
    root = repo_root()
    path = (root / data_path).resolve() if not data_path.is_absolute() else data_path
    if not path.is_file():
        raise FileNotFoundError(
            f"Training parquet not found: {path}\n"
            "Place the Cook County training extract at the expected location or pass --data-path."
        )

    df_sample, n_rows = _read_parquet_sample(path, max_rows=sample_rows)

    summary: Dict[str, Any] = {
        "training_parquet": str(path),
        "n_rows": int(n_rows),
        "n_cols": int(df_sample.shape[1]),
        "sample_rows_loaded": int(df_sample.shape[0]),
        "memory_usage_mb": round(float(df_sample.memory_usage(deep=True).sum()) / (1024 * 1024), 2),
    }

    optional_flags = ["sv_is_outlier", "ind_pin_is_multicard"]
    for col in optional_flags:
        if col in df_sample.columns:
            vc = df_sample[col].dropna()
            summary[f"fraction_{col}_true_sample"] = float(vc.astype(bool).mean()) if len(vc) else float("nan")

    date_cols = [c for c in ("meta_sale_date", "sale_date") if c in df_sample.columns]
    if date_cols:
        col = date_cols[0]
        s = pd.to_datetime(df_sample[col], errors="coerce")
        summary[f"{col}_min_sample"] = str(s.min())
        summary[f"{col}_max_sample"] = str(s.max())

    print("INGEST SUMMARY")
    print("=" * 60)
    for k, v in summary.items():
        print(f"  {k}: {v}")
    _dtypes_summary(df_sample, max_cols=list_columns)
    print("\nIngest OK — training extract is readable.")
    return summary


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Validate the local training parquet (ingest stage).")
    p.add_argument(
        "--data-path",
        type=str,
        default=str(DEFAULT_TRAINING_PARQUET),
        help="Path to training_data.parquet (relative to repo root unless absolute).",
    )
    p.add_argument("--sample-rows", type=int, default=50_000, help="Rows to load for dtype peek (capped by file size).")
    p.add_argument("--list-columns", type=int, default=40, help="How many column dtypes to print.")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    run_ingest(data_path=Path(args.data_path), sample_rows=int(args.sample_rows), list_columns=int(args.list_columns))


if __name__ == "__main__":
    main()
