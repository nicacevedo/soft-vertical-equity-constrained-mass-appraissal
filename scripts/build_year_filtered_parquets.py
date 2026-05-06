"""Create year-filtered training parquets from data/CCAO/2025/training_data.parquet.

The 2025 snapshot contains sales 2017-2024. We simulate two assessment-year
contexts by truncating the max sale year:

- sim2023: sales through 2022 (what a 2023 assessment run would have used)
- sim2024: sales through 2023 (what a 2024 assessment run would have used)

Both outputs are written alongside the source parquet.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq


SRC = Path("data/CCAO/2025/training_data.parquet")
OUT_DIR = SRC.parent
TARGETS = {
    "sim2023": 2022,  # keep time_sale_year <= 2022
    "sim2024": 2023,  # keep time_sale_year <= 2023
}


def _load_year_column(src: Path) -> pd.Series:
    """Load the minimal column used for filtering without blowing up memory."""
    for col in ("time_sale_year", "meta_year"):
        try:
            return pd.read_parquet(src, columns=[col])[col]
        except Exception:
            continue
    raise RuntimeError("Neither time_sale_year nor meta_year is present in the parquet")


def _build(tag: str, max_year: int, src: Path, force: bool) -> Path:
    out = OUT_DIR / f"training_data_{tag}.parquet"
    if out.exists() and not force:
        print(f"[skip] {out} already exists; pass --force to overwrite")
        return out

    print(f"[{tag}] reading full parquet from {src}")
    df = pd.read_parquet(src)
    print(f"[{tag}] loaded rows={len(df):,}, cols={df.shape[1]}")

    # Prefer time_sale_year (float); fall back to meta_year (string-ish)
    mask = None
    if "time_sale_year" in df.columns:
        col = pd.to_numeric(df["time_sale_year"], errors="coerce")
        mask = col <= max_year
    elif "meta_year" in df.columns:
        col = pd.to_numeric(df["meta_year"], errors="coerce")
        mask = col <= max_year
    else:
        raise RuntimeError("No year column found")

    out_df = df.loc[mask].copy()
    dropped = len(df) - len(out_df)
    print(f"[{tag}] kept rows={len(out_df):,} (dropped {dropped:,} with year>{max_year})")

    # Write with pyarrow to match cv_config.yaml's default engines
    out_df.to_parquet(out, engine="pyarrow", index=False)
    print(f"[{tag}] wrote {out} ({out.stat().st_size/1e6:.1f} MB)")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="overwrite existing outputs")
    ap.add_argument("--src", default=str(SRC))
    args = ap.parse_args()

    src = Path(args.src)
    if not src.exists():
        print(f"ERROR: source parquet missing: {src}", file=sys.stderr)
        return 2

    for tag, max_year in TARGETS.items():
        _build(tag, max_year, src, args.force)
    return 0


if __name__ == "__main__":
    sys.exit(main())
