#!/usr/bin/env python3
"""Merge the 9 per-block zero-reference fit tables into zero_reference_fits.csv."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as c

BLOCK_ORDER = [f"fold_{k}_train" for k in range(1, 8)] + \
              ["development_pool", "production_2016_2024"]

def main() -> int:
    parts = sorted(c.TABLES.glob("zero_reference_fits_block*.csv"))
    if len(parts) != 9:
        raise c.ProtocolViolation(
            f"expected 9 per-block tables, found {len(parts)}: "
            f"{[p.name for p in parts]}")
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    if len(df) != 27:
        raise c.ProtocolViolation(f"expected 27 fits, got {len(df)}")
    if set(df.block_id) != set(BLOCK_ORDER):
        raise c.ProtocolViolation(f"block set mismatch: {sorted(set(df.block_id))}")
    for cid in ("A", "B", "C"):
        n = int((df.cell_id == cid).sum())
        if n != 9:
            raise c.ProtocolViolation(f"cell {cid} has {n} blocks, expected 9")
    df["_o"] = df.block_id.map({b: i for i, b in enumerate(BLOCK_ORDER)})
    df = df.sort_values(["cell_id", "_o"]).drop(columns=["_o"]).reset_index(drop=True)
    c.write_table(df, c.TABLES / "zero_reference_fits.csv")
    print(f"merged {len(parts)} blocks -> zero_reference_fits.csv ({len(df)} fits)")
    print(df.groupby("cell_id").size().to_string())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
