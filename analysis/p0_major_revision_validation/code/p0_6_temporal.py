#!/usr/bin/env python3
"""P0-6a -- temporal exposure audit ONLY (Stage 1).

NO model fits.  D-SNAP / D-PURGE / D-UNSEEN fitting is NOT authorized in Stage 1.

Reproduces, from the canonical loader and the exact archived protocol:
  * same-date crossing at all 8 chronological boundaries
      (7 rolling-origin fold train/val cuts + the development/held-out cut),
      with train-side and validation/held-out-side counts on the boundary date;
  * repeat-PIN exposure for each fold, dev <-> held-out, and production <-> 2025.

Emits:
  tables/boundary_exposure_audit.csv
  tables/repeat_pin_exposure.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as c


def _load_pins() -> pd.DataFrame:
    """meta_pin is NOT loaded by the pipeline (run_temporal_cv.py:741), so read it
    separately -- using the SAME pyarrow row-group pushdown as the canonical loader,
    then the same eligibility ordering.  We align by position against the canonical
    frames using (meta_sale_date, meta_sale_price) plus row order, and verify."""
    import pyarrow.parquet as pq
    filters = [("ind_pin_is_multicard", "==", False), ("sv_is_outlier", "==", False)]
    tbl = pq.read_table(
        str(c.DATA_PATH),
        columns=[c.PIN_COL, c.DATE_COL, c.TARGET_COL],
        filters=filters,
    )
    df = tbl.to_pandas()
    df[c.DATE_COL] = pd.to_datetime(df[c.DATE_COL])
    return df


def run() -> int:
    print("[S3] loading canonical splits ...", flush=True)
    df_tv, df_test, df_assess, _p, _c = c.load_canonical_splits()
    folds, archive = c.rebuild_folds(df_tv)

    # ------------------------------------------------------------------ PINs
    raw = _load_pins()
    print(f"[S3] pushdown-filtered parquet rows for PIN join: {len(raw)}", flush=True)

    def attach_pin(frame: pd.DataFrame, label: str) -> pd.Series:
        """Attach meta_pin by exact (date, price) multiset order.

        The canonical loader sorts by meta_sale_date with kind='mergesort', so
        within-date order is the original parquet row order.  Reproducing that
        ordering on the pushdown-filtered frame gives a positional alignment we
        then verify on date and price."""
        ordered = raw.sort_values([c.DATE_COL], kind="mergesort").reset_index(drop=True)
        return ordered

    ordered_raw = raw.sort_values([c.DATE_COL], kind="mergesort").reset_index(drop=True)

    # Universe = 2016-01-01 .. 2024-12-31 ; assessment = calendar 2025
    d = ordered_raw[c.DATE_COL]
    universe = ordered_raw.loc[(d >= "2016-01-01") & (d <= "2024-12-31")].reset_index(drop=True)
    assess = ordered_raw.loc[d.dt.year == 2025].reset_index(drop=True)
    split_idx = int(0.9 * len(universe))
    uni_dev = universe.iloc[:split_idx].reset_index(drop=True)
    uni_test = universe.iloc[split_idx:].reset_index(drop=True)

    checks = {
        "universe_rows": len(universe), "dev_rows": len(uni_dev),
        "test_rows": len(uni_test), "assess_rows": len(assess),
    }
    if (len(uni_dev), len(uni_test), len(assess)) != (c.N_DEVELOPMENT, c.N_HELDOUT, c.N_2025):
        raise c.ProtocolViolation(f"PIN-side split reconstruction mismatch: {checks}")
    # verify the positional alignment against the canonical frames
    for name, a, b in (
        ("development", uni_dev, df_tv),
        ("heldout", uni_test, df_test),
        ("forward_2025", assess, df_assess),
    ):
        if not np.array_equal(a[c.DATE_COL].to_numpy(), pd.to_datetime(b[c.DATE_COL]).to_numpy()):
            raise c.ProtocolViolation(f"{name}: sale-date alignment failed for the PIN join")
        if not np.allclose(a[c.TARGET_COL].to_numpy(dtype=float),
                           b[c.TARGET_COL].to_numpy(dtype=float), rtol=0, atol=0):
            raise c.ProtocolViolation(f"{name}: sale-price alignment failed for the PIN join")
    print("[S3] PIN alignment verified on date and price for all three splits", flush=True)

    # meta_pin is object dtype (14-char zero-padded strings).  Factorize ONCE over the
    # full eligible universe so every downstream set operation is an int64 hash lookup
    # instead of numpy's O(n*m) object-array fallback.  Purely a performance change:
    # factorize is a bijection on the observed PIN values, so membership is identical.
    all_pins = pd.concat(
        [uni_dev[c.PIN_COL], uni_test[c.PIN_COL], assess[c.PIN_COL]], ignore_index=True
    )
    codes, uniques = pd.factorize(all_pins, sort=False)
    n_dev, n_test_, n_as_ = len(uni_dev), len(uni_test), len(assess)
    pin_dev = codes[:n_dev]
    pin_test = codes[n_dev:n_dev + n_test_]
    pin_assess = codes[n_dev + n_test_:]
    pin_prod = codes[:n_dev + n_test_]
    print(f"[S3] PIN factorization: {len(uniques)} distinct PINs over {len(all_pins)} eligible rows",
          flush=True)

    dates_dev = pd.to_datetime(df_tv[c.DATE_COL]).to_numpy()
    dates_test = pd.to_datetime(df_test[c.DATE_COL]).to_numpy()

    # --------------------------------------------------- same-date boundaries
    brows = []
    for rec in folds:
        tr = np.asarray(rec["train_indices"], dtype=int)
        va = np.asarray(rec["val_indices"], dtype=int)
        d_tr, d_va = dates_dev[tr], dates_dev[va]
        b_date = d_va.min()                      # first validation date
        brows.append({
            "boundary_id": f"fold_{int(rec['fold_id'])+1}",
            "boundary_kind": "cv_fold_train_val",
            "boundary_date": pd.Timestamp(b_date).date().isoformat(),
            "train_max_date": pd.Timestamp(d_tr.max()).date().isoformat(),
            "val_min_date": pd.Timestamp(d_va.min()).date().isoformat(),
            "same_date_crossing": bool(pd.Timestamp(d_tr.max()) == pd.Timestamp(d_va.min())),
            "n_on_boundary_date_train_side": int(np.sum(d_tr == b_date)),
            "n_on_boundary_date_val_side": int(np.sum(d_va == b_date)),
            "n_train": int(tr.size), "n_val": int(va.size),
            "share_of_val_block_on_boundary_date": float(np.sum(d_va == b_date) / va.size),
        })
    b_date = dates_test.min()
    brows.append({
        "boundary_id": "development_heldout",
        "boundary_kind": "dev_heldout",
        "boundary_date": pd.Timestamp(b_date).date().isoformat(),
        "train_max_date": pd.Timestamp(dates_dev.max()).date().isoformat(),
        "val_min_date": pd.Timestamp(dates_test.min()).date().isoformat(),
        "same_date_crossing": bool(pd.Timestamp(dates_dev.max()) == pd.Timestamp(dates_test.min())),
        "n_on_boundary_date_train_side": int(np.sum(dates_dev == b_date)),
        "n_on_boundary_date_val_side": int(np.sum(dates_test == b_date)),
        "n_train": int(dates_dev.size), "n_val": int(dates_test.size),
        "share_of_val_block_on_boundary_date": float(np.sum(dates_test == b_date) / dates_test.size),
    })
    d_as = pd.to_datetime(df_assess[c.DATE_COL]).to_numpy()
    brows.append({
        "boundary_id": "production_2025",
        "boundary_kind": "forward_year",
        "boundary_date": "2025-01-01",
        "train_max_date": pd.Timestamp(dates_test.max()).date().isoformat(),
        "val_min_date": pd.Timestamp(d_as.min()).date().isoformat(),
        "same_date_crossing": False,
        "n_on_boundary_date_train_side": 0, "n_on_boundary_date_val_side": 0,
        "n_train": int(dates_dev.size + dates_test.size), "n_val": int(d_as.size),
        "share_of_val_block_on_boundary_date": 0.0,
    })
    df_b = pd.DataFrame(brows)
    c.write_table(df_b, c.TABLES / "boundary_exposure_audit.csv")

    # ------------------------------------------------------- repeat-PIN exposure
    prows = []
    for rec in folds:
        tr = np.asarray(rec["train_indices"], dtype=int)
        va = np.asarray(rec["val_indices"], dtype=int)
        p_tr, p_va = pin_dev[tr], pin_dev[va]
        uniq_va = np.unique(p_va)
        tr_set = np.unique(p_tr)
        seen = np.isin(uniq_va, tr_set, assume_unique=True)
        rows_seen = np.isin(p_va, tr_set)
        prows.append({
            "boundary_id": f"fold_{int(rec['fold_id'])+1}",
            "boundary_kind": "cv_fold_train_val",
            "n_train_rows": int(tr.size), "n_val_rows": int(va.size),
            "n_unique_train_pins": int(tr_set.size),
            "n_unique_val_pins": int(uniq_va.size),
            "n_val_pins_also_in_train": int(seen.sum()),
            "share_val_pins_also_in_train": float(seen.mean()),
            "n_val_rows_with_seen_pin": int(rows_seen.sum()),
            "share_val_rows_with_seen_pin": float(rows_seen.mean()),
        })
    for label, p_tr, p_va, n_tr, n_va in (
        ("development_heldout", pin_dev, pin_test, pin_dev.size, pin_test.size),
        ("production_2025", pin_prod, pin_assess, pin_prod.size, pin_assess.size),
    ):
        uniq_va = np.unique(p_va)
        tr_set = np.unique(p_tr)
        seen = np.isin(uniq_va, tr_set, assume_unique=True)
        rows_seen = np.isin(p_va, tr_set)
        prows.append({
            "boundary_id": label,
            "boundary_kind": "out_of_time",
            "n_train_rows": int(n_tr), "n_val_rows": int(n_va),
            "n_unique_train_pins": int(tr_set.size),
            "n_unique_val_pins": int(uniq_va.size),
            "n_val_pins_also_in_train": int(seen.sum()),
            "share_val_pins_also_in_train": float(seen.mean()),
            "n_val_rows_with_seen_pin": int(rows_seen.sum()),
            "share_val_rows_with_seen_pin": float(rows_seen.mean()),
        })
    df_p = pd.DataFrame(prows)
    c.write_table(df_p, c.TABLES / "repeat_pin_exposure.csv")

    print("\n=== same-date boundaries ===")
    print(df_b[["boundary_id", "boundary_date", "same_date_crossing",
                "n_on_boundary_date_train_side", "n_on_boundary_date_val_side"]].to_string(index=False))
    print("\n=== repeat-PIN exposure ===")
    print(df_p[["boundary_id", "n_val_pins_also_in_train", "n_unique_val_pins",
                "share_val_pins_also_in_train"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
