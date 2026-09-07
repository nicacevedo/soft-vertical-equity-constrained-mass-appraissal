#!/usr/bin/env python3
"""Redundant PARALLEL re-execution of G5a refinement array task 12 — one rho per Slurm task.

WHY THIS EXISTS
---------------
`22151066_12` (region A / fold_7 / Direct / chunk 1-of-2, 11 rho) is healthy but was
co-scheduled onto an oversubscribed node and is running ~15x slower per fit than the
identical cells elsewhere. Its 11 rho fits are scientifically INDEPENDENT — `_run_cell`
constructs a fresh estimator for every rho and never warm-starts — so the same 11 fits can
be executed concurrently as 11 Slurm array elements.

WHAT THIS CHANGES
-----------------
Execution granularity and the OUTPUT PATH. Nothing else.

This wrapper does not reimplement any mathematics. It calls the canonical
`p0_6_temporal_designs._load_block` and `._run_cell` — the same functions the canonical
runner calls — with the same design, block, family, rho, split protocol, LightGBM
parameter vector, seed, feature columns, filters, objective, metric code and thread
settings. The ONLY deviation is that `c.TABLES` (the directory `c.write_table` resolves
against) is repointed at an isolated race workspace, so a race task can never write the
canonical task-12 filename while the original job is still running.

The canonical shard filename is written only later, by an explicit human-audited promotion
step, and only from ONE complete execution lineage.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as c
import p0_6_temporal_designs as td

CANON_SHARD = 12
RACE_DIR = c.P0_DIR / "race" / "task12_parallel"
EXPECT = ("A", "fold_7", "direct", 0, 2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho-index", type=int, required=True)
    a = ap.parse_args()

    reg, block, family, ch, nch, rhos = td._refine_shards()[CANON_SHARD]
    if (reg, block, family, ch, nch) != EXPECT:
        raise c.ProtocolViolation(
            f"refine shard {CANON_SHARD} is {(reg, block, family, ch, nch)}, expected {EXPECT}; "
            "the frozen refinement grid changed and this race is no longer valid")
    if len(rhos) != 11:
        raise c.ProtocolViolation(f"task 12 must hold 11 rho, found {len(rhos)}")
    if not 0 <= a.rho_index < 11:
        raise SystemExit(f"--rho-index must be in [0, 11), got {a.rho_index}")

    rho = float(rhos[a.rho_index])
    print(f"[race] canonical task {CANON_SHARD} -> region={reg} block={block} "
          f"family={family} chunk={ch + 1}/{nch}", flush=True)
    print(f"[race] rho_index={a.rho_index} rho={rho!r}", flush=True)
    print(f"[race] env={c.thread_env()}", flush=True)

    tr_df, ev_df, ev_rid, params, cfg_hash, pred_cols, cat_cols = td._load_block("dsnap", block)

    # ---- OUTPUT PATH ISOLATION ONLY -------------------------------------------------
    # c.TABLES is the directory `c.write_table` resolves relative paths against inside
    # `_run_cell`. Repointing it keeps every race artifact out of tables/ so the still
    # running canonical job owns its filename uncontested. No scientific input changes.
    RACE_DIR.mkdir(parents=True, exist_ok=True)
    c.TABLES = RACE_DIR
    # ---------------------------------------------------------------------------------

    return td._run_cell(
        "dsnap", block, family, tr_df, ev_df, ev_rid, params, cfg_hash,
        {"positive_rhos": [rho]}, pred_cols, cat_cols,
        grid_label="refinement", include_zero=False,
        out_name=f"rho_{a.rho_index:02d}")


if __name__ == "__main__":
    raise SystemExit(main())
