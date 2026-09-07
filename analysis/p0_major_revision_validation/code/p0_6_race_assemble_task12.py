#!/usr/bin/env python3
"""Assemble and validate the parallel-race candidate for refinement task 12.

Writes a NON-CANONICAL candidate plus a manifest. Promotion to the canonical filename is a
separate, explicit step (`--mode promote`) that refuses to run while the fallback job is
still alive and refuses to mix rows across execution lineages.

Modes
-----
  assemble  build race/task12_parallel/dsnap_refine_shard__A__fold_7__direct__c0.RACE.csv
            + race_manifest.json, and run every structural / provenance / cross-execution check
  promote   copy the validated candidate to the canonical tables/ filename (guarded)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import p0_common as c
import p0_6_temporal_designs as td

CANON_SHARD = 12
RACE_DIR = c.P0_DIR / "race" / "task12_parallel"
CAND = RACE_DIR / "dsnap_refine_shard__A__fold_7__direct__c0.RACE.csv"
CANON = c.TABLES / "dsnap_refine_shard__A__fold_7__direct__c0.csv"
SIBLING = c.TABLES / "dsnap_refine_shard__A__fold_7__direct__c1.csv"
FALLBACK_LOG = c.P0_DIR / "logs" / "09_refine_22151066_12.out"
FALLBACK_JOB = "22151066_12"
ATOL = 1e-12

REQUIRED_METRICS = ["R2_price", "MAE_price", "MAPE", "RMSE_log", "median_ratio", "mean_ratio",
                    "weighted_mean_ratio", "COD", "COV", "PRD", "PRB", "MKI", "VEI",
                    "beta_log", "Cov_log_residual_log_price", "dCor_e_y", "Delta_NL",
                    "Delta_NL_raw"]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _spec():
    reg, block, family, ch, nch, rhos = td._refine_shards()[CANON_SHARD]
    if (reg, block, family, ch, nch) != ("A", "fold_7", "direct", 0, 2):
        raise c.ProtocolViolation("frozen refinement grid no longer maps task 12 as expected")
    return reg, block, family, ch, nch, [float(x) for x in rhos]


def _parse_fallback_log():
    """The still-running canonical execution prints one line per completed rho."""
    if not FALLBACK_LOG.exists():
        return {}
    pat = re.compile(r"^\[temporal\] dsnap/fold_7/direct rho=(\S+) beta=(\S+) R2=(\S+) ")
    out = {}
    for ln in FALLBACK_LOG.read_text().splitlines():
        m = pat.match(ln)
        if m:
            out[float(m.group(1))] = (float(m.group(2)), float(m.group(3)))
    return out


def mode_assemble() -> int:
    reg, block, family, ch, nch, rhos = _spec()

    # ---- 1. every rho row present, from ONE lineage (the race), none missing
    parts, missing = [], []
    for i in range(len(rhos)):
        p = RACE_DIR / f"rho_{i:02d}.csv"
        (parts.append((i, p)) if p.exists() else missing.append(p.name))
    if missing:
        raise c.ProtocolViolation(f"race incomplete, missing: {missing}")

    frames = []
    for i, p in parts:
        d = pd.read_csv(p)
        if len(d) != 1:
            raise c.ProtocolViolation(f"{p.name}: expected exactly 1 row, found {len(d)}")
        if not np.isclose(float(d.rho.iloc[0]), rhos[i], rtol=0, atol=ATOL):
            raise c.ProtocolViolation(
                f"{p.name}: rho {d.rho.iloc[0]} != frozen rhos[{i}] {rhos[i]}")
        frames.append(d)
    df = pd.concat(frames, ignore_index=True)          # frozen order = rhos[0::2] order

    checks: list[tuple[str, bool, str]] = []

    def chk(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    # ---- 2-4. row/rho identity
    chk("exactly_11_rows", len(df) == 11, f"{len(df)}")
    chk("exactly_11_unique_rho", df.rho.nunique() == 11, f"{df.rho.nunique()}")
    chk("no_duplicate_rho", not df.rho.duplicated().any())
    chk("rho_set_equals_frozen_task12",
        all(np.isclose(float(a), b, rtol=0, atol=ATOL) for a, b in zip(df.rho, rhos)))
    chk("rho_ascending_frozen_convention",
        list(df.rho) == sorted(df.rho), "region A chunk 0 is ascending by construction")

    # ---- 5-7. cell identity
    chk("design_is_dsnap", set(df.design) == {"dsnap"}, str(set(df.design)))
    chk("design_type_strict_date", set(df.design_type) == {"strict_date_robustness"})
    chk("block_is_fold_7", set(df.block) == {"fold_7"}, str(set(df.block)))
    chk("family_is_direct", set(df.family) == {"direct"}, str(set(df.family)))
    chk("grid_label_refinement", set(df.grid) == {"refinement"})

    # ---- 8-10. model / split / seed identity
    cfg = c.frozen_lgbm_config()
    chk("lgbm_params_sha256_matches_frozen",
        set(df.lgbm_params_sha256) == {cfg["lgbm_params_sha256"]},
        cfg["lgbm_params_sha256"][:16])
    chk("historical_execution_settings",
        set(df.execution_settings) == {"HISTORICAL (no determinism pins)"})
    chk("seed_is_frozen_random_state_2025", cfg["lgbm_params"].get("random_state") == 2025)
    chk("single_split_identity_n_train", df.n_train.nunique() == 1, str(set(df.n_train)))
    chk("single_split_identity_n_eval", df.n_eval.nunique() == 1, str(set(df.n_eval)))

    # split identity must equal the D-SNAP fold_7 protocol, not the primary design
    ba = pd.read_csv(c.TABLES / "dsnap_boundary_audit.csv")
    row = ba[ba.boundary == "fold_7"].iloc[0]
    chk("n_train_equals_dsnap_fold7_protocol",
        int(df.n_train.iloc[0]) == int(row.snap_train),
        f"{int(df.n_train.iloc[0])} vs {int(row.snap_train)}")
    chk("n_eval_equals_dsnap_fold7_protocol",
        int(df.n_eval.iloc[0]) == int(row.snap_val),
        f"{int(df.n_eval.iloc[0])} vs {int(row.snap_val)}")

    # ---- 11. schema identical to a canonical refinement shard
    if SIBLING.exists():
        want = list(pd.read_csv(SIBLING).columns)
        chk("schema_identical_to_canonical_sibling", list(df.columns) == want,
            f"{len(df.columns)} cols")
    else:
        chk("schema_identical_to_canonical_sibling", False, "sibling shard absent")

    # ---- 12. no missing required metric
    nan_cols = [m for m in REQUIRED_METRICS if m in df.columns and df[m].isna().any()]
    chk("no_nan_in_required_metrics", not nan_cols, str(nan_cols))
    chk("all_required_metrics_present",
        not [m for m in REQUIRED_METRICS if m not in df.columns],
        str([m for m in REQUIRED_METRICS if m not in df.columns]))
    chk("pred_sha256_present_and_unique",
        df.pred_sha256.notna().all() and df.pred_sha256.nunique() == 11)

    # ---- 13. cross-execution agreement with the ORIGINAL running job's logged rows
    fb = _parse_fallback_log()
    cross, worst_b, worst_r = [], 0.0, 0.0
    for _, r in df.iterrows():
        key = next((k for k in fb if np.isclose(k, float(r.rho), rtol=0, atol=1e-9)), None)
        if key is None:
            continue
        lb, lr = fb[key]
        db, dr = abs(round(float(r.beta_log), 5) - lb), abs(round(float(r.R2_price), 5) - lr)
        worst_b, worst_r = max(worst_b, db), max(worst_r, dr)
        cross.append({"rho": float(r.rho), "beta_log_race": float(r.beta_log),
                      "beta_log_fallback_logged": lb, "abs_diff_beta_at_log_precision": db,
                      "R2_price_race": float(r.R2_price), "R2_price_fallback_logged": lr,
                      "abs_diff_R2_at_log_precision": dr})
    # the fallback log prints %+.5f / %.5f, so agreement is exact at 5dp or it is a real
    # disagreement -- a half-ulp rounding straddle can only ever cost 1 unit in the 5th place
    chk("cross_execution_beta_log_agrees", worst_b <= 1e-5, f"worst {worst_b:.2e}")
    chk("cross_execution_R2_price_agrees", worst_r <= 1e-5, f"worst {worst_r:.2e}")
    chk("cross_execution_rows_compared", len(cross) > 0, f"{len(cross)} of 11 available")

    # ---- 14. lineage purity
    chk("candidate_is_single_lineage_race", True,
        "every row read from race/task12_parallel/rho_NN.csv; no canonical row spliced")
    chk("canonical_filename_not_written_by_race", not CANON.exists(),
        "canonical shard absent while the race candidate is built")

    c.write_table(df, CAND)

    man = {
        "event": "redundant_parallel_re_execution",
        "canonical_array_task": CANON_SHARD,
        "cell": {"design": "dsnap", "region": reg, "block": block, "family": family,
                 "chunk": f"{ch + 1}/{nch}", "n_rho": len(rhos)},
        "rhos_frozen_order": rhos,
        "fallback_job": FALLBACK_JOB,
        "race_job": "22166048",
        "race_topology": "one rho per Slurm array element (11 elements)",
        "scientific_settings_changed": "none",
        "scheduler_only_changes": [
            "--mem 110G -> 16G (measured MaxRSS of every completed refinement task 2.4-2.7 GB)",
            "--exclude=node2621 (the oversubscribed node the canonical task is stuck on)",
            "--array=0-10 (execution granularity)",
            "NUMBA_CACHE_DIR isolated per task (cache location only)"],
        "lgbm_params_sha256": cfg["lgbm_params_sha256"],
        "n_train": int(df.n_train.iloc[0]), "n_eval": int(df.n_eval.iloc[0]),
        "candidate_file": CAND.name, "candidate_sha256": _sha(CAND),
        "per_rho_files": {f"rho_{i:02d}.csv": _sha(RACE_DIR / f"rho_{i:02d}.csv")
                          for i in range(len(rhos))},
        "cross_execution_check": cross,
        "checks": [{"check": n, "passed": p, "detail": d} for n, p, d in checks],
        "all_checks_passed": all(p for _, p, _ in checks),
    }
    c.write_json(RACE_DIR / "race_manifest.json", man)

    print(f"\n=== RACE CANDIDATE VALIDATION ({len(checks)} checks) ===")
    for n, p, d in checks:
        print(f"  {'PASS' if p else 'FAIL'}  {n}" + (f"  [{d}]" if d else ""))
    print(f"\ncross-execution rows compared against the fallback log: {len(cross)}")
    for x in cross:
        print(f"  rho={x['rho']:<22.16g} beta {x['beta_log_race']:+.5f} vs "
              f"{x['beta_log_fallback_logged']:+.5f} | R2 {x['R2_price_race']:.5f} vs "
              f"{x['R2_price_fallback_logged']:.5f}")
    ok = all(p for _, p, _ in checks)
    print(f"\n{'RACE_CANDIDATE_VALIDATED' if ok else 'RACE_CANDIDATE_REJECTED'}")
    print(f"candidate: {CAND}")
    print(f"sha256:    {man['candidate_sha256']}")
    return 0 if ok else 1


def mode_promote() -> int:
    """Copy the validated candidate to the canonical filename. Guarded."""
    man = json.loads((RACE_DIR / "race_manifest.json").read_text())
    if not man["all_checks_passed"]:
        raise c.ProtocolViolation("candidate did not pass validation; refusing to promote")
    if _sha(CAND) != man["candidate_sha256"]:
        raise c.ProtocolViolation("candidate changed since validation; refusing to promote")
    alive = subprocess.run(["squeue", "-j", FALLBACK_JOB, "-h", "-o", "%t"],
                           capture_output=True, text=True).stdout.strip()
    if alive:
        raise c.ProtocolViolation(
            f"fallback {FALLBACK_JOB} is still in state {alive!r}; cancel it before promoting "
            "so the two executions can never both write the canonical filename")
    if CANON.exists():
        raise c.ProtocolViolation(
            f"{CANON.name} already exists -- the canonical execution won; do not overwrite it")
    df = pd.read_csv(CAND)
    if len(df) != 11:
        raise c.ProtocolViolation(f"candidate has {len(df)} rows")
    # BYTE copy, not a re-serialization. Round-tripping through pd.read_csv/to_csv can shift
    # the last digit of a float by 1 ULP (it did: two COD values on the first promotion), and
    # the promoted artifact must be bit-identical to the candidate the manifest hashed.
    c._assert_write_allowed(CANON)
    CANON.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(CAND, CANON)
    cand_pq, canon_pq = CAND.with_suffix(".parquet"), CANON.with_suffix(".parquet")
    if cand_pq.exists():
        shutil.copyfile(cand_pq, canon_pq)
    if _sha(CANON) != man["candidate_sha256"]:
        raise c.ProtocolViolation("promoted file does not hash to the validated candidate")
    print(f"[promote] {CAND.name} -> {CANON.name} (byte copy)")
    print(f"[promote] canonical sha256 = {_sha(CANON)}  == manifest candidate_sha256")
    print("[promote] race candidate and manifest preserved for provenance")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["assemble", "promote"])
    a = ap.parse_args()
    return mode_assemble() if a.mode == "assemble" else mode_promote()


if __name__ == "__main__":
    raise SystemExit(main())
