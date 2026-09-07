#!/usr/bin/env python3
"""Shared helpers for the P1 inferential-reporting stage.

Binding rules
-------------
* P0 is FROZEN at tag ``p0-major-revision-final-20260907`` (commit 805c426e).
  Nothing under ``analysis/p0_major_revision_validation/`` is ever written here;
  it is imported and read only.  ``p0_common`` itself hard-refuses writes outside
  the P0 tree, so P1 defines its own writers with its own allowed roots.
* Canonical machinery is IMPORTED, never re-implemented -- in particular the D3
  appearance weights (``p0_5_beta_coordinates.appearance_weights``) and the IAAO
  metric functions in ``utils/motivation_utils``.
* Protected paths are never written: paper/, utils/, soft_constrained_models/,
  scripts/, run_temporal_cv.py, output/paper_v6_preselection_994/,
  output/paper_v12_*, analysis/p0_major_revision_validation/.
* Zero refits.  Every input is a cached prediction artifact resolved through the
  frozen P0 config maps.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
P1_DIR = Path(__file__).resolve().parents[1]        # analysis/p1_inferential_reporting
REPO = P1_DIR.parents[1]                            # repository root
P0_DIR = REPO / "analysis" / "p0_major_revision_validation"

# p0_common inserts REPO and REPO/scripts into sys.path itself.
sys.path.insert(0, str(P0_DIR / "code"))
sys.path.insert(0, str(REPO))

import p0_common as p0                                              # noqa: E402
from p0_common import (                                             # noqa: E402,F401
    ProtocolViolation,
    sha256_file, sha256_obj,
    git_state, package_versions, thread_env,
    load_canonical_splits, load_archived_folds, rebuild_folds,
    frozen_lgbm_config, frozen_rho_grid, fitting_blocks,
    DISPLAY_ANCHORS, CELL_A_NAME, CELL_B_NAME, CELL_C_NAME,
    N_DEVELOPMENT, N_HELDOUT, N_2025, N_PRODUCTION,
    TARGET_COL, DATE_COL, PIN_COL, V6, V12,
)

TABLES = P1_DIR / "tables"
REPORTS = P1_DIR / "reports"
CONFIGS = P1_DIR / "configs"
PROVENANCE = P1_DIR / "provenance"
LOGS = P1_DIR / "logs"
FIGURES = P1_DIR / "figures"
P1_OUTPUT_ROOT = REPO / "output" / "p1_inferential_reporting"

# Frozen P0 inputs (read-only).
P0_TABLES = P0_DIR / "tables"
P0_CONFIGS = P0_DIR / "configs"
MATCHED_BETA_FROZEN = P0_CONFIGS / "matched_beta_frozen.json"
FROZEN_CONFIG_MAP = P0_CONFIGS / "frozen_config_map.csv"      # family x rho x {heldout,forward_2025}
FROZEN_CV_RUN_MAP = P0_CONFIGS / "frozen_cv_run_map.csv"      # family x rho x fold
POST_G1_CONVENTION = P0_CONFIGS / "post_g1_reference_convention.yaml"

P0_TAG = "p0-major-revision-final-20260907"
P0_COMMIT = "805c426e1587972a2a07dcaf60220603397c0d3e"

ED2_STATUS = "May-2026 Exposure Draft / proposed guidance; not adopted IAAO guidance."
ED2_ADOPTED_REFERENCE = "IAAO Standard on Ratio Studies (2013) remains the adopted/current guidance."

# The frozen D3 overlap structure (development pool).  Recorded unambiguously:
# 20,988 duplicated UNIQUE rows contribute 41,976 APPEARANCES.
D3_N_APPEARANCES = 151_153
D3_N_UNIQUE = 130_165
D3_UNIQUE_ROWS_MULT_1 = 109_177
D3_DUPLICATED_UNIQUE_ROWS = 20_988          # unique rows appearing twice
D3_DUPLICATED_APPEARANCES = 41_976          # = 2 * D3_DUPLICATED_UNIQUE_ROWS
D3_MAX_MULTIPLICITY = 2


def assert_d3_multiplicity_identities() -> None:
    """The bookkeeping trap: the P0 artifact's m_i_distribution is valued in
    APPEARANCES ({"1": 109177, "2": 41976}), and its n_duplicated_appearances
    (20988) is the EXCESS count n_appearances - n_unique, which coincides with
    the duplicated-unique-row count only because the maximum multiplicity is 2.
    """
    if D3_UNIQUE_ROWS_MULT_1 + D3_DUPLICATED_UNIQUE_ROWS != D3_N_UNIQUE:
        raise ProtocolViolation(
            f"unique-row identity fails: {D3_UNIQUE_ROWS_MULT_1} + {D3_DUPLICATED_UNIQUE_ROWS} "
            f"!= {D3_N_UNIQUE}")
    if D3_UNIQUE_ROWS_MULT_1 + 2 * D3_DUPLICATED_UNIQUE_ROWS != D3_N_APPEARANCES:
        raise ProtocolViolation(
            f"appearance identity fails: {D3_UNIQUE_ROWS_MULT_1} + 2*{D3_DUPLICATED_UNIQUE_ROWS} "
            f"!= {D3_N_APPEARANCES}")
    if D3_DUPLICATED_APPEARANCES != 2 * D3_DUPLICATED_UNIQUE_ROWS:
        raise ProtocolViolation("duplicated_appearances != 2 * duplicated_unique_rows")
    if D3_N_APPEARANCES - D3_N_UNIQUE != D3_DUPLICATED_UNIQUE_ROWS:
        raise ProtocolViolation("excess-appearance identity fails")


# --------------------------------------------------------------------------
# Write isolation
# --------------------------------------------------------------------------
_ALLOWED_WRITE_ROOTS = (P1_DIR.resolve(), P1_OUTPUT_ROOT.resolve())

_PROTECTED = (
    REPO / "paper", REPO / "utils", REPO / "soft_constrained_models", REPO / "scripts",
    REPO / "run_temporal_cv.py", REPO / "output" / "paper_v6_preselection_994",
    REPO / "output" / "paper_v12_lower_rho_extension_994_v2", P0_DIR,
)


def _assert_write_allowed(path: Path) -> None:
    rp = path.resolve()
    for prot in _PROTECTED:
        if str(rp) == str(prot.resolve()) or str(rp).startswith(str(prot.resolve()) + "/"):
            raise ProtocolViolation(f"Refusing to write to a PROTECTED path: {rp}")
    if not any(str(rp).startswith(str(root) + "/") or str(rp) == str(root)
               for root in _ALLOWED_WRITE_ROOTS):
        raise ProtocolViolation(f"Refusing to write outside the approved P1 locations: {rp}")


def write_json(path: Path, payload: Any) -> Path:
    _assert_write_allowed(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n")
    tmp.replace(path)
    return path


def write_table(df: pd.DataFrame, path: Path) -> Path:
    """csv + parquet twin (house convention, matching p0_common.write_table)."""
    _assert_write_allowed(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".csv.tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)
    try:
        df.to_parquet(path.with_suffix(".parquet"), index=False)
    except Exception:
        pass
    return path


def write_text(path: Path, text: str) -> Path:
    _assert_write_allowed(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)
    return path


# --------------------------------------------------------------------------
# D3 appearance weights -- imported, never re-implemented
# --------------------------------------------------------------------------
def appearance_weights(fold_rids: List[np.ndarray]):
    """Delegate to the P0 D3 builder: w_ik = 1/m_i, one sale row = total weight 1."""
    sys.path.insert(0, str(P0_DIR / "code"))
    import p0_5_beta_coordinates as bc
    return bc.appearance_weights(fold_rids)


# --------------------------------------------------------------------------
# Frozen P0 inputs
# --------------------------------------------------------------------------
def matched_beta_frozen() -> dict:
    return json.loads(MATCHED_BETA_FROZEN.read_text())


def ed2_procedure() -> dict:
    return json.loads((CONFIGS / "ed2_vei_procedure.json").read_text())


def ed2_source_manifest() -> dict:
    p = PROVENANCE / "ed2_source_manifest.json"
    if not p.exists():
        raise ProtocolViolation(
            "ED2 source manifest missing -- Task 3 may not run and the evidence freeze is blocked.")
    m = json.loads(p.read_text())
    if not m.get("verified"):
        raise ProtocolViolation("ED2 source manifest is not marked verified.")
    return m


def ed2_source_verified() -> bool:
    try:
        ed2_source_manifest()
        return True
    except Exception:
        return False


_PREFLIGHT_CACHE: Dict[str, Any] = {}


def preflight_block() -> Dict[str, Any]:
    """Provenance block.  Memoised: git_state() runs `git status --porcelain` over a
    repository with a very large untracked output/ tree on a network filesystem and
    costs minutes; it is invariant within a single process run."""
    if not _PREFLIGHT_CACHE:
        _PREFLIGHT_CACHE.update({
            "git": git_state(),
            "versions": package_versions(),
            "thread_env": thread_env(),
            "cwd": str(Path.cwd()),
            "p0_tag": P0_TAG,
            "p0_commit": P0_COMMIT,
        })
    return dict(_PREFLIGHT_CACHE)


# --------------------------------------------------------------------------
# Observation-level loading (shared by Tasks 2, 3, 4)
# --------------------------------------------------------------------------
FOLD_EVALS = [f"fold_{k}" for k in range(1, 8)]
OOS_EVALS = ["heldout", "forward_2025"]
ALL_EVALS = FOLD_EVALS + ["pooled_oof"] + OOS_EVALS


def _cs():
    sys.path.insert(0, str(P0_DIR / "code"))
    import p0_4_centered_spread as cs
    return cs


_ZREF_CACHE: Dict[tuple, tuple] = {}


def _load_eval_cached(ref: str, block_id: str):
    """p0_4_centered_spread.load_eval with its hash verification, memoised.

    The underlying loader re-reads and re-hashes the parquet on every call; the P1
    display set touches the same 18 (cell, block) pairs many times.  Caching changes
    nothing about what is verified -- the hash check still runs once per pair.
    """
    key = (ref, block_id)
    if key not in _ZREF_CACHE:
        cs = _cs()
        _ZREF_CACHE[key] = cs.load_eval(ref, block_id)
    return _ZREF_CACHE[key]


def load_observations(entry: dict, evaluation: str) -> pd.DataFrame:
    """Observation-level (row_id, y_true_log, y_pred_log) for one display entry.

    Zero refits: fitted entries read the cached prediction parquet; post-hoc entries
    apply the frozen centered map f_b = ybar_T + b*(f0 - ybar_T) to the cached
    Stage-1.5 zero-reference predictions.  'pooled_oof' concatenates the seven fold
    validation blocks -- which are NOT disjoint (fold_6 and fold_7 share 20,988 unique
    rows), so it must be used with D3 row-balanced weights, never as an IID sample.
    """
    if entry.get("kind") == "NOT_ATTAINED":
        raise ProtocolViolation("NOT_ATTAINED entry has no observations by construction")

    if evaluation == "pooled_oof":
        parts = []
        for k in range(1, 8):
            d = load_observations(entry, f"fold_{k}").copy()
            d["fold"] = k
            parts.append(d)
        return pd.concat(parts, ignore_index=True)

    if entry["kind"] == "fitted":
        path = entry["artifacts"][evaluation]
        d = pd.read_parquet(REPO / path, columns=["row_id", "y_true_log", "y_pred_log"])
        return d.sort_values("row_id").reset_index(drop=True)

    if entry["kind"] == "posthoc":
        cs = _cs()
        ref = entry["posthoc_ref_cell"]
        blk = cs.BLOCK_FOR[evaluation]
        d, ybar_T, _ = _load_eval_cached(ref, blk)
        out = d[["row_id", "y_true_log"]].copy()
        out["y_pred_log"] = cs.centered_map(d.y_pred_log.to_numpy(), ybar_T, float(entry["b"]))
        return out.reset_index(drop=True)

    raise ProtocolViolation(f"unknown entry kind: {entry.get('kind')}")


def d3_weights_for_pooled(row_ids: np.ndarray) -> np.ndarray:
    """D3 row-balanced weights w_ik = 1/m_i over a concatenated pooled-OOF sample.

    Delegates the multiplicity logic to the same rule that defines D3 in P0:
    each UNIQUE sale row carries total weight exactly one.
    """
    uniq, cnt = np.unique(row_ids, return_counts=True)
    m = dict(zip(uniq.tolist(), cnt.tolist()))
    w = np.asarray([1.0 / m[int(r)] for r in row_ids], dtype=float)
    tot = pd.Series(w).groupby(pd.Series(row_ids)).sum()
    dev = float(np.max(np.abs(tot.to_numpy() - 1.0)))
    if dev > 1e-12:
        raise ProtocolViolation(f"D3 weights do not sum to one per unique row (max dev {dev:.2e})")
    return w


def display_set() -> dict:
    """The frozen display set, hash-verified against its freeze-time hash."""
    p = CONFIGS / "display_set_frozen.json"
    h = json.loads((CONFIGS / "display_set_frozen_hash.json").read_text())
    if h["file_sha256"] != sha256_file(p):
        raise ProtocolViolation("display_set_frozen.json changed after being hashed")
    return json.loads(p.read_text())
