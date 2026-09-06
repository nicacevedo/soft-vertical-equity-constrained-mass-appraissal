#!/usr/bin/env python3
"""Shared Stage-1 helpers for the P0 major-revision validation.

Binding rules (approved plan rev. 3):
  * The canonical loader/split machinery is IMPORTED, never re-implemented.
    In particular the pyarrow row-group pushdown in
    ``run_temporal_cv._load_and_split_data`` is part of the frozen experiment
    definition (plan C-11) and must not be replaced with pandas-side filtering.
  * Canonical sample counts 344,607 / 38,290 / 26,641 are asserted.
  * Archived fold index hashes are verified before any diagnostic is emitted.
  * Nothing outside analysis/p0_major_revision_validation/ and
    output/p0_major_revision_validation/ is written.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
P0_DIR = Path(__file__).resolve().parents[1]          # analysis/p0_major_revision_validation
REPO = P0_DIR.parents[1]                              # repository root
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

TABLES = P0_DIR / "tables"
REPORTS = P0_DIR / "reports"
CONFIGS = P0_DIR / "configs"
PROVENANCE = P0_DIR / "provenance"
LOGS = P0_DIR / "logs"
FIGURES = P0_DIR / "figures"
P0_OUTPUT_ROOT = REPO / "output" / "p0_major_revision_validation"

DATA_PATH = REPO / "data" / "CCAO" / "2025" / "training_data.parquet"
V6 = REPO / "output" / "paper_v6_preselection_994"
V12 = REPO / "output" / "paper_v12_lower_rho_extension_994_v2"

FROZEN_LGBM_CONFIG = V6 / "lgbm_config.json"
ARCHIVED_FOLDS = V6 / "protocol" / "data_id=d4929d43ec19badf" / "split_id=3d464d4a611b131b" / "folds.json"

TARGET_COL = "meta_sale_price"
DATE_COL = "meta_sale_date"
PIN_COL = "meta_pin"

# Canonical, non-negotiable sample counts (plan A.6 / C-11).
N_DEVELOPMENT = 344_607
N_HELDOUT = 38_290
N_2025 = 26_641
N_PRODUCTION = N_DEVELOPMENT + N_HELDOUT      # 382,897

# Fixed cell names (plan A8; never abbreviate, never substitute).
CELL_A_NAME = "Ordinary LightGBM (standard raw-label native)"
CELL_B_NAME = "Parity-aligned native L2"
CELL_C_NAME = "Custom rho=0 origin"

# Frozen config identifiers.
NATIVE_CONFIG_ID = "252a25d9c0ce796b"
LINEAR_CONFIG_ID = "fd63507d2456c789"
DIRECT_RHO0_ID = "1fb838f7d6bfda88"
SURR_RHO0_ID = "5b7875e55e58ac62"
EXPECTED_LGBM_PARAMS_SHA256 = "8f0f2acd83118de782604b5ca7143acfbd2af3fd186ea9376588f9bcf560585b"

# Display anchors on the frozen 82-point positive grid (plan A.4).
DISPLAY_ANCHORS = [0.0104811313415468, 0.1, 0.954095476349994, 10.481131341546853, 100.0]


class ProtocolViolation(RuntimeError):
    """Raised when repository reality conflicts with the approved protocol."""


# --------------------------------------------------------------------------
# Provenance
# --------------------------------------------------------------------------
def _run(cmd: List[str], cwd: Optional[Path] = None) -> str:
    return subprocess.run(
        cmd, cwd=str(cwd or REPO), check=True, capture_output=True, text=True
    ).stdout.strip()


def git_state() -> Dict[str, Any]:
    return {
        "branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"]),
        "commit": _run(["git", "rev-parse", "HEAD"]),
        "dirty": bool(_run(["git", "status", "--porcelain"])),
        "status_porcelain": _run(["git", "status", "--porcelain"]),
    }


def package_versions() -> Dict[str, str]:
    import lightgbm, sklearn, scipy, pyarrow
    out = {
        "python": sys.version.split()[0],
        "lightgbm": lightgbm.__version__,
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
        "pyarrow": pyarrow.__version__,
        "node": platform.node(),
    }
    try:
        import dcor
        out["dcor"] = dcor.__version__
    except Exception:
        out["dcor"] = "MISSING"
    return out


def thread_env() -> Dict[str, str]:
    keys = ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS", "SLURM_JOB_ID", "SLURM_CPUS_PER_TASK"]
    return {k: os.environ.get(k, "") for k in keys}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


# --------------------------------------------------------------------------
# Safe writers (enforce the isolation rule)
# --------------------------------------------------------------------------
_ALLOWED_WRITE_ROOTS = (P0_DIR.resolve(), P0_OUTPUT_ROOT.resolve())


def _assert_write_allowed(path: Path) -> None:
    rp = path.resolve()
    if not any(str(rp).startswith(str(root)) for root in _ALLOWED_WRITE_ROOTS):
        raise ProtocolViolation(
            f"Refusing to write outside the approved P0 locations: {rp}"
        )


def write_json(path: Path, payload: Any) -> Path:
    _assert_write_allowed(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str))
    tmp.replace(path)
    return path


def write_table(df: pd.DataFrame, path: Path) -> Path:
    """Write csv + parquet twin (house convention)."""
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
# Canonical data + split reconstruction
# --------------------------------------------------------------------------
def load_params() -> dict:
    import yaml
    with open(REPO / "params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_canonical_splits(verbose: bool = True):
    """Canonical loader.  Imports run_temporal_cv._load_and_split_data verbatim.

    Returns (df_tv, df_test, df_assess, predictor_cols, categorical_cols).
    """
    from run_temporal_cv import _load_and_split_data

    params = load_params()
    df_tv, df_test, df_assess, pred_cols, cat_cols = _load_and_split_data(
        data_path=str(DATA_PATH),
        params=params,
        target_column=TARGET_COL,
        date_column=DATE_COL,
        assessment_year=2025,
        heldout_test_mode="pre_assessment_tail",
        sample_frac=None,
        sample_seed=2025,
        universe_start="2016-01-01",
        pre_assessment_end="2024-12-31",
    )
    n_tv, n_te, n_as = len(df_tv), len(df_test), len(df_assess)
    if (n_tv, n_te, n_as) != (N_DEVELOPMENT, N_HELDOUT, N_2025):
        raise ProtocolViolation(
            "Canonical split counts do not match the frozen experiment: "
            f"got ({n_tv}, {n_te}, {n_as}), expected "
            f"({N_DEVELOPMENT}, {N_HELDOUT}, {N_2025})."
        )
    if verbose:
        print(f"[p0] canonical splits OK: dev={n_tv} heldout={n_te} 2025={n_as}", flush=True)
    return df_tv, df_test, df_assess, pred_cols, cat_cols


def load_archived_folds() -> dict:
    return json.loads(ARCHIVED_FOLDS.read_text())


def rebuild_folds(df_tv: pd.DataFrame, verify: bool = True) -> Tuple[List[dict], dict]:
    """Rebuild the seven rolling-origin folds and verify archived index hashes."""
    from utils.motivation_utils import build_rolling_origin_protocol

    archive = load_archived_folds()
    sp = archive["split_protocol"]
    rebuilt = build_rolling_origin_protocol(
        df_tv,
        DATE_COL,
        train_mode=str(sp["train_mode"]),
        initial_train_months=int(sp["initial_train_months"]),
        val_fraction=float(sp["val_fraction"]),
        val_window_months=int(sp.get("val_window_months", 15)),
        step_months=int(sp["step_months"]),
        min_train_rows=int(sp["min_train_rows"]),
        min_val_rows=int(sp["min_val_rows"]),
    )
    if verify:
        if len(rebuilt) != len(archive["folds"]):
            raise ProtocolViolation(
                f"Rebuilt {len(rebuilt)} folds but the archive has {len(archive['folds'])}."
            )
        bad = []
        for rec, arch in zip(rebuilt, archive["folds"]):
            ok = (
                rec["train_index_hash"] == arch["train_index_hash"]
                and rec["val_index_hash"] == arch["val_index_hash"]
                and int(rec["train_size"]) == int(arch["train_size"])
                and int(rec["val_size"]) == int(arch["val_size"])
            )
            if not ok:
                bad.append(int(rec["fold_id"]))
        if bad:
            raise ProtocolViolation(
                f"Rebuilt fold index hashes do not match the archived protocol for folds {bad}."
            )
        print(f"[p0] archived fold index hashes verified for all {len(rebuilt)} folds", flush=True)
    return rebuilt, archive


def fitting_blocks(df_tv: pd.DataFrame, df_test: pd.DataFrame, folds: List[dict]) -> List[dict]:
    """The nine canonical fitting blocks (plan E.3).

    folds 1..7 training blocks  +  full development pool  +  full 2016-2024 production block.
    Returns dicts with block id, label, and a positional index array into a
    concatenated (df_tv, df_test) frame -- production uses both.
    """
    blocks: List[dict] = []
    for rec in folds:
        idx = np.asarray(rec["train_indices"], dtype=int)
        blocks.append(
            {
                "block_id": f"fold_{int(rec['fold_id']) + 1}_train",
                "block_kind": "cv_fold_train",
                "fold_id": int(rec["fold_id"]) + 1,
                "n": int(idx.size),
                "source": "development",
                "indices": idx,
            }
        )
    blocks.append(
        {
            "block_id": "development_pool",
            "block_kind": "development_pool",
            "fold_id": None,
            "n": int(len(df_tv)),
            "source": "development",
            "indices": np.arange(len(df_tv), dtype=int),
        }
    )
    blocks.append(
        {
            "block_id": "production_2016_2024",
            "block_kind": "production_block",
            "fold_id": None,
            "n": int(len(df_tv) + len(df_test)),
            "source": "development+heldout",
            "indices": np.arange(len(df_tv) + len(df_test), dtype=int),
        }
    )
    return blocks


def block_log_target(block: dict, y_dev_log: np.ndarray, y_prod_log: np.ndarray) -> np.ndarray:
    """log-price vector for a fitting block."""
    if block["source"] == "development":
        return y_dev_log[block["indices"]]
    return y_prod_log[block["indices"]]


# --------------------------------------------------------------------------
# Frozen configuration
# --------------------------------------------------------------------------
def frozen_lgbm_config() -> dict:
    cfg = json.loads(FROZEN_LGBM_CONFIG.read_text())
    from canonical_experiment import lgbm_params_hash
    got = lgbm_params_hash(cfg["lgbm_params"])
    if got != EXPECTED_LGBM_PARAMS_SHA256:
        raise ProtocolViolation(
            f"Frozen LightGBM parameter hash mismatch: got {got}, "
            f"expected {EXPECTED_LGBM_PARAMS_SHA256}."
        )
    return cfg


def frozen_rho_grid() -> List[float]:
    """The frozen 82-point positive grid + explicit zero (plan A.4).

    Canonical source is V12/protocol/lower_rho_grid_v2.json, the artifact that
    generated the augmented grid and which stores full float64 repr.  The two
    experiment_spec.json files record the same values at slightly lower JSON
    precision (max relative difference 4.97e-16, i.e. 1-2 ULP); that difference is
    a serialization artifact, is verified here, and is scientifically irrelevant
    because rho enters continuously.
    """
    gj = json.loads((V12 / "protocol" / "lower_rho_grid_v2.json").read_text())
    grid = sorted(float(x) for x in gj["augmented_positive_rhos"])
    if len(grid) != 82:
        raise ProtocolViolation(f"Expected 82 positive rho values, got {len(grid)}.")

    v6 = json.loads((V6 / "experiment_spec.json").read_text())
    v12 = json.loads((V12 / "experiment_spec.json").read_text())
    spec_grid = sorted(
        {float(x) for x in v6["cov_rhos"] if float(x) != 0.0}
        | {float(x) for x in v12["cov_rhos"] if float(x) != 0.0}
    )
    if len(spec_grid) != 82:
        raise ProtocolViolation(f"Spec-derived grid has {len(spec_grid)} values, expected 82.")
    rel = max(
        abs(a - b) / max(abs(b), 1e-300) for a, b in zip(grid, spec_grid)
    )
    if rel > 1e-12:
        raise ProtocolViolation(
            f"Protocol grid and experiment-spec grid disagree by rel {rel:.3e} (> 1 ULP scale)."
        )
    # Direct and Surrogate share the same numeric grid in both specs.
    if sorted(v6["cov_rhos"]) != sorted(v6["smooth_rhos"]):
        raise ProtocolViolation("v6 cov/smooth rho grids differ.")
    if sorted(v12["cov_rhos"]) != sorted(v12["smooth_rhos"]):
        raise ProtocolViolation("v12 cov/smooth rho grids differ.")
    return [0.0] + grid


def preflight_block() -> Dict[str, Any]:
    return {
        "git": git_state(),
        "versions": package_versions(),
        "thread_env": thread_env(),
        "cwd": str(Path.cwd()),
    }
