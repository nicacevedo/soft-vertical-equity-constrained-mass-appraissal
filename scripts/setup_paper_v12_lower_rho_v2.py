#!/usr/bin/env python3
"""Freeze the v2 lower-rho grid and copy immutable 994 identity into a new root.

Does not fit models, does not mutate v1 analysis trees, and does not compile TeX.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from utils.transition_regions import sha256_file, validate_canonical_result_root

CANONICAL = REPO / "output" / "paper_v6_preselection_994"
EXT_ROOT = REPO / "output" / "paper_v12_lower_rho_extension_994_v2"
DATA_ID = "d4929d43ec19badf"
SPLIT_ID = "3d464d4a611b131b"
ANALYSIS_BASE = (
    EXT_ROOT / "analysis" / f"data_id={DATA_ID}" / f"split_id={SPLIT_ID}" / "penalty_path_analysis"
)
V1 = (
    CANONICAL
    / "analysis"
    / f"data_id={DATA_ID}"
    / f"split_id={SPLIT_ID}"
    / "penalty_path_analysis"
    / "transition_regions_v1"
)
ASSETS_V1 = V1.parent / "transition_regions_paper_assets_v1"
PAPER_TEX = REPO / "paper" / "paper_v12.tex"


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=str(REPO), text=True).strip()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def sha256_list(values: List[float]) -> str:
    blob = json.dumps(values, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def lower_rho_extension() -> Dict[str, Any]:
    base = np.geomspace(0.1, 100.0, 50)
    q = float(base[1] / base[0])
    old = [float(x) for x in base.tolist()]
    lower: List[float] = []
    rho = float(base[0] / q)
    while rho >= 1e-3 - 1e-18:
        lower.append(float(rho))
        rho /= q
    lower = sorted(lower)
    if any(abs(x - 0.1) < 1e-15 for x in lower):
        raise RuntimeError("lower grid must not duplicate 0.1")
    if len(lower) != 32:
        raise RuntimeError(f"expected 32 lower-rho values, got {len(lower)}")
    augmented_positive = sorted(lower + old)
    if len(augmented_positive) != 82:
        raise RuntimeError(f"expected 82 positive rhos, got {len(augmented_positive)}")
    return {
        "q": q,
        "old_positive_rhos": old,
        "n_old_positive": 50,
        "new_positive_rhos": lower,
        "n_new_positive": 32,
        "augmented_positive_rhos": augmented_positive,
        "n_augmented_positive": 82,
        "min_new": float(min(lower)),
        "max_new": float(max(lower)),
        "min_positive_augmented": float(min(augmented_positive)),
        "max_positive_augmented": float(max(augmented_positive)),
        "old_positive_sha256": sha256_list(old),
        "new_positive_sha256": sha256_list(lower),
        "augmented_positive_sha256": sha256_list(augmented_positive),
        "construction": "base=geomspace(0.1,100,50); q=base[1]/base[0]; rho=base[0]/q; while rho>=1e-3",
        "no_duplicate_0.1": True,
        "do_not_rerun_old_or_zero": True,
    }


def hash_if_exists(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    if path.is_file():
        return {"path": str(path), "exists": True, "sha256": sha256_file(path), "size": path.stat().st_size}
    files = [p for p in sorted(path.rglob("*")) if p.is_file()]
    h = hashlib.sha256()
    for p in files:
        rel = str(p.relative_to(path)).encode("utf-8")
        h.update(rel + b"\0")
        h.update(sha256_file(p).encode("utf-8") + b"\n")
    return {
        "path": str(path),
        "exists": True,
        "n_files": len(files),
        "tree_manifest_sha256": h.hexdigest(),
    }


def main() -> int:
    identity = validate_canonical_result_root(CANONICAL)
    grid = lower_rho_extension()
    EXT_ROOT.mkdir(parents=True, exist_ok=True)
    (EXT_ROOT / "logs").mkdir(exist_ok=True)
    (EXT_ROOT / "protocol").mkdir(exist_ok=True)
    (EXT_ROOT / "provenance").mkdir(exist_ok=True)
    (EXT_ROOT / "qa").mkdir(exist_ok=True)
    ANALYSIS_BASE.mkdir(parents=True, exist_ok=True)

    for name in ("lgbm_config.json", "frozen_baseline.json", "baseline_gate.json"):
        src = CANONICAL / name
        dst = EXT_ROOT / name
        shutil.copy2(src, dst)
        if sha256_file(src) != sha256_file(dst):
            raise RuntimeError(f"copy hash mismatch {name}")

    write_json(EXT_ROOT / "protocol" / "lower_rho_grid_v2.json", grid)
    write_json(
        EXT_ROOT / "experiment_manifest.json",
        {
            "experiment": "paper_v12_lower_rho_extension_994_v2",
            "canonical_source_root": str(CANONICAL),
            "selection_performed": False,
            "no_selection_confirmation": (
                "No rho, penalty family, or penalized configuration was selected or ranked in this analysis."
            ),
            "canonical_identity": {
                "data_id": DATA_ID,
                "split_id": SPLIT_ID,
                "lgbm_config_id": "407d47775760c14d",
                "baseline_gate": "ADOPT_994",
                "seed": 2025,
                "n_estimators": 994,
            },
            "grid": {
                "n_new_positive": 32,
                "n_old_positive": 50,
                "n_augmented_positive": 82,
                "n_path_points_per_family": 83,
                "expected_new_cv_fits": 448,
                "expected_new_oos_fits": 128,
            },
            "v1_immutable": True,
            "identity_ok": identity.get("ok"),
        },
    )

    pre = {
        "utc": utc_now(),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "head": git("rev-parse", "HEAD"),
        "git_status_porcelain": git("status", "--porcelain"),
        "paper_v12_tex": hash_if_exists(PAPER_TEX),
        "canonical_files": {
            "lgbm_config.json": hash_if_exists(CANONICAL / "lgbm_config.json"),
            "frozen_baseline.json": hash_if_exists(CANONICAL / "frozen_baseline.json"),
            "baseline_gate.json": hash_if_exists(CANONICAL / "baseline_gate.json"),
            "experiment_manifest.json": hash_if_exists(CANONICAL / "experiment_manifest.json"),
            "combined_path_table.csv": hash_if_exists(CANONICAL / "analysis" / "combined_path_table.csv"),
            "combined_path_table.parquet": hash_if_exists(CANONICAL / "analysis" / "combined_path_table.parquet"),
        },
        "transition_regions_v1": hash_if_exists(V1 / "qa" / "FINAL_STATUS.json"),
        "transition_regions_v1_tree": hash_if_exists(V1 / "qa"),
        "paper_assets_v1": hash_if_exists(ASSETS_V1 / "qa" / "FINAL_PAPER_ASSET_STATUS.json"),
        "paper_asset_manifest": hash_if_exists(ASSETS_V1 / "provenance" / "paper_asset_manifest.csv"),
        "grid": {
            "n_new": grid["n_new_positive"],
            "min_new": grid["min_new"],
            "max_new": grid["max_new"],
            "new_positive_sha256": grid["new_positive_sha256"],
            "augmented_positive_sha256": grid["augmented_positive_sha256"],
        },
        "no_v1_mutation": True,
        "no_tex_compilation": True,
    }
    write_json(EXT_ROOT / "provenance" / "PREFLIGHT.json", pre)
    print(json.dumps({"status": "READY", "n_new": 32, "min_new": grid["min_new"], "max_new": grid["max_new"], "new_sha256": grid["new_positive_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
