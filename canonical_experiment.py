"""Small helpers for the paper-v6 descriptive penalty-path experiment.

No model selection. These utilities only freeze a LightGBM baseline,
record provenance, and keep rho=0 outside the geometric grid.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

VALID_STAGES = ("baseline-search", "cv", "test", "forward", "all")


def frozen_baseline_path(result_root: str) -> Path:
    return Path(result_root) / "frozen_baseline.json"


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True, default=str)
    tmp.replace(path)


def read_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def package_versions() -> Dict[str, str]:
    versions = {"python": sys.version.split()[0]}
    try:
        import lightgbm
        versions["lightgbm"] = str(lightgbm.__version__)
    except Exception:
        versions["lightgbm"] = "unknown"
    try:
        import numpy
        versions["numpy"] = str(numpy.__version__)
    except Exception:
        versions["numpy"] = "unknown"
    try:
        import pandas
        versions["pandas"] = str(pandas.__version__)
    except Exception:
        versions["pandas"] = "unknown"
    try:
        import sklearn
        versions["scikit-learn"] = str(sklearn.__version__)
    except Exception:
        versions["scikit-learn"] = "unknown"
    return versions


def git_state(repo_root: Optional[Path] = None) -> Dict[str, str]:
    root = repo_root or Path(__file__).resolve().parent
    out = {"git_commit": "unknown", "git_branch": "unknown"}
    try:
        out["git_commit"] = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
        out["git_branch"] = (
            subprocess.check_output(
                ["git", "rev-parse", "--abbrev-ref", "HEAD"],
                cwd=root,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except Exception:
        pass
    return out


def seed_lgbm_candidates_from_repo(base_lgbm_params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Historical CV-only LightGBM configs plus the CCAO default. Drops test-selected configs."""
    extra: List[Dict[str, Any]] = []
    yaml_path = Path("best_lgbm_baseline_configs.yaml")
    if yaml_path.is_file():
        with yaml_path.open("r", encoding="utf-8") as f:
            blob = yaml.safe_load(f) or {}
        for name, rec in dict(blob.get("lgbm_baselines", {})).items():
            if str(name).startswith("test_"):
                continue
            raw = dict(rec.get("lgbm_params", {}))
            cand = dict(base_lgbm_params)
            for key in list(cand.keys()):
                if key in raw and raw[key] is not None:
                    cand[key] = raw[key]
            extra.append(cand)
    try:
        with open("params.yaml", "r", encoding="utf-8") as f:
            params = yaml.safe_load(f)
        hp = dict(params["model"]["hyperparameter"]["default"])
        num_leaves = int(hp["num_leaves"])
        add_depth = int(hp.get("add_to_linked_depth", 4))
        cand = dict(base_lgbm_params)
        cand.update(
            {
                "n_estimators": int(hp["num_iterations"]),
                "learning_rate": float(hp["learning_rate"]),
                "max_bin": int(hp["max_bin"]),
                "num_leaves": num_leaves,
                "max_depth": int(np.floor(np.log2(max(num_leaves, 2))) + add_depth),
                "colsample_bytree": float(hp["feature_fraction"]),
                "min_split_gain": float(hp["min_gain_to_split"]),
                "min_child_samples": int(hp["min_data_in_leaf"]),
                "max_cat_threshold": int(hp["max_cat_threshold"]),
                "min_data_per_group": int(hp["min_data_per_group"]),
                "cat_smooth": float(hp["cat_smooth"]),
                "cat_l2": float(hp["cat_l2"]),
                "reg_alpha": float(hp["lambda_l1"]),
                "reg_lambda": float(hp["lambda_l2"]),
            }
        )
        extra.append(cand)
    except Exception:
        pass
    return extra
