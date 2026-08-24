"""Small helpers for the paper-v6 descriptive penalty-path experiment.

No model selection. These utilities only freeze a LightGBM baseline,
record provenance, and keep rho=0 outside the geometric grid.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import yaml

VALID_STAGES = ("baseline-search", "cv", "test", "forward", "all", "baseline-report")


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
    try:
        import dcor
        versions["dcor"] = str(dcor.__version__)
    except Exception:
        versions["dcor"] = "unknown"
    return versions


class BaselineFreezeError(RuntimeError):
    """Raised when a failed baseline search would otherwise write a freeze file."""


def write_frozen_baseline(path: Path, payload: Dict[str, Any], *, fallback_used: bool) -> None:
    """Write frozen_baseline.json only for an eligible search winner.

    Diagnostics for a failed search must be written by the caller *before*
    this function. A fallback winner must never create or overwrite the
    canonical freeze artifact.
    """
    if bool(fallback_used):
        raise BaselineFreezeError(
            "Baseline search produced no eligible seven-fold candidate; "
            "refusing to write frozen_baseline.json. See search diagnostics."
        )
    write_json(path, payload)


def git_state(repo_root: Optional[Path] = None) -> Dict[str, str]:
    root = repo_root or Path(__file__).resolve().parent
    out = {
        "git_commit": "unknown",
        "git_branch": "unknown",
        "git_dirty": "unknown",
        "git_diff_sha256": "",
    }
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
        porcelain = (
            subprocess.check_output(["git", "status", "--porcelain"], cwd=root, stderr=subprocess.DEVNULL)
            .decode()
        )
        dirty = bool(porcelain.strip())
        out["git_dirty"] = "true" if dirty else "false"
        if dirty:
            diff = subprocess.check_output(["git", "diff"], cwd=root, stderr=subprocess.DEVNULL)
            out["git_diff_sha256"] = hashlib.sha256(diff).hexdigest()
    except Exception:
        pass
    return out


def model_grid_hash(specs: Sequence[Dict[str, Any]]) -> str:
    payload = [{"name": s.get("name"), "config": dict(s.get("config") or {})} for s in specs]
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def frozen_baseline_hash(path: Path) -> str:
    if not path.is_file():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def lgbm_params_hash(params: Dict[str, Any]) -> str:
    blob = json.dumps(dict(params), sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def expected_config_ids(specs: Sequence[Dict[str, Any]]) -> List[str]:
    from utils.motivation_utils import _stable_hash

    return [
        _stable_hash({"model_name": s["name"], "config": dict(s.get("config") or {})})
        for s in specs
    ]


def cv_completion_path(result_root: str) -> Path:
    return Path(result_root) / "cv_completion.json"


def _pairs_from_records(records, *, config_key="config_id", fold_key="fold_id") -> List[Tuple[str, int]]:
    if records is None:
        return []
    if hasattr(records, "empty"):
        if bool(records.empty):
            return []
        cols = set(getattr(records, "columns", []))
        if config_key not in cols or fold_key not in cols:
            return []
        rows = records[[config_key, fold_key]].drop_duplicates().itertuples(index=False, name=None)
        return [(str(c), int(f)) for c, f in rows]
    out: List[Tuple[str, int]] = []
    for rec in records:
        if rec is None:
            continue
        if config_key in rec and fold_key in rec:
            out.append((str(rec[config_key]), int(rec[fold_key])))
    return out


def build_cv_completion(
    *,
    data_id: str,
    split_id: str,
    expected_config_ids: Sequence[str],
    expected_fold_ids: Sequence[int],
    run_records=None,
    failed_records=None,
    invalid_config_ids: Optional[Iterable[str]] = None,
    frozen_baseline_sha: str = "",
    model_grid_sha: str = "",
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    expected_configs = [str(x) for x in expected_config_ids]
    expected_folds = [int(x) for x in expected_fold_ids]
    invalid = sorted({str(x) for x in (invalid_config_ids or [])})
    completed = sorted(set(_pairs_from_records(run_records)))
    failed = sorted(set(_pairs_from_records(failed_records)))
    expected_pairs = {(c, f) for c in expected_configs for f in expected_folds}
    completed_set = set(completed)
    failed_set = set(failed)
    valid_expected = {(c, f) for c, f in expected_pairs if c not in set(invalid)}
    missing = sorted(valid_expected - completed_set)
    failed_valid = sorted(failed_set - {(c, f) for c, f in failed_set if c in set(invalid)})
    status = "complete" if (not missing and not failed_valid) else "incomplete"
    payload = {
        "status": status,
        "data_id": str(data_id),
        "split_id": str(split_id),
        "frozen_baseline_hash": str(frozen_baseline_sha),
        "model_grid_hash": str(model_grid_sha),
        "expected_config_ids": expected_configs,
        "expected_fold_ids": expected_folds,
        "completed_config_fold": [{"config_id": c, "fold_id": f} for c, f in completed],
        "failed_config_fold": [{"config_id": c, "fold_id": f} for c, f in failed],
        "invalid_config_ids": invalid,
        "missing_valid_config_fold": [{"config_id": c, "fold_id": f} for c, f in missing],
        "n_expected_pairs": int(len(expected_pairs)),
        "n_completed_pairs": int(len(completed_set)),
        "n_failed_pairs": int(len(failed_set)),
        "versions": package_versions(),
        **git_state(),
    }
    if extra:
        payload.update(extra)
    return payload


def require_complete_cv(
    result_root: str,
    *,
    data_id: str,
    split_id: str,
    frozen_baseline_sha: str,
    model_grid_sha: str,
    allow_incomplete: bool = False,
) -> Dict[str, Any]:
    path = cv_completion_path(result_root)
    if not path.is_file():
        raise RuntimeError(
            f"Missing {path}. Run --stage cv to completion before test/forward, "
            "or pass --allow-incomplete-cv for a named smoke/debug run."
        )
    blob = read_json(path)
    problems = []
    if str(blob.get("data_id")) != str(data_id):
        problems.append(f"data_id {blob.get('data_id')!r} != {data_id!r}")
    if str(blob.get("split_id")) != str(split_id):
        problems.append(f"split_id {blob.get('split_id')!r} != {split_id!r}")
    if str(blob.get("frozen_baseline_hash", "")) != str(frozen_baseline_sha):
        problems.append("frozen baseline hash mismatch")
    if str(blob.get("model_grid_hash", "")) != str(model_grid_sha):
        problems.append("model/rho grid hash mismatch")
    if str(blob.get("status")) != "complete":
        problems.append(f"CV status is {blob.get('status')!r}, not complete")
    if problems and not allow_incomplete:
        raise RuntimeError(
            "Refusing test/forward on an incompatible or incomplete CV artifact: "
            + "; ".join(problems)
        )
    return blob


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
