#!/usr/bin/env python3
"""
Finalize (CCAO ``05-finalize`` analog).

Cook County writes run metadata (IDs, git SHA, DVC hashes), aggregates stage
timings, and renders reports.

This lightweight stage records **what artifacts exist** for the resolved
``data_id`` / ``split_id``: paths under ``result_root``, optional git metadata,
and merges with ``pipeline_last_context.json``. No Quarto dependency.

Output::

  <result_root>/analysis/data_id=…/split_id=…/pipeline_finalize_manifest.json

Usage::

  python pipeline/05_finalize.py
  python pipeline/05_finalize.py --result-root ./output/robust_rolling_origin_cv
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pipeline._helpers import DEFAULT_RESULT_ROOT, parse_data_split_ids, read_context, repo_root, write_context


def _git_meta(repo: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"git_commit": None, "git_branch": None}
    try:
        out["git_commit"] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True, timeout=5
        ).strip()
        out["git_branch"] = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"], text=True, timeout=5
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pass
    return out


def _count_files(directory: Path, pattern: str = "*.parquet") -> int:
    if not directory.is_dir():
        return 0
    return sum(1 for _ in directory.rglob(pattern))


def _collect_artifact_paths(*, result_root: Path, analysis_dir: Path, data_id: str, split_id: str) -> Dict[str, Any]:
    test_files = [
        "test_eval_status.json",
        "test_metrics.csv",
        "test_predictions.parquet",
        "test_eval_metadata.json",
        "test_flagged_configs.csv",
    ]
    found: Dict[str, Any] = {}
    for name in test_files:
        p = analysis_dir / name
        found[name] = str(p.resolve()) if p.is_file() else None

    selected_dir = analysis_dir / "selected"
    found["selected_dir"] = str(selected_dir.resolve()) if selected_dir.is_dir() else None
    for name in (
        "selected_models.json",
        "selected_models.csv",
        "selected_models_evaluation.csv",
        "selected_models_folds.csv",
        "selected_models_interpret.md",
    ):
        p = selected_dir / name
        found[name] = str(p.resolve()) if p.is_file() else None

    report_dir = selected_dir / "report"
    found["report_dir"] = str(report_dir.resolve()) if report_dir.is_dir() else None
    for name in (
        "three_model_comparison.html",
        "three_model_metrics.csv",
        "three_model_decile.csv",
        "three_model_township_error.csv",
    ):
        p = report_dir / name
        found[name] = str(p.resolve()) if p.is_file() else None

    # Stage-level CV outputs that live outside the analysis split folder.
    stem = f"data_id={data_id}/split_id={split_id}"
    runs_dir = result_root / "runs" / stem
    bootstrap_dir = result_root / "bootstrap_metrics" / stem
    predictions_dir = result_root / "predictions" / stem
    found["runs_dir"] = str(runs_dir.resolve()) if runs_dir.is_dir() else None
    found["runs_n_parquet"] = _count_files(runs_dir, "*.parquet")
    found["bootstrap_metrics_dir"] = str(bootstrap_dir.resolve()) if bootstrap_dir.is_dir() else None
    found["bootstrap_metrics_n_parquet"] = _count_files(bootstrap_dir, "*.parquet")
    found["predictions_dir"] = str(predictions_dir.resolve()) if predictions_dir.is_dir() else None
    found["predictions_n_parquet"] = _count_files(predictions_dir, "*.parquet")
    return found


def _load_selected_summary(analysis_dir: Path) -> Dict[str, Any]:
    json_path = analysis_dir / "selected" / "selected_models.json"
    if not json_path.is_file():
        return {}
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"error": f"could not parse {json_path}"}
    return {
        "accuracy_metric": payload.get("accuracy_metric"),
        "constraint_metrics": payload.get("constraint_metrics"),
        "candidate_pools": payload.get("candidate_pools") or payload.get("candidate_pool"),
        "winners": {
            rule: {
                "config_id": sel.get("config_id"),
                "model_name": sel.get("model_name"),
                "model_family": sel.get("model_family"),
                f"cv_{payload.get('accuracy_metric','RMSE')}_mean":
                    sel.get(f"cv_{payload.get('accuracy_metric','RMSE')}_mean"),
                "test_R2": sel.get("test_R2"),
                "test_RMSE": sel.get("test_RMSE"),
                "test_PRD": sel.get("test_PRD"),
                "test_PRB": sel.get("test_PRB"),
                "test_VEI": sel.get("test_VEI"),
                "nash_log_utility": sel.get("nash_log_utility"),
            }
            for rule, sel in payload.get("selections", {}).items()
        },
    }


def run_finalize(*, result_root: Path, data_id: str, split_id: str) -> Path:
    analysis_dir = result_root / "analysis" / f"data_id={data_id}" / f"split_id={split_id}"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    ctx = read_context()
    payload: Dict[str, Any] = {
        "stage": "finalize",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "data_id": data_id,
        "split_id": split_id,
        "result_root": str(result_root.resolve()),
        "analysis_dir": str(analysis_dir.resolve()),
        "prior_context": ctx,
        "artifacts": _collect_artifact_paths(
            result_root=result_root,
            analysis_dir=analysis_dir,
            data_id=data_id,
            split_id=split_id,
        ),
        "selected_models": _load_selected_summary(analysis_dir),
        **_git_meta(_REPO),
    }

    out_path = analysis_dir / "pipeline_finalize_manifest.json"
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    tmp.replace(out_path)

    # Refresh shared context with finalize pointer
    ctx_update = dict(ctx)
    ctx_update.update(
        {
            "stage": "finalize",
            "data_id": data_id,
            "split_id": split_id,
            "finalize_manifest_json": str(out_path.resolve()),
        }
    )
    write_context(ctx_update)

    print("FINALIZE")
    print("=" * 60)
    print(f"  manifest: {out_path}")
    print(f"  updated:  {repo_root() / 'pipeline' / 'pipeline_last_context.json'}")
    return out_path


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Finalize — write manifest + refresh pipeline context.")
    p.add_argument("--result-root", type=str, default=str(DEFAULT_RESULT_ROOT))
    p.add_argument("--data-id", type=str, default=None)
    p.add_argument("--split-id", type=str, default=None)
    p.add_argument("--no-context", action="store_true")
    return p


def main() -> None:
    args = _build_parser().parse_args()
    result_root = Path(args.result_root).resolve()
    data_id, split_id = parse_data_split_ids(
        data_id=args.data_id,
        split_id=args.split_id,
        result_root=result_root,
        prefer_context=not args.no_context,
    )
    run_finalize(result_root=result_root, data_id=data_id, split_id=split_id)


if __name__ == "__main__":
    main()
