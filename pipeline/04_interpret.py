#!/usr/bin/env python3
"""
Interpret (CCAO ``04-interpret`` analog).

Cook County exports SHAP contributions, global feature importance, and
optional comparable sales. This research codebase does not yet have a
first-class SHAP / comps export wired to the fairness-regularized
estimators.

This stage focuses on the **two selections** produced by stage 02-assess:

- prints each winner's ``config_id``, ``model_name``, and
  ``model_config_json`` so the chosen hyperparameters are visible;
- prints the IAAO-style fairness footprint (CV mean PRD/PRB/VEI plus
  whether each falls inside the IAAO band);
- echoes the held-out test summary (R2, RMSE, MAE, PRD, PRB, VEI);
- writes a small Markdown report next to the other selected/ artifacts;
- optionally invokes ``quick_test_models.py`` (``--run-quick-test``).

Outputs (under ``analysis/data_id=…/split_id=…/selected/``):

- ``selected_models_interpret.md`` — human-readable summary

Usage::

  python pipeline/04_interpret.py
  python pipeline/04_interpret.py --run-quick-test -- --rho 1.0
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from pipeline._helpers import DEFAULT_RESULT_ROOT, parse_data_split_ids, run_repo_script


def _safe_float(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _load_lgbm_base_params() -> Dict[str, Any]:
    path = _REPO / "model_params.yaml"
    if not path.is_file():
        return {}
    try:
        import yaml
    except ImportError:
        return {}
    try:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError:
        return {}
    base = cfg.get("LGBMRegressor")
    return dict(base) if isinstance(base, dict) else {}


def _format_hyperparams(model_config_json: str) -> str:
    if not model_config_json:
        return "  (none recorded)"
    try:
        cfg = json.loads(model_config_json)
    except json.JSONDecodeError:
        return f"  (unparseable JSON: {model_config_json[:120]}…)"
    if not isinstance(cfg, dict):
        return f"  {cfg!r}"
    base_params: Dict[str, Any] = {}
    if "lgbm_base_config_id" in cfg:
        base_params = _load_lgbm_base_params()
    lines: List[str] = []
    for k in sorted(cfg.keys()):
        lines.append(f"  - {k}: {cfg[k]}")
    if base_params:
        lines.append("  - (LightGBM base hyperparameters from model_params.yaml::LGBMRegressor)")
        for k in sorted(base_params.keys()):
            lines.append(f"      · {k}: {base_params[k]}")
    return "\n".join(lines)


def _bounds_status(value: float, *, lower: float | None, upper: float | None) -> str:
    import math

    if not math.isfinite(value):
        return "missing"
    if lower is not None and value < lower:
        return f"below [{lower}, {upper}]"
    if upper is not None and value > upper:
        return f"above [{lower}, {upper}]"
    return f"inside [{lower}, {upper}]"


def _build_markdown(payload: Dict[str, Any]) -> str:
    from pipeline._selection import CONSTRAINT_SPECS

    lines: List[str] = []
    lines.append(f"# Selected models — interpret stage")
    lines.append("")
    lines.append(f"- result_root: `{payload['result_root']}`")
    lines.append(f"- data_id: `{payload['data_id']}`  ")
    lines.append(f"- split_id: `{payload['split_id']}`  ")
    lines.append(f"- accuracy metric: `{payload['accuracy_metric']}`  ")
    lines.append(f"- constraint metrics: `{payload['constraint_metrics']}`")
    pools = payload.get("candidate_pools", {}) or {}
    if pools:
        lines.append("- candidate pools:")
        for rule, pool in pools.items():
            lines.append(
                f"  - `{rule}`: {pool.get('n_configs', '?')} configs across "
                f"{pool.get('n_folds', '?')} folds (families: {pool.get('families', [])})"
            )
    lines.append("")

    for rule, sel in payload["selections"].items():
        lines.append(f"## Selection rule: `{rule}`")
        lines.append(f"- **config_id:** `{sel['config_id']}`")
        lines.append(f"- **model_name:** `{sel['model_name']}`")
        lines.append(f"- **model_family:** `{sel.get('model_family', '')}`")
        lines.append(f"- **n_folds:** {sel.get('n_folds', '?')}")
        if "nash" in str(rule):
            lines.append(f"- **nash_log_utility:** {sel.get('nash_log_utility', float('nan')):.6g}")
        elif rule == "utopia":
            lines.append(f"- **utopia_distance (legacy key):** {sel.get('utopia_distance', float('nan')):.4f}")

        acc = payload["accuracy_metric"]
        cv_acc = sel.get(f"cv_{acc}_mean", float("nan"))
        cv_std = sel.get(f"cv_{acc}_std", float("nan"))
        lines.append(f"- **CV {acc} (mean ± std):** {cv_acc:.6g} ± {cv_std:.6g}")

        for cid in payload.get("constraint_metrics", []):
            spec = CONSTRAINT_SPECS.get(cid)
            mean_val = _safe_float(sel.get(f"cv_{cid}_mean"))
            status = _bounds_status(mean_val, lower=getattr(spec, "lower", None), upper=getattr(spec, "upper", None))
            lines.append(f"- **CV {cid} mean:** {mean_val:.6g}  ({status})")

        lines.append("")
        lines.append("**Held-out test metrics:**")
        for col in ("R2", "RMSE", "MAE", "PRD", "PRB", "VEI", "COD"):
            v = sel.get(f"test_{col}", None)
            if v is not None:
                lines.append(f"  - {col}: {_safe_float(v):.6g}")
        lines.append("")
        lines.append("**Hyperparameters (`model_config_json`):**")
        lines.append("")
        lines.append("```")
        lines.append(_format_hyperparams(sel.get("model_config_json", "")))
        lines.append("```")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main() -> None:
    argv = sys.argv[1:]
    qt_argv: list[str] = []
    if "--" in argv:
        i = argv.index("--")
        argv, qt_argv = argv[:i], argv[i + 1 :]

    p = argparse.ArgumentParser(description="Interpret stage — focused on the two selected models.")
    p.add_argument("--result-root", type=str, default=str(DEFAULT_RESULT_ROOT))
    p.add_argument("--data-id", type=str, default=None)
    p.add_argument("--split-id", type=str, default=None)
    p.add_argument("--no-context", action="store_true")
    p.add_argument("--run-quick-test", action="store_true", help="Run quick_test_models.py (extras after --).")
    args = p.parse_args(argv)

    result_root = Path(args.result_root).resolve()
    data_id, split_id = parse_data_split_ids(
        data_id=args.data_id,
        split_id=args.split_id,
        result_root=result_root,
        prefer_context=not args.no_context,
    )

    selected_path = (
        result_root / "analysis" / f"data_id={data_id}" / f"split_id={split_id}" / "selected" / "selected_models.json"
    )
    if not selected_path.is_file():
        raise FileNotFoundError(f"selected_models.json not found at {selected_path}. Run 02_assess first.")
    payload = json.loads(selected_path.read_text(encoding="utf-8"))

    md = _build_markdown(payload)
    out_path = selected_path.parent / "selected_models_interpret.md"
    out_path.write_text(md, encoding="utf-8")

    print("=" * 70)
    print("INTERPRET — selected models summary")
    print("=" * 70)
    print(md)
    print(f"  → {out_path}")

    if args.run_quick_test:
        run_repo_script("quick_test_models.py", qt_argv)


if __name__ == "__main__":
    main()
