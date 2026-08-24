#!/usr/bin/env python3
"""Paper-v6 pre-selection reporting. No rho/family/penalized-model selection."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from canonical_experiment import git_state  # noqa: F401

DIRECT = "LGBCovPenalty"
SURROGATE = "LGBSmoothPenalty"
NATIVE = "LGBMRegressor"
LINEAR = "LinearRegression"
ANCHORS = (0.0, 0.1, 1.0, 10.0, 100.0)
FORBIDDEN_PHRASES = (
    "PLACEHOLDER",
    "populate from",
    "Reference layout only",
    "results_reference_assets",
    "after the 728-fit CV completes",
)
METRIC_MAP = {
    "R2_price": "R2_price",
    "MAE_price": "MAE_price",
    "MAPE": "MAPE",
    "RMSE_log": "RMSE_log",
    "Median ratio": "median_ratio",
    "Mean ratio": "mean_ratio",
    "W. Mean ratio": "weighted_mean_ratio",
    "COD": "COD",
    "COV_IAAO": "COV",
    "PRD": "PRD",
    "PRB": "PRB",
    "MKI": "MKI",
    "VEI": "VEI",
    "Beta_log": "Beta_log",
    "Cov_log_residual_log_price": "Cov_log_residual_log_price",
    "dCor_e_y": "dCor_e_y",
}
PATH_METRICS = list(METRIC_MAP.values())
DIRECT_COLOR = "#1D4ED8"
SURR_COLOR = "#C2410C"
NATIVE_COLOR = "#111827"
LINEAR_COLOR = "#6B7280"

ROOT = REPO / "output" / "paper_v6_preselection"
BASELINE_ROOT = ROOT / "baseline_reporting"
PREVIEW_ROOT = ROOT / "reporting_preview"
ANALYSIS_ROOT = ROOT / "analysis"
PAPER_OUT = ROOT / "paper_outputs"
FIG_OUT = PAPER_OUT / "figures"
FIG_OUT = FIG_OUT
TAB_OUT = PAPER_OUT / "tables"
MANIFEST_ROOT = ROOT / "manifests"
PAPER_TEX = REPO / "paper" / "paper_v6.tex"
PAPER_IMG = REPO / "paper" / "img" / "generated_v6_preselection"


def configure_paths(result_root: Optional[str] = None) -> Path:
    global ROOT, BASELINE_ROOT, PREVIEW_ROOT, ANALYSIS_ROOT, PAPER_OUT
    global FIG_OUT, FIG_OUT, TAB_OUT, MANIFEST_ROOT, PAPER_IMG
    raw = result_root if result_root is not None else os.environ.get("P6_RESULT_ROOT", "output/paper_v6_preselection")
    path = Path(raw)
    ROOT = path.resolve() if path.is_absolute() else (REPO / path).resolve()
    BASELINE_ROOT = ROOT / "baseline_reporting"
    PREVIEW_ROOT = ROOT / "reporting_preview"
    ANALYSIS_ROOT = ROOT / "analysis"
    PAPER_OUT = ROOT / "paper_outputs"
    FIG_OUT = PAPER_OUT / "figures"
    FIG_OUT = FIG_OUT
    TAB_OUT = PAPER_OUT / "tables"
    MANIFEST_ROOT = ROOT / "manifests"
    PAPER_IMG = REPO / "paper" / "img" / "generated_v6_preselection"
    return ROOT


configure_paths()


def parse_config(raw: Any) -> Dict[str, Any]:
    if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
        return {}
    if isinstance(raw, dict):
        return dict(raw)
    text = str(raw).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def rho_from_row(row: pd.Series) -> float:
    if "rho" in row.index and pd.notna(row.get("rho")):
        try:
            return float(row["rho"])
        except (TypeError, ValueError):
            pass
    cfg = parse_config(row.get("model_config_json"))
    if "rho" in cfg:
        try:
            return float(cfg["rho"])
        except (TypeError, ValueError):
            return float("nan")
    return float("nan")


def family_from_name(name: str) -> str:
    return {LINEAR: "Linear", NATIVE: "LightGBM", DIRECT: "Direct", SURROGATE: "Surrogate"}.get(str(name), str(name))


def nearest_anchor(grid: Sequence[float], target: float) -> float:
    arr = np.asarray(list(grid), dtype=float)
    if arr.size == 0:
        return float(target)
    return float(arr[int(np.argmin(np.abs(arr - float(target))))])


def map_display_anchors(grid: Sequence[float], targets: Sequence[float] = ANCHORS) -> List[float]:
    return [nearest_anchor(grid, t) for t in targets]


def expected_canonical_rhos() -> List[float]:
    from run_temporal_cv import _build_rho_values, _prepend_explicit_zero

    return [float(x) for x in _prepend_explicit_zero(_build_rho_values([0.1, 100.0], rho_count=50, rho_scale="geom"))]


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def standardize_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    name_col = "model_name" if "model_name" in out.columns else "name"
    out["model_name"] = out[name_col].astype(str)
    out["family"] = out["model_name"].map(family_from_name)
    if "config_id" not in out.columns:
        out["config_id"] = ""
    out["config_id"] = out["config_id"].astype(str)
    out["rho"] = out.apply(rho_from_row, axis=1)
    for src, dst in METRIC_MAP.items():
        if src in out.columns:
            out[dst] = pd.to_numeric(out[src], errors="coerce")
    return out


def concat_parquets(paths: Iterable[Path]) -> pd.DataFrame:
    files = [Path(p) for p in paths if Path(p).is_file()]
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)


def config_metadata_table(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return pd.DataFrame(columns=["config_id", "family", "rho", "model_name"])
    keep = [c for c in ("config_id", "family", "rho", "model_name") if c in metrics.columns]
    return metrics[keep].drop_duplicates("config_id").copy()


def _config_id_col(df: pd.DataFrame) -> str:
    for col in ("config_id", "config_id"):
        if col in df.columns:
            return col
    raise RuntimeError("Prediction shards are missing config_id.")


def join_prediction_metadata(preds: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    if preds.empty:
        return preds
    pred_col = _config_id_col(preds)
    meta_src = metrics.copy()
    if pred_col not in meta_src.columns and "config_id" in meta_src.columns:
        meta_src = meta_src.rename(columns={"config_id": pred_col})
    if pred_col not in meta_src.columns and "config_id" in meta_src.columns:
        meta_src = meta_src.rename(columns={"config_id": pred_col})
    meta = config_metadata_table(meta_src)
    if pred_col not in meta.columns and "config_id" in meta.columns:
        meta = meta.rename(columns={"config_id": pred_col})
    out = preds.copy()
    out[pred_col] = out[pred_col].astype(str)
    meta[pred_col] = meta[pred_col].astype(str)
    out = out.drop(columns=[c for c in ("family", "rho") if c in out.columns], errors="ignore")
    out = out.merge(meta, on=pred_col, how="left", validate="many_to_one")
    if "model_name" in out.columns:
        custom = out["model_name"].isin([DIRECT, SURROGATE])
        rho_ok = np.isfinite(pd.to_numeric(out["rho"], errors="coerce"))
        bad = out.loc[custom & (out["family"].isna() | ~rho_ok)]
        if not bad.empty:
            raise RuntimeError(f"{len(bad)} custom prediction rows lack family/rho after config_id join.")
    return out


def assert_no_duplicate_pairs(df: pd.DataFrame, keys: Sequence[str], label: str) -> None:
    if df.empty:
        return
    dup = df.duplicated(list(keys), keep=False)
    if dup.any():
        raise RuntimeError(f"Duplicate {label} keys {list(keys)}: {int(dup.sum())} rows.")


def exact_oos_family_rho_set(
    oos: pd.DataFrame,
    *,
    family: str,
    evaluation: str,
    expected_rhos: Optional[Sequence[float]] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    rhos = expected_rhos if expected_rhos is not None else kwargs.get("expected_rhos")
    if rhos is None:
        raise TypeError("expected_rhos is required")
    sub = oos.loc[(oos["family"] == family) & (oos["evaluation"] == evaluation)].copy()
    got = sorted({float(x) for x in sub["rho"].dropna().tolist()})
    want = [float(x) for x in rhos]
    missing = [r for r in want if not any(np.isclose(r, g, atol=1e-12) for g in got)]
    extra = [g for g in got if not any(np.isclose(g, r, atol=1e-12) for r in want)]
    dups = int(sub.duplicated(["config_id"], keep=False).sum()) if "config_id" in sub.columns else 0
    return {
        "family": family,
        "evaluation": evaluation,
        "n_rows": int(len(sub)),
        "n_unique_rho": int(len(got)),
        "missing_rho": missing,
        "extra_rho": extra,
        "n_duplicate_config_rows": dups,
        "ok": (not missing) and (not extra) and dups == 0,
    }


def metric_value(df: pd.DataFrame, family: str, rho: Optional[float], metric: str, evaluation: str) -> float:
    if df is None or df.empty or metric not in df.columns:
        return float("nan")
    sub = df.loc[(df["family"] == family) & (df["evaluation"] == evaluation)]
    if rho is None:
        sub = sub.loc[sub["rho"].isna() | ~np.isfinite(pd.to_numeric(sub["rho"], errors="coerce"))]
    else:
        sub = sub.loc[np.isclose(pd.to_numeric(sub["rho"], errors="coerce"), float(rho), atol=1e-10, equal_nan=False)]
    if sub.empty:
        return float("nan")
    return float(pd.to_numeric(sub[metric], errors="coerce").iloc[0])


def build_combined_table(cv: pd.DataFrame, oos: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    records: List[Tuple[str, Optional[float], str]] = [("Linear", None, LINEAR), ("LightGBM", None, NATIVE)]
    for rho in expected_canonical_rhos():
        records.append(("Direct", float(rho), DIRECT))
        records.append(("Surrogate", float(rho), SURROGATE))
    freeze = load_json(ROOT / "frozen_baseline.json")
    spec = load_json(ROOT / "experiment_spec.json")
    completion = load_json(ROOT / "cv_completion.json")
    analysis = git_state()
    rows: List[Dict[str, Any]] = []
    for family, rho, model_name in records:
        row: Dict[str, Any] = {
            "family": family,
            "rho": np.nan if rho is None else float(rho),
            "model_name": model_name,
            "config_id": "",
            "data_id": spec.get("data_id") or completion.get("data_id"),
            "split_id": spec.get("split_id") or completion.get("split_id"),
            "source_fit_git_commit": spec.get("git_commit") or freeze.get("git_commit") or completion.get("git_commit"),
            "analysis_git_commit": analysis.get("git_commit"),
            "baseline_hash": freeze.get("lgbm_params_sha256") or spec.get("frozen_baseline_hash"),
            "grid_hash": spec.get("canonical_model_grid_hash") or completion.get("model_grid_hash"),
        }
        src_cv = cv.loc[cv["family"] == family].copy() if not cv.empty else pd.DataFrame()
        if rho is not None and not src_cv.empty and "rho" in src_cv.columns:
            src_cv = src_cv.loc[np.isclose(src_cv["rho"].astype(float), float(rho), atol=1e-10)]
        if not src_cv.empty and "config_id" in src_cv.columns:
            row["config_id"] = str(src_cv["config_id"].iloc[0])
        oos_src = oos if family in {"Direct", "Surrogate"} else baseline
        for metric in PATH_METRICS:
            fold_vals: List[float] = []
            if not src_cv.empty and metric in src_cv.columns and "fold_id" in src_cv.columns:
                for fold in range(7):
                    part = src_cv.loc[src_cv["fold_id"] == fold, metric]
                    val = float(pd.to_numeric(part, errors="coerce").iloc[0]) if not part.empty else float("nan")
                    row[f"{metric}__fold_{fold + 1}"] = val
                    if np.isfinite(val):
                        fold_vals.append(val)
            row[f"{metric}__CV_mean"] = float(np.mean(fold_vals)) if fold_vals else float("nan")
            row[f"{metric}__CV_sd"] = float(np.std(fold_vals, ddof=1)) if len(fold_vals) > 1 else float("nan")
            row[f"{metric}__heldout"] = metric_value(oos_src, family, rho, metric, "heldout")
            row[f"{metric}__forward_2025"] = metric_value(oos_src, family, rho, metric, "forward_2025")
        rows.append(row)
    return pd.DataFrame(rows)


def compute_rho0_control(native: pd.DataFrame, direct0: pd.DataFrame, surr0: pd.DataFrame) -> Dict[str, float]:
    def _col(df: pd.DataFrame) -> str:
        for c in ("y_pred_log", "yhat_log", "pred_log"):
            if c in df.columns:
                return c
        raise KeyError("no log-prediction column")

    def _id(df: pd.DataFrame) -> str:
        for c in ("row_id",):
            if c in df.columns:
                return c
        raise KeyError("no row id column")

    def _align(a: pd.DataFrame, b: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        left = a.set_index(_id(a))
        right = b.set_index(_id(b))
        idx = left.index.intersection(right.index)
        if len(idx) == 0:
            raise RuntimeError("rho=0 control could not align predictions on row_id.")
        return left.loc[idx, _col(a)].to_numpy(dtype=float), right.loc[idx, _col(b)].to_numpy(dtype=float)

    out: Dict[str, float] = {}
    for name, other in (("direct", direct0), ("surrogate", surr0)):
        nlog, olog = _align(native, other)
        delta = olog - nlog
        mean_abs = float(np.mean(np.abs(delta)))
        max_abs = float(np.max(np.abs(delta)))
        out[f"{name}_mean_abs_delta_log"] = mean_abs
        out[f"{name}_max_abs_delta_log"] = max_abs
        out[f"{name}_mean_abs_delta_log"] = mean_abs
        out[f"{name}_max_abs_delta_log"] = max_abs
    dlog, slog = _align(direct0, surr0)
    ds = slog - dlog
    out["direct_vs_surrogate_mean_abs_delta_log"] = float(np.mean(np.abs(ds)))
    out["direct_vs_surrogate_max_abs_delta_log"] = float(np.max(np.abs(ds)))
    out["direct_vs_surrogate_mean_abs_delta_log"] = out["direct_vs_surrogate_mean_abs_delta_log"]
    out["direct_vs_surrogate_max_abs_delta_log"] = out["direct_vs_surrogate_max_abs_delta_log"]
    return out


def replace_latex_environment_by_label(tex: str, env: str, label: str, new_env: str) -> str:
    needle = r"\label{" + label + "}"
    pos = tex.find(needle)
    if pos < 0:
        raise RuntimeError(f"LaTeX label {label} not found.")
    begin = tex.rfind(r"\begin{" + env + "}", 0, pos)
    if begin < 0:
        raise RuntimeError(f"Could not find \\begin{{{env}}} before {label}.")
    end_token = r"\end{" + env + "}"
    end = tex.find(end_token, pos)
    if end < 0:
        raise RuntimeError(f"Could not find \\end{{{env}}} after {label}.")
    end += len(end_token)
    return tex[:begin] + new_env.strip() + tex[end:]


def plot_accuracy_equity_r2(oos: pd.DataFrame, ymetrics: Sequence[str], stem: str) -> Path:
    dest = Path(globals().get("FIG_OUT", globals().get("FIG_OUT", FIG_OUT)))
    dest.mkdir(parents=True, exist_ok=True)
    xcol = "R2_price" if "R2_price" in oos.columns else "R2_price"
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.0), constrained_layout=True)
    evals = ("heldout", "forward_2025")
    n_drawn = 0
    for r, ev in enumerate(evals):
        for c, metric in enumerate(list(ymetrics)[:2]):
            ax = axes[r, c]
            for fam, color in (("Direct", DIRECT_COLOR), ("Surrogate", SURR_COLOR)):
                part = oos.loc[(oos["family"] == fam) & (oos["evaluation"] == ev)].sort_values("rho")
                if part.empty or metric not in part.columns or xcol not in part.columns:
                    continue
                ax.plot(part[xcol], part[metric], color=color, marker="o", ms=3, lw=1.2, label=fam)
                n_drawn += int(part[xcol].notna().sum() > 0)
            ax.set_xlabel(r"$R^2_P$")
            ax.set_ylabel(metric)
            ax.set_title(f"{ev}: R2 vs {metric}")
    if n_drawn == 0:
        raise RuntimeError("accuracy-equity figure has no R2-versus-metric curves")
    pdf = dest / f"{stem}.pdf"
    fig.savefig(pdf)
    plt.close(fig)
    meta = {"x": "R2_price", "y": list(ymetrics)[:2], "vs_rho": False, "vs_rho": False}
    write_json(dest / f"{stem}.meta.json", meta)
    return pdf


def load_oos_shards() -> pd.DataFrame:
    frames = []
    mapping = {"heldout": "test_run_metrics", "forward_2025": "assess_run_metrics"}
    for eval_name, shard in mapping.items():
        files = list(PREVIEW_ROOT.glob(f"{eval_name}/**/{shard}/*.parquet"))
        if not files:
            continue
        df = standardize_metrics(concat_parquets(files))
        df["evaluation"] = eval_name
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_cv_runs() -> pd.DataFrame:
    files = list(ROOT.glob("runs/**/*.parquet"))
    if not files:
        return pd.DataFrame()
    df = standardize_metrics(concat_parquets(files))
    df["evaluation"] = "cv_fold"
    if "fold_id" in df.columns:
        df["fold_id"] = pd.to_numeric(df["fold_id"], errors="coerce").astype("Int64")
    return df


def load_baseline_metrics() -> pd.DataFrame:
    rows = []
    for split, fname in (("heldout", "test_metrics.csv"), ("forward_2025", "assess_metrics.csv")):
        for path in BASELINE_ROOT.glob(f"analysis/**/{fname}"):
            df = standardize_metrics(pd.read_csv(path))
            df["evaluation"] = split
            rows.append(df)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def require_oos_complete(oos: pd.DataFrame) -> None:
    rhos = expected_canonical_rhos()
    problems = []
    for fam in ("Direct", "Surrogate"):
        for ev in ("heldout", "forward_2025"):
            chk = exact_oos_family_rho_set(oos, family=fam, evaluation=ev, expected_rhos=rhos)
            if not chk["ok"]:
                problems.append(chk)
    if problems:
        raise RuntimeError("OOS exact-set QA failed: " + json.dumps(problems, default=str)[:2000])


def cmd_cv_qa() -> int:
    cv = load_cv_runs()
    n = 0 if cv.empty else int(cv[["config_id", "fold_id"]].drop_duplicates().shape[0]) if {"config_id", "fold_id"} <= set(cv.columns) else len(cv)
    payload = {"selection_performed": False, "n_completed_pairs": n, "n_expected_pairs": 728, "ok": n == 728}
    write_json(MANIFEST_ROOT / "cv_qa.json", payload)
    print(json.dumps(payload, indent=2))
    if n != 728:
        print("FAIL CV QA")
        return 1
    print("PASS CV QA")
    return 0


def cmd_preview() -> int:
    oos = load_oos_shards()
    if oos.empty:
        print("FAIL preview: no OOS shards")
        return 1
    require_oos_complete(oos)
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    oos.to_csv(ANALYSIS_ROOT / "oos_preview_metrics.csv", index=False)
    plot_accuracy_equity_r2(oos, ["PRB", "MKI"], "prb_mki_accuracy_equity")
    print("PASS preview")
    return 0


def cmd_merge() -> int:
    cv = load_cv_runs()
    oos = load_oos_shards()
    baseline = load_baseline_metrics()
    require_oos_complete(oos)
    combined = build_combined_table(cv, oos, baseline)
    dict_cols = [c for c in combined.columns if combined[c].map(lambda x: isinstance(x, dict)).any()]
    if dict_cols:
        raise RuntimeError(f"combined table still has dict-valued columns: {dict_cols}")
    ANALYSIS_ROOT.mkdir(parents=True, exist_ok=True)
    combined.to_csv(ANALYSIS_ROOT / "combined_path_table.csv", index=False)
    combined.to_parquet(ANALYSIS_ROOT / "combined_path_table.parquet", index=False)
    plot_accuracy_equity_r2(oos, ["PRB", "MKI"], "prb_mki_accuracy_equity")
    write_json(MANIFEST_ROOT / "merge.json", {"n_rows": int(len(combined)), "selection_performed": False})
    print(f"PASS merge n_rows={len(combined)}")
    return 0


def cmd_populate() -> int:
    configure_paths(os.environ.get("P6_RESULT_ROOT", "output/paper_v6_preselection_994"))
    if "paper_v6_preselection_994" not in str(ROOT):
        raise RuntimeError("Refusing to populate the manuscript from the 500-tree root after ADOPT_994.")
    gate = load_json(ROOT / "baseline_gate.json")
    if str(gate.get("decision")) != "ADOPT_994":
        raise RuntimeError("baseline_gate.json is not ADOPT_994.")
    from populate_paper_v6_994 import populate

    return populate()


def _live_tex(text: str) -> str:
    lines = []
    for line in text.splitlines():
        if line.lstrip().startswith("%"):
            continue
        lines.append(re.sub(r"(?<!\\)%.*", "", line))
    return "\n".join(lines)


def cmd_final_qa() -> int:
    configure_paths(os.environ.get("P6_RESULT_ROOT", "output/paper_v6_preselection_994"))
    problems = []
    if "paper_v6_preselection_994" not in str(ROOT):
        problems.append("final QA must run from the 994 result root")
    tex = PAPER_TEX.read_text(encoding="utf-8") if PAPER_TEX.is_file() else ""
    live = _live_tex(tex)
    for phrase in FORBIDDEN_PHRASES:
        if phrase in live:
            problems.append(f"forbidden phrase: {phrase}")
    for stale in ("344,610", "382,900", "populate from full", "PLACEHOLDER", "results_reference_assets"):
        if stale in live:
            problems.append(f"stale live text: {stale}")
    payload = {"problems": problems, "selection_performed": False}
    write_json(MANIFEST_ROOT / "final_qa.json", payload)
    print(json.dumps(payload, indent=2))
    if problems:
        print("FAIL final QA")
        return 1
    print("PASS final QA")
    return 0


# Names used by focused reporting tests and the 994 populate script.
assert_no_duplicate_pairs = assert_no_duplicate_pairs
exact_oos_family_rho_set = exact_oos_family_rho_set
nearest_anchor = nearest_anchor
map_display_anchors = map_display_anchors
join_prediction_metadata = join_prediction_metadata
compute_rho0_control = compute_rho0_control
replace_latex_environment_by_label = replace_latex_environment_by_label
plot_accuracy_equity_r2 = plot_accuracy_equity_r2
expected_canonical_rhos = expected_canonical_rhos
FORBIDDEN_PHRASES = FORBIDDEN_PHRASES
load_json = load_json
write_json = write_json
load_json = load_json
write_json = write_json
FIG_OUT = FIG_OUT
FIG_OUT = FIG_OUT
configure_paths = configure_paths
ROOT = ROOT
DIRECT_COLOR = DIRECT_COLOR
SURR_COLOR = SURR_COLOR
LINEAR_COLOR = LINEAR_COLOR
NATIVE_COLOR = NATIVE_COLOR
join_prediction_metadata = join_prediction_metadata
exact_oos_family_rho_set = exact_oos_family_rho_set
assert_no_duplicate_pairs = assert_no_duplicate_pairs
nearest_anchor = nearest_anchor
map_display_anchors = map_display_anchors
compute_rho0_control = compute_rho0_control
replace_latex_environment_by_label = replace_latex_environment_by_label
plot_accuracy_equity_r2 = plot_accuracy_equity_r2
expected_canonical_rhos = expected_canonical_rhos


def main() -> int:
    configure_paths()
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["cv-qa", "preview", "merge", "populate", "final-qa"])
    args = p.parse_args()
    return {
        "cv-qa": cmd_cv_qa,
        "preview": cmd_preview,
        "merge": cmd_merge,
        "populate": cmd_populate,
        "final-qa": cmd_final_qa,
    }[args.command]()


if __name__ == "__main__":
    raise SystemExit(main())
