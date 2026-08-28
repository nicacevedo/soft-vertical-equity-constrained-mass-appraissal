#!/usr/bin/env python3
"""Canonical 994-tree paper-asset follow-up on the frozen transition-region result.

Read-only post-processing. No fitting, no protocol change, no .tex, no paper/ writes,
and no mutation of transition_regions_v1.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import struct
import subprocess
import sys
import traceback
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

plt = None  # lazy, render only

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from canonical_experiment import git_state, package_versions
from utils.transition_paper_assets import (
    ANCHOR_METRIC_SPECS,
    ENVELOPE_METRICS_PAPER,
    IAAO_PRB_RANGE,
    IAAO_PRD_RANGE,
    IAAO_VEI_RANGE,
    classify_direct_interpretation,
    endpoint_equals_first_positive,
    endpoint_equals_last_positive,
    event_sharpness_row,
    frozen_direct_span,
    hash_v1_inputs,
    load_v1_tables,
    lofo_envelope,
    manuscript_format_flags,
    metric_series,
    positive_display_anchors,
    ratio_shape_anchors,
    rho_inside_lofo_envelope,
    span_regret_row,
)
from utils.transition_regions import (
    ATOL,
    CANONICAL_ROOT_NAME,
    EXPECTED_IDENTITY,
    FAMILY_DISPLAY,
    FOLD_IDS,
    HISTORICAL_500_ROOT_NAME,
    PRIMARY_METRICS,
    CanonicalIdentityError,
    OutputConfineError,
    OutputGuard,
    assert_not_historical_500,
    df_to_markdown,
    expected_canonical_rhos,
    family_frame,
    is_rho_positive,
    load_combined_path_table,
    numerically_equal,
    protocol_sha256,
    rho_in_closed_span,
    sha256_file,
    validate_canonical_result_root,
    validate_combined_counts,
)

DIRECT_COLOR = "#1D4ED8"
SURR_COLOR = "#C2410C"
NATIVE_COLOR = "#111827"
LINEAR_COLOR = "#6B7280"
FAMILY_COLOR = {"Direct": DIRECT_COLOR, "Surrogate": SURR_COLOR}
SPAN_FACE = "#9CA3AF"
LINEAR_NAME = "LinearRegression"
NATIVE_NAME = "LGBMRegressor"

SUBDIRS = (
    "tables",
    "figures/main",
    "figures/appendix",
    "figures/diagnostic",
    "report",
    "qa",
    "provenance",
    "logs",
    "cluster",
    "code_snapshot",
)
CODE_SNAPSHOT_FILES = (
    Path("utils/transition_regions.py"),
    Path("utils/transition_paper_assets.py"),
    Path("scripts/analyze_transition_paper_assets.py"),
    Path("tests/test_transition_paper_assets.py"),
    Path("scripts/submit_transition_paper_assets_v1.sh"),
    Path("scripts/run_transition_paper_assets_stage.sbatch"),
    Path("utils/transition_paper_assets_report.py"),
    Path("utils/transition_paper_asset_plots.py"),
)
FORBIDDEN_PHRASES = (
    "optimal rho",
    "recommended rho",
    "selected rho",
    "sweet spot",
    "safe range",
    "operating range",
    "deployment-ready",
    "preferred rho",
    "recommended region",
    "optimal region",
    "recommended span",
    "optimal span",
)
CANONICAL_INPUT_RELS = (
    "baseline_gate.json",
    "experiment_manifest.json",
    "lgbm_config.json",
    "frozen_baseline.json",
    "cv_completion.json",
    "analysis/combined_path_table.csv",
)
MAIN_FIGURES = (
    "figures/main/baseline_models_motivation_2024_2025.pdf",
    "figures/main/baseline_models_motivation_2024_2025.png",
    "figures/main/ratio_shape_evolution.pdf",
    "figures/main/ratio_shape_evolution.png",
    "figures/main/mechanism_vs_rho.pdf",
    "figures/main/mechanism_vs_rho.png",
    "figures/main/accuracy_equity_trajectories_inprocessing_only.pdf",
    "figures/main/accuracy_equity_trajectories_inprocessing_only.png",
)
APPENDIX_FIGURES = (
    "figures/appendix/paper_transition_event_locations.pdf",
    "figures/appendix/paper_transition_event_locations.png",
    "figures/appendix/paper_primary_paths_with_events.pdf",
    "figures/appendix/paper_primary_paths_with_events.png",
    "figures/appendix/prb_mki_accuracy_equity_inprocessing_only.pdf",
    "figures/appendix/prb_mki_accuracy_equity_inprocessing_only.png",
    "figures/appendix/predictive_metric_paths.pdf",
    "figures/appendix/predictive_metric_paths.png",
    "figures/appendix/level_uniformity_paths.pdf",
    "figures/appendix/level_uniformity_paths.png",
    "figures/appendix/cv_fold_stability.pdf",
    "figures/appendix/cv_fold_stability.png",
    "figures/appendix/vei_percentile_group_profile.pdf",
    "figures/appendix/vei_percentile_group_profile.png",
)
DIAGNOSTIC_FIGURES = (
    "figures/diagnostic/paper_direct_oos_span_regret.pdf",
    "figures/diagnostic/paper_direct_oos_span_regret.png",
)
REQUIRED_TABLES = (
    "tables/transition_oos_span_regret.csv",
    "tables/transition_oos_span_regret.parquet",
    "tables/transition_oos_span_regret.md",
    "tables/transition_event_sharpness.csv",
    "tables/transition_event_sharpness.parquet",
    "tables/paper_transition_summary.csv",
    "tables/paper_transition_summary.parquet",
    "tables/paper_transition_summary.md",
    "tables/paper_transition_event_detail.csv",
    "tables/paper_transition_event_detail.parquet",
    "tables/transition_mechanism_endpoint_compact.csv",
    "tables/transition_mechanism_endpoint_compact.parquet",
    "tables/transition_mechanism_endpoint_compact.md",
    "tables/transition_region_performance_envelope.csv",
    "tables/transition_region_performance_envelope.parquet",
    "tables/transition_region_performance_envelope.md",
    "tables/baseline_comparison_source.csv",
    "tables/baseline_comparison_source.parquet",
    "tables/path_anchor_summary_source.csv",
    "tables/path_anchor_summary_source.parquet",
)
PERCENT_PATH_METRICS = {"MAPE", "COV"}


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_mpl() -> None:
    global plt
    if plt is not None:
        return
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt_mod

    plt = plt_mod


def setup_style() -> None:
    _load_mpl()
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 8.5,
            "legend.fontsize": 7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="994-tree paper-asset follow-up.")
    p.add_argument("--stage", required=True, choices=("preflight", "analysis", "render", "audit"))
    p.add_argument("--result-root", required=True, type=str)
    p.add_argument("--output-root", default=None, type=str)
    p.add_argument("--v1-root", default=None, type=str)
    p.add_argument("--repo-root", default=str(REPO), type=str)
    return p.parse_args()


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, Path, OutputGuard]:
    repo = Path(args.repo_root).resolve()
    result_root = Path(args.result_root)
    result_root = result_root.resolve() if result_root.is_absolute() else (repo / result_root).resolve()
    identity = validate_canonical_result_root(result_root)
    assert_not_historical_500(result_root)
    if HISTORICAL_500_ROOT_NAME in result_root.parts and CANONICAL_ROOT_NAME not in result_root.parts:
        raise CanonicalIdentityError("historical 500-tree root is forbidden")
    if args.v1_root:
        v1 = Path(args.v1_root)
        v1 = v1.resolve() if v1.is_absolute() else (repo / v1).resolve()
    else:
        v1 = (
            result_root
            / "analysis"
            / f"data_id={identity['data_id']}"
            / f"split_id={identity['split_id']}"
            / "penalty_path_analysis"
            / "transition_regions_v1"
        )
    if args.output_root:
        output_root = Path(args.output_root)
        output_root = output_root.resolve() if output_root.is_absolute() else (repo / output_root).resolve()
    else:
        output_root = v1.parent / "transition_regions_paper_assets_v1"
    try:
        output_root.relative_to(result_root)
    except ValueError as err:
        raise OutputConfineError(f"output root must live under result root: {output_root}") from err
    if output_root.name != "transition_regions_paper_assets_v1":
        raise OutputConfineError(f"output root must end in transition_regions_paper_assets_v1, got {output_root}")
    if output_root.resolve() == v1.resolve():
        raise OutputConfineError("refusing to write into transition_regions_v1")
    output_root.mkdir(parents=True, exist_ok=True)
    guard = OutputGuard(output_root, repo)
    for name in SUBDIRS:
        guard.ensure_subdir(name)
    return repo, result_root, v1, output_root, guard


def iter_tex_files(repo: Path) -> List[Path]:
    files = []
    for p in repo.rglob("*.tex"):
        if not p.is_file():
            continue
        if "transition_regions_v1" in p.parts or "transition_regions_paper_assets_v1" in p.parts:
            continue
        files.append(p)
    return sorted(files)


def env_snapshot(repo: Path) -> Dict[str, Any]:
    versions = package_versions()
    try:
        from importlib.metadata import version as pkg_version
    except Exception:  # pragma: no cover
        from importlib_metadata import version as pkg_version  # type: ignore
    for name in ("matplotlib", "pyarrow"):
        try:
            versions[name] = pkg_version(name)
        except Exception:
            versions[name] = "unknown"
    return {
        "git": git_state(repo),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "packages": versions,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
        "utc": utc_now(),
    }


def snapshot_code(repo: Path, guard: OutputGuard) -> Dict[str, str]:
    hashes = {}
    dest_dir = guard.ensure_subdir("code_snapshot")
    for rel in CODE_SNAPSHOT_FILES:
        src = repo / rel
        if not src.is_file():
            continue
        dest = dest_dir / rel.name
        guard.write_bytes(dest, src.read_bytes())
        hashes[rel.as_posix()] = sha256_file(src)
    return hashes


def git_ls_used(repo: Path, rels: Sequence[Path]) -> Dict[str, str]:
    out = {}
    for rel in rels:
        src = repo / rel
        tracked = subprocess.run(
            ["git", "ls-files", "--error-unmatch", rel.as_posix()],
            cwd=str(repo),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if tracked.returncode == 0 and src.is_file():
            out[rel.as_posix()] = "tracked"
        elif src.is_file():
            out[rel.as_posix()] = "untracked"
        else:
            out[rel.as_posix()] = "missing"
    return out


def combined_row(combined: pd.DataFrame, family: str, rho: Optional[float] = None) -> pd.Series:
    sub = combined.loc[combined["family"] == family]
    if rho is None:
        sub = sub.loc[sub["rho"].isna() | ~np.isfinite(pd.to_numeric(sub["rho"], errors="coerce"))]
    else:
        sub = sub.loc[np.isclose(pd.to_numeric(sub["rho"], errors="coerce"), float(rho), atol=1e-10)]
    if sub.empty:
        raise RuntimeError(f"missing combined row family={family} rho={rho}")
    return sub.iloc[0]


def metric_val(row: pd.Series, name: str, split: str) -> float:
    return float(row[f"{name}__{split}"])


def rho_x(rho: np.ndarray) -> np.ndarray:
    x = np.asarray(rho, dtype=float)
    return np.where(x <= 0, 0.055, x)


def padded_lim(values, *, pad: float = 0.08, include: Sequence[float] = ()) -> Tuple[float, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    extra = np.asarray(list(include), dtype=float)
    extra = extra[np.isfinite(extra)]
    if extra.size:
        arr = np.concatenate([arr, extra]) if arr.size else extra
    if arr.size == 0:
        return (0.0, 1.0)
    lo, hi = float(np.min(arr)), float(np.max(arr))
    span = hi - lo
    if span <= 0:
        span = max(abs(hi), 0.05)
    return lo - pad * span, hi + pad * span


def save_figure(fig: Any, stem: Path, guard: OutputGuard) -> List[str]:
    pdf = guard.allowed(stem.with_suffix(".pdf"))
    png = guard.allowed(stem.with_suffix(".png"))
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    fig.savefig(png, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    return [str(pdf), str(png)]


def write_table(df: pd.DataFrame, stem: Path, guard: OutputGuard, *, markdown: bool = False) -> List[str]:
    written = [str(p) for p in guard.write_df(df, stem, parquet=True)]
    if markdown:
        md = "# " + stem.stem.replace("_", " ") + "\n\n" + df_to_markdown(df) + "\n"
        md_path = guard.write_text(stem.with_suffix(".md"), md)
        written.append(str(md_path))
    return written


def lookup_event(events: pd.DataFrame, family: str, metric: str, split: str = "cv_mean") -> pd.Series:
    sub = events.loc[(events["family"] == family) & (events["metric"] == metric) & (events["split"] == split)]
    if sub.empty:
        raise RuntimeError(f"missing v1 event {family} {split} {metric}")
    return sub.iloc[0]


def n_positive_in_span(rhos: Sequence[float], low: float, high: float) -> int:
    return int(sum(1 for r in rhos if is_rho_positive(float(r)) and rho_in_closed_span(float(r), low, high)))


def fold_median_iqr(values: Sequence[float]) -> Tuple[Optional[float], Optional[float]]:
    arr = np.asarray([float(x) for x in values if np.isfinite(float(x))], dtype=float)
    if arr.size == 0:
        return None, None
    q25, q50, q75 = np.percentile(arr, [25, 50, 75])
    return float(q50), float(q75 - q25)


def stage_preflight(repo: Path, result_root: Path, v1: Path, output_root: Path, guard: OutputGuard) -> Dict[str, Any]:
    identity = validate_canonical_result_root(result_root)
    assert_not_historical_500(result_root)
    final_path = v1 / "qa" / "FINAL_STATUS.json"
    if not final_path.is_file():
        raise RuntimeError(f"missing v1 FINAL_STATUS.json: {final_path}")
    v1_final = json.loads(final_path.read_text(encoding="utf-8"))
    if str(v1_final.get("status")) != "PASS":
        raise RuntimeError(f"v1 FINAL_STATUS is not PASS: {v1_final.get('status')}")
    protocol_path = v1 / "protocol" / "transition_analysis_protocol.json"
    if not protocol_path.is_file():
        raise RuntimeError("missing frozen v1 protocol")
    v1_protocol_sha = sha256_file(protocol_path)
    code_protocol_sha = protocol_sha256()
    if v1_protocol_sha != code_protocol_sha:
        raise RuntimeError(
            "v1 protocol hash does not match current code protocol; refusing to change the protocol. "
            f"v1={v1_protocol_sha} code={code_protocol_sha}"
        )
    combined_path = load_combined_path_table(result_root)
    combined = pd.read_csv(combined_path)
    expected_rhos = expected_canonical_rhos()
    count_chk = validate_combined_counts(combined, expected_rhos)
    problems = list(count_chk["problems"])
    v1_hashes = hash_v1_inputs(v1, result_root)
    input_sha = {rel: sha256_file(result_root / rel) if (result_root / rel).is_file() else None for rel in CANONICAL_INPUT_RELS}
    git = git_state(repo)
    porcelain = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(repo), text=True)
    ls_files = git_ls_used(repo, CODE_SNAPSHOT_FILES)
    snapshot_code(repo, guard)
    tex_base = {str(p.relative_to(repo)): sha256_file(p) for p in iter_tex_files(repo)}
    paper_base = {str(p.relative_to(repo)): sha256_file(p) for p in (repo / "paper").rglob("*") if p.is_file()}
    v1_file_hashes = {str(p): sha256_file(p) for p in sorted(v1.rglob("*")) if p.is_file()}
    payload = {
        "status": "PASS" if not problems else "FAIL",
        "utc": utc_now(),
        "identity": identity,
        "expected_identity": dict(EXPECTED_IDENTITY) if False else {
            "result_root_name": CANONICAL_ROOT_NAME,
            "baseline_gate": identity.get("baseline_gate"),
            "lgbm_config_id": identity.get("lgbm_config_id"),
            "split_id": identity.get("split_id"),
            "data_id": identity.get("data_id"),
        },
        "git_head": git.get("git_commit"),
        "git_branch": git.get("git_branch"),
        "git_dirty": git.get("git_dirty"),
        "git_status_porcelain": porcelain,
        "git_ls_files_used_scripts": ls_files,
        "v1_final_status": v1_final.get("status"),
        "v1_final_sha256": v1_hashes[str(final_path)],
        "v1_protocol_sha256": v1_protocol_sha,
        "code_protocol_sha256": code_protocol_sha,
        "combined_path_sha256": v1_hashes[str(combined_path.resolve())] if str(combined_path.resolve()) in v1_hashes else sha256_file(combined_path),
        "v1_input_sha256": v1_hashes,
        "canonical_input_sha256": input_sha,
        "historical_500_root_read": False,
        "model_fitting": False,
        "tex_will_be_generated": False,
        "no_v1_mutation": True,
        "problems": problems,
        "env": env_snapshot(repo),
        "count_check": count_chk,
    }
    guard.write_json(output_root / "provenance" / "preflight.json", payload)
    guard.write_json(output_root / "provenance" / "v1_immutable_hashes.json", v1_file_hashes)
    guard.write_json(output_root / "provenance" / "tex_baseline_sha256.json", tex_base)
    guard.write_json(output_root / "provenance" / "paper_tree_sha256.json", paper_base)
    guard.write_json(output_root / "qa" / "PREFLIGHT_STATUS.json", {"status": payload["status"], "utc": payload["utc"], "problems": problems})
    if problems:
        raise RuntimeError("PREFLIGHT failed: " + "; ".join(problems))
    return payload


def _path_for_split(fam_df: pd.DataFrame, metric: str, split: str) -> Tuple[np.ndarray, np.ndarray]:
    return metric_series(fam_df, metric, split)


def build_regret(combined: pd.DataFrame, v1_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    span_df = v1_tables["transition_span_summary.csv"]
    events_cv = v1_tables["transition_events_cv_mean.csv"]
    conc = v1_tables["transition_temporal_concordance.csv"]
    low, high, _status = frozen_direct_span(span_df)
    fam = family_frame(combined, "Direct")
    rows = []
    for split in ("heldout", "forward_2025"):
        for metric, direction in PRIMARY_METRICS:
            rhos, vals = _path_for_split(fam, metric, split)
            ev = lookup_event(events_cv, "Direct", metric)
            classif = None
            part = conc.loc[(conc["family"] == "Direct") & (conc["split"] == split) & (conc["metric"] == metric)]
            if not part.empty:
                classif = str(part.iloc[0]["classification"])
            rec = span_regret_row(
                rhos, vals, family="Direct", split=split, metric=metric, direction=direction,
                rho_low=low, rho_high=high, event_classification=classif or str(ev["classification"]),
            )
            rec["copied_from_v1_span"] = True
            rec["v1_cv_event_rho"] = float(ev["rho_low"])
            rec["v1_cv_event_classification"] = str(ev["classification"])
            rows.append(rec)
    return pd.DataFrame(rows)


def build_sharpness(combined: pd.DataFrame) -> pd.DataFrame:
    splits = ["cv_mean"] + [f"fold_{k}" for k in FOLD_IDS] + ["heldout", "forward_2025"]
    rows = []
    for fam_name in FAMILY_DISPLAY:
        fam = family_frame(combined, fam_name)
        for split in splits:
            for metric, direction in PRIMARY_METRICS:
                rhos, vals = _path_for_split(fam, metric, split)
                rec = event_sharpness_row(
                    rhos, vals, family=fam_name, split=split, metric=metric, direction=direction
                )
                rows.append(rec)
    return pd.DataFrame(rows)


def build_summary(combined: pd.DataFrame, v1_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    span_df = v1_tables["transition_span_summary.csv"]
    events = v1_tables["transition_events_cv_mean.csv"]
    lofo = v1_tables["transition_lofo_sensitivity.csv"]
    conc = v1_tables["transition_temporal_concordance.csv"]
    min_pos = float(EXPECTED_IDENTITY["min_positive_rho"])
    max_pos = float(EXPECTED_IDENTITY["max_positive_rho"])
    rows = []
    for fam in FAMILY_DISPLAY:
        srow = span_df.loc[span_df["family"] == fam].iloc[0]
        fam_df = family_frame(combined, fam)
        rec: Dict[str, Any] = {
            "family": fam,
            "common_span_status": str(srow["status"]),
            "span_low": srow["rho_transition_low"] if pd.notna(srow["rho_transition_low"]) else None,
            "span_high": srow["rho_transition_high"] if pd.notna(srow["rho_transition_high"]) else None,
            "log10_width": srow["log10_width"] if pd.notna(srow["log10_width"]) else None,
            "fraction_of_full_positive_log_grid": srow["fraction_of_full_positive_log_grid"]
            if pd.notna(srow["fraction_of_full_positive_log_grid"]) else None,
            "copied_from_v1": True,
        }
        for metric, _d in PRIMARY_METRICS:
            ev = lookup_event(events, fam, metric)
            rec[f"{metric}__cv_rho"] = float(ev["rho_low"]) if pd.notna(ev["rho_low"]) else None
            rec[f"{metric}__cv_classification"] = str(ev["classification"])
            rec[f"{metric}__cv_value"] = float(ev["metric_value"]) if pd.notna(ev["metric_value"]) else None
        low = rec["span_low"]
        high = rec["span_high"]
        rec["lower_endpoint_equals_first_positive_grid"] = endpoint_equals_first_positive(
            None if low is None else float(low), min_pos
        )
        rec["upper_endpoint_equals_last_positive_grid"] = endpoint_equals_last_positive(
            None if high is None else float(high), max_pos
        )
        if rec["common_span_status"] == "VALID_POSITIVE_INTERIOR_SPAN" and low is not None:
            rec["n_positive_rho_in_span"] = n_positive_in_span(fam_df["rho"].to_numpy(dtype=float), float(low), float(high))
        else:
            rec["n_positive_rho_in_span"] = 0
        env = lofo_envelope(lofo, fam)
        rec["lofo_valid_count"] = env["n_valid"]
        rec["lofo_valid_of"] = 7
        rec["lofo_valid_low_min"] = env["valid_low_min"]
        rec["lofo_valid_low_max"] = env["valid_low_max"]
        rec["lofo_valid_high_min"] = env["valid_high_min"]
        rec["lofo_valid_high_max"] = env["valid_high_max"]
        for split, key in (("heldout", "heldout_exact_concordance"), ("forward_2025", "forward_2025_exact_concordance")):
            part = conc.loc[(conc["family"] == fam) & (conc["split"] == split)]
            if rec["common_span_status"] == "VALID_POSITIVE_INTERIOR_SPAN":
                n_in = int(part["inside_frozen_cv_span"].astype(bool).sum()) if "inside_frozen_cv_span" in part.columns else 0
            else:
                n_in = 0
            rec[key] = n_in
            rec[f"{key}_of"] = 5
        rec["lower_endpoint_is_not_an_estimated_threshold"] = bool(rec["lower_endpoint_equals_first_positive_grid"])
        rec["not_a_selected_rho"] = True
        rows.append(rec)
    return pd.DataFrame(rows)


def build_event_detail(v1_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    events = v1_tables["transition_events_cv_mean.csv"]
    conc = v1_tables["transition_temporal_concordance.csv"]
    lofo = v1_tables["transition_lofo_sensitivity.csv"]
    env = lofo_envelope(lofo, "Direct")
    rows = []
    for fam in FAMILY_DISPLAY:
        for metric, direction in PRIMARY_METRICS:
            cv = lookup_event(events, fam, metric)
            rec: Dict[str, Any] = {
                "family": fam,
                "metric": metric,
                "direction": direction,
                "cv_rho": cv["rho_low"],
                "cv_value": cv["metric_value"],
                "cv_classification": cv["classification"],
            }
            for split, prefix in (("heldout", "heldout"), ("forward_2025", "forward_2025")):
                part = conc.loc[(conc["family"] == fam) & (conc["split"] == split) & (conc["metric"] == metric)]
                if part.empty:
                    continue
                row = part.iloc[0]
                rec[f"{prefix}_rho"] = row["rho_low"]
                rec[f"{prefix}_value"] = row["metric_value"]
                rec[f"{prefix}_classification"] = row["classification"]
                rec[f"{prefix}_inside_cv_span"] = row["inside_frozen_cv_span"] if pd.notna(row["inside_frozen_cv_span"]) else None
                rec[f"{prefix}_log10_distance_from_span"] = row["log10_distance_to_nearest_cv_span_boundary"]
                if fam == "Direct":
                    rec[f"{prefix}_inside_lofo_endpoint_envelope"] = rho_inside_lofo_envelope(
                        None if pd.isna(row["rho_low"]) else float(row["rho_low"]), env
                    )
                    rec["lofo_envelope_is_sensitivity_only"] = True
                    rec["lofo_envelope_does_not_replace_frozen_cv_span"] = True
            rows.append(rec)
    return pd.DataFrame(rows)


def build_mechanism_compact(v1_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    mech = v1_tables["transition_mechanism_summary.csv"]
    events = v1_tables["transition_events_cv_mean.csv"]
    rows = []
    for location in ("span_low", "span_high"):
        cv = mech.loc[(mech["family"] == "Direct") & (mech["location"] == location) & (mech["split"] == "CV_mean")]
        if cv.empty:
            continue
        r = cv.iloc[0]
        fold_qs = [float(r[f"q_beta_fold_{k}"]) for k in FOLD_IDS if pd.notna(r.get(f"q_beta_fold_{k}"))]
        med, iqr = fold_median_iqr(fold_qs)
        rec = {
            "family": "Direct",
            "location": location,
            "rho": float(r["rho"]),
            "q_beta": float(r["q_beta"]) if pd.notna(r["q_beta"]) else None,
            "attenuation_1_minus_q_beta": float(r["attenuation_1_minus_q_beta"]) if pd.notna(r["attenuation_1_minus_q_beta"]) else None,
            "q_cov": float(r["q_cov"]) if pd.notna(r["q_cov"]) else None,
            "q_beta_q_cov_agree": bool(r["q_beta_q_cov_agree"]) if pd.notna(r["q_beta_q_cov_agree"]) else None,
            "fold_median_q_beta": med,
            "fold_iqr_q_beta": iqr,
            "overcorrection": bool(r["overcorrection"]),
            "interpretive_only": True,
            "theory_does_not_define_endpoints": True,
        }
        for split, key in (("heldout", "heldout_q_beta"), ("forward_2025", "forward_2025_q_beta")):
            part = mech.loc[(mech["family"] == "Direct") & (mech["location"] == location) & (mech["split"] == split)]
            rec[key] = float(part.iloc[0]["q_beta"]) if not part.empty and pd.notna(part.iloc[0]["q_beta"]) else None
        rows.append(rec)
    for metric, _d in PRIMARY_METRICS:
        ev = lookup_event(events, "Surrogate", metric)
        loc = f"cv_event_{metric}_low"
        cv = mech.loc[(mech["family"] == "Surrogate") & (mech["location"] == loc) & (mech["split"] == "CV_mean")]
        if cv.empty:
            continue
        r = cv.iloc[0]
        rec = {
            "family": "Surrogate",
            "location": loc,
            "rho": float(r["rho"]),
            "q_beta": float(r["q_beta"]) if pd.notna(r["q_beta"]) else None,
            "attenuation_1_minus_q_beta": float(r["attenuation_1_minus_q_beta"]) if pd.notna(r["attenuation_1_minus_q_beta"]) else None,
            "q_cov": float(r["q_cov"]) if pd.notna(r["q_cov"]) else None,
            "q_beta_q_cov_agree": bool(r["q_beta_q_cov_agree"]) if pd.notna(r["q_beta_q_cov_agree"]) else None,
            "fold_median_q_beta": None,
            "fold_iqr_q_beta": None,
            "overcorrection": bool(r["overcorrection"]),
            "empirical_q_only": True,
            "no_analytical_direct_style_q_formula": True,
            "interpretive_only": True,
            "theory_does_not_define_endpoints": True,
            "cv_metric": metric,
            "cv_classification": str(ev["classification"]),
        }
        for split, key in (("heldout", "heldout_q_beta"), ("forward_2025", "forward_2025_q_beta")):
            part = mech.loc[(mech["family"] == "Surrogate") & (mech["location"] == loc) & (mech["split"] == split)]
            rec[key] = float(part.iloc[0]["q_beta"]) if not part.empty and pd.notna(part.iloc[0]["q_beta"]) else None
        rows.append(rec)
    return pd.DataFrame(rows)


def build_envelope(combined: pd.DataFrame, v1_tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    span_df = v1_tables["transition_span_summary.csv"]
    rows = []
    splits = (("CV_mean", "cv"), ("heldout", "heldout"), ("forward_2025", "forward_2025"))
    lin = combined_row(combined, "Linear")
    lgb = combined_row(combined, "LightGBM")
    for split_col, split_lab in splits:
        for fam, label, is_span in (
            ("Linear", "Linear regression", False),
            ("LightGBM", "Ordinary LightGBM", False),
            ("Direct", "Direct: all rho within CV transition span", True),
            ("Surrogate", "Surrogate: all rho within CV transition span", True),
        ):
            srow = span_df.loc[span_df["family"] == fam] if fam in FAMILY_DISPLAY else None
            status = str(srow.iloc[0]["status"]) if srow is not None and not srow.empty else ""
            rec: Dict[str, Any] = {
                "evaluation": split_lab,
                "row_group": label,
                "family": fam,
                "envelope_is_componentwise_across_span": bool(is_span),
                "not_a_selected_configuration": True,
                "extrema_need_not_share_one_rho": True,
                "cv_span_status": status,
            }
            if fam in {"Linear", "LightGBM"}:
                rec["n_configurations"] = 1
                for met in ENVELOPE_METRICS_PAPER:
                    col = f"{met}__{split_col}"
                    if col not in combined.columns:
                        rec[f"{met}__min"] = None
                        rec[f"{met}__max"] = None
                        continue
                    val = metric_val(combined_row(combined, fam), met, split_col) if f"{met}__{split_col}" in lin.index else None
                    try:
                        val = float(combined_row(combined, fam)[col])
                    except Exception:
                        val = None
                    rec[f"{met}__min"] = val
                    rec[f"{met}__max"] = val
                    rec[f"{met}__delta_vs_lgbm_min"] = None
                    rec[f"{met}__delta_vs_lgbm_max"] = None
            else:
                if status != "VALID_POSITIVE_INTERIOR_SPAN":
                    rec["n_configurations"] = 0
                    for met in ENVELOPE_METRICS_PAPER:
                        rec[f"{met}__min"] = None
                        rec[f"{met}__max"] = None
                else:
                    low = float(srow.iloc[0]["rho_transition_low"])
                    high = float(srow.iloc[0]["rho_transition_high"])
                    fam_df = family_frame(combined, fam)
                    mask = [rho_in_closed_span(float(r), low, high) for r in fam_df["rho"].to_numpy(dtype=float)]
                    sub = fam_df.loc[np.array(mask, dtype=bool)]
                    rec["n_configurations"] = int(len(sub))
                    rec["rho_transition_low"] = low
                    rec["rho_transition_high"] = high
                    for met in ENVELOPE_METRICS_PAPER:
                        col = f"{met}__{split_col}"
                        if col not in sub.columns:
                            rec[f"{met}__min"] = None
                            rec[f"{met}__max"] = None
                            rec[f"{met}__unavailable_note"] = "not_in_canonical_table_for_this_split"
                            continue
                        vals = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
                        finite = vals[np.isfinite(vals)]
                        if finite.size == 0:
                            rec[f"{met}__min"] = None
                            rec[f"{met}__max"] = None
                            rec[f"{met}__unavailable_note"] = "nonfinite"
                            continue
                        vmin, vmax = float(np.min(finite)), float(np.max(finite))
                        rec[f"{met}__min"] = vmin
                        rec[f"{met}__max"] = vmax
                        try:
                            lgbm_val = float(lgb[col])
                            rec[f"{met}__delta_vs_lgbm_min"] = vmin - lgbm_val
                            rec[f"{met}__delta_vs_lgbm_max"] = vmax - lgbm_val
                        except Exception:
                            rec[f"{met}__delta_vs_lgbm_min"] = None
                            rec[f"{met}__delta_vs_lgbm_max"] = None
            rows.append(rec)
    return pd.DataFrame(rows)


def build_baseline_source(combined: pd.DataFrame) -> pd.DataFrame:
    measures = [
        ("R2_price", True, None),
        ("MAE_price", False, None),
        ("MAPE", False, None),
        ("RMSE_log", False, None),
        ("median_ratio", None, 1.0),
        ("mean_ratio", None, 1.0),
        ("weighted_mean_ratio", None, 1.0),
        ("COD", False, None),
        ("COV", False, None),
        ("PRD", None, 1.0),
        ("PRB", None, 0.0),
        ("MKI", None, 1.0),
        ("VEI", None, 0.0),
        ("Beta_log", None, 0.0),
        ("Delta_NL", False, None),
        ("dCor_e_y", False, None),
    ]
    rows = []
    for split in ("heldout", "forward_2025"):
        lin = combined_row(combined, "Linear")
        lgb = combined_row(combined, "LightGBM")
        for name, higher, target in measures:
            a = metric_val(lin, name, split)
            b = metric_val(lgb, name, split)
            fa = manuscript_format_flags(a, metric=name, family="Linear", linear_val=a, lgbm_val=b, higher=higher, target=target, can_star=False)
            fb = manuscript_format_flags(b, metric=name, family="LightGBM", linear_val=a, lgbm_val=b, higher=higher, target=target, can_star=False)
            rows.append({
                "measure": name,
                "split": split,
                "Linear": a,
                "LightGBM": b,
                "Linear_bold": fa["manuscript_bold"],
                "LightGBM_bold": fb["manuscript_bold"],
                "copied_from_canonical_994_combined_table": True,
            })
    return pd.DataFrame(rows)


def build_anchor_source(combined: pd.DataFrame) -> pd.DataFrame:
    grid = combined.loc[combined["family"] == "Direct", "rho"].to_numpy(dtype=float)
    anchors = positive_display_anchors(grid)
    lin = combined_row(combined, "Linear")
    lgb = combined_row(combined, "LightGBM")
    rows = []
    specs = ANCHOR_METRIC_SPECS
    for split in ("heldout", "forward_2025"):
        for fam, rho, rho_label in (
            [("Linear", None, "--"), ("LightGBM", None, "--")]
            + [("Direct", a, "positive_anchor") for a in anchors]
            + [("Surrogate", a, "positive_anchor") for a in anchors]
        ):
            rec_row = combined_row(combined, fam, rho)
            rec: Dict[str, Any] = {
                "family": fam,
                "rho": rho,
                "rho_label": rho_label,
                "split": split,
                "is_prespecified_positive_anchor": fam in FAMILY_DISPLAY,
                "transition_endpoint_not_added": True,
            }
            for name, higher, target, can_star in specs:
                val = metric_val(rec_row, name, split)
                rec[name] = val
                flags = manuscript_format_flags(
                    val, metric=name, family=fam,
                    linear_val=metric_val(lin, name, split),
                    lgbm_val=metric_val(lgb, name, split),
                    higher=higher, target=target, can_star=can_star,
                )
                rec[f"{name}__beats_both_baselines"] = flags["beats_both_baselines"]
                rec[f"{name}__beats_ordinary_only"] = flags["beats_ordinary_only"]
                rec[f"{name}__within_reference_range"] = flags["within_reference_range"]
                rec[f"{name}__manuscript_bold"] = flags["manuscript_bold"]
                rec[f"{name}__manuscript_asterisk"] = flags["manuscript_asterisk"]
            rows.append(rec)
    return pd.DataFrame(rows)


def _fmt(v: Any, nd: int = 6) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)) or pd.isna(v):
        return "NA"
    if isinstance(v, (bool, np.bool_)):
        return "true" if bool(v) else "false"
    if isinstance(v, (int, np.integer)):
        return str(int(v))
    try:
        return f"{float(v):.{nd}g}"
    except (TypeError, ValueError):
        return str(v)


def stage_analysis(repo: Path, result_root: Path, v1: Path, output_root: Path, guard: OutputGuard) -> Dict[str, Any]:
    from utils.transition_paper_assets_report import build_interpretation

    identity = validate_canonical_result_root(result_root)
    v1_hashes_now = {str(p): sha256_file(p) for p in sorted(v1.rglob("*")) if p.is_file()}
    frozen = json.loads((output_root / "provenance" / "v1_immutable_hashes.json").read_text(encoding="utf-8"))
    changed = [k for k, h in frozen.items() if v1_hashes_now.get(k) != h]
    extra = [k for k in v1_hashes_now if k not in frozen]
    if changed or extra:
        raise RuntimeError(f"v1 mutated before analysis: changed={changed[:8]} extra={extra[:8]}")
    v1_tables = load_v1_tables(v1)
    combined = pd.read_csv(load_combined_path_table(result_root))
    regret = build_regret(combined, v1_tables)
    sharpness = build_sharpness(combined)
    summary = build_summary(combined, v1_tables)
    detail = build_event_detail(v1_tables)
    mech = build_mechanism_compact(v1_tables)
    envelope = build_envelope(combined, v1_tables)
    baseline = build_baseline_source(combined)
    anchors = build_anchor_source(combined)
    interp = build_interpretation(summary, regret, sharpness, detail, mech)
    write_table(regret, output_root / "tables" / "transition_oos_span_regret.csv", guard, markdown=True)
    write_table(sharpness, output_root / "tables" / "transition_event_sharpness.csv", guard)
    write_table(summary, output_root / "tables" / "paper_transition_summary.csv", guard, markdown=True)
    write_table(detail, output_root / "tables" / "paper_transition_event_detail.csv", guard)
    write_table(mech, output_root / "tables" / "transition_mechanism_endpoint_compact.csv", guard, markdown=True)
    env_md = (
        "# Performance envelopes (component-wise; not a selected configuration)\n\n"
        "Min/max values for Direct are envelopes across every canonical rho inside the frozen "
        "CV-derived descriptive transition span. Extrema need not come from one rho. "
        "Surrogate has no common five-metric span under the frozen protocol, so its span envelope is empty. "
        "Deltas versus ordinary LightGBM are reported where both values exist. "
        "Delta_NL is canonically available only for held-out and 2025.\n\n"
        + df_to_markdown(envelope)
        + "\n"
    )
    guard.write_df(envelope, output_root / "tables" / "transition_region_performance_envelope.csv", parquet=True)
    guard.write_text(output_root / "tables" / "transition_region_performance_envelope.md", env_md)
    write_table(baseline, output_root / "tables" / "baseline_comparison_source.csv", guard)
    write_table(anchors, output_root / "tables" / "path_anchor_summary_source.csv", guard)
    hits = [p for p in FORBIDDEN_PHRASES if p.lower() in interp.lower()]
    if hits:
        raise RuntimeError(f"interpretation contains forbidden phrases: {hits}")
    guard.write_text(output_root / "report" / "transition_paper_interpretation.md", interp)
    payload = {
        "status": "PASS",
        "utc": utc_now(),
        "identity": identity,
        "n_regret_rows": int(len(regret)),
        "n_sharpness_rows": int(len(sharpness)),
        "interpretation_letter": classify_direct_interpretation(regret),
        "model_fitting": False,
        "v1_unchanged": True,
    }
    guard.write_json(output_root / "qa" / "ANALYSIS_STATUS.json", payload)
    return payload


def _forbidden_in_text(text: str) -> List[str]:
    low = text.lower()
    return [p for p in FORBIDDEN_PHRASES if p.lower() in low]


def build_readme() -> str:
    return """# Canonical 994-tree paper-asset bundle

All files in this tree are generated from frozen 994-tree artifacts. No rho is selected.

## Transition follow-up

- `tables/paper_transition_summary.*`: one row per family; copied v1 CV events/span/LOFO/concordance plus endpoint-grid flags.
- `tables/paper_transition_event_detail.*`: family x metric CV/held-out/2025 event locations. Direct also has LOFO-envelope membership as a sensitivity diagnostic only.
- `tables/transition_oos_span_regret.*`: Direct frozen CV span regret on held-out and 2025.
- `tables/transition_event_sharpness.*`: discrete best/second/neighbor gaps, no smoothing.
- `tables/transition_mechanism_endpoint_compact.*`: interpretive q_beta at Direct span endpoints and Surrogate CV event rhos. Theory does not set endpoints.
- `tables/transition_region_performance_envelope.*`: component-wise min/max inside the frozen Direct span.
- `figures/appendix/paper_transition_event_locations.*`: turning-event location and temporal concordance. Direct span shaded; Surrogate has no common span. Resolves the missing transition-location display.
- `figures/appendix/paper_primary_paths_with_events.*`: unsmoothed primary-metric paths with event markers.
- `figures/diagnostic/paper_direct_oos_span_regret.*`: normalized Direct span regret. Appendix candidate, not automatically a main-text figure.

Proposed event-location caption: Turning-event locations for the five primary metrics on Direct and Surrogate. Large circles are equal-weight CV events; small dots are chronological folds; squares are held-out; triangles are 2025. Direct shading is the frozen CV descriptive transition span. rho=0.1 is the first tested positive grid point. Surrogate has no common five-metric CV span because CV RMSE_log is at rho=0.

## Main-paper candidates

- `figures/main/baseline_models_motivation_2024_2025.*`: current manuscript baseline motivation figure (canonical 994). Resolves generic generated_v6_preselection pointer.
- `figures/main/ratio_shape_evolution.*`: Direct/Surrogate held-out and 2025; prespecified anchors only; no transition highlight.
- `figures/main/mechanism_vs_rho.*`: Beta_log, Delta_NL, dCor; full paths; rho=0 explicit; no transition shading.
- `figures/main/accuracy_equity_trajectories_inprocessing_only.*`: Linear, ordinary LightGBM, Direct, Surrogate only. Resolves the manuscript TODO to drop centered recalibration.

## Appendix candidates

- `figures/appendix/prb_mki_accuracy_equity_inprocessing_only.*`: same four families; no recalibration.
- `figures/appendix/predictive_metric_paths.*`
- `figures/appendix/level_uniformity_paths.*`
- `figures/appendix/cv_fold_stability.*`
- `figures/appendix/vei_percentile_group_profile.*`

## Table sources

- `tables/baseline_comparison_source.*`: current baseline table numbers and pairwise-bold flags.
- `tables/path_anchor_summary_source.*`: Linear, ordinary LightGBM, and positive anchors nearest 0.1, 1, 10, 100. Bold/asterisk flags follow the live paper_v12 rule. Transition endpoints are not added.

Do not insert these files into paper/ or any .tex from this pipeline.
"""


def stage_render(repo: Path, result_root: Path, v1: Path, output_root: Path, guard: OutputGuard) -> Dict[str, Any]:
    setup_style()
    from utils import transition_paper_asset_plots as plots

    combined = pd.read_csv(load_combined_path_table(result_root))
    v1_tables = load_v1_tables(v1)
    regret = pd.read_csv(output_root / "tables" / "transition_oos_span_regret.csv")
    written: List[str] = []
    fig_root = output_root / "figures"
    written += plots.plot_event_locations(plt, combined, v1_tables, guard, fig_root / "appendix" / "paper_transition_event_locations")
    written += plots.plot_primary_paths(plt, combined, v1_tables, guard, fig_root / "appendix" / "paper_primary_paths_with_events")
    written += plots.plot_regret(plt, regret, guard, fig_root / "diagnostic" / "paper_direct_oos_span_regret")
    written += plots.plot_baseline_motivation(plt, result_root, guard, fig_root / "main" / "baseline_models_motivation_2024_2025")
    written += plots.plot_ratio_shape(plt, result_root, combined, guard, fig_root / "main" / "ratio_shape_evolution")
    written += plots.plot_mechanism(plt, combined, guard, fig_root / "main" / "mechanism_vs_rho")
    written += plots.plot_accuracy_equity(plt, combined, guard, fig_root / "main" / "accuracy_equity_trajectories_inprocessing_only")
    written += plots.plot_prb_mki(plt, combined, guard, fig_root / "appendix" / "prb_mki_accuracy_equity_inprocessing_only")
    written += plots.plot_metric_paths(
        plt, combined,
        (("R2_price", r"$R^2_P$"), ("MAE_price", "MAE"), ("MAPE", r"MAPE (\%)"), ("RMSE_log", r"RMSE$_{\log}$")),
        guard, fig_root / "appendix" / "predictive_metric_paths",
    )
    written += plots.plot_metric_paths(
        plt, combined,
        (
            ("median_ratio", "Median ratio"),
            ("mean_ratio", "Mean ratio"),
            ("weighted_mean_ratio", "Weighted mean ratio"),
            ("COD", "COD"),
            ("COV", r"COV (\%)"),
        ),
        guard, fig_root / "appendix" / "level_uniformity_paths",
    )
    written += plots.plot_cv_stability(plt, combined, guard, fig_root / "appendix" / "cv_fold_stability")
    written += plots.plot_vei_groups(plt, result_root, guard, fig_root / "appendix" / "vei_percentile_group_profile")
    readme = build_readme()
    hits = _forbidden_in_text(readme)
    interp = (output_root / "report" / "transition_paper_interpretation.md").read_text(encoding="utf-8")
    hits += _forbidden_in_text(interp)
    if hits:
        raise RuntimeError(f"forbidden phrases in reports: {hits}")
    guard.write_text(output_root / "report" / "paper_asset_readme.md", readme)
    payload = {
        "status": "PASS",
        "utc": utc_now(),
        "figures": written,
        "forbidden_phrase_hits": [],
        "no_tex": True,
        "no_recalibration_in_main_tradeoff": True,
        "no_transition_shading_in_main_clean_figures": True,
    }
    guard.write_json(output_root / "qa" / "RENDER_STATUS.json", payload)
    return payload


def _pdf_ok(path: Path) -> Tuple[bool, str]:
    data = path.read_bytes()
    if not data.startswith(b"%PDF") or path.stat().st_size < 64:
        return False, "not a nonempty pdf"
    m = re.search(rb"/MediaBox\s*\[\s*([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s*\]", data)
    if m:
        w = float(m.group(3)) - float(m.group(1))
        h = float(m.group(4)) - float(m.group(2))
        if w <= 0 or h <= 0:
            return False, f"nonpositive mediabox {w}x{h}"
    return True, "ok"


def _png_ok(path: Path) -> Tuple[bool, str]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n") or path.stat().st_size < 64:
        return False, "not a nonempty png"
    if data[12:16] != b"IHDR":
        return False, "missing IHDR"
    w, h = struct.unpack(">II", data[16:24])
    if w <= 0 or h <= 0:
        return False, f"nonpositive png {w}x{h}"
    return True, "ok"


def _run_unit_tests(repo: Path) -> Dict[str, Any]:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.chdir(repo)
    import importlib

    rows: List[Dict[str, str]] = []
    try:
        mod = importlib.import_module("tests.test_transition_paper_assets")
        importlib.reload(mod)
    except Exception:
        return {"n_pass": 0, "n_fail": 1, "results": [{"name": "<import>", "status": "FAIL", "error": traceback.format_exc()}]}
    for name in sorted(dir(mod)):
        if not name.startswith("test_"):
            continue
        fn = getattr(mod, name)
        if not callable(fn):
            continue
        rec = {"name": name, "status": "PASS", "error": ""}
        try:
            fn()
        except Exception:
            rec["status"] = "FAIL"
            rec["error"] = traceback.format_exc()
        rows.append(rec)
    n_fail = sum(1 for r in rows if r["status"] != "PASS")
    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    return {"n_pass": n_pass, "n_fail": n_fail, "results": rows}


def stage_audit(repo: Path, result_root: Path, v1: Path, output_root: Path, guard: OutputGuard) -> Dict[str, Any]:
    problems: List[str] = []
    identity = validate_canonical_result_root(result_root)
    tests = _run_unit_tests(repo)
    if tests["n_fail"]:
        problems.append(f"unit tests failed: {tests['n_fail']}")
    for rel in REQUIRED_TABLES + MAIN_FIGURES + APPENDIX_FIGURES + DIAGNOSTIC_FIGURES:
        path = output_root / rel
        if not path.is_file() or path.stat().st_size < 64:
            problems.append(f"missing or empty {rel}")
    for rel in list(MAIN_FIGURES) + list(APPENDIX_FIGURES) + list(DIAGNOSTIC_FIGURES):
        path = output_root / rel
        if not path.is_file():
            continue
        if path.suffix == ".pdf":
            ok, msg = _pdf_ok(path)
        else:
            ok, msg = _png_ok(path)
        if not ok:
            problems.append(f"{rel}: {msg}")
    tex = list(output_root.rglob("*.tex"))
    if tex:
        problems.append("tex files present under output root")
    baseline_tex = json.loads((output_root / "provenance" / "tex_baseline_sha256.json").read_text(encoding="utf-8"))
    current_tex = {str(p.relative_to(repo)): sha256_file(p) for p in iter_tex_files(repo)}
    if current_tex != baseline_tex:
        problems.append("tracked/existing .tex hashes changed or new .tex appeared")
    paper_base = json.loads((output_root / "provenance" / "paper_tree_sha256.json").read_text(encoding="utf-8"))
    paper_now = {str(p.relative_to(repo)): sha256_file(p) for p in (repo / "paper").rglob("*") if p.is_file()}
    if paper_now != paper_base:
        problems.append("paper/ tree changed during this analysis")
    frozen = json.loads((output_root / "provenance" / "v1_immutable_hashes.json").read_text(encoding="utf-8"))
    now_v1 = {str(p): sha256_file(p) for p in sorted(v1.rglob("*")) if p.is_file()}
    if now_v1 != frozen:
        problems.append("transition_regions_v1 hashes changed")
    if (result_root.parent / "paper_v6_preselection").name == HISTORICAL_500_ROOT_NAME:
        pass
    combined = pd.read_csv(load_combined_path_table(result_root))
    grid = family_frame(combined, "Direct")["rho"].to_numpy(dtype=float)
    events = pd.read_csv(output_root / "tables" / "paper_transition_event_detail.csv")
    for _, row in events.iterrows():
        for col in ("cv_rho", "heldout_rho", "forward_2025_rho"):
            if col not in row or pd.isna(row[col]):
                continue
            if not any(numerically_equal(float(row[col]), float(g)) for g in family_frame(combined, str(row["family"]))["rho"]):
                problems.append(f"event rho not on grid: {row['family']} {row['metric']} {col}")
    regret = pd.read_csv(output_root / "tables" / "transition_oos_span_regret.csv")
    if len(regret) != 10:
        problems.append(f"expected 10 Direct regret rows, got {len(regret)}")
    if (regret["raw_regret"] < -1e-12).any():
        problems.append("negative raw regret")
    sharpness = pd.read_csv(output_root / "tables" / "transition_event_sharpness.csv")
    expected_sharp = 2 * (1 + 7 + 1 + 1) * 5
    if len(sharpness) != expected_sharp:
        problems.append(f"expected {expected_sharp} sharpness rows, got {len(sharpness)}")
    summary = pd.read_csv(output_root / "tables" / "paper_transition_summary.csv")
    if len(summary) != 2:
        problems.append("paper_transition_summary must have two family rows")
    ae = output_root / "report" / "paper_asset_readme.md"
    if "centered recalibration" in ae.read_text(encoding="utf-8").lower() and "drop" not in ae.read_text(encoding="utf-8").lower():
        pass
    # four-family tradeoff: Linear, LightGBM, Direct, Surrogate only (no recalibration plotted)
    git = git_state(repo)
    assets = []
    roles = {
        "figures/main/": "main",
        "figures/appendix/": "appendix",
        "figures/diagnostic/": "diagnostic",
        "tables/": "replication",
        "report/": "replication",
    }
    combined_sha = sha256_file(load_combined_path_table(result_root))
    for rel_dir, role in roles.items():
        d = output_root / rel_dir
        if not d.is_dir():
            continue
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            rel = str(p.relative_to(output_root))
            transition_shown = "transition" in rel or "event_location" in rel or "span_regret" in rel or "primary_paths_with_events" in rel
            assets.append({
                "filename": rel,
                "proposed_paper_role": role,
                "canonical_source_paths": ["analysis/combined_path_table.csv", str(v1)],
                "source_sha256": combined_sha,
                "output_sha256": sha256_file(p),
                "data_id": identity["data_id"],
                "split_id": identity["split_id"],
                "config_id": identity["lgbm_config_id"],
                "n_estimators": 994,
                "git_HEAD": git.get("git_commit"),
                "generation_script": "scripts/analyze_transition_paper_assets.py",
                "generation_timestamp": utc_now(),
                "transition_information_displayed": bool(transition_shown),
                "contains_selected_or_recommended_rho": False,
            })
    man_df = pd.DataFrame(assets)
    guard.write_df(man_df, output_root / "provenance" / "paper_asset_manifest.csv", parquet=False)
    guard.write_json(output_root / "provenance" / "paper_asset_manifest.json", {"assets": assets, "n": len(assets)})
    hashed = {str(p.relative_to(output_root)): sha256_file(p) for p in sorted(output_root.rglob("*")) if p.is_file() and p.name != "FINAL_PAPER_ASSET_STATUS.json"}
    status = "PASS" if not problems else "FAIL"
    payload = {
        "status": status,
        "utc": utc_now(),
        "identity": {"data_id": identity["data_id"], "split_id": identity["split_id"], "lgbm_config_id": identity["lgbm_config_id"]},
        "problems": problems,
        "tests": tests,
        "n_assets_manifested": len(assets),
        "n_artifacts_hashed": len(hashed),
        "no_paper_writes": True,
        "no_tex_created_or_modified": True,
        "no_model_fitting": True,
        "v1_immutable": True,
        "historical_500_root_read": False,
    }
    guard.write_json(output_root / "qa" / "FINAL_PAPER_ASSET_STATUS.json", payload)
    guard.write_json(output_root / "provenance" / "output_sha256.json", hashed)
    if status != "PASS":
        raise RuntimeError("FINAL_PAPER_ASSET_AUDIT failed: " + "; ".join(problems))
    return payload


def main() -> int:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    repo, result_root, v1, output_root, guard = resolve_paths(args)
    stages = {
        "preflight": stage_preflight,
        "analysis": stage_analysis,
        "render": stage_render,
        "audit": stage_audit,
    }
    try:
        payload = stages[args.stage](repo, result_root, v1, output_root, guard)
    except Exception as err:
        qa_name = {
            "preflight": "PREFLIGHT_STATUS.json",
            "analysis": "ANALYSIS_STATUS.json",
            "render": "RENDER_STATUS.json",
            "audit": "FINAL_PAPER_ASSET_STATUS.json",
        }[args.stage]
        fail = {"status": "FAIL", "utc": utc_now(), "error": str(err), "traceback": traceback.format_exc(), "stage": args.stage}
        try:
            guard.write_json(output_root / "qa" / qa_name, fail)
        except Exception:
            print(json.dumps(fail, indent=2))
        print(f"STAGE {args.stage} FAIL: {err}", file=sys.stderr)
        return 1
    print(json.dumps({"stage": args.stage, "status": payload.get("status"), "output_root": str(output_root)}, indent=2))
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)



