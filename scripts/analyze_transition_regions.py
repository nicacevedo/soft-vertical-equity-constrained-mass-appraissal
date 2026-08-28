#!/usr/bin/env python3
"""994-tree CV transition-region analysis.

Read-only post-processing of frozen canonical results. No fitting, no rho
selection, no .tex, and no writes under paper/.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

plt = None  # lazy-loaded in render only
PdfPages = None

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from canonical_experiment import git_state, package_versions
from utils.transition_regions import (
    ATOL,
    BAND_METRICS,
    CANONICAL_ROOT_NAME,
    ENVELOPE_METRICS,
    EXPECTED_IDENTITY,
    FAMILY_DISPLAY,
    FOLD_IDS,
    FORBIDDEN_SPAN_DESCRIPTORS,
    PRIMARY_METRICS,
    PROTOCOL,
    CanonicalIdentityError,
    DiscreteEvent,
    OutputConfineError,
    OutputGuard,
    PathDataError,
    TransitionSpan,
    assert_cv_mean_is_equal_weight,
    attenuation,
    concordance_row,
    construct_transition_span,
    default_output_root,
    df_to_markdown,
    event_table_row,
    expected_canonical_rhos,
    extract_discrete_event,
    extract_primary_events_from_frame,
    family_frame,
    fold_col,
    fold_matrix_from_frame,
    load_combined_path_table,
    lofo_events_and_span,
    lofo_span_summary,
    metric_col,
    numerically_equal,
    protocol_canonical_json,
    protocol_sha256,
    q_beta_cov_agree,
    q_ratio,
    rho_in_closed_span,
    sha256_file,
    span_segment_mask,
    summarize_fold_events_logrho,
    validate_canonical_result_root,
    validate_combined_counts,
)


DIRECT_COLOR = "#1D4ED8"
SURR_COLOR = "#C2410C"
FAMILY_COLOR = {"Direct": DIRECT_COLOR, "Surrogate": SURR_COLOR}
SPAN_FACE = "#9CA3AF"
SUBDIRS = (
    "protocol",
    "tables",
    "figures",
    "provenance",
    "logs",
    "cluster",
    "report",
    "code_snapshot",
    "qa",
)
FORBIDDEN_REPORT_PHRASES = (
    "optimal rho",
    "recommended rho",
    "selected rho",
    "safe range",
    "deployment-ready",
    "preferred rho",
    "recommended range",
    "optimal span",
    "recommended span",
)
CODE_SNAPSHOT_FILES = (
    Path("utils/transition_regions.py"),
    Path("scripts/analyze_transition_regions.py"),
    Path("tests/test_transition_regions.py"),
    Path("scripts/submit_transition_regions_v1.sh"),
)
EXISTING_TEST_MODULES = (
    "tests.test_canonical_grid",
    "tests.test_canonical_metrics",
    "tests.test_canonical_objectives",
    "tests.test_paper_v6_guards",
    "tests.test_paper_v6_reporting",
    "tests.test_transition_regions",
)
CANONICAL_INPUT_RELS = (
    "baseline_gate.json",
    "experiment_manifest.json",
    "lgbm_config.json",
    "frozen_baseline.json",
    "cv_completion.json",
    "analysis/combined_path_table.csv",
    "manifests/cv_qa.json",
    "manifests/final_qa.json",
    "paper_outputs/paper_results_manifest.json",
    "final_local_results/FINAL_LOCAL_RESULTS_STATUS.json",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _load_mpl() -> None:
    """Import matplotlib only for render. Precheck/core must not load it."""
    global plt, PdfPages
    if plt is not None:
        return
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt_mod
    from matplotlib.backends.backend_pdf import PdfPages as PdfPagesMod

    plt = plt_mod
    PdfPages = PdfPagesMod


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
        }
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="994-tree descriptive transition-region analysis.")
    p.add_argument("--stage", required=True, choices=("precheck", "tests", "core", "render", "audit"))
    p.add_argument("--result-root", required=True, type=str)
    p.add_argument("--output-root", default=None, type=str)
    p.add_argument("--repo-root", default=str(REPO), type=str)
    return p.parse_args()


def resolve_paths(args: argparse.Namespace) -> Tuple[Path, Path, Path, OutputGuard]:
    repo = Path(args.repo_root).resolve()
    result_root = Path(args.result_root)
    if not result_root.is_absolute():
        result_root = (repo / result_root).resolve()
    else:
        result_root = result_root.resolve()
    identity = validate_canonical_result_root(result_root)
    if args.output_root:
        output_root = Path(args.output_root)
        if not output_root.is_absolute():
            output_root = (repo / output_root).resolve()
        else:
            output_root = output_root.resolve()
    else:
        output_root = default_output_root(result_root, identity["data_id"], identity["split_id"])
    try:
        output_root.relative_to(result_root)
    except ValueError as err:
        raise OutputConfineError(f"output root must live under result root: {output_root}") from err
    if output_root.name != "transition_regions_v1":
        raise OutputConfineError(f"output root must end in transition_regions_v1, got {output_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    guard = OutputGuard(output_root, repo)
    for name in SUBDIRS:
        guard.ensure_subdir(name)
    return repo, result_root, output_root, guard


def hash_tree(paths: Iterable[Path]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for path in paths:
        p = Path(path)
        if p.is_file():
            out[str(p)] = sha256_file(p)
    return out


def iter_tex_files(repo: Path) -> List[Path]:
    files = []
    for p in repo.rglob("*.tex"):
        if not p.is_file():
            continue
        parts = set(p.parts)
        if "transition_regions_v1" in parts:
            continue
        files.append(p)
    return sorted(files)


def env_snapshot(repo: Path) -> Dict[str, Any]:
    versions = package_versions()
    extra = {}
    try:
        from importlib.metadata import version as pkg_version
    except Exception:  # pragma: no cover
        from importlib_metadata import version as pkg_version  # type: ignore
    for name in ("matplotlib", "pyarrow"):
        try:
            extra[name] = pkg_version(name)
        except Exception:
            extra[name] = "unknown"
    versions.update(extra)
    return {
        "git": git_state(repo),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "packages": versions,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_partition": os.environ.get("SLURM_JOB_PARTITION"),
        "hostname": os.uname().nodename if hasattr(os, "uname") else "",
        "cwd": os.getcwd(),
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
    cluster_src = guard.output_root / "cluster"
    if cluster_src.is_dir():
        snap_cluster = guard.ensure_subdir("code_snapshot/cluster")
        for src in sorted(cluster_src.glob("*")):
            if src.is_file() and src.suffix.lower() != ".tex":
                guard.write_bytes(snap_cluster / src.name, src.read_bytes())
                hashes[f"cluster/{src.name}"] = sha256_file(src)
    return hashes


def freeze_protocol(guard: OutputGuard) -> Dict[str, Any]:
    path = guard.output_root / "protocol" / "transition_analysis_protocol.json"
    blob = protocol_canonical_json()
    want = protocol_sha256()
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        got = hashlib.sha256(existing.encode("utf-8")).hexdigest()
        if got != want:
            raise CanonicalIdentityError(
                "frozen protocol hash mismatch; refusing to continue. "
                f"disk={got} code={want}"
            )
    else:
        guard.write_text(path, blob)
    return {"path": str(path), "sha256": want, "frozen": True}


def lookup_at_rho(df: pd.DataFrame, col: str, rho: float) -> Optional[float]:
    if col not in df.columns:
        return None
    for r, v in zip(df["rho"].to_numpy(dtype=float), pd.to_numeric(df[col], errors="coerce")):
        if numerically_equal(r, rho):
            if pd.isna(v) or not np.isfinite(float(v)):
                return None
            return float(v)
    return None


def require_grid_complete(df: pd.DataFrame, family: str, expected: Sequence[float], splits: Sequence[str], metrics: Sequence[str]) -> List[str]:
    problems = []
    sub = family_frame(df, family)
    got = [float(x) for x in sub["rho"].tolist()]
    for e in expected:
        if not any(numerically_equal(e, g) for g in got):
            problems.append(f"{family} missing canonical rho {e}")
    for metric in metrics:
        for suffix in splits:
            col = metric_col(metric, suffix)
            if col not in sub.columns:
                problems.append(f"{family} missing column {col}")
                continue
            vals = pd.to_numeric(sub[col], errors="coerce")
            if vals.isna().any() or not np.isfinite(vals.to_numpy(dtype=float)).all():
                problems.append(f"{family} non-finite values in {col}")
    return problems


def display_rho(rho: float, min_positive: float) -> float:
    if numerically_equal(rho, 0.0):
        return float(min_positive) * 0.55
    return float(rho)


def shade_span(ax, span: TransitionSpan) -> None:
    if span.status != "VALID_POSITIVE_INTERIOR_SPAN":
        return
    if span.rho_transition_low is None or span.rho_transition_high is None:
        return
    ax.axvspan(
        span.rho_transition_low,
        span.rho_transition_high,
        color=SPAN_FACE,
        alpha=0.22,
        lw=0,
        zorder=0,
        label="CV transition span",
    )


def save_figure(fig: Any, stem: Path, guard: OutputGuard) -> List[str]:
    pdf = guard.allowed(stem.with_suffix(".pdf"))
    png = guard.allowed(stem.with_suffix(".png"))
    fig.savefig(pdf, format="pdf", bbox_inches="tight")
    fig.savefig(png, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [str(pdf), str(png)]


# ---------------------------------------------------------------------------
# Stages
# ---------------------------------------------------------------------------

def stage_precheck(repo: Path, result_root: Path, output_root: Path, guard: OutputGuard) -> Dict[str, Any]:
    identity = validate_canonical_result_root(result_root)
    protocol = freeze_protocol(guard)
    combined_path = load_combined_path_table(result_root)
    combined = pd.read_csv(combined_path)
    expected_rhos = expected_canonical_rhos()
    count_chk = validate_combined_counts(combined, expected_rhos)
    problems = list(count_chk["problems"])
    primary_names = [m for m, _d in PRIMARY_METRICS]
    problems.extend(assert_cv_mean_is_equal_weight(combined, primary_names))
    split_suffixes = ["CV_mean", "heldout", "forward_2025"] + [f"fold_{i}" for i in FOLD_IDS]
    for fam in FAMILY_DISPLAY:
        problems.extend(require_grid_complete(combined, fam, expected_rhos, split_suffixes, primary_names))
        for metric in ("Beta_log", "dCor_e_y", "Cov_log_residual_log_price"):
            problems.extend(require_grid_complete(combined, fam, expected_rhos, ["CV_mean", "heldout", "forward_2025"], [metric]))
        for metric in ("Delta_NL",):
            col_h = metric_col(metric, "heldout")
            col_f = metric_col(metric, "forward_2025")
            if col_h not in combined.columns or col_f not in combined.columns:
                problems.append(f"Delta_NL OOS columns missing ({col_h}, {col_f})")
            else:
                sub = family_frame(combined, fam)
                if sub[col_h].isna().any() or sub[col_f].isna().any():
                    problems.append(f"{fam} Delta_NL missing on held-out or 2025")
        if metric_col("Corr_log_residual_log_price", "CV_mean") not in combined.columns:
            # Not a failure; recorded as unavailable.
            pass
        if family_frame(combined, fam)["rho"].isna().any():
            problems.append(f"{fam} has NA rho")

    tex_files = iter_tex_files(repo)
    tex_hashes = {str(p.relative_to(repo)): sha256_file(p) for p in tex_files}
    paper_files = [p for p in (repo / "paper").rglob("*") if p.is_file()]
    paper_hashes = {str(p.relative_to(repo)): sha256_file(p) for p in paper_files}
    input_hashes = {}
    for rel in CANONICAL_INPUT_RELS:
        path = result_root / rel
        if path.is_file():
            input_hashes[rel] = sha256_file(path)
        else:
            input_hashes[rel] = None

    if (result_root.parent / "paper_v6_preselection").resolve() == result_root.resolve():
        problems.append("result root resolved to historical 500-tree directory")

    env = env_snapshot(repo)
    code_hashes = snapshot_code(repo, guard)
    guard.write_json(output_root / "provenance" / "tex_baseline_sha256.json", tex_hashes)
    guard.write_json(output_root / "provenance" / "paper_tree_sha256.json", paper_hashes)
    guard.write_json(output_root / "provenance" / "input_manifest.json", {
        "result_root": str(result_root),
        "output_root": str(output_root),
        "canonical_identity": {k: identity[k] for k in (
            "data_id", "split_id", "lgbm_config_id", "baseline_gate", "n_estimators", "seed", "n_cv_fits"
        )},
        "combined_path_table": str(combined_path),
        "combined_sha256": sha256_file(combined_path),
        "input_sha256": input_hashes,
        "protocol_sha256": protocol["sha256"],
        "code_snapshot_sha256": code_hashes,
        "historical_500_tree_root_used": False,
        "read_only_frozen_artifacts": True,
    })
    guard.write_json(output_root / "provenance" / "environment.json", env)
    status = "PASS" if not problems else "FAIL"
    payload = {
        "status": status,
        "utc": utc_now(),
        "problems": problems,
        "identity": {
            "data_id": identity["data_id"],
            "split_id": identity["split_id"],
            "lgbm_config_id": identity["lgbm_config_id"],
            "baseline_gate": identity["baseline_gate"],
            "n_estimators": identity["n_estimators"],
            "seed": identity["seed"],
            "n_cv_fits": identity["n_cv_fits"],
            "heldout_identity": identity["heldout_identity"],
            "forward_2025_identity": identity["forward_2025_identity"],
        },
        "counts": count_chk,
        "n_tex_baseline": len(tex_hashes),
        "protocol": protocol,
        "output_root": str(output_root),
        "result_root": str(result_root),
        "no_tex_written": True,
        "no_paper_writes": True,
        "no_model_fitting": True,
    }
    guard.write_json(output_root / "qa" / "PRECHECK_STATUS.json", payload)
    if status != "PASS":
        raise CanonicalIdentityError("PRECHECK failed: " + "; ".join(problems))
    return payload


def _run_test_module(mod_name: str) -> List[Dict[str, str]]:
    rows = []
    mod = importlib.import_module(mod_name)
    for name in sorted(dir(mod)):
        if not name.startswith("test_"):
            continue
        fn = getattr(mod, name)
        if not callable(fn):
            continue
        rec = {"module": mod_name, "name": name, "status": "PASS", "error": ""}
        try:
            fn()
        except Exception:
            rec["status"] = "FAIL"
            rec["error"] = traceback.format_exc()
        rows.append(rec)
    return rows


def stage_tests(repo: Path, result_root: Path, output_root: Path, guard: OutputGuard) -> Dict[str, Any]:
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    os.chdir(repo)
    rows: List[Dict[str, str]] = []
    for mod_name in EXISTING_TEST_MODULES:
        try:
            rows.extend(_run_test_module(mod_name))
        except Exception:
            rows.append({
                "module": mod_name,
                "name": "<import>",
                "status": "FAIL",
                "error": traceback.format_exc(),
            })
    n_fail = sum(1 for r in rows if r["status"] != "PASS")
    n_pass = sum(1 for r in rows if r["status"] == "PASS")
    status = "PASS" if n_fail == 0 else "FAIL"
    payload = {
        "status": status,
        "utc": utc_now(),
        "n_pass": n_pass,
        "n_fail": n_fail,
        "results": rows,
        "output_root": str(output_root),
        "result_root": str(result_root),
    }
    guard.write_json(output_root / "qa" / "TEST_STATUS.json", payload)
    if status != "PASS":
        raise RuntimeError(f"TESTS failed: {n_fail} failures")
    return payload


def _span_from_events(family: str, events: Sequence[DiscreteEvent], expected_rhos: Sequence[float]) -> TransitionSpan:
    positives = [r for r in expected_rhos if r > 0]
    return construct_transition_span(
        family,
        events,
        min_positive_rho=min(positives),
        max_positive_rho=max(positives),
    )


def _q_block(family: str, df: pd.DataFrame, rho: float, suffix: str, location: str) -> Dict[str, Any]:
    beta = lookup_at_rho(df, metric_col("Beta_log", suffix), rho)
    beta0 = lookup_at_rho(df, metric_col("Beta_log", suffix), 0.0)
    cov = lookup_at_rho(df, metric_col("Cov_log_residual_log_price", suffix), rho)
    cov0 = lookup_at_rho(df, metric_col("Cov_log_residual_log_price", suffix), 0.0)
    qb, qb_note = q_ratio(beta, beta0) if beta is not None and beta0 is not None else (None, "missing")
    qc, qc_note = q_ratio(cov, cov0) if cov is not None and cov0 is not None else (None, "missing")
    agree, delta, agree_note = q_beta_cov_agree(qb, qc)
    q_name = "q_beta" if family == "Direct" else "empirical_remaining_first_order_fraction"
    fold_q = {}
    if suffix == "CV_mean":
        for fid in FOLD_IDS:
            b = lookup_at_rho(df, fold_col("Beta_log", fid), rho)
            b0 = lookup_at_rho(df, fold_col("Beta_log", fid), 0.0)
            qf, ntf = q_ratio(b, b0) if b is not None and b0 is not None else (None, "missing")
            fold_q[f"q_beta_fold_{fid}"] = qf
            fold_q[f"q_beta_fold_{fid}_note"] = ntf
    rec = {
        "family": family,
        "location": location,
        "rho": rho,
        "split": suffix,
        "quantity_name": q_name,
        "Beta_log": beta,
        "Beta_log_rho0": beta0,
        "q_beta": qb,
        "q_beta_note": qb_note,
        "attenuation_1_minus_q_beta": attenuation(qb),
        "Cov_log_residual_log_price": cov,
        "Cov_log_residual_log_price_rho0": cov0,
        "q_cov": qc,
        "q_cov_note": qc_note,
        "q_beta_q_cov_agree": agree,
        "q_beta_minus_q_cov": delta,
        "q_agree_note": agree_note,
        "overcorrection": bool(qb is not None and qb < 0),
        "direct_theory_q": None,
        "direct_theory_note": "no_canonical_994_projection_theory_artifact_under_result_root",
    }
    rec.update(fold_q)
    return rec


def stage_core(repo: Path, result_root: Path, output_root: Path, guard: OutputGuard) -> Dict[str, Any]:
    freeze_protocol(guard)
    identity = validate_canonical_result_root(result_root)
    combined = pd.read_csv(load_combined_path_table(result_root))
    expected_rhos = expected_canonical_rhos()
    count_chk = validate_combined_counts(combined, expected_rhos)
    if not count_chk["ok"]:
        raise CanonicalIdentityError("canonical counts failed in core: " + "; ".join(count_chk["problems"]))

    cv_event_rows = []
    fold_event_rows = []
    fold_summary_rows = []
    span_rows = []
    lofo_frames = []
    conc_rows = []
    mech_rows = []
    band_frames = []
    spans: Dict[str, TransitionSpan] = {}
    cv_events_by_family: Dict[str, List[DiscreteEvent]] = {}

    for fam in FAMILY_DISPLAY:
        sub = family_frame(combined, fam)
        if sub["rho"].isna().any() or len(sub) != 51:
            raise PathDataError(f"{fam} path is incomplete")
        cv_events = extract_primary_events_from_frame(sub, "CV_mean")
        cv_events_by_family[fam] = cv_events
        for ev in cv_events:
            cv_event_rows.append(event_table_row(fam, "cv_mean", ev))
        span = _span_from_events(fam, cv_events, expected_rhos)
        spans[fam] = span
        rec = span.to_dict()
        rec["blocking_metrics"] = ",".join(span.blocking_metrics)
        rec["plateau_metrics"] = ",".join(span.plateau_metrics)
        rec["label"] = "CV-derived descriptive transition span"
        rec["not_a_selected_rho"] = True
        span_rows.append(rec)

        for metric, direction in PRIMARY_METRICS:
            rhos, mat = fold_matrix_from_frame(sub, metric)
            fold_events = []
            for j, fid in enumerate(FOLD_IDS):
                ev = extract_discrete_event(rhos, mat[:, j], metric=metric, direction=direction)
                fold_events.append(ev)
                fold_event_rows.append(event_table_row(fam, f"fold_{fid}", ev, {"fold_id": int(fid)}))
            summary = summarize_fold_events_logrho(fold_events)
            summary.update({"family": fam, "metric": metric, "direction": direction})
            fold_summary_rows.append(summary)

        lofo = lofo_events_and_span(
            sub,
            fam,
            min_positive_rho=min(r for r in expected_rhos if r > 0),
            max_positive_rho=max(expected_rhos),
        )
        lofo_frames.append(lofo)

        frozen = span
        for split, suffix in (("heldout", "heldout"), ("forward_2025", "forward_2025")):
            oos_events = extract_primary_events_from_frame(sub, suffix)
            for ev in oos_events:
                conc_rows.append(concordance_row(fam, split, ev, frozen))

        q_locations: List[Tuple[str, float]] = []
        if span.status == "VALID_POSITIVE_INTERIOR_SPAN":
            q_locations.append(("span_low", float(span.rho_transition_low)))
            if not numerically_equal(span.rho_transition_low, span.rho_transition_high):
                q_locations.append(("span_high", float(span.rho_transition_high)))
        for ev in cv_events:
            if ev.rho_low is not None:
                q_locations.append((f"cv_event_{ev.metric}_low", float(ev.rho_low)))
            if ev.rho_high is not None and (ev.rho_low is None or not numerically_equal(ev.rho_low, ev.rho_high)):
                q_locations.append((f"cv_event_{ev.metric}_high", float(ev.rho_high)))
        seen = set()
        for loc, rho in q_locations:
            key = (loc, round(rho, 12))
            if key in seen:
                continue
            seen.add(key)
            for suffix in ("CV_mean", "heldout", "forward_2025"):
                mech_rows.append(_q_block(fam, sub, rho, suffix, loc))

        if span.status == "VALID_POSITIVE_INTERIOR_SPAN":
            mask = [
                rho_in_closed_span(float(r), span.rho_transition_low, span.rho_transition_high)
                for r in sub["rho"].to_numpy(dtype=float)
            ]
            band = sub.loc[np.array(mask)].copy()
            band.insert(0, "in_cv_transition_span", True)
            band_frames.append(band)

    events_cv = pd.DataFrame(cv_event_rows)
    events_fold = pd.DataFrame(fold_event_rows)
    fold_sum = pd.DataFrame(fold_summary_rows)
    span_df = pd.DataFrame(span_rows)
    lofo_df = pd.concat(lofo_frames, ignore_index=True) if lofo_frames else pd.DataFrame()
    conc_df = pd.DataFrame(conc_rows)
    mech_df = pd.DataFrame(mech_rows)
    band_df = pd.concat(band_frames, ignore_index=True) if band_frames else pd.DataFrame()

    baselines = combined.loc[combined["family"].astype(str).isin(["Linear", "LightGBM"])].copy()
    baselines.insert(0, "role", "unpenalized_baseline")

    env_rows = []
    for evaluation, suffix in (("heldout", "heldout"), ("forward_2025", "forward_2025")):
        for fam, n_cfg in (("Linear", 1), ("LightGBM", 1)):
            sub = combined.loc[combined["family"].astype(str) == fam]
            if sub.empty:
                continue
            rec: Dict[str, Any] = {
                "evaluation": evaluation,
                "row_group": "Linear regression" if fam == "Linear" else "Ordinary LightGBM",
                "family": fam,
                "n_configurations": int(len(sub)),
                "envelope_is_componentwise_across_span": False,
                "not_a_selected_configuration": True,
                "cv_span_status": "",
            }
            for metric in ENVELOPE_METRICS:
                col = metric_col(metric, suffix)
                val = float(pd.to_numeric(sub[col], errors="coerce").iloc[0]) if col in sub.columns else np.nan
                rec[f"{metric}__min"] = val
                rec[f"{metric}__max"] = val
            env_rows.append(rec)
        for fam in FAMILY_DISPLAY:
            span = spans[fam]
            rec = {
                "evaluation": evaluation,
                "row_group": f"{fam}: all rho within CV transition span",
                "family": fam,
                "n_configurations": 0,
                "envelope_is_componentwise_across_span": True,
                "not_a_selected_configuration": True,
                "cv_span_status": span.status,
            }
            if span.status == "VALID_POSITIVE_INTERIOR_SPAN" and not band_df.empty:
                part = band_df.loc[band_df["family"].astype(str) == fam]
                rec["n_configurations"] = int(len(part))
                for metric in ENVELOPE_METRICS:
                    col = metric_col(metric, suffix)
                    if col not in part.columns:
                        rec[f"{metric}__min"] = np.nan
                        rec[f"{metric}__max"] = np.nan
                        continue
                    vals = pd.to_numeric(part[col], errors="coerce")
                    rec[f"{metric}__min"] = float(vals.min()) if vals.notna().any() else np.nan
                    rec[f"{metric}__max"] = float(vals.max()) if vals.notna().any() else np.nan
            else:
                for metric in ENVELOPE_METRICS:
                    rec[f"{metric}__min"] = np.nan
                    rec[f"{metric}__max"] = np.nan
            env_rows.append(rec)
    env_df = pd.DataFrame(env_rows)

    tables = {
        "transition_events_cv_mean": events_cv,
        "transition_events_by_fold": events_fold,
        "transition_fold_stability_summary": fold_sum,
        "transition_span_summary": span_df,
        "transition_lofo_sensitivity": lofo_df,
        "transition_temporal_concordance": conc_df,
        "transition_mechanism_summary": mech_df,
        "transition_band_configs": band_df,
        "transition_baseline_comparison": baselines,
        "transition_region_performance_envelope": env_df,
    }
    written = []
    for name, df in tables.items():
        written.extend(guard.write_df(df, output_root / "tables" / f"{name}.csv", parquet=True))
    env_md = "# Performance envelopes (component-wise; not a unique penalized rho)\n\n"
    env_md += (
        "Min/max values for Direct and Surrogate are envelopes across every canonical "
        "rho inside that family's CV-derived descriptive transition span. They do not "
        "generally come from a single penalized rho.\n\n"
    )
    env_md += df_to_markdown(env_df) + "\n"
    guard.write_text(output_root / "tables" / "transition_region_performance_envelope.md", env_md)

    conc_summary = {}
    for fam in FAMILY_DISPLAY:
        conc_summary[fam] = {}
        for split in ("heldout", "forward_2025"):
            part = conc_df.loc[(conc_df["family"] == fam) & (conc_df["split"] == split)]
            inside = part["inside_frozen_cv_span"]
            n_in = int(inside.fillna(False).sum()) if not part.empty else 0
            conc_summary[fam][split] = f"{n_in} / {len(part)} events inside CV span"

    lofo_summaries = {}
    for fam in FAMILY_DISPLAY:
        lofo_summaries[fam] = lofo_span_summary(lofo_df.loc[lofo_df["family"] == fam])

    payload = {
        "status": "PASS",
        "utc": utc_now(),
        "identity": {
            "data_id": identity["data_id"],
            "split_id": identity["split_id"],
            "lgbm_config_id": identity["lgbm_config_id"],
        },
        "span": {fam: spans[fam].to_dict() for fam in FAMILY_DISPLAY},
        "concordance": conc_summary,
        "lofo": lofo_summaries,
        "n_band_configs": {fam: int((band_df["family"] == fam).sum()) if not band_df.empty else 0 for fam in FAMILY_DISPLAY},
        "written": written,
        "no_tex": True,
        "no_selection": True,
        "cv_span_frozen_before_oos_events": True,
    }
    guard.write_json(output_root / "qa" / "CORE_STATUS.json", payload)
    return payload


def _load_span_df(output_root: Path) -> Dict[str, TransitionSpan]:
    path = output_root / "tables" / "transition_span_summary.csv"
    df = pd.read_csv(path)
    out: Dict[str, TransitionSpan] = {}
    for _, row in df.iterrows():
        blocking = [x for x in str(row.get("blocking_metrics") or "").split(",") if x]
        plateaus = [x for x in str(row.get("plateau_metrics") or "").split(",") if x]
        out[str(row["family"])] = TransitionSpan(
            family=str(row["family"]),
            status=str(row["status"]),
            rho_transition_low=None if pd.isna(row["rho_transition_low"]) else float(row["rho_transition_low"]),
            rho_transition_high=None if pd.isna(row["rho_transition_high"]) else float(row["rho_transition_high"]),
            log10_width=None if pd.isna(row["log10_width"]) else float(row["log10_width"]),
            fraction_of_full_positive_log_grid=None
            if pd.isna(row["fraction_of_full_positive_log_grid"])
            else float(row["fraction_of_full_positive_log_grid"]),
            n_primary_events=int(row["n_primary_events"]),
            n_supporting_events=int(row["n_supporting_events"]),
            blocking_metrics=blocking,
            plateau_metrics=plateaus,
            min_positive_rho=float(row["min_positive_rho"]),
            max_positive_rho=float(row["max_positive_rho"]),
            used_tied_intervals=bool(row["used_tied_intervals"]),
            notes=str(row.get("notes") or ""),
        )
    return out


def _plot_family_path(ax, sub: pd.DataFrame, metric: str, color: str, span: TransitionSpan, min_pos: float) -> None:
    rhos = sub["rho"].to_numpy(dtype=float)
    x = np.array([display_rho(r, min_pos) for r in rhos])
    mean_col = metric_col(metric, "CV_mean")
    if mean_col not in sub.columns:
        ax.set_axis_off()
        return
    y = pd.to_numeric(sub[mean_col], errors="coerce").to_numpy(dtype=float)
    shade_span(ax, span)
    for fid in FOLD_IDS:
        col = fold_col(metric, fid)
        if col not in sub.columns:
            continue
        yf = pd.to_numeric(sub[col], errors="coerce").to_numpy(dtype=float)
        ax.plot(x, yf, color=color, alpha=0.18, lw=0.8, zorder=2)
    ax.plot(x, y, color=color, lw=1.8, marker="o", markersize=3.2, label="CV mean", zorder=3)
    if metric_col(metric, "heldout") in sub.columns:
        ax.plot(x, pd.to_numeric(sub[metric_col(metric, "heldout")], errors="coerce"),
                color=color, ls="--", lw=1.2, marker="s", markersize=3, label="Held-out", zorder=3)
    if metric_col(metric, "forward_2025") in sub.columns:
        ax.plot(x, pd.to_numeric(sub[metric_col(metric, "forward_2025")], errors="coerce"),
                color=color, ls=":", lw=1.2, marker="^", markersize=3.4, label="2025", zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(min_pos * 0.45, float(np.nanmax(rhos)) * 1.15)


def stage_render(repo: Path, result_root: Path, output_root: Path, guard: OutputGuard) -> Dict[str, Any]:
    _load_mpl()
    setup_style()
    freeze_protocol(guard)
    combined = pd.read_csv(load_combined_path_table(result_root))
    spans = _load_span_df(output_root)
    events_cv = pd.read_csv(output_root / "tables" / "transition_events_cv_mean.csv")
    events_fold = pd.read_csv(output_root / "tables" / "transition_events_by_fold.csv")
    lofo = pd.read_csv(output_root / "tables" / "transition_lofo_sensitivity.csv")
    conc = pd.read_csv(output_root / "tables" / "transition_temporal_concordance.csv")
    span_df = pd.read_csv(output_root / "tables" / "transition_span_summary.csv")
    mech = pd.read_csv(output_root / "tables" / "transition_mechanism_summary.csv")
    env_df = pd.read_csv(output_root / "tables" / "transition_region_performance_envelope.csv")
    fold_sum = pd.read_csv(output_root / "tables" / "transition_fold_stability_summary.csv")
    band = pd.read_csv(output_root / "tables" / "transition_band_configs.csv") if (
        output_root / "tables" / "transition_band_configs.csv"
    ).is_file() else pd.DataFrame()
    min_pos = 0.1
    fig_dir = output_root / "figures"
    written: List[str] = []

    # Figure 1 — event map
    fig, axes = plt.subplots(len(PRIMARY_METRICS), 2, figsize=(9.6, 11.2), sharex=True)
    for col, fam in enumerate(FAMILY_DISPLAY):
        sub = family_frame(combined, fam)
        span = spans[fam]
        color = FAMILY_COLOR[fam]
        x = np.array([display_rho(r, min_pos) for r in sub["rho"].to_numpy(dtype=float)])
        for row, (metric, direction) in enumerate(PRIMARY_METRICS):
            ax = axes[row, col]
            shade_span(ax, span)
            y = pd.to_numeric(sub[metric_col(metric, "CV_mean")], errors="coerce").to_numpy(dtype=float)
            ax.plot(x, y, color=color, lw=1.3, zorder=2)
            cv_ev = events_cv.loc[(events_cv["family"] == fam) & (events_cv["metric"] == metric)].iloc[0]
            ax.scatter(
                [display_rho(float(cv_ev["rho_low"]), min_pos)],
                [float(cv_ev["metric_value"])],
                s=70,
                color=color,
                zorder=5,
                label="CV event",
            )
            if not numerically_equal(cv_ev["rho_low"], cv_ev["rho_high"]):
                ax.plot(
                    [display_rho(float(cv_ev["rho_low"]), min_pos), display_rho(float(cv_ev["rho_high"]), min_pos)],
                    [float(cv_ev["metric_value"])] * 2,
                    color=color,
                    lw=3,
                    solid_capstyle="round",
                    zorder=4,
                )
            fold_part = events_fold.loc[(events_fold["family"] == fam) & (events_fold["metric"] == metric)]
            for _, fr in fold_part.iterrows():
                if pd.isna(fr["rho_low"]):
                    continue
                ax.scatter(
                    [display_rho(float(fr["rho_low"]), min_pos)],
                    [float(fr["metric_value"])] if pd.notna(fr["metric_value"]) else [np.nan],
                    s=16,
                    color=color,
                    alpha=0.55,
                    zorder=4,
                )
            valid_lofo = lofo.loc[(lofo["family"] == fam) & (lofo["valid_positive_interior_five_event_span"].astype(bool))]
            if not valid_lofo.empty:
                ax.axvline(float(valid_lofo["rho_transition_low"].min()), color="#6B7280", ls=":", lw=0.7, alpha=0.7)
                ax.axvline(float(valid_lofo["rho_transition_high"].max()), color="#6B7280", ls=":", lw=0.7, alpha=0.7)
            ax.set_xscale("log")
            ylabel = f"{metric} ({direction})"
            ax.set_ylabel(ylabel)
            if row == 0:
                ax.set_title(fam)
            if row == len(PRIMARY_METRICS) - 1:
                ax.set_xlabel(r"$\rho$")
            if span.status != "VALID_POSITIVE_INTERIOR_SPAN":
                ax.text(0.02, 0.92, "FULL_COMMON_SPAN_NOT_SUPPORTED", transform=ax.transAxes, fontsize=6.5, color="#991B1B")
    fig.suptitle("CV transition events (discrete grid). Large markers: equal-weight CV; small: fold events.", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    written.extend(save_figure(fig, fig_dir / "transition_event_map", guard))

    # Figure 2 — compact rho evolution
    compact = ["R2_price", "MAE_price", "RMSE_log", "COD", "Beta_log", "dCor_e_y"]
    fig, axes = plt.subplots(len(compact), 2, figsize=(9.4, 12.4), sharex=True)
    for col, fam in enumerate(FAMILY_DISPLAY):
        sub = family_frame(combined, fam)
        span = spans[fam]
        color = FAMILY_COLOR[fam]
        for row, metric in enumerate(compact):
            ax = axes[row, col]
            _plot_family_path(ax, sub, metric, color, span, min_pos)
            ax.set_ylabel(metric)
            if row == 0:
                ax.set_title(fam)
            if row == len(compact) - 1:
                ax.set_xlabel(r"$\rho$")
            if row == 0 and col == 1:
                ax.legend(loc="best", frameon=False)
    fig.suptitle("CV mean, fold paths, held-out, and 2025 with frozen CV transition span", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    written.extend(save_figure(fig, fig_dir / "transition_rho_evolution", guard))

    # Figure 3 — atlas
    atlas_metrics = [
        "MAPE", "median_ratio", "mean_ratio", "weighted_mean_ratio",
        "PRD", "PRB", "MKI", "VEI", "Beta_log", "Cov_log_residual_log_price", "dCor_e_y", "Delta_NL",
        "R2_price", "MAE_price", "RMSE_log", "COD",
    ]
    atlas_pdf = guard.allowed(fig_dir / "transition_rho_atlas.pdf")
    page_pngs = []
    with PdfPages(atlas_pdf) as pdf:
        chunk = 4
        pages = [atlas_metrics[i:i + chunk] for i in range(0, len(atlas_metrics), chunk)]
        for pi, mets in enumerate(pages, start=1):
            fig, axes = plt.subplots(len(mets), 2, figsize=(9.4, 2.35 * len(mets)), sharex=True)
            if len(mets) == 1:
                axes = np.array([axes])
            for row, metric in enumerate(mets):
                for col, fam in enumerate(FAMILY_DISPLAY):
                    ax = axes[row, col]
                    sub = family_frame(combined, fam)
                    color = FAMILY_COLOR[fam]
                    span = spans[fam]
                    if metric == "Delta_NL":
                        rhos = sub["rho"].to_numpy(dtype=float)
                        xx = np.array([display_rho(r, min_pos) for r in rhos])
                        shade_span(ax, span)
                        for suffix, ls, mk, lab in (
                            ("heldout", "--", "s", "Held-out"),
                            ("forward_2025", ":", "^", "2025"),
                        ):
                            coln = metric_col("Delta_NL", suffix)
                            if coln in sub.columns:
                                ax.plot(xx, pd.to_numeric(sub[coln], errors="coerce"), color=color, ls=ls, marker=mk, lw=1.3, label=lab)
                        ax.set_xscale("log")
                    elif metric_col(metric, "CV_mean") in sub.columns:
                        _plot_family_path(ax, sub, metric, color, span, min_pos)
                    else:
                        ax.set_axis_off()
                        continue
                    ax.set_ylabel(metric)
                    if row == 0:
                        ax.set_title(fam)
                    if row == len(mets) - 1:
                        ax.set_xlabel(r"$\rho$")
            fig.suptitle(f"Appendix rho atlas (page {pi})", fontsize=10)
            fig.tight_layout(rect=[0, 0, 1, 0.96])
            pdf.savefig(fig)
            png_path = guard.allowed(fig_dir / f"transition_rho_atlas_p{pi:02d}.png")
            fig.savefig(png_path, dpi=300, bbox_inches="tight")
            page_pngs.append(str(png_path))
            plt.close(fig)
    written.append(str(atlas_pdf))
    written.extend(page_pngs)

    # Figure 4 — mechanism / nonlinear
    fig, axes = plt.subplots(3, 2, figsize=(9.4, 8.6), sharex=True)
    mech_metrics = [("Beta_log", "CV"), ("Delta_NL", "OOS"), ("dCor_e_y", "CV")]
    for col, fam in enumerate(FAMILY_DISPLAY):
        sub = family_frame(combined, fam)
        span = spans[fam]
        color = FAMILY_COLOR[fam]
        for row, (metric, kind) in enumerate(mech_metrics):
            ax = axes[row, col]
            if kind == "OOS":
                rhos = sub["rho"].to_numpy(dtype=float)
                xx = np.array([display_rho(r, min_pos) for r in rhos])
                shade_span(ax, span)
                for suffix, ls, mk, lab in (("heldout", "--", "s", "Held-out"), ("forward_2025", ":", "^", "2025")):
                    coln = metric_col(metric, suffix)
                    if coln in sub.columns:
                        ax.plot(xx, pd.to_numeric(sub[coln], errors="coerce"), color=color, ls=ls, marker=mk, lw=1.4, label=lab)
                ax.set_xscale("log")
            else:
                _plot_family_path(ax, sub, metric, color, span, min_pos)
            ax.set_ylabel(metric)
            if row == 0:
                ax.set_title(fam)
            if row == 2:
                ax.set_xlabel(r"$\rho$")
            if row == 1 and col == 1:
                ax.legend(loc="best", frameon=False)
    fig.suptitle("First-order mechanism and broader dependence with frozen CV transition span", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    written.extend(save_figure(fig, fig_dir / "transition_mechanism_structure", guard))

    # Figure 5 — accuracy-equity path segments
    pairs = [("R2_price", "PRD"), ("R2_price", "VEI"), ("R2_price", "PRB"), ("R2_price", "MKI")]
    evals = [("CV_mean", "CV mean"), ("heldout", "Held-out"), ("forward_2025", "2025")]
    fig, axes = plt.subplots(len(pairs), 3, figsize=(10.6, 10.8))
    for row, (xm, ym) in enumerate(pairs):
        for col, (suffix, title) in enumerate(evals):
            ax = axes[row, col]
            for fam in FAMILY_DISPLAY:
                sub = family_frame(combined, fam).sort_values("rho")
                xc = metric_col(xm, suffix)
                yc = metric_col(ym, suffix)
                if xc not in sub.columns or yc not in sub.columns:
                    continue
                x = pd.to_numeric(sub[xc], errors="coerce").to_numpy(dtype=float)
                y = pd.to_numeric(sub[yc], errors="coerce").to_numpy(dtype=float)
                color = FAMILY_COLOR[fam]
                ax.plot(x, y, color=color, lw=1.0, alpha=0.45, marker="o", markersize=2.2, label=fam)
                span = spans[fam]
                if span.status == "VALID_POSITIVE_INTERIOR_SPAN":
                    mask = span_segment_mask(sub["rho"].to_numpy(dtype=float), span.rho_transition_low, span.rho_transition_high)
                    if mask.any():
                        ax.plot(x[mask], y[mask], color=color, lw=2.6, marker="o", markersize=3.4, label=f"{fam} CV-span segment")
            if row == 0:
                ax.set_title(title)
            ax.set_xlabel(xm)
            ax.set_ylabel(ym)
            if row == 0 and col == 2:
                ax.legend(fontsize=6, frameon=False)
    fig.suptitle("Ordered-rho accuracy–equity trajectories; thick segment = frozen CV transition span", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    written.extend(save_figure(fig, fig_dir / "transition_accuracy_equity_trajectories", guard))

    # Optional mechanism-normalized figure
    fig, axes = plt.subplots(3, 2, figsize=(9.4, 8.8), sharex=True)
    ymetrics = ["R2_price", "dCor_e_y", "Delta_NL"]
    for col, fam in enumerate(FAMILY_DISPLAY):
        sub = family_frame(combined, fam).sort_values("rho")
        beta0 = lookup_at_rho(sub, metric_col("Beta_log", "CV_mean"), 0.0)
        q_cv = []
        for rho, beta in zip(sub["rho"], pd.to_numeric(sub[metric_col("Beta_log", "CV_mean")], errors="coerce")):
            q, _n = q_ratio(float(beta), beta0) if np.isfinite(beta) and beta0 is not None else (None, "")
            q_cv.append(q)
        att = np.array([np.nan if q is None else 1.0 - q for q in q_cv], dtype=float)
        color = FAMILY_COLOR[fam]
        span = spans[fam]
        for row, metric in enumerate(ymetrics):
            ax = axes[row, col]
            if metric == "Delta_NL":
                y = pd.to_numeric(sub[metric_col("Delta_NL", "heldout")], errors="coerce").to_numpy(dtype=float) if metric_col("Delta_NL", "heldout") in sub.columns else None
                ylab = "Delta_NL (held-out)"
            else:
                y = pd.to_numeric(sub[metric_col(metric, "CV_mean")], errors="coerce").to_numpy(dtype=float)
                ylab = metric
            if y is None:
                ax.set_axis_off()
                continue
            ax.plot(att, y, color=color, lw=1.2, marker="o", markersize=3, alpha=0.7)
            if span.status == "VALID_POSITIVE_INTERIOR_SPAN":
                mask = span_segment_mask(sub["rho"].to_numpy(dtype=float), span.rho_transition_low, span.rho_transition_high)
                ax.plot(att[mask], y[mask], color=color, lw=2.5)
            ax.set_ylabel(ylab)
            if row == 0:
                ax.set_title(fam)
            if row == 2:
                xlab = "1 - q_beta" if fam == "Direct" else "1 - empirical remaining first-order fraction"
                ax.set_xlabel(xlab)
    fig.suptitle("Appendix: metrics versus achieved first-order attenuation (not a selection rule)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    written.extend(save_figure(fig, fig_dir / "transition_mechanism_normalized", guard))

    report_md = build_report(
        result_root=result_root,
        output_root=output_root,
        spans=spans,
        events_cv=events_cv,
        fold_sum=fold_sum,
        lofo=lofo,
        conc=conc,
        mech=mech,
        env_df=env_df,
        band=band,
        span_df=span_df,
    )
    low = report_md.lower()
    hits = [p for p in FORBIDDEN_REPORT_PHRASES if p in low]
    if hits:
        raise RuntimeError(f"report contains forbidden selection language: {hits}")
    guard.write_text(output_root / "report" / "transition_results_report.md", report_md)
    html = build_html_report(report_md, written, env_df)
    guard.write_text(output_root / "report" / "transition_results_report.html", html)

    payload = {
        "status": "PASS",
        "utc": utc_now(),
        "figures": written,
        "report_md": str(output_root / "report" / "transition_results_report.md"),
        "report_html": str(output_root / "report" / "transition_results_report.html"),
        "no_tex": True,
        "forbidden_phrase_hits": hits,
    }
    guard.write_json(output_root / "qa" / "RENDER_STATUS.json", payload)
    return payload


def build_report(
    *,
    result_root: Path,
    output_root: Path,
    spans: Dict[str, TransitionSpan],
    events_cv: pd.DataFrame,
    fold_sum: pd.DataFrame,
    lofo: pd.DataFrame,
    conc: pd.DataFrame,
    mech: pd.DataFrame,
    env_df: pd.DataFrame,
    band: pd.DataFrame,
    span_df: pd.DataFrame,
) -> str:
    pre = json.loads((output_root / "qa" / "PRECHECK_STATUS.json").read_text(encoding="utf-8"))
    identity = pre["identity"]
    lines = [
        "# Transition-region analysis report",
        "",
        "Descriptive analysis of frozen 994-tree CCAO Direct and Surrogate regularization paths.",
        "No rho, penalty family, or penalized configuration is selected or recommended.",
        "",
        "## 1. Canonical experiment identity",
        "",
        f"- Result root: `{result_root}`",
        f"- Baseline gate: `{identity['baseline_gate']}`",
        f"- LightGBM config ID: `{identity['lgbm_config_id']}`",
        f"- data_id: `{identity['data_id']}`",
        f"- split_id: `{identity['split_id']}`",
        f"- seed: `{identity['seed']}`",
        f"- estimator count: `{identity['n_estimators']}`",
        f"- expected CV fits: `{identity['n_cv_fits']}`",
        f"- held-out identity: `{identity['heldout_identity']}`",
        f"- 2025 identity: `{identity['forward_2025_identity']}`",
        "",
        "## 2. Provenance gate status",
        "",
        f"- PRECHECK: `{pre['status']}`",
        f"- Protocol SHA256: `{pre['protocol']['sha256']}`",
        f"- Historical 500-tree root used: no",
        "",
    ]
    for fam in FAMILY_DISPLAY:
        lines += [f"## Primary CV events: {fam}", ""]
        part = events_cv.loc[events_cv["family"] == fam]
        lines.append(df_to_markdown(part[[
            "metric", "expected_optimization_direction", "classification",
            "rho_low", "rho_high", "metric_value", "local_turn_verified", "n_tied", "notes",
        ]]))
        lines.append("")
        sp = spans[fam]
        lines += [
            f"## CV transition span: {fam}",
            "",
            f"- Status: `{sp.status}`",
            f"- Label: CV-derived descriptive transition span",
            f"- rho_transition_low: `{sp.rho_transition_low}`",
            f"- rho_transition_high: `{sp.rho_transition_high}`",
            f"- log10_width: `{sp.log10_width}`",
            f"- fraction_of_full_positive_log_grid: `{sp.fraction_of_full_positive_log_grid}`",
            f"- blocking metrics: {sp.blocking_metrics or 'none'}",
            f"- plateau metrics: {sp.plateau_metrics or 'none'}",
            "",
        ]
    lines += ["## 7. Fold-specific stability (log-rho; temporal diagnostics, not IID replicates)", "", df_to_markdown(fold_sum), ""]
    lines += ["## 8. Leave-one-fold-out stability", "", df_to_markdown(lofo), ""]
    for fam in FAMILY_DISPLAY:
        sub = lofo.loc[lofo["family"] == fam]
        n_valid = int(sub["valid_positive_interior_five_event_span"].astype(bool).sum()) if not sub.empty else 0
        lines.append(f"- {fam}: {n_valid} / {len(sub)} LOFO replications retain an all-five positive-interior span.")
        if n_valid:
            v = sub.loc[sub["valid_positive_interior_five_event_span"].astype(bool)]
            lines.append(
                f"  - lower endpoints [{v['rho_transition_low'].min():.6g}, {v['rho_transition_low'].max():.6g}]; "
                f"upper [{v['rho_transition_high'].min():.6g}, {v['rho_transition_high'].max():.6g}]; "
                f"log-width [{v['log10_width'].min():.6g}, {v['log10_width'].max():.6g}]"
            )
    lines += ["", "## 9–10. Retrospective temporal concordance", ""]
    lines.append("Held-out and 2025 paths were already inspected during paper development. This is not prospective confirmation.")
    lines.append("")
    for fam in FAMILY_DISPLAY:
        for split, label in (("heldout", "held-out"), ("forward_2025", "2025")):
            part = conc.loc[(conc["family"] == fam) & (conc["split"] == split)]
            n_in = int(part["inside_frozen_cv_span"].fillna(False).sum()) if not part.empty else 0
            lines.append(f"- {fam} {label}: **{n_in} / {len(part)}** events inside the frozen CV transition span")
    lines.append("")
    lines.append(df_to_markdown(conc))
    lines += ["", "## 11. First-order mechanism (q_beta / q_cov) at CV endpoints", ""]
    lines.append(
        "Raw rho is not comparable across families. Direct q_beta is the remaining first-order slope fraction. "
        "Surrogate uses the same ratio only as an empirical remaining first-order fraction; the Direct projection formula is not applied."
    )
    lines.append("")
    if not mech.empty:
        endp = mech.loc[mech["location"].isin(["span_low", "span_high"])]
        show = endp if not endp.empty else mech
        cols = [c for c in (
            "family", "location", "rho", "split", "quantity_name", "q_beta", "q_cov",
            "attenuation_1_minus_q_beta", "overcorrection", "q_beta_q_cov_agree", "q_agree_note"
        ) if c in show.columns]
        lines.append(df_to_markdown(show[cols]))
    lines += ["", "## 12. Configurations inside each CV transition span", ""]
    if band.empty:
        lines.append("No valid span, or no band rows.")
    else:
        for fam in FAMILY_DISPLAY:
            n = int((band["family"] == fam).sum())
            lines.append(f"- {fam}: {n} canonical rho-grid configurations")
    lines += ["", "## 13. Performance envelopes", ""]
    lines.append("Component-wise min/max across span members; they do not identify a unique penalized rho.")
    lines.append("")
    lines.append(df_to_markdown(env_df))
    lines += [
        "",
        "## 14. Scientific interpretation",
        "",
        "Three distinct objects:",
        "",
        "1. **CV transition span** — a descriptive path phenomenon on equal-weight chronological CV for the five frozen primary metrics.",
        "2. **First-order mechanism scale** — achieved attenuation of Beta_log / covariance, summarized by signed q.",
        "3. **Fixed-space theory** — a mechanism benchmark only. No canonical 994-tree projection-theory artifact was found under the result root, so theory q is not reported as an empirical turning-point explanation.",
        "",
        "Theory can explain why targeted first-order dependence should attenuate systematically. It does not predict the empirical R2 maximum, MAE/MAPE/RMSE_log/COD minima, or high-rho nonlinear behavior.",
        "",
        "The five primary metrics arise from the same predictions and are not five independent statistical confirmations.",
        "No new selection rule is introduced. A wide span, a missing interior event, unstable LOFO, or poor held-out/2025 concordance is reported as such.",
        "",
        "## 15. Limitations",
        "",
        "- Discrete events on a 51-point grid; no interpolation.",
        "- Seven expanding chronological folds are temporal diagnostics, not IID replicates.",
        "- Held-out/2025 concordance is retrospective.",
        "- Delta_NL is available only for held-out and 2025; fold-level CV Delta_NL was not computed.",
        "- Corr_log_residual_log_price is omitted if absent from the canonical combined path table.",
        "- Direct and Surrogate raw rho values are not commensurate.",
        "- Historical 500-tree ranges and six-county rho values were not used.",
        "",
        f"Output root: `{output_root}`",
        "",
    ]
    return "\n".join(lines) + "\n"


def build_html_report(md: str, figures: Sequence[str], env_df: pd.DataFrame) -> str:
    img_tags = []
    for path in figures:
        p = Path(path)
        if p.suffix.lower() == ".png":
            rel = os.path.relpath(p, start=str(Path(path).parents[1] / "report")) if False else f"../figures/{p.name}"
            img_tags.append(f'<h3>{p.stem}</h3><p><img src="../figures/{p.name}" alt="{p.stem}" style="max-width:100%;"></p>')
    env_html = env_df.to_html(index=False, border=0) if env_df is not None else ""
    escaped = (
        md.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    )
    return (
        "<!DOCTYPE html><html><head><meta charset='utf-8'><title>Transition-region report</title>"
        "<style>body{font-family:Georgia,serif;max-width:980px;margin:24px auto;padding:0 16px;color:#111}"
        "img{border:1px solid #e5e7eb} table{border-collapse:collapse;font-size:13px}"
        "td,th{border:1px solid #d1d5db;padding:4px 6px} pre{white-space:pre-wrap}</style></head><body>"
        "<h1>Transition-region analysis</h1>"
        "<p>Descriptive CV transition span. No configuration is selected.</p>"
        + "".join(img_tags)
        + "<h2>Envelope table</h2>"
        + env_html
        + "<h2>Markdown source</h2><pre>"
        + escaped
        + "</pre></body></html>\n"
    )


def _file_nonzero(path: Path, min_bytes: int = 64) -> bool:
    return path.is_file() and path.stat().st_size >= min_bytes


def stage_audit(repo: Path, result_root: Path, output_root: Path, guard: OutputGuard) -> Dict[str, Any]:
    problems: List[str] = []
    identity = validate_canonical_result_root(result_root)
    combined = pd.read_csv(load_combined_path_table(result_root))
    expected_rhos = expected_canonical_rhos()
    count_chk = validate_combined_counts(combined, expected_rhos)
    problems.extend(count_chk["problems"])

    required = [
        "tables/transition_events_cv_mean.csv",
        "tables/transition_span_summary.csv",
        "tables/transition_span_summary.parquet",
        "tables/transition_events_by_fold.csv",
        "tables/transition_lofo_sensitivity.csv",
        "tables/transition_temporal_concordance.csv",
        "tables/transition_temporal_concordance.parquet",
        "tables/transition_mechanism_summary.csv",
        "tables/transition_mechanism_summary.parquet",
        "tables/transition_band_configs.csv",
        "tables/transition_band_configs.parquet",
        "tables/transition_region_performance_envelope.csv",
        "tables/transition_region_performance_envelope.parquet",
        "tables/transition_region_performance_envelope.md",
        "figures/transition_event_map.pdf",
        "figures/transition_event_map.png",
        "figures/transition_rho_evolution.pdf",
        "figures/transition_rho_evolution.png",
        "figures/transition_rho_atlas.pdf",
        "figures/transition_mechanism_structure.pdf",
        "figures/transition_mechanism_structure.png",
        "figures/transition_accuracy_equity_trajectories.pdf",
        "figures/transition_accuracy_equity_trajectories.png",
        "report/transition_results_report.md",
        "protocol/transition_analysis_protocol.json",
        "qa/PRECHECK_STATUS.json",
        "qa/TEST_STATUS.json",
        "qa/CORE_STATUS.json",
        "qa/RENDER_STATUS.json",
        "provenance/input_manifest.json",
        "provenance/tex_baseline_sha256.json",
    ]
    for rel in required:
        if not _file_nonzero(output_root / rel):
            problems.append(f"missing or empty {rel}")

    tex_in_output = list(output_root.rglob("*.tex"))
    if tex_in_output:
        problems.append("tex files present under output root: " + ", ".join(str(p) for p in tex_in_output))

    baseline_tex = json.loads((output_root / "provenance" / "tex_baseline_sha256.json").read_text(encoding="utf-8"))
    current_tex = {str(p.relative_to(repo)): sha256_file(p) for p in iter_tex_files(repo)}
    if current_tex != baseline_tex:
        problems.append("tracked/existing .tex hashes changed or new .tex appeared")
    paper_base = json.loads((output_root / "provenance" / "paper_tree_sha256.json").read_text(encoding="utf-8"))
    paper_now = {str(p.relative_to(repo)): sha256_file(p) for p in (repo / "paper").rglob("*") if p.is_file()}
    if paper_now != paper_base:
        problems.append("paper/ tree changed during this analysis")

    in_man = json.loads((output_root / "provenance" / "input_manifest.json").read_text(encoding="utf-8"))
    for rel, old in (in_man.get("input_sha256") or {}).items():
        path = result_root / rel
        if old is None:
            continue
        if not path.is_file():
            problems.append(f"canonical input disappeared: {rel}")
            continue
        if sha256_file(path) != old:
            problems.append(f"canonical input changed: {rel}")

    events_cv = pd.read_csv(output_root / "tables" / "transition_events_cv_mean.csv")
    span_df = pd.read_csv(output_root / "tables" / "transition_span_summary.csv")
    conc = pd.read_csv(output_root / "tables" / "transition_temporal_concordance.csv")
    mech = pd.read_csv(output_root / "tables" / "transition_mechanism_summary.csv")
    recon_spans = {}
    for fam in FAMILY_DISPLAY:
        sub = family_frame(combined, fam)
        evs = extract_primary_events_from_frame(sub, "CV_mean")
        span = _span_from_events(fam, evs, expected_rhos)
        recon_spans[fam] = span
        saved = events_cv.loc[events_cv["family"] == fam]
        for ev in evs:
            row = saved.loc[saved["metric"] == ev.metric].iloc[0]
            if str(row["classification"]) != ev.classification:
                problems.append(f"{fam} {ev.metric} classification mismatch")
            if ev.rho_low is not None and not numerically_equal(row["rho_low"], ev.rho_low):
                problems.append(f"{fam} {ev.metric} rho_low mismatch")
        saved_span = span_df.loc[span_df["family"] == fam].iloc[0]
        if str(saved_span["status"]) != span.status:
            problems.append(f"{fam} span status mismatch")
        if span.status == "VALID_POSITIVE_INTERIOR_SPAN":
            if not any(numerically_equal(span.rho_transition_low, r) for r in sub["rho"]):
                problems.append(f"{fam} span low is not a grid point")
            if not any(numerically_equal(span.rho_transition_high, r) for r in sub["rho"]):
                problems.append(f"{fam} span high is not a grid point")
            oos = extract_primary_events_from_frame(sub, "heldout")
            for ev in oos:
                row = conc.loc[(conc["family"] == fam) & (conc["split"] == "heldout") & (conc["metric"] == ev.metric)].iloc[0]
                recon = concordance_row(fam, "heldout", ev, span)
                if bool(row["inside_frozen_cv_span"]) != bool(recon["inside_frozen_cv_span"]):
                    problems.append(f"{fam} held-out membership mismatch for {ev.metric}")
            oos2 = extract_primary_events_from_frame(sub, "forward_2025")
            for ev in oos2:
                row = conc.loc[(conc["family"] == fam) & (conc["split"] == "forward_2025") & (conc["metric"] == ev.metric)].iloc[0]
                recon = concordance_row(fam, "forward_2025", ev, span)
                if bool(row["inside_frozen_cv_span"]) != bool(recon["inside_frozen_cv_span"]):
                    problems.append(f"{fam} 2025 membership mismatch for {ev.metric}")
            for loc in ("span_low", "span_high"):
                part = mech.loc[(mech["family"] == fam) & (mech["location"] == loc) & (mech["split"] == "CV_mean")]
                if part.empty:
                    continue
                rho = float(part.iloc[0]["rho"])
                recon_q = _q_block(fam, sub, rho, "CV_mean", loc)
                if not numerically_equal(part.iloc[0]["q_beta"], recon_q["q_beta"]) and not (
                    pd.isna(part.iloc[0]["q_beta"]) and recon_q["q_beta"] is None
                ):
                    problems.append(f"{fam} {loc} q_beta mismatch")

    status = "PASS" if not problems else "FAIL"
    payload = {
        "status": status,
        "utc": utc_now(),
        "problems": problems,
        "identity": {
            "data_id": identity["data_id"],
            "split_id": identity["split_id"],
            "lgbm_config_id": identity["lgbm_config_id"],
        },
        "n_artifacts_hashed": 0,
        "tarball": None,
        "no_tex_created_or_modified": (not tex_in_output) and ("tex hashes" not in " ".join(problems)) and ("new .tex" not in " ".join(problems)),
        "no_paper_writes": "paper/ tree changed" not in " ".join(problems),
        "recon_span_status": {fam: recon_spans[fam].status for fam in recon_spans},
    }
    guard.write_json(output_root / "qa" / "FINAL_STATUS.json", payload)
    guard.write_json(output_root / "provenance" / "final_result_manifest.json", payload)

    artifact_hashes = {}
    for p in sorted(output_root.rglob("*")):
        if p.is_file() and p.name not in {"SHA256SUMS", "transition_regions_v1_bundle.tar.gz"}:
            try:
                rel = str(p.relative_to(output_root))
            except ValueError:
                continue
            artifact_hashes[rel] = sha256_file(p)
    sums_txt = "\n".join(f"{h}  {rel}" for rel, h in sorted(artifact_hashes.items())) + "\n"
    guard.write_text(output_root / "provenance" / "SHA256SUMS", sums_txt)
    payload["n_artifacts_hashed"] = len(artifact_hashes)
    guard.write_json(output_root / "qa" / "FINAL_STATUS.json", payload)

    tarball = None
    try:
        import tarfile

        tar_path = output_root / "provenance" / "transition_regions_v1_bundle.tar.gz"
        with tarfile.open(tar_path, "w:gz") as tar:
            for p in sorted(output_root.rglob("*")):
                if not p.is_file():
                    continue
                if p.name == "transition_regions_v1_bundle.tar.gz":
                    continue
                tar.add(p, arcname=str(p.relative_to(output_root.parent)))
        tarball = str(tar_path)
        payload["tarball"] = tarball
        guard.write_json(output_root / "qa" / "FINAL_STATUS.json", payload)
    except Exception as err:
        problems.append(f"tarball failed: {err}")
        payload["problems"] = problems
        payload["status"] = "FAIL"
        guard.write_json(output_root / "qa" / "FINAL_STATUS.json", payload)
        status = "FAIL"

    if status != "PASS":
        raise RuntimeError("FINAL_AUDIT failed: " + "; ".join(problems))
    return payload


def main() -> int:
    args = parse_args()
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MPLBACKEND", "Agg")
    repo, result_root, output_root, guard = resolve_paths(args)
    stages = {
        "precheck": stage_precheck,
        "tests": stage_tests,
        "core": stage_core,
        "render": stage_render,
        "audit": stage_audit,
    }
    try:
        payload = stages[args.stage](repo, result_root, output_root, guard)
    except Exception as err:
        qa_name = {
            "precheck": "PRECHECK_STATUS.json",
            "tests": "TEST_STATUS.json",
            "core": "CORE_STATUS.json",
            "render": "RENDER_STATUS.json",
            "audit": "FINAL_STATUS.json",
        }[args.stage]
        fail = {
            "status": "FAIL",
            "utc": utc_now(),
            "error": str(err),
            "traceback": traceback.format_exc(),
            "stage": args.stage,
        }
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
    # Skip atexit/finalizer hangs (matplotlib font cache on NFS) so Slurm sees the exit code.
    os._exit(rc)
