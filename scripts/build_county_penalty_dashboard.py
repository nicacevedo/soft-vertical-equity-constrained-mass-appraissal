#!/usr/bin/env python3
"""Build a self-contained HTML report for the six-county LGBCovPenalty comparison.

Reads the artifacts written by ``scripts/other_counties_benchmars.py`` for each
county run folder and assembles one page covering sample construction, the
baseline vertical-equity diagnosis, the theory calibration of rho, the accuracy /
equity trade-off, and the recommended operating point per county.

Every figure is a base64 PNG and every table is inlined, so the output file can be
opened or emailed on its own.

Usage:
    python scripts/build_county_penalty_dashboard.py \
        --run output/county_bench_17031_floor50000 \
        --run output/county_bench_42003_floor50000 \
        --run output/county_bench_04013_floor50000 \
        --run output/county_bench_53033_floor50000 \
        --run output/county_bench_12086_floor50000 \
        --run output/county_bench_25017_floor50000 \
        --out output/county_penalty/county_penalty_report.html
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# Reused so the recalibration strawman below is scored by exactly the same code as
# every model in the comparison table.  The schema and transformer hash are also
# imported from the producer so this report cannot silently combine artifacts
# written by an older experiment implementation.
from scripts.other_counties_benchmars import (  # noqa: E402
    ARTIFACT_SCHEMA_VERSION,
    NEIGHBOR_TRANSFORMER_PROVENANCE,
    score_predictions,
)

OUT_HTML = ROOT / "output/county_penalty/county_penalty_report.html"

COUNTY_NAMES = {
    "17031": "Cook, IL",
    "42003": "Allegheny, PA",
    "04013": "Maricopa, AZ",
    "53033": "King, WA",
    "12086": "Miami-Dade, FL",
    "25017": "Middlesex, MA",
}
COUNTY_COLORS = {
    "17031": "#1d4ed8",
    "42003": "#b45309",
    "04013": "#047857",
    "53033": "#be123c",
    "12086": "#7c3aed",
    "25017": "#0f766e",
}
BASELINE_KEY = "lgbm_baseline"
CONTROL_KEY = "cov_rho_0"
EXPECTED_NEIGHBOR_KEYS = ("geo", "geo_time", "geo_time_features")
REQUIRED_PAIR_IDS = {
    "baseline_penalty_vs_baseline",
    "scaling_vs_baseline",
    "neighbor_vs_baseline",
    "neighbor_penalty_vs_neighbor",
}
REQUIRED_PROVENANCE_PARAMETERS = {"max_distance_km", "max_time_distance_days"}
# IAAO Standard on Ratio Studies acceptance ranges for single-family residential.
IAAO_PRD = (0.98, 1.03)
IAAO_PRB = (-0.05, 0.05)
IAAO_COD_MAX = 15.0
LEVEL_PLOT_START = pd.Timestamp("2016-01-01")
LEVEL_PLOT_END = pd.Timestamp("2027-01-01")
# Metrics shown in the headline comparison table, with the direction that is better.
HEADLINE_METRICS = [
    ("R2 (log)", "higher"), ("RMSE (log)", "lower"), ("MAPE", "lower"),
    ("COD", "lower"), ("PRD", "toward_one"), ("PRB", "toward_zero"),
    ("MKI", "toward_one"), ("Corr(r,logprice)", "toward_zero"),
    ("Cov(e,logprice)", "toward_zero"),
]


@dataclass
class CountyRun:
    fips: str
    label: str
    folder: Path
    report: dict
    metrics: pd.DataFrame
    rho_plan: pd.DataFrame
    bootstrap: pd.DataFrame
    draws: pd.DataFrame
    predictions: pd.DataFrame
    train_predictions: pd.DataFrame
    candidates: pd.DataFrame
    local_equity_summary: pd.DataFrame
    local_equity_groups: pd.DataFrame

    @property
    def test(self) -> pd.DataFrame:
        return self.metrics.loc[self.metrics["split"].eq("test")].copy()

    @property
    def baseline(self) -> pd.Series:
        return self.test.set_index("model_key").loc[BASELINE_KEY]

    @property
    def theory(self) -> dict:
        return self.report["comparison"]["theory"]


@dataclass(frozen=True)
class PendingRun:
    """A requested county whose artifacts are not yet complete enough to report."""
    fips: str
    label: str
    folder: Path
    reason: str


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _require_mapping(value, *, path: str, folder: Path) -> dict:
    if not isinstance(value, dict):
        raise ValueError(f"{folder} requires object metadata at {path}.")
    return value


def _require_string(value, *, path: str, folder: Path) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{folder} requires a non-empty string at {path}.")
    return value


def _require_nash_selection(value, *, path: str, folder: Path) -> dict:
    """Require persisted evidence that a selector used the shared four-metric rule."""
    selection = _require_mapping(value, path=path, folder=folder)
    if selection.get("name") != "pareto_nash_hyperrectangle_v1":
        raise ValueError(f"{folder} {path} does not use the required Pareto–Nash selection rule.")
    objectives = selection.get("objectives")
    if not isinstance(objectives, list) or set(objectives) != {"MAPE", "COD", "abs(PRD-1)", "abs(PRB)"}:
        raise ValueError(f"{folder} {path} lacks the required MAPE/COD/PRD/PRB objectives.")
    if selection.get("reference_population") != "pareto_frontier":
        raise ValueError(
            f"{folder} {path} does not define its Nash reference on the Pareto frontier."
        )
    return selection


def _require_files(folder: Path, names: list[str]) -> None:
    missing = [name for name in names if not (folder / name).is_file() or (folder / name).stat().st_size == 0]
    if missing:
        raise ValueError(f"{folder} is missing required dashboard artifacts: {missing}.")


def _validate_report_header(report: dict, folder: Path) -> None:
    """Reject stale/incomplete artifact families before loading their large tables."""
    if report.get("artifact_schema_version") != ARTIFACT_SCHEMA_VERSION:
        raise ValueError(
            f"{folder} has artifact schema {report.get('artifact_schema_version')!r}; "
            f"dashboard requires {ARTIFACT_SCHEMA_VERSION}. Rerun the county benchmark."
        )
    _require_mapping(report.get("sample"), path="sample", folder=folder)
    base_selection = _require_mapping(report.get("selection"), path="selection", folder=folder)
    _require_nash_selection(
        base_selection.get("selection"), path="selection.selection", folder=folder,
    )
    comparison = _require_mapping(report.get("comparison"), path="comparison", folder=folder)
    for field in (
        "selection_protocol", "baseline_penalty", "linear_recalibration", "neighbor_search",
        "neighbor_models", "selected_neighbor_model_key", "selected_neighbor_penalty_model_key",
        "neighbor_transformer_provenance", "paired_comparisons", "model_manifest", "local_equity",
        "log_baseline_candidate_selection",
    ):
        if field not in comparison:
            raise ValueError(f"{folder} lacks comparison.{field}; rerun the county benchmark.")
    if not isinstance(comparison.get("n_bootstrap"), int) or comparison["n_bootstrap"] < 1:
        raise ValueError(f"{folder} lacks a positive comparison.n_bootstrap value.")

    protocol = _require_mapping(comparison["selection_protocol"], path="comparison.selection_protocol", folder=folder)
    if protocol.get("name") != "pareto_nash_hyperrectangle_v1":
        raise ValueError(f"{folder} does not record the required Pareto–Nash selection rule.")
    if protocol.get("validation_split") != "chronological_validation" or protocol.get("test_used_for_selection") is not False:
        raise ValueError(f"{folder} does not record test-free model selection.")
    objectives = protocol.get("objectives")
    if not isinstance(objectives, list) or set(objectives) != {"MAPE", "COD", "abs(PRD-1)", "abs(PRB)"}:
        raise ValueError(f"{folder} lacks the required MAPE/COD/PRD/PRB selection objectives.")
    _require_nash_selection(
        comparison["log_baseline_candidate_selection"],
        path="comparison.log_baseline_candidate_selection", folder=folder,
    )

    baseline_penalty = _require_mapping(comparison["baseline_penalty"], path="comparison.baseline_penalty", folder=folder)
    _require_string(baseline_penalty.get("selected_model_key"), path="comparison.baseline_penalty.selected_model_key", folder=folder)
    if baseline_penalty.get("selection_split") != "chronological_validation":
        raise ValueError(f"{folder} baseline penalty was not selected on chronological validation.")
    _require_mapping(
        baseline_penalty.get("selected_validation_metrics"),
        path="comparison.baseline_penalty.selected_validation_metrics", folder=folder,
    )
    _require_nash_selection(
        baseline_penalty.get("selection"), path="comparison.baseline_penalty.selection", folder=folder,
    )
    scaling = _require_mapping(comparison["linear_recalibration"], path="comparison.linear_recalibration", folder=folder)
    _require_string(scaling.get("selected_model_key"), path="comparison.linear_recalibration.selected_model_key", folder=folder)
    if scaling.get("selection_rule") != protocol["name"]:
        raise ValueError(f"{folder} first-degree scaling does not record the common Pareto–Nash rule.")
    if not isinstance(scaling.get("candidates"), list) or not scaling["candidates"]:
        raise ValueError(f"{folder} lacks first-degree scaling candidates.")
    _require_nash_selection(
        scaling.get("selection"), path="comparison.linear_recalibration.selection", folder=folder,
    )

    provenance = _require_mapping(
        comparison["neighbor_transformer_provenance"], path="comparison.neighbor_transformer_provenance", folder=folder,
    )
    _require_string(provenance.get("source_sha256"), path="comparison.neighbor_transformer_provenance.source_sha256", folder=folder)
    parameters = provenance.get("required_constructor_parameters")
    if not isinstance(parameters, list) or not REQUIRED_PROVENANCE_PARAMETERS.issubset(parameters):
        raise ValueError(f"{folder} lacks the capped-neighbor transformer provenance.")
    if provenance["source_sha256"] != NEIGHBOR_TRANSFORMER_PROVENANCE["source_sha256"]:
        raise ValueError(
            f"{folder} was generated with a different neighbor transformer source hash; rerun the county benchmark."
        )

    neighbor_models = comparison["neighbor_models"]
    if not isinstance(neighbor_models, list):
        raise ValueError(f"{folder} comparison.neighbor_models must be a list.")
    neighbor_by_key: dict[str, dict] = {}
    for position, neighbor in enumerate(neighbor_models):
        neighbor = _require_mapping(neighbor, path=f"comparison.neighbor_models[{position}]", folder=folder)
        key = _require_string(neighbor.get("key"), path=f"comparison.neighbor_models[{position}].key", folder=folder)
        if key in neighbor_by_key:
            raise ValueError(f"{folder} has duplicate neighbor representation {key!r}.")
        _require_string(neighbor.get("label"), path=f"comparison.neighbor_models[{position}].label", folder=folder)
        _require_string(neighbor.get("lgbm_model_key"), path=f"comparison.neighbor_models[{position}].lgbm_model_key", folder=folder)
        validation = _require_mapping(
            neighbor.get("validation"), path=f"comparison.neighbor_models[{position}].validation", folder=folder,
        )
        specification = _require_mapping(
            validation.get("specification"),
            path=f"comparison.neighbor_models[{position}].validation.specification", folder=folder,
        )
        for field in ("k", "max_distance_km", "max_time_distance_days", "time_weight", "feature_weight"):
            if field not in specification:
                raise ValueError(f"{folder} lacks neighbor validation specification field {field!r}.")
        _require_mapping(
            validation.get("metrics"), path=f"comparison.neighbor_models[{position}].validation.metrics", folder=folder,
        )
        _require_mapping(
            neighbor.get("final_feature_metadata"),
            path=f"comparison.neighbor_models[{position}].final_feature_metadata", folder=folder,
        )
        metadata = neighbor["final_feature_metadata"]
        if metadata.get("candidate_multiplier_used") is not False or not str(
            metadata.get("candidate_retrieval", "")
        ).startswith("exact"):
            raise ValueError(
                f"{folder} neighbor representation {key!r} does not record exact capped candidate retrieval."
            )
        neighbor_by_key[key] = neighbor
    if set(neighbor_by_key) != set(EXPECTED_NEIGHBOR_KEYS):
        raise ValueError(
            f"{folder} neighbor representations must be {list(EXPECTED_NEIGHBOR_KEYS)}, "
            f"not {sorted(neighbor_by_key)}."
        )
    selected_neighbors = [row for row in neighbor_by_key.values() if row.get("selected_representation") is True]
    if len(selected_neighbors) != 1:
        raise ValueError(f"{folder} must have exactly one selected neighbor representation.")
    selected_neighbor = selected_neighbors[0]
    selected_penalty_key = _require_string(
        selected_neighbor.get("penalized_model_key"), path="selected neighbor penalized_model_key", folder=folder,
    )
    selected_penalty = _require_mapping(
        selected_neighbor.get("penalty_selection"), path="selected neighbor penalty_selection", folder=folder,
    )
    if selected_penalty.get("selection_split") != "chronological_validation":
        raise ValueError(f"{folder} selected neighbor penalty was not selected on chronological validation.")
    _require_mapping(
        selected_penalty.get("selected_validation_metrics"),
        path="selected neighbor penalty_selection.selected_validation_metrics", folder=folder,
    )
    _require_nash_selection(
        selected_penalty.get("selection"), path="selected neighbor penalty_selection.selection", folder=folder,
    )
    for key, neighbor in neighbor_by_key.items():
        penalty_key = neighbor.get("penalized_model_key")
        if key == selected_neighbor["key"]:
            continue
        if penalty_key is not None:
            raise ValueError(f"{folder} has a penalty model for unselected neighbor representation {key!r}.")
    if comparison["selected_neighbor_model_key"] != selected_neighbor["lgbm_model_key"]:
        raise ValueError(f"{folder} selected_neighbor_model_key disagrees with neighbor_models.")
    if comparison["selected_neighbor_penalty_model_key"] != selected_penalty_key:
        raise ValueError(f"{folder} selected_neighbor_penalty_model_key disagrees with neighbor_models.")

    neighbor_search = _require_mapping(comparison["neighbor_search"], path="comparison.neighbor_search", folder=folder)
    shared_specification = _require_mapping(
        neighbor_search.get("shared_selected_specification"),
        path="comparison.neighbor_search.shared_selected_specification", folder=folder,
    )
    _require_string(
        shared_specification.get("key"),
        path="comparison.neighbor_search.shared_selected_specification.key", folder=folder,
    )
    search_candidates = neighbor_search.get("search_candidates")
    if not isinstance(search_candidates, list) or len(search_candidates) != 6:
        raise ValueError(f"{folder} lacks the required bounded neighbor hyperparameter search.")
    candidate_spec_keys = {candidate.get("key") for candidate in search_candidates if isinstance(candidate, dict)}
    if shared_specification["key"] not in candidate_spec_keys:
        raise ValueError(f"{folder} selected neighbor specification is absent from the recorded search grid.")
    if neighbor_search.get("selected_representation_key") != selected_neighbor["key"]:
        raise ValueError(f"{folder} selected neighbor representation disagrees with neighbor search metadata.")
    for key, neighbor in neighbor_by_key.items():
        if neighbor["validation"]["specification"].get("key") != shared_specification["key"]:
            raise ValueError(f"{folder} neighbor representation {key!r} does not use the shared selected specification.")
    _require_nash_selection(
        neighbor_search.get("search_selection"), path="comparison.neighbor_search.search_selection", folder=folder,
    )
    _require_nash_selection(
        neighbor_search.get("representation_selection"),
        path="comparison.neighbor_search.representation_selection", folder=folder,
    )

    manifest = _require_mapping(comparison["model_manifest"], path="comparison.model_manifest", folder=folder)
    for field in ("expected_model_keys", "expected_neighbor_model_keys", "bootstrap_model_keys", "test_oracle_frontier_model_keys"):
        values = manifest.get(field)
        if not isinstance(values, list) or not values or not all(isinstance(value, str) and value for value in values):
            raise ValueError(f"{folder} has invalid comparison.model_manifest.{field}.")
        if len(values) != len(set(values)):
            raise ValueError(f"{folder} has duplicate keys in comparison.model_manifest.{field}.")
    expected_neighbor_model_keys = {
        *(neighbor["lgbm_model_key"] for neighbor in neighbor_by_key.values()), selected_penalty_key,
    }
    if set(manifest["expected_neighbor_model_keys"]) != expected_neighbor_model_keys:
        raise ValueError(f"{folder} expected_neighbor_model_keys disagrees with neighbor_models.")
    expected_model_keys = set(manifest["expected_model_keys"])
    required_model_keys = {
        BASELINE_KEY, baseline_penalty["selected_model_key"], scaling["selected_model_key"],
        *expected_neighbor_model_keys,
    }
    if not required_model_keys.issubset(expected_model_keys):
        raise ValueError(f"{folder} expected_model_keys omits one or more required selected comparator models.")
    if not set(manifest["test_oracle_frontier_model_keys"]).issubset(expected_model_keys):
        raise ValueError(f"{folder} test-oracle frontier includes models absent from expected_model_keys.")

    pairs = comparison["paired_comparisons"]
    if not isinstance(pairs, list):
        raise ValueError(f"{folder} comparison.paired_comparisons must be a list.")
    pair_ids = set()
    for position, pair in enumerate(pairs):
        pair = _require_mapping(pair, path=f"comparison.paired_comparisons[{position}]", folder=folder)
        pair_id = _require_string(pair.get("id"), path=f"comparison.paired_comparisons[{position}].id", folder=folder)
        if pair_id in pair_ids:
            raise ValueError(f"{folder} has duplicate paired-comparison id {pair_id!r}.")
        pair_ids.add(pair_id)
        _require_string(pair.get("label"), path=f"comparison.paired_comparisons[{position}].label", folder=folder)
        _require_string(pair.get("reference_model_key"), path=f"comparison.paired_comparisons[{position}].reference_model_key", folder=folder)
        _require_string(pair.get("candidate_model_key"), path=f"comparison.paired_comparisons[{position}].candidate_model_key", folder=folder)
    if pair_ids != REQUIRED_PAIR_IDS:
        raise ValueError(f"{folder} paired comparisons must be {sorted(REQUIRED_PAIR_IDS)}, not {sorted(pair_ids)}.")
    pair_by_id = {pair["id"]: pair for pair in pairs}
    expected_pairs = {
        "baseline_penalty_vs_baseline": (BASELINE_KEY, baseline_penalty["selected_model_key"]),
        "scaling_vs_baseline": (BASELINE_KEY, scaling["selected_model_key"]),
        "neighbor_vs_baseline": (BASELINE_KEY, selected_neighbor["lgbm_model_key"]),
        "neighbor_penalty_vs_neighbor": (selected_neighbor["lgbm_model_key"], selected_penalty_key),
    }
    for pair_id, (reference_key, candidate_key) in expected_pairs.items():
        pair = pair_by_id[pair_id]
        if pair["reference_model_key"] != reference_key or pair["candidate_model_key"] != candidate_key:
            raise ValueError(f"{folder} paired comparison {pair_id!r} does not match the selected model keys.")
    pair_model_keys = {
        key for pair in pairs for key in (pair["reference_model_key"], pair["candidate_model_key"])
    }
    if set(manifest["bootstrap_model_keys"]) != pair_model_keys:
        raise ValueError(f"{folder} bootstrap manifest does not exactly cover the paired comparisons.")

    local_equity = _require_mapping(comparison["local_equity"], path="comparison.local_equity", folder=folder)
    for field in ("summary_file", "groups_file", "summary"):
        if field not in local_equity:
            raise ValueError(f"{folder} lacks comparison.local_equity.{field}.")
    _require_string(local_equity["summary_file"], path="comparison.local_equity.summary_file", folder=folder)
    _require_string(local_equity["groups_file"], path="comparison.local_equity.groups_file", folder=folder)


def load_run(folder: Path) -> CountyRun | None:
    """Load a completed, schema-compatible county artifact family."""
    report_path = folder / "metrics.json"
    if not report_path.exists():
        return None
    report = json.loads(report_path.read_text())
    _validate_report_header(report, folder)
    comparison = report["comparison"]
    local_equity = comparison["local_equity"]
    _require_files(folder, [
        "model_comparison_metrics.csv", "rho_plan.csv", "model_comparison_bootstrap_summary.csv",
        "model_comparison_bootstrap_draws.parquet", "predictions.parquet", "validation_candidates.csv",
        str(local_equity["summary_file"]), str(local_equity["groups_file"]),
    ])
    fips = str(report["sample"]["county_fips"]).zfill(5)
    predictions = pd.read_parquet(folder / "predictions.parquet")
    run = CountyRun(
        fips=fips,
        label=f"{COUNTY_NAMES.get(fips, fips)} ({fips})",
        folder=folder,
        report=report,
        metrics=pd.read_csv(folder / "model_comparison_metrics.csv"),
        rho_plan=pd.read_csv(folder / "rho_plan.csv"),
        bootstrap=pd.read_csv(folder / "model_comparison_bootstrap_summary.csv"),
        draws=pd.read_parquet(folder / "model_comparison_bootstrap_draws.parquet"),
        predictions=predictions.loc[predictions["split"].eq("test")],
        train_predictions=predictions.loc[predictions["split"].eq("train")],
        candidates=pd.read_csv(folder / "validation_candidates.csv"),
        local_equity_summary=pd.read_csv(folder / str(local_equity["summary_file"])),
        local_equity_groups=pd.read_csv(folder / str(local_equity["groups_file"])),
    )
    _validate_run_artifacts(run)
    return run


def _validate_run_artifacts(run: CountyRun) -> None:
    """Validate all cross-file references before rendering a single chart or table."""
    comparison = run.report["comparison"]
    manifest = comparison["model_manifest"]
    expected_model_keys = set(manifest["expected_model_keys"])
    expected_bootstrap_keys = set(manifest["bootstrap_model_keys"])
    metric_keys: dict[str, set[str]] = {}
    for split in ("train", "test"):
        keys = run.metrics.loc[run.metrics["split"].eq(split), "model_key"].astype(str)
        duplicates = keys.loc[keys.duplicated()].unique().tolist()
        if duplicates:
            raise ValueError(f"{run.folder} has duplicate {split} model keys: {duplicates}.")
        metric_keys[split] = set(keys)

    test_keys = metric_keys["test"]
    if test_keys != expected_model_keys or metric_keys["train"] != expected_model_keys:
        raise ValueError(
            f"{run.folder} metric/model-manifest mismatch; train={sorted(metric_keys['train'])}, "
            f"test={sorted(test_keys)}, expected={sorted(expected_model_keys)}."
        )
    if BASELINE_KEY not in test_keys:
        raise ValueError(f"{run.folder} lacks the {BASELINE_KEY} held-out metric row.")
    missing_predictions = [
        key for key in sorted(test_keys)
        if f"predicted_sale_price__{key}" not in run.predictions.columns
    ]
    if missing_predictions:
        raise ValueError(f"{run.folder} lacks prediction columns for model keys: {missing_predictions}.")

    if "draw" not in run.draws or "model_key" not in run.draws:
        raise ValueError(f"{run.folder} bootstrap draws lack draw/model_key columns.")
    missing_draw_metrics = [metric for metric in PAIRED_METRICS if metric not in run.draws.columns]
    if missing_draw_metrics:
        raise ValueError(f"{run.folder} bootstrap draws lack paired-comparison metrics: {missing_draw_metrics}.")
    draw_keys = set(run.draws["model_key"].astype(str))
    if draw_keys != expected_bootstrap_keys:
        raise ValueError(
            f"{run.folder} bootstrap/model-manifest mismatch; bootstrap={sorted(draw_keys)}, "
            f"expected={sorted(expected_bootstrap_keys)}."
        )
    draw_ids: dict[str, set[int]] = {}
    for key, frame in run.draws.groupby(run.draws["model_key"].astype(str), sort=False):
        if frame["draw"].duplicated().any():
            raise ValueError(f"{run.folder} has duplicate bootstrap draws for {key!r}.")
        draw_ids[str(key)] = set(frame["draw"].astype(int))
    for pair in comparison["paired_comparisons"]:
        reference, candidate = pair["reference_model_key"], pair["candidate_model_key"]
        if reference not in expected_bootstrap_keys or candidate not in expected_bootstrap_keys:
            raise ValueError(f"{run.folder} paired comparison {pair['id']!r} uses a non-bootstrap model key.")
        if draw_ids.get(reference) != draw_ids.get(candidate):
            raise ValueError(f"{run.folder} paired comparison {pair['id']!r} lacks shared bootstrap draw IDs.")
    bootstrap_summary_keys = set(run.bootstrap["model_key"].astype(str)) if "model_key" in run.bootstrap else set()
    if bootstrap_summary_keys != expected_bootstrap_keys:
        raise ValueError(f"{run.folder} bootstrap summary keys do not match the bootstrap manifest.")
    if "model_key" not in run.rho_plan:
        raise ValueError(f"{run.folder} rho_plan.csv lacks model_key.")
    penalty_keys = run.rho_plan["model_key"].dropna().astype(str)
    duplicates = penalty_keys.loc[penalty_keys.duplicated()].unique().tolist()
    if duplicates:
        raise ValueError(f"{run.folder} rho_plan.csv has duplicate model keys: {duplicates}.")
    oracle_keys = set(manifest["test_oracle_frontier_model_keys"])
    if set(penalty_keys) != oracle_keys:
        raise ValueError(f"{run.folder} rho_plan keys do not match the declared test-oracle frontier.")
    missing_penalty_rows = sorted(set(penalty_keys) - test_keys)
    if missing_penalty_rows:
        raise ValueError(f"{run.folder} rho_plan keys are absent from held-out metrics: {missing_penalty_rows}.")
    baseline_key = comparison["baseline_penalty"]["selected_model_key"]
    scaling_key = comparison["linear_recalibration"]["selected_model_key"]
    selected_neighbor_key = comparison["selected_neighbor_model_key"]
    selected_neighbor_penalty_key = comparison["selected_neighbor_penalty_model_key"]
    required_selected = {baseline_key, scaling_key, selected_neighbor_key, selected_neighbor_penalty_key}
    if not required_selected.issubset(test_keys):
        raise ValueError(f"{run.folder} lacks held-out rows for one or more validation-selected models.")
    if baseline_key not in oracle_keys:
        raise ValueError(f"{run.folder} selected baseline penalty is absent from the declared rho frontier.")
    local_required = {BASELINE_KEY, selected_neighbor_key, selected_neighbor_penalty_key}
    local_columns = {
        "model_key", "split", "minimum_group_n", "n_eligible_groups", "eligible_group_sale_coverage",
        "weighted_median_local_median_ratio_deviation", "weighted_p90_local_median_ratio_deviation",
        "weighted_median_local_cod", "weighted_p90_local_cod", "moran_i_log_residual_knn",
        "moran_i_n", "moran_i_k",
    }
    if not local_columns.issubset(run.local_equity_summary.columns):
        raise ValueError(f"{run.folder} local-equity summary lacks required diagnostics.")
    local_summary_keys = set(run.local_equity_summary.loc[
        run.local_equity_summary["split"].eq("test"), "model_key"
    ].astype(str))
    if local_summary_keys != local_required:
        raise ValueError(f"{run.folder} local-equity summary keys do not match selected neighbor comparisons.")
    group_columns = {"model_key", "split", "geographic_group", "n", "median_ratio", "local_median_ratio_deviation", "local_cod"}
    if not group_columns.issubset(run.local_equity_groups.columns):
        raise ValueError(f"{run.folder} local-equity group artifact lacks required columns.")


def pending_run(folder: Path, reason: str) -> PendingRun:
    """Describe an incomplete requested run without inventing any results for it."""
    match = re.search(r"county_bench_(\d{5})(?:_|$)", folder.name)
    fips = match.group(1) if match else folder.name
    return PendingRun(
        fips=fips,
        label=f"{COUNTY_NAMES.get(fips, fips)} ({fips})",
        folder=folder,
        reason=reason,
    )


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _png(fig, *, dpi: int = 120) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    buf.seek(0)
    return "data:image/png;base64," + base64.b64encode(buf.read()).decode("ascii")


def _figure(src: str, caption: str) -> str:
    return (
        f"<figure class='plot-card'><img src='{src}' alt='{html.escape(caption)}'>"
        f"<figcaption>{caption}</figcaption></figure>"
    )


def _fmt(value: float, digits: int = 3, missing: str = "&ndash;") -> str:
    x = float(value)
    if not np.isfinite(x):
        return missing
    if x != 0 and abs(x) < 10 ** (-digits):
        return f"{x:.1e}"
    if abs(x) >= 10_000:
        return f"{x:,.0f}"
    return f"{x:.{digits}f}"


def _table(df: pd.DataFrame, *, digits: int = 3, highlight: str | None = None) -> str:
    if df is None or df.empty:
        return "<p class='muted'>No rows available.</p>"
    show = df.copy()
    flags = show.pop("_flag") if "_flag" in show.columns else None
    header = "".join(f"<th>{html.escape(str(c))}</th>" for c in show.columns)
    rows = []
    for position, (_, row) in enumerate(show.iterrows()):
        cells = "".join(
            f"<td>{_fmt(v, digits) if isinstance(v, (int, float, np.floating, np.integer)) else html.escape(str(v))}</td>"
            for v in row
        )
        classes = []
        if flags is not None and bool(flags.iloc[position]):
            classes.append("row-pick")
        if highlight is not None and str(row.iloc[0]) == highlight:
            classes.append("row-base")
        attr = f" class='{' '.join(classes)}'" if classes else ""
        rows.append(f"<tr{attr}>{cells}</tr>")
    return f"<table class='table'><thead><tr>{header}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


# Metrics that can be written as a distance from an ideal value, so that a paired
# difference is unambiguously "better" when it is negative.
IDEAL_ONE = {"PRD", "MKI"}
IDEAL_ZERO = {"PRB", "Corr(r,logprice)", "Corr(r,price)", "Cov(e,logprice)"}
ALREADY_A_DISTANCE = {"COD", "RMSE (log)", "MAPE", "MAE (log)", "COV_IAAO"}
PAIRED_METRICS = ["PRD", "PRB", "Cov(e,logprice)", "MKI", "COD", "RMSE (log)", "MAPE"]


def _distance(metric: str, value):
    """Distance from the ideal value, so every metric compares as 'lower is better'."""
    if metric in IDEAL_ONE:
        return abs(value - 1.0)
    if metric in IDEAL_ZERO:
        return abs(value)
    if metric in ALREADY_A_DISTANCE:
        return value
    raise KeyError(f"No ideal value defined for {metric!r}; add it before pairing on it.")


# ---------------------------------------------------------------------------
# Validation-selected operating points and descriptive held-out frontiers
# ---------------------------------------------------------------------------

def operating_points(run: CountyRun) -> pd.DataFrame:
    """Score every penalized fit against the baseline on the held-out block."""
    test = run.test
    baseline = run.baseline
    # ``rho_plan`` is written only for the ordinary LGBCovPenalty path. Neighbor
    # models have separate model keys and only the selected representation has a
    # validation-selected penalty, so including them here would mislabel them as
    # baseline-path frontier points.
    baseline_penalty_keys = set(run.rho_plan["model_key"].dropna().astype(str))
    penalized = test.loc[test["model_key"].astype(str).isin(baseline_penalty_keys)].sort_values("rho").copy()
    penalized["accuracy_cost_r2_log"] = baseline["R2 (log)"] - penalized["R2 (log)"]
    penalized["rmse_log_increase_pct"] = 100.0 * (
        penalized["RMSE (log)"] / baseline["RMSE (log)"] - 1.0
    )
    penalized["prd_gap"] = (penalized["PRD"] - 1.0).abs()
    penalized["prb_gap"] = penalized["PRB"].abs()
    penalized["prd_gap_reduction"] = abs(baseline["PRD"] - 1.0) - penalized["prd_gap"]
    penalized["prb_gap_reduction"] = abs(baseline["PRB"]) - penalized["prb_gap"]
    penalized["cod_reduction"] = baseline["COD"] - penalized["COD"]
    penalized["cov_reduction"] = 1.0 - penalized["Cov(e,logprice)"] / baseline["Cov(e,logprice)"]
    penalized["in_iaao_prd"] = penalized["PRD"].between(*IAAO_PRD)
    penalized["in_iaao_prb"] = penalized["PRB"].between(*IAAO_PRB)
    return penalized


def compliant(run: CountyRun, criterion: str) -> bool:
    band, value = (IAAO_PRB, run.baseline["PRB"]) if criterion == "PRB" else (IAAO_PRD, run.baseline["PRD"])
    return bool(band[0] <= value <= band[1])


def selected_row(run: CountyRun) -> pd.Series:
    """Return the baseline penalty selected on chronological validation.

    The artifact declares this model key after refitting on the full training block.
    It is deliberately never chosen by inspecting the held-out table below.
    """
    key = str(run.report["comparison"]["baseline_penalty"]["selected_model_key"])
    # Retain the descriptive held-out deltas used by the report while deriving
    # them only *after* the model key has been declared by validation.
    rows = operating_points(run).loc[lambda frame: frame["model_key"].astype(str).eq(key)]
    if len(rows) != 1:
        raise ValueError(f"{run.folder} lacks exactly one held-out row for selected baseline penalty {key!r}.")
    return rows.iloc[0]


def selected_validation_metrics(run: CountyRun) -> dict:
    """Validation scores used to choose the deployable baseline-penalty point."""
    return run.report["comparison"]["baseline_penalty"]["selected_validation_metrics"]


def selection_protocol_label(run: CountyRun) -> str:
    protocol = run.report["comparison"]["selection_protocol"]
    return str(protocol.get("name", "pareto_nash_hyperrectangle_v1"))


def recalibrate(run: CountyRun, model_key: str) -> dict:
    """Score a model after a log-linear recalibration fitted on the training block.

    Regressivity in a fitted model is largely shrinkage toward the mean, so regressing
    log price on the log prediction over the training block and applying the fitted
    line to held-out predictions removes the average part of it for free. Applied to
    the baseline it is the strawman the penalty has to beat; applied to a penalized
    fit it says whether the penalty does anything a rescaling could not.
    """
    train, test = run.train_predictions, run.predictions
    column = f"predicted_sale_price__{model_key}"
    slope, intercept = np.polyfit(
        np.log(train[column].to_numpy()), np.log(train["sale_price"].to_numpy()), 1
    )
    corrected = np.exp(intercept + slope * np.log(test[column].to_numpy()))
    scored = score_predictions(
        test["sale_price"].to_numpy(), corrected, train["sale_price"].to_numpy()
    )
    return {"model_key": model_key, "slope": float(slope), "intercept": float(intercept), **scored}


def recalibrated_frontier(run: CountyRun) -> pd.DataFrame:
    """The whole rho grid, rescored after the same training-fitted recalibration."""
    cached = _CACHE.get((run.fips, "recal"))
    if cached is not None:
        return cached
    baseline_rmse = run.baseline["RMSE (log)"]
    positive = operating_points(run).query("rho > 0")
    keys = [BASELINE_KEY] + positive["model_key"].tolist()
    frame = pd.DataFrame([recalibrate(run, key) for key in keys])
    frame["rho"] = [0.0] + positive["rho"].tolist()
    frame["rmse_log_increase_pct"] = 100.0 * (frame["RMSE (log)"] / baseline_rmse - 1.0)
    frame["prd_gap_reduction"] = abs(run.baseline["PRD"] - 1.0) - (frame["PRD"] - 1.0).abs()
    frame["prb_gap_reduction"] = abs(run.baseline["PRB"]) - frame["PRB"].abs()
    frame["cod_reduction"] = run.baseline["COD"] - frame["COD"]
    frame["in_iaao_prb"] = frame["PRB"].between(*IAAO_PRB)
    _CACHE[(run.fips, "recal")] = frame
    return frame


# Slopes applied to the centred baseline log prediction. 1.0 is the baseline itself;
# larger values spread predictions further apart, which is what undoes regressivity.
RESCALE_SLOPES = (1.0, 1.02, 1.05, 1.08, 1.11, 1.14, 1.18, 1.22, 1.26, 1.30)
_CACHE: dict[tuple[str, str], pd.DataFrame] = {}


def rescaling_frontier(run: CountyRun) -> pd.DataFrame:
    """Accuracy/equity frontier traced by rescaling the baseline log prediction alone.

    ``log p_new = mean(log y) + slope * (log p_hat - mean(log p_hat))`` with both means
    taken on the training block, so every point is deployable without touching test
    data. This is the one-parameter competitor the penalty has to beat: if the penalty
    is doing something a rescaling cannot, its frontier must sit above this one.
    """
    cached = _CACHE.get((run.fips, "rescale"))
    if cached is not None:
        return cached
    train, test = run.train_predictions, run.predictions
    column = f"predicted_sale_price__{BASELINE_KEY}"
    train_price = train["sale_price"].to_numpy()
    centre = float(np.log(train[column].to_numpy()).mean())
    level = float(np.log(train_price).mean())
    test_log_pred = np.log(test[column].to_numpy())
    rows = []
    for slope in RESCALE_SLOPES:
        corrected = np.exp(level + slope * (test_log_pred - centre))
        rows.append({"slope": slope, **score_predictions(
            test["sale_price"].to_numpy(), corrected, train_price
        )})
    frame = pd.DataFrame(rows)
    frame["rmse_log_increase_pct"] = 100.0 * (frame["RMSE (log)"] / run.baseline["RMSE (log)"] - 1.0)
    frame["prd_gap_reduction"] = abs(run.baseline["PRD"] - 1.0) - (frame["PRD"] - 1.0).abs()
    frame["prb_gap_reduction"] = abs(run.baseline["PRB"]) - frame["PRB"].abs()
    frame["cod_reduction"] = run.baseline["COD"] - frame["COD"]
    frame["in_iaao_prb"] = frame["PRB"].between(*IAAO_PRB)
    frame["in_iaao_prd"] = frame["PRD"].between(*IAAO_PRD)
    _CACHE[(run.fips, "rescale")] = frame
    return frame


RECALIBRATION_COLOR = "#0ea5e9"
NEIGHBOR_COLORS = {"geo": "#f59e0b", "geo_time": "#7c3aed", "geo_time_features": "#059669"}
NEIGHBOR_MARKERS = {"geo": "s", "geo_time": "D", "geo_time_features": "P"}


def linear_recalibration_spec(run: CountyRun) -> dict:
    """Read the rolling-origin first-degree scaling artifact."""
    spec = run.report["comparison"].get("linear_recalibration")
    if spec is None:
        raise ValueError(
            f"{run.folder} lacks rolling-origin linear-recalibration coefficients; rerun the county benchmark."
        )
    return spec


def reanchored_level_log_prediction(run: CountyRun, prediction_log: np.ndarray | None = None) -> np.ndarray:
    """Apply only the final-training level anchor, without a scaling correction."""
    spec = linear_recalibration_spec(run)
    source = (
        np.log(run.predictions[f"predicted_sale_price__{BASELINE_KEY}"].to_numpy())
        if prediction_log is None else np.asarray(prediction_log, dtype=float)
    )
    return source + float(spec["final_training_outcome_level"]) - float(spec["final_training_prediction_center"])


def recalibration_candidate(spec: dict, strength: float) -> dict:
    return next(row for row in spec["candidates"] if np.isclose(float(row["strength"]), strength))


def recalibrated_log_prediction(
    run: CountyRun, strength: float = 1.0, prediction_log: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the saved rolling-origin first-degree map with final-training anchors."""
    spec = linear_recalibration_spec(run)
    candidate = recalibration_candidate(spec, strength)
    source = (
        np.log(run.predictions[f"predicted_sale_price__{BASELINE_KEY}"].to_numpy())
        if prediction_log is None else np.asarray(prediction_log, dtype=float)
    )
    x = source - float(spec["final_training_prediction_center"])
    support_low, support_high = map(float, spec["oof_centered_prediction_range"])
    x = np.clip(x, support_low, support_high)
    return float(spec["final_training_outcome_level"]) + float(candidate["coefficient"]) * x


def linear_recalibration_frontier(run: CountyRun) -> pd.DataFrame:
    """Held-out scores for every rolling-origin first-degree scaling strength."""
    cached = _CACHE.get((run.fips, "linear_recalibration"))
    if cached is not None:
        return cached
    spec = linear_recalibration_spec(run)
    train_price = run.train_predictions["sale_price"].to_numpy()
    test_price = run.predictions["sale_price"].to_numpy()
    baseline_test_log = np.log(run.predictions[f"predicted_sale_price__{BASELINE_KEY}"].to_numpy())
    centered_test_log = baseline_test_log - float(spec["final_training_prediction_center"])
    support_low, support_high = map(float, spec["oof_centered_prediction_range"])
    outside_support_share = float(((centered_test_log < support_low) | (centered_test_log > support_high)).mean())
    rows = []
    for candidate in spec["candidates"]:
        strength = float(candidate["strength"])
        scored = score_predictions(
            test_price, np.exp(recalibrated_log_prediction(run, strength)), train_price,
        )
        rows.append({
            "strength": strength,
            "method": f"rolling first-degree scale, strength {strength:g}",
            "coefficient": candidate["coefficient"],
            "held_out_prediction_outside_oof_support_share": outside_support_share,
            "validation_metrics": candidate["validation_metrics"],
            "selected_on_rolling_validation": (
                spec["selected_strength"] is not None
                and np.isclose(strength, float(spec["selected_strength"]))
            ),
            "selection_rule": spec["selection_rule"],
            **scored,
        })
    frame = pd.DataFrame(rows)
    frame["rmse_log_increase_pct"] = 100.0 * (frame["RMSE (log)"] / run.baseline["RMSE (log)"] - 1.0)
    frame["prd_gap_reduction"] = abs(run.baseline["PRD"] - 1.0) - (frame["PRD"] - 1.0).abs()
    frame["prb_gap_reduction"] = abs(run.baseline["PRB"]) - frame["PRB"].abs()
    frame["cod_reduction"] = run.baseline["COD"] - frame["COD"]
    frame["in_iaao_prb"] = frame["PRB"].between(*IAAO_PRB)
    frame["in_iaao_prd"] = frame["PRD"].between(*IAAO_PRD)
    _CACHE[(run.fips, "linear_recalibration")] = frame
    return frame


def metric_row(run: CountyRun, model_key: str, split: str) -> pd.Series:
    """Return one persisted metric row, rejecting ambiguous artifact references."""
    rows = run.metrics.loc[
        run.metrics["split"].eq(split) & run.metrics["model_key"].astype(str).eq(str(model_key))
    ]
    if len(rows) != 1:
        raise ValueError(
            f"{run.folder} expected one {split} metric row for {model_key!r}, found {len(rows)}."
        )
    return rows.iloc[0]


def comparator_table(runs: list[CountyRun]) -> pd.DataFrame:
    """Validation provenance and one-time held-out scores for every comparator."""
    rows = []
    for run in runs:
        comparison = run.report["comparison"]
        spec = linear_recalibration_spec(run)
        selected_scale_candidate = recalibration_candidate(spec, float(spec["selected_strength"]))
        selected_scale = metric_row(run, comparison["linear_recalibration"]["selected_model_key"], "test")
        selected_penalty = selected_row(run)
        entries = [
            {
                "name": "LGBM baseline", "fit": "reference; chronological validation", "held_out": run.baseline,
                "chronological": comparison["baseline_penalty"]["baseline_validation_metrics"],
            },
            {
                "name": f"Validation-selected baseline penalty (rho={selected_penalty['rho']:.3g})",
                "fit": f"chronological validation; {selection_protocol_label(run)}", "held_out": selected_penalty,
                "chronological": selected_validation_metrics(run), "rho": selected_penalty["rho"],
            },
            {
                "name": "Validation-selected first-degree scaling",
                "fit": f"rolling-origin OOF; {selection_protocol_label(run)}", "held_out": selected_scale,
                "rolling": selected_scale_candidate["validation_metrics"],
                "strength": selected_scale_candidate["strength"], "coefficient": selected_scale_candidate["coefficient"],
            },
        ]
        for neighbor in comparison["neighbor_models"]:
            entries.append({
                "name": f"{neighbor['label']} LGBM",
                "fit": f"chronological validation; {selection_protocol_label(run)}",
                "held_out": metric_row(run, neighbor["lgbm_model_key"], "test"),
                "chronological": neighbor["validation"]["metrics"], "neighbor": neighbor,
            })
            if neighbor["penalized_model_key"] is not None:
                penalty = neighbor["penalty_selection"]
                entries.append({
                    "name": f"{neighbor['label']} + validation-selected covariance penalty",
                    "fit": f"chronological validation; {selection_protocol_label(run)}",
                    "held_out": metric_row(run, neighbor["penalized_model_key"], "test"),
                    "chronological": penalty["selected_validation_metrics"],
                    "neighbor": neighbor, "rho": neighbor["rho"],
                })
        for entry in entries:
            name = entry["name"]
            held_out = entry["held_out"]
            rolling = entry.get("rolling")
            chronological = entry.get("chronological")
            neighbor = entry.get("neighbor", {})
            metadata = neighbor.get("final_feature_metadata", {})
            neighbor_spec = neighbor.get("validation", {}).get("specification", {})
            rows.append({
                "County": run.label,
                "Method": name,
                "Fit / selection": entry["fit"],
                "Scale strength": entry.get("strength", np.nan),
                "Scale coefficient": entry.get("coefficient", np.nan),
                "Comparable k": neighbor_spec.get("k", np.nan),
                "Max distance (km)": neighbor_spec.get("max_distance_km", np.nan),
                "Max age (days)": neighbor_spec.get("max_time_distance_days", np.nan),
                "Geo / time / feature weights": (
                    "" if not neighbor else "/".join(
                        _fmt(metadata.get(field, np.nan), 2)
                        for field in ("geo_weight", "time_weight", "feature_weight")
                    )
                ),
                "Rolling OOF RMSE (log)": np.nan if rolling is None else rolling["RMSE (log)"],
                "Rolling OOF PRD": np.nan if rolling is None else rolling["PRD"],
                "Rolling OOF PRB": np.nan if rolling is None else rolling["PRB"],
                "Chronological validation RMSE (log)": np.nan if chronological is None else chronological["RMSE (log)"],
                "Chronological validation PRD": np.nan if chronological is None else chronological["PRD"],
                "Chronological validation PRB": np.nan if chronological is None else chronological["PRB"],
                "Held-out outside OOF support share": (
                    np.nan if "held_out_prediction_outside_oof_support_share" not in held_out
                    else held_out["held_out_prediction_outside_oof_support_share"]
                ),
                "Held-out RMSE (log) increase %": 100.0 * (
                    held_out["RMSE (log)"] / run.baseline["RMSE (log)"] - 1.0
                ),
                "Held-out COD": held_out["COD"],
                "Held-out PRD": held_out["PRD"],
                "Held-out PRB": held_out["PRB"],
                "Held-out MKI": held_out["MKI"],
            })
    return pd.DataFrame(rows)


def neighbor_diagnostics_table(runs: list[CountyRun]) -> pd.DataFrame:
    """Expose bounded-search choices and cap-aware comparable coverage."""
    rows = []
    for run in runs:
        search = run.report["comparison"]["neighbor_search"]
        for neighbor in run.report["comparison"]["neighbor_models"]:
            metadata = neighbor["final_feature_metadata"]
            validation = neighbor["validation"]
            search_spec = validation["specification"]
            rows.append({
                "County": run.label,
                "Comparable representation": neighbor["label"],
                "Selected representation": "yes" if neighbor["selected_representation"] else "no",
                "Selection split": "chronological validation",
                "Shared search specification": search["shared_selected_specification"]["key"],
                "k": search_spec["k"],
                "Maximum distance (km)": search_spec["max_distance_km"],
                "Maximum age (days)": search_spec["max_time_distance_days"],
                "Geo / time / structural weights": "/".join(
                    _fmt(metadata.get(field, 0.0), 2)
                    for field in ("geo_weight", "time_weight", "feature_weight")
                ),
                "Target-trend protocol": (
                    metadata.get("time_trend_fit_mode", "not recorded (legacy)")
                    if metadata.get("time_trend") else "not used"
                ),
                "Train coordinate coverage": metadata.get("train_coordinate_coverage", np.nan),
                "Held-out coordinate coverage": metadata.get("evaluation_coordinate_coverage", np.nan),
                "Train comparable coverage after caps": metadata.get("train_comparable_coverage", np.nan),
                "Held-out comparable coverage after caps": metadata.get("evaluation_comparable_coverage", np.nan),
                "Held-out full-k coverage after caps": metadata.get("evaluation_comparable_coverage_after_caps_full_k", np.nan),
                "Candidate retrieval": metadata.get("candidate_retrieval", "not recorded"),
                "Structural columns": ", ".join(metadata.get("structural_similarity_columns", [])) or "—",
            })
    return pd.DataFrame(rows)


def local_equity_table(runs: list[CountyRun]) -> pd.DataFrame:
    """Compact held-out local-equity comparison for baseline and selected neighbors."""
    rows = []
    for run in runs:
        comparison = run.report["comparison"]
        labels = {
            BASELINE_KEY: "LGBM baseline",
            comparison["selected_neighbor_model_key"]: "Validation-selected neighbor LGBM",
            comparison["selected_neighbor_penalty_model_key"]: "Validation-selected neighbor + penalty",
        }
        summary = run.local_equity_summary.loc[run.local_equity_summary["split"].eq("test")]
        for _, row in summary.iterrows():
            key = str(row["model_key"])
            rows.append({
                "County": run.label,
                "Model": labels.get(key, key),
                "Minimum group n": row["minimum_group_n"],
                "Eligible groups": row["n_eligible_groups"],
                "Eligible-group sale coverage": row["eligible_group_sale_coverage"],
                "Weighted median |local median ratio - 1|": row["weighted_median_local_median_ratio_deviation"],
                "Weighted P90 |local median ratio - 1|": row["weighted_p90_local_median_ratio_deviation"],
                "Weighted median local COD": row["weighted_median_local_cod"],
                "Weighted P90 local COD": row["weighted_p90_local_cod"],
                "Moran's I (log residual)": row["moran_i_log_residual_knn"],
                "Moran n / k": f"{int(row['moran_i_n']):,} / {int(row['moran_i_k'])}",
            })
    return pd.DataFrame(rows)


def matched_cost_table(runs: list[CountyRun]) -> pd.DataFrame:
    """Accuracy cost of reaching the same PRB by penalty and by rescaling."""
    rows = []
    for run in runs:
        pick = selected_row(run)
        if pick is None:
            continue
        rescale = rescaling_frontier(run).sort_values("PRB")
        target = float(pick["PRB"])
        # The rescaling frontier is monotone in slope, so interpolating its accuracy
        # cost at the penalty's PRB gives the like-for-like comparison.
        cost = float(np.interp(target, rescale["PRB"], rescale["rmse_log_increase_pct"]))
        slope = float(np.interp(target, rescale["PRB"], rescale["slope"]))
        in_band = rescale.loc[rescale["in_iaao_prd"]]
        rows.append({
            "County": run.label,
            "Target PRB (penalty operating point)": target,
            "rho achieving it": pick["rho"],
            "Penalty cost, RMSE (log) %": pick["rmse_log_increase_pct"],
            "Rescaling slope achieving it": slope,
            "Rescaling cost, RMSE (log) %": cost,
            "Cheaper route": "rescaling" if cost < pick["rmse_log_increase_pct"] else "penalty",
            "Closest |PRD-1|, penalty": float(operating_points(run)["prd_gap"].min()),
            "Closest |PRD-1|, rescaling": float(rescale["PRD"].sub(1.0).abs().min()),
            "Rescaling PRD point estimate in IAAO guidance band": "yes" if not in_band.empty else "no",
        })
    return pd.DataFrame(rows)


def recalibration_table(runs: list[CountyRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        base, pick = run.baseline, selected_row(run)
        fix = recalibrate(run, BASELINE_KEY)
        frontier = recalibrated_frontier(run)
        cheapest = frontier.loc[frontier["in_iaao_prb"]]
        cheapest = cheapest.iloc[0] if not cheapest.empty else None
        entries = [
            ("LGBM baseline", base, np.nan),
            ("Baseline + recalibration", fix, fix["slope"]),
        ]
        if pick is not None:
            entries.append((f"LGBCovPenalty rho={pick['rho']:.3g}", pick, np.nan))
        if cheapest is not None:
            entries.append((
                f"LGBCovPenalty rho={cheapest['rho']:.3g} + recalibration",
                cheapest, cheapest["slope"],
            ))
        for name, source, slope in entries:
            rows.append({
                "County": run.label,
                "Model": name,
                "Recalibration slope": slope,
                "R2 (log)": source["R2 (log)"],
                "RMSE (log) increase %": 100.0 * (source["RMSE (log)"] / base["RMSE (log)"] - 1.0),
                "COD": source["COD"],
                "PRD": source["PRD"],
                "PRB": source["PRB"],
                "MKI": source["MKI"],
                "Cov(e,logprice)": source["Cov(e,logprice)"],
                "PRB point estimate in IAAO guidance band": (
                    "yes" if IAAO_PRB[0] <= source["PRB"] <= IAAO_PRB[1] else "no"
                ),
            })
    return pd.DataFrame(rows)


def paired_bootstrap(
    run: CountyRun,
    *,
    reference_model_key: str,
    candidate_model_key: str,
    metrics: list[str],
) -> pd.DataFrame:
    """Bootstrap distribution of (candidate - reference) on shared time blocks."""
    draws = run.draws
    reference = draws.loc[draws["model_key"].eq(reference_model_key)].set_index("draw")
    candidate = draws.loc[draws["model_key"].eq(candidate_model_key)].set_index("draw")
    common = reference.index.intersection(candidate.index)
    if len(common) != len(reference) or len(common) != len(candidate):
        raise ValueError(
            f"{run.folder} has non-paired bootstrap draws for {reference_model_key!r} and {candidate_model_key!r}."
        )
    rows = []
    for metric in metrics:
        if metric not in reference.columns or metric not in candidate.columns:
            continue
        delta = _distance(metric, candidate.loc[common, metric]) - _distance(metric, reference.loc[common, metric])
        rows.append({
            "metric": f"|{metric}| gap" if metric in IDEAL_ONE | IDEAL_ZERO else metric,
            "reference": float(_distance(metric, reference.loc[common, metric]).mean()),
            "candidate": float(_distance(metric, candidate.loc[common, metric]).mean()),
            "difference": float(delta.mean()),
            "ci_2_5": float(delta.quantile(0.025)),
            "ci_97_5": float(delta.quantile(0.975)),
            "p_improved": float((delta < 0).mean()),
        })
    return pd.DataFrame(rows)


def paired_comparison_table(run: CountyRun) -> pd.DataFrame:
    """All predeclared paired comparisons, with one row per metric and interval."""
    rows = []
    for pair in run.report["comparison"]["paired_comparisons"]:
        table = paired_bootstrap(
            run,
            reference_model_key=pair["reference_model_key"],
            candidate_model_key=pair["candidate_model_key"],
            metrics=PAIRED_METRICS,
        )
        if table.empty:
            continue
        table.insert(0, "Comparison", pair["label"])
        rows.append(table)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------

def sample_table(runs: list[CountyRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        sample = run.report["sample"]
        rows.append({
            "County": run.label,
            "Raw recorder transfers": sample["n_raw_county_recorder_transfers"],
            "Qualified sales": sample["n_transactions_before_history"],
            "History-matched": sample["n_history_matches"],
            "Modeled (use 385)": sample["n_property_use_eligible"],
            "History match rate": sample["assessor_history_match_rate"],
            "Location usable": sample["tax_assessor_usable_location_rate"],
            "ACS matched": sample["acs_match_rate"],
            "Train / test": f"{run.report['n_train']:,} / {run.report['n_test']:,}",
            "Test window": " to ".join(run.report["test_date_range"]),
        })
    return pd.DataFrame(rows)


def baseline_table(runs: list[CountyRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        base = run.baseline
        rows.append({
            "County": run.label,
            "Feature set": run.report["comparison"]["feature_set"],
            "LGBM config": run.report["comparison"]["lgbm_config"],
            "R2 (log)": base["R2 (log)"],
            "RMSE (log)": base["RMSE (log)"],
            "MAPE": base["MAPE"],
            "COD": base["COD"],
            "PRD": base["PRD"],
            "PRB": base["PRB"],
            "MKI": base["MKI"],
            "Median ratio": base["Median ratio"],
            "PRD point estimate in IAAO guidance band": (
                "yes" if IAAO_PRD[0] <= base["PRD"] <= IAAO_PRD[1] else "no"
            ),
            "PRB point estimate in IAAO guidance band": (
                "yes" if IAAO_PRB[0] <= base["PRB"] <= IAAO_PRB[1] else "no"
            ),
            "COD point estimate within IAAO guidance": "yes" if base["COD"] <= IAAO_COD_MAX else "no",
        })
    return pd.DataFrame(rows)


def theory_table(runs: list[CountyRun]) -> pd.DataFrame:
    rows = []
    for run in runs:
        theory = run.theory
        rows.append({
            "County": run.label,
            "A = Var(f0)": theory["A_var_f0_log"],
            "B = MSE(log)": theory["B_mse_log"],
            "C0 = Cov(e, logY)": theory["C0_cov_log_residual_logprice"],
            "Baseline PRD (train)": theory["prd"],
            "Baseline PRD (test)": run.baseline["PRD"],
            "rho @ 25% shrink": theory["rho_shrink_25pct"],
            "rho @ 50% shrink": theory["rho_shrink_50pct"],
            "rho @ 67% shrink": theory["rho_shrink_67pct"],
            "rho @ 1% MSE budget": theory["rho_budget_1pct_mse"],
            "PRD target costs > 1% MSE": (
                "yes" if theory["rho_prd_guidance"] > theory["rho_budget_1pct_mse"] else "no"
            ),
        })
    return pd.DataFrame(rows)


def generalization_table(runs: list[CountyRun]) -> pd.DataFrame:
    """In-sample, validation, and held-out equity for the selected baseline.

    The validation column is the selected configuration scored on one chronological
    validation slice during model selection, so it is not a cross-validated estimate.
    """
    rows = []
    for run in runs:
        selection = run.report["selection"]
        candidates = run.candidates
        match = candidates.loc[
            candidates["feature_set"].eq(selection["feature_set"])
            & candidates["lgbm_config"].eq(selection["lgbm_config"])
            & candidates["target_scale"].eq(selection["target_scale"])
        ]
        if match.empty:
            continue
        oof = match.iloc[0]
        in_sample = run.report["train_metrics"]
        test = run.report["test_metrics"]
        rows.append({
            "County": run.label,
            "PRD in-sample": in_sample["PRD"],
            "PRD validation": oof["PRD"],
            "PRD held-out": test["PRD"],
            "PRB in-sample": in_sample["PRB"],
            "PRB validation": oof["PRB"],
            "PRB held-out": test["PRB"],
            "COD in-sample": in_sample["COD"],
            "COD validation": oof["COD"],
            "COD held-out": test["COD"],
        })
    return pd.DataFrame(rows)


def control_table(runs: list[CountyRun]) -> pd.DataFrame:
    """rho = 0 versus the plain LightGBM baseline: the objective-plumbing check."""
    rows = []
    for run in runs:
        test = run.test.set_index("model_key")
        if CONTROL_KEY not in test.index:
            continue
        base, control = test.loc[BASELINE_KEY], test.loc[CONTROL_KEY]
        rows.append({
            "County": run.label,
            "R2 (log) baseline": base["R2 (log)"],
            "R2 (log) rho=0": control["R2 (log)"],
            "R2 (log) difference": control["R2 (log)"] - base["R2 (log)"],
            "PRD baseline": base["PRD"],
            "PRD rho=0": control["PRD"],
            "PRD difference": control["PRD"] - base["PRD"],
            "COD difference": control["COD"] - base["COD"],
        })
    return pd.DataFrame(rows)


def comparison_table(run: CountyRun) -> pd.DataFrame:
    """Held-out baseline-penalty path; only its marked row was validation-selected."""
    # Neighbor models are intentionally reported in the explicit comparator table below.
    baseline_keys = {BASELINE_KEY, *run.rho_plan["model_key"].dropna().astype(str)}
    test = run.test.loc[run.test["model_key"].astype(str).isin(baseline_keys)].sort_values(
        "rho", na_position="first",
    )
    plan = run.rho_plan.set_index("model_key")
    rows = []
    operating_key = str(run.report["comparison"]["baseline_penalty"]["selected_model_key"])
    for _, row in test.iterrows():
        key = str(row["model_key"])
        rows.append({
            "Model": "LGBM baseline" if key == BASELINE_KEY else f"LGBCovPenalty rho={row['rho']:.3g}",
            "Requested cov. reduction": plan.loc[key, "requested_covariance_reduction"] if key in plan.index else np.nan,
            "Realized cov. reduction": plan.loc[key, "realized_covariance_reduction_test"] if key in plan.index else np.nan,
            **{name: row[name] for name, _ in HEADLINE_METRICS},
            "_flag": key == operating_key,
        })
    return pd.DataFrame(rows)


def operating_summary(runs: list[CountyRun]) -> pd.DataFrame:
    """One deployable baseline-penalty selection per county, chosen before test."""
    rows = []
    for run in runs:
        comparison = run.report["comparison"]
        base = run.baseline
        pick = selected_row(run)
        validation = selected_validation_metrics(run)
        selected_meta = comparison["baseline_penalty"]
        rows.append({
            "County": run.label,
            "Selection split": selected_meta["selection_split"],
            "Selection rule": selection_protocol_label(run),
            "Selected rho": pick["rho"],
            "Requested cov. reduction": selected_meta["selected_requested_covariance_reduction"],
            "Validation MAPE": validation["MAPE"],
            "Validation COD": validation["COD"],
            "Validation PRD": validation["PRD"],
            "Validation PRB": validation["PRB"],
            "Held-out MAPE": pick["MAPE"],
            "Held-out COD": pick["COD"],
            "Held-out PRD": pick["PRD"],
            "Held-out PRB": pick["PRB"],
            "Held-out RMSE(log) increase %": 100.0 * (pick["RMSE (log)"] / base["RMSE (log)"] - 1.0),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def ratio_curve_penalized_row(run: CountyRun) -> tuple[pd.Series, str]:
    """Return the validation-selected penalty, never a test-picked fallback."""
    pick = selected_row(run)
    return pick, f"LGBCovPenalty rho={pick['rho']:.3g} (validation-selected)"


def fig_baseline_ratio_curves(runs: list[CountyRun]) -> str:
    """Median assessment ratio by sale-price decile: a descriptive diagnostic."""
    ncols = min(3, len(runs))
    nrows = (len(runs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.5 * nrows), sharey=True, squeeze=False)
    for ax, run in zip(axes.ravel(), runs):
        pick, pick_label = ratio_curve_penalized_row(run)
        frame = run.predictions
        decile = pd.qcut(frame["sale_price"], 10, labels=False, duplicates="drop")
        for key, name, style in (
            (BASELINE_KEY, "LGBM baseline", {"color": "#334155", "marker": "o"}),
            (pick["model_key"], pick_label,
             {"color": COUNTY_COLORS.get(run.fips, "#1d4ed8"), "marker": "s"}),
        ):
            column = f"predicted_sale_price__{key}"
            if column not in frame:
                continue
            ratio = frame[column] / frame["sale_price"]
            # Median, not mean: the ratio distribution has a heavy right tail in the
            # cheapest decile that would otherwise set the scale for the whole panel.
            grouped = ratio.groupby(decile).median()
            ax.plot(grouped.index + 1, grouped.to_numpy(), linewidth=1.8, markersize=4, label=name, **style)
        ax.axhline(1.0, color="#94a3b8", linestyle="--", linewidth=1)
        ax.set_title(run.label, fontsize=10)
        ax.set_xlabel("Sale-price decile")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7.5, frameon=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("Median ratio (predicted / sale)")
    for ax in axes.ravel()[len(runs):]:
        ax.set_visible(False)
    fig.suptitle("Median assessment ratio by sale-price decile, held-out sales", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _png(fig)


def fig_rho_evolution(runs: list[CountyRun], *, x_axis: str) -> str:
    """Metric trajectories against either raw rho or the requested covariance reduction."""
    panels = [
        ("PRD", "PRD", IAAO_PRD), ("PRB", "PRB", IAAO_PRB), ("COD", "COD", None),
        ("MKI", "MKI", None), ("R2 (log)", "R2 (log)", None),
        ("Cov(e,logprice)", "Cov(e, log price)", None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.2))
    for ax, (metric, title, band) in zip(axes.ravel(), panels):
        for run in runs:
            points = operating_points(run).merge(
                run.rho_plan[["model_key", "requested_covariance_reduction"]], on="model_key", how="left",
            )
            points = points.loc[points["rho"].gt(0)]
            x = points["rho"] if x_axis == "rho" else points["requested_covariance_reduction"]
            color = COUNTY_COLORS.get(run.fips, "#334155")
            ax.plot(x, points[metric], marker="o", markersize=3.5, linewidth=1.6, color=color, label=run.label)
            baseline_value = run.baseline[metric]
            ax.axhline(baseline_value, color=color, linestyle=":", linewidth=1.0, alpha=0.65)
        if band is not None:
            ax.axhspan(band[0], band[1], color="#22c55e", alpha=0.10)
        if x_axis == "rho":
            ax.set_xscale("log")
            ax.set_xlabel("rho (log scale)")
        else:
            ax.set_xlabel("Requested covariance reduction")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    axes.ravel()[0].legend(fontsize=7.5, frameon=False)
    subtitle = "penalty strength" if x_axis == "rho" else "requested covariance reduction (comparable across counties)"
    fig.suptitle(
        f"Held-out metric evolution against {subtitle}. Dotted lines are the unpenalized baseline; "
        "green bands are the IAAO guidance ranges for point estimates.",
        fontsize=11,
    )
    fig.tight_layout()
    return _png(fig)


def calibration_factor(run: CountyRun) -> float:
    """Least-squares slope of realized on requested covariance reduction, through zero.

    A slope of 1 would mean the closed-form rho delivers exactly what it promises.
    """
    plan = run.rho_plan
    x = plan["requested_covariance_reduction"].to_numpy(dtype=float)
    y = plan["realized_covariance_reduction_test"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    return float((x[mask] @ y[mask]) / (x[mask] @ x[mask])) if mask.any() else float("nan")


def fig_theory_vs_realized(runs: list[CountyRun]) -> str:
    """Does the rank-one theory deliver the covariance reduction it promises?"""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for run in runs:
        plan = run.rho_plan.sort_values("requested_covariance_reduction")
        color = COUNTY_COLORS.get(run.fips, "#334155")
        axes[0].plot(
            plan["requested_covariance_reduction"], plan["realized_covariance_reduction_test"],
            marker="o", markersize=4, linewidth=1.6, color=color,
            label=f"{run.label} (slope {calibration_factor(run):.2f})",
        )
        points = operating_points(run)
        points = points.loc[points["rho"].gt(0)].merge(
            plan[["model_key", "delta_mse_log_frac_of_baseline"]], on="model_key", how="left",
        )
        realized_mse = (points["RMSE (log)"] ** 2) / (run.baseline["RMSE (log)"] ** 2) - 1.0
        axes[1].plot(
            points["delta_mse_log_frac_of_baseline"], realized_mse,
            marker="o", markersize=4, linewidth=1.6, color=color, label=run.label,
        )
    for ax, title, label in (
        (axes[0], "Covariance reduction: requested vs realized", "covariance reduction"),
        (axes[1], "Accuracy price: theory vs realized", "MSE(log) increase as a fraction of baseline"),
    ):
        limit = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, limit], [0, limit], color="#94a3b8", linestyle="--", linewidth=1, label="theory exact")
        ax.set_xlabel(f"Theory-predicted {label}")
        ax.set_ylabel(f"Realized {label}")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    return _png(fig)


def fig_tradeoff(runs: list[CountyRun]) -> str:
    """Descriptive held-out penalty frontier, with the validation choice ringed."""
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.3))
    specs = [
        ("prd_gap_reduction", "Reduction in |PRD - 1|"),
        ("prb_gap_reduction", "Reduction in |PRB|"),
        ("cod_reduction", "Reduction in COD (points)"),
    ]
    for ax, (column, ylabel) in zip(axes, specs):
        for run in runs:
            points = operating_points(run)
            points = points.loc[points["rho"].gt(0)].sort_values("rho")
            color = COUNTY_COLORS.get(run.fips, "#334155")
            ax.plot(points["rmse_log_increase_pct"], points[column], marker="o", markersize=3.5,
                    linewidth=1.6, color=color, label=run.label)
            pick = selected_row(run)
            if pick is not None and pick["rho"] > 0:
                ax.scatter([pick["rmse_log_increase_pct"]], [pick[column]], s=110, facecolors="none",
                           edgecolors=color, linewidths=2.0, zorder=5)
        ax.axhline(0.0, color="#94a3b8", linestyle="--", linewidth=1)
        ax.set_xlabel("RMSE(log) increase vs baseline (%)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=7.5, frameon=False)
    fig.suptitle(
        "Accuracy / equity frontier. Up is fairer, right is less accurate; the ringed marker is the "
        "predeclared chronological-validation selection. The remaining held-out points are descriptive only.",
        fontsize=11,
    )
    fig.tight_layout()
    return _png(fig)


def fig_comparator_tradeoff(runs: list[CountyRun]) -> str:
    """Compare baseline penalty, first-degree scale, and comparable-sales models."""
    ncols = min(3, len(runs))
    county_rows = (len(runs) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        2 * county_rows, ncols, figsize=(4.4 * ncols, 3.35 * 2 * county_rows),
        sharex=False, sharey=False, squeeze=False,
    )
    metrics = (("prb_gap_reduction", "Reduction in |PRB|"), ("prd_gap_reduction", "Reduction in |PRD - 1|"))
    for county_number, run in enumerate(runs):
        county_row, column_number = divmod(county_number, ncols)
        penalty = operating_points(run).query("rho > 0").sort_values("rho")
        scale = linear_recalibration_frontier(run).sort_values("strength")
        test_rows = run.test.set_index("model_key")
        for metric_number, (metric, ylabel) in enumerate(metrics):
            row_number = 2 * county_row + metric_number
            ax = axes[row_number, column_number]
            ax.scatter([0.0], [0.0], marker="x", color="#334155", s=28, label="LGBM baseline" if county_number == 0 and metric_number == 0 else None)
            ax.plot(penalty["rmse_log_increase_pct"], penalty[metric], color="#334155", marker="o", markersize=3, linewidth=1.3,
                    label="covariance-penalty path" if county_number == 0 and metric_number == 0 else None)
            ax.plot(scale["rmse_log_increase_pct"], scale[metric], color=RECALIBRATION_COLOR, marker="^", markersize=3, linewidth=1.2,
                    linestyle="--", label="first-degree scaling frontier" if county_number == 0 and metric_number == 0 else None)
            for neighbor in run.report["comparison"]["neighbor_models"]:
                color = NEIGHBOR_COLORS[neighbor["key"]]
                marker = NEIGHBOR_MARKERS[neighbor["key"]]
                candidates = [(neighbor["lgbm_model_key"], False, "neighbor LGBM")]
                if neighbor["penalized_model_key"] is not None:
                    candidates.append((neighbor["penalized_model_key"], True, "neighbor LGBM + penalty"))
                for key, filled, label in candidates:
                    point = test_rows.loc[key]
                    x = 100.0 * (point["RMSE (log)"] / run.baseline["RMSE (log)"] - 1.0)
                    y = (
                        abs(run.baseline["PRB"]) - abs(point["PRB"])
                        if metric == "prb_gap_reduction"
                        else abs(run.baseline["PRD"] - 1.0) - abs(point["PRD"] - 1.0)
                    )
                    ax.scatter(
                        [x], [y], marker=marker, s=36, edgecolors=color,
                        facecolors=color if filled else "white", linewidths=1.0,
                        label=f"{neighbor['label']} ({label})" if county_number == 0 and metric_number == 0 else None,
                    )
            ax.axhline(0.0, color="#94a3b8", linestyle=":", linewidth=1)
            ax.grid(alpha=0.25)
            ax.set_title(run.label if metric_number == 0 else "", fontsize=8.5)
            if metric_number == 1:
                ax.set_xlabel("RMSE(log) increase vs baseline (%)")
            if column_number == 0:
                ax.set_ylabel(ylabel)
    for county_number in range(len(runs), county_rows * ncols):
        county_row, column_number = divmod(county_number, ncols)
        axes[2 * county_row, column_number].set_visible(False)
        axes[2 * county_row + 1, column_number].set_visible(False)
    axes[0, 0].legend(fontsize=5.6, frameon=False, loc="best")
    fig.suptitle(
        "Held-out screen: solid paths are baseline covariance penalties; dashed paths are first-degree scaling; "
        "open markers are neighbor LGBM fits; the one filled marker is the selected-neighbor validation-selected penalty.",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _png(fig)


def fig_linear_mapping(runs: list[CountyRun]) -> str:
    """Show level anchoring separately from the validation-selected scale map."""
    ncols = min(3, len(runs))
    nrows = (len(runs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.4 * ncols, 3.5 * nrows), squeeze=False)
    for ax, run in zip(axes.ravel(), runs):
        spec = linear_recalibration_spec(run)
        baseline_log = np.log(run.predictions[f"predicted_sale_price__{BASELINE_KEY}"].to_numpy())
        centre = float(spec["final_training_prediction_center"])
        support_low, support_high = [centre + float(value) for value in spec["oof_centered_prediction_range"]]
        low = min(float(np.quantile(baseline_log, 0.01)), support_low)
        high = max(float(np.quantile(baseline_log, 0.99)), support_high)
        grid = np.linspace(low, high, 300)
        ax.axvspan(support_low, support_high, color="#94a3b8", alpha=0.13, label="rolling OOF support")
        ax.plot(grid, grid, color="#334155", linestyle="--", linewidth=1.2, label="raw baseline")
        ax.plot(
            grid, reanchored_level_log_prediction(run, grid), color="#64748b", linewidth=1.4,
            label="final-training level anchor",
        )
        ax.plot(
            grid, recalibrated_log_prediction(
                run, strength=float(spec["selected_strength"]), prediction_log=grid,
            ),
            color=RECALIBRATION_COLOR, linewidth=1.7, label="validation-selected first-degree scale",
        )
        ax.set_title(run.label, fontsize=9.5)
        ax.set_xlabel("Baseline log prediction")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.6, frameon=False, loc="best")
    for ax in axes[:, 0]:
        ax.set_ylabel("Recalibrated log prediction")
    for ax in axes.ravel()[len(runs):]:
        ax.set_visible(False)
    fig.suptitle(
        "Validation-selected rolling-origin first-degree scaling with final-training level anchors; "
        "shading marks OOF support",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _png(fig)


def fig_linear_ratio_curves(runs: list[CountyRun]) -> str:
    """Held-out ratio deciles for baseline, penalty, and first-degree scaling."""
    ncols = min(3, len(runs))
    nrows = (len(runs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 3.7 * nrows), sharey=True, squeeze=False)
    for ax, run in zip(axes.ravel(), runs):
        frame = run.predictions
        decile = pd.qcut(frame["sale_price"], 10, labels=False, duplicates="drop")
        penalty, penalty_label = ratio_curve_penalized_row(run)
        lines = [
            ("LGBM baseline", frame[f"predicted_sale_price__{BASELINE_KEY}"].to_numpy(),
             {"color": "#334155", "marker": "o", "linewidth": 1.8}),
            (penalty_label, frame[f"predicted_sale_price__{penalty['model_key']}"].to_numpy(),
             {"color": COUNTY_COLORS.get(run.fips, "#1d4ed8"), "marker": "P", "linewidth": 1.5}),
            ("final-training level anchor", np.exp(reanchored_level_log_prediction(run)),
             {"color": "#64748b", "marker": "x", "linewidth": 1.2}),
        ]
        lines.append((
            "validation-selected rolling first-degree scale",
            frame[f"predicted_sale_price__{run.report['comparison']['linear_recalibration']['selected_model_key']}"]
            .to_numpy(),
            {"color": RECALIBRATION_COLOR, "marker": "^", "linewidth": 1.4},
        ))
        for label, prediction, style in lines:
            ratio = prediction / frame["sale_price"].to_numpy()
            grouped = pd.DataFrame({"decile": decile, "ratio": ratio}).groupby("decile", observed=True)["ratio"].median()
            ax.plot(grouped.index.to_numpy() + 1, grouped.to_numpy(), label=label, markersize=3.3, **style)
        ax.axhline(1.0, color="#94a3b8", linestyle="--", linewidth=1)
        ax.set_title(run.label, fontsize=9.5)
        ax.set_xlabel("Sale-price decile")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=5.9, frameon=False, loc="best")
    for ax in axes[:, 0]:
        ax.set_ylabel("Median ratio (predicted / sale)")
    for ax in axes.ravel()[len(runs):]:
        ax.set_visible(False)
    fig.suptitle("Held-out ratio curves: raw baseline, level anchor, covariance penalty, and first-degree scaling", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _png(fig)


def fig_neighbor_ratio_curves(runs: list[CountyRun]) -> str:
    """Held-out deciles for all neighbor levels and the selected penalty fit."""
    ncols = min(3, len(runs))
    nrows = (len(runs) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.7 * ncols, 3.7 * nrows), sharey=True, squeeze=False)
    for ax, run in zip(axes.ravel(), runs):
        frame = run.predictions
        decile = pd.qcut(frame["sale_price"], 10, labels=False, duplicates="drop")
        lines = [("LGBM baseline", frame[f"predicted_sale_price__{BASELINE_KEY}"].to_numpy(), {"color": "#334155", "marker": "o", "linewidth": 1.7})]
        for neighbor in run.report["comparison"]["neighbor_models"]:
            color = NEIGHBOR_COLORS[neighbor["key"]]
            marker = NEIGHBOR_MARKERS[neighbor["key"]]
            lines.append((
                f"{neighbor['label']} LGBM",
                frame[f"predicted_sale_price__{neighbor['lgbm_model_key']}"].to_numpy(),
                {"color": color, "marker": marker, "linestyle": "--", "linewidth": 1.1},
            ))
            if neighbor["penalized_model_key"] is not None:
                lines.append((
                    f"{neighbor['label']} + validation-selected penalty",
                    frame[f"predicted_sale_price__{neighbor['penalized_model_key']}"].to_numpy(),
                    {"color": color, "marker": marker, "linewidth": 1.5},
                ))
        for label, prediction, style in lines:
            ratio = prediction / frame["sale_price"].to_numpy()
            grouped = pd.DataFrame({"decile": decile, "ratio": ratio}).groupby("decile", observed=True)["ratio"].median()
            ax.plot(grouped.index.to_numpy() + 1, grouped.to_numpy(), label=label, markersize=3.1, **style)
        ax.axhline(1.0, color="#94a3b8", linestyle="--", linewidth=1)
        ax.set_title(run.label, fontsize=9.5)
        ax.set_xlabel("Sale-price decile")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=5.7, frameon=False, loc="best")
    for ax in axes[:, 0]:
        ax.set_ylabel("Median ratio (predicted / sale)")
    for ax in axes.ravel()[len(runs):]:
        ax.set_visible(False)
    fig.suptitle(
        "Held-out ratio curves: all comparable-sales levels and the selected-neighbor validation-selected penalty",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    return _png(fig)


def fig_paired_intervals(runs: list[CountyRun]) -> str:
    """Paired test-block intervals for all four predeclared comparisons."""
    pair_count = len(REQUIRED_PAIR_IDS)
    fig, axes = plt.subplots(
        len(runs), pair_count, figsize=(3.15 * pair_count, 2.75 * len(runs)),
        sharex=False, squeeze=False,
    )
    for county_number, run in enumerate(runs):
        pairs = run.report["comparison"]["paired_comparisons"]
        for pair_number, pair in enumerate(pairs):
            ax = axes[county_number, pair_number]
            table = paired_bootstrap(
                run,
                reference_model_key=pair["reference_model_key"],
                candidate_model_key=pair["candidate_model_key"],
                metrics=PAIRED_METRICS,
            )
            y = np.arange(len(table))[::-1]
            # Differences live on different scales, so normalize by the paired
            # reference distance.  Zero remains no change; negative is better.
            scale = table["reference"].abs().replace(0.0, np.nan)
            ax.errorbar(
                table["difference"] / scale, y,
                xerr=[(table["difference"] - table["ci_2_5"]).abs() / scale,
                      (table["ci_97_5"] - table["difference"]).abs() / scale],
                fmt="o", markersize=4, capsize=2.5, linewidth=1.2,
                color=COUNTY_COLORS.get(run.fips, "#334155"),
            )
            ax.axvline(0.0, color="#94a3b8", linestyle="--", linewidth=1)
            ax.set_yticks(y)
            ax.set_yticklabels(table["metric"], fontsize=6.8)
            ax.set_title(pair["label"], fontsize=7.4)
            if county_number == len(runs) - 1:
                ax.set_xlabel("Candidate − reference\n(relative distance)", fontsize=7.2)
            if pair_number == 0:
                ax.set_ylabel(run.label, fontsize=8.2)
            ax.grid(alpha=0.25, axis="x")
    fig.suptitle(
        "Paired time-block bootstrap for predeclared validation-selected comparisons. "
        "Negative candidate − reference distance is better.",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return _png(fig)


def fig_level_drift(runs: list[CountyRun]) -> str:
    """Quarterly median ratio from 2016 through 2026, with test windows shaded.

    A level shift inside the test window contaminates every ratio statistic computed
    on it, so this is checked before any of those statistics are interpreted.
    """
    fig, axes = plt.subplots(1, len(runs), figsize=(3.9 * len(runs), 3.3), sharey=True)
    axes = np.atleast_1d(axes)
    column = f"predicted_sale_price__{BASELINE_KEY}"
    visible_levels = []
    for ax, run in zip(axes, runs):
        frame = pd.concat([run.train_predictions, run.predictions])
        frame = frame.assign(
            sale_date=pd.to_datetime(frame["sale_date"]),
            ratio=frame[column] / frame["sale_price"],
        )
        quarterly = frame.set_index("sale_date").resample("QE")["ratio"].median()
        visible = quarterly.loc[(quarterly.index >= LEVEL_PLOT_START) & (quarterly.index < LEVEL_PLOT_END)]
        visible_levels.append(visible)
        ax.plot(visible.index, visible.to_numpy(), linewidth=1.6,
                color=COUNTY_COLORS.get(run.fips, "#334155"))
        test_start = pd.to_datetime(run.report["test_date_range"][0])
        if not visible.empty:
            ax.axvspan(max(test_start, LEVEL_PLOT_START), visible.index.max(), color="#94a3b8", alpha=0.16)
        ax.axhline(1.0, color="#94a3b8", linestyle="--", linewidth=1)
        ax.set_title(run.label, fontsize=9.5)
        ax.set_xlim(LEVEL_PLOT_START, LEVEL_PLOT_END)
        ax.set_xticks(pd.date_range("2016-01-01", "2026-01-01", freq="2YS"))
        ax.tick_params(axis="x", labelsize=7.5, rotation=45)
        ax.grid(alpha=0.25)
    values = pd.concat(visible_levels).dropna()
    if not values.empty:
        low, high = float(values.min()), float(values.max())
        pad = max(0.03, 0.10 * (high - low))
        lower = np.floor((min(low - pad, 1.0) * 20.0)) / 20.0
        upper = np.ceil((max(high + pad, 1.0) * 20.0)) / 20.0
        for ax in axes:
            ax.set_ylim(lower, upper)
    axes[0].set_ylabel("Median ratio (predicted / sale)")
    fig.suptitle(
        "Median assessment ratio by quarter, 2016–2026. The shaded region is the held-out block; a step "
        "inside it is a level break the model could not have known about.",
        fontsize=10,
    )
    fig.tight_layout()
    return _png(fig)


def level_drift_table(runs: list[CountyRun]) -> pd.DataFrame:
    """Size of the level shift between the training block and the held-out block."""
    column = f"predicted_sale_price__{BASELINE_KEY}"
    rows = []
    for run in runs:
        train_ratio = (run.train_predictions[column] / run.train_predictions["sale_price"]).median()
        test_ratio = (run.predictions[column] / run.predictions["sale_price"]).median()
        rows.append({
            "County": run.label,
            "Median ratio, train": train_ratio,
            "Median ratio, test": test_ratio,
            "Level shift %": 100.0 * (test_ratio / train_ratio - 1.0),
            "Test window": " to ".join(run.report["test_date_range"]),
        })
    return pd.DataFrame(rows)


def fig_price_tail(runs: list[CountyRun]) -> str:
    """Where the held-out price mass sits, which governs price-scale R2 and RMSE."""
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for run in runs:
        price = np.sort(run.predictions["sale_price"].to_numpy())
        share = np.cumsum(price) / price.sum()
        ax.plot(np.linspace(0, 100, price.size), 100 * share, linewidth=1.8,
                color=COUNTY_COLORS.get(run.fips, "#334155"), label=run.label)
    ax.set_xlabel("Percentile of held-out sales, ordered by price")
    ax.set_ylabel("Cumulative share of total sale value (%)")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("Concentration of transaction value in the upper tail", fontsize=10)
    fig.tight_layout()
    return _png(fig)


# ---------------------------------------------------------------------------
# Page assembly
# ---------------------------------------------------------------------------

CSS = """
<style>
:root { --bg:#f8fafc; --fg:#0f172a; --muted:#64748b; --line:#dbe3ef; --card:#ffffff; --accent:#1d4ed8; }
body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif; color:var(--fg); background:var(--bg); }
header { padding:30px 40px; background:#0f172a; color:white; }
header h1 { margin:0 0 10px; font-size:30px; }
header p { margin:5px 0; color:#dbeafe; max-width:1150px; line-height:1.5; }
nav { position:sticky; top:0; z-index:3; background:white; border-bottom:1px solid var(--line); padding:10px 40px; }
nav a { color:var(--accent); margin-right:16px; text-decoration:none; font-weight:600; font-size:13.5px; }
main { max-width:1420px; margin:0 auto; padding:24px 28px 60px; }
section { background:var(--card); border:1px solid var(--line); border-radius:8px; padding:24px; margin:0 0 20px; box-shadow:0 1px 2px rgba(15,23,42,0.04); }
h2 { margin:0 0 6px; font-size:22px; }
h3 { margin:24px 0 10px; font-size:16px; }
p, li { line-height:1.55; font-size:14.5px; }
.muted { color:var(--muted); font-size:13.5px; }
.cards { display:grid; grid-template-columns:repeat(auto-fit,minmax(200px,1fr)); gap:12px; margin:14px 0; }
.card { border:1px solid var(--line); border-radius:8px; padding:14px; background:#fbfdff; font-size:13px; }
.card b { display:block; font-size:23px; margin-bottom:3px; color:#0f172a; }
.table { border-collapse:collapse; width:100%; font-size:12.5px; margin-bottom:6px; }
.table th, .table td { border-bottom:1px solid var(--line); padding:6px 8px; text-align:right; white-space:nowrap; }
.table th { background:#f1f5f9; color:#334155; font-weight:600; }
.table td:first-child, .table th:first-child { text-align:left; }
.table tr.row-pick { background:#ecfdf5; font-weight:600; }
.table tr.row-base { background:#f1f5f9; }
.scroll { overflow-x:auto; }
.plot-card { margin:0 0 16px; border:1px solid var(--line); border-radius:8px; overflow:hidden; background:white; }
.plot-card img { display:block; width:100%; height:auto; }
.plot-card figcaption { padding:9px 12px; font-size:13px; color:#334155; border-top:1px solid var(--line); background:#fbfdff; }
code { background:#e2e8f0; padding:1px 5px; border-radius:4px; font-size:12.5px; }
.callout { border-left:4px solid var(--accent); background:#f0f6ff; padding:12px 16px; margin:14px 0; border-radius:0 6px 6px 0; }
.warn { border-left-color:#d97706; background:#fffbeb; }
</style>
"""


def build_html(
    runs: list[CountyRun], *, floor: str, pending: list[PendingRun] | None = None,
) -> str:
    pending = pending or []
    generated = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")
    county_count = len(runs)
    requested_count = county_count + len(pending)
    county_list = ", ".join(COUNTY_NAMES.get(run.fips, run.fips) for run in runs)
    model_counts = sorted({int(run.test["model_key"].nunique()) for run in runs})
    n_models = str(model_counts[0]) if len(model_counts) == 1 else f"{model_counts[0]}–{model_counts[-1]}"
    total_sales = sum(run.report["n_transactions"] for run in runs)
    total_test = sum(run.report["n_test"] for run in runs)
    level_series_starts = "; ".join(
        f"{COUNTY_NAMES.get(run.fips, run.fips)}: "
        f"{max(LEVEL_PLOT_START.year, pd.to_datetime(run.report['sale_date_range'][0]).year)}"
        for run in runs
    )

    county_sections = []
    for run in runs:
        pick = selected_row(run)
        validation = selected_validation_metrics(run)
        paired = paired_comparison_table(run)
        pick_text = (
            f"The baseline-penalty rho <code>{pick['rho']:.4g}</code> was selected once on the earlier "
            f"chronological validation block by <code>{html.escape(selection_protocol_label(run))}</code> "
            f"over MAPE, COD, |PRD−1|, and |PRB|. Its validation MAPE/COD/PRD/PRB were "
            f"{validation['MAPE']:.3f} / {validation['COD']:.3f} / {validation['PRD']:.3f} / "
            f"{validation['PRB']:.3f}; the held-out block was not used to choose it."
        )
        county_sections.append(f"""
        <section id='c{run.fips}'>
          <h2>{html.escape(run.label)}</h2>
          <p class='muted'>{run.report['n_train']:,} training sales and {run.report['n_test']:,} held-out sales;
          feature set <code>{html.escape(run.report['comparison']['feature_set'])}</code>,
          hyperparameters <code>{html.escape(run.report['comparison']['lgbm_config'])}</code>. {pick_text}</p>
          <h3>Held-out metrics by penalty strength</h3>
          <div class='scroll'>{_table(comparison_table(run), highlight="LGBM baseline")}</div>
          <p class='muted'>The highlighted green row is the rho selected on chronological validation before
          the full-training refit. Every other row is a descriptive held-out frontier point; it was not eligible
          to replace the selected model.</p>
          <h3>Paired bootstrap for predeclared comparisons</h3>
          <div class='scroll'>{_table(paired, digits=4)}</div>
          <p class='muted'>Values are distances from the ideal (|PRD−1|, |PRB|, |Cov|); <code>difference</code>
          is candidate minus reference, so negative is better. <code>p_improved</code> is the share of the
          {run.report['comparison']['n_bootstrap']} shared resampled time blocks in which it is negative.
          The four comparisons were predeclared from validation-selected model keys.</p>
        </section>
        """)

    pending_sections = []
    for run in pending:
        pending_sections.append(f"""
        <section id='c{html.escape(run.fips)}'>
          <h2>{html.escape(run.label)} — pending</h2>
          <div class='callout warn'>
            <p><b>Results are not yet available.</b> This county is intentionally omitted from every numeric
            table, figure, sample total, and cross-county conclusion in this partial report.</p>
            <p class='muted'>Expected artifact folder: <code>{html.escape(str(run.folder))}</code>.<br>
            Current status: {html.escape(run.reason)}</p>
          </div>
        </section>
        """)

    nav = " ".join(
        ["<a href='#overview'>Overview</a>", "<a href='#design'>Design</a>",
         "<a href='#baseline'>Baseline</a>", "<a href='#theory'>Theory</a>",
         "<a href='#results'>Results</a>", "<a href='#tradeoff'>Trade-off</a>",
         "<a href='#mechanism'>Mechanism</a>", "<a href='#caveats'>Caveats</a>"]
        + [f"<a href='#c{run.fips}'>{html.escape(COUNTY_NAMES.get(run.fips, run.fips))}</a>" for run in runs]
        + [f"<a href='#c{run.fips}'>{html.escape(COUNTY_NAMES.get(run.fips, run.fips))} (pending)</a>" for run in pending]
    )

    pending_notice = ""
    if pending:
        pending_names = ", ".join(run.label for run in pending)
        pending_notice = f"""
<div class='callout warn'>
  <p><b>Partial report: {county_count} of {requested_count} requested counties are complete.</b>
  {html.escape(pending_names)} is pending. Its numeric results are not substituted or imputed; every
  cross-county table, figure, total, and conclusion below uses only the {county_count} completed {'county' if county_count == 1 else 'counties'}.</p>
</div>
"""

    summary = operating_summary(runs)
    comparator_metrics = comparator_table(runs)
    neighbor_diagnostics = neighbor_diagnostics_table(runs)
    local_equity_metrics = local_equity_table(runs)
    legacy_trend_runs = []
    for run in runs:
        for neighbor in run.report["comparison"].get("neighbor_models", []):
            metadata = neighbor.get("final_feature_metadata", {})
            if metadata.get("time_trend") and metadata.get("time_trend_fit_mode") != "causal_prior":
                legacy_trend_runs.append(run.label)
                break
    legacy_trend_notice = ""
    if legacy_trend_runs:
        legacy_trend_notice = f"""
  <div class='callout warn'>
    <p><b>Time-adjusted neighbor results are legacy artifacts.</b> The artifacts for
    {html.escape(', '.join(legacy_trend_runs))} do not record the strictly-prior causal trend protocol.
    Their geographic-only comparable models remain past-only, but their time-adjusted comparable results
    should be treated as descriptive until rerun with
    the recorded causal-prior implementation.</p>
  </div>
"""
    baseline_outside_guidance = [
        run.label for run in runs if not (compliant(run, "PRB") and compliant(run, "PRD"))
    ]
    selected_rows = [(run, selected_row(run)) for run in runs]
    selected_within_guidance = [
        run.label for run, row in selected_rows
        if bool(row["PRB"] >= IAAO_PRB[0] and row["PRB"] <= IAAO_PRB[1]
                and row["PRD"] >= IAAO_PRD[0] and row["PRD"] <= IAAO_PRD[1])
    ]
    selected_costs = [float(row["rmse_log_increase_pct"]) for _, row in selected_rows]

    return f"""<!doctype html>
<html lang='en'>
<head>
<meta charset='utf-8'>
<meta name='viewport' content='width=device-width, initial-scale=1'>
<title>Soft vertical-equity constraints across {county_count} completed counties</title>
{CSS}
</head>
<body>
<header>
  <h1>Soft vertical-equity constraints: {county_count}-county ATTOM benchmark</h1>
  <p>Does penalizing covariance between the <b>log residual and log price</b> buy meaningful vertical equity,
  and at what accuracy cost? Unpenalized LightGBM is compared against <code>LGBCovPenalty[diff]</code> at a
  theory-calibrated grid of penalty strengths, on identical features and identical chronological splits, in
  {html.escape(county_list)}.</p>
  <p>Generated {html.escape(generated)} from <code>scripts/other_counties_benchmars.py</code> outputs
  (minimum qualified sale price ${html.escape(floor)}).</p>
</header>
<nav>{nav}</nav>
<main>

{pending_notice}

<section id='overview'>
  <h2>Overview</h2>
  <div class='cards'>
    <div class='card'><b>{len(runs)}</b>completed counties, one pipeline</div>
    <div class='card'><b>{total_sales:,}</b>qualified single-family sales modeled</div>
    <div class='card'><b>{total_test:,}</b>held-out sales</div>
    <div class='card'><b>{n_models}</b>models fitted per county</div>
  </div>
  <div class='callout'>
    <p><b>Headline.</b> The unpenalized baseline has at least one vertical-equity point estimate outside
    IAAO guidance in {html.escape(', '.join(baseline_outside_guidance)) if baseline_outside_guidance else 'none of the counties'}.
    The one baseline-penalty model selected on chronological validation falls within both PRD and PRB guidance
    on the held-out block in {len(selected_within_guidance)} of {county_count} counties
    {'(' + html.escape(', '.join(selected_within_guidance)) + ')' if selected_within_guidance else ''}; its
    held-out RMSE(log) change ranges from {_fmt(min(selected_costs), 2)}% to {_fmt(max(selected_costs), 2)}%.
    These are one-time evaluation results, not test-block selections.</p>
    <p><b>First-degree scaling and comparable-sales models are explicit comparators.</b> The <a href='#mechanism'>mechanism
    section</a> compares the existing covariance-penalty path with a centered linear scaling and three
    comparable-sales neighbor-information levels, without changing the penalty objective.</p>
    <p><b>Check level stability before interpreting equity.</b> The level-calibration panel and table show
    each county's held-out ratio path, including any test-window level shift that can contaminate ratio
    statistics.</p>
  </div>
  <h3>Sample construction</h3>
  <div class='scroll'>{_table(sample_table(runs))}</div>
  <p class='muted'>Every county goes through the same code path: Recorder <code>TRANSFERAMOUNT</code> as the
  target, an as-of join to the last Assessor History snapshot that closed before the sale, the Tax Assessor
  location crosswalk, and tract-level ACS lagged two years. Only <code>--county-fips</code> and
  <code>--assessor-dir</code> differ between the {county_count} runs.</p>
</section>

<section id='design'>
  <h2>Experimental design</h2>
  <p><b>Target and split.</b> The target is the Recorder transfer amount for property-use code 385
  (single-family). Sales are ordered by date; the last 20% form the held-out test block and the last 10% of
  the remainder is the validation block used only for model selection. No test-period information reaches
  the fit: the assessor snapshot must have closed before the sale, prior-sale features look strictly
  backwards, and ACS vintages lag the sale year by two.</p>
  <p><b>What varies and what does not.</b> The feature set, LightGBM configuration, penalty strength,
  first-degree scaling strength, comparable-sales search specification, and comparable representation are
  each selected on development/chronological-validation data only. Every such selection applies the same
  Pareto–Nash maximum-volume-hyperrectangle rule over MAPE, COD, |PRD−1|, and |PRB|. The split and random
  seed are held fixed. The held-out block is evaluated once after the final training refit.</p>
  <p><b>The penalty.</b> <code>LGBCovPenalty[diff]</code> trains on log price and adds a quadratic penalty on
  the covariance between the log residual and log price. On the per-observation scale the objective is</p>
  <p style='text-align:center'><code>MSE(log) + 0.5 &middot; rho &middot; Cov(f - Y, Y)<sup>2</sup></code>,
  &nbsp; Y = log(sale price), f = prediction.</p>
  <p>That covariance is related to, but not equivalent to, the ratio-price dependence summarised by PRD,
  PRB, and MKI. A regressive assessment often has a negative <code>Cov(f - Y, Y)</code>, because cheap
  properties are over-valued and expensive ones under-valued; reducing this one global log-scale covariance
  does not guarantee monotonic improvement in any of the ratio-scale statistics.</p>
  <p><b>Choosing rho.</b> rho is not swept blindly. Under the rank-one approximation in
  <code>scripts/theory_informed_rho_range_v2.py</code>, the covariance that survives training is
  <code>q(rho) = 1 / (1 + rho &middot; A / 2)</code> with <code>A = Var(f0)</code> the variance of the
  baseline prediction. Inverting it turns a <em>requested covariance reduction</em> into an explicit rho.
  Each county therefore gets its own rho grid calibrated to its own baseline, and the same requested
  reduction means the same thing in all counties even though the rho delivering it does not. The
  validation-selected point is the Pareto–Nash choice on that compact theory-calibrated grid; the full
  held-out rho path is retained only as a descriptive frontier bound.</p>
</section>

<section id='baseline'>
  <h2>Baseline: is there vertical inequity to fix?</h2>
  <div class='scroll'>{_table(baseline_table(runs))}</div>
  <p class='muted'>A PRD point estimate above 1.03 indicates regressivity. PRB is the IAAO regression slope
  of median-normalized ratios against log2 of a sale-price/indicated-value proxy; negative indicates
  regressivity. MKI below 1 is regressive. COD measures horizontal dispersion, with 15 as IAAO guidance for
  single-family. These are point-estimate comparisons with guidance bands, not confidence-interval tests.</p>
  {_figure(fig_baseline_ratio_curves(runs), "Median assessment ratio by held-out sale-price decile. Every panel pairs the baseline with its chronological-validation-selected penalty model. The curves are descriptive: sale price is both the ratio denominator and the binning variable.")}
  <p class='muted'>The decile curves show whether regressivity is distributed gradually across price levels
  or concentrated in a tail. The covariance penalty acts globally, while the first-degree scaling comparator
  changes only the global log-prediction slope; their trade-offs should be interpreted against the observed pattern in
  each county. They are not a
  standalone vertical-equity test and should be read with the IAAO statistics above.</p>
  <h3>Level calibration over time</h3>
  <p class='muted'>Every statistic on this page is computed from ratios, so a shift in the <em>level</em> of
  the ratio inside the held-out window would contaminate all of them at once. This is checked first.</p>
  {_figure(fig_level_drift(runs), "Quarterly median ratio from 2016 through 2026 on a shared y-scale, with the held-out block shaded for every county.")}
  <p class='muted'>Visible modeled-sales series begin: {html.escape(level_series_starts)}. This is the
  completed model sample after the existing sale-validation, as-of history-match, and property-use filters,
  not necessarily the first raw county record. Blank earlier periods have no eligible modeled sale yet.
  Miami-Dade and Middlesex have older modeled sales, but their pre-2016 observations are intentionally
  omitted from this common display window. All completed runs end in 2025, so 2026 is intentionally blank.</p>
  <div class='scroll'>{_table(level_drift_table(runs), digits=3)}</div>
  <div class='callout warn'>
    <p><b>Level shifts are a separate diagnostic.</b> A step in the median ratio within a held-out block can
    make ratio statistics reflect a level error rather than vertical inequity. Use the county-specific
    level-drift table before drawing that distinction.</p>
  </div>
</section>

<section id='theory'>
  <h2>Theory calibration of rho</h2>
  <div class='scroll'>{_table(theory_table(runs))}</div>
  <p class='muted'>A, B and C0 are measured on the training block from the baseline fit. Because
  <code>rho @ q</code> scales as <code>1/A</code>, counties with more dispersed predictions need smaller rho
  for the same effect, which is exactly why a single hard-coded rho would not transfer. The last column
  flags counties where the rho the theory says is needed to reach the IAAO PRD guidance boundary is already larger
  than the rho a 1%-of-MSE accuracy budget would allow, so the two criteria genuinely conflict.</p>
  {_figure(fig_theory_vs_realized(runs), "Descriptive test-frontier diagnostic. Left: the covariance reduction the theory predicts against the reduction observed once on held-out sales. Right: predicted versus realized accuracy cost. Points below the dashed line mean the theory is optimistic.")}

  <h3>Why theory calibration is not itself a model-selection rule</h3>
  <p>The theory is fed <code>A</code>, <code>B</code> and <code>C0</code> measured from the baseline's
  predictions <em>on the data it was fitted to</em>. Those predictions track their own training prices too
  closely, so the ratio-price dependence they exhibit is far smaller than the dependence the same model
  shows on new sales. The rho needed to close a small gap is small, and the resulting penalty is too weak
  for the gap that actually exists. The theory is therefore used to make a compact, comparable grid; its
  candidate is selected by the validation Pareto–Nash rule rather than by a test-period threshold.</p>
  <div class='scroll'>{_table(generalization_table(runs), digits=3)}</div>
  <p class='muted'>The validation column is the selected configuration as scored during model selection:
  fitted on the earlier part of the training block and evaluated on one later chronological validation block.
  It is not an out-of-fold or cross-validated estimate.</p>
  <div class='callout'>
    <p><b>Compare in-sample, validation, and held-out diagnostics.</b> The table above exposes the
    generalization gap separately for every county. Validation determines the deployable candidate;
    held-out diagnostics remain a one-time evaluation.</p>
  </div>
</section>

<section id='results'>
  <h2>How the metrics move with rho</h2>
  {_figure(fig_rho_evolution(runs, x_axis="requested"), "Descriptive held-out metrics against the requested covariance reduction. This common scale makes all counties directly comparable, but does not select a model.")}
  {_figure(fig_rho_evolution(runs, x_axis="rho"), "The same descriptive held-out trajectories against raw rho, showing how differently calibrated the counties are in rho units.")}
</section>

<section id='tradeoff'>
  <h2>What the equity gain costs</h2>
  {_figure(fig_tradeoff(runs), "Descriptive held-out equity improvement against accuracy loss. Each curve traces one county across its rho grid; the ring marks the point selected before test on chronological validation.")}
  <h3>Validation-selected operating point</h3>
  <div class='scroll'>{_table(summary, digits=4)}</div>
  <p class='muted'>There is exactly one row per county. It is chosen on the chronological validation block
  using the Pareto–Nash maximum-volume hyperrectangle over MAPE, COD, |PRD−1|, and |PRB|, then refit on the
  complete training block. The held-out columns are an evaluation of that predeclared choice. The remaining
  test-path points in the figures and county tables are descriptive frontier bounds and were never eligible
  to replace it.</p>
  {_figure(fig_paired_intervals(runs), "Paired time-block bootstrap for the four predeclared validation-selected comparisons: baseline penalty versus baseline, scaling versus baseline, neighbor versus baseline, and neighbor-plus-penalty versus neighbor. Shared resampling removes noise common to each pair.")}
</section>

<section id='mechanism'>
  <h2>Can linear scaling or comparable-sales information improve the residual pattern?</h2>
  <p><b>First-degree scaling.</b> Three fixed rolling origins refit the selected log-LightGBM pipeline on an
  earlier prefix and predict the following validation-sized block. The one scaling coefficient is fitted only
  on those out-of-sample predictions. Its deployed map uses the final model's training centre and level:</p>
  <p style='text-align:center'><code>log p&#770;<sub>new</sub> = mean(log y<sub>train</sub>) + a<sub>1</sub>
  [log p&#770; &minus; mean(log p&#770;<sub>train</sub>)]</code>.</p>
  <p>The level-only anchor is shown separately. Candidate scaling strengths form a monotone rolling-origin
  frontier and are selected with the same Pareto–Nash rule over MAPE, COD, |PRD−1|, and |PRB|. Values outside
  the rolling-OOF prediction support are clamped at its edge, so the held-out block supplies evaluation only.</p>
  <p><b>Comparable-sales features.</b> The baseline already uses available coordinates and geographic
  categories; these models are an explicit target-derived comparable-sales augmentation, not its first use of
  location. The three levels are geographic only; geographic plus a log-price trend and temporal kernel
  distance; and that time-adjusted variant plus standardized structural similarity. Direct comparable targets
  exclude the query row and use only earlier sales; recorded <code>causal_prior</code> trends also use only
  targets strictly before each query date. Each feature set supplies only the weighted adjusted log-price,
  weighted log-price dispersion, neighbor count, and effective neighbor count to the same LightGBM pipeline.
  A compact six-point, space-filling validation search varies <code>k</code> (3, 5, 8), geographic eligibility
  cap <code>Dmax</code> (1, 2, 5 km), past-sale age cap <code>Tmax</code> (365.25 or 730.5 days), and relative
  temporal/structural weights (0.25, 0.50, 0.75). Geographic weight remains 1 and normalized bandwidths remain
  fixed to avoid redundant weight–bandwidth scaling. The caps are enforced before ranking: all past eligible
  sales inside both caps are retrieved exactly, then ranked by the full composite distance. The legacy
  candidate multiplier is recorded but cannot truncate this capped exact-retrieval route. The full
  geographic+time+structural specification chooses the shared search point by the Pareto–Nash rule; the same
  selected point is then used for the controlled representation ablations. Only the selected representation
  receives a theory-calibrated rho grid and a second chronological-validation Pareto–Nash penalty selection.</p>
  {legacy_trend_notice}
  {_figure(fig_comparator_tradeoff(runs), "Descriptive held-out equity versus log-RMSE cost for the baseline penalty path, first-degree scaling frontier, all comparable-sales LGBM variants, and the validation-selected penalty for the selected neighbor representation.")}
  <h3>Comparator fit and held-out metrics</h3>
  <div class='scroll'>{_table(comparator_metrics, digits=4)}</div>
  <p class='muted'>The table separates rolling-origin and chronological-validation provenance from held-out
  evaluation. Every listed selected model was fixed before the test block; paired intervals are reported in
  the trade-off and county sections.</p>
  <h3>Comparable-sales coverage after eligibility caps and selected settings</h3>
  <div class='scroll'>{_table(neighbor_diagnostics, digits=3)}</div>
  <p class='muted'>Coverage is the share of rows with usable coordinates/date inputs or at least one valid
  comparable after the geographic and age caps, respectively. Structural columns are the actually available
  pre-specified core columns for each county; they are not tuned by the held-out block.</p>
  <h3>Held-out local-equity diagnostics</h3>
  <div class='scroll'>{_table(local_equity_metrics, digits=4)}</div>
  <p class='muted'>For geographic groups with the fixed minimum sample size, the table reports local
  median-ratio deviation from one and local COD as sale-weighted median/P90 summaries, plus Moran’s I of log
  residuals. These diagnostics test whether the selected neighbor representation changes local behavior;
  they do not by themselves establish an equity improvement.</p>
  {_figure(fig_linear_mapping(runs), "Raw baseline, final-training level anchor, and first-degree rolling-origin scale map. Shading marks rolling-OOF prediction support.")}
  {_figure(fig_linear_ratio_curves(runs), "Median held-out assessment ratio by sale-price decile for the raw baseline, final-training level anchor, validation-selected covariance penalty, and validation-selected first-degree rolling scale. The curves are descriptive because sale price is both denominator and binning variable.")}
  {_figure(fig_neighbor_ratio_curves(runs), "Median held-out assessment ratio by sale-price decile for every comparable-sales level and the validation-selected penalty of the selected representation. The curves are descriptive because sale price is both denominator and binning variable.")}
  <div class='callout warn'>
    <p><b>Interpretation.</b> Neighbor features address remaining local market and temporal information before
    the covariance penalty is applied. They do not guarantee simultaneous improvement in COD, PRD, PRB, or
    local diagnostics; those remain explicit one-time held-out evaluations.</p>
  </div>
</section>

{''.join(county_sections)}
{''.join(pending_sections)}

<section id='caveats'>
  <h2>Reading these results carefully</h2>
  <div class='callout warn'>
    <p><b>Price-scale accuracy is not the only yardstick here.</b> The transaction-value concentration plot
    shows how strongly price-scale R2 and RMSE can be governed by high-value transfers in each county.
    Accuracy is therefore compared on the log scale throughout, which is also the scale the penalty
    operates on, and equity is compared with the ratio statistics the IAAO standard defines.</p>
  </div>
  {_figure(fig_price_tail(runs), "Cumulative share of held-out transaction value by price percentile for every county.")}
  <ul>
    <li><b>Sale validation is code-based, not verified.</b> The broad cohort keeps transfers whose ATTOM codes
    are undocumented rather than demonstrably defective, so some non-arms-length transactions survive. The
    strict cohort is available but removes most of the sample in Cook.</li>
    <li><b>The theory is an approximation.</b> The rank-one derivation assumes the baseline is close to the
    conditional mean and that the penalty acts along a single direction. The theory-versus-realized plot
    reports its calibration separately for every county; requested reduction is an ordering device, not a
    quantitative guarantee.</li>
    <li><b>PRD and PRB can disagree.</b> PRD is a ratio of means and also responds to ratio dispersion,
    whereas PRB isolates a slope against value. The metric trajectories identify any county-specific
    saturation or trade-off, so neither metric should be read alone.</li>
    <li><b>In-sample and held-out equity can differ.</b> The generalization table compares training,
    one chronological validation block, and held-out diagnostics. A deployable calibration rule should be based on validation or
    held-out-like predictions rather than on the model's fitted values alone.</li>
    <li><b>The hyperparameter configurations are inherited, not re-tuned per county.</b> Three stored
    configurations compete, and the winner is chosen on each county's validation block, so the selection
    is clean. But the configurations themselves were tuned in an earlier Cook County experiment, and one
    of them carries the name <code>test_best_r2</code> from how it was derived there. Nothing on this page
    depends on that config being optimal, only on it being applied identically to every model in a
    county.</li>
    <li><b>Equity and accuracy are not the only trade-off.</b> Reducing the ratio-price covariance can raise
    COD, because compressing the systematic component of the ratio does not compress its dispersion. Both
    directions are reported side by side rather than collapsed into one score.</li>
    <li><b>One held-out block per county.</b> Every metric refers to a single chronological test window, so
    the bootstrap intervals capture sampling variation within that window, not variation across windows or
    market regimes. The level-drift panel provides a separate diagnostic for this limitation.</li>
    <li><b>Assessment values as features.</b> Where the validation block selects
    <code>status_quo_augmented</code>, the model sees the jurisdiction's own assessed values. That improves
    accuracy but partly imports the incumbent assessment's regressivity, which is a reason the penalty has
    work to do.</li>
  </ul>
</section>
</main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, action="append", required=True,
                        help="County run folder written by other_counties_benchmars.py (repeatable).")
    parser.add_argument("--out", type=Path, default=OUT_HTML)
    parser.add_argument("--floor", default="50,000", help="Minimum qualified sale price, for the header text.")
    parser.add_argument(
        "--allow-pending", action="store_true",
        help=("Build a partial report when a requested run is incomplete; pending counties get a "
              "labeled placeholder and are excluded from all numeric results."),
    )
    args = parser.parse_args()

    runs: list[CountyRun] = []
    pending: list[PendingRun] = []
    for folder in args.run:
        try:
            run = load_run(folder)
            if run is None:
                raise ValueError("required completed dashboard artifacts are not available")
            # This is the dashboard-specific completion signal.  It is written only after the
            # rolling-origin and comparable-sales stage has completed.
            linear_recalibration_spec(run)
        except (OSError, ValueError, KeyError, pd.errors.EmptyDataError) as exc:
            if not args.allow_pending:
                raise
            status = str(exc).strip() or type(exc).__name__
            pending.append(pending_run(folder, status))
            print(f"[county-penalty] pending {folder}: {status}", file=sys.stderr)
        else:
            runs.append(run)
    if not runs:
        raise SystemExit("No run folder contained a completed penalty comparison.")
    order = list(COUNTY_NAMES)
    runs.sort(key=lambda run: order.index(run.fips) if run.fips in order else len(order))
    pending.sort(key=lambda run: order.index(run.fips) if run.fips in order else len(order))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(build_html(runs, floor=args.floor, pending=pending), encoding="utf-8")
    suffix = f"; {len(pending)} pending" if pending else ""
    print(
        f"[county-penalty] wrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB) "
        f"for {len(runs)} completed counties{suffix}"
    )


if __name__ == "__main__":
    main()
