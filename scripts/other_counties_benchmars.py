#!/usr/bin/env python3
"""Build a bounded Cook County ATTOM sale-price experiment.

The target is Recorder ``TRANSFERAMOUNT``.  Each transaction is joined to the
latest Assessor History record whose assessor year ended before the sale, so no
post-sale property characteristics enter the model.  The current Tax Assessor
extract supplies only a checked property-location crosswalk; time-varying
characteristics always come from Assessor History.

``--sale-cohort`` chooses how undocumented Recorder codes are treated.  The broad
default excludes a transaction only when a code shows the sale is unsuitable, and
carries codes that ATTOM has not documented as audit flags.  ``strict`` admits only
reviewed inclusion codes and is the sensitivity cohort: because
``TRANSFERAMOUNTINFOACCURACY`` is populated with the "full amount stated" code on a
very small minority of Cook County transfers, the strict rule alone removes most of
the county's sales.

After the unpenalized LightGBM baseline is selected, the same features, split, and
hyperparameters are reused to fit ``LGBCovPenalty[diff]`` at a grid of penalty
strengths.  The grid is not guessed: each rho is the closed-form value that the
rank-one theory in ``scripts/theory_informed_rho_range_v2.py`` predicts will remove
a requested fraction of ``Cov(f - log price, log price)``, so the same requested
shrinkage means the same thing in every county even though the rho that delivers it
does not.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import yaml
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.neighbors import NearestNeighbors

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.motivation_utils import _build_time_block_bootstrap_indices, _compute_extended_metrics
from soft_constrained_models.boosting_models import LGBCovPenalty
from preprocessing.spatiotemporal_neighbors import SpatioTemporalKernelTargetNeighbors
from scripts.theory_informed_rho_range_v2 import compute_theory_for_predictions


def _capture_neighbor_transformer_provenance() -> dict:
    """Pin the local transformer implementation and API used by this experiment."""
    source = Path(inspect.getfile(SpatioTemporalKernelTargetNeighbors)).resolve()
    expected_source = (ROOT / "preprocessing/spatiotemporal_neighbors.py").resolve()
    if source != expected_source:
        raise RuntimeError(
            "County benchmark must use the repository's spatiotemporal-neighbor transformer; "
            f"imported {source}, expected {expected_source}."
        )
    required_parameters = {
        "k", "lat_col", "lon_col", "date_col", "kernel", "bandwidth", "bandwidth_scale",
        "geo_weight", "target_transform", "include_aggregate", "include_diagnostics",
        "categorical_filter_roots", "use_feature_distance", "numeric_feature_cols", "feature_scaler",
        "feature_alpha", "feature_bandwidth", "candidate_multiplier", "batch_query_size",
        "full_pool_batch_size", "n_jobs", "use_time_trend", "time_trend", "time_trend_fit_mode",
        "use_time_decay", "time_weight", "time_bandwidth_days", "neighbor_time_rule",
        "insufficient_neighbors", "exclude_self", "feature_prefix",
        "max_distance_km", "max_time_distance_days",
    }
    available = set(inspect.signature(SpatioTemporalKernelTargetNeighbors).parameters)
    missing = sorted(required_parameters - available)
    if missing:
        raise RuntimeError(
            "The imported spatiotemporal-neighbor transformer lacks required parameters: "
            f"{missing}."
        )
    return {
        "module_path": str(source),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "required_constructor_parameters": sorted(required_parameters),
    }


# Capture this at module import, before a long-running job can observe later edits
# to the shared working tree.
NEIGHBOR_TRANSFORMER_PROVENANCE = _capture_neighbor_transformer_provenance()


ASSESSOR_DIR = ROOT / "data/dewey-downloads/cookcounty-2016-2025-all-features"
RECORDER_DIR = ROOT / "data/dewey-downloads/10-counties-recorder-2016-2025"
TAX_ASSESSOR_DIR = ROOT / "data/dewey-downloads/9-counties-tax-assessor-missingharris-anyyear"
ACS_DIR = ROOT / "data/CensusData/acs5"
OUTPUT_DIR = ROOT / "output/attom_recorder_sample"
COOK_FIPS = "17031"
LGBM_CONFIG_PATH = ROOT / "best_lgbm_baseline_configs.yaml"
SALE_VALIDATION_DICTIONARY_PATH = (
    ROOT / "data/dewey-downloads/data_dictionaries/attom_recorder_residential_avm_sale_validation_dictionary.csv"
)
RECORDER_COLUMNS = [
    "ATTOMID", "TRANSACTIONID", "DOCUMENTRECORDINGCOUNTYFIPS",
    "INSTRUMENTDATE", "RECORDINGDATE", "TRANSFERAMOUNT", "DOCUMENTTYPECODE", "ARMSLENGTHFLAG",
    "TRANSFERINFOMULTIPARCELFLAG", "FORECLOSUREAUCTIONSALE", "QUITCLAIMFLAG",
    "TRANSFERINFODISTRESSCIRCUMSTANCECODE", "PARTIALINTEREST", "TRANSACTIONTYPE",
    "TRANSFERAMOUNTINFOACCURACY", "TRANSFERINFOPURCHASETYPECODE",
]
RECORDER_PRIOR_COLUMNS = RECORDER_COLUMNS.copy()
TAX_ASSESSOR_COLUMNS = [
    "ATTOMID", "SITUSSTATECOUNTYFIPS", "PARCELNUMBERFORMATTED", "PARCELNUMBERPREVIOUS",
    "PROPERTYADDRESSFULL", "LATITUDE", "LONGITUDE", "CENSUSTRACT", "PUBLICATIONDATE",
    "ASSRLASTUPDATED", "LASTASSESSORTAXROLLUPDATE", "TAXYEARASSESSED",
]
ACS_FEATURES = [
    "B01003_001E", "B19013_001E", "B25001_001E", "B25003_002E", "B25003_003E", "B25064_001E",
]
NUMERIC_FEATURES = [
    "AREABUILDING", "AREALOTSF", "AREA1STFLOOR", "AREA2NDFLOOR", "AREAUPPERFLOORS",
    "AREAGROSS", "AREALOTACRES", "AREALOTDEPTH", "AREALOTWIDTH", "YEARBUILT",
    "YEARBUILTEFFECTIVE", "BEDROOMSCOUNT", "BATHCOUNT", "BATHPARTIALCOUNT", "ROOMSCOUNT",
    "STORIESCOUNT", "UNITSCOUNT", "PARKINGGARAGEAREA", "PARKINGSPACECOUNT", "FIREPLACECOUNT",
    "ROOMSBASEMENTAREA", "ROOMSBASEMENTAREAFINISHED", "ROOMSBASEMENTAREAUNFINISHED", "PORCHAREA",
    "PATIOAREA", "DECKAREA", "BALCONYAREA", "POOLAREA", "LATITUDE", "LONGITUDE",
    "tax_assessor_latitude", "tax_assessor_longitude", *ACS_FEATURES,
    "TAXASSESSEDVALUEIMPROVEMENTS", "TAXASSESSEDVALUELAND", "TAXASSESSEDVALUETOTAL",
    "TAXMARKETVALUEIMPROVEMENTS", "TAXMARKETVALUELAND", "TAXMARKETVALUETOTAL", "TAXBILLEDAMOUNT",
]
CATEGORICAL_FEATURES = [
    "PROPERTYUSESTANDARDIZED", "STRUCTURESTYLE", "EXTERIOR1CODE", "FOUNDATION", "CONSTRUCTION",
    "HVACCOOLINGDETAIL", "HVACHEATINGDETAIL", "HVACHEATINGFUEL", "PARKINGGARAGE", "FIREPLACE",
    "POOL", "PORCHCODE", "ROOFCONSTRUCTION", "ROOFMATERIAL", "PROPERTYADDRESSZIP",
    "PROPERTYADDRESSCITY", "PROPERTYJURISDICTIONNAME", "NEIGHBORHOODCODE", "CENSUSTRACT",
    "LEGALTOWNSHIP", "MINORCIVILDIVISIONNAME", "ZONEDCODELOCAL", "tax_assessor_geoid",
]
HISTORY_COLUMNS = list(dict.fromkeys([
    "ATTOMID", "SITUSSTATECOUNTYFIPS", "ASSESSORHISTORYYEAR", "TAXYEARASSESSED",
    "PARCELNUMBERFORMATTED", "PARCELNUMBERPREVIOUS", "PROPERTYADDRESSFULL",
    "ASSESSORLASTSALEDATE", "ASSESSORLASTSALEAMOUNT",
    *NUMERIC_FEATURES, *CATEGORICAL_FEATURES,
]))
ASSESSMENT_VALUE_COLUMNS = {
    "TAXASSESSEDVALUEIMPROVEMENTS", "TAXASSESSEDVALUELAND", "TAXASSESSEDVALUETOTAL",
    "TAXMARKETVALUEIMPROVEMENTS", "TAXMARKETVALUELAND", "TAXMARKETVALUETOTAL", "TAXBILLEDAMOUNT",
}
FEATURE_SET_OPTIONS = {
    "ccao_core_acs": (False, False),
    "attom_market_history": (True, False),
    "status_quo_augmented": (True, True),
    # Backward-compatible labels used by the first version of this script.
    "ccao_like": (False, False),
    "ccao_like_plus_prior_sale": (True, False),
}
BASELINE_MODEL = "LGBMRegressor"
BASELINE_KEY = "lgbm_baseline"
PENALIZED_MODEL = "LGBCovPenalty[diff]"
SCALING_MODEL = "Rolling first-degree scaling"
SCALING_MODEL_KEY = "baseline_linear_scale_selected"
NEIGHBOR_LGBM_MODEL = "Neighbor LGBMRegressor"
NEIGHBOR_PENALIZED_MODEL = "Neighbor LGBCovPenalty[diff]"
ARTIFACT_SCHEMA_VERSION = 3
# Requested reductions of the baseline log-residual/log-price covariance.  Each one
# is mapped to a rho by the rank-one theory, so a county's grid is calibrated to its
# own baseline rather than copied from another county.
# A compact, theory-calibrated path is enough to trace the relevant trade-off
# without multiplying an already expensive cross-county study.  The command-line
# option remains available when a denser descriptive path is genuinely needed.
COVARIANCE_SHRINKAGE_TARGETS = (0.10, 0.25, 0.50, 0.67, 0.80, 0.90, 0.97)
# Theory anchors reported alongside the requested-shrinkage grid.
THEORY_ANCHORS = {
    "rho_prd_guidance": "prd_guidance_boundary",
    "rho_budget_1pct_mse": "one_percent_mse_budget",
}
PRD_TARGETS = (0.98, 1.0, 1.02, 1.03, 1.05)
ACCURACY_BUDGETS = (0.005, 0.01, 0.02, 0.05)
# The search is deliberately a small space-filling design, rather than a
# Cartesian product.  It exercises every requested spatial hyperparameter while
# remaining viable for the largest county.  Geographic and temporal eligibility
# are hard caps, not soft kernel preferences.  The selected full specification
# is propagated to the two ablations so they remain controlled comparisons.
NEIGHBOR_FEATURE_COLUMNS = (
    "BEDROOMSCOUNT", "YEARBUILT", "AREABUILDING", "AREALOTSF", "BATHCOUNT", "BATHPARTIALCOUNT",
)


@dataclass(frozen=True)
class NeighborSearchSpec:
    """A bounded comparable-sales search point selected on validation only."""

    key: str
    k: int
    max_distance_km: float
    max_time_distance_days: float
    time_weight: float
    feature_weight: float

    def as_dict(self) -> dict:
        return {
            "key": self.key,
            "k": int(self.k),
            "max_distance_km": float(self.max_distance_km),
            "max_time_distance_days": float(self.max_time_distance_days),
            "time_weight": float(self.time_weight),
            "feature_weight": float(self.feature_weight),
        }


# The largest radius occurs only in two of six screening points.  This keeps
# exact radius retrieval tractable in dense counties while still allowing the
# validation selector to trade coverage against locality in sparse areas.
NEIGHBOR_SEARCH_SPECS = (
    NeighborSearchSpec("s01", 3, 1.0, 365.25, 0.25, 0.25),
    NeighborSearchSpec("s02", 3, 2.0, 730.50, 0.50, 0.75),
    NeighborSearchSpec("s03", 5, 5.0, 365.25, 0.75, 0.50),
    NeighborSearchSpec("s04", 5, 1.0, 730.50, 0.75, 0.50),
    NeighborSearchSpec("s05", 8, 2.0, 365.25, 0.25, 0.75),
    NeighborSearchSpec("s06", 8, 5.0, 730.50, 0.50, 0.25),
)
LOCAL_EQUITY_MIN_GROUP_N = 30
LOCAL_EQUITY_MORAN_K = 8
NASH_SELECTION_METRICS = (
    ("MAPE", "minimize"),
    ("COD", "minimize"),
    ("PRD", "target_one"),
    ("PRB", "target_zero"),
)
NASH_SELECTION_EPS = 1e-6


@dataclass(frozen=True)
class NeighborVariant:
    """One leakage-controlled comparable-sales feature specification."""

    key: str
    label: str
    use_time_trend: bool = False
    use_time_decay: bool = False
    use_feature_distance: bool = False
    geo_weight: float = 1.0
    time_weight: float = 1.0
    feature_weight: float = 1.0


NEIGHBOR_VARIANTS = (
    NeighborVariant("geo", "Geographic comparables"),
    NeighborVariant(
        "geo_time", "Geographic + time-adjusted comparables",
        use_time_trend=True, use_time_decay=True, time_weight=0.5,
    ),
    NeighborVariant(
        "geo_time_features", "Geographic + time + structural comparables",
        use_time_trend=True, use_time_decay=True, use_feature_distance=True,
        time_weight=0.5, feature_weight=0.5,
    ),
)


@dataclass(frozen=True)
class SaleValidationPolicy:
    """Documented target-sale rules, kept separate from model fitting."""

    minimum_sale_price: float
    dictionary_path: str
    dictionary_sha256: str
    code_decisions: dict[str, dict[str, str]]
    arms_length_only: bool
    single_parcel_only: bool
    cohort: str


INCLUDE_CANDIDATE = "INCLUDE_CANDIDATE"
INCLUDE_WITH_MISSINGNESS_FLAG = "INCLUDE_WITH_MISSINGNESS_FLAG"
NOT_USED_PLACEHOLDER = "NOT_USED_PLACEHOLDER"
MISSING_CODE = "<NULL>"
BROAD_COHORT = "broad"
STRICT_COHORT = "strict"
SALE_COHORTS = (BROAD_COHORT, STRICT_COHORT)
# Decisions that establish a transaction is unsuitable on the evidence of the code
# itself.  Every other negative decision (EXCLUDE_PENDING_MAPPING,
# EXCLUDE_PENDING_NULL_SEMANTICS, and unseen codes) records missing vendor
# documentation rather than a defective sale, so the broad cohort keeps those rows
# and relies on the per-field decision columns to flag them.
HARD_EXCLUDE_DECISIONS = frozenset({"EXCLUDE", "EXCLUDE_SEPARATE_MODEL"})
# Trailing secondary-unit part of an address.  Each token must be followed by a
# space, a digit, or the end of the string so ordinary street names that merely
# begin with one of them ("FLOWER", "PHEASANT") are left intact.
UNIT_DESIGNATOR = (
    r"\s(?:#|APT|UNIT|STE|SUITE|PH|FL|FLR|FLOOR|BLDG|RM|SPC|TRLR|LOT)(?:[\s\d].*)?$"
)
REQUIRED_SALE_VALIDATION_FIELDS = {
    "DOCUMENTTYPECODE",
    "TRANSFERAMOUNTINFOACCURACY",
    "PARTIALINTEREST",
    "TRANSACTIONTYPE",
    "TRANSFERINFOPURCHASETYPECODE",
    "ARMSLENGTHFLAG",
    "TRANSFERINFOMULTIPARCELFLAG",
    "TRANSFERINFODISTRESSCIRCUMSTANCECODE",
    "FORECLOSUREAUCTIONSALE",
    "QUITCLAIMFLAG",
}


def report_progress(label: str, completed: int, total: int, started_at: float) -> None:
    """Print a compact elapsed-time progress bar for bounded long-running work."""
    width = 24
    filled = round(width * completed / total)
    bar = "#" * filled + "-" * (width - filled)
    elapsed = perf_counter() - started_at
    end = "\n" if completed == total else "\r"
    print(f"[{elapsed:7.1f}s] {label}: [{bar}] {completed}/{total}", end=end, flush=True)


def sample_files(directory: Path, pattern: str, count: int) -> list[Path]:
    """Return evenly spaced shards, limiting I/O while covering the extract."""
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} files in {directory}")
    if not 1 <= count <= len(files):
        raise ValueError(f"Sample size must be between 1 and {len(files)}")
    return [files[i] for i in np.linspace(0, len(files) - 1, count, dtype=int)]


def files_or_sample(directory: Path, pattern: str, count: int) -> list[Path]:
    """Return all shards when count is 0; otherwise return an even shard sample."""
    files = sorted(directory.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No {pattern} files in {directory}")
    if count == 0:
        return files
    return sample_files(directory, pattern, count)


def readable_parquet_files(files: list[Path]) -> tuple[list[Path], list[str]]:
    """Split shards into readable ones and those with an unusable Parquet footer.

    An interrupted download can leave an HTML error page behind the ``.parquet``
    name, which aborts a whole scan.  Skipping the shard keeps a county runnable,
    so the names are returned for the run report instead of being swallowed.
    """
    readable, unreadable = [], []
    for file in files:
        try:
            pq.ParquetFile(file).metadata
        except Exception:
            unreadable.append(file.name)
        else:
            readable.append(file)
    if not readable:
        raise ValueError(f"No readable Parquet shards among {len(files)} files.")
    return readable, unreadable


def normalize_property_use(series: pd.Series) -> pd.Series:
    """Normalize stored property-use codes so they compare equal to user input."""
    value = series.astype("string").str.strip().str.upper().str.replace(r"\.0$", "", regex=True)
    return value.where(value.notna() & value.ne(""))


def parse_property_use_codes(raw: str | None) -> set[str]:
    """Parse user-supplied ATTOM property-use codes without inferring their meaning.

    An empty selection means the modeled universe is every observed code.
    """
    codes = pd.Series([value for value in (raw or "").split(",")], dtype="string")
    return set(normalize_property_use(codes).dropna())


def clean_fips(series: pd.Series) -> pd.Series:
    return series.astype("string").str.replace(r"\.0$", "", regex=True).str.zfill(5)


def fips_scan_filter(source: ds.Dataset, column: str, county_fips: str) -> ds.Expression:
    """Return a parquet predicate covering the FIPS forms accepted by ``clean_fips``."""
    field = source.schema.field(column)
    county_number = str(int(county_fips))
    if pa.types.is_string(field.type) or pa.types.is_large_string(field.type):
        return ds.field(column).isin([county_fips, county_number, f"{county_number}.0"])
    return ds.field(column) == int(county_fips)


def clean_identifier(series: pd.Series) -> pd.Series:
    """Standardize an APN or address for an equality check without imputing it."""
    value = series.astype("string").str.upper().str.replace(r"[^A-Z0-9]", "", regex=True)
    return value.where(value.notna() & value.ne(""))


def clean_street_address(series: pd.Series) -> pd.Series:
    """Standardize an address down to its building, dropping any unit designator.

    Two records that differ only by unit share a building, hence a coordinate pair
    and a Census tract, so a unit-level difference is not evidence that the sources
    describe different properties.
    """
    value = series.astype("string").str.upper().str.replace(r"\s+", " ", regex=True).str.strip()
    value = value.str.replace(UNIT_DESIGNATOR, "", regex=True)
    value = value.str.replace(r"[^A-Z0-9]", "", regex=True)
    return value.where(value.notna() & value.ne(""))


def clean_recorder_code(series: pd.Series) -> pd.Series:
    """Normalize an opaque ATTOM code while preserving its documented identity."""
    value = series.astype("string").str.strip().str.upper()
    return value.where(value.notna() & value.ne(""))


def load_sale_validation_dictionary(path: Path) -> tuple[dict[str, dict[str, str]], str]:
    """Load the reviewed code decisions and reject incomplete or ambiguous dictionaries."""
    required_columns = {"field", "code", "decision"}
    try:
        with path.open(newline="", encoding="utf-8") as file:
            reader = csv.DictReader(file)
            if reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
                raise ValueError(f"must contain columns: {', '.join(sorted(required_columns))}")
            decisions: dict[str, dict[str, str]] = {}
            for row_number, row in enumerate(reader, start=2):
                field = (row["field"] or "").strip().upper()
                code = (row["code"] or "").strip().upper()
                decision = (row["decision"] or "").strip().upper()
                if not field or not code or not decision:
                    raise ValueError(f"row {row_number} has an empty field, code, or decision")
                if code == MISSING_CODE:
                    code = MISSING_CODE
                if code in decisions.setdefault(field, {}):
                    raise ValueError(f"duplicate code {code!r} for {field} on row {row_number}")
                decisions[field][code] = decision
    except FileNotFoundError as error:
        raise FileNotFoundError(f"Sale-validation dictionary not found: {path}") from error
    except csv.Error as error:
        raise ValueError(f"Invalid sale-validation dictionary {path}: {error}") from error
    missing_fields = sorted(REQUIRED_SALE_VALIDATION_FIELDS - set(decisions))
    if missing_fields:
        raise ValueError(
            f"Sale-validation dictionary is missing fields: {', '.join(missing_fields)}"
        )
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return decisions, digest


def dictionary_decision(
    data: pd.DataFrame, field: str, policy: SaleValidationPolicy,
) -> tuple[pd.Series, pd.Series]:
    """Return normalized codes and reviewed decisions; unseen codes remain unmapped."""
    codes = clean_recorder_code(data[field]).fillna(MISSING_CODE)
    decisions = codes.map(policy.code_decisions[field])
    return codes, decisions.astype("string")


def clean_tract(series: pd.Series) -> pd.Series:
    """Return a six-digit Census tract code, leaving malformed values null."""
    value = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    value = value.where(value.str.fullmatch(r"\d{1,6}"), pd.NA)
    return value.str.zfill(6)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.where(denominator.gt(0))
    return numerator / denominator


def apply_sale_validation(
    data: pd.DataFrame, policy: SaleValidationPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Apply documented sale rules and retain every exclusion reason for audit.

    The strict cohort admits only codes reviewed as positive inclusions.  The broad
    cohort drops a transaction only when a code proves it unsuitable, so codes that
    are merely undocumented stay in the sample carrying their decision columns and
    ``sale_validation_has_unverified_code``.
    """
    result = data.copy()
    decisions: dict[str, pd.Series] = {}
    codes: dict[str, pd.Series] = {}
    for field in REQUIRED_SALE_VALIDATION_FIELDS:
        codes[field], decisions[field] = dictionary_decision(result, field, policy)
        result[f"sale_validation_{field.lower()}_decision"] = decisions[field].fillna("UNMAPPED")
    result["partial_interest_missing_or_unknown"] = (
        codes["PARTIALINTEREST"].eq(MISSING_CODE) | codes["PARTIALINTEREST"].eq("102")
    )

    strict = policy.cohort == STRICT_COHORT
    no_rows = pd.Series(False, index=result.index, dtype="boolean")

    def excluded(field: str, *allowed_decisions: str) -> pd.Series:
        """Fail the field gate: on any non-positive decision when strict, else only on
        a decision that itself proves the transaction unsuitable."""
        if strict:
            return ~decisions[field].isin(allowed_decisions)
        # fillna keeps unmapped codes eligible; the shared fillna(True) below would
        # otherwise turn an undecided gate back into an exclusion.
        return decisions[field].isin(HARD_EXCLUDE_DECISIONS).fillna(False)

    def unmapped(field: str) -> pd.Series:
        """Codes absent from the reviewed dictionary; only the strict cohort drops them."""
        return decisions[field].isna() if strict else no_rows

    unverified = pd.Series(False, index=result.index, dtype="boolean")
    for field, allowed in (
        ("DOCUMENTTYPECODE", (INCLUDE_CANDIDATE,)),
        ("TRANSFERAMOUNTINFOACCURACY", (INCLUDE_CANDIDATE,)),
        ("PARTIALINTEREST", (INCLUDE_CANDIDATE, INCLUDE_WITH_MISSINGNESS_FLAG)),
        ("TRANSACTIONTYPE", (INCLUDE_CANDIDATE, NOT_USED_PLACEHOLDER)),
    ):
        unverified |= ~decisions[field].isin(allowed).fillna(False)
    result["sale_validation_has_unverified_code"] = unverified

    duplicate_transaction = result["TRANSACTIONID"].notna() & result.duplicated("TRANSACTIONID", keep="first")
    checks: list[tuple[str, pd.Series]] = [
        ("missing_attomid", result["ATTOMID"].isna()),
        ("missing_transaction_id", result["TRANSACTIONID"].isna()),
        ("missing_sale_date", result["sale_date"].isna()),
        ("missing_or_nonpositive_sale_price", result["sale_price"].isna() | result["sale_price"].le(0)),
        (
            "below_minimum_sale_price",
            result["sale_price"].gt(0) & result["sale_price"].lt(policy.minimum_sale_price),
        ),
        ("document_type_unmapped", unmapped("DOCUMENTTYPECODE")),
        ("document_type_not_allowed", excluded("DOCUMENTTYPECODE", INCLUDE_CANDIDATE)),
        ("transfer_amount_accuracy_unmapped", unmapped("TRANSFERAMOUNTINFOACCURACY")),
        (
            "transfer_amount_accuracy_not_allowed",
            excluded("TRANSFERAMOUNTINFOACCURACY", INCLUDE_CANDIDATE),
        ),
        ("partial_interest_unmapped", unmapped("PARTIALINTEREST")),
        (
            "partial_interest_not_allowed",
            excluded("PARTIALINTEREST", INCLUDE_CANDIDATE, INCLUDE_WITH_MISSINGNESS_FLAG),
        ),
        ("transaction_type_unmapped", unmapped("TRANSACTIONTYPE")),
        (
            "transaction_type_not_allowed",
            excluded("TRANSACTIONTYPE", INCLUDE_CANDIDATE, NOT_USED_PLACEHOLDER),
        ),
    ]
    if policy.arms_length_only:
        checks.extend([
            ("arms_length_unmapped", unmapped("ARMSLENGTHFLAG")),
            ("not_arms_length", excluded("ARMSLENGTHFLAG", INCLUDE_CANDIDATE)),
        ])
    if policy.single_parcel_only:
        checks.extend([
            ("multi_parcel_unmapped", unmapped("TRANSFERINFOMULTIPARCELFLAG")),
            (
                "multi_parcel_or_unallocated_price",
                excluded("TRANSFERINFOMULTIPARCELFLAG", INCLUDE_CANDIDATE),
            ),
        ])
    checks.extend([
        ("foreclosure_auction_unmapped", unmapped("FORECLOSUREAUCTIONSALE")),
        ("foreclosure_or_auction_sale", excluded("FORECLOSUREAUCTIONSALE", INCLUDE_CANDIDATE)),
        ("quitclaim_unmapped", unmapped("QUITCLAIMFLAG")),
        ("quitclaim", excluded("QUITCLAIMFLAG", INCLUDE_CANDIDATE)),
        ("distress_circumstance_unmapped", unmapped("TRANSFERINFODISTRESSCIRCUMSTANCECODE")),
        ("distress_circumstance", excluded("TRANSFERINFODISTRESSCIRCUMSTANCECODE", INCLUDE_CANDIDATE)),
        ("duplicate_transaction_id", duplicate_transaction),
    ])

    reasons = pd.Series("", index=result.index, dtype="string")
    first_reason = pd.Series(pd.NA, index=result.index, dtype="string")
    eligible = pd.Series(True, index=result.index, dtype="boolean")
    waterfall_rows = [{
        "stage": "raw_county_recorder_transfers",
        "n_excluded_at_stage": 0,
        "n_remaining": int(len(result)),
    }]
    for reason, failed in checks:
        failed = failed.fillna(True).astype(bool)
        newly_excluded = eligible & failed
        prior_reasons = reasons.loc[failed]
        reasons.loc[failed] = prior_reasons.mask(prior_reasons.eq(""), reason).mask(
            prior_reasons.ne(""), prior_reasons + ";" + reason,
        )
        first_reason.loc[newly_excluded] = reason
        eligible &= ~failed
        waterfall_rows.append({
            "stage": reason,
            "n_excluded_at_stage": int(newly_excluded.sum()),
            "n_remaining": int(eligible.sum()),
        })
    result["sale_validation_eligible"] = eligible
    result["sale_validation_first_exclusion_reason"] = first_reason.fillna("included")
    result["sale_validation_exclusion_reasons"] = reasons.mask(reasons.eq(""), pd.NA)
    return (
        result.loc[eligible].copy().sort_values("sale_date").reset_index(drop=True),
        result,
        pd.DataFrame(waterfall_rows),
    )


def read_transactions(
    files: list[Path], county_fips: str, policy: SaleValidationPolicy,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read Recorder targets and preserve a complete, policy-based validation audit."""
    source = ds.dataset(files, format="parquet")
    raw = source.to_table(
        columns=RECORDER_COLUMNS,
        filter=fips_scan_filter(source, "DOCUMENTRECORDINGCOUNTYFIPS", county_fips),
    ).to_pandas()
    data = raw.loc[clean_fips(raw["DOCUMENTRECORDINGCOUNTYFIPS"]).eq(county_fips)].copy()
    data["sale_date"] = pd.to_datetime(data["INSTRUMENTDATE"], errors="coerce").fillna(
        pd.to_datetime(data["RECORDINGDATE"], errors="coerce")
    )
    data["sale_price"] = pd.to_numeric(data["TRANSFERAMOUNT"], errors="coerce")
    data["ATTOMID"] = pd.to_numeric(data["ATTOMID"], errors="coerce").astype("Int64")
    return apply_sale_validation(data, policy)


def read_history(files: list[Path], transaction_ids: pd.Series, county_fips: str) -> pd.DataFrame:
    """Read every Assessor characteristic only for sampled transaction properties."""
    ids = pa.array(transaction_ids.dropna().astype("int64").unique())
    source = ds.dataset(files, format="parquet")
    columns = [column for column in HISTORY_COLUMNS if column in source.schema.names]
    data = source.to_table(
        columns=columns,
        filter=ds.field("ATTOMID").isin(ids) & fips_scan_filter(source, "SITUSSTATECOUNTYFIPS", county_fips),
    ).to_pandas()
    data = data.loc[clean_fips(data["SITUSSTATECOUNTYFIPS"]).eq(county_fips)].copy()
    data["ATTOMID"] = pd.to_numeric(data["ATTOMID"], errors="coerce").astype("Int64")
    # ATTOM documents ASSESSORHISTORYYEAR as the assessor's assessment year;
    # TAXYEARASSESSED can refer to a different tax period.  With no dated
    # historical publication event in this extract, year end is conservative.
    assessor_year = pd.to_numeric(data["ASSESSORHISTORYYEAR"], errors="coerce").astype("Int64")
    data["assessed_through"] = pd.to_datetime(assessor_year.astype("string") + "-12-31", errors="coerce")
    return data.loc[data["ATTOMID"].notna() & data["assessed_through"].notna()].copy()


def match_history(transactions: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest history row strictly preceding each Recorder sale."""
    left = transactions.sort_values(["sale_date", "ATTOMID"])
    right = history.sort_values(["assessed_through", "ATTOMID"])
    result = pd.merge_asof(
        left, right, left_on="sale_date", right_on="assessed_through", by="ATTOMID",
        direction="backward", allow_exact_matches=False,
    ).dropna(subset=["assessed_through"]).sort_values("sale_date").reset_index(drop=True)
    result["history_lag_days"] = (result["sale_date"] - result["assessed_through"]).dt.days
    result["history_lag_years"] = result["history_lag_days"] / 365.25
    return result


def attach_recorder_prior_sales(
    data: pd.DataFrame, recorder_files: list[Path], county_fips: str, policy: SaleValidationPolicy,
) -> pd.DataFrame:
    """Add strictly pre-sale Recorder history features for each target property."""
    result = data.copy()
    ids = pa.array(result["ATTOMID"].dropna().astype("int64").unique())
    source = ds.dataset(recorder_files, format="parquet")
    raw = source.to_table(
        columns=RECORDER_PRIOR_COLUMNS,
        filter=ds.field("ATTOMID").isin(ids) & fips_scan_filter(source, "DOCUMENTRECORDINGCOUNTYFIPS", county_fips),
    ).to_pandas()
    history = raw.loc[clean_fips(raw["DOCUMENTRECORDINGCOUNTYFIPS"]).eq(county_fips)].copy()
    history["ATTOMID"] = pd.to_numeric(history["ATTOMID"], errors="coerce").astype("Int64")
    history["sale_date"] = pd.to_datetime(history["INSTRUMENTDATE"], errors="coerce").fillna(
        pd.to_datetime(history["RECORDINGDATE"], errors="coerce")
    )
    history["sale_price"] = pd.to_numeric(history["TRANSFERAMOUNT"], errors="coerce")
    history, _, _ = apply_sale_validation(history, policy)
    history = history.rename(columns={"sale_date": "recorder_sale_date", "sale_price": "recorder_sale_price"})
    history = history.sort_values(["recorder_sale_date", "ATTOMID"])

    left = result.reset_index(names="_rowid").sort_values(["sale_date", "ATTOMID"])
    latest = pd.merge_asof(
        left, history, left_on="sale_date", right_on="recorder_sale_date", by="ATTOMID",
        direction="backward", allow_exact_matches=False,
    ).set_index("_rowid")
    result["recorder_prior_sale_amount"] = latest["recorder_sale_price"]
    result["recorder_prior_sale_age_years"] = (
        (pd.to_datetime(result["sale_date"]) - latest["recorder_sale_date"]).dt.days / 365.25
    )
    result["recorder_log_prior_sale_price"] = np.log(result["recorder_prior_sale_amount"].where(
        result["recorder_prior_sale_amount"].gt(0)
    ))

    # The former implementation filtered every history row once per property.
    # These as-of joins retain its strict-before-date semantics while scanning
    # each sorted table once, including when several sales occur on one date.
    history["_prior_sale_count"] = history.groupby("ATTOMID", sort=False).cumcount().add(1)

    def count_before(cutoff: pd.Series) -> pd.Series:
        left = result[["ATTOMID"]].copy()
        left["_rowid"] = result.index
        left["_cutoff"] = cutoff
        prior = pd.merge_asof(
            left.sort_values(["_cutoff", "ATTOMID"]),
            history[["ATTOMID", "recorder_sale_date", "_prior_sale_count"]],
            left_on="_cutoff",
            right_on="recorder_sale_date",
            by="ATTOMID",
            direction="backward",
            allow_exact_matches=False,
        ).set_index("_rowid")["_prior_sale_count"]
        return prior.reindex(result.index).fillna(0).astype("int64")

    sale_dates = pd.to_datetime(result["sale_date"])
    result["recorder_prior_sale_count_all"] = count_before(sale_dates)
    result["recorder_prior_sale_count_3yr"] = (
        result["recorder_prior_sale_count_all"] - count_before(sale_dates - pd.DateOffset(years=3))
    )
    result["recorder_prior_sale_count_5yr"] = (
        result["recorder_prior_sale_count_all"] - count_before(sale_dates - pd.DateOffset(years=5))
    )
    return result


def read_tax_assessor(files: list[Path], transaction_ids: pd.Series, county_fips: str) -> pd.DataFrame:
    """Read a current, one-row-per-property location crosswalk without tax values."""
    source = ds.dataset(files, format="parquet")
    missing = sorted(set(TAX_ASSESSOR_COLUMNS) - set(source.schema.names))
    if missing:
        raise ValueError(f"Tax Assessor data is missing required crosswalk columns: {', '.join(missing)}")
    ids = pa.array(transaction_ids.dropna().astype("int64").unique())
    data = source.to_table(
        columns=TAX_ASSESSOR_COLUMNS,
        filter=ds.field("ATTOMID").isin(ids) & fips_scan_filter(source, "SITUSSTATECOUNTYFIPS", county_fips),
    ).to_pandas()
    data = data.loc[clean_fips(data["SITUSSTATECOUNTYFIPS"]).eq(county_fips)].copy()
    data["ATTOMID"] = pd.to_numeric(data["ATTOMID"], errors="coerce").astype("Int64")
    data = data.loc[data["ATTOMID"].notna()].copy()
    data["tax_assessor_tract"] = clean_tract(data.pop("CENSUSTRACT"))
    data["tax_assessor_geoid"] = county_fips + data["tax_assessor_tract"]
    latitude = pd.to_numeric(data.pop("LATITUDE"), errors="coerce")
    longitude = pd.to_numeric(data.pop("LONGITUDE"), errors="coerce")
    data["tax_assessor_coordinate_valid"] = latitude.between(-90, 90) & longitude.between(-180, 180)
    data["tax_assessor_latitude"] = latitude.where(data["tax_assessor_coordinate_valid"])
    data["tax_assessor_longitude"] = longitude.where(data["tax_assessor_coordinate_valid"])
    rename = {
        "PARCELNUMBERFORMATTED": "tax_assessor_apn",
        "PARCELNUMBERPREVIOUS": "tax_assessor_prior_apn",
        "PROPERTYADDRESSFULL": "tax_assessor_address",
        "PUBLICATIONDATE": "tax_assessor_publication_date",
        "ASSRLASTUPDATED": "tax_assessor_last_updated",
        "LASTASSESSORTAXROLLUPDATE": "tax_assessor_tax_roll_updated",
        "TAXYEARASSESSED": "tax_assessor_tax_year_assessed",
    }
    data = data.drop(columns=["SITUSSTATECOUNTYFIPS"]).rename(columns=rename)
    duplicate_columns = [
        "tax_assessor_tract", "tax_assessor_latitude", "tax_assessor_longitude",
        "tax_assessor_apn", "tax_assessor_prior_apn", "tax_assessor_address",
    ]
    conflicts = data.groupby("ATTOMID", dropna=False)[duplicate_columns].nunique(dropna=False).gt(1).any(axis=1)
    data["tax_assessor_attomid_ambiguous"] = data["ATTOMID"].map(conflicts).astype("boolean")
    # Equal duplicate rows can arise from repeated extract shards.  Conflicting
    # crosswalks are retained only as an ambiguity flag and never drive ACS.
    # Dates are parsed so ordering follows the calendar rather than string form, and
    # undated rows sort first so a dated record always wins the keep="last" pick.
    for column in ("tax_assessor_last_updated", "tax_assessor_publication_date"):
        data[column] = pd.to_datetime(data[column], errors="coerce")
    data = data.sort_values(["tax_assessor_last_updated", "tax_assessor_publication_date"], na_position="first")
    return data.drop_duplicates("ATTOMID", keep="last").reset_index(drop=True)


def attach_tax_assessor(data: pd.DataFrame, tax_assessor: pd.DataFrame) -> pd.DataFrame:
    """Attach the current location only when it agrees with the history record."""
    result = data.merge(tax_assessor, on="ATTOMID", how="left", validate="m:1", indicator="_tax_assessor_match")
    result["tax_assessor_matched"] = result.pop("_tax_assessor_match").eq("both")
    history_apns = [
        clean_identifier(result["PARCELNUMBERFORMATTED"]), clean_identifier(result["PARCELNUMBERPREVIOUS"]),
    ]
    current_apns = [
        clean_identifier(result["tax_assessor_apn"]), clean_identifier(result["tax_assessor_prior_apn"]),
    ]
    comparable_apns = (
        pd.concat(history_apns, axis=1).notna().any(axis=1)
        & pd.concat(current_apns, axis=1).notna().any(axis=1)
    )
    # Current and prior parcel numbers agree when any pair matches; comparing the
    # four pairs directly keeps the set-intersection meaning without a per-row loop.
    apn_overlap = pd.Series(False, index=result.index, dtype="boolean")
    for history_apn in history_apns:
        for current_apn in current_apns:
            apn_overlap |= history_apn.eq(current_apn).fillna(False)
    result["tax_assessor_apn_consistent"] = apn_overlap.where(comparable_apns).astype("boolean")
    history_address = clean_identifier(result["PROPERTYADDRESSFULL"])
    current_address = clean_identifier(result["tax_assessor_address"])
    comparable_address = history_address.notna() & current_address.notna()
    address_consistent = pd.Series(pd.NA, index=result.index, dtype="boolean")
    address_consistent.loc[comparable_address] = history_address.loc[comparable_address].eq(current_address.loc[comparable_address])
    result["tax_assessor_address_consistent"] = address_consistent
    # The exact flag above stays as an audit signal, but the usable-location gate
    # compares buildings: erasing a verified coordinate because one source wrote
    # "APT 3" and the other "# 3" discards a location that is in fact agreed.
    history_street = clean_street_address(result["PROPERTYADDRESSFULL"])
    current_street = clean_street_address(result["tax_assessor_address"])
    comparable_street = history_street.notna() & current_street.notna()
    result["tax_assessor_street_address_consistent"] = (
        history_street.eq(current_street).where(comparable_street).astype("boolean")
    )
    coordinate_valid = result["tax_assessor_coordinate_valid"].astype("boolean").fillna(False)
    attomid_unambiguous = ~result["tax_assessor_attomid_ambiguous"].astype("boolean").fillna(True)
    apn_not_conflicting = result["tax_assessor_apn_consistent"].astype("boolean").ne(False).fillna(True)
    address_not_conflicting = (
        result["tax_assessor_street_address_consistent"].astype("boolean").ne(False).fillna(True)
    )
    result["tax_assessor_location_usable"] = (
        result["tax_assessor_matched"]
        & coordinate_valid
        & attomid_unambiguous
        & apn_not_conflicting
        & address_not_conflicting
    )
    result.loc[~result["tax_assessor_location_usable"], ["tax_assessor_latitude", "tax_assessor_longitude", "tax_assessor_geoid"]] = np.nan
    return result


def read_acs(directory: Path, county_fips: str) -> pd.DataFrame:
    """Read the local ACS tract panel and enforce one record per vintage/tract."""
    files = sorted(directory.glob(f"year=*/county_fips={county_fips}/tracts.parquet"))
    if not files:
        raise FileNotFoundError(f"No ACS tract files for county {county_fips} in {directory}")
    data = ds.dataset(files, format="parquet").to_table().to_pandas()
    required = {"GEOID", "acs_vintage", "county_fips", *ACS_FEATURES}
    missing = sorted(required - set(data.columns))
    if missing:
        raise ValueError(f"ACS data is missing required columns: {', '.join(missing)}")
    data["GEOID"] = data["GEOID"].astype("string")
    data["acs_vintage"] = pd.to_numeric(data["acs_vintage"], errors="raise").astype("int16")
    if not clean_fips(data["county_fips"]).eq(county_fips).all():
        raise ValueError(f"ACS files for {county_fips} contain a different county.")
    if data.duplicated(["acs_vintage", "GEOID"]).any():
        raise ValueError("ACS panel contains duplicate vintage/tract rows.")
    return data[["acs_vintage", "GEOID", *ACS_FEATURES]].copy()


def attach_acs(data: pd.DataFrame, acs: pd.DataFrame) -> pd.DataFrame:
    """Join a tract's newest ACS vintage definitely available before its sale year."""
    result = data.copy()
    # A 5-year ACS vintage is released after its end year.  The two-year lag is
    # safe for every sale date without assuming a common release day; 2016 sales
    # appropriately remain unmatched because this local panel starts in 2015.
    result["acs_vintage"] = (pd.to_datetime(result["sale_date"]).dt.year - 2).astype("Int16")
    result = result.merge(
        acs, left_on=["acs_vintage", "tax_assessor_geoid"], right_on=["acs_vintage", "GEOID"],
        how="left", validate="m:1", indicator="_acs_match",
    )
    result["acs_matched"] = result.pop("_acs_match").eq("both")
    return result.drop(columns=["GEOID"])


def score_predictions(actual: np.ndarray, predicted: np.ndarray, train_actual: np.ndarray) -> dict:
    """Use the repository's accuracy/equity metrics plus log-scale errors."""
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    train_actual = np.asarray(train_actual, dtype=float)
    actual_log = np.log(actual)
    predicted_log = np.log(predicted)
    metrics = _compute_extended_metrics(
        y_true_log=actual_log,
        y_pred_log=predicted_log,
        y_train_log=np.log(train_actual),
        ratio_mode="div",
    )
    residual_log = predicted_log - actual_log
    metrics.update(
        {
            "N": int(actual.size),
            "RMSE (log)": float(mean_squared_error(actual_log, predicted_log) ** 0.5),
            "MAE (log)": float(mean_absolute_error(actual_log, predicted_log)),
            # The quantity LGBCovPenalty[diff] actually penalizes, reported for every
            # model so the penalty's own objective can be read off the same table.
            "Cov(e,logprice)": float(
                np.mean((residual_log - residual_log.mean()) * (actual_log - actual_log.mean()))
            ),
        }
    )
    return metrics


def _selection_losses(frame: pd.DataFrame) -> pd.DataFrame:
    """Return the four predeclared losses used by every model selector.

    They are deliberately all losses: MAPE and COD are minimized directly, while
    PRD and PRB are minimized by distance to their equity ideals.  This keeps the
    Pareto relation invariant to units and avoids silently privileging accuracy
    over either vertical-equity diagnostic.
    """
    required = [name for name, _ in NASH_SELECTION_METRICS]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"Nash selection is missing required metrics: {missing}.")
    return pd.DataFrame(
        {
            "loss_mape": pd.to_numeric(frame["MAPE"], errors="coerce"),
            "loss_cod": pd.to_numeric(frame["COD"], errors="coerce"),
            "loss_prd_gap": (pd.to_numeric(frame["PRD"], errors="coerce") - 1.0).abs(),
            "loss_prb_gap": pd.to_numeric(frame["PRB"], errors="coerce").abs(),
        },
        index=frame.index,
    )


def _pareto_efficient_losses(losses: np.ndarray) -> np.ndarray:
    """Mark nondominated rows for a finite all-minimization loss matrix."""
    efficient = np.ones(len(losses), dtype=bool)
    for row, value in enumerate(losses):
        if not efficient[row]:
            continue
        dominates = np.all(losses <= value, axis=1) & np.any(losses < value, axis=1)
        if dominates.any():
            efficient[row] = False
    return efficient


def select_pareto_nash(
    candidates: pd.DataFrame,
    *,
    candidate_id: str = "candidate_id",
) -> tuple[pd.DataFrame, pd.Series, dict]:
    """Select the maximum-volume Pareto hyperrectangle on validation metrics.

    The requested Nash rule is operationalized as follows.  First remove points
    dominated on MAPE, COD, ``abs(PRD - 1)``, and ``abs(PRB)``.  For every
    remaining point, calculate the volume of its four-dimensional improvement
    hyperrectangle from a predeclared strictly-worse reference to the observed
    ideal.  Maximizing that volume is the usual Nash product rule, without a
    hidden accuracy tie-break.  A stable candidate ID resolves exact ties only.
    """
    if candidates.empty:
        raise ValueError("Cannot select from an empty candidate table.")
    if candidate_id not in candidates:
        raise ValueError(f"Nash selection requires a {candidate_id!r} column.")

    # Some stages intentionally refine a previously ranked candidate table
    # (for example, selecting the log-only baseline for the penalty arm after
    # the all-target baseline selection).  Strip only fields generated by this
    # selector so the operation is idempotent rather than creating duplicate
    # ``loss_*``/``nash_*`` labels on the second pass.
    generated_columns = {
        "loss_mape", "loss_cod", "loss_prd_gap", "loss_prb_gap",
        "selection_valid", "pareto_optimal", "nash_log_hypervolume",
        "nash_hypervolume", "selected_by_pareto_nash",
    }
    generated_columns.update(
        column for column in candidates.columns
        if column.startswith("nash_")
    )
    ranked = candidates.drop(columns=sorted(generated_columns & set(candidates.columns))).copy()
    losses = _selection_losses(ranked)
    loss_columns = list(losses.columns)
    ranked = pd.concat([ranked, losses], axis=1)
    valid = np.isfinite(losses.to_numpy(dtype=float)).all(axis=1)
    if not valid.any():
        raise ValueError("No candidate has finite MAPE, COD, PRD, and PRB for Nash selection.")

    ranked["selection_valid"] = valid
    ranked["pareto_optimal"] = False
    ranked["nash_log_hypervolume"] = np.nan
    ranked["nash_hypervolume"] = np.nan
    for column in loss_columns:
        ranked[f"nash_utility_{column.removeprefix('loss_')}"] = np.nan

    valid_index = ranked.index[valid]
    valid_losses = losses.loc[valid_index].to_numpy(dtype=float)
    pareto = _pareto_efficient_losses(valid_losses)
    ranked.loc[valid_index[pareto], "pareto_optimal"] = True

    # The requested reference point belongs to the Pareto frontier, not to a
    # dominated screening candidate.  This makes the maximum-volume choice
    # invariant when an objectively dominated trial is added to a search.
    pareto_losses = valid_losses[pareto]
    ideal = np.nanmin(pareto_losses, axis=0)
    observed_worst = np.nanmax(pareto_losses, axis=0)
    span = observed_worst - ideal
    # The reference lies strictly beyond the observed nadir.  The 5% margin is
    # fixed before selection and makes endpoint utility positive, so a useful
    # product remains defined when a candidate is best in one dimension.
    margin = np.maximum(0.05 * span, NASH_SELECTION_EPS)
    reference = observed_worst + margin
    denominator = reference - ideal
    utilities = (reference - valid_losses) / denominator
    utilities = np.clip(utilities, NASH_SELECTION_EPS, 1.0)
    log_volume = np.sum(np.log(utilities), axis=1)
    log_volume[~pareto] = -np.inf

    utility_columns = [f"nash_utility_{column.removeprefix('loss_')}" for column in loss_columns]
    ranked.loc[valid_index, utility_columns] = utilities
    finite_volume = np.isfinite(log_volume)
    ranked.loc[valid_index[finite_volume], "nash_log_hypervolume"] = log_volume[finite_volume]
    ranked.loc[valid_index[finite_volume], "nash_hypervolume"] = np.exp(log_volume[finite_volume])

    finalists = ranked.loc[ranked["pareto_optimal"]].copy()
    finalists["_candidate_id_sort"] = finalists[candidate_id].astype(str)
    finalists = finalists.sort_values(
        ["nash_log_hypervolume", "_candidate_id_sort"], ascending=[False, True], kind="mergesort",
    )
    selected = finalists.iloc[0].drop(labels="_candidate_id_sort")
    ranked["selected_by_pareto_nash"] = ranked.index.to_numpy() == selected.name
    metadata = {
        "name": "pareto_nash_hyperrectangle_v1",
        "candidate_id_column": candidate_id,
        "objectives": ["MAPE", "COD", "abs(PRD-1)", "abs(PRB)"],
        "loss_columns": loss_columns,
        "ideal_losses": {column: float(value) for column, value in zip(loss_columns, ideal)},
        "reference_losses": {column: float(value) for column, value in zip(loss_columns, reference)},
        "reference_margin_rule": "observed_nadir_plus_five_percent_range_or_epsilon",
        "reference_population": "pareto_frontier",
        "pareto_candidates": int(pareto.sum()),
        "valid_candidates": int(valid.sum()),
        "tie_breaker": f"lexicographic {candidate_id} only after equal hypervolume",
        "selected_candidate_id": str(selected[candidate_id]),
    }
    return ranked, ranked.loc[selected.name].copy(), metadata


def feature_frame(
    data: pd.DataFrame, train_rows: int, include_repeat_sale: bool, include_assessment_values: bool,
) -> tuple[pd.DataFrame, list[str]]:
    """Create predictors for an explicit CCAO-like or status-quo feature set."""
    features = pd.DataFrame(index=data.index)
    for column in NUMERIC_FEATURES:
        if column in data and (include_assessment_values or column not in ASSESSMENT_VALUE_COLUMNS):
            features[column] = pd.to_numeric(data[column], errors="coerce")
    numeric = lambda column: pd.to_numeric(data[column], errors="coerce") if column in data else pd.Series(np.nan, index=data.index)
    sale_date = pd.to_datetime(data["sale_date"])
    features["sale_year"] = sale_date.dt.year
    features["sale_month"] = sale_date.dt.month
    features["sale_quarter"] = sale_date.dt.quarter
    features["sale_day_of_year"] = sale_date.dt.dayofyear
    tax_year = pd.to_numeric(data["TAXYEARASSESSED"], errors="coerce")
    features["tax_year_assessed"] = tax_year
    for column in ("AREABUILDING", "AREALOTSF", "TAXMARKETVALUETOTAL", "TAXASSESSEDVALUETOTAL"):
        if column in features:
            features[f"log_{column.lower()}"] = np.log1p(features[column].clip(lower=0))
    if "YEARBUILT" in features:
        features["property_age"] = (features["sale_year"] - features["YEARBUILT"]).where(lambda x: x >= 0)
    if "YEARBUILTEFFECTIVE" in features:
        features["effective_age"] = (features["sale_year"] - features["YEARBUILTEFFECTIVE"]).where(lambda x: x >= 0)
    building_area = numeric("AREABUILDING")
    land_area = numeric("AREALOTSF")
    rooms = numeric("ROOMSCOUNT")
    bedrooms = numeric("BEDROOMSCOUNT")
    units = numeric("UNITSCOUNT")
    basement_area = numeric("ROOMSBASEMENTAREA")
    basement_finished = numeric("ROOMSBASEMENTAREAFINISHED")
    features["building_to_land_area_ratio"] = safe_divide(building_area, land_area)
    features["building_area_per_unit"] = safe_divide(building_area, units)
    features["land_area_per_unit"] = safe_divide(land_area, units)
    features["area_per_room"] = safe_divide(building_area, rooms)
    features["area_per_bedroom"] = safe_divide(building_area, bedrooms)
    features["finished_basement_share"] = safe_divide(basement_finished, basement_area)
    features["has_basement"] = basement_area.gt(0).astype(float)
    for source, output in [
        ("ROOMSBASEMENTAREAFINISHED", "has_finished_basement"),
        ("PARKINGGARAGEAREA", "has_garage_area"),
        ("PARKINGSPACECOUNT", "has_parking_spaces"),
        ("FIREPLACECOUNT", "has_fireplace_count"),
        ("PORCHAREA", "has_porch_area"),
        ("PATIOAREA", "has_patio_area"),
        ("DECKAREA", "has_deck_area"),
        ("POOLAREA", "has_pool_area"),
    ]:
        features[output] = numeric(source).gt(0).astype(float)
    if "YEARBUILT" in data and "YEARBUILTEFFECTIVE" in data:
        features["effective_year_gap"] = (numeric("YEARBUILTEFFECTIVE") - numeric("YEARBUILT")).where(lambda x: x >= 0)
    if include_assessment_values:
        market_total = numeric("TAXMARKETVALUETOTAL")
        assessed_total = numeric("TAXASSESSEDVALUETOTAL")
        features["market_total_per_building_sf"] = safe_divide(market_total, building_area)
        features["assessed_total_per_building_sf"] = safe_divide(assessed_total, building_area)
        features["assessed_to_market_ratio"] = safe_divide(assessed_total, market_total)
        features["tax_bill_to_market_value"] = safe_divide(numeric("TAXBILLEDAMOUNT"), market_total)
        features["market_land_value_share"] = safe_divide(numeric("TAXMARKETVALUELAND"), market_total)
        features["assessed_land_value_share"] = safe_divide(numeric("TAXASSESSEDVALUELAND"), assessed_total)
    occupied = numeric("B25003_002E") + numeric("B25003_003E")
    features["acs_log_population"] = np.log1p(numeric("B01003_001E").clip(lower=0))
    features["acs_log_median_income"] = np.log1p(numeric("B19013_001E").clip(lower=0))
    features["acs_log_median_rent"] = np.log1p(numeric("B25064_001E").clip(lower=0))
    features["acs_owner_occupied_share"] = safe_divide(numeric("B25003_002E"), occupied)
    features["acs_renter_occupied_share"] = safe_divide(numeric("B25003_003E"), occupied)
    features["acs_housing_units_per_capita"] = safe_divide(numeric("B25001_001E"), numeric("B01003_001E"))
    if include_repeat_sale:
        for column in [
            "recorder_prior_sale_amount", "recorder_log_prior_sale_price", "recorder_prior_sale_age_years",
            "recorder_prior_sale_count_all", "recorder_prior_sale_count_3yr", "recorder_prior_sale_count_5yr",
        ]:
            if column in data:
                features[column] = pd.to_numeric(data[column], errors="coerce")
    if include_repeat_sale and {"ASSESSORLASTSALEDATE", "ASSESSORLASTSALEAMOUNT"}.issubset(data.columns):
        prior_date = pd.to_datetime(data["ASSESSORLASTSALEDATE"], errors="coerce")
        prior_price = pd.to_numeric(data["ASSESSORLASTSALEAMOUNT"], errors="coerce").where(prior_date < sale_date)
        features["assessor_log_prior_sale_price"] = np.log(prior_price.where(prior_price > 0))
        features["assessor_prior_sale_age_years"] = ((sale_date - prior_date).dt.days / 365.25).where(prior_date < sale_date)
    categorical = []
    for column in CATEGORICAL_FEATURES:
        if column not in data:
            continue
        values = data[column].astype("string").fillna("__missing__")
        known = pd.Index(values.iloc[:train_rows].unique())
        if known.size < 2:
            continue
        values = values.where(values.isin(known), "__unknown__")
        features[column] = pd.Categorical(values, categories=known.union(pd.Index(["__unknown__"]), sort=False))
        categorical.append(column)
    train_features = features.iloc[:train_rows]
    keep = [column for column in features if train_features[column].nunique(dropna=True) > 1]
    return features[keep], [column for column in categorical if column in keep]


def _neighbor_raw_frame(data: pd.DataFrame, train_rows: int) -> tuple[pd.DataFrame, list[str]]:
    """Return raw comparable-sales inputs with a stable, unique row identity."""
    raw = pd.DataFrame(index=pd.RangeIndex(len(data)))
    numeric = lambda column: (
        pd.to_numeric(data[column], errors="coerce")
        if column in data else pd.Series(np.nan, index=data.index, dtype=float)
    )
    # The Tax Assessor extract is the audited location crosswalk.  History
    # coordinates remain a fallback for rows without a usable crosswalk match.
    raw["neighbor_latitude"] = numeric("tax_assessor_latitude").combine_first(
        numeric("LATITUDE")
    ).to_numpy()
    raw["neighbor_longitude"] = numeric("tax_assessor_longitude").combine_first(
        numeric("LONGITUDE")
    ).to_numpy()
    raw["neighbor_sale_date"] = pd.to_datetime(data["sale_date"], errors="coerce").to_numpy()
    numeric_cols = [
        column for column in NEIGHBOR_FEATURE_COLUMNS
        if column in data and pd.to_numeric(data[column].iloc[:train_rows], errors="coerce").notna().any()
    ]
    for column in numeric_cols:
        raw[column] = pd.to_numeric(data[column], errors="coerce").to_numpy()
    return raw, numeric_cols


def _neighbor_transformer(
    variant: NeighborVariant,
    spec: NeighborSearchSpec,
    numeric_cols: list[str],
) -> SpatioTemporalKernelTargetNeighbors:
    """Build the pinned comparable transformer for one bounded search point."""
    return SpatioTemporalKernelTargetNeighbors(
        k=int(spec.k),
        lat_col="neighbor_latitude",
        lon_col="neighbor_longitude",
        date_col="neighbor_sale_date",
        kernel="gaussian",
        bandwidth="adaptive",
        bandwidth_scale=1.0,
        geo_weight=float(variant.geo_weight),
        target_transform="log",
        include_aggregate=True,
        include_diagnostics=True,
        categorical_filter_roots=None,
        use_feature_distance=bool(variant.use_feature_distance),
        numeric_feature_cols=numeric_cols if variant.use_feature_distance else None,
        feature_scaler="standard",
        feature_alpha=float(variant.feature_weight),
        feature_bandwidth=1.0,
        # Finite caps activate the transformer's exact radius-then-time query
        # route, so this legacy multiplier cannot truncate a composite candidate.
        candidate_multiplier=10,
        batch_query_size=256,
        full_pool_batch_size=32,
        n_jobs=-1,
        use_time_trend=bool(variant.use_time_trend),
        time_trend="linear",
        time_trend_fit_mode="causal_prior",
        use_time_decay=bool(variant.use_time_decay),
        time_weight=float(spec.time_weight) if variant.use_time_decay else 0.0,
        time_bandwidth_days=365.25,
        neighbor_time_rule="past",
        max_distance_km=float(spec.max_distance_km),
        max_time_distance_days=float(spec.max_time_distance_days),
        insufficient_neighbors="nan",
        exclude_self=True,
        feature_prefix="neighbor",
    )


def neighbor_feature_frame(
    data: pd.DataFrame,
    target_log: pd.Series,
    *,
    train_rows: int,
    end_rows: int,
    variant: NeighborVariant,
    spec: NeighborSearchSpec,
) -> tuple[pd.DataFrame, dict]:
    """Build strictly prior-sale comparable features through ``end_rows``.

    Training rows exclude themselves and use only earlier sales.  Evaluation rows
    use only fitted sales that predate their valuation date.  For time-adjusted
    variants, the target-derived linear trend is also fit from only sales strictly
    before each query date.
    """
    raw, numeric_cols = _neighbor_raw_frame(data.iloc[:end_rows], train_rows)
    if variant.use_feature_distance and not numeric_cols:
        raise ValueError(f"{variant.label} has no usable structural similarity columns.")
    coordinate_columns = ["neighbor_latitude", "neighbor_longitude", "neighbor_sale_date"]
    valid = raw[coordinate_columns].notna().all(axis=1)
    valid_train = raw.iloc[:train_rows].loc[valid.iloc[:train_rows]]
    valid_eval = raw.iloc[train_rows:end_rows].loc[valid.iloc[train_rows:end_rows]]
    if valid_train.empty:
        raise ValueError(f"{variant.label} has no train rows with usable coordinates and sale dates.")

    transformer = _neighbor_transformer(variant, spec, numeric_cols)
    # The shared transformer owns the log transform so its comparable target and
    # train-only time trend are both on the intended log-price scale.
    train_target = pd.Series(
        np.exp(np.asarray(target_log.iloc[:train_rows], dtype=float)), index=raw.index[:train_rows],
    )
    transformer.fit(valid_train, train_target.loc[valid_train.index])
    transformed_train = transformer.transform(valid_train)
    transformed_eval = transformer.transform(valid_eval) if not valid_eval.empty else pd.DataFrame(index=valid_eval.index)
    transformed = pd.concat([transformed_train, transformed_eval]).sort_index()

    weighted_columns = [
        f"neighbor_neighbor_{number}_kernel_adjusted_target" for number in range(1, int(spec.k) + 1)
    ]
    weight_columns = [f"neighbor_neighbor_{number}_weight" for number in range(1, int(spec.k) + 1)]
    weights = transformed[weight_columns].to_numpy(dtype=float)
    weighted_targets = transformed[weighted_columns].to_numpy(dtype=float)
    adjusted_targets = np.divide(
        weighted_targets, weights, out=np.full_like(weighted_targets, np.nan), where=weights > 0.0,
    )
    local_mean = transformed["neighbor_local_kernel_adjusted_target_mean"].to_numpy(dtype=float)
    dispersion = np.sqrt(np.nansum(weights * (adjusted_targets - local_mean[:, None]) ** 2, axis=1))
    dispersion[~np.isfinite(local_mean)] = np.nan
    effective_n = np.divide(
        1.0, np.sum(weights ** 2, axis=1), out=np.full(len(transformed), np.nan), where=np.sum(weights ** 2, axis=1) > 0.0,
    )

    prefix = f"neighbor_{variant.key}"
    features = pd.DataFrame(np.nan, index=raw.index, columns=[
        f"{prefix}_log_price", f"{prefix}_log_price_dispersion", f"{prefix}_n", f"{prefix}_effective_n",
    ])
    features.loc[transformed.index, f"{prefix}_log_price"] = local_mean
    features.loc[transformed.index, f"{prefix}_log_price_dispersion"] = dispersion
    features.loc[transformed.index, f"{prefix}_n"] = transformed["neighbor_n_valid_neighbors"].to_numpy(dtype=float)
    features.loc[transformed.index, f"{prefix}_effective_n"] = effective_n
    features.index = data.index[:end_rows]
    eligible_pool_column = "neighbor_eligible_pool_size_after_caps"
    eligible_pools = (
        pd.to_numeric(transformed[eligible_pool_column], errors="coerce")
        if eligible_pool_column in transformed else pd.Series(np.nan, index=transformed.index)
    )
    eligible_quantiles = {
        f"p{int(q * 100):02d}": float(eligible_pools.quantile(q))
        for q in (0.05, 0.50, 0.95)
        if eligible_pools.notna().any()
    }
    train_full_coverage = float(
        (pd.to_numeric(transformed.loc[transformed.index < train_rows, "neighbor_n_valid_neighbors"], errors="coerce") >= int(spec.k)).mean()
    ) if len(transformed) else np.nan
    evaluation_full_coverage = float(
        (pd.to_numeric(transformed.loc[transformed.index >= train_rows, "neighbor_n_valid_neighbors"], errors="coerce") >= int(spec.k)).mean()
    ) if len(transformed) and (transformed.index >= train_rows).any() else np.nan
    return features, {
        **spec.as_dict(),
        "geo_weight": float(variant.geo_weight),
        "time_weight": float(spec.time_weight) if variant.use_time_decay else 0.0,
        "feature_weight": float(spec.feature_weight) if variant.use_feature_distance else 0.0,
        "time_trend": bool(variant.use_time_trend),
        "time_trend_fit_mode": (
            str(transformer.time_trend_fit_mode) if variant.use_time_trend else "not_used"
        ),
        "transformer_parameters": transformer.get_params(deep=False),
        "time_decay": bool(variant.use_time_decay),
        "structural_similarity_columns": numeric_cols if variant.use_feature_distance else [],
        "candidate_retrieval": getattr(transformer, "candidate_retrieval_", "not_recorded"),
        "candidate_multiplier_used": bool(getattr(transformer, "candidate_multiplier_used_", True)),
        "train_coordinate_coverage": float(valid.iloc[:train_rows].mean()),
        "evaluation_coordinate_coverage": float(valid.iloc[train_rows:end_rows].mean()),
        "train_comparable_coverage": float(features.iloc[:train_rows, 0].notna().mean()),
        "evaluation_comparable_coverage": float(features.iloc[train_rows:end_rows, 0].notna().mean()),
        "train_comparable_coverage_after_caps_full_k": train_full_coverage,
        "evaluation_comparable_coverage_after_caps_full_k": evaluation_full_coverage,
        "eligible_pool_size_after_caps_quantiles": eligible_quantiles,
    }


def feature_coverage(features: pd.DataFrame, categorical: list[str]) -> dict[str, dict]:
    """Summarize which engineered feature families are actually usable."""
    families = {
        "acs": lambda column: column.startswith("acs_") or (column.startswith("B") and column[1:2].isdigit()),
        "assessor_value": lambda column: "TAX" in column or "market_" in column or "assessed_" in column,
        "derived_structure": lambda column: column.startswith(("building_", "land_", "area_", "finished_", "has_", "effective_")),
        "location": lambda column: column in {"LATITUDE", "LONGITUDE", "tax_assessor_latitude", "tax_assessor_longitude", "tax_assessor_geoid", "CENSUSTRACT"},
        "recorder_history": lambda column: column.startswith("recorder_"),
        "sale_time": lambda column: column.startswith("sale_") or column.startswith("tax_year_"),
    }
    report = {}
    for family, selector in families.items():
        columns = [column for column in features.columns if selector(column)]
        report[family] = {
            "n_features": int(len(columns)),
            "features": columns,
            "mean_non_null_rate": float(features[columns].notna().mean().mean()) if columns else None,
        }
    report["categorical"] = {
        "n_features": int(len(categorical)),
        "features": categorical,
        "mean_non_null_rate": float(features[categorical].notna().mean().mean()) if categorical else None,
    }
    return report


def load_lgbm_configs(path: Path, keys: str, threads: int | None = None) -> dict[str, dict]:
    """Load the CCAO-selected LightGBM configurations without modifying them.

    ``threads`` overrides only ``n_jobs``, which sizes the fit to the machine
    rather than describing the model, leaving every tuned hyperparameter intact.
    """
    with path.open(encoding="utf-8") as file:
        configured = yaml.safe_load(file)["lgbm_baselines"]
    requested = [key.strip() for key in keys.split(",") if key.strip()]
    missing = [key for key in requested if key not in configured]
    if missing:
        raise ValueError(f"Unknown LightGBM configuration(s): {', '.join(missing)}")
    configs = {key: dict(configured[key]["lgbm_params"]) for key in requested}
    if threads is not None:
        for params in configs.values():
            params["n_jobs"] = threads
    return configs


def bootstrap_scores(test: pd.DataFrame, prediction: np.ndarray, train_price: np.ndarray, n_bootstrap: int, block_freq: str, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarize test metrics over CCAO-style resampled calendar blocks."""
    started_at = perf_counter()
    indices = _build_time_block_bootstrap_indices(
        val_dates=pd.to_datetime(test["sale_date"]), n_bootstrap=n_bootstrap, block_freq=block_freq, rng_seed=seed,
    )
    rows = []
    update_every = max(1, n_bootstrap // 20)
    for number, idx in enumerate(indices, start=1):
        rows.append(score_predictions(test.sale_price.to_numpy()[idx], prediction[idx], train_price))
        if number % update_every == 0 or number == n_bootstrap:
            report_progress("Bootstrap scores", number, n_bootstrap, started_at)
    draws = pd.DataFrame(rows)
    numeric = draws.select_dtypes(include="number")
    summary = pd.DataFrame(
        {
            "metric": numeric.columns,
            "mean": numeric.mean().to_numpy(),
            "std": numeric.std(ddof=1).to_numpy(),
            "ci_2_5": numeric.quantile(0.025).to_numpy(),
            "ci_97_5": numeric.quantile(0.975).to_numpy(),
        }
    )
    return draws, summary


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    """Deterministic weighted quantile for local-equity group summaries."""
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0.0)
    if not valid.any():
        return np.nan
    values, weights = values[valid], weights[valid]
    order = np.argsort(values, kind="mergesort")
    values, weights = values[order], weights[order]
    cutoff = float(quantile) * float(weights.sum())
    return float(values[min(np.searchsorted(np.cumsum(weights), cutoff, side="left"), len(values) - 1)])


def _moran_i_knn(
    latitude: np.ndarray,
    longitude: np.ndarray,
    residual: np.ndarray,
    *,
    k: int,
) -> tuple[float, int]:
    """Directed binary-kNN Moran's I without a heavyweight spatial dependency."""
    latitude = np.asarray(latitude, dtype=float)
    longitude = np.asarray(longitude, dtype=float)
    residual = np.asarray(residual, dtype=float)
    valid = np.isfinite(latitude) & np.isfinite(longitude) & np.isfinite(residual)
    if valid.sum() < 3:
        return np.nan, int(valid.sum())
    coords = np.radians(np.column_stack([latitude[valid], longitude[valid]]))
    z = residual[valid] - float(np.mean(residual[valid]))
    if not np.isfinite(z @ z) or np.isclose(z @ z, 0.0):
        return np.nan, int(len(z))
    n_neighbors = min(int(k), len(z) - 1)
    neighbors = NearestNeighbors(metric="haversine", algorithm="ball_tree").fit(coords)
    indices = neighbors.kneighbors(coords, n_neighbors=n_neighbors + 1, return_distance=False)
    # sklearn returns the query row itself first; use the remaining deterministic
    # neighbours to form a directed binary graph with S0 = n * k.
    neighbor_indices = indices[:, 1:]
    numerator = float(np.sum(z * np.sum(z[neighbor_indices], axis=1)))
    s0 = float(len(z) * n_neighbors)
    return float((len(z) / s0) * numerator / float(z @ z)), int(len(z))


def local_equity_diagnostics(
    data: pd.DataFrame,
    predictions: pd.DataFrame,
    *,
    split: int,
    model_keys: Sequence[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Evaluate tract-like local ratio diagnostics and residual spatial dependence."""
    test = data.iloc[split:].reset_index(drop=True)
    if "tax_assessor_geoid" not in test:
        raise ValueError("Local-equity evaluation requires tax_assessor_geoid in the matched sample.")
    group = test["tax_assessor_geoid"].astype("string")
    numeric_column = lambda column: (
        pd.to_numeric(test[column], errors="coerce") if column in test
        else pd.Series(np.nan, index=test.index, dtype=float)
    )
    latitude = numeric_column("tax_assessor_latitude").combine_first(numeric_column("LATITUDE"))
    longitude = numeric_column("tax_assessor_longitude").combine_first(numeric_column("LONGITUDE"))
    actual = test.sale_price.to_numpy(dtype=float)
    group_rows, summary_rows = [], []
    for model_key in model_keys:
        column = f"predicted_sale_price__{model_key}"
        if column not in predictions:
            raise ValueError(f"Local-equity evaluation lacks prediction column {column}.")
        predicted = predictions[column].iloc[split:].to_numpy(dtype=float)
        valid = np.isfinite(actual) & np.isfinite(predicted) & (actual > 0.0) & (predicted > 0.0)
        ratio = np.divide(predicted, actual, out=np.full(len(actual), np.nan), where=valid)
        eligible_group_rows = []
        group_frame = pd.DataFrame({"group": group, "ratio": ratio}).loc[valid & group.notna().to_numpy()]
        for group_key, frame in group_frame.groupby("group", sort=True):
            values = frame["ratio"].to_numpy(dtype=float)
            n = int(len(values))
            if n < LOCAL_EQUITY_MIN_GROUP_N:
                continue
            median = float(np.median(values))
            cod = float(100.0 * np.mean(np.abs(values - median)) / median) if median > 0.0 else np.nan
            row = {
                "model_key": model_key,
                "split": "test",
                "geographic_group": str(group_key),
                "n": n,
                "median_ratio": median,
                "local_median_ratio_deviation": abs(median - 1.0),
                "local_cod": cod,
            }
            group_rows.append(row)
            eligible_group_rows.append(row)
        eligible = pd.DataFrame(eligible_group_rows)
        residual = np.log(predicted[valid]) - np.log(actual[valid])
        moran_i, moran_n = _moran_i_knn(
            latitude.to_numpy()[valid], longitude.to_numpy()[valid], residual, k=LOCAL_EQUITY_MORAN_K,
        )
        if eligible.empty:
            weighted_median_deviation = weighted_p90_deviation = weighted_median_cod = weighted_p90_cod = np.nan
            covered_sales = 0
        else:
            weights = eligible["n"].to_numpy(dtype=float)
            weighted_median_deviation = _weighted_quantile(
                eligible["local_median_ratio_deviation"].to_numpy(), weights, 0.50,
            )
            weighted_p90_deviation = _weighted_quantile(
                eligible["local_median_ratio_deviation"].to_numpy(), weights, 0.90,
            )
            weighted_median_cod = _weighted_quantile(eligible["local_cod"].to_numpy(), weights, 0.50)
            weighted_p90_cod = _weighted_quantile(eligible["local_cod"].to_numpy(), weights, 0.90)
            covered_sales = int(eligible["n"].sum())
        summary_rows.append({
            "model_key": model_key,
            "split": "test",
            "group_column": "tax_assessor_geoid",
            "minimum_group_n": LOCAL_EQUITY_MIN_GROUP_N,
            "n_valid_ratio_sales": int(valid.sum()),
            "n_eligible_groups": int(len(eligible_group_rows)),
            "eligible_group_sale_coverage": float(covered_sales / valid.sum()) if valid.any() else np.nan,
            "weighted_median_local_median_ratio_deviation": weighted_median_deviation,
            "weighted_p90_local_median_ratio_deviation": weighted_p90_deviation,
            "weighted_median_local_cod": weighted_median_cod,
            "weighted_p90_local_cod": weighted_p90_cod,
            "moran_i_log_residual_knn": moran_i,
            "moran_i_n": moran_n,
            "moran_i_k": min(LOCAL_EQUITY_MORAN_K, max(moran_n - 1, 0)),
        })
    protocol = {
        "split": "held_out_test",
        "group_column": "tax_assessor_geoid",
        "minimum_group_n": LOCAL_EQUITY_MIN_GROUP_N,
        "local_ratio_metrics": ["median_ratio", "abs(median_ratio-1)", "COD"],
        "aggregation": "sale-count weighted median and P90 across eligible geographic groups",
        "spatial_dependence": "directed binary haversine kNN Moran's I of log residuals",
        "moran_k": LOCAL_EQUITY_MORAN_K,
    }
    # Preserve a readable, schema-stable CSV even when no geographic group meets
    # the fixed minimum size (a legitimate outcome in a sparse county).  Without
    # explicit columns, pandas writes an empty file and the dashboard cannot
    # distinguish that outcome from a missing artifact.
    group_columns = [
        "model_key", "split", "geographic_group", "n", "median_ratio",
        "local_median_ratio_deviation", "local_cod",
    ]
    summary_columns = [
        "model_key", "split", "group_column", "minimum_group_n", "n_valid_ratio_sales",
        "n_eligible_groups", "eligible_group_sale_coverage",
        "weighted_median_local_median_ratio_deviation",
        "weighted_p90_local_median_ratio_deviation", "weighted_median_local_cod",
        "weighted_p90_local_cod", "moran_i_log_residual_knn", "moran_i_n", "moran_i_k",
    ]
    return (
        pd.DataFrame(group_rows, columns=group_columns),
        pd.DataFrame(summary_rows, columns=summary_columns),
        protocol,
    )


def parse_shrinkage_targets(raw: str | None) -> tuple[float, ...]:
    """Parse requested covariance reductions, each strictly inside (0, 1)."""
    if raw is None or not raw.strip():
        return COVARIANCE_SHRINKAGE_TARGETS
    values = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        value = float(token)
        if not 0.0 < value < 1.0:
            raise ValueError(f"Covariance shrinkage target must lie in (0, 1); got {value}.")
        values.append(value)
    if not values:
        raise ValueError("--cov-shrinkage-targets was supplied without a usable value.")
    return tuple(sorted(set(values)))


def plan_rho_grid(
    y_log: np.ndarray,
    f0_log: np.ndarray,
    *,
    county_fips: str,
    config_key: str,
    shrinkage_targets: Sequence[float],
    include_anchors: bool = True,
) -> tuple[pd.DataFrame, dict, dict[str, pd.DataFrame]]:
    """Turn baseline training predictions into a theory-calibrated rho grid.

    Rank-one theory gives ``q(rho) = 1 / (1 + rho * A / 2)`` for the remaining
    covariance fraction, with ``A = Var(f0)``.  Inverting it makes each requested
    reduction an explicit, county-specific rho, and reports the accuracy price the
    same theory predicts for it.
    """
    summary, shrink_df, prd_df, budget_df = compute_theory_for_predictions(
        y_log=np.asarray(y_log, dtype=float),
        f0_log=np.asarray(f0_log, dtype=float),
        split_label="train",
        data_source=county_fips,
        config_key=config_key,
        assessment_year=int(pd.Timestamp.now().year),
        prd_targets=PRD_TARGETS,
        shrinkage_q_values=[1.0 - t for t in shrinkage_targets],
        accuracy_budgets=ACCURACY_BUDGETS,
    )
    plan = (
        shrink_df.rename(columns={"rho_theory": "rho"})
        .assign(
            rho_source="requested_covariance_reduction",
            requested_covariance_reduction=lambda frame: frame["covariance_reduction"],
        )
        [[
            "rho", "rho_source", "requested_covariance_reduction",
            "q_remaining_covariance", "delta_mse_log_theory", "delta_mse_log_frac_of_baseline",
        ]]
    )
    if include_anchors:
        anchors = []
        for field, label in THEORY_ANCHORS.items():
            rho = float(summary.get(field, np.nan))
            if not np.isfinite(rho) or rho <= 0.0:
                continue
            q = float(1.0 / (1.0 + rho * summary["A_var_f0_log"] / 2.0))
            delta = (summary["C0_cov_log_residual_logprice"] ** 2 / summary["A_var_f0_log"]) * (1.0 - q) ** 2
            anchors.append({
                "rho": rho,
                "rho_source": label,
                "requested_covariance_reduction": 1.0 - q,
                "q_remaining_covariance": q,
                "delta_mse_log_theory": delta,
                "delta_mse_log_frac_of_baseline": delta / summary["B_mse_log"],
            })
        plan = pd.concat([plan, pd.DataFrame(anchors)], ignore_index=True)
    plan = plan.loc[np.isfinite(plan["rho"]) & plan["rho"].gt(0.0)].copy()
    # Anchors routinely land on a grid rho; keep the first (requested-reduction)
    # label so one fit is never repeated under two names.
    plan["rho"] = plan["rho"].astype(float).round(6)
    plan = plan.drop_duplicates("rho", keep="first").sort_values("rho").reset_index(drop=True)
    return plan, summary, {"shrinkage": shrink_df, "prd": prd_df, "accuracy_budget": budget_df}


def fit_penalized_predictions(
    features: pd.DataFrame,
    target_log: pd.Series,
    *,
    split: int,
    rho: float,
    params: dict,
    early_stopping_rounds: int | None,
) -> np.ndarray:
    """Fit LGBCovPenalty[diff] on the training block and predict every row.

    The estimator is constructed exactly as in ``quick_test_models.py`` so results
    stay comparable with the existing rho-sweep experiments.  Categorical columns
    are already pandas ``category`` dtype, which LightGBM picks up on its own.
    """
    model = LGBCovPenalty(
        rho=float(rho),
        ratio_mode="diff",
        early_stopping_rounds=early_stopping_rounds,
        zero_grad_tol=1e-12,
        lgbm_params=params,
        verbose=False,
    )
    model.fit(features.iloc[:split], target_log.iloc[:split])
    return np.exp(model.predict(features))


def _shrinkage_key(prefix: str, requested_reduction: float) -> str:
    """Stable model identifier based on the transferable shrinkage target, not rho."""
    text = f"{float(requested_reduction):.6f}".rstrip("0").rstrip(".")
    return f"{prefix}_shrink_{text}".replace(".", "p").replace("-", "m")


def select_penalty_on_validation(
    features: pd.DataFrame,
    target_log: pd.Series,
    data: pd.DataFrame,
    *,
    development_split: int,
    validation_end: int,
    params: dict,
    county_fips: str,
    config_key: str,
    shrinkage_targets: Sequence[float],
    early_stopping_rounds: int | None,
) -> dict:
    """Select a theory-calibrated penalty target on a chronological validation block.

    A requested covariance reduction is the portable hyperparameter.  Its numeric
    rho is calibrated on the development prefix here, then recalibrated after the
    selected target is refit on the full training block.  Test outcomes never enter
    this function.
    """
    baseline = LGBMRegressor(**params)
    baseline.fit(features.iloc[:development_split], target_log.iloc[:development_split])
    baseline_prediction = np.exp(baseline.predict(features.iloc[:validation_end]))
    baseline_validation_metrics = score_predictions(
        data.sale_price.iloc[development_split:validation_end], baseline_prediction[development_split:],
        data.sale_price.iloc[:development_split],
    )
    plan, theory, theory_tables = plan_rho_grid(
        target_log.to_numpy()[:development_split],
        np.log(baseline_prediction[:development_split]),
        county_fips=county_fips,
        config_key=config_key,
        shrinkage_targets=shrinkage_targets,
        include_anchors=False,
    )
    candidates = []
    train_price = data.sale_price.iloc[:development_split].to_numpy()
    for row in plan.itertuples(index=False):
        prediction = fit_penalized_predictions(
            features.iloc[:validation_end], target_log.iloc[:validation_end],
            split=development_split, rho=float(row.rho), params=params,
            early_stopping_rounds=early_stopping_rounds,
        )
        metrics = score_predictions(
            data.sale_price.iloc[development_split:validation_end], prediction[development_split:], train_price,
        )
        candidates.append({
            "candidate_id": _shrinkage_key("shrink", float(row.requested_covariance_reduction)),
            "requested_covariance_reduction": float(row.requested_covariance_reduction),
            "development_rho": float(row.rho),
            "validation_metrics": metrics,
        })
    selection_input = pd.DataFrame([
        {"candidate_id": row["candidate_id"], **row["validation_metrics"]}
        for row in candidates
    ])
    selected_frame, selected_row, selection_metadata = select_pareto_nash(selection_input)
    selection_by_id = selected_frame.set_index("candidate_id")
    for candidate in candidates:
        selection = selection_by_id.loc[candidate["candidate_id"]]
        candidate["selection"] = {
            name: (
                bool(selection[name]) if name in {"selection_valid", "pareto_optimal", "selected_by_pareto_nash"}
                else float(selection[name])
            )
            for name in (
                "selection_valid", "pareto_optimal", "selected_by_pareto_nash",
                "nash_log_hypervolume", "nash_hypervolume",
            )
        }
    selected = next(row for row in candidates if row["candidate_id"] == selected_row["candidate_id"])
    return {
        "fit_split": "development_prefix",
        "selection_split": "chronological_validation",
        "validation_rows": int(validation_end - development_split),
        "theory": theory,
        "candidates": candidates,
        "selection": selection_metadata,
        "selected_requested_covariance_reduction": float(selected["requested_covariance_reduction"]),
        "selected_development_rho": float(selected["development_rho"]),
        "selected_validation_metrics": selected["validation_metrics"],
        "baseline_validation_metrics": baseline_validation_metrics,
    }


ROLLING_RECALIBRATION_FOLDS = 3
RECALIBRATION_STRENGTHS = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)


def fit_rolling_recalibration(
    data: pd.DataFrame,
    target_log: pd.Series,
    *,
    split: int,
    fold_rows: int,
    include_repeat_sale: bool,
    include_assessment_values: bool,
    params: dict,
    final_train_prediction_log: np.ndarray,
    started_at: float,
) -> dict:
    """Fit a first-degree map on rolling-origin predictions, then re-anchor it.

    Each map sees only predictions made by a model trained before its calibration
    fold.  Within each fold, prediction and outcome levels are removed before the
    scaling coefficient is fitted.  The deployed map is then anchored to the final
    baseline fit's training prediction centre and outcome level, avoiding transport
    of a validation model's absolute intercept.
    """
    first_origin = split - ROLLING_RECALIBRATION_FOLDS * fold_rows
    if first_origin < 1 or fold_rows < 1:
        raise ValueError("The training block is too small for three rolling recalibration folds.")

    oof_prediction, oof_actual, oof_level, oof_x = [], [], [], []
    folds = []
    for fold in range(ROLLING_RECALIBRATION_FOLDS):
        origin = first_origin + fold * fold_rows
        end = split if fold == ROLLING_RECALIBRATION_FOLDS - 1 else origin + fold_rows
        features, categorical = feature_frame(
            data, origin, include_repeat_sale, include_assessment_values,
        )
        model = LGBMRegressor(**params)
        model.fit(features.iloc[:origin], target_log.iloc[:origin], categorical_feature=categorical)
        train_prediction = model.predict(features.iloc[:origin])
        validation_prediction = model.predict(features.iloc[origin:end])
        centre = float(train_prediction.mean())
        level = float(target_log.iloc[:origin].mean())
        x = validation_prediction - centre
        oof_prediction.append(validation_prediction)
        oof_actual.append(target_log.iloc[origin:end].to_numpy())
        oof_level.append(np.full(end - origin, level))
        oof_x.append(x)
        folds.append({
            "origin_rows": int(origin),
            "validation_rows": int(end - origin),
            "validation_date_range": [
                str(pd.to_datetime(data.sale_date.iloc[origin]).date()),
                str(pd.to_datetime(data.sale_date.iloc[end - 1]).date()),
            ],
        })
        report_progress("Rolling recalibration fits", fold + 1, ROLLING_RECALIBRATION_FOLDS, started_at)

    prediction = np.concatenate(oof_prediction)
    actual_log = np.concatenate(oof_actual)
    level = np.concatenate(oof_level)
    x = np.concatenate(oof_x)
    calibration_train_actual = data.sale_price.iloc[:first_origin].to_numpy()
    actual = np.exp(actual_log)
    raw_prediction = np.exp(prediction)
    level_only_prediction = np.exp(level + x)
    level_only_metrics = score_predictions(actual, level_only_prediction, calibration_train_actual)

    final_train_prediction_log = np.asarray(final_train_prediction_log, dtype=float)
    final_centre = float(final_train_prediction_log.mean())
    final_level = float(target_log.iloc[:split].mean())
    target = actual_log - level
    denominator = float(x @ x)
    mse_coefficient = float(x @ target / denominator) if denominator > 0.0 else 1.0
    mse_coefficient = max(0.0, mse_coefficient)
    candidates = []
    for strength in RECALIBRATION_STRENGTHS:
        coefficient = 1.0 + strength * (mse_coefficient - 1.0)
        corrected = np.exp(level + coefficient * x)
        metrics = score_predictions(actual, corrected, calibration_train_actual)
        candidates.append({
            "strength": float(strength),
            "coefficient": float(coefficient),
            "validation_metrics": metrics,
            "in_iaao_prd": bool(0.98 <= metrics["PRD"] <= 1.03),
            "in_iaao_prb": bool(-0.05 <= metrics["PRB"] <= 0.05),
        })
    selection_frame = pd.DataFrame([
        {"candidate_id": f"strength_{candidate['strength']:.6g}", **candidate["validation_metrics"]}
        for candidate in candidates
    ])
    selection_frame, selected_row, selection_metadata = select_pareto_nash(selection_frame)
    selected_position = int(selection_frame.index.get_loc(selected_row.name))
    for position, candidate in enumerate(candidates):
        candidate["selection"] = {
            field: (bool(selection_frame.loc[selection_frame.index[position], field])
                    if field in {"selection_valid", "pareto_optimal", "selected_by_pareto_nash"}
                    else float(selection_frame.loc[selection_frame.index[position], field]))
            for field in [
                "selection_valid", "pareto_optimal", "selected_by_pareto_nash",
                "nash_log_hypervolume", "nash_hypervolume",
            ]
        }
    selected = candidates[selected_position]
    return {
        "fit_split": "rolling_origin",
        "n_folds": ROLLING_RECALIBRATION_FOLDS,
        "folds": folds,
        "n_oof": int(len(actual)),
        "frontier_strengths": list(RECALIBRATION_STRENGTHS),
        "oof_baseline_metrics": score_predictions(actual, raw_prediction, calibration_train_actual),
        "oof_level_only_metrics": level_only_metrics,
        "final_training_prediction_center": final_centre,
        "final_training_outcome_level": final_level,
        "oof_centered_prediction_range": [float(x.min()), float(x.max())],
        "mse_coefficient": mse_coefficient,
        "candidates": candidates,
        "selected_strength": float(selected["strength"]),
        "selection_rule": selection_metadata["name"],
        "selection": selection_metadata,
    }


def apply_rolling_recalibration(
    prediction_log: np.ndarray,
    specification: dict,
) -> np.ndarray:
    """Apply the validation-selected first-degree map with final-train anchors."""
    selected_strength = float(specification["selected_strength"])
    candidate = next(
        row for row in specification["candidates"]
        if np.isclose(float(row["strength"]), selected_strength)
    )
    x = np.asarray(prediction_log, dtype=float) - float(specification["final_training_prediction_center"])
    support_low, support_high = map(float, specification["oof_centered_prediction_range"])
    x = np.clip(x, support_low, support_high)
    return np.exp(float(specification["final_training_outcome_level"]) + float(candidate["coefficient"]) * x)


def select_neighbor_variants(
    data: pd.DataFrame,
    target_log: pd.Series,
    *,
    validation_split: int,
    split: int,
    include_repeat_sale: bool,
    include_assessment_values: bool,
    params: dict,
) -> dict:
    """Tune one bounded full comparable specification, then score ablations.

    The six-point full-specification screen is the only spatial hyperparameter
    search.  Its selected caps/count/weights are subsequently shared by the
    geographic and geographic-plus-time ablations, making those models controlled
    information comparisons rather than three costly independent searches.
    """
    features, categorical = feature_frame(
        data, validation_split, include_repeat_sale, include_assessment_values,
    )
    full_variant = next(variant for variant in NEIGHBOR_VARIANTS if variant.key == "geo_time_features")
    search_candidates = []
    for spec in NEIGHBOR_SEARCH_SPECS:
        neighbor_features, metadata = neighbor_feature_frame(
            data, target_log, train_rows=validation_split, end_rows=split,
            variant=full_variant, spec=spec,
        )
        augmented = pd.concat([features.iloc[:split], neighbor_features], axis=1)
        model = LGBMRegressor(**params)
        model.fit(
            augmented.iloc[:validation_split], target_log.iloc[:validation_split],
            categorical_feature=categorical,
        )
        prediction = np.exp(model.predict(augmented.iloc[validation_split:split]))
        search_candidates.append({
            "candidate_id": spec.key,
            "spec": spec,
            "metadata": metadata,
            "validation_metrics": score_predictions(
                data.sale_price.iloc[validation_split:split], prediction,
                data.sale_price.iloc[:validation_split],
            ),
            # Retain only the winning frame in memory for the next validation
            # stage; it avoids re-querying a dense county for the selected spec.
            "_neighbor_features": neighbor_features,
        })
    search_input = pd.DataFrame([
        {"candidate_id": row["candidate_id"], **row["spec"].as_dict(), **row["validation_metrics"]}
        for row in search_candidates
    ])
    search_frame, search_selected, search_metadata = select_pareto_nash(search_input)
    search_by_id = search_frame.set_index("candidate_id")
    for candidate in search_candidates:
        selection = search_by_id.loc[candidate["candidate_id"]]
        candidate["selection"] = {
            name: (
                bool(selection[name]) if name in {"selection_valid", "pareto_optimal", "selected_by_pareto_nash"}
                else float(selection[name])
            )
            for name in (
                "selection_valid", "pareto_optimal", "selected_by_pareto_nash",
                "nash_log_hypervolume", "nash_hypervolume",
            )
        }
    selected_search = next(
        candidate for candidate in search_candidates if candidate["candidate_id"] == search_selected["candidate_id"]
    )
    shared_spec: NeighborSearchSpec = selected_search["spec"]

    variants = []
    for variant in NEIGHBOR_VARIANTS:
        if variant.key == full_variant.key:
            neighbor_features = selected_search["_neighbor_features"]
            metadata = selected_search["metadata"]
            validation_metrics = selected_search["validation_metrics"]
        else:
            neighbor_features, metadata = neighbor_feature_frame(
                data, target_log, train_rows=validation_split, end_rows=split,
                variant=variant, spec=shared_spec,
            )
            augmented = pd.concat([features.iloc[:split], neighbor_features], axis=1)
            model = LGBMRegressor(**params)
            model.fit(
                augmented.iloc[:validation_split], target_log.iloc[:validation_split],
                categorical_feature=categorical,
            )
            prediction = np.exp(model.predict(augmented.iloc[validation_split:split]))
            validation_metrics = score_predictions(
                data.sale_price.iloc[validation_split:split], prediction,
                data.sale_price.iloc[:validation_split],
            )
        variants.append({
            "candidate_id": variant.key,
            "key": variant.key,
            "label": variant.label,
            "variant": variant,
            "spec": shared_spec,
            "metadata": metadata,
            "validation_metrics": validation_metrics,
        })
    representation_input = pd.DataFrame([
        {"candidate_id": row["candidate_id"], **row["validation_metrics"]} for row in variants
    ])
    representation_frame, representation_selected, representation_metadata = select_pareto_nash(representation_input)
    representation_by_id = representation_frame.set_index("candidate_id")
    for variant in variants:
        selection = representation_by_id.loc[variant["candidate_id"]]
        variant["representation_selection"] = {
            name: (
                bool(selection[name]) if name in {"selection_valid", "pareto_optimal", "selected_by_pareto_nash"}
                else float(selection[name])
            )
            for name in (
                "selection_valid", "pareto_optimal", "selected_by_pareto_nash",
                "nash_log_hypervolume", "nash_hypervolume",
            )
        }
        variant["selected_representation"] = bool(selection["selected_by_pareto_nash"])
    return {
        "shared_spec": shared_spec,
        "search_candidates": search_candidates,
        "search_selection": search_metadata,
        "variants": variants,
        "representation_selection": representation_metadata,
        "selected_variant_key": str(representation_selected["candidate_id"]),
    }


def fit_neighbor_comparisons(
    data: pd.DataFrame,
    target_log: pd.Series,
    *,
    split: int,
    validation_split: int,
    features: pd.DataFrame,
    categorical: list[str],
    include_repeat_sale: bool,
    include_assessment_values: bool,
    params: dict,
    county_fips: str,
    config_key: str,
    selected_neighbor_search: dict,
    shrinkage_targets: Sequence[float],
    early_stopping_rounds: int | None,
) -> tuple[list[tuple[str, float, str, np.ndarray]], list[dict], dict]:
    """Refit controlled neighbor ablations and validate one selected penalty path."""
    fitted = []
    metadata = []
    selected_variant_key = selected_neighbor_search["selected_variant_key"]
    selected_penalty_metadata = None
    for selected in selected_neighbor_search["variants"]:
        variant = selected["variant"]
        spec: NeighborSearchSpec = selected["spec"]
        neighbor_features, final_metadata = neighbor_feature_frame(
            data, target_log, train_rows=split, end_rows=len(data), variant=variant, spec=spec,
        )
        augmented = pd.concat([features, neighbor_features], axis=1)
        baseline = LGBMRegressor(**params)
        baseline.fit(augmented.iloc[:split], target_log.iloc[:split], categorical_feature=categorical)
        baseline_prediction = np.exp(baseline.predict(augmented))
        lgbm_key = f"neighbor_{variant.key}_lgbm"
        cov_key = f"neighbor_{variant.key}_cov"
        fitted.append((f"{NEIGHBOR_LGBM_MODEL}: {variant.label}", np.nan, lgbm_key, baseline_prediction))
        model_metadata = {
            "key": variant.key,
            "label": variant.label,
            "lgbm_model_key": lgbm_key,
            "penalized_model_key": None,
            "selected_representation": bool(selected["selected_representation"]),
            "validation": {
                "specification": spec.as_dict(),
                "metrics": selected["validation_metrics"],
                "representation_selection": selected["representation_selection"],
            },
            "final_feature_metadata": final_metadata,
            "n_features": int(augmented.shape[1]),
        }
        if variant.key == selected_variant_key:
            development_features, development_categorical = feature_frame(
                data, validation_split,
                include_repeat_sale=include_repeat_sale,
                include_assessment_values=include_assessment_values,
            )
            development_neighbor_features, _ = neighbor_feature_frame(
                data, target_log, train_rows=validation_split, end_rows=split,
                variant=variant, spec=spec,
            )
            development_augmented = pd.concat(
                [development_features.iloc[:split], development_neighbor_features], axis=1,
            )
            penalty_selection = select_penalty_on_validation(
                development_augmented, target_log, data,
                development_split=validation_split, validation_end=split, params=params,
                county_fips=county_fips, config_key=f"{config_key}__{variant.key}",
                shrinkage_targets=shrinkage_targets, early_stopping_rounds=early_stopping_rounds,
            )
            selected_target = penalty_selection["selected_requested_covariance_reduction"]
            full_plan, full_theory, full_theory_tables = plan_rho_grid(
                target_log.to_numpy()[:split], np.log(baseline_prediction[:split]),
                county_fips=county_fips, config_key=f"{config_key}__{variant.key}",
                shrinkage_targets=shrinkage_targets, include_anchors=False,
            )
            full_row = full_plan.loc[np.isclose(full_plan["requested_covariance_reduction"], selected_target)]
            if full_row.empty:
                raise RuntimeError("Selected neighbor shrinkage target was absent from the full-training theory grid.")
            rho = float(full_row.iloc[0]["rho"])
            penalized_prediction = fit_penalized_predictions(
                augmented, target_log, split=split, rho=rho, params=params,
                early_stopping_rounds=early_stopping_rounds,
            )
            fitted.append((f"{NEIGHBOR_PENALIZED_MODEL}: {variant.label}", rho, cov_key, penalized_prediction))
            model_metadata.update({
                "penalized_model_key": cov_key,
                "rho": rho,
                "requested_covariance_reduction": float(selected_target),
                "penalty_selection": {
                    **penalty_selection,
                    "selected_final_rho": rho,
                    "full_training_theory": full_theory,
                    "full_training_theory_tables": {
                        name: table.to_dict(orient="records") for name, table in full_theory_tables.items()
                    },
                },
            })
            selected_penalty_metadata = model_metadata["penalty_selection"]
        metadata.append(model_metadata)
    if selected_penalty_metadata is None:
        raise RuntimeError("The selected neighbor representation did not produce a penalty model.")
    return fitted, metadata, selected_penalty_metadata


def _model_key(model: str, rho: float) -> str:
    """Short, filename-safe identifier that is unique per (model, rho)."""
    if model == BASELINE_MODEL:
        return "lgbm_baseline"
    return f"cov_rho_{rho:g}".replace(".", "p").replace("-", "m")


def _bootstrap_one_model(job: dict) -> pd.DataFrame:
    """Score one model over shared time-block bootstrap indices."""
    rows = []
    for draw, idx in enumerate(job["indices"]):
        rows.append({
            "model": job["model"],
            "rho": job["rho"],
            "model_key": job["model_key"],
            "draw": draw,
            **score_predictions(job["actual"][idx], job["prediction"][idx], job["train_actual"]),
        })
    return pd.DataFrame(rows)


def compare_penalized_models(
    data: pd.DataFrame,
    *,
    configs: dict[str, dict],
    candidates: pd.DataFrame,
    split: int,
    validation_split: int,
    target_log: pd.Series,
    shrinkage_targets: Sequence[float],
    county_fips: str,
    n_bootstrap: int,
    bootstrap_block_freq: str,
    seed: int,
    early_stopping_rounds: int | None,
) -> dict:
    """Fit the log-scale LGBM baseline and its LGBCovPenalty[diff] counterparts.

    Everything except the objective is held fixed -- same features, same
    chronological split, same tuned hyperparameters -- so a difference between two
    rows of the returned metric table is attributable to the penalty strength.
    """
    log_candidates = candidates.loc[candidates["target_scale"].eq("log")].copy()
    if log_candidates.empty:
        raise ValueError(
            "The covariance penalty is defined on log residuals; include 'log' in --target-scales."
        )
    log_candidates["candidate_id"] = (
        log_candidates["feature_set"].astype(str) + "|" + log_candidates["target_scale"].astype(str)
        + "|" + log_candidates["lgbm_config"].astype(str)
    )
    _, best, log_candidate_selection = select_pareto_nash(log_candidates)
    include_repeat_sale, include_assessment_values = FEATURE_SET_OPTIONS[str(best.feature_set)]
    features, categorical = feature_frame(data, split, include_repeat_sale, include_assessment_values)
    params = configs[str(best.lgbm_config)]
    train, test = data.iloc[:split], data.iloc[split:]
    train_price = train.sale_price.to_numpy()

    started_at = perf_counter()
    baseline = LGBMRegressor(**params)
    baseline.fit(features.iloc[:split], target_log.iloc[:split], categorical_feature=categorical)
    baseline_prediction = np.exp(baseline.predict(features))
    linear_recalibration = fit_rolling_recalibration(
        data, target_log, split=split, fold_rows=split - validation_split,
        include_repeat_sale=include_repeat_sale, include_assessment_values=include_assessment_values,
        params=params, final_train_prediction_log=np.log(baseline_prediction[:split]), started_at=started_at,
    )
    scaling_prediction = apply_rolling_recalibration(np.log(baseline_prediction), linear_recalibration)

    report_progress("Neighbor validation selection", 0, len(NEIGHBOR_SEARCH_SPECS), started_at)
    selected_neighbor_search = select_neighbor_variants(
        data, target_log, validation_split=validation_split, split=split,
        include_repeat_sale=include_repeat_sale, include_assessment_values=include_assessment_values,
        params=params,
    )
    report_progress("Neighbor validation selection", len(NEIGHBOR_SEARCH_SPECS), len(NEIGHBOR_SEARCH_SPECS), started_at)

    development_features, _ = feature_frame(
        data, validation_split, include_repeat_sale, include_assessment_values,
    )
    baseline_penalty_selection = select_penalty_on_validation(
        development_features.iloc[:split], target_log, data,
        development_split=validation_split, validation_end=split, params=params,
        county_fips=county_fips, config_key=str(best.lgbm_config),
        shrinkage_targets=shrinkage_targets, early_stopping_rounds=early_stopping_rounds,
    )
    selected_baseline_target = baseline_penalty_selection["selected_requested_covariance_reduction"]
    rho_plan, theory, theory_tables = plan_rho_grid(
        target_log.to_numpy()[:split], np.log(baseline_prediction[:split]),
        county_fips=county_fips, config_key=str(best.lgbm_config),
        shrinkage_targets=shrinkage_targets, include_anchors=False,
    )
    validation_penalty_by_target = {
        round(float(candidate["requested_covariance_reduction"]), 8): candidate
        for candidate in baseline_penalty_selection["candidates"]
    }
    rho_plan = rho_plan.copy()
    rho_plan["county_fips"] = county_fips
    rho_plan["model_key"] = [
        _shrinkage_key("cov", value) for value in rho_plan["requested_covariance_reduction"]
    ]
    rho_plan["development_rho"] = [
        validation_penalty_by_target[round(float(value), 8)]["development_rho"]
        for value in rho_plan["requested_covariance_reduction"]
    ]
    rho_plan["selected_on_chronological_validation"] = [
        bool(np.isclose(float(value), selected_baseline_target))
        for value in rho_plan["requested_covariance_reduction"]
    ]
    for field in ("selection_valid", "pareto_optimal", "selected_by_pareto_nash", "nash_log_hypervolume", "nash_hypervolume"):
        rho_plan[f"validation_{field}"] = [
            validation_penalty_by_target[round(float(value), 8)]["selection"][field]
            for value in rho_plan["requested_covariance_reduction"]
        ]
    for metric in ("MAPE", "COD", "PRD", "PRB", "RMSE (log)", "Cov(e,logprice)"):
        rho_plan[f"validation_{metric}"] = [
            validation_penalty_by_target[round(float(value), 8)]["validation_metrics"][metric]
            for value in rho_plan["requested_covariance_reduction"]
        ]

    fitted: list[tuple[str, float, str, np.ndarray]] = [
        (BASELINE_MODEL, np.nan, _model_key(BASELINE_MODEL, float("nan")), baseline_prediction),
        (SCALING_MODEL, np.nan, SCALING_MODEL_KEY, scaling_prediction),
    ]
    total_fits = 2 + len(rho_plan) + len(NEIGHBOR_VARIANTS) + 1
    completed_fits = 2
    report_progress("Comparison fits", completed_fits, total_fits, started_at)
    for row in rho_plan.itertuples(index=False):
        fitted.append((PENALIZED_MODEL, float(row.rho), str(row.model_key), fit_penalized_predictions(
            features, target_log, split=split, rho=float(row.rho), params=params,
            early_stopping_rounds=early_stopping_rounds,
        )))
        completed_fits += 1
        report_progress("Comparison fits", completed_fits, total_fits, started_at)
    neighbor_fitted, neighbor_models, selected_neighbor_penalty = fit_neighbor_comparisons(
        data, target_log, split=split, validation_split=validation_split, features=features, categorical=categorical,
        include_repeat_sale=include_repeat_sale, include_assessment_values=include_assessment_values, params=params,
        county_fips=county_fips, config_key=str(best.lgbm_config), selected_neighbor_search=selected_neighbor_search,
        shrinkage_targets=shrinkage_targets, early_stopping_rounds=early_stopping_rounds,
    )
    fitted.extend(neighbor_fitted)
    model_keys = [key for _, _, key, _ in fitted]
    duplicate_model_keys = sorted({key for key in model_keys if model_keys.count(key) > 1})
    if duplicate_model_keys:
        raise RuntimeError(
            "Comparison model keys must be unique before writing predictions; duplicates: "
            f"{duplicate_model_keys}."
        )
    completed_fits += len(neighbor_fitted)
    report_progress("Comparison fits", completed_fits, total_fits, started_at)

    metric_rows, jobs = [], []
    indices = _build_time_block_bootstrap_indices(
        val_dates=pd.to_datetime(test["sale_date"]), n_bootstrap=n_bootstrap,
        block_freq=bootstrap_block_freq, rng_seed=seed,
    )
    wide = pd.DataFrame(index=data.index)
    for model, rho, key, prediction in fitted:
        wide[f"predicted_sale_price__{key}"] = prediction
        for label, frame, slice_ in (("train", train, slice(None, split)), ("test", test, slice(split, None))):
            metric_rows.append({
                "county_fips": county_fips, "model": model, "rho": rho, "model_key": key,
                "split": label, **score_predictions(frame.sale_price, prediction[slice_], train_price),
            })
    selected_baseline_rows = rho_plan.loc[
        rho_plan["selected_on_chronological_validation"], ["model_key", "rho"]
    ]
    if len(selected_baseline_rows) != 1:
        raise RuntimeError(
            "The validation-selected baseline shrinkage target must occur exactly once "
            "in the full-training rho plan."
        )
    baseline_penalty_model_key = str(selected_baseline_rows.iloc[0]["model_key"])
    selected_neighbor = next(model for model in neighbor_models if model["selected_representation"])
    selected_neighbor_lgbm_key = str(selected_neighbor["lgbm_model_key"])
    selected_neighbor_cov_key = str(selected_neighbor["penalized_model_key"])
    paired_comparisons = [
        {
            "id": "baseline_penalty_vs_baseline", "label": "Validation-selected baseline penalty vs baseline",
            "reference_model_key": BASELINE_KEY, "candidate_model_key": baseline_penalty_model_key,
        },
        {
            "id": "scaling_vs_baseline", "label": "Validation-selected first-degree scaling vs baseline",
            "reference_model_key": BASELINE_KEY, "candidate_model_key": SCALING_MODEL_KEY,
        },
        {
            "id": "neighbor_vs_baseline", "label": "Validation-selected neighbor model vs baseline",
            "reference_model_key": BASELINE_KEY, "candidate_model_key": selected_neighbor_lgbm_key,
        },
        {
            "id": "neighbor_penalty_vs_neighbor", "label": "Validation-selected neighbor penalty vs neighbor model",
            "reference_model_key": selected_neighbor_lgbm_key, "candidate_model_key": selected_neighbor_cov_key,
        },
    ]
    bootstrap_model_keys = list(dict.fromkeys(
        [key for pair in paired_comparisons for key in (pair["reference_model_key"], pair["candidate_model_key"])]
    ))
    fitted_by_key = {key: (model, rho, prediction) for model, rho, key, prediction in fitted}
    for key in bootstrap_model_keys:
        model, rho, prediction = fitted_by_key[key]
        jobs.append({
            "model": model, "rho": rho, "model_key": key, "indices": indices,
            "actual": test.sale_price.to_numpy(), "prediction": prediction[split:],
            "train_actual": train_price,
        })

    # Persist development-prefix validation scores under the final, portable model
    # keys.  Their full-training/test rows below are never used to choose a model.
    metric_rows.append({
        "county_fips": county_fips, "model": BASELINE_MODEL, "rho": np.nan,
        "model_key": BASELINE_KEY, "split": "chronological_validation",
        **baseline_penalty_selection["baseline_validation_metrics"],
    })
    for candidate in baseline_penalty_selection["candidates"]:
        target = float(candidate["requested_covariance_reduction"])
        metric_rows.append({
            "county_fips": county_fips, "model": PENALIZED_MODEL,
            "rho": float(candidate["development_rho"]),
            "model_key": _shrinkage_key("cov", target), "split": "chronological_validation",
            **candidate["validation_metrics"],
        })
    selected_scaling_candidate = next(
        candidate for candidate in linear_recalibration["candidates"]
        if np.isclose(float(candidate["strength"]), float(linear_recalibration["selected_strength"]))
    )
    metric_rows.append({
        "county_fips": county_fips, "model": SCALING_MODEL, "rho": np.nan,
        "model_key": SCALING_MODEL_KEY, "split": "rolling_origin_oof",
        **selected_scaling_candidate["validation_metrics"],
    })
    for neighbor in neighbor_models:
        metric_rows.append({
            "county_fips": county_fips, "model": f"{NEIGHBOR_LGBM_MODEL}: {neighbor['label']}",
            "rho": np.nan, "model_key": neighbor["lgbm_model_key"], "split": "chronological_validation",
            **neighbor["validation"]["metrics"],
        })
        if neighbor["penalized_model_key"]:
            metric_rows.append({
                "county_fips": county_fips, "model": f"{NEIGHBOR_PENALIZED_MODEL}: {neighbor['label']}",
                "rho": float(neighbor["penalty_selection"]["selected_development_rho"]),
                "model_key": neighbor["penalized_model_key"], "split": "chronological_validation",
                **neighbor["penalty_selection"]["selected_validation_metrics"],
            })

    # Scored in-process on purpose: LightGBM has already initialised OpenMP by this
    # point, and forking a worker pool on top of that deadlocks.  Parallelism lives
    # at the county level instead, where the SLURM array already provides it.
    bootstrap_started_at = perf_counter()
    draw_frames = []
    for number, job in enumerate(jobs, start=1):
        draw_frames.append(_bootstrap_one_model(job))
        report_progress("Comparison bootstrap", number, len(jobs), bootstrap_started_at)
    draws = pd.concat(draw_frames, ignore_index=True)
    identifiers = ["model", "rho", "model_key", "draw"]
    numeric = [c for c in draws.select_dtypes(include="number").columns if c not in identifiers]
    summary = (
        draws.melt(id_vars=identifiers, value_vars=numeric, var_name="metric", value_name="value")
        .groupby(["model", "rho", "model_key", "metric"], as_index=False, dropna=False)
        .agg(
            mean=("value", "mean"),
            std=("value", "std"),
            ci_2_5=("value", lambda s: s.quantile(0.025)),
            ci_97_5=("value", lambda s: s.quantile(0.975)),
        )
        .assign(county_fips=county_fips)
    )

    metrics = pd.DataFrame(metric_rows)
    # Realized shrinkage is the honest counterpart of the requested one: it says
    # whether the theory-implied rho actually delivered the covariance reduction.
    test_metrics = metrics.loc[metrics["split"].eq("test")].set_index("model_key")
    baseline_cov = float(test_metrics.loc[BASELINE_KEY, "Cov(e,logprice)"])
    rho_plan["realized_covariance_reduction_test"] = [
        1.0 - float(test_metrics.loc[key, "Cov(e,logprice)"]) / baseline_cov
        if key in test_metrics.index and baseline_cov != 0.0 else np.nan
        for key in rho_plan["model_key"]
    ]
    local_model_keys = [BASELINE_KEY, selected_neighbor_lgbm_key, selected_neighbor_cov_key]
    local_groups, local_summary, local_protocol = local_equity_diagnostics(
        data, wide, split=split, model_keys=local_model_keys,
    )
    neighbor_search_artifact = {
        # This bounded specification search is deliberately run on the richest
        # comparable representation; the separately selected representation is
        # recorded below as ``selected_representation_key``.
        "search_representation": "geo_time_features",
        "search_selection": selected_neighbor_search["search_selection"],
        "search_candidates": [
            {
                **candidate["spec"].as_dict(),
                "metadata": candidate["metadata"],
                "validation_metrics": candidate["validation_metrics"],
                "selection": candidate["selection"],
            }
            for candidate in selected_neighbor_search["search_candidates"]
        ],
        "shared_selected_specification": selected_neighbor_search["shared_spec"].as_dict(),
        "representation_selection": selected_neighbor_search["representation_selection"],
        "selected_representation_key": selected_neighbor_search["selected_variant_key"],
    }
    model_keys = [key for _, _, key, _ in fitted]
    return {
        "comparison_selection": {
            "feature_set": str(best.feature_set), "lgbm_config": str(best.lgbm_config),
            "target_scale": "log", "n_features": int(features.shape[1]),
            "validation_r2": float(best["R2"]), "validation_r2_log": float(best["R2 (log)"]),
            "early_stopping_rounds": early_stopping_rounds,
            "n_bootstrap": int(n_bootstrap), "bootstrap_block_freq": bootstrap_block_freq,
            "selection_protocol": {
                "name": "pareto_nash_hyperrectangle_v1",
                "development_split": "development_prefix",
                "validation_split": "chronological_validation",
                "test_used_for_selection": False,
                "objectives": ["MAPE", "COD", "abs(PRD-1)", "abs(PRB)"],
            },
            "log_baseline_candidate_selection": log_candidate_selection,
            "linear_recalibration": {
                **linear_recalibration,
                "selected_model_key": SCALING_MODEL_KEY,
            },
            "baseline_penalty": {
                **baseline_penalty_selection,
                "selected_model_key": baseline_penalty_model_key,
                "selected_final_rho": float(
                    rho_plan.loc[rho_plan["selected_on_chronological_validation"], "rho"].iloc[0]
                ),
            },
            "neighbor_search": neighbor_search_artifact,
            "neighbor_models": neighbor_models,
            "selected_neighbor_model_key": selected_neighbor_lgbm_key,
            "selected_neighbor_penalty_model_key": selected_neighbor_cov_key,
            "neighbor_transformer_provenance": NEIGHBOR_TRANSFORMER_PROVENANCE,
            "paired_comparisons": paired_comparisons,
            "model_manifest": {
                "expected_model_keys": model_keys,
                "expected_neighbor_model_keys": [
                    key for neighbor in neighbor_models
                    for key in (neighbor["lgbm_model_key"], neighbor["penalized_model_key"])
                    if key
                ],
                "bootstrap_model_keys": bootstrap_model_keys,
                "test_oracle_frontier_model_keys": rho_plan["model_key"].tolist(),
            },
            "local_equity": {
                **local_protocol,
                "summary_file": "model_comparison_local_equity_summary.csv",
                "groups_file": "model_comparison_local_equity_groups.csv",
                "summary": local_summary.to_dict(orient="records"),
            },
        },
        "theory": theory,
        "metrics": metrics,
        "bootstrap_draws": draws,
        "bootstrap_summary": summary,
        "rho_plan": rho_plan,
        "theory_tables": theory_tables,
        "predictions": wide,
        "local_equity_groups": local_groups,
        "local_equity_summary": local_summary,
    }


def run_model(data: pd.DataFrame, test_fraction: float, validation_fraction: float, config_path: Path, config_keys: str, feature_sets: str, target_scales: str, n_bootstrap: int, bootstrap_block_freq: str, seed: int, lgbm_threads: int | None = None, comparison: dict | None = None) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict | None]:
    """Select by recent validation, refit on all training sales, then bootstrap test metrics."""
    split = int(len(data) * (1 - test_fraction))
    validation_split = int(split * (1 - validation_fraction))
    if validation_split < 1 or validation_split >= split or split >= len(data):
        raise ValueError("The matched sample is too small for the requested chronological splits.")
    target_log = np.log(data["sale_price"].astype(float))
    configs = load_lgbm_configs(config_path, config_keys, lgbm_threads)
    variants = [name.strip() for name in feature_sets.split(",") if name.strip()]
    scales = [name.strip() for name in target_scales.split(",") if name.strip()]
    if not variants or set(variants) - set(FEATURE_SET_OPTIONS):
        raise ValueError(f"--feature-sets must contain only: {', '.join(sorted(FEATURE_SET_OPTIONS))}.")
    if set(scales) - {"log", "raw"}:
        raise ValueError("--target-scales must contain log and/or raw.")
    candidate_rows = []
    candidate_count = len(variants) * len(scales) * len(configs)
    candidate_started_at = perf_counter()
    candidate_number = 0
    for feature_set in variants:
        include_repeat_sale, include_assessment_values = FEATURE_SET_OPTIONS[feature_set]
        features, categorical = feature_frame(data, validation_split, include_repeat_sale, include_assessment_values)
        for target_scale in scales:
            target = target_log if target_scale == "log" else data["sale_price"]
            for name, params in configs.items():
                model = LGBMRegressor(**params)
                model.fit(features.iloc[:validation_split], target.iloc[:validation_split], categorical_feature=categorical)
                raw_prediction = model.predict(features.iloc[validation_split:split])
                prediction = np.exp(raw_prediction) if target_scale == "log" else np.maximum(raw_prediction, 1.0)
                scores = score_predictions(data.sale_price.iloc[validation_split:split], prediction, data.sale_price.iloc[:validation_split])
                candidate_rows.append({"feature_set": feature_set, "target_scale": target_scale, "lgbm_config": name, "n_features": int(features.shape[1]), **scores})
                candidate_number += 1
                report_progress("Validation models", candidate_number, candidate_count, candidate_started_at)
    candidates = pd.DataFrame(candidate_rows)
    candidates["candidate_id"] = (
        candidates["feature_set"].astype(str) + "|" + candidates["target_scale"].astype(str)
        + "|" + candidates["lgbm_config"].astype(str)
    )
    candidates, best, base_candidate_selection = select_pareto_nash(candidates)
    candidates = candidates.sort_values(
        ["selected_by_pareto_nash", "pareto_optimal", "nash_log_hypervolume", "candidate_id"],
        ascending=[False, False, False, True], kind="mergesort",
    ).reset_index(drop=True)
    include_repeat_sale, include_assessment_values = FEATURE_SET_OPTIONS[str(best.feature_set)]
    features, categorical = feature_frame(data, split, include_repeat_sale, include_assessment_values)
    params = configs[str(best.lgbm_config)]
    model = LGBMRegressor(**params)
    target = target_log if best.target_scale == "log" else data["sale_price"]
    model.fit(features.iloc[:split], target.iloc[:split], categorical_feature=categorical)
    raw_prediction = model.predict(features)
    prediction = np.exp(raw_prediction) if best.target_scale == "log" else np.maximum(raw_prediction, 1.0)
    prediction_log = np.log(prediction)
    train = data.iloc[:split]
    test = data.iloc[split:]
    bootstrap_draws, bootstrap_summary = bootstrap_scores(
        test, prediction[split:], train.sale_price.to_numpy(), n_bootstrap, bootstrap_block_freq, seed,
    )
    report = {
        "n_transactions": int(len(data)), "n_train": split, "n_test": int(len(test)),
        "sale_date_range": [str(data.sale_date.min().date()), str(data.sale_date.max().date())],
        "test_date_range": [str(test.sale_date.min().date()), str(test.sale_date.max().date())],
        "selection": {
            "validation_fraction": validation_fraction,
            "development_rows": int(validation_split),
            "chronological_validation_rows": int(split - validation_split),
            "feature_set": str(best.feature_set), "target_scale": str(best.target_scale),
            "lgbm_config": str(best.lgbm_config), "validation_r2": float(best["R2"]),
            "validation_r2_log": float(best["R2 (log)"]),
            "selection": base_candidate_selection,
        },
        "model": params,
        "features": list(features.columns),
        "categorical_features": categorical,
        "feature_coverage": feature_coverage(features, categorical),
        "train_metrics": score_predictions(train.sale_price, prediction[:split], train.sale_price),
        "test_metrics": score_predictions(test.sale_price, prediction[split:], train.sale_price),
        "test_bootstrap": {"n_bootstrap": n_bootstrap, "block_freq": bootstrap_block_freq, "metrics": bootstrap_summary.set_index("metric").to_dict(orient="index")},
    }
    prediction_columns = [
        "ATTOMID", "TRANSACTIONID", "DOCUMENTTYPECODE", "ARMSLENGTHFLAG", "TRANSFERINFOMULTIPARCELFLAG",
        "sale_date", "sale_price", "TAXYEARASSESSED", "tax_assessor_geoid", "tax_assessor_latitude",
        "tax_assessor_longitude", "LATITUDE", "LONGITUDE",
    ]
    predictions = data[[column for column in prediction_columns if column in data]].copy()
    predictions["split"] = np.where(np.arange(len(data)) < split, "train", "test")
    predictions["sale_log_price"] = target_log
    predictions["predicted_log_sale_price"] = prediction_log
    predictions["predicted_sale_price"] = prediction
    metrics_table = pd.DataFrame([{"split": "train", **report["train_metrics"]}, {"split": "test", **report["test_metrics"]}])

    penalized = None
    if comparison is not None:
        penalized = compare_penalized_models(
            data, configs=configs, candidates=candidates, split=split, validation_split=validation_split, target_log=target_log,
            n_bootstrap=n_bootstrap, bootstrap_block_freq=bootstrap_block_freq, seed=seed,
            **comparison,
        )
        report["comparison"] = {
            **penalized["comparison_selection"],
            "theory": penalized["theory"],
            "rho_grid": penalized["rho_plan"].to_dict(orient="records"),
        }
        predictions = pd.concat([predictions, penalized.pop("predictions")], axis=1)
    return report, predictions, metrics_table, candidates, bootstrap_draws, bootstrap_summary, penalized


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessor-dir", type=Path, default=ASSESSOR_DIR)
    parser.add_argument("--recorder-dir", type=Path, default=RECORDER_DIR)
    parser.add_argument("--tax-assessor-dir", type=Path, default=TAX_ASSESSOR_DIR)
    parser.add_argument("--acs-dir", type=Path, default=ACS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--assessor-files", type=int, default=0, help="Assessor History shards to read; 0 reads all shards.")
    parser.add_argument("--recorder-files", type=int, default=0, help="Recorder shards to read; 0 reads all shards (required for final results).")
    parser.add_argument("--tax-assessor-files", type=int, default=0, help="Tax Assessor shards to read; 0 reads all shards.")
    parser.add_argument("--county-fips", default=COOK_FIPS)
    parser.add_argument(
        "--property-use-codes",
        default=None,
        help="Comma-separated, verified ATTOM PROPERTYUSESTANDARDIZED codes for the modeled "
             "residential universe; omit to keep every observed code.",
    )
    parser.add_argument(
        "--sale-cohort",
        choices=SALE_COHORTS,
        default=BROAD_COHORT,
        help="broad keeps transactions whose codes are undocumented rather than defective; "
             "strict admits only reviewed inclusion codes (default: %(default)s).",
    )
    parser.add_argument(
        "--sale-validation-dictionary",
        type=Path,
        default=SALE_VALIDATION_DICTIONARY_PATH,
        help="Reviewed Recorder code-decision CSV (default: %(default)s).",
    )
    parser.add_argument(
        "--minimum-sale-price",
        type=float,
        default=10_000.0,
        help="Minimum positive consideration for a qualified sale (default: %(default).0f).",
    )
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--lgbm-config-path", type=Path, default=LGBM_CONFIG_PATH)
    parser.add_argument("--lgbm-config-keys", default="test_best_r2,cv_top1_r2,cv_top2_r2")
    parser.add_argument(
        "--lgbm-threads",
        type=int,
        default=None,
        help="Override the stored LightGBM n_jobs, which the tuned configs pin to 1; "
             "-1 uses every core on the node (default: keep the stored value).",
    )
    parser.add_argument("--feature-sets", default="ccao_core_acs,attom_market_history,status_quo_augmented")
    parser.add_argument("--target-scales", default="log,raw")
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--bootstrap-block-freq", default="M")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument(
        "--skip-penalty-comparison",
        action="store_true",
        help="Fit only the unpenalized LightGBM baseline and skip the LGBCovPenalty arm.",
    )
    parser.add_argument(
        "--cov-shrinkage-targets",
        default=",".join(str(value) for value in COVARIANCE_SHRINKAGE_TARGETS),
        help="Requested reductions of the baseline log-residual/log-price covariance. Each is "
             "converted to a county-specific rho by the rank-one theory (default: %(default)s).",
    )
    parser.add_argument(
        "--penalty-early-stopping-rounds",
        type=int,
        default=0,
        help="Training-loss patience for LGBCovPenalty; 0 disables it so every model in the "
             "comparison uses the same tree count as the baseline (default: %(default)d).",
    )
    parser.add_argument("--include-non-arms-length", action="store_true")
    parser.add_argument("--include-multi-parcel", action="store_true")
    args = parser.parse_args()
    property_use_codes = parse_property_use_codes(args.property_use_codes)
    if args.property_use_codes is not None and not property_use_codes:
        parser.error("--property-use-codes was supplied without a usable ATTOM code.")
    if args.minimum_sale_price <= 0:
        parser.error("--minimum-sale-price must be positive.")
    sale_validation_decisions, sale_validation_dictionary_sha256 = load_sale_validation_dictionary(
        args.sale_validation_dictionary,
    )
    sale_validation_policy = SaleValidationPolicy(
        minimum_sale_price=args.minimum_sale_price,
        dictionary_path=str(args.sale_validation_dictionary),
        dictionary_sha256=sale_validation_dictionary_sha256,
        code_decisions=sale_validation_decisions,
        arms_length_only=not args.include_non_arms_length,
        single_parcel_only=not args.include_multi_parcel,
        cohort=args.sale_cohort,
    )

    started_at = perf_counter()
    def stage(message: str) -> None:
        print(f"[{perf_counter() - started_at:7.1f}s] {message}", flush=True)

    stage("Selecting input shards")
    assessor_files = files_or_sample(args.assessor_dir, "assessor-history_*.parquet", args.assessor_files)
    recorder_files = files_or_sample(args.recorder_dir, "recorder_*.parquet", args.recorder_files)
    recorder_history_files = files_or_sample(args.recorder_dir, "recorder_*.parquet", 0)
    tax_assessor_files = files_or_sample(args.tax_assessor_dir, "tax-assessor_*.parquet", args.tax_assessor_files)
    unreadable_files: dict[str, list[str]] = {}
    for label, selected in (
        ("assessor_history", assessor_files), ("recorder", recorder_files),
        ("recorder_history", recorder_history_files), ("tax_assessor", tax_assessor_files),
    ):
        selected[:], skipped = readable_parquet_files(selected)
        if skipped:
            unreadable_files[label] = skipped
            stage(f"WARNING: skipping {len(skipped)} unreadable {label} shard(s): {', '.join(skipped)}")
    stage("Reading and filtering recorder transactions")
    transactions, transaction_validation_audit, transaction_validation_waterfall = read_transactions(
        recorder_files, args.county_fips, sale_validation_policy,
    )
    stage(f"Retained {len(transactions):,} transactions; reading assessor history")
    history = read_history(assessor_files, transactions.ATTOMID, args.county_fips)
    stage("Matching assessor history and recorder prior sales")
    matched = match_history(transactions, history)
    n_history_matches = len(matched)
    n_history_id_overlap = int(transactions["ATTOMID"].isin(history["ATTOMID"]).sum())
    if property_use_codes:
        matched = matched.loc[
            normalize_property_use(matched["PROPERTYUSESTANDARDIZED"]).isin(property_use_codes)
        ].copy()
        if matched.empty:
            raise ValueError("No history-matched transactions remain after the verified property-use filter.")
        stage(f"Retained {len(matched):,} transactions in the verified property-use universe")
    else:
        stage(f"Retained all {len(matched):,} history-matched transactions; no property-use filter")
    matched = attach_recorder_prior_sales(
        matched, recorder_history_files, args.county_fips, sale_validation_policy,
    )
    stage("Reading tax-assessor location crosswalk")
    tax_assessor = read_tax_assessor(tax_assessor_files, matched.ATTOMID, args.county_fips)
    matched = attach_tax_assessor(matched, tax_assessor)
    stage("Reading and attaching ACS features")
    matched = attach_acs(matched, read_acs(args.acs_dir, args.county_fips))
    stage(f"Fitting models on {len(matched):,} matched sales")
    comparison = None
    if not args.skip_penalty_comparison:
        comparison = {
            "shrinkage_targets": parse_shrinkage_targets(args.cov_shrinkage_targets),
            "county_fips": args.county_fips,
            "early_stopping_rounds": (
                None if args.penalty_early_stopping_rounds <= 0 else args.penalty_early_stopping_rounds
            ),
        }
    report, predictions, metrics_table, candidates, bootstrap_draws, bootstrap_summary, penalized = run_model(
        matched, args.test_fraction, args.validation_fraction, args.lgbm_config_path,
        args.lgbm_config_keys, args.feature_sets, args.target_scales, args.n_bootstrap, args.bootstrap_block_freq, args.seed,
        args.lgbm_threads, comparison,
    )
    report["sample"] = {
        "county_fips": args.county_fips,
        "property_use_codes": sorted(property_use_codes) or "all_observed",
        "test_fraction": args.test_fraction,
        "sale_validation_policy": {
            "minimum_sale_price": sale_validation_policy.minimum_sale_price,
            "dictionary_path": sale_validation_policy.dictionary_path,
            "dictionary_sha256": sale_validation_policy.dictionary_sha256,
            "arms_length_only": sale_validation_policy.arms_length_only,
            "single_parcel_only": sale_validation_policy.single_parcel_only,
            "cohort": sale_validation_policy.cohort,
        },
        "unreadable_files": unreadable_files,
        "assessor_files": [file.name for file in assessor_files],
        "recorder_files": [file.name for file in recorder_files],
        "recorder_history_files": [file.name for file in recorder_history_files],
        "tax_assessor_files": [file.name for file in tax_assessor_files],
        "n_raw_county_recorder_transfers": int(len(transaction_validation_audit)),
        "n_transactions_before_history": int(len(transactions)),
        "n_history_matches": int(n_history_matches),
        "n_property_use_eligible": int(len(matched)),
        "unverified_code_share": float(matched["sale_validation_has_unverified_code"].mean()),
        # Identifier coverage and the strictly-before-sale rule are separate losses:
        # a sale whose only history row shares its assessor year is an overlap
        # without a temporal match.
        "assessor_history_attomid_overlap_rate": (
            float(n_history_id_overlap / len(transactions)) if len(transactions) else None
        ),
        "assessor_history_match_rate": float(n_history_matches / len(transactions)) if len(transactions) else None,
        "recorder_prior_sale_rate": float(matched["recorder_prior_sale_amount"].notna().mean()),
        "tax_assessor_match_rate": float(matched["tax_assessor_matched"].mean()),
        "tax_assessor_usable_location_rate": float(matched["tax_assessor_location_usable"].mean()),
        "acs_match_rate": float(matched["acs_matched"].mean()),
    }
    # This is deliberately written only after every other artifact succeeds; the
    # dashboard treats it as the completion marker for a coherent county run.
    report["artifact_schema_version"] = ARTIFACT_SCHEMA_VERSION

    stage("Writing outputs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    transaction_validation_audit.to_parquet(args.output_dir / "transaction_validation_audit.parquet", index=False)
    transaction_validation_waterfall.to_csv(args.output_dir / "transaction_validation_waterfall.csv", index=False)
    matched.to_parquet(args.output_dir / "matched_sales.parquet", index=False)
    predictions.to_parquet(args.output_dir / "predictions.parquet", index=False)
    audit_columns = [
        "PROPERTYUSESTANDARDIZED", "AREABUILDING", "AREALOTSF", "UNITSCOUNT", "ASSESSORHISTORYYEAR",
        "TAXYEARASSESSED", "assessed_through", "history_lag_days", "history_lag_years",
        "tax_assessor_apn_consistent", "tax_assessor_address_consistent", "tax_assessor_location_usable",
    ]
    error_audit = predictions.merge(
        matched[["TRANSACTIONID", *audit_columns]], on="TRANSACTIONID", how="left", validate="1:1",
    )
    error_audit["ratio"] = error_audit["predicted_sale_price"] / error_audit["sale_price"]
    error_audit["absolute_percentage_error"] = (error_audit["ratio"] - 1).abs()
    error_audit.nlargest(1_000, "absolute_percentage_error").to_parquet(
        args.output_dir / "largest_errors_audit.parquet", index=False,
    )
    metrics_table.to_csv(args.output_dir / "metrics.csv", index=False)
    candidates.to_csv(args.output_dir / "validation_candidates.csv", index=False)
    bootstrap_draws.to_csv(args.output_dir / "test_bootstrap_draws.csv", index=False)
    bootstrap_summary.to_csv(args.output_dir / "test_bootstrap_summary.csv", index=False)
    if penalized is not None:
        penalized["metrics"].to_csv(args.output_dir / "model_comparison_metrics.csv", index=False)
        penalized["rho_plan"].to_csv(args.output_dir / "rho_plan.csv", index=False)
        penalized["bootstrap_summary"].to_csv(args.output_dir / "model_comparison_bootstrap_summary.csv", index=False)
        penalized["bootstrap_draws"].to_parquet(args.output_dir / "model_comparison_bootstrap_draws.parquet", index=False)
        penalized["local_equity_groups"].to_csv(
            args.output_dir / "model_comparison_local_equity_groups.csv", index=False,
        )
        penalized["local_equity_summary"].to_csv(
            args.output_dir / "model_comparison_local_equity_summary.csv", index=False,
        )
        for name, table in penalized["theory_tables"].items():
            table.to_csv(args.output_dir / f"theory_{name}.csv", index=False)
    metrics_path = args.output_dir / "metrics.json"
    metrics_tmp_path = metrics_path.with_suffix(".json.tmp")
    metrics_tmp_path.write_text(json.dumps(report, indent=2) + "\n")
    metrics_tmp_path.replace(metrics_path)
    stage("Complete")
    print(json.dumps(report, indent=2))
    print(f"Wrote validation audit, matched sales, predictions, and metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
