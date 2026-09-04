#!/usr/bin/env python3
"""Shared paths, jurisdiction table and constants for external_jurisdiction_benchmark_v1.

Does not import modeling code at module import beyond stdlib/pyarrow, so
lightweight audits can run without LightGBM (some Sloan batch nodes cannot
dlopen the fairness_env lib_lightgbm.so).
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ANALYSIS = REPO / "analysis" / "external_jurisdiction_benchmark_v1"
OUTPUT = REPO / "output" / "external_jurisdiction_benchmark_v1"

PYTHON = "/home/nacevedo/.conda/envs/fairness_env/bin/python"
PARTITION = "sched_mit_sloan_batch_r8"

DEWEY = REPO / "data" / "dewey-downloads"
SALE_VALIDATION_DICTIONARY = (
    DEWEY / "data_dictionaries" / "attom_recorder_residential_avm_sale_validation_dictionary.csv"
)
LGBM_CONFIG_PATH = ANALYSIS / "baseline" / "shared_lgbm_grid.yaml"

# County -> {fips, assessor Assessor-History dir, Recorder dir, label}.
# The county->folder mapping previously existed only as a bash `case` block in
# attom_county_benchmark.sh:52-59; lifted here as the single source of truth.
# Assessor History folders are single-product (History only) everywhere;
# Recorder always comes from one of the two shared multi-county folders.
_RECORDER_10 = DEWEY / "10-counties-recorder-2016-2025"
_RECORDER_WPS = DEWEY / "wayne-philadelphia-st-louis-2003-2025-recorder"

JURISDICTIONS = (
    {"key": "wayne", "fips": "26163", "label": "Wayne County, MI",
     "assessor_dir": DEWEY / "wayne-philadelphia-st-louis-2004-2025-history",
     "recorder_dir": _RECORDER_WPS, "berry_unit": "Detroit, MI", "role": "external"},
    {"key": "philadelphia", "fips": "42101", "label": "Philadelphia County, PA",
     "assessor_dir": DEWEY / "wayne-philadelphia-st-louis-2004-2025-history",
     "recorder_dir": _RECORDER_WPS, "berry_unit": "Philadelphia, PA", "role": "external"},
    {"key": "st_louis_county", "fips": "29189", "label": "St. Louis County, MO",
     "assessor_dir": DEWEY / "wayne-philadelphia-st-louis-2004-2025-history",
     "recorder_dir": _RECORDER_WPS, "berry_unit": "St. Louis County, MO", "role": "external"},
    {"key": "allegheny", "fips": "42003", "label": "Allegheny County, PA",
     # Canonical source resolved 2026-09-03 (Step 1): the narrow
     # "2016-2025" folder is truncated to ASSESSORHISTORYYEAR>=2019 and has
     # 1 unreadable shard. The "2006-2025" folder is a coherent same-product
     # superset (100% key overlap, 100% feature agreement on the overlap,
     # identical missingness, 0 unreadable shards) -- see
     # audits/history_source_resolution.yaml. Sale window unchanged (2016-2025).
     "assessor_dir": DEWEY / "allegheny-county-2006-2025-all-features",
     "recorder_dir": _RECORDER_10, "berry_unit": None, "role": "external"},
    {"key": "maricopa", "fips": "04013", "label": "Maricopa County, AZ",
     "assessor_dir": DEWEY / "maricopa-county-2016-2025-all-features",
     "recorder_dir": _RECORDER_10, "berry_unit": None, "role": "external"},
    {"key": "king", "fips": "53033", "label": "King County, WA",
     "assessor_dir": DEWEY / "king-county-2015-2025-all-features",
     "recorder_dir": _RECORDER_10, "berry_unit": None, "role": "external"},
    {"key": "miami_dade", "fips": "12086", "label": "Miami-Dade County, FL",
     "assessor_dir": DEWEY / "miami-dade-county-2006-2025-all-features",
     "recorder_dir": _RECORDER_10, "berry_unit": None, "role": "external"},
    {"key": "middlesex", "fips": "25017", "label": "Middlesex County, MA",
     "assessor_dir": DEWEY / "middlesex-county-2006-2025-all-features",
     "recorder_dir": _RECORDER_10, "berry_unit": None, "role": "external"},
    {"key": "cook", "fips": "17031", "label": "Cook County, IL",
     # Canonical source resolved 2026-09-03 (Step 1): the narrow
     # "2016-2025" folder is truncated to ASSESSORHISTORYYEAR>=2019. The
     # "2006-2025" folder is a coherent same-product superset (100% key
     # overlap, 100% feature agreement on the overlap, identical missingness)
     # -- see audits/history_source_resolution.yaml. Sale window unchanged.
     "assessor_dir": DEWEY / "cook-2006-2025-all-features",
     "recorder_dir": _RECORDER_10, "berry_unit": None, "role": "bridge_reference"},
)
JURISDICTION_BY_KEY = {j["key"]: j for j in JURISDICTIONS}
FIPS = {j["key"]: j["fips"] for j in JURISDICTIONS}
ST_LOUIS_CITY_FIPS = "29510"

# 3-county pilot per the staged-compute decision.
PILOT_KEYS = ("wayne", "philadelphia", "cook")
EXTERNAL_KEYS = tuple(j["key"] for j in JURISDICTIONS if j["role"] == "external")
ALL_KEYS = tuple(j["key"] for j in JURISDICTIONS)

# Reuse the v2/v3 cache column lists verbatim -- scientifically unchanged.
from analysis.berry_attom_validation_v2.scripts.v2_common import (  # noqa: E402
    HISTORY_CACHE_COLUMNS,
    RECORDER_CACHE_COLUMNS,
)

SEED = 2025
TEST_FRACTION = 0.20  # forward = calendar 2025 once temporal_design.yaml is frozen (Step 3)
MIN_SALE_PRICE = 50_000.0
SALE_WINDOW = ("2016-01-01", "2025-12-31")
N_BOOTSTRAP = 200

# Legacy bridge only -- see cohort/residential_code_mapping.yaml once the ATTOM
# code list is supplied (Phase B, gated). Never treated as semantic here.
LEGACY_385_ONLY = ("385",)

ASSESSMENT_VALUE_COLUMNS = {
    "TAXASSESSEDVALUEIMPROVEMENTS", "TAXASSESSEDVALUELAND", "TAXASSESSEDVALUETOTAL",
    "TAXMARKETVALUEIMPROVEMENTS", "TAXMARKETVALUELAND", "TAXMARKETVALUETOTAL", "TAXBILLEDAMOUNT",
    "PREVIOUSASSESSEDVALUE",
}

# CCAO reference candidate-region endpoints (raw rho), frozen in paper_v12.tex.
# Converted to rho_tilde = rho * Vy_CCAO in the objective-scaling audit (Step 1),
# NOT via plan_rho_grid's internal A = Var(baseline predictions) normalizer --
# see protocol_external_benchmark_v1.yaml `rho_normalization` for the distinction.
CCAO_DIRECT_RAW_RHO = {"activity": 0.355648, "guardrail": 2.559548}
CCAO_SURROGATE_RAW_RHO = {"activity": 0.202359, "guardrail": 2.222996}

# CCAO reference Vy used ONLY to set the initial normalized rho_tilde grid
# span -- never to select a jurisdiction, a deployment rho, or any candidate-
# region endpoint (those are computed independently per jurisdiction on this
# benchmark's own data). Computed from CCAO's own frozen fold-7 VALIDATION
# block (output/robust_rolling_origin_cv_2025/predictions/.../fold_id=7,
# y_true_log, n=33,859, ddof=0) as a fast, defensible proxy for the fold's
# training-block Vy -- the two are drawn from adjacent chronological windows
# of the same population and are expected to be very close; this benchmark's
# own protocol only requires the initial grid to be "generously wide", not an
# exact reproduction of CCAO's internal training variance.
CCAO_REFERENCE_VY = 0.4993351642786339
CCAO_REFERENCE_VY_SOURCE = (
    "output/robust_rolling_origin_cv_2025/predictions/data_id=d2b6dd81ad6ca613/"
    "split_id=a72eb9f19bef3733/fold_id=7 (validation block, y_true_log, ddof=0, n=33859)"
)

# Normalized rho_tilde grid (Step 6). Span = 2 log-decades below the smaller of
# the two families' reference activity points, to 2 log-decades above the
# larger of their reference guardrail points -- one shared, generously wide
# grid for both families, rounded to clean bounds.
#   Direct:    activity rho_tilde = 0.355648 * Vy = 0.1776; guardrail = 1.278
#   Surrogate: activity rho_tilde = 0.202359 * Vy = 0.1010; guardrail = 1.110
#   -> span [min(0.1776,0.1010)*1e-2, max(1.278,1.110)*1e2] = [0.00101, 127.8]
#   -> rounded to [1e-3, 150] for a clean, documented boundary.
NORMALIZED_RHO_GRID_MIN = 1e-3
NORMALIZED_RHO_GRID_MAX = 150.0
NORMALIZED_RHO_GRID_POINTS = 33
NORMALIZED_RHO_GRID_EXTENSION_DECADES = 1.0  # predeclared one-decade CV-only extension step


def normalized_rho_tilde_grid(include_zero: bool = True, extra_decades: float = 0.0):
    """The frozen zero + ~33 log-spaced positive rho_tilde grid (Step 6).

    `extra_decades` implements the predeclared one-decade CV-only grid
    extension (Step 6/QA gate) when an endpoint is boundary-censored; it must
    never be used to chase a favorable or unfavorable outcome.
    """
    import numpy as np

    hi = NORMALIZED_RHO_GRID_MAX * (10.0 ** extra_decades)
    lo = NORMALIZED_RHO_GRID_MIN / (10.0 ** extra_decades) if extra_decades else NORMALIZED_RHO_GRID_MIN
    positive = np.geomspace(lo, hi, NORMALIZED_RHO_GRID_POINTS)
    return np.concatenate([[0.0], positive]) if include_zero else positive

# Canonical-objective argument contract (Direct/Surrogate). Any other
# combination falls through to the legacy exploratory objective with gradient
# flooring -- see soft_constrained_models/boosting_models.py.
DIRECT_CANONICAL_KWARGS = {"ratio_mode": "diff", "match_native_init": True}
SURROGATE_CANONICAL_KWARGS = {
    "ratio_mode": "diff", "weighting_proxy_mode": "identity", "match_native_init": True,
}


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def population_variance(series) -> float:
    """ddof=0 variance, the convention used throughout this benchmark's rho
    normalization (Vy_T) and NMSE denominators. Never confused with pandas'
    default ddof=1."""
    import numpy as np

    arr = np.asarray(series, dtype=float)
    return float(np.mean((arr - arr.mean()) ** 2))


def chronological_splits(n: int, test_fraction: float = TEST_FRACTION) -> int:
    """Index of the first forward-period row under a simple fraction split.
    Superseded by the frozen calendar-year rule in temporal_design.yaml once
    Step 3 completes; kept here only for early smoke checks."""
    split = int(n * (1 - test_fraction))
    if not (1 <= split < n):
        raise ValueError(f"sample too small for chronological split: n={n}")
    return split
