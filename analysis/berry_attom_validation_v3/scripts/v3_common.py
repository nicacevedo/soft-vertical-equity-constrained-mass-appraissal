#!/usr/bin/env python3
"""Shared v3 paths. Does not import modeling code at module import beyond stdlib."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
ANALYSIS = REPO / "analysis" / "berry_attom_validation_v3"
V2 = REPO / "analysis" / "berry_attom_validation_v2"
OUTPUT = REPO / "output" / "berry_attom_validation_v3"
RECORDER_DIR = REPO / "data/dewey-downloads/wayne-philadelphia-st-louis-2003-2025-recorder"
HISTORY_DIR = REPO / "data/dewey-downloads/wayne-philadelphia-st-louis-2004-2025-history"
BERRY_RAW = REPO / "data/berry_cmf/raw"
SALE_VALIDATION_DICTIONARY = (
    REPO / "data/dewey-downloads/data_dictionaries/attom_recorder_residential_avm_sale_validation_dictionary.csv"
)
LGBM_CONFIG_PATH = REPO / "best_lgbm_baseline_configs.yaml"
PYTHON = "/home/nacevedo/.conda/envs/fairness_env/bin/python"
PARTITION = "sched_mit_sloan_batch_r8"

FIPS = {
    "wayne": "26163",
    "philadelphia": "42101",
    "st_louis_county": "29189",
}
ST_LOUIS_CITY_FIPS = "29510"

COUNTIES = (
    {"key": "wayne", "fips": "26163", "label": "Wayne County, MI", "berry_unit": "Detroit, MI",
     "berry_file": "detroit_mi_transactions.parquet"},
    {"key": "philadelphia", "fips": "42101", "label": "Philadelphia County, PA", "berry_unit": "Philadelphia, PA",
     "berry_file": "philadelphia_pa_transactions.parquet"},
    {"key": "st_louis_county", "fips": "29189", "label": "St. Louis County, MO", "berry_unit": "St. Louis County, MO",
     "berry_file": "st_louis_county_mo_transactions.parquet"},
)

SEED = 2025
TEST_FRACTION = 0.20
VALIDATION_FRACTION = 0.10
N_BOOTSTRAP = 200
MIN_SALE_PRICE = 50_000.0
PROPERTY_USE_CODES = ("385",)

# Philadelphia sensitivity cohort. The primary cohort keeps PROPERTYUSESTANDARDIZED
# 385 only, which drops 82% of Philadelphia's safe-history sales (208,508 -> 38,043)
# against ~15% in Wayne and St. Louis, because Philadelphia's Assessor History stock
# is coded 366 (5.36M rows) rather than 385 (2.68M). Dewey ships no
# PROPERTYUSESTANDARDIZED code dictionary, so no code may be *called* residential.
# The sensitivity cohort is therefore defined by published structural facts, never
# by a semantic guess and never by any price, sale, or model outcome. The same rule
# is evaluated for all three counties so it cannot be Philadelphia-specific special
# pleading; see feature_audit/<county>_property_use_profile.csv.
#
# Rule version 2. Version 1 required building area to be *present* and admitted
# codes 401 (all three counties) and 397 (St. Louis) whose median AREABUILDING is
# exactly 0 -- parcels with no building, i.e. land. A cohort described as carrying
# dwelling structure must not contain them, so a positive median area is now
# required. The trigger is that structural observation, not any model metric;
# version 1's validation results had already been produced and are preserved
# under *_rule_v1 rather than discarded. See feature_audit/*_property_use_profile.csv.
BROAD_RESIDENTIAL_RULE = {
    "rule_version": 2,
    "min_share_area_building_present": 0.50,
    "min_median_area_building": 1.0,
    "min_share_year_built_present": 0.50,
    "max_median_units_count": 4.0,
    "min_share_of_matched_rows": 0.005,
}
PROPERTY_USE_SET_NAMES = ("primary_385", "broad_residential")
SALE_WINDOW = ("2016-01-01", "2025-12-31")
STL_LINKAGE_START = "2005-01-01"
STL_LINKAGE_END = "2019-12-31"

ASSESSMENT_VALUE_COLUMNS = {
    "TAXASSESSEDVALUEIMPROVEMENTS", "TAXASSESSEDVALUELAND", "TAXASSESSEDVALUETOTAL",
    "TAXMARKETVALUEIMPROVEMENTS", "TAXMARKETVALUELAND", "TAXMARKETVALUETOTAL", "TAXBILLEDAMOUNT",
    "PREVIOUSASSESSEDVALUE",
}

UNIQUE_APN_STATUSES = ("EXACT_RAW_APN", "EXACT_NORMALIZED_APN", "EXACT_PREVIOUS_APN")


SURROGATE_NOISE_FLOOR = 0.01
SURROGATE_REVERSAL_FRAC = 0.05
SURROGATE_GRID_MIN_RHO = 1e-3
SURROGATE_GRID_POINTS = 25
SURROGATE_GRID_MAX_MULTIPLE_OF_DIRECT = 4.0
# Bottom fifth of the log grid. At these rho the penalty is negligible against
# the county's own Direct anchors (smallest Direct rho is 0.47-0.88, so rho<=1e-2
# is under 2% of it), which makes the reductions measured there an estimate of
# pure validation noise rather than of any mechanism.
SURROGATE_NOISE_PROBE_POINTS = 5
SURROGATE_NOISE_MULTIPLE = 1.5


def estimate_surrogate_noise_floor(
    achieved,
    probe_points: int = SURROGATE_NOISE_PROBE_POINTS,
    multiple: float = SURROGATE_NOISE_MULTIPLE,
    min_floor: float = SURROGATE_NOISE_FLOOR,
) -> float:
    """Noise amplitude of the achieved-reduction curve, from its inactive tail.

    A fixed floor cannot work here: the observed low-rho noise envelope is 0.017
    in St. Louis, 0.028 in Philadelphia and 0.042 in Wayne, so the 0.01 constant
    used in passes 1 and 2 sat *below* the noise in every county. That let a
    wiggle open or close a branch -- Wayne's branch was cut at 0.042 achieved
    reduction while the real mechanism rises monotonically to 0.639, and
    Philadelphia's recorded "reversal" of 0.023 was smaller than its own 0.028
    noise envelope.

    Scaling the floor to the measured envelope implements the intended rule (the
    first branch where the mechanism actually changes in the intended direction)
    rather than changing it. Uses validation-block reductions only.
    """
    probe = [float(r) for r in list(achieved)[:probe_points] if r == r and abs(r) != float("inf")]
    if not probe:
        return min_floor
    return max(min_floor, multiple * max(abs(min(probe)), abs(max(probe))))


def surrogate_rho_grid(max_direct_rho: float):
    """Pre-test rho grid for Surrogate calibration.

    Declared rule (protocol_v3.yaml surrogate_calibration): span
    SURROGATE_GRID_MIN_RHO to SURROGATE_GRID_MAX_MULTIPLE_OF_DIRECT times the
    county's largest Direct rho, which is itself mapped from the pretest block
    only. Pass 1 used a fixed 1e-6..1e2 grid whose ceiling sat below the Direct
    97% anchor, so strong-penalty targets were reported UNATTAINED when the grid
    simply stopped early.
    """
    import numpy as np

    top = float(max_direct_rho) * SURROGATE_GRID_MAX_MULTIPLE_OF_DIRECT
    if not np.isfinite(top) or top <= SURROGATE_GRID_MIN_RHO:
        raise ValueError(f"unusable max_direct_rho={max_direct_rho!r}")
    return np.geomspace(SURROGATE_GRID_MIN_RHO, top, SURROGATE_GRID_POINTS)


def first_branch_calibrate(
    rhos,
    achieved,
    targets=(0.10, 0.25, 0.50, 0.67, 0.80, 0.90, 0.97),
    noise_floor: float | None = None,
    reversal_frac: float = SURROGATE_REVERSAL_FRAC,
):
    """Increase rho geometrically; keep the first contiguous intended-direction branch.

    A branch opens only once the achieved first-order reduction clears
    ``noise_floor``, and closes only on a *material* reversal (a drop of more
    than ``max(noise_floor, reversal_frac * |prev|)``). Pass 1 used a bare
    ``red + 1e-8 < prev`` test with no opening floor, so a noise-level positive
    reduction at negligible rho could open a one-point branch that the next
    point immediately closed -- which is what made every St. Louis target
    UNATTAINED.

    ``noise_floor=None`` (the default) estimates the floor from the curve's own
    inactive low-rho tail via ``estimate_surrogate_noise_floor``. Passing a fixed
    number reproduces the earlier passes and is used only by tests.

    ``branch_terminated_by`` records why the branch ended, so an UNATTAINED row
    distinguishes a genuine bend (MATERIAL_REVERSAL) from running out of grid
    (GRID_CEILING), a failed fit (FIT_FAILURE), or never clearing the noise
    floor at all (NEVER_STARTED).

    Lives here so tests can import it without loading LightGBM (some Sloan
    batch nodes cannot dlopen the fairness_env lib_lightgbm.so).
    """
    import numpy as np
    import pandas as pd

    if noise_floor is None:
        noise_floor = estimate_surrogate_noise_floor(achieved)
    branch_rho, branch_red = [], []
    started = False
    prev = -np.inf
    terminated_by = None
    for rho, red in zip(rhos, achieved):
        if not np.isfinite(red):
            if started:
                terminated_by = "FIT_FAILURE"
                break
            continue
        if not started:
            if red <= noise_floor:
                continue
            started = True
            prev = float(red)
            branch_rho.append(float(rho))
            branch_red.append(float(red))
            continue
        if red < prev - max(noise_floor, reversal_frac * abs(prev)):
            terminated_by = "MATERIAL_REVERSAL"
            break
        branch_rho.append(float(rho))
        branch_red.append(float(red))
        prev = float(red)
    if terminated_by is None:
        terminated_by = "GRID_CEILING" if started else "NEVER_STARTED"

    br = pd.DataFrame({"rho": branch_rho, "achieved_reduction": branch_red})
    branch_max = float(br["achieved_reduction"].max()) if len(br) else float("nan")
    rows = []
    for t in targets:
        rec = {
            "requested_reduction": t,
            "rho": np.nan,
            "status": "UNATTAINED",
            "unattained_reason": "BRANCH_TOO_SHORT" if len(br) < 2 else terminated_by,
        }
        if len(br) >= 2 and t <= branch_max + 1e-8:
            for i in range(len(br) - 1):
                a, b = br["achieved_reduction"].iloc[i], br["achieved_reduction"].iloc[i + 1]
                lo, hi = min(a, b), max(a, b)
                if lo - 1e-12 <= t <= hi + 1e-12 and abs(b - a) > 1e-12:
                    w = (t - a) / (b - a)
                    rec = {
                        "requested_reduction": t,
                        "rho": float(
                            br["rho"].iloc[i] + w * (br["rho"].iloc[i + 1] - br["rho"].iloc[i])
                        ),
                        "status": "interpolated_first_branch",
                        "unattained_reason": "",
                    }
                    break
        rec["branch_terminated_by"] = terminated_by
        rec["noise_floor_used"] = float(noise_floor)
        rec["branch_n_points"] = int(len(br))
        rec["branch_max_reduction"] = branch_max
        rows.append(rec)
    return pd.DataFrame(rows), br

# Reuse v2 column lists (scientifically unchanged).
from analysis.berry_attom_validation_v2.scripts.v2_common import (  # noqa: E402
    HISTORY_CACHE_COLUMNS,
    RECORDER_CACHE_COLUMNS,
)


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")


def chronological_splits(n: int) -> tuple[int, int]:
    split = int(n * (1 - TEST_FRACTION))
    validation_split = int(split * (1 - VALIDATION_FRACTION))
    if not (1 <= validation_split < split < n):
        raise ValueError(f"sample too small for chronological splits: n={n}")
    return split, validation_split


def lr_feature_groups(train, categorical: list[str], max_levels: int = 32):
    """LR columns from the training/development prefix only.

    High-cardinality or non-numeric fields are dropped, never sent to
    median imputation (Wayne/St. Louis city names exceeded 32 levels).
    """
    import pandas as pd

    cats, numeric, dropped = [], [], []
    for column in train.columns:
        series = train[column]
        nuniq = int(series.nunique(dropna=True))
        cat_like = column in categorical or not pd.api.types.is_numeric_dtype(series)
        if cat_like:
            if 1 < nuniq <= max_levels:
                cats.append(column)
            else:
                dropped.append(column)
        else:
            numeric.append(column)
    return numeric, cats, dropped
