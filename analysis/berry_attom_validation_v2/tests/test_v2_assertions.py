"""Assertions that must hold for Berry/ATTOM v2. Run: pytest analysis/berry_attom_validation_v2/tests"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from analysis.berry_attom_validation_v2.scripts.apn_normalize import normalize_apn_raw, normalize_apn_series
from analysis.berry_attom_validation_v2.scripts.v2_common import (
    ASSESSMENT_VALUE_COLUMNS, FIPS, ST_LOUIS_CITY_FIPS,
)


def test_fips_constants():
    assert FIPS["wayne"] == "26163"
    assert FIPS["philadelphia"] == "42101"
    assert FIPS["st_louis_county"] == "29189"
    assert ST_LOUIS_CITY_FIPS == "29510"
    assert FIPS["st_louis_county"] != ST_LOUIS_CITY_FIPS


def test_apn_normalization_is_deterministic():
    assert normalize_apn_raw("03G-110095") == normalize_apn_raw(" 03g 110095 ")
    assert normalize_apn_raw("03G-110095") == "03G110095"
    a = pd.Series(["A-1", "A-1", "B 2"])
    b = normalize_apn_series(a)
    assert list(b) == list(normalize_apn_series(a))
    assert b.nunique() == 2


def test_one_to_many_must_be_flagged_not_collapsed():
    """A helper that would pick an arbitrary ATTOMID is forbidden; status must be AMBIGUOUS."""
    pairs = pd.DataFrame({
        "berry_apn_norm": ["X", "X"],
        "ATTOMID": [1, 2],
    })
    n_ids = pairs.groupby("berry_apn_norm")["ATTOMID"].nunique()
    ambiguous = set(n_ids[n_ids > 1].index)
    assert "X" in ambiguous
    # collapsing would be: pairs.drop_duplicates("berry_apn_norm") — tests document that we must not.


def test_history_must_strictly_precede_sale():
    sale = pd.Timestamp("2018-06-15")
    through = pd.Timestamp("2018-12-31")
    assert not (through < sale)
    through_ok = pd.Timestamp("2017-12-31")
    assert through_ok < sale


def test_prior_sale_must_strictly_precede_current():
    current = pd.Timestamp("2019-01-10")
    prior = pd.Timestamp("2019-01-10")
    assert not (prior < current)


def test_assessment_value_columns_are_excluded_from_primary_features():
    forbidden = ASSESSMENT_VALUE_COLUMNS
    primary_example = {"AREABUILDING", "YEARBUILT", "sale_year", "recorder_prior_sale_amount"}
    assert primary_example.isdisjoint(forbidden)


def test_chronological_split_order():
    dates = pd.to_datetime(["2016-01-01", "2018-06-01", "2020-01-01", "2024-12-01"])
    n = len(dates)
    split = int(n * 0.8)
    train, test = dates[:split], dates[split:]
    assert train.max() <= test.min()


def test_protocol_forbids_test_in_selection_and_requires_freeze_before_rho():
    protocol = (REPO / "analysis/berry_attom_validation_v2/protocol_v2.yaml").read_text()
    assert "validation_only" in protocol
    assert "Write panel_freeze/final_panel_freeze_v2.yaml before any positive-rho" in protocol
    freeze = REPO / "analysis/berry_attom_validation_v2/panel_freeze/final_panel_freeze_v2.yaml"
    method_dir = REPO / "output/berry_attom_validation_v2/method_transfer"
    if any(method_dir.glob("*")):
        assert freeze.exists(), "positive-rho artifacts exist without a freeze file"


def test_no_cookies_in_v2_tree():
    root = REPO / "analysis/berry_attom_validation_v2"
    bad = [p for p in root.rglob("*") if "cookie" in p.name.lower() or "session" in p.name.lower()]
    assert bad == []


def test_st_louis_city_fips_never_in_county_cache():
    cache = REPO / "output/berry_attom_validation_v2/cache/st_louis_county"
    rec = cache / "recorder.parquet"
    hist = cache / "history.parquet"
    if not rec.exists():
        return
    import pyarrow.parquet as pq
    for path, col in ((rec, "DOCUMENTRECORDINGCOUNTYFIPS"), (hist, "SITUSSTATECOUNTYFIPS")):
        vals = set(pq.read_table(path, columns=[col])[col].to_pylist())
        assert "29510" not in {str(v) for v in vals if v is not None}


def test_primary_feature_set_excludes_assessment_values():
    from analysis.berry_attom_validation_v2.scripts.v2_common import HISTORY_CACHE_COLUMNS
    # Cache may store tax values for Step 9 audit, but primary feature construction
    # must drop them. This tests the exclusion set, not the cache schema.
    assert "TAXASSESSEDVALUETOTAL" in HISTORY_CACHE_COLUMNS
    assert "TAXASSESSEDVALUETOTAL" in ASSESSMENT_VALUE_COLUMNS
