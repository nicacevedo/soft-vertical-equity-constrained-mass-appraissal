#!/usr/bin/env python3
"""Build a bounded Cook County ATTOM sale-price experiment.

The target is Recorder ``TRANSFERAMOUNT``.  Each transaction is joined to the
latest Assessor History record whose assessor year ended before the sale, so no
post-sale property characteristics enter the model.  The current Tax Assessor
extract supplies only a checked property-location crosswalk; time-varying
characteristics always come from Assessor History.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import yaml
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.motivation_utils import _build_time_block_bootstrap_indices, _compute_extended_metrics


ASSESSOR_DIR = ROOT / "data/dewey-downloads/cookcounty-2016-2025-all-features"
RECORDER_DIR = ROOT / "data/dewey-downloads/10-counties-recorder-2016-2025"
TAX_ASSESSOR_DIR = ROOT / "data/dewey-downloads/9-counties-tax-assessor-missingharris-anyyear"
ACS_DIR = ROOT / "data/CensusData/acs5"
OUTPUT_DIR = ROOT / "output/attom_recorder_sample"
COOK_FIPS = "17031"
LGBM_CONFIG_PATH = ROOT / "best_lgbm_baseline_configs.yaml"
RECORDER_COLUMNS = [
    "ATTOMID", "TRANSACTIONID", "DOCUMENTRECORDINGCOUNTYFIPS",
    "INSTRUMENTDATE", "RECORDINGDATE", "TRANSFERAMOUNT", "DOCUMENTTYPECODE", "ARMSLENGTHFLAG",
    "TRANSFERINFOMULTIPARCELFLAG",
]
RECORDER_PRIOR_COLUMNS = [
    "ATTOMID", "TRANSACTIONID", "DOCUMENTRECORDINGCOUNTYFIPS", "INSTRUMENTDATE", "RECORDINGDATE", "TRANSFERAMOUNT",
    "ARMSLENGTHFLAG", "TRANSFERINFOMULTIPARCELFLAG",
]
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


def clean_tract(series: pd.Series) -> pd.Series:
    """Return a six-digit Census tract code, leaving malformed values null."""
    value = series.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    value = value.where(value.str.fullmatch(r"\d{1,6}"), pd.NA)
    return value.str.zfill(6)


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    denominator = denominator.where(denominator.gt(0))
    return numerator / denominator


def read_transactions(files: list[Path], county_fips: str, arms_length_only: bool, single_parcel_only: bool) -> pd.DataFrame:
    """Read positive Cook County transfers and retain CCAO-like valid sales."""
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
    data = data.loc[data["ATTOMID"].notna() & data["sale_date"].notna() & data["sale_price"].gt(0)]
    if arms_length_only:
        data = data.loc[pd.to_numeric(data["ARMSLENGTHFLAG"], errors="coerce").eq(1)]
    if single_parcel_only:
        data = data.loc[pd.to_numeric(data["TRANSFERINFOMULTIPARCELFLAG"], errors="coerce").eq(0)]
    return data.drop_duplicates("TRANSACTIONID").sort_values("sale_date").reset_index(drop=True)


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
    return pd.merge_asof(
        left, right, left_on="sale_date", right_on="assessed_through", by="ATTOMID",
        direction="backward", allow_exact_matches=False,
    ).dropna(subset=["assessed_through"]).sort_values("sale_date").reset_index(drop=True)


def attach_recorder_prior_sales(
    data: pd.DataFrame, recorder_files: list[Path], county_fips: str, arms_length_only: bool, single_parcel_only: bool
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
    history["recorder_sale_date"] = pd.to_datetime(history["INSTRUMENTDATE"], errors="coerce").fillna(
        pd.to_datetime(history["RECORDINGDATE"], errors="coerce")
    )
    history["recorder_sale_price"] = pd.to_numeric(history["TRANSFERAMOUNT"], errors="coerce")
    history = history.loc[
        history["ATTOMID"].notna() & history["recorder_sale_date"].notna() & history["recorder_sale_price"].gt(0)
    ].copy()
    if arms_length_only:
        history = history.loc[pd.to_numeric(history["ARMSLENGTHFLAG"], errors="coerce").eq(1)]
    if single_parcel_only:
        history = history.loc[pd.to_numeric(history["TRANSFERINFOMULTIPARCELFLAG"], errors="coerce").eq(0)]
    history = history.drop_duplicates("TRANSACTIONID").sort_values(["recorder_sale_date", "ATTOMID"])

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

    result["recorder_prior_sale_count_all"] = 0
    result["recorder_prior_sale_count_3yr"] = 0
    result["recorder_prior_sale_count_5yr"] = 0
    for attomid, index in result.groupby("ATTOMID", dropna=True).groups.items():
        dates = history.loc[history["ATTOMID"].eq(attomid), "recorder_sale_date"].sort_values().to_numpy()
        if dates.size == 0:
            continue
        sale_dates = pd.to_datetime(result.loc[index, "sale_date"])
        all_counts = np.searchsorted(dates, sale_dates.to_numpy(), side="left")
        count_3yr = all_counts - np.searchsorted(dates, (sale_dates - pd.DateOffset(years=3)).to_numpy(), side="left")
        count_5yr = all_counts - np.searchsorted(dates, (sale_dates - pd.DateOffset(years=5)).to_numpy(), side="left")
        result.loc[index, "recorder_prior_sale_count_all"] = all_counts
        result.loc[index, "recorder_prior_sale_count_3yr"] = count_3yr
        result.loc[index, "recorder_prior_sale_count_5yr"] = count_5yr
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
    data = data.sort_values(["tax_assessor_last_updated", "tax_assessor_publication_date"], na_position="last")
    return data.drop_duplicates("ATTOMID", keep="last").reset_index(drop=True)


def attach_tax_assessor(data: pd.DataFrame, tax_assessor: pd.DataFrame) -> pd.DataFrame:
    """Attach the current location only when it agrees with the history record."""
    result = data.merge(tax_assessor, on="ATTOMID", how="left", validate="m:1", indicator="_tax_assessor_match")
    result["tax_assessor_matched"] = result.pop("_tax_assessor_match").eq("both")
    history_apns = pd.concat(
        [clean_identifier(result["PARCELNUMBERFORMATTED"]), clean_identifier(result["PARCELNUMBERPREVIOUS"])], axis=1,
    )
    current_apns = pd.concat(
        [clean_identifier(result["tax_assessor_apn"]), clean_identifier(result["tax_assessor_prior_apn"])], axis=1,
    )
    comparable_apns = history_apns.notna().any(axis=1) & current_apns.notna().any(axis=1)
    apn_consistent = pd.Series(pd.NA, index=result.index, dtype="boolean")
    apn_consistent.loc[comparable_apns] = [
        bool(set(history_apns.loc[index].dropna()) & set(current_apns.loc[index].dropna()))
        for index in result.index[comparable_apns]
    ]
    result["tax_assessor_apn_consistent"] = apn_consistent
    history_address = clean_identifier(result["PROPERTYADDRESSFULL"])
    current_address = clean_identifier(result["tax_assessor_address"])
    comparable_address = history_address.notna() & current_address.notna()
    address_consistent = pd.Series(pd.NA, index=result.index, dtype="boolean")
    address_consistent.loc[comparable_address] = history_address.loc[comparable_address].eq(current_address.loc[comparable_address])
    result["tax_assessor_address_consistent"] = address_consistent
    coordinate_valid = result["tax_assessor_coordinate_valid"].astype("boolean").fillna(False)
    attomid_unambiguous = ~result["tax_assessor_attomid_ambiguous"].astype("boolean").fillna(True)
    apn_not_conflicting = result["tax_assessor_apn_consistent"].astype("boolean").ne(False).fillna(True)
    address_not_conflicting = result["tax_assessor_address_consistent"].astype("boolean").ne(False).fillna(True)
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
    metrics.update(
        {
            "N": int(actual.size),
            "RMSE (log)": float(mean_squared_error(actual_log, predicted_log) ** 0.5),
            "MAE (log)": float(mean_absolute_error(actual_log, predicted_log)),
        }
    )
    return metrics


def feature_frame(data: pd.DataFrame, train_rows: int, include_prior_sale: bool) -> tuple[pd.DataFrame, list[str]]:
    """Create CCAO-like building, tax, location, and sale-time predictors."""
    features = pd.DataFrame(index=data.index)
    for column in NUMERIC_FEATURES:
        if column in data:
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
    market_total = numeric("TAXMARKETVALUETOTAL")
    assessed_total = numeric("TAXASSESSEDVALUETOTAL")
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
    for column in [
        "recorder_prior_sale_amount", "recorder_log_prior_sale_price", "recorder_prior_sale_age_years",
        "recorder_prior_sale_count_all", "recorder_prior_sale_count_3yr", "recorder_prior_sale_count_5yr",
    ]:
        if column in data:
            features[column] = pd.to_numeric(data[column], errors="coerce")
    if include_prior_sale and {"ASSESSORLASTSALEDATE", "ASSESSORLASTSALEAMOUNT"}.issubset(data.columns):
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
    keep = [column for column in features if features[column].nunique(dropna=True) > 1]
    return features[keep], [column for column in categorical if column in keep]


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


def load_lgbm_configs(path: Path, keys: str) -> dict[str, dict]:
    """Load the CCAO-selected LightGBM configurations without modifying them."""
    with path.open(encoding="utf-8") as file:
        configured = yaml.safe_load(file)["lgbm_baselines"]
    requested = [key.strip() for key in keys.split(",") if key.strip()]
    missing = [key for key in requested if key not in configured]
    if missing:
        raise ValueError(f"Unknown LightGBM configuration(s): {', '.join(missing)}")
    return {key: dict(configured[key]["lgbm_params"]) for key in requested}


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


def run_model(data: pd.DataFrame, test_fraction: float, validation_fraction: float, config_path: Path, config_keys: str, feature_sets: str, target_scales: str, n_bootstrap: int, bootstrap_block_freq: str, seed: int) -> tuple[dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Select by recent validation, refit on all training sales, then bootstrap test metrics."""
    split = int(len(data) * (1 - test_fraction))
    validation_split = int(split * (1 - validation_fraction))
    if validation_split < 1 or validation_split >= split or split >= len(data):
        raise ValueError("The matched sample is too small for the requested chronological splits.")
    target_log = np.log(data["sale_price"].astype(float))
    ##########################################
    # Temp: some plotting of the target
    # y_train = target.iloc[:validation_split]
    import matplotlib.pyplot as plt
    import seaborn as sns

    print(data["PROPERTYUSESTANDARDIZED"].unique())

    uses_to_keep = ["385", "386"] # "401", "361"

# import matplotlib.pyplot as plt
#     import seaborn as sns

#     uses_to_keep = ["385", "386", "401", "361"]
    
    # 1. Create a mask to filter only the desired property uses
    mask = data["PROPERTYUSESTANDARDIZED"].isin(uses_to_keep)
    
    # 2. Apply the mask to align both your x-data and hue data
    filtered_target_log = target_log[mask]
    filtered_hue = data.loc[mask, "PROPERTYUSESTANDARDIZED"]

    print(filtered_hue.unique())

    plt.figure(figsize=(6,4))

    # Create the hued histogram using the filtered data
    ax = sns.histplot(
        x=filtered_target_log, 
        hue=filtered_hue, 
        bins=100, 
        stat="percent",      # Changes y-axis from raw frequency to percentage
        common_norm=False,   # Normalizes each category independently so small categories aren't hidden
        multiple="layer",    # Overlapping layout for independent percentages
        palette="tab10",
        alpha=0.4            # Adds transparency to make overlapping layers readable
    )

    # Make the legend super tiny and place it outside the plot area
    sns.move_legend(
        ax, 
        "upper left", 
        bbox_to_anchor=(1.02, 1), # Pushes the legend just outside the right edge
        fontsize='xx-small',      # Makes the category labels super tiny
        title_fontsize='x-small', # Makes the legend title slightly larger than the labels
        frameon=False             # Removes the box around the legend for a cleaner look
    )

    # bbox_inches="tight" ensures the external legend is not cut off in the saved PDF
    plt.savefig("tmp/delete/histogram.pdf", dpi=600, bbox_inches="tight")
    plt.close()
    # exit()
    ##########################################
    configs = load_lgbm_configs(config_path, config_keys)
    variants = [name.strip() for name in feature_sets.split(",") if name.strip()]
    scales = [name.strip() for name in target_scales.split(",") if name.strip()]
    if set(variants) - {"ccao_like", "ccao_like_plus_prior_sale"}:
        raise ValueError("--feature-sets must contain ccao_like and/or ccao_like_plus_prior_sale.")
    if set(scales) - {"log", "raw"}:
        raise ValueError("--target-scales must contain log and/or raw.")
    candidate_rows = []
    candidate_count = len(variants) * len(scales) * len(configs)
    candidate_started_at = perf_counter()
    candidate_number = 0
    for feature_set in variants:
        features, categorical = feature_frame(data, validation_split, feature_set != "ccao_like")
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
    candidates = pd.DataFrame(candidate_rows).sort_values(["R2", "R2 (log)"], ascending=False).reset_index(drop=True)
    best = candidates.iloc[0]
    features, categorical = feature_frame(data, split, best.feature_set != "ccao_like")
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
        "selection": {"validation_fraction": validation_fraction, "feature_set": str(best.feature_set), "target_scale": str(best.target_scale), "lgbm_config": str(best.lgbm_config), "validation_r2": float(best["R2"]), "validation_r2_log": float(best["R2 (log)"])},
        "model": params,
        "features": list(features.columns),
        "categorical_features": categorical,
        "feature_coverage": feature_coverage(features, categorical),
        "train_metrics": score_predictions(train.sale_price, prediction[:split], train.sale_price),
        "test_metrics": score_predictions(test.sale_price, prediction[split:], train.sale_price),
        "test_bootstrap": {"n_bootstrap": n_bootstrap, "block_freq": bootstrap_block_freq, "metrics": bootstrap_summary.set_index("metric").to_dict(orient="index")},
    }
    predictions = data[["ATTOMID", "TRANSACTIONID", "DOCUMENTTYPECODE", "ARMSLENGTHFLAG", "TRANSFERINFOMULTIPARCELFLAG", "sale_date", "sale_price", "TAXYEARASSESSED"]].copy()
    predictions["split"] = np.where(np.arange(len(data)) < split, "train", "test")
    predictions["sale_log_price"] = target_log
    predictions["predicted_log_sale_price"] = prediction_log
    predictions["predicted_sale_price"] = prediction
    metrics_table = pd.DataFrame([{"split": "train", **report["train_metrics"]}, {"split": "test", **report["test_metrics"]}])
    return report, predictions, metrics_table, candidates, bootstrap_draws, bootstrap_summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assessor-dir", type=Path, default=ASSESSOR_DIR)
    parser.add_argument("--recorder-dir", type=Path, default=RECORDER_DIR)
    parser.add_argument("--tax-assessor-dir", type=Path, default=TAX_ASSESSOR_DIR)
    parser.add_argument("--acs-dir", type=Path, default=ACS_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--assessor-files", type=int, default=0, help="Assessor History shards to read; 0 reads all shards.")
    parser.add_argument("--recorder-files", type=int, default=16)
    parser.add_argument("--tax-assessor-files", type=int, default=0, help="Tax Assessor shards to read; 0 reads all shards.")
    parser.add_argument("--county-fips", default=COOK_FIPS)
    parser.add_argument("--test-fraction", type=float, default=0.20)
    parser.add_argument("--validation-fraction", type=float, default=0.10)
    parser.add_argument("--lgbm-config-path", type=Path, default=LGBM_CONFIG_PATH)
    parser.add_argument("--lgbm-config-keys", default="test_best_r2,cv_top1_r2,cv_top2_r2")
    parser.add_argument("--feature-sets", default="ccao_like,ccao_like_plus_prior_sale")
    parser.add_argument("--target-scales", default="log,raw")
    parser.add_argument("--n-bootstrap", type=int, default=200)
    parser.add_argument("--bootstrap-block-freq", default="M")
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--include-non-arms-length", action="store_true")
    parser.add_argument("--include-multi-parcel", action="store_true")
    args = parser.parse_args()

    started_at = perf_counter()
    def stage(message: str) -> None:
        print(f"[{perf_counter() - started_at:7.1f}s] {message}", flush=True)

    stage("Selecting input shards")
    assessor_files = files_or_sample(args.assessor_dir, "assessor-history_*.parquet", args.assessor_files)
    recorder_files = sample_files(args.recorder_dir, "recorder_*.parquet", args.recorder_files)
    recorder_history_files = files_or_sample(args.recorder_dir, "recorder_*.parquet", 0)
    tax_assessor_files = files_or_sample(args.tax_assessor_dir, "tax-assessor_*.parquet", args.tax_assessor_files)
    stage("Reading and filtering recorder transactions")
    transactions = read_transactions(
        recorder_files, args.county_fips, not args.include_non_arms_length, not args.include_multi_parcel,
    )
    stage(f"Retained {len(transactions):,} transactions; reading assessor history")
    history = read_history(assessor_files, transactions.ATTOMID, args.county_fips)
    stage("Matching assessor history and recorder prior sales")
    matched = match_history(transactions, history)
    matched = attach_recorder_prior_sales(
        matched, recorder_history_files, args.county_fips, not args.include_non_arms_length, not args.include_multi_parcel,
    )
    stage("Reading tax-assessor location crosswalk")
    tax_assessor = read_tax_assessor(tax_assessor_files, matched.ATTOMID, args.county_fips)
    matched = attach_tax_assessor(matched, tax_assessor)
    stage("Reading and attaching ACS features")
    matched = attach_acs(matched, read_acs(args.acs_dir, args.county_fips))
    stage(f"Fitting models on {len(matched):,} matched sales")
    report, predictions, metrics_table, candidates, bootstrap_draws, bootstrap_summary = run_model(
        matched, args.test_fraction, args.validation_fraction, args.lgbm_config_path,
        args.lgbm_config_keys, args.feature_sets, args.target_scales, args.n_bootstrap, args.bootstrap_block_freq, args.seed,
    )
    report["sample"] = {
        "county_fips": args.county_fips,
        "test_fraction": args.test_fraction,
        "arms_length_only": not args.include_non_arms_length,
        "single_parcel_only": not args.include_multi_parcel,
        "assessor_files": [file.name for file in assessor_files],
        "recorder_files": [file.name for file in recorder_files],
        "recorder_history_files": [file.name for file in recorder_history_files],
        "tax_assessor_files": [file.name for file in tax_assessor_files],
        "n_transactions_before_history": int(len(transactions)),
        "n_history_matches": int(len(matched)),
        "assessor_history_match_rate": float(len(matched) / len(transactions)) if len(transactions) else None,
        "recorder_prior_sale_rate": float(matched["recorder_prior_sale_amount"].notna().mean()),
        "tax_assessor_match_rate": float(matched["tax_assessor_matched"].mean()),
        "tax_assessor_usable_location_rate": float(matched["tax_assessor_location_usable"].mean()),
        "acs_match_rate": float(matched["acs_matched"].mean()),
    }

    stage("Writing outputs")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    matched.to_parquet(args.output_dir / "matched_sales.parquet", index=False)
    predictions.to_parquet(args.output_dir / "predictions.parquet", index=False)
    metrics_table.to_csv(args.output_dir / "metrics.csv", index=False)
    candidates.to_csv(args.output_dir / "validation_candidates.csv", index=False)
    bootstrap_draws.to_csv(args.output_dir / "test_bootstrap_draws.csv", index=False)
    bootstrap_summary.to_csv(args.output_dir / "test_bootstrap_summary.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    stage("Complete")
    print(json.dumps(report, indent=2))
    print(f"Wrote matched sales, predictions, and metrics to {args.output_dir}")


if __name__ == "__main__":
    main()
