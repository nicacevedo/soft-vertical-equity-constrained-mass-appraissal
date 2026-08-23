#!/usr/bin/env python3
"""Prepare ATTOM Assessor History for a CCAO RES-AVM-style experiment.

Purpose
-------
This script uses only ATTOM Assessor History. It creates one CCAO-like modeling
row per property by selecting the latest available assessor snapshot, then uses
that property's assessor-reported last sale amount/date as the prediction target.

This is the closest available analogue to the CCAO workflow because the CCAO
training sample uses historical sales together with a common property-data
vintage. It is not a substitute for a verified Recorder transaction sample:
Assessor History does not provide enough information to verify arms-length,
non-distressed, single-parcel transactions.

Outputs
-------
1. A model-ready parquet with:
   - meta_sale_price and meta_sale_date
   - CCAO-like structural, coarse-location, and time predictors
   - audit/status-quo fields kept outside the predictor list
   - ind_pin_is_multicard=False and sv_is_outlier=False so the current project
     loader can read the file without code changes
2. A YAML file containing predictor and categorical-column lists.
3. CSV audit tables for sample construction, missingness, and property-use codes.

The script deliberately does NOT:
- use tax/assessment values as predictors;
- use owner names or mailing addresses;
- guess ATTOM property-use code meanings;
- winsorize the target or remove price tails;
- treat ATTOM code values 0 or 999 as missing without a codebook.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import yaml


# -----------------------------------------------------------------------------
# Raw ATTOM columns used by this preparation.
# Missing optional columns are tolerated; missing required columns are not.
# -----------------------------------------------------------------------------
REQUIRED_RAW_COLUMNS = [
    "ATTOMID",
    "ASSESSORLASTSALEAMOUNT",
    "ASSESSORLASTSALEDATE",
    "ASSESSORHISTORYYEAR",
    "PARCELNUMBERRAW",
    "SITUSSTATECOUNTYFIPS",
    "PROPERTYUSESTANDARDIZED",
]

NUMERIC_RAW_COLUMNS = [
    "AREABUILDING",
    "AREAGROSS",
    "AREALOTSF",
    "BATHCOUNT",
    "BATHPARTIALCOUNT",
    "BEDROOMSCOUNT",
    "PARKINGGARAGEAREA",
    "ROOMSATTICAREA",
    "ROOMSBASEMENTAREAFINISHED",
    "ROOMSBASEMENTAREAUNFINISHED",
    "ROOMSCOUNT",
    "STORIESCOUNT",
    "UNITSCOUNT",
    "YEARBUILT",
    "YEARBUILTEFFECTIVE",
    "ASSESSORHISTORYYEAR",
    "TAXYEARASSESSED",
    "TAXFISCALYEAR",
    "TAXASSESSEDVALUEIMPROVEMENTS",
    "TAXASSESSEDVALUELAND",
    "TAXASSESSEDVALUETOTAL",
    "TAXBILLEDAMOUNT",
    "TAXMARKETVALUEIMPROVEMENTS",
    "TAXMARKETVALUELAND",
    "TAXMARKETVALUETOTAL",
]

DATE_RAW_COLUMNS = [
    "ASSESSORLASTSALEDATE",
    "LASTOWNERSHIPTRANSFERDATE",
    "PUBLICATIONDATE",
]

CODE_RAW_COLUMNS = [
    "PROPERTYUSESTANDARDIZED",
    "STRUCTURESTYLE",
    "EXTERIOR1CODE",
    "FOUNDATION",
    "HVACCOOLINGDETAIL",
    "HVACHEATINGDETAIL",
    "PARKINGGARAGE",
    "FIREPLACE",
    "POOL",
    "PORCHCODE",
    "ROOFMATERIAL",
    "PROPERTYADDRESSZIP",
    "PROPERTYADDRESSCITY",
    "PROPERTYADDRESSCRRT",
    "SITUSSTATECOUNTYFIPS",
    "SITUSSTATECODE",
    "SITUSCOUNTY",
]

# Cook County township codes used in the CCAO project. For a Cook County PIN,
# the first two digits identify the township code.
VALID_COOK_TOWNSHIP_CODES = {
    *(str(i) for i in range(10, 40)),
    *(str(i) for i in range(70, 78)),
}

# Structural fields used to choose the most complete row when duplicate latest
# snapshots exist for the same ATTOMID.
SNAPSHOT_QUALITY_COLUMNS = [
    "AREABUILDING",
    "AREALOTSF",
    "YEARBUILT",
    "BATHCOUNT",
    "BEDROOMSCOUNT",
    "ROOMSCOUNT",
    "PROPERTYUSESTANDARDIZED",
    "STRUCTURESTYLE",
    "EXTERIOR1CODE",
    "HVACCOOLINGDETAIL",
    "HVACHEATINGDETAIL",
    "PROPERTYADDRESSZIP",
    "PROPERTYADDRESSCITY",
    "PROPERTYADDRESSCRRT",
]


# Final model columns. These are ATTOM analogues of the available CCAO blocks;
# names are intentionally explicit when an ATTOM field is not identical to a
# CCAO field.
NUMERIC_PREDICTORS = [
    "meta_sale_count_past_4_years",
    "char_yrblt",
    "char_yrblt_effective",
    "char_property_age",
    "char_effective_age",
    "char_bldg_sf",
    "char_gross_sf",
    "char_land_sf",
    "char_beds",
    "char_baths_total",
    "char_baths_partial",
    "char_rooms",
    "char_stories",
    "char_units",
    "char_garage_area",
    "char_attic_area",
    "char_bsmt_fin_area",
    "char_bsmt_unfin_area",
    "char_has_attic",
    "char_has_bsmt",
    "char_has_finished_bsmt",
    "time_sale_year",
    "time_sale_day",
    "time_sale_month_of_year",
    "time_sale_quarter_of_year_num",
    "time_sale_day_of_year",
    "time_sale_day_of_month",
    "time_sale_day_of_week",
]

CATEGORICAL_PREDICTORS = [
    "meta_township_code",
    "char_property_use_attom",
    "char_structure_style_attom",
    "char_ext_wall_attom",
    "char_foundation_attom",
    "char_cooling_attom",
    "char_heating_attom",
    "char_garage_type_attom",
    "char_fireplace_attom",
    "char_pool_attom",
    "char_porch_attom",
    "char_roof_material_attom",
    "loc_zip",
    "loc_city",
    "loc_carrier_route",
    "time_sale_quarter_of_year",
]

PREDICTOR_COLUMNS = NUMERIC_PREDICTORS + CATEGORICAL_PREDICTORS

# These fields are retained for auditing and optional comparisons, but are not
# predictors. The same-year ratio is only computed when assessment year equals
# sale year.
AUDIT_COLUMNS = [
    "meta_attomid",
    "meta_pin",
    "meta_assessor_history_year",
    "meta_tax_year_assessed",
    "meta_publication_date",
    "meta_property_use_raw",
    "meta_sale_price",
    "meta_sale_date",
    "target_log_sale_price",
    "split",
    "status_quo_market_value_total",
    "status_quo_assessed_value_total",
    "status_quo_market_value_land",
    "status_quo_market_value_improvements",
    "status_quo_assessed_value_land",
    "status_quo_assessed_value_improvements",
    "status_quo_tax_billed_amount",
    "status_quo_ratio_same_year",
    "status_quo_same_year_available",
    "ind_pin_is_multicard",
    "sv_is_outlier",
]


def _load_table(path: Path) -> pd.DataFrame:
    """Load parquet or CSV and normalize column names to uppercase."""
    suffix = path.suffix.lower()
    if suffix in {".parquet", ".pq"}:
        df = pd.read_parquet(path)
    elif suffix in {".csv", ".txt"}:
        df = pd.read_csv(path, low_memory=False)
    else:
        raise ValueError(f"Unsupported input format: {path.suffix}")

    df = df.copy()
    df.columns = [str(c).strip().upper() for c in df.columns]
    return df


def _require_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise KeyError(
            "The ATTOM file is missing required columns: " + ", ".join(missing)
        )


def _existing(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    return [c for c in columns if c in df.columns]


def _clean_string_codes(series: pd.Series) -> pd.Series:
    """Normalize text codes without guessing that 0 or 999 means missing."""
    out = series.astype("string").str.strip()
    return out.mask(out.eq(""), pd.NA)


def _to_numeric(df: pd.DataFrame, columns: Sequence[str]) -> None:
    for col in _existing(df, columns):
        df[col] = pd.to_numeric(df[col], errors="coerce")


def _to_dates(df: pd.DataFrame, columns: Sequence[str]) -> None:
    for col in _existing(df, columns):
        df[col] = pd.to_datetime(df[col], errors="coerce")


def _derive_township_from_pin(pin: pd.Series) -> pd.Series:
    """Derive Cook County township code from the first two PIN digits."""
    digits = pin.astype("string").str.replace(r"\D", "", regex=True)
    township = digits.str[:2]
    return township.where(township.isin(VALID_COOK_TOWNSHIP_CODES), pd.NA)


def _rolling_prior_count(
    frame: pd.DataFrame,
    *,
    group_col: str,
    date_col: str,
    years: int,
) -> pd.Series:
    """Count earlier sales in the same area during the prior `years` years.

    The interval is [sale_date - years, sale_date), so observations on the same
    date do not count one another. This avoids using current or future sales.
    """
    result = pd.Series(0, index=frame.index, dtype="int64")
    valid = frame[group_col].notna() & frame[date_col].notna()

    for _, idx in frame.loc[valid].groupby(group_col, sort=False).groups.items():
        idx = pd.Index(idx)
        dates = frame.loc[idx, date_col].sort_values()
        values = dates.to_numpy(dtype="datetime64[ns]")

        # Search each row's strictly earlier observations.
        upper = np.searchsorted(values, values, side="left")
        lower_dates = (
            dates - pd.DateOffset(years=years)
        ).to_numpy(dtype="datetime64[ns]")
        lower = np.searchsorted(values, lower_dates, side="left")
        counts = upper - lower
        result.loc[dates.index] = counts.astype("int64")

    return result


def _record_stage(stages: list[dict[str, object]], name: str, df: pd.DataFrame) -> None:
    stages.append(
        {
            "stage": name,
            "rows": int(df.shape[0]),
            "columns": int(df.shape[1]),
            "unique_attomid": (
                int(df["ATTOMID"].nunique(dropna=True))
                if "ATTOMID" in df.columns
                else np.nan
            ),
        }
    )


def prepare_attom_assessor_history(
    raw: pd.DataFrame,
    *,
    county_fips: str = "17031",
    min_sale_year: int = 2016,
    max_sale_year: int = 2024,
    single_family_codes: set[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Create the CCAO-like ATTOM modeling table and audit outputs."""
    stages: list[dict[str, object]] = []
    df = raw.copy()
    _record_stage(stages, "raw", df)

    # 1. Remove columns containing no observed value. This is safe because they
    # cannot carry information in this extract.
    df = df.dropna(axis=1, how="all").copy()
    _record_stage(stages, "drop_all_null_columns", df)

    _require_columns(df, REQUIRED_RAW_COLUMNS)

    # 2. Normalize types before filtering or sorting.
    _to_numeric(df, NUMERIC_RAW_COLUMNS + ["ATTOMID", "ASSESSORLASTSALEAMOUNT"])
    _to_dates(df, DATE_RAW_COLUMNS)
    for col in _existing(df, CODE_RAW_COLUMNS + ["PARCELNUMBERRAW"]):
        df[col] = _clean_string_codes(df[col])

    county_fips = str(county_fips).zfill(5)
    df["SITUSSTATECOUNTYFIPS"] = df["SITUSSTATECOUNTYFIPS"].str.zfill(5)

    # 3. Keep only the intended county. The uploaded Cook sample contains a few
    # inconsistent state/FIPS values, so county FIPS is the decisive filter.
    df = df.loc[df["SITUSSTATECOUNTYFIPS"].eq(county_fips)].copy()
    _record_stage(stages, "filter_county_fips", df)

    # 4. Choose one common, latest assessor snapshot per property. This mirrors
    # the CCAO design more closely than mixing several historical snapshots of
    # the same property in the same model.
    quality_cols = _existing(df, SNAPSHOT_QUALITY_COLUMNS)
    df["_snapshot_quality"] = df[quality_cols].notna().sum(axis=1)

    sort_cols = ["ATTOMID"]
    for col in ["ASSESSORHISTORYYEAR", "TAXYEARASSESSED", "PUBLICATIONDATE"]:
        if col in df.columns:
            sort_cols.append(col)
    sort_cols.append("_snapshot_quality")

    df = (
        df.sort_values(sort_cols, na_position="first")
        .drop_duplicates(subset=["ATTOMID"], keep="last")
        .copy()
    )
    _record_stage(stages, "latest_snapshot_per_attomid", df)

    # 5. The only usable target/date pair in this product is the assessor's last
    # sale amount/date. Require a positive price and a valid date. No further
    # price-tail deletion is applied because it could mechanically alter the
    # regressivity pattern under study.
    df["ASSESSORLASTSALEAMOUNT"] = pd.to_numeric(
        df["ASSESSORLASTSALEAMOUNT"], errors="coerce"
    )
    df["ASSESSORLASTSALEDATE"] = pd.to_datetime(
        df["ASSESSORLASTSALEDATE"], errors="coerce"
    )
    df = df.loc[
        df["ASSESSORLASTSALEAMOUNT"].gt(0)
        & df["ASSESSORLASTSALEDATE"].notna()
    ].copy()
    _record_stage(stages, "positive_sale_price_and_valid_date", df)

    sale_year = df["ASSESSORLASTSALEDATE"].dt.year
    df = df.loc[sale_year.between(min_sale_year, max_sale_year)].copy()
    _record_stage(stages, "filter_sale_year_range", df)

    # 6. Restrict to ATTOM single-family codes only when the verified code list
    # is supplied. The script never guesses the meaning of ATTOM codes.
    if single_family_codes:
        codes = {str(x).strip() for x in single_family_codes}
        df = df.loc[df["PROPERTYUSESTANDARDIZED"].isin(codes)].copy()
        _record_stage(stages, "filter_verified_single_family_codes", df)

    # 7. Remove duplicate parcel-sale observations that survive under different
    # ATTOMIDs or duplicate source records. Keep the most complete snapshot.
    # Use ATTOMID as a fallback when the raw parcel number is unavailable so
    # unrelated missing-PIN records are never collapsed together.
    df["_parcel_or_attomid"] = df["PARCELNUMBERRAW"].where(
        df["PARCELNUMBERRAW"].notna(),
        "ATTOMID:" + df["ATTOMID"].astype("Int64").astype("string"),
    )
    dedup_key = [
        "_parcel_or_attomid",
        "ASSESSORLASTSALEDATE",
        "ASSESSORLASTSALEAMOUNT",
    ]
    df = (
        df.sort_values(["_snapshot_quality"], na_position="first")
        .drop_duplicates(subset=dedup_key, keep="last")
        .copy()
    )
    _record_stage(stages, "deduplicate_parcel_sale", df)

    # 8. Convert only physically impossible zeros to missing. We do not convert
    # zero bedroom/bath/unit counts automatically because those fields can use
    # jurisdiction-specific coding and some residential subtypes permit zeros.
    for col in _existing(df, ["AREABUILDING", "AREAGROSS", "AREALOTSF"]):
        df[col] = df[col].where(df[col].gt(0), np.nan)

    sale_year = df["ASSESSORLASTSALEDATE"].dt.year.astype("float64")
    if "YEARBUILT" in df.columns:
        df["YEARBUILT"] = df["YEARBUILT"].where(
            df["YEARBUILT"].gt(0) & df["YEARBUILT"].le(sale_year), np.nan
        )
    if "YEARBUILTEFFECTIVE" in df.columns:
        valid_effective = (
            df["YEARBUILTEFFECTIVE"].gt(0)
            & df["YEARBUILTEFFECTIVE"].le(sale_year)
        )
        if "YEARBUILT" in df.columns:
            valid_effective &= (
                df["YEARBUILT"].isna()
                | df["YEARBUILTEFFECTIVE"].ge(df["YEARBUILT"])
            )
        df["YEARBUILTEFFECTIVE"] = df["YEARBUILTEFFECTIVE"].where(
            valid_effective, np.nan
        )

    # Negative structural counts or areas cannot be valid.
    nonnegative_cols = [
        "BATHCOUNT",
        "BATHPARTIALCOUNT",
        "BEDROOMSCOUNT",
        "PARKINGGARAGEAREA",
        "ROOMSATTICAREA",
        "ROOMSBASEMENTAREAFINISHED",
        "ROOMSBASEMENTAREAUNFINISHED",
        "ROOMSCOUNT",
        "STORIESCOUNT",
        "UNITSCOUNT",
    ]
    for col in _existing(df, nonnegative_cols):
        df[col] = df[col].where(df[col].ge(0), np.nan)

    # 9. Build the model table. Tax values remain audit outcomes and are not
    # included in the predictor lists.
    out = pd.DataFrame(index=df.index)
    out["meta_attomid"] = df["ATTOMID"].astype("Int64")
    out["meta_pin"] = df["PARCELNUMBERRAW"].astype("string")
    out["meta_township_code"] = _derive_township_from_pin(df["PARCELNUMBERRAW"])

    out["meta_sale_price"] = df["ASSESSORLASTSALEAMOUNT"].astype("float64")
    out["meta_sale_date"] = df["ASSESSORLASTSALEDATE"]
    out["target_log_sale_price"] = np.log(out["meta_sale_price"])

    out["meta_assessor_history_year"] = df.get("ASSESSORHISTORYYEAR")
    out["meta_tax_year_assessed"] = df.get("TAXYEARASSESSED")
    out["meta_publication_date"] = df.get("PUBLICATIONDATE")
    out["meta_property_use_raw"] = df["PROPERTYUSESTANDARDIZED"]

    # Structural numeric features.
    numeric_map = {
        "YEARBUILT": "char_yrblt",
        "YEARBUILTEFFECTIVE": "char_yrblt_effective",
        "AREABUILDING": "char_bldg_sf",
        "AREAGROSS": "char_gross_sf",
        "AREALOTSF": "char_land_sf",
        "BEDROOMSCOUNT": "char_beds",
        "BATHCOUNT": "char_baths_total",
        "BATHPARTIALCOUNT": "char_baths_partial",
        "ROOMSCOUNT": "char_rooms",
        "STORIESCOUNT": "char_stories",
        "UNITSCOUNT": "char_units",
        "PARKINGGARAGEAREA": "char_garage_area",
        "ROOMSATTICAREA": "char_attic_area",
        "ROOMSBASEMENTAREAFINISHED": "char_bsmt_fin_area",
        "ROOMSBASEMENTAREAUNFINISHED": "char_bsmt_unfin_area",
    }
    for raw_col, final_col in numeric_map.items():
        out[final_col] = df[raw_col] if raw_col in df.columns else np.nan

    out["char_property_age"] = (
        out["meta_sale_date"].dt.year.astype("float64") - out["char_yrblt"]
    )
    out["char_effective_age"] = (
        out["meta_sale_date"].dt.year.astype("float64")
        - out["char_yrblt_effective"]
    )

    out["char_has_attic"] = out["char_attic_area"].gt(0).astype("int8")
    out["char_has_bsmt"] = (
        out[["char_bsmt_fin_area", "char_bsmt_unfin_area"]]
        .fillna(0)
        .sum(axis=1)
        .gt(0)
        .astype("int8")
    )
    out["char_has_finished_bsmt"] = (
        out["char_bsmt_fin_area"].fillna(0).gt(0).astype("int8")
    )

    # Structural categorical features. Raw ATTOM codes are kept as unordered
    # strings; they must not be interpreted as numeric magnitudes.
    categorical_map = {
        "PROPERTYUSESTANDARDIZED": "char_property_use_attom",
        "STRUCTURESTYLE": "char_structure_style_attom",
        "EXTERIOR1CODE": "char_ext_wall_attom",
        "FOUNDATION": "char_foundation_attom",
        "HVACCOOLINGDETAIL": "char_cooling_attom",
        "HVACHEATINGDETAIL": "char_heating_attom",
        "PARKINGGARAGE": "char_garage_type_attom",
        "FIREPLACE": "char_fireplace_attom",
        "POOL": "char_pool_attom",
        "PORCHCODE": "char_porch_attom",
        "ROOFMATERIAL": "char_roof_material_attom",
        "PROPERTYADDRESSZIP": "loc_zip",
        "PROPERTYADDRESSCITY": "loc_city",
        "PROPERTYADDRESSCRRT": "loc_carrier_route",
    }
    for raw_col, final_col in categorical_map.items():
        if raw_col in df.columns:
            out[final_col] = _clean_string_codes(df[raw_col])
        else:
            out[final_col] = pd.Series(pd.NA, index=out.index, dtype="string")

    # Time features copied from the CCAO design when they can be constructed
    # exactly from the sale date.
    date = out["meta_sale_date"]
    out["time_sale_year"] = date.dt.year.astype("int16")
    out["time_sale_day"] = date.dt.day.astype("int8")
    out["time_sale_month_of_year"] = date.dt.month.astype("int8")
    out["time_sale_quarter_of_year_num"] = date.dt.quarter.astype("int8")
    out["time_sale_quarter_of_year"] = (
        "Q" + date.dt.quarter.astype("string")
    ).astype("string")
    out["time_sale_day_of_year"] = date.dt.dayofyear.astype("int16")
    out["time_sale_day_of_month"] = date.dt.day.astype("int8")
    out["time_sale_day_of_week"] = date.dt.dayofweek.astype("int8")

    # Use the finest reliable available area proxy for a past-sales count:
    # carrier route when present, otherwise ZIP. This feature uses only sales
    # strictly before the current observation.
    out["_market_area"] = out["loc_carrier_route"].where(
        out["loc_carrier_route"].notna(), out["loc_zip"]
    )
    out["_market_area"] = (
        county_fips + "|" + out["_market_area"].astype("string")
    ).where(out["_market_area"].notna(), pd.NA)
    out["meta_sale_count_past_4_years"] = _rolling_prior_count(
        out,
        group_col="_market_area",
        date_col="meta_sale_date",
        years=4,
    )
    out = out.drop(columns=["_market_area"])

    # Recalculate ages after the time columns exist and remove impossible
    # negative values defensively.
    out["char_property_age"] = (
        out["time_sale_year"].astype("float64") - out["char_yrblt"]
    ).where(lambda s: s.ge(0), np.nan)
    out["char_effective_age"] = (
        out["time_sale_year"].astype("float64") - out["char_yrblt_effective"]
    ).where(lambda s: s.ge(0), np.nan)

    # Audit/status-quo fields. These are intentionally excluded from X.
    audit_map = {
        "TAXMARKETVALUETOTAL": "status_quo_market_value_total",
        "TAXASSESSEDVALUETOTAL": "status_quo_assessed_value_total",
        "TAXMARKETVALUELAND": "status_quo_market_value_land",
        "TAXMARKETVALUEIMPROVEMENTS": "status_quo_market_value_improvements",
        "TAXASSESSEDVALUELAND": "status_quo_assessed_value_land",
        "TAXASSESSEDVALUEIMPROVEMENTS": "status_quo_assessed_value_improvements",
        "TAXBILLEDAMOUNT": "status_quo_tax_billed_amount",
    }
    for raw_col, final_col in audit_map.items():
        out[final_col] = df[raw_col] if raw_col in df.columns else np.nan

    same_year = (
        pd.to_numeric(out["meta_tax_year_assessed"], errors="coerce")
        == out["time_sale_year"]
    )
    valid_same_year_ratio = (
        same_year
        & pd.to_numeric(out["status_quo_market_value_total"], errors="coerce").gt(0)
    )
    out["status_quo_same_year_available"] = valid_same_year_ratio
    out["status_quo_ratio_same_year"] = np.where(
        valid_same_year_ratio,
        pd.to_numeric(out["status_quo_market_value_total"], errors="coerce")
        / out["meta_sale_price"],
        np.nan,
    )

    # Exact temporal blocks currently used in the paper/project.
    out["split"] = np.select(
        [
            out["time_sale_year"].between(min_sale_year, 2022),
            out["time_sale_year"].eq(2023),
            out["time_sale_year"].eq(2024),
        ],
        ["development", "test", "assessment"],
        default="outside_protocol",
    )

    # Compatibility columns expected by the existing loader. They are not
    # genuine ATTOM flags, so they are set to False only after this script's
    # explicit filtering and deduplication.
    out["ind_pin_is_multicard"] = False
    out["sv_is_outlier"] = False

    # Enforce expected string/categorical types.
    for col in CATEGORICAL_PREDICTORS + ["meta_pin", "meta_property_use_raw", "split"]:
        if col in out.columns:
            out[col] = out[col].astype("string")

    # Stable chronological order for downstream rolling-origin splitting.
    out = out.sort_values(["meta_sale_date", "meta_attomid"]).reset_index(drop=True)
    _record_stage(stages, "final_model_table", out.rename(columns={"meta_attomid": "ATTOMID"}))

    # Audit tables.
    stage_df = pd.DataFrame(stages)
    missingness_df = pd.DataFrame(
        {
            "column": out.columns,
            "missing_count": [int(out[c].isna().sum()) for c in out.columns],
            "missing_share": [float(out[c].isna().mean()) for c in out.columns],
            "n_unique": [int(out[c].nunique(dropna=True)) for c in out.columns],
            "dtype": [str(out[c].dtype) for c in out.columns],
        }
    ).sort_values(["missing_share", "column"], ascending=[False, True])

    property_use_df = (
        out["meta_property_use_raw"]
        .value_counts(dropna=False)
        .rename_axis("property_use_code")
        .reset_index(name="rows")
    )
    property_use_df["share"] = property_use_df["rows"] / max(len(out), 1)

    return out, stage_df, missingness_df, property_use_df


def _write_predictor_yaml(path: Path) -> None:
    payload = {
        "model": {
            "predictor": {
                "all": PREDICTOR_COLUMNS,
                "categorical": CATEGORICAL_PREDICTORS,
                "numeric": NUMERIC_PREDICTORS,
            }
        },
        "target_column": "meta_sale_price",
        "date_column": "meta_sale_date",
        "notes": [
            "Tax and assessment values are audit outcomes, not predictors.",
            "ATTOM code columns are categorical strings.",
            "Imputation and category fitting must be learned on each training fold only.",
        ],
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _parse_codes(raw: str | None) -> set[str] | None:
    if raw is None or not raw.strip():
        return None
    return {x.strip() for x in raw.split(",") if x.strip()}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare ATTOM Assessor History for a CCAO-like AVM experiment."
    )
    parser.add_argument("--input", required=True, help="Input ATTOM parquet or CSV.")
    parser.add_argument(
        "--output-dir",
        default="./data/ATTOM/processed",
        help="Directory for the prepared parquet and audit outputs.",
    )
    parser.add_argument("--county-fips", default="17031")
    parser.add_argument("--min-sale-year", type=int, default=2016)
    parser.add_argument("--max-sale-year", type=int, default=2024)
    parser.add_argument(
        "--single-family-codes",
        default=None,
        help=(
            "Comma-separated verified ATTOM PROPERTYUSESTANDARDIZED codes for "
            "single-family homes. When omitted, no property-use restriction is made."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw = _load_table(input_path)
    model_df, stages, missingness, property_use = prepare_attom_assessor_history(
        raw,
        county_fips=args.county_fips,
        min_sale_year=args.min_sale_year,
        max_sale_year=args.max_sale_year,
        single_family_codes=_parse_codes(args.single_family_codes),
    )

    model_path = output_dir / "attom_assessor_history_ccao_like.parquet"
    stages_path = output_dir / "sample_construction.csv"
    missingness_path = output_dir / "missingness_report.csv"
    property_use_path = output_dir / "property_use_frequencies.csv"
    predictor_yaml_path = output_dir / "attom_predictor_config.yaml"
    metadata_path = output_dir / "preparation_metadata.json"

    model_df.to_parquet(model_path, index=False)
    stages.to_csv(stages_path, index=False)
    missingness.to_csv(missingness_path, index=False)
    property_use.to_csv(property_use_path, index=False)
    _write_predictor_yaml(predictor_yaml_path)

    metadata = {
        "input": str(input_path),
        "output": str(model_path),
        "county_fips": str(args.county_fips).zfill(5),
        "sale_year_range": [args.min_sale_year, args.max_sale_year],
        "single_family_codes": sorted(_parse_codes(args.single_family_codes) or []),
        "rows": int(model_df.shape[0]),
        "predictor_count": len(PREDICTOR_COLUMNS),
        "numeric_predictor_count": len(NUMERIC_PREDICTORS),
        "categorical_predictor_count": len(CATEGORICAL_PREDICTORS),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Prepared rows: {len(model_df):,}")
    print(f"Model parquet: {model_path}")
    print(f"Predictor config: {predictor_yaml_path}")
    print(f"Sample construction: {stages_path}")
    print(f"Missingness report: {missingness_path}")
    print(f"Property-use frequencies: {property_use_path}")
    if not args.single_family_codes:
        print(
            "WARNING: No verified single-family PROPERTYUSESTANDARDIZED codes "
            "were supplied. Inspect property_use_frequencies.csv and the ATTOM "
            "codebook before treating the output as a single-family sample."
        )


if __name__ == "__main__":
    main()
