#!/usr/bin/env python3
"""Shared v2 paths, FIPS, and hash helpers. Does not import modeling code at module level beyond stdlib."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
ANALYSIS = REPO / "analysis" / "berry_attom_validation_v2"
OUTPUT = REPO / "output" / "berry_attom_validation_v2"
RECORDER_DIR = REPO / "data/dewey-downloads/wayne-philadelphia-st-louis-2003-2025-recorder"
HISTORY_DIR = REPO / "data/dewey-downloads/wayne-philadelphia-st-louis-2004-2025-history"
BERRY_RAW = REPO / "data/berry_cmf/raw"
SALE_VALIDATION_DICTIONARY = (
    REPO / "data/dewey-downloads/data_dictionaries/attom_recorder_residential_avm_sale_validation_dictionary.csv"
)
LGBM_CONFIG_PATH = REPO / "best_lgbm_baseline_configs.yaml"

FIPS = {
    "wayne": "26163",
    "philadelphia": "42101",
    "st_louis_county": "29189",
}
ST_LOUIS_CITY_FIPS = "29510"
DETROIT_LABEL_FORBIDDEN_FOR_WAYNE = True

COUNTIES = (
    {"key": "wayne", "fips": "26163", "label": "Wayne County, MI", "berry_unit": "Detroit, MI"},
    {"key": "philadelphia", "fips": "42101", "label": "Philadelphia County, PA", "berry_unit": "Philadelphia, PA"},
    {"key": "st_louis_county", "fips": "29189", "label": "St. Louis County, MO", "berry_unit": "St. Louis County, MO"},
)

SEED = 2025
TEST_FRACTION = 0.20
VALIDATION_FRACTION = 0.10
N_BOOTSTRAP = 200
MIN_SALE_PRICE = 50_000.0
PROPERTY_USE_CODES = ("385",)
SALE_WINDOW = ("2016-01-01", "2025-12-31")
SALE_COHORT = "broad"

ASSESSMENT_VALUE_COLUMNS = {
    "TAXASSESSEDVALUEIMPROVEMENTS", "TAXASSESSEDVALUELAND", "TAXASSESSEDVALUETOTAL",
    "TAXMARKETVALUEIMPROVEMENTS", "TAXMARKETVALUELAND", "TAXMARKETVALUETOTAL", "TAXBILLEDAMOUNT",
    "PREVIOUSASSESSEDVALUE",
}

HISTORY_CACHE_COLUMNS = list(dict.fromkeys([
    "ATTOMID", "SITUSSTATECOUNTYFIPS", "ASSESSORHISTORYYEAR", "TAXYEARASSESSED",
    "PARCELNUMBERFORMATTED", "PARCELNUMBERPREVIOUS", "PARCELNUMBERRAW", "PARCELNUMBERALTERNATE",
    "PROPERTYADDRESSFULL", "PROPERTYADDRESSCITY", "PROPERTYADDRESSZIP",
    "PROPERTYJURISDICTIONNAME", "MINORCIVILDIVISIONNAME", "SITUSCOUNTY",
    "ASSESSORLASTSALEDATE", "ASSESSORLASTSALEAMOUNT",
    "AREABUILDING", "AREALOTSF", "AREA1STFLOOR", "AREA2NDFLOOR", "AREAUPPERFLOORS",
    "AREAGROSS", "AREALOTACRES", "YEARBUILT", "YEARBUILTEFFECTIVE",
    "BEDROOMSCOUNT", "BATHCOUNT", "BATHPARTIALCOUNT", "ROOMSCOUNT", "STORIESCOUNT", "UNITSCOUNT",
    "PARKINGGARAGEAREA", "PARKINGSPACECOUNT", "FIREPLACECOUNT",
    "ROOMSBASEMENTAREA", "ROOMSBASEMENTAREAFINISHED", "ROOMSBASEMENTAREAUNFINISHED",
    "PORCHAREA", "PATIOAREA", "DECKAREA", "BALCONYAREA", "POOLAREA",
    "PROPERTYUSESTANDARDIZED", "STRUCTURESTYLE", "EXTERIOR1CODE", "FOUNDATION", "CONSTRUCTION",
    "HVACCOOLINGDETAIL", "HVACHEATINGDETAIL", "HVACHEATINGFUEL", "PARKINGGARAGE", "FIREPLACE",
    "POOL", "PORCHCODE", "ROOFCONSTRUCTION", "ROOFMATERIAL", "NEIGHBORHOODCODE",
    "LEGALTOWNSHIP", "ZONEDCODELOCAL",
    # retained for optional Step 9 semantics check only; never as AVM predictors
    "TAXASSESSEDVALUETOTAL", "TAXMARKETVALUETOTAL",
]))

RECORDER_CACHE_COLUMNS = [
    "ATTOMID", "TRANSACTIONID", "DOCUMENTRECORDINGCOUNTYFIPS",
    "DOCUMENTRECORDINGCOUNTYNAME", "DOCUMENTRECORDINGJURISDICTIONNAME",
    "INSTRUMENTDATE", "RECORDINGDATE", "TRANSFERAMOUNT",
    "DOCUMENTTYPECODE", "ARMSLENGTHFLAG", "TRANSFERINFOMULTIPARCELFLAG",
    "FORECLOSUREAUCTIONSALE", "QUITCLAIMFLAG", "TRANSFERINFODISTRESSCIRCUMSTANCECODE",
    "PARTIALINTEREST", "TRANSACTIONTYPE", "TRANSFERAMOUNTINFOACCURACY",
    "TRANSFERINFOPURCHASETYPECODE", "PROPERTYUSESTANDARDIZED",
    "APNFORMATTED", "APNORIGINAL",
    "PROPERTYADDRESSFULL", "PROPERTYADDRESSCITY", "PROPERTYADDRESSZIP",
]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str) + "\n", encoding="utf-8")
