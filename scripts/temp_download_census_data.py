#!/usr/bin/env python3
"""Download tract-level ACS 5-year data for the ATTOM Recorder counties.

The county scope is read from
``data/dewey-downloads/10-counties-recorder-2016-2025`` so this downloader
cannot silently drift from the ATTOM extract.  Each (ACS vintage, county) is a
separate Parquet chunk under ``data/CensusData/acs5``; complete chunks are
validated and reused on subsequent runs.

The Census API now requires a key for all queries.  Set ``CENSUS_API_KEY`` in
the environment (do not put the key in this file) before downloading.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


ROOT = Path(__file__).resolve().parents[1]
RECORDER_DIR = ROOT / "data" / "dewey-downloads" / "10-counties-recorder-2016-2025"
OUTPUT_DIR = ROOT / "data" / "CensusData" / "acs5"
API_URL_TEMPLATE = "https://api.census.gov/data/{year}/acs/acs5"
YEARS = tuple(range(2015, 2025))

# Keep the original requested variables.  Estimates are stored as nullable
# integers; Census's negative missing-value sentinels become nulls.
ESTIMATE_VARIABLES = (
    "B01003_001E",  # Total population
    "B19013_001E",  # Median household income
    "B25001_001E",  # Total housing units
    "B25003_002E",  # Owner-occupied housing units
    "B25003_003E",  # Renter-occupied housing units
    "B25064_001E",  # Median gross rent
)
API_VARIABLES = ("NAME", *ESTIMATE_VARIABLES)
RECORDER_SCOPE_COLUMNS = (
    "DOCUMENTRECORDINGCOUNTYFIPS",
    "DOCUMENTRECORDINGCOUNTYNAME",
    "DOCUMENTRECORDINGSTATECODE",
)
OUTPUT_SCHEMA = pa.schema(
    [
        pa.field("NAME", pa.string()),
        *[pa.field(variable, pa.int64()) for variable in ESTIMATE_VARIABLES],
        pa.field("state", pa.string()),
        pa.field("county", pa.string()),
        pa.field("tract", pa.string()),
        pa.field("GEOID", pa.string()),
        pa.field("county_fips", pa.string()),
        pa.field("acs_vintage", pa.int16()),
    ]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", nargs="+", type=int, default=list(YEARS), help="ACS vintages to download.")
    parser.add_argument(
        "--counties",
        nargs="+",
        help="Optional five-digit county FIPS subset. It must be a subset of the Recorder extract.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Redownload chunks that already validate.")
    parser.add_argument("--dry-run", action="store_true", help="Validate scope and print planned chunks without API calls.")
    parser.add_argument("--request-pause", type=float, default=0.2, help="Seconds to pause between API requests (default: 0.2).")
    return parser.parse_args()


def normalize_fips(value: Any, width: int) -> str:
    """Return a zero-padded FIPS string, rejecting malformed source values."""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    if not text.isdigit() or len(text) > width:
        raise ValueError(f"Invalid FIPS value {value!r}; expected at most {width} digits.")
    return text.zfill(width)


def recorder_counties(recorder_dir: Path) -> dict[str, dict[str, str]]:
    """Read the exact county scope from ATTOM without loading its other fields."""
    if not recorder_dir.is_dir():
        raise FileNotFoundError(f"Recorder directory does not exist: {recorder_dir}")
    source = ds.dataset(recorder_dir, format="parquet")
    missing = sorted(set(RECORDER_SCOPE_COLUMNS) - set(source.schema.names))
    if missing:
        raise ValueError(f"Recorder data is missing required scope columns: {', '.join(missing)}")
    table = source.to_table(columns=list(RECORDER_SCOPE_COLUMNS))
    counties: dict[str, dict[str, str]] = {}
    for fips, county_name, state_code in zip(*[table.column(column).to_pylist() for column in RECORDER_SCOPE_COLUMNS]):
        if fips is None:
            continue
        county_fips = normalize_fips(fips, 5)
        state_fips, county_code = county_fips[:2], county_fips[2:]
        candidate = {
            "county_fips": county_fips,
            "state_fips": state_fips,
            "county_code": county_code,
            "county_name": str(county_name).strip() if county_name is not None else "",
            "state_code": str(state_code).strip() if state_code is not None else "",
        }
        previous = counties.setdefault(county_fips, candidate)
        if previous["county_name"] != candidate["county_name"] or previous["state_code"] != candidate["state_code"]:
            raise ValueError(f"Recorder scope has conflicting county metadata for {county_fips}.")
    if not counties:
        raise ValueError("No recording-county FIPS values were found in the ATTOM Recorder data.")
    return dict(sorted(counties.items()))


def session_with_retries() -> requests.Session:
    retry = Retry(
        total=4,
        backoff_factor=1.0,
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=("GET",),
        raise_on_status=False,
    )
    session = requests.Session()
    session.mount("https://", HTTPAdapter(max_retries=retry))
    return session


def validate_requested_variables(session: requests.Session, year: int, api_key: str) -> None:
    """Fail before data writes if a requested variable is unavailable in a vintage."""
    response = session.get(
        f"{API_URL_TEMPLATE.format(year=year)}/variables.json", params={"key": api_key}, timeout=60
    )
    response.raise_for_status()
    variables = response.json().get("variables", {})
    missing = sorted(set(ESTIMATE_VARIABLES) - set(variables))
    if missing:
        raise ValueError(f"ACS {year} does not expose required variables: {', '.join(missing)}")


def parse_estimate(value: Any, *, variable: str, year: int, county_fips: str) -> int | None:
    """Convert an ACS estimate while retaining API nulls as missing values."""
    if value is None:
        return None
    try:
        estimate = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"ACS {year} returned a non-integer {variable} value {value!r} for {county_fips}."
        ) from error
    # Census uses large negative values (for example -666666666) as
    # unavailable/not-applicable sentinels in many detailed tables.
    return None if estimate <= -666_666_000 else estimate


def census_rows(session: requests.Session, year: int, county: dict[str, str], api_key: str) -> list[dict[str, Any]]:
    params = {
        "get": ",".join(API_VARIABLES),
        "for": "tract:*",
        "in": f"state:{county['state_fips']} county:{county['county_code']}",
        "key": api_key,
    }
    response = session.get(API_URL_TEMPLATE.format(year=year), params=params, timeout=120)
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list) or len(payload) < 2:
        raise ValueError(f"ACS {year} returned no tract rows for Recorder county {county['county_fips']}.")
    headers, *rows = payload
    expected_headers = set(API_VARIABLES) | {"state", "county", "tract"}
    if not expected_headers.issubset(headers):
        raise ValueError(f"ACS {year} response is missing expected columns for {county['county_fips']}: {headers}")
    result: list[dict[str, Any]] = []
    for row in rows:
        values = dict(zip(headers, row))
        state = normalize_fips(values["state"], 2)
        county_code = normalize_fips(values["county"], 3)
        tract = normalize_fips(values["tract"], 6)
        county_fips = state + county_code
        if county_fips != county["county_fips"]:
            raise ValueError(f"ACS response escaped its requested county: {county_fips} != {county['county_fips']}")
        record: dict[str, Any] = {
            "NAME": values["NAME"],
            "state": state,
            "county": county_code,
            "tract": tract,
            "GEOID": state + county_code + tract,
            "county_fips": county_fips,
            "acs_vintage": year,
        }
        for variable in ESTIMATE_VARIABLES:
            record[variable] = parse_estimate(
                values[variable], variable=variable, year=year, county_fips=county_fips
            )
        result.append(record)
    geoids = [row["GEOID"] for row in result]
    if len(set(geoids)) != len(geoids):
        raise ValueError(f"ACS {year} returned duplicate tract GEOIDs for {county['county_fips']}.")
    return result


def chunk_path(output_dir: Path, year: int, county_fips: str) -> Path:
    return output_dir / f"year={year}" / f"county_fips={county_fips}" / "tracts.parquet"


def chunk_is_valid(path: Path, year: int, county_fips: str) -> bool:
    if not path.is_file():
        return False
    try:
        table = pq.read_table(path)
    except Exception:
        return False
    if table.schema != OUTPUT_SCHEMA or table.num_rows == 0:
        return False
    records = table.select(["county_fips", "acs_vintage", "GEOID", "state", "county"]).to_pylist()
    return (
        all(
            row["county_fips"] == county_fips
            and row["acs_vintage"] == year
            and len(row["GEOID"]) == 11
            and row["GEOID"] == row["state"] + row["county"] + row["GEOID"][-6:]
            for row in records
        )
        and len({row["GEOID"] for row in records}) == len(records)
    )


def write_chunk(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".parquet.tmp")
    pq.write_table(pa.Table.from_pylist(records, schema=OUTPUT_SCHEMA), temporary, compression="zstd")
    temporary.replace(path)


def selected_counties(all_counties: dict[str, dict[str, str]], requested: Iterable[str] | None) -> dict[str, dict[str, str]]:
    if requested is None:
        return all_counties
    requested_fips = {normalize_fips(value, 5) for value in requested}
    unknown = sorted(requested_fips - set(all_counties))
    if unknown:
        raise ValueError(f"Requested counties are not in the ATTOM Recorder extract: {', '.join(unknown)}")
    return {fips: all_counties[fips] for fips in sorted(requested_fips)}


def main() -> None:
    args = parse_args()
    years = tuple(sorted(set(args.years)))
    invalid_years = sorted(set(years) - set(YEARS))
    if invalid_years:
        raise ValueError(f"Only ACS vintages {YEARS[0]}-{YEARS[-1]} are in scope; got {invalid_years}")
    if args.request_pause < 0:
        raise ValueError("--request-pause cannot be negative.")

    all_counties = recorder_counties(RECORDER_DIR)
    counties = selected_counties(all_counties, args.counties)
    print(f"Recorder scope: {len(all_counties)} counties: {', '.join(all_counties)}")
    print(f"Planned ACS chunks: {len(years) * len(counties)} ({years[0]}-{years[-1]})")
    if args.dry_run:
        return

    api_key = os.environ.get("CENSUS_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("CENSUS_API_KEY is required by the Census API. Set it in the environment and rerun.")

    session = session_with_retries()
    for year in years:
        validate_requested_variables(session, year, api_key)
        for county_fips, county in counties.items():
            output_path = chunk_path(OUTPUT_DIR, year, county_fips)
            if not args.overwrite and chunk_is_valid(output_path, year, county_fips):
                print(f"skip  {year} {county_fips}: validated {output_path.relative_to(ROOT)}")
                continue
            records = census_rows(session, year, county, api_key)
            write_chunk(records, output_path)
            if not chunk_is_valid(output_path, year, county_fips):
                raise RuntimeError(f"Post-write validation failed: {output_path}")
            print(f"write {year} {county_fips}: {len(records):,} tracts -> {output_path.relative_to(ROOT)}")
            if args.request_pause:
                time.sleep(args.request_pause)


if __name__ == "__main__":
    main()
