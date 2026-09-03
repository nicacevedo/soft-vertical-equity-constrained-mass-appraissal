# New Dewey delivery inventory (Step 2)

Folder names were **not** treated as evidence of contents. Shards were hashed and scanned.

## Shard counts and sizes

| product | shards | footer rows | bytes |
|---|---:|---:|---:|
| Recorder | 64 | 6,901,475 | 1,079,113,474 |
| Assessor History | 64 | 49,523,812 | 6,956,928,070 |

SHA-256 values are in `new_dewey_file_manifest.csv`.

## Actual FIPS (do not trust the folder title)

Recorder FIPS present: `26163`, `29189`, `29223`, `42101`, `42127`.

History FIPS present include the three targets **plus many extras**
(`42127` Wayne County PA, `29223` Washington County MO, `26155`, `13305`, …, and a malformed `5499`).

| FIPS | Role |
|---|---|
| 26163 | Wayne County MI (intended) |
| 42101 | Philadelphia County PA (intended) |
| 29189 | St. Louis County MO (intended) |
| 29510 | St. Louis City — **absent** (good) |
| 42127 | Wayne County PA — extra; never treat as Detroit/Wayne MI |
| 29223 | Washington County MO — extra |

County caches **predicate-filter** to 26163 / 42101 / 29189 only.

## Dates / years

- Recorder instrument dates: 1900-01-01 through 2026-08-21 (pre-2003 rows are residual; modeled window remains 2016–2025).
- Recorder recording dates: 1930-12-10 through 2026-08-21.
- Assessor History years: 2003–2025 (folder name said 2004–2025).

## Identifiers

- ATTOMID missing: 0 (both products).
- TRANSACTIONID non-null: all recorder rows; per-shard distinct count sums to N (no within-shard duplicate IDs). Global uniqueness across shards is not a cryptographic uniqueness proof.
- Recorder `APNFORMATTED` non-null on every row; `APNORIGINAL` sparse (~1%).
- History `PARCELNUMBERFORMATTED` is **100% missing**. Parcel linkage must use `PARCELNUMBERRAW` / `PARCELNUMBERPREVIOUS`, not formatted APN.
- History `PROPERTYJURISDICTIONNAME` is **100% missing**, so a Detroit-city subset cannot be validated from that field in this extract. `MINORCIVILDIVISIONNAME` remains a later sensitivity candidate only if cache coverage supports it.

## Feature coverage (history, all shards)

Near-complete: YEARBUILT, BEDROOMSCOUNT, AREABUILDING, PROPERTYUSESTANDARDIZED, TAXASSESSEDVALUETOTAL, PARCELNUMBERRAW, PARCELNUMBERPREVIOUS.
Small missingness: BATHCOUNT ~0.39%, AREALOTSF ~0.23%.
