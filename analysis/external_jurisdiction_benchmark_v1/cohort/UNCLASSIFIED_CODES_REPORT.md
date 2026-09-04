# Property-use code classification: empirical anomalies flagged before cohort freeze

Per protocol: "if a proposed code appears semantically inconsistent with the published legend,
stop and report it." No cohort has been built on the current mapping. This report is that stop.

Source: `cohort/jurisdiction_code_frequency.csv` / `jurisdiction_code_frequency_pivot.csv`, built
from the full Assessor History cache for all 9 jurisdictions (~105M rows).

## 1. Only 3 of the 14 given PRIMARY_RESIDENTIAL codes are observed at all

`{363, 364, 376, 380, 382, 383, 384, 390, 423, 447, 452}` — 11 of 14 codes — have **zero** rows
across all nine jurisdictions combined. Observed: `385` (57.5M rows), `386` (2.16M), `377` (40,224,
almost entirely Maricopa). Some absences are plausible for genuinely rare categories (Barndominium,
Tiny House, Zero-Lot-Line). Others are harder to explain this way: PUD and Townhouse are common
housing forms, yet PUD (377) is a rounding error and Townhouse (390) is entirely absent.

## 2. Code 385's volume is inconsistent with its given label ("Seasonal/Cabin Residence")

385 is **54.7% of all History rows**, present in **every one of the nine jurisdictions** at large
volume: Wayne 13.26M, Maricopa 11.13M, Miami-Dade 6.52M, Cook 6.00M, St. Louis 5.60M, Middlesex
5.04M, King 4.73M, Philadelphia 2.68M, Allegheny 2.56M.

A code describing seasonal/vacation-cabin residences would be a small, geographically concentrated
minority — high in rural/lake/mountain counties, negligible in dense urban counties. Being the
*majority* land-use code in Cook County, Maricopa County, and Miami-Dade County is not consistent
with that description. This pattern — one code at roughly half of all records, present everywhere
— is the signature of a generic/default residential code, not a niche seasonal category.

This is corroborated independently: `analysis/berry_attom_validation_v3/scripts/v3_common.py`
already documented, from its own empirical profiling, that code 385 functions as the dominant
general-residential code in Wayne and St. Louis (v3 built its entire primary cohort on 385 alone,
treating it as "the" single-family code — not as a seasonal/cabin subset).

## 3. Code 386's volume is inconsistent with its given label ("Single Family Residence")

386, given as the flagship general single-family code, is only 2.05% of all rows and is **entirely
absent** in Wayne (0) and Middlesex (0), and nearly absent in St. Louis (1,330 of 5.6M+385-coded
rows). A code labeled "Single Family Residence" would be expected to appear broadly across every
jurisdiction's housing stock, not vanish in a third of them.

## 4. Code 366 is NOT in PRIMARY_RESIDENTIAL, but behaves like a mainstream residential code

366 is **19.2% of all rows** (20.2M), present in **all nine** jurisdictions, and is the single
largest code in Philadelphia (5.36M, ~50% of Philadelphia's coded rows) and comparable to 385 in
volume in Miami-Dade (6.10M vs 385's 6.52M there). Nothing in the categories named for exclusion
(condos/coops, duplex/triplex/quadplex, multifamily/apartments, manufactured/mobile, timeshares,
common-area records, vacant land) plausibly accounts for ~19% of parcels in nine major metro
counties simultaneously. v3 independently reached the same conclusion for Philadelphia specifically.

## Top codes by total volume, with primary-set membership

| use_code | total_n | share_of_all_rows | n_counties_present | in_primary_residential |
|---|---|---|---|---|
| 385 | 57,506,198 | 54.70% | 9 | **yes** |
| 366 | 20,219,214 | 19.23% | 9 | no |
| (missing/null) | 5,343,526 | 5.08% | 9 | no |
| 401 | 3,167,455 | 3.01% | 9 | no |
| 397 | 2,811,905 | 2.67% | 9 | no |
| 361 | 2,194,656 | 2.09% | 9 | no |
| 386 | 2,155,718 | 2.05% | 7 | **yes** |
| 294 | 1,792,194 | 1.70% | 9 | no |
| 369 | 1,448,514 | 1.38% | 7 | no |
| 375 | 785,164 | 0.75% | 9 | no |

Full table: `cohort/jurisdiction_code_frequency_pivot.csv` (85 distinct codes observed overall).

## What this does NOT mean

This is not evidence that the codes are wrong in the ATTOM system — only that the specific
code-to-label pairing supplied (385=Seasonal/Cabin, 386=Single Family) does not match the observed
volume/geography pattern well enough to freeze without confirmation. Two live possibilities:

1. The numbering supplied has an off-by-something or list-order error (e.g., 385/386 are swapped
   relative to their intended positions, or the source table's codes shifted between versions).
2. ATTOM's field is used inconsistently across data vintages/deliveries, and 385 genuinely is a
   generic catch-all in this particular Dewey delivery regardless of the reference table's label.

Either way, no cohort should be built, and no report/paper text should describe 385 as
"seasonal/cabin residence" or 386 as "the" single-family code, until this is resolved.

## Not yet actionable regardless of the above

`EXPLICITLY_EXCLUDED_FROM_PRIMARY` and `BROAD_RESIDENTIAL_APPENDIX` were specified by category name,
not exact code. Codes 401 (3.01%), 397 (2.67%), 361 (2.09%), 294 (1.70%), 369 (1.38%) and others
cannot yet be assigned to "vacant land" / "multifamily" / "manufactured home" / etc. without exact
code values for those categories.
