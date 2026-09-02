# Berry/CMF reproduction notes

Written after source registry, hashed downloads, and schema inspection.
This documents the **assessment-ratio** replication, not the paper AVM.

## Method

CMF local pages call `cmfproperty::reformat_data` then `make_report` /
`calc_iaao_stats`. R/Stata were not used for the numeric pass. Operations
were translated from:

- `data/berry_cmf/raw/_shared/cmfproperty/R/reformat_data.R`
- `data/berry_cmf/raw/_shared/cmfproperty/R/iaao_stats.R`
- jurisdiction page code / Box Rmd files

`reformat_data` (default) creates `RATIO = AV/SP` for `SP>100` and drops
year-specific Tukey IQR ratio outliers. Inflation adjustment writes
`SALE_PRICE_ADJ` only; IAAO point estimates use unadjusted `RATIO`,
`SALE_PRICE`, and `ASSESSED_VALUE`.

COD/PRD/PRB here are **CMF formulas**, not `utils.motivation_utils`
paper metrics.

Executable logs: `analysis/berry_cmf_validation/logs/reproduction/`.

## Detroit, MI — EXACT_OR_ROUNDING_MATCH

Source: `detroit_replication_code.Rmd` + `combined.csv`.
Report: *An Evaluation of Residential Property Tax Assessments in the
City of Detroit, 2016-2018* (February 2020; Box PDF and CMF site PDF).

Filters (Rmd): `Property Class == 401`, `Terms of Sale == 'VALID ARMS
LENGTH'`, `reformat_data(..., filter=FALSE)` so the official arms-length
flag is kept. Before reappraisal = `quarter_num <= 5` (through 2017Q1).

| Quantity | Reported | Reproduced |
|---|---|---|
| 2016Q2 N | 975 | 975 |
| Window N (sum of Table 1) | 9653 | 9653 |
| COD before / after | 46.26 / 50.08 | 46.2205 / 50.1694 |
| PRD before / after | 1.30 / 1.35 | 1.3012 / 1.3466 |
| PRB before / after | -0.26 / -0.45 | -0.2574 / -0.4458 |

Relative discrepancies are well under 1%. This is independent confirmation
of documented Detroit assessment regressivity.

## Philadelphia County, PA — PARTIAL_REPRODUCTION

Followed `philadelphia-raw-data-code`: rbind `Ratio_Analysis_arms_length.dta`
and `Ratio_Analysis_total.dta`, then default IQR `reformat_data`.
Reproduced N=88,811 (2012–2018) after IQR filter; overall COD 35.70,
PRD 1.30, PRB -0.11.

No local CMF PDF with published N/COD/PRD/PRB was linked. The page
“Report” URL is the **nationwide CoreLogic** HTML, which CMF says cannot
be redistributed as microdata and is a different dataset. Rbind of
total+arms-length files may duplicate rows. Clean CSV
`final-Philadelphia.csv` is a five-column extract.

## Orleans Parish, LA — PARTIAL_REPRODUCTION

README: FOIA quote ~$75,000; CMF used GitHub user `bhelx` scrape of the
assessor site. Rebuilt `nola.R` CSV join (residential, `price<1e7`, max
price per property-year, `sale>=2000`) and applied the `/0.1` assessed
value scaling used when writing `nola_joined.RDS`. Skipped `tidycensus`
tract inner join, so N may exceed the RDS used for the HTML report.
Reproduced N=30,939 (2015–2019), COD 46.68, PRD 1.24, PRB -0.075.

## Franklin County / Columbus, OH — PARTIAL_REPRODUCTION

Working replication page is the hub typo URL `columnbs-raw-data-code`
(Columbus city). Box contains only `columbusonly.csv` (PID, PRICE,
SALE_YEAR, ASSESSED_VALUE, LAND_CLASS). `franklin-code` / `franklin-raw-data-code`
overwrite `df` with `cmfproperty::example_data` and have no data download.
Page “Report” points at **Columbus County, North Carolina** CoreLogic HTML.

Applied `LAND_CLASS` 510–530 and default IQR filter: N=186,231 (2002–2018),
COD 33.96, PRD 1.17, PRB -0.197. No published local table to match.
GeoDatabase cited in page code was not in the Box share.

## St. Louis County, MO — PARTIAL_REPRODUCTION (code defect)

The published snippet creates `sold_only` (`PRICE`, `SALEVAL=="X"`) then
**ignores it** and runs `make_report` on `fullroll.csv` with
`SALE_PRICE = APRTOT` (appraised total) and `ASSESSED_VALUE = ASMTOT`.
As-written: N=3,715,468, COD≈0.012, PRD=1.0, PRB=0 — i.e. appraisal vs
assessment on the full roll, not a sale-ratio study.

Intended sold-only construction on `joined.csv`: N=73,743 after IQR,
COD 11.87, PRD 1.015, PRB -0.007. Use **sold-only** for any scientific
sale-ratio claim. Box HTML report was not parsed for published cells in
this pass.

Yearly extracts include `dwelling` snapshots (YRBLT, RMBED, FIXBATH,
SFLA, GRADE, CDU) with tax year on the row.

## Cook County, IL — PARTIAL_REPRODUCTION

Concatenated `res_sales2002.csv`–`res_sales2015.csv`, `bor_class` in the
published residential list, `SALE_PRICE=NetConsideration`,
`ASSESSED_VALUE=bor_CCAO_ass`, `SALE_YEAR=joinyr`, default IQR filter.
N=641,147 (2002–2015). Overall COD 19.14, PRD 1.039, PRB -0.007.
2011 is the most regressive year in this reconstruction (PRD 1.16,
PRB -0.086).

No local CMF numeric report to match. Page “Report” is CoreLogic
nationwide HTML. **These files end in 2015; CCAO `training_data.parquet`
starts 2016 — no overlapping sale years for transaction matching.**

## What was not run in R

`module avail` lists R/4.4.3 and R/4.5.2. Numeric translations were
used instead of installing `cmfproperty` dependencies (`tidycensus`,
`sf`, etc.). Detroit’s match to the published table indicates the
translation of `calc_iaao_stats` is adequate for this audit.
