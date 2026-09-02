#!/usr/bin/env python3
"""Write post-reproduction audit tables (steps 6-11, 13 scaffolding). No Direct/Surrogate."""
from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml

REPO = Path(__file__).resolve().parents[3]
A = REPO / "analysis" / "berry_cmf_validation"
RAW = REPO / "data" / "berry_cmf" / "raw"
NOW = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def sha(p: Path) -> str:
    if not p.exists():
        return ""
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path: Path, rows: list, fields: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main() -> int:
    # ---- Step 6 usability ----
    use_fields = [
        "jurisdiction", "geographic_unit", "usability_class", "classified_before_direct_surrogate",
        "sale_target", "sale_dates", "qualification_logic", "predictor_sufficiency",
        "temporal_concern", "linkage_needed", "evidence",
    ]
    use_rows = [
        dict(jurisdiction="detroit_mi", geographic_unit="Detroit, MI",
             usability_class="A_EXTERNAL_REGRESSIVITY_ONLY",
             classified_before_direct_surrogate="true",
             sale_target="Adj. Sale $ on combined.csv",
             sale_dates="Sale Date 2016-2018",
             qualification_logic="Terms of Sale == VALID ARMS LENGTH; class 401",
             predictor_sufficiency="Floor Area, Building Style, District present; no year built, lot, beds/baths in combined file",
             temporal_concern="Assessor workbook attached to sales; snapshot date vs sale not documented (Appr. Date often empty)",
             linkage_needed="no ATTOM Detroit extract in repo",
             evidence="EXACT_OR_ROUNDING_MATCH to 2020 CMF table; thin/timing-unresolved AVM features"),
        dict(jurisdiction="philadelphia_pa", geographic_unit="Philadelphia County, PA",
             usability_class="C_PUBLIC_ENRICHMENT_CANDIDATE",
             classified_before_direct_surrogate="true",
             sale_target="sale_price in OPA ratio-analysis Stata",
             sale_dates="saledate / sale_year 2012-2018",
             qualification_logic="sale_type_to_use; plus CMF IQR in page code",
             predictor_sufficiency="DTA contains totlivarea, yrbuilt, nobath, stories but bundled in a ratio-study extract",
             temporal_concern="Characteristic timing vs sale not documented in CMF code; likely current OPA attrs on historical sales",
             linkage_needed="OpenDataPhilly assessment history is the documented governmental source",
             evidence="PARTIAL_REPRODUCTION; clean CSV is five columns only"),
        dict(jurisdiction="orleans_la", geographic_unit="Orleans Parish, LA",
             usability_class="A_EXTERNAL_REGRESSIVITY_ONLY",
             classified_before_direct_surrogate="true",
             sale_target="sales.price",
             sale_dates="sales.date 2015-2019 in reconstructed join",
             qualification_logic="nola.R price filters; no official arms-length flag",
             predictor_sufficiency="properties table has land/building area and coords; year built/beds absent",
             temporal_concern="properties.csv is a website scrape (inserted_at/updated_at); not a historical snapshot",
             linkage_needed="no ATTOM Orleans extract; FOIA for official history quoted at ~$75k",
             evidence="README scrape provenance; ACS join skipped in reproduction"),
        dict(jurisdiction="franklin_oh", geographic_unit="Columbus city (not full Franklin County)",
             usability_class="A_EXTERNAL_REGRESSIVITY_ONLY",
             classified_before_direct_surrogate="true",
             sale_target="PRICE",
             sale_dates="SALE_YEAR 2002-2018",
             qualification_logic="LAND_CLASS 510-530 plus CMF IQR; no deed/arms-length field",
             predictor_sufficiency="only PID, PRICE, SALE_YEAR, ASSESSED_VALUE, LAND_CLASS in Box file",
             temporal_concern="n/a — no structural predictors in file",
             linkage_needed="page cites FCA_SDE_Web_Prod.gdb not included in Box",
             evidence="SOURCE incomplete for countywide full.csv; wrong CoreLogic report geography"),
        dict(jurisdiction="st_louis_county_mo", geographic_unit="St. Louis County, MO",
             usability_class="B_STANDALONE_AVM_READY",
             classified_before_direct_surrogate="true",
             sale_target="PRICE on sold_only/joined.csv (not APRTOT)",
             sale_dates="SALEDT via 2019 sales extract joined by PARID/year 2009-2019",
             qualification_logic="SALEVAL==X as published sold_only construction",
             predictor_sufficiency="yearly dwelling snapshots: YRBLT, RMBED, FIXBATH, SFLA, GRADE, CDU, STYLE, STORIES",
             temporal_concern="use dwelling TAXYR matching sale year (or prior); file header truncated — map 39 fields from data dictionary",
             linkage_needed="not required if yearly extracts used",
             evidence="intended sold-only N~79k before IQR; as-written make_report path scientifically invalid"),
        dict(jurisdiction="cook_il", geographic_unit="Cook County, IL",
             usability_class="A_EXTERNAL_REGRESSIVITY_ONLY",
             classified_before_direct_surrogate="true",
             sale_target="NetConsideration on CMF res_sales2002-2015",
             sale_dates="joinyr 2002-2015 (disjoint from CCAO 2016-2025 modeling sample)",
             qualification_logic="bor_class residential list + CMF IQR",
             predictor_sufficiency="area, class, township/neighborhood only in res_sales extracts",
             temporal_concern="BOR extract; characteristic vintage vs sale not documented",
             linkage_needed="CCAO training_data.parquet remains the primary AVM source; ATTOM Cook is separate sensitivity",
             evidence="PARTIAL_REPRODUCTION; no year overlap with canonical CCAO experiment"),
        dict(jurisdiction="nyc_ny", geographic_unit="New York City, NY",
             usability_class="E_NOT_RECOMMENDED",
             classified_before_direct_surrogate="true",
             sale_target="inventory only; large files not downloaded",
             sale_dates="", qualification_logic="", predictor_sufficiency="data folder ~3.9GB not downloaded this pass",
             temporal_concern="", linkage_needed="",
             evidence="secondary_inventory_only per protocol"),
        dict(jurisdiction="erie_ny", geographic_unit="Buffalo City, NY",
             usability_class="E_NOT_RECOMMENDED",
             classified_before_direct_surrogate="true",
             sale_target="inventory; assessment rolls skipped as large",
             sale_dates="Sales 2009-2019.xlsx downloaded",
             qualification_logic="page filters Prop Class 210/220/230, Sale_Year>2017, distinct SBL",
             predictor_sufficiency="2020 disclosure notice is a single-year snapshot on 2018-2019 sales",
             temporal_concern="POST_SALE_LEAKAGE risk: 2020 roll on earlier sales",
             linkage_needed="",
             evidence="secondary; page itself uses a 2020 extract for 2018+ sales"),
        dict(jurisdiction="clark_nv", geographic_unit="Las Vegas City, NV",
             usability_class="E_NOT_RECOMMENDED",
             classified_before_direct_surrogate="true",
             sale_target="inventory; yearly CSVs skipped",
             sale_dates="", qualification_logic="", predictor_sufficiency="",
             temporal_concern="", linkage_needed="",
             evidence="secondary_inventory_only; ~800MB yearly files not downloaded"),
        dict(jurisdiction="los_angeles_ca", geographic_unit="Los Angeles County, CA",
             usability_class="E_NOT_RECOMMENDED",
             classified_before_direct_surrogate="true",
             sale_target="not downloaded",
             sale_dates="", qualification_logic="", predictor_sufficiency="",
             temporal_concern="page code overwrites SALE_PRICE/ASSESSED_VALUE inconsistently",
             linkage_needed="",
             evidence="secondary; source-code quality risk even if files were fetched"),
        dict(jurisdiction="maricopa_az", geographic_unit="Maricopa County, AZ",
             usability_class="E_NOT_RECOMMENDED",
             classified_before_direct_surrogate="true",
             sale_target="not downloaded (PHX_* files up to 3.2GB)",
             sale_dates="", qualification_logic="", predictor_sufficiency="",
             temporal_concern="page uses AZ limited property value (lpv) as ASSESSED_VALUE",
             linkage_needed="existing ATTOM Maricopa remains sensitivity layer",
             evidence="secondary_inventory_only; ATTOM already covers Maricopa"),
    ]
    write_csv(A / "modeling_readiness/jurisdiction_usability_matrix.csv", use_rows, use_fields)

    # ---- Step 7 temporal ----
    t_fields = ["jurisdiction", "source_table", "source_variable", "semantic",
                "temporal_class", "evidence", "allowed_in_main_avm"]
    t_rows = [
        dict(jurisdiction="detroit_mi", source_table="combined.csv", source_variable="Adj. Sale $",
             semantic="sale_price", temporal_class="SAFE_AS_OF_SALE",
             evidence="transaction price on sale row", allowed_in_main_avm="n/a_outcome"),
        dict(jurisdiction="detroit_mi", source_table="combined.csv", source_variable="Sale Date",
             semantic="sale_date", temporal_class="SAFE_AS_OF_SALE",
             evidence="sale event date", allowed_in_main_avm="n/a_outcome"),
        dict(jurisdiction="detroit_mi", source_table="combined.csv", source_variable="Asd. when Sold",
             semantic="assessed_value", temporal_class="SAFE_AS_OF_SALE",
             evidence="label claims assessment at sale; diagnostic only, not AVM target",
             allowed_in_main_avm="false_diagnostic"),
        dict(jurisdiction="detroit_mi", source_table="combined.csv", source_variable="Floor Area",
             semantic="building_area", temporal_class="TIMING_UNRESOLVED",
             evidence="no snapshot date; Appr. Date often missing", allowed_in_main_avm="false"),
        dict(jurisdiction="detroit_mi", source_table="combined.csv", source_variable="Building Style",
             semantic="style", temporal_class="TIMING_UNRESOLVED",
             evidence="same workbook, no as-of date", allowed_in_main_avm="false"),
        dict(jurisdiction="philadelphia_pa", source_table="Ratio_Analysis_*.dta",
             source_variable="sale_price", semantic="sale_price",
             temporal_class="SAFE_AS_OF_SALE", evidence="transaction field", allowed_in_main_avm="n/a_outcome"),
        dict(jurisdiction="philadelphia_pa", source_table="Ratio_Analysis_*.dta",
             source_variable="assmt_at_sale", semantic="assessed_value",
             temporal_class="SAFE_AS_OF_SALE", evidence="named assessment at sale; diagnostic",
             allowed_in_main_avm="false_diagnostic"),
        dict(jurisdiction="philadelphia_pa", source_table="Ratio_Analysis_*.dta",
             source_variable="totlivarea", semantic="building_area",
             temporal_class="TIMING_UNRESOLVED",
             evidence="CMF code never dates characteristics relative to saledate",
             allowed_in_main_avm="false"),
        dict(jurisdiction="philadelphia_pa", source_table="Ratio_Analysis_*.dta",
             source_variable="yrbuilt", semantic="year_built",
             temporal_class="TIMING_UNRESOLVED",
             evidence="stable if truly year built, but vintage of OPA extract unknown",
             allowed_in_main_avm="false"),
        dict(jurisdiction="orleans_la", source_table="sales.csv", source_variable="price",
             semantic="sale_price", temporal_class="SAFE_AS_OF_SALE",
             evidence="sale row", allowed_in_main_avm="n/a_outcome"),
        dict(jurisdiction="orleans_la", source_table="values.csv", source_variable="total_assessed_value",
             semantic="assessed_value", temporal_class="HISTORICAL_SNAPSHOT_REQUIRED",
             evidence="yearly values; join on sale year is as-of-year assessment",
             allowed_in_main_avm="false_diagnostic"),
        dict(jurisdiction="orleans_la", source_table="properties.csv", source_variable="building_area_sq_ft",
             semantic="building_area", temporal_class="POST_SALE_LEAKAGE",
             evidence="scrape with updated_at; not a historical roll",
             allowed_in_main_avm="false"),
        dict(jurisdiction="franklin_oh", source_table="columbusonly.csv", source_variable="PRICE",
             semantic="sale_price", temporal_class="SAFE_AS_OF_SALE",
             evidence="sale year only (no day)", allowed_in_main_avm="n/a_outcome"),
        dict(jurisdiction="franklin_oh", source_table="columbusonly.csv", source_variable="ASSESSED_VALUE",
             semantic="assessed_value", temporal_class="TIMING_UNRESOLVED",
             evidence="no assessment stage/year distinct from SALE_YEAR",
             allowed_in_main_avm="false_diagnostic"),
        dict(jurisdiction="st_louis_county_mo", source_table="2019/sales.csv", source_variable="PRICE",
             semantic="sale_price", temporal_class="SAFE_AS_OF_SALE",
             evidence="SALEDT on sales extract", allowed_in_main_avm="n/a_outcome"),
        dict(jurisdiction="st_louis_county_mo", source_table="dwelling.{csv,txt}", source_variable="SFLA",
             semantic="building_area", temporal_class="SAFE_AS_OF_SALE",
             evidence="yearly extract TAXYR; join TAXYR==sale year (or prior year) is a historical snapshot",
             allowed_in_main_avm="true_if_year_matched"),
        dict(jurisdiction="st_louis_county_mo", source_table="dwelling.{csv,txt}", source_variable="YRBLT",
             semantic="year_built", temporal_class="SAFE_AS_OF_SALE",
             evidence="year built on that tax-year card", allowed_in_main_avm="true_if_year_matched"),
        dict(jurisdiction="st_louis_county_mo", source_table="dwelling.{csv,txt}", source_variable="RMBED",
             semantic="bedrooms", temporal_class="SAFE_AS_OF_SALE",
             evidence="tax-year dwelling card", allowed_in_main_avm="true_if_year_matched"),
        dict(jurisdiction="st_louis_county_mo", source_table="fullroll.csv APRTOT",
             source_variable="APRTOT", semantic="appraised_value",
             temporal_class="NOT_APPLICABLE",
             evidence="must not be used as sale price (published snippet defect)",
             allowed_in_main_avm="false"),
        dict(jurisdiction="cook_il", source_table="res_salesYYYY.csv", source_variable="NetConsideration",
             semantic="sale_price", temporal_class="SAFE_AS_OF_SALE",
             evidence="BOR sale extract", allowed_in_main_avm="n/a_outcome"),
        dict(jurisdiction="cook_il", source_table="res_salesYYYY.csv", source_variable="area",
             semantic="building_area", temporal_class="TIMING_UNRESOLVED",
             evidence="single area field, no assessor-year of characteristics",
             allowed_in_main_avm="false"),
        dict(jurisdiction="cook_il", source_table="CCAO training_data.parquet",
             source_variable="char_*", semantic="ccao_characteristics",
             temporal_class="SAFE_AS_OF_SALE",
             evidence="canonical CCAO modeling extract; not modified this pass",
             allowed_in_main_avm="true_primary_application_only"),
    ]
    write_csv(A / "feature_coverage/temporal_integrity_audit.csv", t_rows, t_fields)

    # ---- Step 8 public enrichment ----
    p_fields = ["jurisdiction", "usability_before", "usability_after", "source_name",
                "source_url", "what_it_could_supply", "temporal_validity",
                "acquired", "reason"]
    p_rows = [
        dict(jurisdiction="philadelphia_pa", usability_before="C_PUBLIC_ENRICHMENT_CANDIDATE",
             usability_after="C_PUBLIC_ENRICHMENT_CANDIDATE",
             source_name="OpenDataPhilly OPA properties and assessment history",
             source_url="https://opendataphilly.org/datasets/philadelphia-properties-and-assessment-history/",
             what_it_could_supply="year-by-year assessed values and some characteristics",
             temporal_validity="assessment history CSV is yearly; characteristic as-of-sale still needs a dated roll",
             acquired="false",
             reason="clearly useful only after a dated historical characteristic extract is confirmed; not bulk-downloaded this pass"),
        dict(jurisdiction="franklin_oh", usability_before="A_EXTERNAL_REGRESSIVITY_ONLY",
             usability_after="A_EXTERNAL_REGRESSIVITY_ONLY",
             source_name="Franklin County Auditor GIS (FCA_SDE_Web_Prod.gdb cited by CMF)",
             source_url="https://www.franklincountyauditor.com/",
             what_it_could_supply="HistoricalParcelCentroids / municipal boundary used in Columbus page code",
             temporal_validity="CMF used HistoricalParcelCentroids; current live GDB may not be as-of-sale",
             acquired="false",
             reason="Box share omitted the GDB; live auditor GIS is not a demonstrated historical snapshot"),
        dict(jurisdiction="detroit_mi", usability_before="A_EXTERNAL_REGRESSIVITY_ONLY",
             usability_after="A_EXTERNAL_REGRESSIVITY_ONLY",
             source_name="City of Detroit Office of the Assessor / open data",
             source_url="https://detroitmi.gov/departments/office-chief-financial-officer/ocfo-divisions/office-assessor",
             what_it_could_supply="parcel characteristics if historical rolls exist",
             temporal_validity="current assessor portals typically lack sale-dated snapshots",
             acquired="false",
             reason="no demonstrated historical characteristic API; CMF already shipped the ratio-study workbook"),
        dict(jurisdiction="orleans_la", usability_before="A_EXTERNAL_REGRESSIVITY_ONLY",
             usability_after="A_EXTERNAL_REGRESSIVITY_ONLY",
             source_name="Orleans Parish Assessor / bhelx scrape",
             source_url="https://github.com/bhelx/nola-assessor-data",
             what_it_could_supply="already the CMF raw CSVs",
             temporal_validity="website scrape, not certified historical rolls",
             acquired="false",
             reason="CMF Box already contains the scrape; FOIA for official history was $75k; do not treat GitHub as a new official source"),
        dict(jurisdiction="st_louis_county_mo", usability_before="B_STANDALONE_AVM_READY",
             usability_after="B_STANDALONE_AVM_READY",
             source_name="n/a", source_url="",
             what_it_could_supply="yearly extracts already in CMF Box",
             temporal_validity="tax-year files", acquired="false",
             reason="public enrichment not required"),
        dict(jurisdiction="cook_il", usability_before="A_EXTERNAL_REGRESSIVITY_ONLY",
             usability_after="A_EXTERNAL_REGRESSIVITY_ONLY",
             source_name="CCAO research extract already in repo",
             source_url="data/CCAO/2025/training_data.parquet",
             what_it_could_supply="primary AVM features (do not replace with CMF 2002-2015 files)",
             temporal_validity="CCAO modeling protocol", acquired="false",
             reason="canonical CCAO experiment must not be changed"),
    ]
    write_csv(A / "public_enrichment/public_enrichment_audit.csv", p_rows, p_fields)

    # ---- Step 9 ATTOM hybrid: none remain D ----
    l_fields = ["jurisdiction", "hybrid_attempted", "reason",
                "eligible_local_n", "matched_n", "unmatched_n", "unique_match_rate",
                "ambiguous_match_rate", "conflicting_id_rate", "approved"]
    write_csv(A / "linkage/attom_join_feasibility.csv", [
        dict(jurisdiction="ALL_PRIMARY", hybrid_attempted="false",
             reason="No jurisdiction remained D_ATTOM_ENRICHMENT_CANDIDATE after public-enrichment reclassification. Existing ATTOM six-county benchmark is preserved separately. Cook ATTOM is used only in three-source validation, not as a replacement for CCAO or Berry outcomes.",
             eligible_local_n="", matched_n="", unmatched_n="", unique_match_rate="",
             ambiguous_match_rate="", conflicting_id_rate="", approved="false"),
    ], l_fields)
    b_fields = ["jurisdiction", "group", "variable", "matched_stat", "unmatched_stat", "note"]
    write_csv(A / "linkage/matched_unmatched_balance.csv", [
        dict(jurisdiction="ALL_PRIMARY", group="n/a", variable="n/a",
             matched_stat="", unmatched_stat="",
             note="No hybrid join executed; balance diagnostics not applicable."),
    ], b_fields)

    # ---- Step 10 Cook validation tables ----
    ccao_n, ccao_min, ccao_max = 444692, "2016-01-01", "2025-12-29"
    agr_fields = ["comparison", "field", "n_compared", "exact_agree", "tolerance_agree",
                  "note"]
    agr_rows = [
        dict(comparison="ccao_vs_berry_cmf_res_sales", field="sale_year_coverage",
             n_compared=0, exact_agree="", tolerance_agree="",
             note="CCAO training_data.parquet sales 2016-01-01 to 2025-12-29; CMF res_sales joinyr 2002-2015. Empty transaction overlap by year."),
        dict(comparison="ccao_vs_berry_cmf_res_sales", field="parcel_pin",
             n_compared=0, exact_agree="", tolerance_agree="",
             note="PIN formats are comparable (pin/pin10 vs meta_pin) but years do not overlap, so no transaction match attempted."),
        dict(comparison="ccao_vs_attom_cook", field="construction",
             n_compared="", exact_agree="", tolerance_agree="",
             note="ATTOM Cook uses Recorder TRANSFERAMOUNT, use code 385, $50k floor, pre-sale Assessor History (scripts/other_counties_benchmars.py). Different universe than CCAO residential workflow. Existing exploratory benchmark not overwritten."),
        dict(comparison="berry_cmf_vs_attom_cook", field="years",
             n_compared=0, exact_agree="", tolerance_agree="",
             note="CMF BOR files end 2015; ATTOM Cook construction is 2016-2025 recorder transfers. Empty overlap."),
    ]
    write_csv(A / "cook_validation/cook_source_agreement.csv", agr_rows, agr_fields)

    coh_fields = ["source", "n", "year_min", "year_max", "price_p10", "price_p50", "price_p90",
                  "geography", "property_class", "note"]
    coh_rows = [
        dict(source="ccao_training_data_parquet", n=ccao_n, year_min=ccao_min, year_max=ccao_max,
             price_p10="", price_p50="", price_p90="",
             geography="Cook County triads/townships in CCAO extract",
             property_class="CCAO residential modeling group",
             note="Canonical primary application; not modified."),
        dict(source="berry_cmf_res_sales_2002_2015_after_class_filter", n=641147,
             year_min="2002", year_max="2015",
             price_p10="", price_p50="", price_p90="",
             geography="Cook townships/neighborhoods in BOR extract",
             property_class="bor_class in {202-212,234,278,295,299}",
             note="IQR-filtered modeling N from reproduction; different years than CCAO."),
        dict(source="attom_cook_paper_benchmark", n=109793,
             year_min="(see paper tab:attom_samples)", year_max="2025-12-31",
             price_p10="", price_p50="", price_p90="",
             geography="Cook County FIPS 17031",
             property_class="ATTOM property use 385 single-family",
             note="Paper table: Cook 109,793 eligible / 87,834 train / 21,959 held-out. Not recomputed this pass."),
    ]
    write_csv(A / "cook_validation/cook_common_cohort_summary.csv", coh_rows, coh_fields)

    rob_fields = ["model", "source", "status", "note"]
    rob_rows = [
        dict(model="LinearRegression", source="harmonized_ccao_attom_berry",
             status="NOT_RUN",
             note="No common sale years between Berry CMF (2002-2015) and CCAO/ATTOM (2016+). 10D requires a common cohort; forcing one would mix incompatible universes."),
        dict(model="LGBMRegressor", source="harmonized_ccao_attom_berry",
             status="NOT_RUN",
             note="Same blocker. Existing paper already reports CCAO and ATTOM LightGBM separately under different protocols."),
    ]
    write_csv(A / "cook_validation/cook_baseline_source_robustness.csv", rob_rows, rob_fields)

    (A / "cook_validation/COOK_VALIDATION_REPORT.md").write_text(f"""# Cook County three-source validation

Frozen: {NOW}
Canonical CCAO experiment was not modified.

## Distinct sources

1. **CCAO research/modeling data** — `data/CCAO/2025/training_data.parquet` (N={ccao_n}, sales {ccao_min} to {ccao_max}).
2. **Berry/CMF local-government files** — Box `res_sales2002.csv`–`res_sales2015.csv` plus layout xlsx; assessed value `bor_CCAO_ass`, price `NetConsideration`.
3. **ATTOM Cook construction** — `scripts/other_counties_benchmars.py` on `data/dewey-downloads/cookcounty-2016-2025-all-features` (paper exploratory benchmark).

## 10A. Source agreement

Transaction-level matching between Berry/CMF and CCAO is **not defensible**: the CMF BOR extracts stop in 2015 and the CCAO modeling sample starts in 2016. PIN fields are conceptually compatible (`pin`/`pin10` vs `meta_pin`) but there is no shared sale year.

Berry/CMF vs ATTOM Cook is likewise disjoint (2015 vs 2016+ recorder transfers).

ATTOM vs CCAO overlap in calendar time (2016+) but **not** in institutional universe (ATTOM use code 385 + $50k floor + recorder amount vs CCAO residential workflow). Existing paper already treats them as separate constructions.

## 10B. Cohort agreement

See `cook_common_cohort_summary.csv`. Definitions differ; cohorts were not forced identical.

## 10C. External regressivity benchmark

Reproduced CMF Cook code path (class filter + IQR `reformat_data`) on 2002–2015 BOR sales: N=641,147, COD=19.14, PRD=1.039, PRB=-0.007. This is an **official-assessment / sale-price** ratio, not the paper’s model valuation ratio. No local CMF published table was available to score exact numeric match; the page “Report” is CoreLogic nationwide HTML and was not used as a target.

## 10D. Harmonized-model source validation

**Not run.** A common row set with common pre-sale features does not exist across all three sources. Running LR/LightGBM on mismatched years would not be source-robustness of the same experiment.

## Interpretation

Cook remains the paper’s primary AVM application via CCAO. Berry/CMF Cook files are useful as a **pre-2016 assessment-regressivity provenance check**, not as a substitute modeling sample. ATTOM Cook remains an exploratory sensitivity layer.
""", encoding="utf-8")

    # ---- Step 11 harmonized schema ----
    schema = {
        "schema_version": 1,
        "written_at_utc": NOW,
        "rule": "Do not force an extremely weak common schema solely for 100% coverage.",
        "CORE_HARMONIZED": {
            "purpose": "cross-jurisdiction transfer if multiple B-ready sources exist",
            "fields": [
                {"name": "sale_price", "role": "target"},
                {"name": "sale_date", "role": "split"},
                {"name": "building_area", "role": "predictor", "timing": "SAFE_AS_OF_SALE or yearly snapshot"},
                {"name": "year_built", "role": "predictor"},
                {"name": "lot_area", "role": "predictor", "optional_if_missing_locally": True},
                {"name": "bathrooms", "role": "predictor", "optional_if_missing_locally": True},
                {"name": "bedrooms", "role": "predictor", "optional_if_missing_locally": True},
                {"name": "residential_class_or_style", "role": "predictor"},
                {"name": "geography_id", "role": "predictor"},
            ],
            "jurisdictions_with_credible_core": ["st_louis_county_mo"],
            "note": "Only St. Louis County currently supplies a leakage-safe CORE set among new CMF sources. CCAO primary application uses a richer local schema and is not reduced to CORE.",
        },
        "LOCAL_ENRICHED": {
            "st_louis_county_mo": ["GRADE", "CDU", "STYLE", "STORIES", "SALEVAL", "CLASS", "LUC"],
            "ccao_primary": "params.yaml model.predictor list (unchanged)",
        },
    }
    (A / "modeling_readiness/harmonized_feature_schema.yaml").write_text(
        yaml.safe_dump(schema, sort_keys=False), encoding="utf-8")
    cov_fields = ["jurisdiction", "schema", "field", "present", "timing_class", "note"]
    cov_rows = []
    for j, present in [
        ("st_louis_county_mo", True),
        ("detroit_mi", False),
        ("philadelphia_pa", False),
        ("orleans_la", False),
        ("franklin_oh", False),
        ("cook_il_cmf", False),
        ("cook_il_ccao", True),
    ]:
        for field in ["building_area", "year_built", "lot_area", "bathrooms", "bedrooms",
                      "residential_class_or_style", "geography_id"]:
            if j == "st_louis_county_mo":
                ok = field != "lot_area"  # ACRES exists on parcel extract; confirm in smoke
                timing = "SAFE_AS_OF_SALE" if ok else "TIMING_UNRESOLVED"
            elif j == "cook_il_ccao":
                ok = True
                timing = "SAFE_AS_OF_SALE"
            else:
                ok = False
                timing = "TIMING_UNRESOLVED"
            cov_rows.append(dict(jurisdiction=j, schema="CORE_HARMONIZED", field=field,
                                 present=str(ok and present).lower(), timing_class=timing,
                                 note="pre-smoke coverage; lot_area for STL from parcel extract if joined"))
    write_csv(A / "modeling_readiness/harmonized_feature_coverage.csv", cov_rows, cov_fields)

    print("wrote audit tables", NOW)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
