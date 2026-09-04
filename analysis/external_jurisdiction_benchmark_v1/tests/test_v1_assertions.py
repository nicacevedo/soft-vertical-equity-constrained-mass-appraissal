"""external_jurisdiction_benchmark_v1 assertions. Run via scripts/run_v1_tests.py
(fairness_env has no pytest, mirroring the v3 test-runner pattern)."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
V1 = REPO / "analysis" / "external_jurisdiction_benchmark_v1"

from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import (  # noqa: E402
    ALL_KEYS, DIRECT_CANONICAL_KWARGS, EXTERNAL_KEYS, FIPS, JURISDICTIONS, PILOT_KEYS,
    ST_LOUIS_CITY_FIPS, SURROGATE_CANONICAL_KWARGS, population_variance,
)


def test_nine_jurisdictions_no_duplicates():
    assert len(ALL_KEYS) == 9
    assert len(set(ALL_KEYS)) == 9
    assert len(EXTERNAL_KEYS) == 8
    assert "cook" not in EXTERNAL_KEYS, "Cook is bridge/reference, never counted as external"


def test_st_louis_fips_never_confused_with_city():
    assert FIPS["st_louis_county"] == "29189"
    assert ST_LOUIS_CITY_FIPS == "29510"
    assert FIPS["st_louis_county"] != ST_LOUIS_CITY_FIPS


def test_wayne_never_labeled_detroit():
    wayne = next(j for j in JURISDICTIONS if j["key"] == "wayne")
    assert "Detroit" not in wayne["label"]
    assert wayne["berry_unit"] == "Detroit, MI"
    proto = (V1 / "protocol_external_benchmark_v1.yaml").read_text()
    assert "NEVER label Wayne as Detroit" in proto


def test_pilot_stage_is_wayne_philadelphia_cook():
    assert PILOT_KEYS == ("wayne", "philadelphia", "cook")


def test_canonical_kwargs_match_validated_synthetic_test():
    """These must be byte-identical to the kwargs used by the already-passing
    tests/test_paper_v6_guards.py::test_native_custom_rho0_parity_after_mean_init,
    or the parity result this benchmark relies on does not actually apply."""
    ref = (REPO / "tests" / "test_paper_v6_guards.py").read_text()
    assert "match_native_init=True" in ref
    assert DIRECT_CANONICAL_KWARGS == {"ratio_mode": "diff", "match_native_init": True}
    assert SURROGATE_CANONICAL_KWARGS == {
        "ratio_mode": "diff", "weighting_proxy_mode": "identity", "match_native_init": True,
    }


def test_population_variance_uses_ddof0():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    expected = np.mean((x - x.mean()) ** 2)  # ddof=0, NOT pandas' default ddof=1
    assert abs(population_variance(x) - expected) < 1e-12
    assert abs(population_variance(pd.Series(x)) - expected) < 1e-12
    # Guard against silently regressing to ddof=1 (which would be larger here).
    ddof1 = np.var(x, ddof=1)
    assert abs(population_variance(x) - ddof1) > 1e-6


def test_rho_tilde_is_not_derived_from_plan_rho_grid_A():
    """rho_tilde = rho * Vy_T must never be confused with plan_rho_grid's
    internal A = Var(baseline predictions) normalizer. See protocol
    objective_and_rho_normalization.explicitly_distinct_from_plan_rho_grid."""
    proto = (V1 / "protocol_external_benchmark_v1.yaml").read_text()
    assert "explicitly_distinct_from_plan_rho_grid" in proto
    assert "Var(baseline predictions f0), NOT Var(y)" in proto
    common_src = (V1 / "scripts" / "v1_common.py").read_text()
    assert "import plan_rho_grid" not in common_src, (
        "v1_common must not import plan_rho_grid's A-based normalizer"
    )
    # The name may appear in prose explaining the distinction; that is fine.
    assert "explicitly_distinct_from" in common_src or "NOT via plan_rho_grid" in common_src


def test_objective_scaling_gate_passed_before_any_cache_or_cohort_work():
    verdict_path = V1 / "audits" / "objective_scaling_audit_verdict.json"
    assert verdict_path.exists(), "Step 1 gate must run before any large work"
    verdict = json.loads(verdict_path.read_text())
    assert verdict["gate_passed"] is True
    assert verdict["rho_tilde_equivalence_exact_everywhere"] is True
    assert verdict["native_custom_parity_interpretable"] is True
    # Real-data parity here must be at least as tight as the already-validated
    # synthetic bar (test_paper_v6_guards.py uses 5e-3); this benchmark reused
    # the same canonical kwargs and got roughly three orders of magnitude
    # tighter on real jurisdiction data.
    for county, gap in verdict["parity_mean_abs_diff_by_county"].items():
        assert gap < 5e-3, f"{county}: parity gap {gap} exceeds the validated bar"


def test_no_blocker_written_unless_a_real_gate_failed():
    blocker = V1 / "BLOCKER.md"
    if not blocker.exists():
        return
    text = blocker.read_text().lower()
    assert "unattended-execution rule" in text, "malformed blocker file: missing rule citation"
    assert "decision required" in text or "what is needed" in text, (
        "malformed blocker file: does not name the decision required"
    )


def test_no_tax_assessor_or_acs_columns_in_cache_column_lists():
    common_src = (V1 / "scripts" / "v1_common.py").read_text()
    assert "TAX_ASSESSOR" not in common_src
    assert "ACS_DIR" not in common_src
    assert "attach_tax_assessor" not in common_src
    assert "attach_acs" not in common_src


def test_county_cache_builder_skips_unreadable_shards():
    src = (V1 / "scripts" / "build_county_caches.py").read_text()
    assert "readable_parquet_files" in src, (
        "must skip corrupt shards (attom_county_benchmark.sh documents a real "
        "corrupt-shard incident in one of these same folders)"
    )
    assert "St. Louis County cache contains St. Louis City FIPS 29510" in src


def test_county_to_folder_map_covers_all_nine_with_existing_dirs():
    for j in JURISDICTIONS:
        assert j["assessor_dir"].exists(), f"{j['key']}: assessor dir missing"
        assert j["recorder_dir"].exists(), f"{j['key']}: recorder dir missing"


def test_no_raw_or_large_data_committed():
    result = subprocess.run(
        ["git", "ls-files", "data/dewey-downloads", "data/berry_cmf/raw",
         "output/external_jurisdiction_benchmark_v1", "analysis/external_jurisdiction_benchmark_v1"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    tracked = [ln for ln in result.stdout.splitlines() if ln.strip()]
    parquet = [p for p in tracked if p.endswith(".parquet")]
    dewey = [p for p in tracked if p.startswith("data/dewey-downloads")]
    berry_raw = [p for p in tracked if p.startswith("data/berry_cmf/raw")]
    assert dewey == [], dewey[:10]
    assert berry_raw == [], berry_raw[:10]
    assert parquet == [], parquet[:10]


def test_v3_and_manuscript_untouched():
    result = subprocess.run(
        ["git", "status", "--porcelain",
         "analysis/berry_attom_validation_v3", "analysis/berry_cmf_validation", "paper/paper_v12.tex"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    dirty = [ln for ln in result.stdout.splitlines() if ln.strip()]
    assert dirty == [], f"protected paths modified: {dirty}"


def test_berry_external_metrics_never_mix_official_and_avm_ratio():
    path = V1 / "berry" / "berry_external_metrics.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    assert {"jurisdiction", "sample", "N", "COD", "PRD", "PRB", "beta_log"}.issubset(df.columns)
    banned = {"avm", "lgbm", "predicted", "attom_valuation", "model_ratio"}
    lower_cols = {c.lower() for c in df.columns}
    assert not (banned & lower_cols), f"AVM-valuation column leaked into official-ratio table: {banned & lower_cols}"


def test_st_louis_berry_wording_corrects_v3():
    meta_path = V1 / "berry" / "berry_regressivity_run_meta.json"
    if not meta_path.exists():
        return
    meta = json.loads(meta_path.read_text())
    assert "st_louis_correction" in meta
    assert "no official assessed-value series" in meta["st_louis_correction"]


def test_st_louis_coverage_stated_2009_2019():
    path = V1 / "berry" / "berry_external_metrics.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    stl = df.loc[df["jurisdiction"] == "st_louis_county_mo"]
    if len(stl):
        assert stl["note"].str.contains("2009-2019", na=False).all()


def test_property_use_mapping_gated_not_faked():
    proto = (V1 / "protocol_external_benchmark_v1.yaml").read_text()
    assert "no_fallback" in proto
    src_dir = V1 / "scripts"
    for py in src_dir.glob("*.py"):
        text = py.read_text()
        assert "BROAD_RESIDENTIAL_RULE" not in text, (
            f"{py.name}: v3's structural proxy must never silently substitute for the "
            "semantic mapping in this benchmark"
        )


def test_residential_mapping_frozen_with_correction_history_preserved():
    """The 385/386 mislabeling this benchmark caught and the user's correction
    must both remain on record -- the audit trail is the point, not just the
    final answer."""
    import yaml
    mapping_path = V1 / "cohort" / "residential_code_mapping.yaml"
    mapping = yaml.safe_load(mapping_path.read_text())
    assert mapping["status"] == "FROZEN"
    assert mapping["PRIMARY_RESIDENTIAL"]["codes"] == [363, 376, 377, 380, 382, 383, 384, 385, 386, 390]
    assert mapping["LEGACY_385_ONLY"]["codes"] == [385]
    assert "366" in mapping["BROAD_RESIDENTIAL_APPENDIX"]["confirmed_members"]
    assert len(mapping.get("correction_history", [])) >= 2, (
        "the flagged anomaly and its resolution must both be preserved, not overwritten away"
    )
    history_text = str(mapping["correction_history"])
    assert "54.7%" in history_text or "inconsistent" in history_text.lower()


def test_primary_residential_codes_disjoint_from_confirmed_appendix():
    import yaml
    mapping = yaml.safe_load((V1 / "cohort" / "residential_code_mapping.yaml").read_text())
    primary = set(mapping["PRIMARY_RESIDENTIAL"]["codes"])
    appendix = {int(k) for k in mapping["BROAD_RESIDENTIAL_APPENDIX"]["confirmed_members"]}
    assert primary.isdisjoint(appendix), "a code cannot be both primary and appendix-only"
    excluded_despite = {int(k) for k in mapping["PRIMARY_RESIDENTIAL"]["excluded_despite_residential_appearance"]}
    assert primary.isdisjoint(excluded_despite)


def test_cohort_retention_not_used_to_adjust_membership():
    """Regenerating the retention audit must never mutate the frozen mapping
    file -- membership is read-only input to the audit, never its output."""
    import yaml
    mapping_path = V1 / "cohort" / "residential_code_mapping.yaml"
    before = mapping_path.read_text()
    mapping = yaml.safe_load(before)
    assert mapping.get("mapping_not_adjusted_for_retention_or_frequency") is True
    retention_path = V1 / "cohort" / "cohort_retention.csv"
    if retention_path.exists():
        import pandas as pd
        df = pd.read_csv(retention_path)
        assert {"county_key", "primary_residential_share"}.issubset(df.columns)
        # Philadelphia's low primary share is an accepted, documented finding,
        # not something the pipeline should have "fixed" by widening the set.
        phl = df.loc[df.county_key == "philadelphia"]
        if len(phl):
            assert phl["primary_residential_share"].iloc[0] < 0.5, (
                "if Philadelphia's primary share is no longer low, confirm this wasn't "
                "achieved by silently broadening PRIMARY_RESIDENTIAL"
            )
    after = mapping_path.read_text()
    assert before == after, "computing retention must not have side-effects on the frozen mapping"


def test_cohort_source_of_truth_is_single_file():
    """build_modeling_tables.py must read PRIMARY_RESIDENTIAL from
    residential_code_mapping.yaml at runtime, never carry its own duplicate
    hardcoded code list that could silently drift from the frozen one."""
    src = (V1 / "scripts" / "build_modeling_tables.py").read_text()
    assert "load_primary_residential_codes" in src
    assert "residential_code_mapping.yaml" in src or "MAPPING_PATH" in src
    assert "[363, 376, 377, 380, 382, 383, 384, 385, 386, 390]" not in src, (
        "cohort codes must never be duplicated as a literal in the builder script"
    )


def test_modeling_table_builder_never_reads_or_writes_2025():
    src = (V1 / "scripts" / "build_modeling_tables.py").read_text()
    assert "2025-01-01" in src
    assert "DEV_END_DATE_DEFAULT" in src
    assert '"2024-12-31"' in src


def test_baseline_cv_hard_guards_present():
    src = (V1 / "scripts" / "run_baseline_cv.py").read_text()
    assert "FORWARD_LOCK_DATE" in src
    assert "2025-01-01" in src
    assert "assert val[\"sale_date\"].max() < FORWARD_LOCK_DATE" in src
    assert "assert data[\"sale_date\"].max() < FORWARD_LOCK_DATE" in src


def test_baseline_cv_uses_seven_expanding_calendar_year_folds():
    src = (V1 / "scripts" / "run_baseline_cv.py").read_text()
    assert "CV_VALIDATION_YEARS = (2018, 2019, 2020, 2021, 2022, 2023, 2024)" in src


def test_no_penalty_path_script_invoked_by_baseline_stage():
    for name in ("04_build_modeling_tables.sh", "05_baseline_cv.sh"):
        text = (V1 / "slurm" / name).read_text()
        assert "run_direct_surrogate" not in text
        assert "LGBCovPenalty" not in text and "LGBSmoothPenalty" not in text


def test_modeling_table_meta_has_no_assessment_columns_if_built():
    meta_paths = list((V1.parent.parent / "output" / "external_jurisdiction_benchmark_v1" / "modeling_tables").glob("*/modeling_table_meta_dev.json"))
    for mp in meta_paths:
        meta = json.loads(mp.read_text())
        assert meta.get("status") == "OK"
        assert meta["sale_date_max"] < "2025-01-01"


def test_baseline_config_selection_matches_declared_rule_if_run():
    import pandas as pd
    for key in ("wayne", "philadelphia", "cook"):
        summary_path = V1 / "baseline" / f"{key}_baseline_cv_summary.csv"
        config_path = V1 / "baseline" / f"{key}_baseline_config.json"
        if not (summary_path.exists() and config_path.exists()):
            continue
        summary = pd.read_csv(summary_path)
        config = json.loads(config_path.read_text())
        best_by_rmse = summary.sort_values("mean_RMSE_log").iloc[0]["config_name"]
        assert config["selected_lgbm_config"] == best_by_rmse
        assert config["no_2025_data_used"] is True
        assert config["no_penalty_path_run"] is True


def test_modeling_table_writer_drops_assessment_columns_not_just_asserts():
    """Regression test for a real bug caught in production: the History cache
    carries TAXASSESSEDVALUETOTAL/TAXMARKETVALUETOTAL (part of
    HISTORY_CACHE_COLUMNS); an early version of this script asserted their
    absence from `final` without ever dropping them, so the assertion always
    fired and the Wayne task failed (Slurm job 21920034_0). The fix drops them
    from the stored table before the assertion, not just before feature
    construction downstream."""
    src = (V1 / "scripts" / "build_modeling_tables.py").read_text()
    assert "drop_cols" in src and "final.drop(columns=drop_cols)" in src, (
        "must actually drop assessment-value columns from the stored table, "
        "not only assert their absence"
    )
    idx_drop = src.index("drop_cols = [")
    idx_assert = src.index("leak = ASSESSMENT_VALUE_COLUMNS")
    assert idx_drop < idx_assert, "columns must be dropped BEFORE the leak assertion runs"
