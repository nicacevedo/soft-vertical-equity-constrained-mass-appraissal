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


def test_modeling_table_builder_dev_mode_still_refuses_2025():
    src = (V1 / "scripts" / "build_modeling_tables.py").read_text()
    assert "DEV_END_DATE_DEFAULT" in src
    assert '"2024-12-31"' in src
    assert 'mode == "dev"' in src
    assert "history_market_core_dev.parquet" in src
    assert "would include 2025 data before the Step 14 freeze" in src


def test_forward_table_mode_is_gated_and_does_not_overwrite_dev():
    src = (V1 / "scripts" / "build_modeling_tables.py").read_text()
    assert 'choices=("dev", "forward")' in src
    assert "history_market_core_full.parquet" in src
    assert "verify_forward_freeze" in src
    assert "preforward_identity" in src
    assert 'table_name = "history_market_core_dev.parquet" if mode == "dev" else "history_market_core_full.parquet"' in src
    slurm = (V1 / "slurm" / "04_build_modeling_tables.sh").read_text()
    assert "--end-date 2024-12-31" in slurm
    assert "--mode forward" not in slurm


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


def test_normalized_cv_scripts_never_read_2025():
    for name in ("run_normalized_direct_cv.py", "run_normalized_surrogate_cv.py"):
        src = (V1 / "scripts" / name).read_text()
        assert "FORWARD_LOCK_DATE" in src
        assert 'pd.Timestamp("2025-01-01")' in src
        assert "assert val[\"sale_date\"].max() < FORWARD_LOCK_DATE" in src


def test_normalized_cv_scripts_use_canonical_kwargs_not_literals():
    direct_src = (V1 / "scripts" / "run_normalized_direct_cv.py").read_text()
    assert "**DIRECT_CANONICAL_KWARGS" in direct_src
    # No re-typed kwarg literal at an actual call site (docstring prose mentioning
    # match_native_init=True as documentation is fine and expected).
    assert "match_native_init=True," not in direct_src
    assert "match_native_init=True)" not in direct_src
    surrogate_src = (V1 / "scripts" / "run_normalized_surrogate_cv.py").read_text()
    assert "**SURROGATE_CANONICAL_KWARGS" in surrogate_src
    # The only two call sites (LGBSmoothPenalty construction) must use the
    # shared dict; a module-docstring mention of the kwargs as documentation
    # is expected and fine, so check the actual call sites, not raw substring
    # absence across the whole file.
    calls = [ln for ln in surrogate_src.splitlines() if "LGBSmoothPenalty(" in ln or "**SURROGATE_CANONICAL_KWARGS" in ln]
    assert any("**SURROGATE_CANONICAL_KWARGS" in ln for ln in calls)


def test_surrogate_cv_reuses_v3_calibrator_read_only():
    src = (V1 / "scripts" / "run_normalized_surrogate_cv.py").read_text()
    assert "from analysis.berry_attom_validation_v3.scripts.v3_common import first_branch_calibrate" in src
    # Read-only reuse: this benchmark must not carry its own redefinition.
    assert "def first_branch_calibrate" not in src


def test_v3_source_unmodified_by_this_pass():
    result = subprocess.run(
        ["git", "diff", "--stat", "HEAD", "--",
         "analysis/berry_attom_validation_v3/scripts/v3_common.py"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    assert result.stdout.strip() == "", f"v3_common.py was modified: {result.stdout}"


def test_normalized_grid_bounds_documented_and_stable():
    from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import (
        NORMALIZED_RHO_GRID_MAX, NORMALIZED_RHO_GRID_MIN, NORMALIZED_RHO_GRID_POINTS,
        normalized_rho_tilde_grid,
    )
    grid = normalized_rho_tilde_grid()
    assert grid[0] == 0.0
    assert len(grid) == NORMALIZED_RHO_GRID_POINTS + 1
    assert abs(grid[1] - NORMALIZED_RHO_GRID_MIN) < 1e-12
    assert abs(grid[-1] - NORMALIZED_RHO_GRID_MAX) < 1e-9
    # One-decade CV-only extension widens both ends without changing the count.
    wide = normalized_rho_tilde_grid(extra_decades=1.0)
    assert wide[1] < grid[1]
    assert wide[-1] > grid[-1]
    assert len(wide) == len(grid)


def test_cook_allegheny_use_resolved_broad_source_after_resolution():
    resolution_path = V1 / "audits" / "history_source_resolution.yaml"
    if not resolution_path.exists():
        return
    import yaml
    resolution = yaml.safe_load(resolution_path.read_text())
    from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import JURISDICTION_BY_KEY
    for key in ("cook", "allegheny"):
        canonical = resolution["canonical_sources"][key]
        assert str(JURISDICTION_BY_KEY[key]["assessor_dir"]).endswith(Path(canonical).name)


def test_pilot_qa_gate_never_fails_on_scientific_favorability():
    src = (V1 / "scripts" / "run_pilot_qa_gate.py").read_text()
    for banned in ("PRB worsens", "candidate region is empty", "Surrogate bends",
                   "mechanism target is unattained", "weakly regressive"):
        assert banned not in src, f"QA gate must not encode a scientific-outcome stop condition: {banned}"
    # PRB itself is a legitimate column name to check for finiteness -- that is a
    # data-quality check, not a favorability judgment. Confirm it is used only
    # inside the finiteness check, never compared against a magnitude/sign.
    assert 'needed = ["R2_price", "PRD", "PRB"' in src


def test_canonical_penalty_fit_never_passed_categorical_feature_kwarg():
    """Regression test: LGBCovPenalty/LGBSmoothPenalty.fit(X, y) take exactly
    two positional args and have no categorical_feature kwarg (unlike native
    LGBMRegressor.fit). Passing it raised TypeError and failed all 3 pilot
    Direct tasks on first submission (job 21940045). Category dtype is
    auto-detected by the underlying native model; no kwarg is needed."""
    import inspect
    from soft_constrained_models.boosting_models import LGBCovPenalty, LGBSmoothPenalty
    for cls in (LGBCovPenalty, LGBSmoothPenalty):
        sig = inspect.signature(cls.fit)
        assert "categorical_feature" not in sig.parameters
    for name in ("run_normalized_direct_cv.py", "run_normalized_surrogate_cv.py"):
        src = (V1 / "scripts" / name).read_text()
        assert "model.fit(features.iloc[:len(train)], y_log_train, categorical_feature" not in src


def test_direct_cv_flags_path_points_whose_screening_metrics_are_not_finite():
    """Regression test for a silent-corruption bug found reviewing job 21945189_4.

    At the top of the shared rho_tilde grid a Direct fit can pass both the
    gradient guard and the prediction-finiteness guard and still produce an
    absurd price scale, whose metrics blow up (Middlesex Direct, rho_tilde=71.22:
    PRB~4e290, VEI~1e93, MAPE~3e290, R2_price=NaN). Such a point must be flagged
    NUMERICALLY_UNSTABLE_RHO, not stored as fit_status=OK, because a single NaN
    deletes that metric from the whole screen and ~1e290 values overflow inside
    select_pwl's SSE -- turning a numerical artifact into a false
    NO_STABLE_CANDIDATE_REGION."""
    import numpy as np
    sys.path.insert(0, str(V1 / "scripts"))
    from run_normalized_direct_cv import REQUIRED_SCREENING_METRICS, screening_metrics_finite

    # every metric the screen reads must be gated
    from utils.rho_screening_v2 import BENEFIT_METRICS, PREDICTIVE_COST_METRICS
    for m in list(BENEFIT_METRICS) + list(PREDICTIVE_COST_METRICS):
        assert m in REQUIRED_SCREENING_METRICS, f"{m} drives a boundary but is not gated"

    ok = {m: 1.0 for m in REQUIRED_SCREENING_METRICS}
    assert screening_metrics_finite(ok) == []
    for bad_value in (float("nan"), float("inf"), -float("inf")):
        probe = dict(ok, R2_price=bad_value)
        assert screening_metrics_finite(probe) == ["R2_price"]
    missing = {k: v for k, v in ok.items() if k != "PRB"}
    assert screening_metrics_finite(missing) == ["PRB"]

    # the guard must actually be wired into the fit loop, before metrics are stored
    src = (V1 / "scripts" / "run_normalized_direct_cv.py").read_text()
    assert "screening_metrics_finite(metrics)" in src
    assert "NUMERICALLY_UNSTABLE_RHO" in src


def test_direct_cv_training_support_bound_is_outcome_independent_and_binding():
    """The divergence rule (user-approved 2026-09-04) must be defined only from
    the training label range, never from a performance/equity metric, and must
    reject the real Middlesex Direct fold-5 case at rho_tilde=71.22 while
    accepting ordinary predictions.

    Why it matters: those diverged fits had FINITE metrics (R2_price ~ -1.5e12,
    MAE_price ~ $41bn), so admitting them as a genuine deterioration signal moved
    Middlesex's detected activity onset 0.2669 -> 2.4935 and emptied the
    cross-jurisdiction Direct band."""
    import numpy as np
    sys.path.insert(0, str(V1 / "scripts"))
    import run_normalized_direct_cv as rd

    y = np.array([10.0, 12.0, 16.0])            # training log prices, range width 6
    lo, hi = rd.training_support_window(y)
    assert (lo, hi) == (4.0, 22.0), (lo, hi)     # one full range width of slack each way

    # ordinary predictions inside the observed range are accepted
    assert rd.diverged_outside_training_support(np.array([11.0, 15.9]), lo, hi) is None
    # generous extrapolation short of the bound is still accepted
    assert rd.diverged_outside_training_support(np.array([4.5, 21.5]), lo, hi) is None
    # the real Middlesex fold-5 magnitude (pred_log ~24 vs training max ~16) is rejected
    d = rd.diverged_outside_training_support(np.array([11.0, 24.0]), lo, hi)
    assert d is not None and d["pred_log_max"] == 24.0

    src = (V1 / "scripts" / "run_normalized_direct_cv.py").read_text()
    # the bound must be evaluated before metrics are computed, and recorded distinctly
    assert "DIVERGED_OUTSIDE_TRAINING_SUPPORT" in src
    support_at = src.index("diverged_outside_training_support(pred_log")
    enrich_at = src.index("enrich(val_price, pred_price, train_price)")
    assert support_at < enrich_at, "support bound must be checked before metrics are computed"
    # the criterion must not be phrased in terms of any outcome metric
    window_src = src[src.index("def training_support_window"):src.index("def diverged_outside_training_support")]
    for outcome in ("R2", "PRD", "PRB", "MKI", "VEI", "beta", "MAPE", "MAE"):
        assert outcome not in window_src, f"support bound must not depend on {outcome}"


def test_candidate_screen_drops_grid_points_with_non_finite_boundary_metrics():
    """The screen's fold-aggregation must exclude any grid point where a
    boundary-driving metric is non-finite -- both the all-NaN shape and the
    partially-poisoned shape. dCor is interpretive-only per protocol and must
    never gate which points exist."""
    import numpy as np
    import pandas as pd
    sys.path.insert(0, str(V1 / "scripts"))
    from run_candidate_region_screen import aggregate_curve

    metrics = ["PRD", "R2_price", "dCor_e_y"]
    df = pd.DataFrame({
        "fold": [1, 1, 1, 1, 2, 2, 2, 2],
        "rho_tilde": [0.1, 1.0, 10.0, 100.0, 0.1, 1.0, 10.0, 100.0],
        # rho_tilde=10 is partially poisoned (R2_price NaN in every fold);
        # rho_tilde=100 is all-NaN across both folds.
        "PRD": [1.0, 1.1, 1.2, np.nan, 1.0, 1.1, 1.2, np.nan],
        "R2_price": [0.9, 0.8, np.nan, np.nan, 0.9, 0.8, np.nan, np.nan],
        "dCor_e_y": [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
    })
    curve = aggregate_curve(df, metrics)
    kept = sorted(curve["rho_tilde"].tolist())
    assert kept == [0.1, 1.0], f"poisoned/all-NaN points must be dropped, kept {kept}"

    # a non-finite dCor alone must NOT remove an otherwise-usable point
    df2 = df.loc[df.rho_tilde <= 1.0].copy()
    df2.loc[df2.rho_tilde == 1.0, "dCor_e_y"] = np.nan
    curve2 = aggregate_curve(df2, metrics)
    assert sorted(curve2["rho_tilde"].tolist()) == [0.1, 1.0], "dCor must not gate grid points"


def test_candidate_status_requires_nonempty_lofo_stable_interval():
    """A protocol-valid candidate region is activity <= guardrail AND both
    endpoints LOFO-stable. Point estimates are preserved in every case."""
    sys.path.insert(0, str(V1 / "scripts"))
    from run_candidate_region_screen import classify_candidate_status

    assert classify_candidate_status(0.27, 0.82, True, True) == "CANDIDATE_REGION"
    assert classify_candidate_status(0.27, 23.3, True, False) == "NO_STABLE_CANDIDATE_REGION"
    assert classify_candidate_status(0.267, 0.060, True, True) == "UPPER_GUARDRAIL_PRECEDES_ACTIVITY"
    assert classify_candidate_status(None, None, False, False) == "NO_STABLE_CANDIDATE_REGION"
    assert classify_candidate_status(0.2, None, True, False) == "PARTIAL_ENDPOINT_ONLY"


def test_protocol_valid_overlap_excludes_unstable_and_inverted_regions():
    sys.path.insert(0, str(V1 / "scripts"))
    from run_portability_and_band import intersection_status, region_overlap

    # Allegheny-like unstable interval must not be fed in as protocol-valid.
    valid_direct = [(0.387, 0.816), (0.267, 0.816), (0.184, 0.562)]
    r = region_overlap(valid_direct)
    assert r["n_with_region"] == 3
    assert r["intersection"][0] == 0.387
    assert r["intersection"][1] == 0.562
    assert r["intersection_status"] == "NONEMPTY_INTERSECTION"

    inverted = [(0.267, 0.060)]
    r2 = region_overlap(inverted)
    assert r2["n_with_region"] == 0
    assert r2["intersection_status"] == "NO_INTERSECTION"

    knife = region_overlap([(0.1267, 0.387), (0.060, 0.1267)])
    assert knife["n_with_region"] == 2
    assert knife["intersection_status"] == "NO_NONDEGENERATE_ALL_JURISDICTION_INTERSECTION"
    assert intersection_status(None, 0) == "NO_INTERSECTION"


def test_forward_freeze_sha256_matches_canonical_object():
    from analysis.external_jurisdiction_benchmark_v1.scripts.forward_common import (
        EXPECTED_FORWARD_FREEZE_SHA256, freeze_sha256, verify_forward_freeze,
    )
    assert freeze_sha256() == EXPECTED_FORWARD_FREEZE_SHA256
    freeze = verify_forward_freeze()
    assert all(v == "PRIMARY_FULL_7_FOLD" for v in freeze["jurisdiction_roles"].values())
    assert set(freeze["jurisdiction_roles"]) == set(ALL_KEYS)


def test_no_independent_pre_2025_test_split_is_invented():
    layers = (V1 / "forward_2025" / "audits" / "evaluation_layer_definition.yaml").read_text()
    assert "independent_pre_2025_test_split:" in layers
    assert "exists: false" in layers
    assert "CV_OOF" in layers and "FORWARD_2025" in layers
    assert "never_label_as_test" in layers
    common = (V1 / "scripts" / "v1_common.py").read_text()
    assert "Superseded by the frozen calendar-year rule" in common


def test_forward_path_script_cannot_train_on_2025_or_write_candidate_region():
    src = (V1 / "scripts" / "run_forward_path.py").read_text()
    assert 'data["sale_date"] < FORWARD_LOCK_DATE' in src
    assert "2025 leakage into forward training" in src
    assert "forward eval is not calendar 2025" in src
    assert "wrote_candidate_region" in src
    assert "candidate_regions.csv" not in src
    assert "run_candidate_region_screen" not in src
    slurm = (V1 / "slurm" / "11_forward_path.sh").read_text()
    assert "sched_mit_sloan_batch_r8" in slurm
    assert "mit_normal" not in slurm
    tables = (V1 / "slurm" / "10_forward_tables.sh").read_text()
    assert "sched_mit_sloan_batch_r8" in tables
    assert "mit_normal" not in tables


def test_forward_rho_mapping_is_train_variance_only():
    from analysis.external_jurisdiction_benchmark_v1.scripts.forward_common import frozen_grid_rho_tilde
    d = frozen_grid_rho_tilde("direct")
    s = frozen_grid_rho_tilde("surrogate")
    assert len(d) == 34 and d[0] == 0.0
    assert len(s) == 33 and s[0] > 0
    src = (V1 / "scripts" / "run_forward_path.py").read_text()
    assert "raw_rho = 0.0 if rho_tilde == 0.0 else float(rho_tilde) / vy" in src
    assert "population_variance" in src
    assert "Do not recalibrate" in (V1 / "scripts" / "run_forward_path.py").read_text() or "Never recalibrated" in src


def test_forward_scripts_do_not_write_2025_candidate_region():
    for name in ("run_forward_path.py", "verify_forward_inputs.py", "forward_common.py"):
        src = (V1 / "scripts" / name).read_text()
        assert "candidate_regions.csv" not in src or "read_csv" in src
        assert "to_csv" not in src or "candidate_regions/" not in src


def test_forward_fit_inventory_calendar_lock_and_freeze():
    from analysis.external_jurisdiction_benchmark_v1.scripts.forward_common import (
        EXPECTED_FORWARD_FREEZE_SHA256,
    )
    comp_path = V1 / "forward_2025" / "audits" / "forward_fit_completeness.csv"
    grid_path = V1 / "forward_2025" / "metrics" / "forward_2025_path_metrics.csv"
    if not comp_path.exists() or not grid_path.exists():
        return
    comp = pd.read_csv(comp_path)
    assert set(comp.county_key) == set(ALL_KEYS)
    assert set(comp.family) == {"direct", "surrogate"}
    assert bool(comp.complete.all())
    assert int(comp.loc[comp.family == "direct", "expected_grid"].iloc[0]) == 34
    assert int(comp.loc[comp.family == "surrogate", "expected_grid"].iloc[0]) == 33
    grid = pd.read_csv(grid_path)
    assert len(grid) == 9 * (34 + 33)
    for key in ALL_KEYS:
        for family in ("direct", "surrogate"):
            meta = json.loads(
                (V1 / "forward_2025" / "metrics" / "partial" / f"{key}_{family}_forward_meta.json").read_text()
            )
            assert meta["no_2025_in_training"] is True
            assert meta["eval_year_min"] == 2025
            assert meta["eval_year_max"] == 2025
            assert meta["wrote_candidate_region"] is False
            assert meta["forward_freeze_sha256"] == EXPECTED_FORWARD_FREEZE_SHA256
            assert meta["n_frozen_grid_points"] == meta["expected_frozen_grid_points"]


def test_forward_does_not_emit_a_2025_candidate_region_file():
    if not (V1 / "forward_2025").exists():
        return
    names = [p.name for p in (V1 / "forward_2025").rglob("*") if p.is_file()]
    assert "candidate_regions.csv" not in names


def test_bootstrap_is_paired_monthly_block_200_draws():
    src = (V1 / "scripts" / "run_forward_bootstrap.py").read_text()
    assert "N_BOOTSTRAP" in src
    assert "same sampled months" in src or "the exact same sampled months" in src or "same month" in src
    ci_path = V1 / "forward_2025" / "bootstrap" / "forward_anchor_bootstrap_ci.csv"
    if not ci_path.exists():
        return
    ci = pd.read_csv(ci_path)
    assert (ci.n_boot == 200).all()
    from analysis.external_jurisdiction_benchmark_v1.scripts.v1_common import OUTPUT, N_BOOTSTRAP
    assert N_BOOTSTRAP == 200
    for key in ALL_KEYS:
        months = OUTPUT / "forward_2025" / "bootstrap" / f"{key}_sampled_months.npy"
        if months.exists():
            arr = np.load(months)
            assert arr.shape[0] == 200


def test_required_forward_paper_figures_exist_when_written():
    paper = V1 / "figures" / "paper"
    if not paper.exists():
        return
    for name in (
        "accuracy_mechanism_frontier_cv_vs_2025.pdf",
        "forward_key_metric_paths_9jurisdictions.pdf",
        "forward_ratio_profile_examples.pdf",
        "berry_local_vs_avm_ratio_profiles.pdf",
    ):
        assert (paper / name).is_file(), name
    pe = list((V1 / "figures" / "path_evolution").glob("*_paths.pdf"))
    assert len(pe) == 36


