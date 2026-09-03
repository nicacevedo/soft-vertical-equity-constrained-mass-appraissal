"""v3 assertions. Run: pytest analysis/berry_attom_validation_v3/tests"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
V3 = REPO / "analysis" / "berry_attom_validation_v3"

from analysis.berry_attom_validation_v2.scripts.apn_normalize import normalize_apn_raw
from analysis.berry_attom_validation_v3.scripts.v3_common import (
    ASSESSMENT_VALUE_COLUMNS, BROAD_RESIDENTIAL_RULE, FIPS, ST_LOUIS_CITY_FIPS,
    SURROGATE_NOISE_FLOOR, SURROGATE_NOISE_MULTIPLE, estimate_surrogate_noise_floor,
    first_branch_calibrate, surrogate_rho_grid,
)

COUNTY_KEYS = ("wayne", "philadelphia", "st_louis_county")


def _text_files(*folders: Path) -> str:
    parts = []
    for folder in folders:
        if not folder.exists():
            continue
        for p in folder.rglob("*"):
            if p.suffix in {".py", ".yaml", ".md", ".sh"} and p.is_file():
                if p.name.startswith("test_"):
                    continue
                parts.append(p.read_text(errors="ignore"))
    return "\n".join(parts)


def test_no_forest_city_or_datacenter_in_v3_operational_files():
    blob = _text_files(V3 / "scripts", V3 / "slurm", V3 / "tests").lower()
    for term in ("forest city", "forest_city", "forect city", "prineville", "kfqd", "datacenter"):
        assert term not in blob
    proto = (V3 / "protocol_v3.yaml").read_text()
    assert "THIS_IS_NOT_A_DATA_CENTER_TASK" in proto


def test_fips():
    assert FIPS["wayne"] == "26163"
    assert FIPS["philadelphia"] == "42101"
    assert FIPS["st_louis_county"] == "29189"
    assert ST_LOUIS_CITY_FIPS == "29510"
    assert FIPS["st_louis_county"] != ST_LOUIS_CITY_FIPS


def test_apn_deterministic():
    assert normalize_apn_raw("03G-110095") == normalize_apn_raw(" 03g 110095 ")
    assert normalize_apn_raw("03G-110095") == "03G110095"


def test_one_to_many_not_collapsed_in_source():
    src = (V3 / "scripts/link_berry_attom.py").read_text()
    assert "AMBIGUOUS_APN" in src
    assert 'drop_duplicates("apn_norm")' not in src
    assert "len(ids) > 1" in src


def test_history_strictly_before_sale_in_source():
    src = (V3 / "scripts/link_berry_attom.py").read_text() + (V3 / "scripts/build_modeling_tables.py").read_text()
    assert 'assert (matched["assessed_through"] < matched["sale_date"]).all()' in src
    assert "allow_exact_matches=False" in (REPO / "scripts/other_counties_benchmars.py").read_text()


def test_prior_recorder_strictly_before_target():
    src = (V3 / "scripts/build_modeling_tables.py").read_text()
    assert "recorder_prior_sale_age_years" in src
    assert "non-positive prior-sale age" in src


def test_no_assessment_predictors_primary():
    src = (V3 / "scripts/run_prefreeze_baselines.py").read_text()
    assert "include_assessment_values" in src or "include_prior, False)" in src
    assert "ASSESSMENT_VALUE_COLUMNS" in src
    primary_example = {"AREABUILDING", "YEARBUILT", "sale_year", "recorder_prior_sale_amount"}
    assert primary_example.isdisjoint(ASSESSMENT_VALUE_COLUMNS)


def test_test_block_not_scored_before_freeze():
    pre = (V3 / "scripts/run_prefreeze_baselines.py").read_text()
    assert '"test_block_scored": False' in pre
    assert "n_test_held_out_unscored" in pre
    assert "model.predict(features.iloc[split:])" not in pre
    assert "test_predictions_written" in pre
    final = (V3 / "scripts/run_final_baselines.py").read_text()
    assert "Freeze file missing" in final
    proto = (V3 / "protocol_v3.yaml").read_text()
    assert "held_out_test_before_freeze: FORBIDDEN" in proto


def test_freeze_before_heldout_and_positive_rho_runtime():
    freeze = V3 / "panel_freeze" / "final_panel_freeze_v3.yaml"
    held = list((REPO / "output/berry_attom_validation_v3/final_models").glob("*/heldout_predictions.parquet"))
    rho = list((REPO / "output/berry_attom_validation_v3/method_transfer").glob("*/*heldout.parquet"))
    if held or rho:
        assert freeze.exists(), "held-out or positive-rho artifacts exist without freeze"


def test_freeze_before_positive_rho_source():
    src = (V3 / "scripts/run_direct_surrogate.py").read_text()
    assert "did not authorize Direct/Surrogate" in src
    assert "first_branch_calibrate" in src
    assert 'sort_values("achieved_reduction")' not in src
    assert "np.interp" not in src


def test_lr_training_only():
    src = (V3 / "scripts/run_prefreeze_baselines.py").read_text()
    assert "lr_feature_groups" in src
    assert "train = features.iloc[:n_dev]" in src
    common = (V3 / "scripts/v3_common.py").read_text()
    assert "max_levels: int = 32" in common


def test_lr_drops_high_cardinality_strings():
    import pandas as pd
    from analysis.berry_attom_validation_v3.scripts.v3_common import lr_feature_groups
    train = pd.DataFrame({
        "AREABUILDING": list(range(40)),
        "PROPERTYADDRESSCITY": [f"C{i}" for i in range(40)],
    })
    numeric, cats, dropped = lr_feature_groups(train, ["PROPERTYADDRESSCITY"], max_levels=32)
    assert "AREABUILDING" in numeric
    assert "PROPERTYADDRESSCITY" in dropped
    assert "PROPERTYADDRESSCITY" not in cats


def test_stl_window_predeclared():
    proto = (V3 / "protocol_v3.yaml").read_text()
    assert 'attom_linkage_cohort_start: "2005-01-01"' in proto
    src = (V3 / "scripts/link_berry_attom.py").read_text()
    assert "STL_LINKAGE_START" in src
    assert "1975" not in src or "1975-2019" not in src


def test_ratio_metrics_uses_one_dimensional_sale_price():
    src = (V3 / "scripts/link_berry_attom.py").read_text()
    assert '"SALE_PRICE": pd.to_numeric(frame["berry_sale_price"]' in src
    assert 'rename(columns={"berry_sale_price": "SALE_PRICE"' not in src


def test_modeling_retention_audit_exists_in_source():
    src = (V3 / "scripts/build_modeling_tables.py").read_text()
    assert "retention_table" in src
    assert "p_final_model" in src


def test_surrogate_first_contiguous_branch():
    rhos = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    # intended increase then a later reverse (S-shape / later branch)
    achieved = np.array([0.10, 0.30, 0.55, 0.40, 0.90])
    frozen, branch = first_branch_calibrate(rhos, achieved, noise_floor=0.01)
    assert list(branch["rho"]) == [1.0, 2.0, 4.0]
    unattained = frozen.loc[frozen["requested_reduction"].eq(0.90)].iloc[0]
    assert unattained["status"] == "UNATTAINED"
    hit = frozen.loc[frozen["requested_reduction"].eq(0.25)].iloc[0]
    assert hit["status"] == "interpolated_first_branch"
    assert 1.0 < hit["rho"] < 2.0


def test_surrogate_branch_does_not_open_on_sub_noise_reduction():
    """Pass 1's St. Louis failure mode: a noise-level positive reduction at
    negligible rho opened a one-point branch that the next point closed."""
    rhos = np.array([1e-6, 1e-4, 1e-2, 1.0, 10.0, 100.0])
    achieved = np.array([0.008, 0.004, 0.006, 0.20, 0.45, 0.70])
    frozen, branch = first_branch_calibrate(rhos, achieved, noise_floor=0.01)
    assert 1e-6 not in set(branch["rho"]), "sub-noise-floor point opened a branch"
    assert len(branch) == 3
    assert frozen.loc[frozen["requested_reduction"].eq(0.25)].iloc[0]["status"] == (
        "interpolated_first_branch"
    )
    # Guard the floor itself: 0.008 must sit below it, or this test proves nothing.
    assert 0.008 < SURROGATE_NOISE_FLOOR


def test_surrogate_unattained_distinguishes_ceiling_from_reversal():
    ceiling, _ = first_branch_calibrate(
        np.array([1.0, 10.0, 100.0]), np.array([0.05, 0.18, 0.32]), noise_floor=0.01,
    )
    assert ceiling["branch_terminated_by"].eq("GRID_CEILING").all()
    assert ceiling.loc[ceiling["requested_reduction"].eq(0.50)].iloc[0]["unattained_reason"] == (
        "GRID_CEILING"
    )
    bend, _ = first_branch_calibrate(
        np.array([1.0, 2.0, 4.0, 8.0]), np.array([0.10, 0.30, 0.55, 0.05]), noise_floor=0.01,
    )
    assert bend["branch_terminated_by"].eq("MATERIAL_REVERSAL").all()
    never, _ = first_branch_calibrate(
        np.array([1.0, 2.0]), np.array([0.001, 0.002]), noise_floor=0.01,
    )
    assert never["branch_terminated_by"].eq("NEVER_STARTED").all()


def test_surrogate_grid_reaches_beyond_direct_rho():
    """A grid ceiling below the Direct 97% anchor is what made pass 1's
    strong-penalty targets UNATTAINED for the wrong reason."""
    for key in COUNTY_KEYS:
        plan = V3 / "method_transfer" / key / "direct_rho_from_pretest.csv"
        if not plan.exists():
            continue
        max_direct = float(pd.read_csv(plan)["rho"].max())
        grid = surrogate_rho_grid(max_direct)
        assert grid.max() >= max_direct, f"{key}: surrogate grid stops below Direct rho"
        assert grid.min() > 0


def test_surrogate_pass1_preserved_when_pass2_exists():
    """Pass-1 evidence must never be partially lost, and a county with no pass-1
    snapshot at all must be one whose pass 1 failed before writing -- which the
    recalibration log has to say. Wayne is that case: 21882241_0 died in dcor's
    numba cache, so there is genuinely nothing to preserve."""
    log = (V3 / "panel_freeze" / "SURROGATE_RECALIBRATION_LOG.md").read_text()
    names = ("surrogate_pass1_heldout.csv", "surrogate_pass1_branch_curve.csv",
             "surrogate_pass1_rho_first_branch.csv")
    for key in COUNTY_KEYS:
        ana = V3 / "method_transfer" / key
        current = ana / "surrogate_heldout.csv"
        if not current.exists():
            continue
        frame = pd.read_csv(current)
        if "surrogate_pass" not in frame.columns or int(frame["surrogate_pass"].iloc[0]) < 2:
            continue
        present = [n for n in names if (ana / n).exists()]
        if present:
            assert len(present) == len(names), f"{key}: pass-1 evidence partially discarded"
            continue
        failed = [
            ln for ln in log.splitlines()
            if key in ln and "FAILED" in ln
        ]
        assert failed, (
            f"{key}: no pass-1 snapshot and the recalibration log does not record a "
            "failed pass 1 for it"
        )


def test_surrogate_recalibration_is_disclosed():
    proto = (V3 / "protocol_v3.yaml").read_text()
    assert "calibration_amendment" in proto
    assert "second_heldout_look_disclosed: true" in proto
    log = V3 / "panel_freeze" / "SURROGATE_RECALIBRATION_LOG.md"
    assert log.exists(), "surrogate pass 2 requires a written recalibration log"


def test_broad_residential_rule_uses_no_outcome_quantity():
    """The sensitivity cohort rule must be structural. If a price, sale, or
    model-performance term ever enters it, the cohort stops being a neutral
    probe of the primary filter."""
    for field in BROAD_RESIDENTIAL_RULE:
        for banned in ("price", "sale", "r2", "prb", "cod", "beta", "ratio", "resid"):
            assert banned not in field.lower(), field
    src = (V3 / "scripts/build_modeling_tables.py").read_text()
    body = src[src.index("def broad_residential_mask"):src.index("def use_set_retention_by_decile")]
    for banned in ("sale_price", "median_sale_price", "R2", "PRB"):
        assert banned not in body, f"{banned} leaked into broad_residential_mask"


def test_broad_residential_excludes_zero_building_area():
    """Rule v1 admitted codes whose median AREABUILDING is 0 -- land, not a
    dwelling. A cohort described as carrying dwelling structure must exclude
    them, so the rule requires a positive median area, not merely a present one."""
    import pandas as pd

    from analysis.berry_attom_validation_v3.scripts.build_modeling_tables import (
        broad_residential_mask,
    )
    assert BROAD_RESIDENTIAL_RULE["min_median_area_building"] > 0
    assert BROAD_RESIDENTIAL_RULE["rule_version"] >= 2
    profile = pd.DataFrame([
        # a dwelling code
        {"use_code": "385", "share_area_building_present": 1.0, "median_area_building": 1440.0,
         "share_year_built_present": 1.0, "median_units_count": 1.0,
         "share_of_matched_rows": 0.80},
        # land: area column present on every row, but the median is zero
        {"use_code": "401", "share_area_building_present": 1.0, "median_area_building": 0.0,
         "share_year_built_present": 1.0, "median_units_count": 0.0,
         "share_of_matched_rows": 0.04},
    ])
    mask = broad_residential_mask(profile)
    assert bool(mask.iloc[0]) is True
    assert bool(mask.iloc[1]) is False, "zero-median-building-area code entered the cohort"


def test_broad_residential_rule_v1_evidence_preserved():
    v2 = V3 / "baselines_pre_freeze" / "philadelphia_broad_residential_sensitivity"
    v1 = V3 / "baselines_pre_freeze" / "philadelphia_broad_residential_sensitivity_rule_v1"
    if not v2.exists():
        return
    assert v1.exists(), "rule-v1 sensitivity results were discarded rather than preserved"
    assert (v1 / "run_meta.json").exists()


def test_property_use_sensitivity_is_labeled_and_never_scored_on_test():
    tables = V3.parent.parent / "output/berry_attom_validation_v3/modeling_tables"
    for meta in tables.glob("*/modeling_table_meta_broad_residential.json"):
        assert json.loads(meta.read_text())["freeze_status"] == (
            "SENSITIVITY_ONLY_NOT_A_FREEZE_REVISION"
        )
    for run_meta in (V3 / "baselines_pre_freeze").glob("*_sensitivity/run_meta.json"):
        meta = json.loads(run_meta.read_text())
        assert meta["freeze_status"] == "SENSITIVITY_ONLY_NOT_A_FREEZE_REVISION"
        assert meta["test_block_scored"] is False
        assert meta["test_predictions_written"] is False
        holder = run_meta.parent
        assert not list(holder.glob("heldout*")), f"{holder.name} scored the test block"
        assert not list(holder.glob("direct*")) and not list(holder.glob("surrogate*"))


def test_freeze_file_untouched_by_this_pass():
    """The Philadelphia property-use finding and the surrogate repair must not
    move any frozen jurisdiction status."""
    freeze = V3 / "panel_freeze" / "final_panel_freeze_v3.yaml"
    text = freeze.read_text()
    assert "frozen_at_utc: '2026-09-03T13:40:04Z'" in text
    assert "heldout_test_not_used: true" in text
    for unit in ("wayne", "philadelphia", "st_louis_county"):
        assert f"key: {unit}" in text
    assert text.count("MODEL_TRANSFER_STATUS: PRIMARY") == 3


def test_no_mit_normal_as_partition_in_v3_slurm():
    slurm = V3 / "slurm"
    assert slurm.exists()
    for p in slurm.glob("*.sh"):
        txt = p.read_text()
        for line in txt.splitlines():
            if line.startswith("#SBATCH") and "partition" in line:
                assert "mit_normal" not in line
                assert "sched_mit_sloan_batch" in line


def test_no_raw_large_data_committed():
    result = subprocess.run(
        ["git", "ls-files",
         "data/dewey-downloads",
         "data/berry_cmf/raw",
         "output/berry_attom_validation_v3",
         "analysis/berry_attom_validation_v3"],
        cwd=REPO, capture_output=True, text=True, check=False,
    )
    tracked = [ln for ln in result.stdout.splitlines() if ln.strip()]
    parquet = [p for p in tracked if p.endswith(".parquet")]
    dewey = [p for p in tracked if p.startswith("data/dewey-downloads")]
    berry_raw = [p for p in tracked if p.startswith("data/berry_cmf/raw")]
    assert dewey == [], dewey[:10]
    assert berry_raw == [], berry_raw[:10]
    assert parquet == [], parquet[:10]


def test_st_louis_city_fips_never_in_county_cache():
    cache = REPO / "output/berry_attom_validation_v3/cache/st_louis_county"
    rec = cache / "recorder.parquet"
    hist = cache / "history.parquet"
    if not rec.exists():
        return
    import pyarrow.parquet as pq
    for path, col in ((rec, "DOCUMENTRECORDINGCOUNTYFIPS"), (hist, "SITUSSTATECOUNTYFIPS")):
        vals = set(pq.read_table(path, columns=[col])[col].to_pylist())
        assert "29510" not in {str(v) for v in vals if v is not None}


def test_noise_floor_is_estimated_above_the_observed_noise_envelope():
    """Passes 1 and 2 used a hard-coded 0.01 floor. The measured low-rho noise
    envelope is 0.017 in St. Louis, 0.028 in Philadelphia and 0.042 in Wayne, so
    that constant sat below the noise everywhere and a wiggle could cut a
    branch. The floor must scale to each curve's own inactive tail."""
    wayne_tail = [0.026437, 0.023539, 0.029764, 0.024722, 0.041952]
    floor = estimate_surrogate_noise_floor(wayne_tail)
    assert floor > max(wayne_tail), "floor does not clear the observed noise envelope"
    assert abs(floor - SURROGATE_NOISE_MULTIPLE * 0.041952) < 1e-9
    # Philadelphia's tail is negative; magnitude is what matters, not sign.
    phl = estimate_surrogate_noise_floor([-0.028229, -0.024492, -0.009801, -0.024703, -0.022342])
    assert phl > 0.028
    # An all-quiet tail must not drop the floor below the declared minimum.
    assert estimate_surrogate_noise_floor([0.0, 0.0, 0.0, 0.0, 0.0]) == SURROGATE_NOISE_FLOOR
    assert estimate_surrogate_noise_floor([]) == SURROGATE_NOISE_FLOOR


def test_estimated_floor_rescues_a_noise_cut_branch():
    """Wayne pass 2: a monotone mechanism rising to 0.639 was reported as
    UNATTAINED at every anchor because a 0.015 noise wiggle at rho~0.017 closed
    the branch at 0.042. With the floor scaled to the tail, the branch survives."""
    rho = np.array([0.001, 0.00176, 0.0031, 0.0055, 0.00964, 0.01699, 0.0298, 0.0528,
                    0.093, 0.1638, 0.2887, 0.5088, 0.8965, 1.5798, 2.784, 4.906])
    red = np.array([0.026437, 0.023539, 0.029764, 0.024722, 0.041952, 0.026722, 0.029462,
                    0.049471, 0.063697, 0.082033, 0.096245, 0.134213, 0.174904, 0.247256,
                    0.324455, 0.385613])
    old, _ = first_branch_calibrate(rho, red, noise_floor=0.01)
    assert old["branch_terminated_by"].iloc[0] == "MATERIAL_REVERSAL"
    assert (old["status"] == "UNATTAINED").all(), "pass-2 behaviour changed unexpectedly"
    new, branch = first_branch_calibrate(rho, red)
    assert new["branch_terminated_by"].iloc[0] == "GRID_CEILING"
    assert len(branch) > len(_), "estimated floor did not rescue the branch"
    assert new.loc[new["requested_reduction"].eq(0.25)].iloc[0]["status"] == (
        "interpolated_first_branch"
    )
