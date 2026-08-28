"""Synthetic tests for the 994-tree paper-asset follow-up.

No canonical event recomputation; span-regret and sharpness only.
"""

from __future__ import annotations

import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.transition_paper_assets import (
    ATOL,
    classify_direct_interpretation,
    endpoint_equals_first_positive,
    endpoint_equals_last_positive,
    event_sharpness_row,
    manuscript_format_flags,
    neighbor_gaps,
    ordinal_rank_of_value,
    positive_display_anchors,
    second_best_event,
    span_regret_row,
    surrogate_rmse_log_zero_vs_positive,
)
from utils.transition_regions import (
    HISTORICAL_500_ROOT_NAME,
    OutputConfineError,
    OutputGuard,
    assert_not_historical_500,
    expected_canonical_rhos,
    numerically_equal,
    sha256_file,
)


def _rhos():
    return np.array([0.0, 0.1, 1.0, 10.0, 100.0], dtype=float)


def test_span_regret_max_inside_zero():
    rhos = _rhos()
    values = np.array([0.80, 0.85, 0.90, 0.88, 0.70])
    row = span_regret_row(
        rhos, values, family="Direct", split="heldout", metric="R2_price",
        direction="max", rho_low=0.1, rho_high=10.0,
    )
    assert numerically_equal(row["raw_regret"], 0.0)
    assert numerically_equal(row["best_inside_value"], 0.90)
    assert row["best_inside_ordinal_rank"] == 1
    assert numerically_equal(row["normalized_regret"], 0.0)


def test_span_regret_min_inside_zero():
    rhos = _rhos()
    values = np.array([20.0, 18.0, 12.0, 14.0, 25.0])
    row = span_regret_row(
        rhos, values, family="Direct", split="heldout", metric="COD",
        direction="min", rho_low=0.1, rho_high=10.0,
    )
    assert numerically_equal(row["raw_regret"], 0.0)
    assert numerically_equal(row["best_inside_value"], 12.0)


def test_span_regret_positive_when_opt_outside():
    rhos = _rhos()
    values = np.array([0.80, 0.82, 0.84, 0.95, 0.70])
    row = span_regret_row(
        rhos, values, family="Direct", split="heldout", metric="R2_price",
        direction="max", rho_low=0.1, rho_high=1.0,
    )
    assert row["raw_regret"] > 0
    assert numerically_equal(row["raw_regret"], 0.95 - 0.84)
    rng = 0.95 - 0.70
    assert numerically_equal(row["normalized_regret"], (0.95 - 0.84) / rng)
    assert row["best_inside_ordinal_rank"] == 2
    assert row["log10_distance_global_opt_to_cv_span"] is not None
    assert row["log10_distance_global_opt_to_cv_span"] > 0


def test_span_regret_min_positive_outside():
    rhos = _rhos()
    values = np.array([20.0, 18.0, 16.0, 10.0, 25.0])
    row = span_regret_row(
        rhos, values, family="Direct", split="forward_2025", metric="MAE_price",
        direction="min", rho_low=0.1, rho_high=1.0,
    )
    assert numerically_equal(row["raw_regret"], 16.0 - 10.0)
    assert row["best_inside_ordinal_rank"] == 2


def test_constant_path_handling():
    rhos = _rhos()
    values = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    row = span_regret_row(
        rhos, values, family="Direct", split="heldout", metric="R2_price",
        direction="max", rho_low=0.1, rho_high=1.0,
    )
    assert numerically_equal(row["raw_regret"], 0.0)
    assert numerically_equal(row["path_range"], 0.0)
    assert row["normalized_regret"] is None
    assert row["best_inside_ordinal_rank"] == 1


def test_rank_of_best_in_span():
    values = np.array([5.0, 1.0, 3.0, 2.0, 4.0])
    assert ordinal_rank_of_value(values, 1.0, direction="min") == 1
    assert ordinal_rank_of_value(values, 2.0, direction="min") == 2
    assert ordinal_rank_of_value(values, 5.0, direction="max") == 1


def test_second_best_event_extraction():
    rhos = _rhos()
    values = np.array([0.80, 0.90, 0.88, 0.85, 0.70])
    rec = second_best_event(rhos, values, direction="max")
    assert numerically_equal(rec["second_best_rho"], 1.0)
    assert numerically_equal(rec["second_best_value"], 0.88)
    assert numerically_equal(rec["best_minus_second_gap"], 0.02)


def test_neighbor_gap_extraction():
    rhos = _rhos()
    values = np.array([0.80, 0.82, 0.90, 0.85, 0.70])
    rec = neighbor_gaps(rhos, values, opt_rho=1.0, opt_value=0.90, direction="max")
    assert numerically_equal(rec["lower_neighbor_rho"], 0.1)
    assert numerically_equal(rec["lower_neighbor_gap"], 0.08)
    assert numerically_equal(rec["higher_neighbor_rho"], 10.0)
    assert numerically_equal(rec["higher_neighbor_gap"], 0.05)


def test_boundary_event_handling():
    rhos = _rhos()
    values = np.array([0.10, 0.20, 0.30, 0.40, 0.50])
    ev = event_sharpness_row(
        rhos, values, family="Surrogate", split="cv_mean", metric="RMSE_log", direction="min"
    )
    assert ev["classification"] == "boundary_zero"
    assert numerically_equal(ev["optimum_rho"], 0.0)
    assert ev["lower_neighbor_rho"] is None
    assert ev["higher_neighbor_rho"] is not None
    assert ev["rmse_log_rho0"] is not None
    assert ev["best_positive_minus_zero"] > 0


def test_surrogate_rho0_versus_best_positive():
    rhos = _rhos()
    values = np.array([0.29, 0.30, 0.31, 0.33, 0.40])
    rec = surrogate_rmse_log_zero_vs_positive(rhos, values)
    assert numerically_equal(rec["rmse_log_rho0"], 0.29)
    assert numerically_equal(rec["best_positive_rmse_log"], 0.30)
    assert numerically_equal(rec["best_positive_rho"], 0.1)
    rng = 0.40 - 0.29
    assert numerically_equal(rec["best_positive_minus_zero"], 0.01)
    assert numerically_equal(rec["best_positive_minus_zero_over_path_range"], 0.01 / rng)


def test_zero_log_axis_event_representation():
    rhos = expected_canonical_rhos()
    assert numerically_equal(rhos[0], 0.0)
    assert all(x > 0 for x in rhos[1:])
    display0 = 0.1 * 0.55
    assert display0 < 0.1


def test_lower_endpoint_equals_first_positive_grid_flag():
    assert endpoint_equals_first_positive(0.1, 0.1) is True
    assert endpoint_equals_first_positive(1.0985411419875584, 0.1) is False
    assert endpoint_equals_last_positive(100.0, 100.0) is True
    assert endpoint_equals_last_positive(1.0985411419875584, 100.0) is False
    assert endpoint_equals_first_positive(None, 0.1) is None


def test_positive_display_anchors_exclude_zero():
    grid = expected_canonical_rhos()
    anchors = positive_display_anchors(grid)
    assert len(anchors) == 4
    assert all(x > 0 for x in anchors)
    assert numerically_equal(anchors[0], 0.1)
    assert numerically_equal(anchors[-1], 100.0)


def test_paper_asset_confinement_and_no_tex():
    with TemporaryDirectory() as tmp:
        repo = Path(tmp) / "repo"
        paper = repo / "paper"
        paper.mkdir(parents=True)
        (paper / "keep.txt").write_text("x")
        out = Path(tmp) / "assets"
        out.mkdir()
        guard = OutputGuard(out, repo)
        dest = guard.write_text(out / "tables" / "ok.md", "hello")
        assert dest.is_file()
        try:
            guard.write_text(out / "bad.tex", "nope")
            raise AssertionError("tex write should fail")
        except OutputConfineError:
            pass
        try:
            guard.write_text(paper / "sneak.md", "nope")
            raise AssertionError("paper write should fail")
        except OutputConfineError:
            pass
        assert not list(out.rglob("*.tex"))
        assert (paper / "keep.txt").read_text() == "x"


def test_no_500_root_reads():
    with TemporaryDirectory() as tmp:
        bad = Path(tmp) / HISTORICAL_500_ROOT_NAME
        bad.mkdir()
        try:
            assert_not_historical_500(bad)
            raise AssertionError("500-tree root must be rejected")
        except Exception as err:
            assert "500" in str(err) or "forbidden" in str(err).lower()


def test_manuscript_flags_beats_both_and_ordinary_only():
    flags = manuscript_format_flags(
        0.90, metric="R2_price", family="Direct",
        linear_val=0.80, lgbm_val=0.89, higher=True, target=None, can_star=False,
    )
    assert flags["beats_both_baselines"] is True
    assert flags["manuscript_bold"] is True
    assert flags["manuscript_asterisk"] is False
    flags2 = manuscript_format_flags(
        -0.040, metric="PRB", family="Direct",
        linear_val=-0.016, lgbm_val=-0.091, higher=None, target=0.0, can_star=True,
    )
    assert flags2["beats_both_baselines"] is False
    assert flags2["beats_ordinary_only"] is True
    assert flags2["within_reference_range"] is True
    assert flags2["manuscript_asterisk"] is True


def test_interpretation_classes_no_threshold():
    a = pd.DataFrame({"family": ["Direct", "Direct"], "raw_regret": [0.0, 0.0]})
    b = pd.DataFrame({"family": ["Direct", "Direct"], "raw_regret": [0.01, 0.02]})
    c = pd.DataFrame({"family": ["Direct", "Direct"], "raw_regret": [0.0, 0.02]})
    assert classify_direct_interpretation(a) == "A"
    assert classify_direct_interpretation(b) == "B"
    assert classify_direct_interpretation(c) == "C"
    _ = ATOL


def test_v1_hash_immutability_helper():
    with TemporaryDirectory() as tmp:
        p = Path(tmp) / "x.csv"
        p.write_text("a,b\n1,2\n")
        h1 = sha256_file(p)
        assert h1 == sha256_file(p)
        p.write_text("a,b\n1,3\n")
        assert sha256_file(p) != h1
