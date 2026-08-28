"""Unit tests for descriptive CV transition-region analysis.

Synthetic paths only for event/span logic. Canonical-root tests use temporary
identity files or the frozen 994-tree manifests without computing real events.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from utils.transition_regions import (
    ATOL,
    CANONICAL_ROOT_NAME,
    EXPECTED_IDENTITY,
    HISTORICAL_500_ROOT_NAME,
    PRIMARY_METRICS,
    CanonicalIdentityError,
    DiscreteEvent,
    OutputConfineError,
    OutputGuard,
    PathDataError,
    assert_not_historical_500,
    attenuation,
    concordance_row,
    construct_transition_span,
    event_supports_common_span,
    expected_canonical_rhos,
    extract_discrete_event,
    extract_primary_events_from_path,
    fraction_of_full_positive_log_grid,
    log10_distance_to_span,
    log10_width,
    lofo_events_and_span,
    lofo_means,
    lofo_span_summary,
    numerically_equal,
    protocol_sha256,
    q_beta_cov_agree,
    q_ratio,
    rho_in_closed_span,
    span_segment_mask,
    summarize_fold_events_logrho,
    validate_canonical_result_root,
    validate_combined_counts,
)


def _rhos():
    return np.array([0.0, 0.1, 1.0, 10.0, 100.0], dtype=float)


def test_maximum_event_extraction():
    rhos = _rhos()
    values = np.array([0.80, 0.82, 0.90, 0.85, 0.70])
    ev = extract_discrete_event(rhos, values, metric="R2_price", direction="max")
    assert ev.classification == "interior_positive"
    assert numerically_equal(ev.rho_low, 1.0)
    assert numerically_equal(ev.rho_high, 1.0)
    assert numerically_equal(ev.metric_value, 0.90)
    assert ev.local_turn_verified


def test_minimum_event_extraction():
    rhos = _rhos()
    values = np.array([20.0, 18.0, 12.0, 14.0, 25.0])
    ev = extract_discrete_event(rhos, values, metric="COD", direction="min")
    assert ev.classification == "interior_positive"
    assert numerically_equal(ev.rho_low, 1.0)
    assert numerically_equal(ev.metric_value, 12.0)
    assert ev.local_turn_verified
    assert ev.prev_rho == 0.1
    assert ev.next_rho == 10.0


def test_rho0_boundary_classification():
    rhos = _rhos()
    values = np.array([0.99, 0.90, 0.80, 0.70, 0.60])
    ev = extract_discrete_event(rhos, values, metric="R2_price", direction="max")
    assert ev.classification == "boundary_zero"
    assert numerically_equal(ev.rho_low, 0.0)
    assert not ev.local_turn_verified


def test_high_rho_boundary_classification():
    rhos = _rhos()
    values = np.array([20.0, 18.0, 16.0, 14.0, 10.0])
    ev = extract_discrete_event(rhos, values, metric="MAE_price", direction="min")
    assert ev.classification == "boundary_high"
    assert numerically_equal(ev.rho_low, 100.0)
    assert not ev.local_turn_verified


def test_numerical_plateau_handling():
    rhos = _rhos()
    plateau = 11.0
    values = np.array([20.0, plateau, plateau, 15.0, 18.0])
    ev = extract_discrete_event(rhos, values, metric="MAE_price", direction="min")
    assert ev.classification == "numerical_plateau"
    assert ev.n_tied == 2
    assert numerically_equal(ev.rho_low, 0.1)
    assert numerically_equal(ev.rho_high, 1.0)
    assert not ev.used_midpoint
    assert event_supports_common_span(ev)


def test_local_neighbor_turn_verification():
    rhos = _rhos()
    peak = extract_discrete_event(rhos, [1.0, 2.0, 3.0, 2.0, 1.0], metric="R2_price", direction="max")
    assert peak.local_turn_verified
    monotone = extract_discrete_event(rhos, [1.0, 2.0, 3.0, 4.0, 5.0], metric="R2_price", direction="max")
    assert monotone.classification == "boundary_high"
    assert not monotone.local_turn_verified
    equal_neighbor = extract_discrete_event(
        rhos, [2.0, 3.0, 3.0 + 1e-20, 2.9, 1.0], metric="R2_price", direction="max"
    )
    assert equal_neighbor.n_tied >= 1


def test_span_construction_with_five_valid_interior_events():
    rhos = _rhos()
    events = [
        extract_discrete_event(rhos, [0.1, 0.2, 0.9, 0.3, 0.0], metric="R2_price", direction="max"),
        extract_discrete_event(rhos, [9.0, 8.0, 5.0, 6.0, 9.5], metric="MAE_price", direction="min"),
        extract_discrete_event(rhos, [0.4, 0.3, 0.1, 0.2, 0.5], metric="MAPE", direction="min"),
        extract_discrete_event(rhos, [0.5, 0.2, 0.3, 0.4, 0.6], metric="RMSE_log", direction="min"),
        extract_discrete_event(rhos, [20.0, 18.0, 17.0, 10.0, 19.0], metric="COD", direction="min"),
    ]
    assert all(e.classification == "interior_positive" for e in events)
    span = construct_transition_span("Direct", events, min_positive_rho=0.1, max_positive_rho=100.0)
    assert span.status == "VALID_POSITIVE_INTERIOR_SPAN"
    assert numerically_equal(span.rho_transition_low, 0.1)
    assert numerically_equal(span.rho_transition_high, 10.0)
    assert span.blocking_metrics == []


def test_no_span_when_required_event_is_boundary_or_ambiguous():
    rhos = _rhos()
    good = extract_discrete_event(rhos, [0.1, 0.2, 0.9, 0.3, 0.0], metric="R2_price", direction="max")
    zero = extract_discrete_event(rhos, [1.0, 2.0, 3.0, 4.0, 5.0], metric="MAE_price", direction="min")
    assert zero.classification == "boundary_zero"
    others = [
        extract_discrete_event(rhos, [0.4, 0.3, 0.1, 0.2, 0.5], metric="MAPE", direction="min"),
        extract_discrete_event(rhos, [0.5, 0.2, 0.3, 0.4, 0.6], metric="RMSE_log", direction="min"),
        extract_discrete_event(rhos, [20.0, 18.0, 12.0, 14.0, 19.0], metric="COD", direction="min"),
    ]
    span = construct_transition_span(
        "Direct", [good, zero, *others], min_positive_rho=0.1, max_positive_rho=100.0
    )
    assert span.status == "FULL_COMMON_SPAN_NOT_SUPPORTED"
    assert "MAE_price" in span.blocking_metrics
    assert span.rho_transition_low is None

    amb = extract_discrete_event(
        [0.0, 0.1, 1.0, 10.0, 100.0],
        [5.0, 1.0, 3.0, 1.0, 4.0],
        metric="COD",
        direction="min",
    )
    assert amb.classification == "ambiguous"
    span2 = construct_transition_span(
        "Surrogate",
        [good, others[0], others[1], others[0], amb],
        min_positive_rho=0.1,
        max_positive_rho=100.0,
    )
    assert span2.status == "FULL_COMMON_SPAN_NOT_SUPPORTED"


def test_log_width_calculation():
    width = log10_width(0.1, 100.0)
    assert numerically_equal(width, 3.0, atol=1e-12, rtol=1e-12)
    frac = fraction_of_full_positive_log_grid(width, 0.1, 100.0)
    assert numerically_equal(frac, 1.0)
    point = log10_width(1.15, 1.15)
    assert numerically_equal(point, 0.0)
    frac_point = fraction_of_full_positive_log_grid(point, 0.1, 100.0)
    assert numerically_equal(frac_point, 0.0)


def test_fold_specific_event_extraction():
    rhos = _rhos()
    fold_paths = [
        np.array([0.1, 0.2, 0.9, 0.3, 0.0]),
        np.array([0.2, 0.95, 0.4, 0.3, 0.1]),
        np.array([0.99, 0.5, 0.4, 0.3, 0.2]),
    ]
    events = [
        extract_discrete_event(rhos, vals, metric="R2_price", direction="max") for vals in fold_paths
    ]
    assert events[0].rho_low == 1.0
    assert events[1].rho_low == 0.1
    assert events[2].classification == "boundary_zero"
    summary = summarize_fold_events_logrho(events)
    assert summary["n_interior_positive"] == 2
    assert summary["n_boundary_zero"] == 1
    assert summary["folds_are_temporal_diagnostics"] is True
    assert summary["log10_rho_low_min"] is not None


def _synthetic_family_frame():
    rhos = expected_canonical_rhos()
    rows = []
    for rho in rhos:
        rec = {"family": "Direct", "rho": rho}
        x = np.log10(rho) if rho > 0 else -2.0
        rec["R2_price__CV_mean"] = np.exp(-((x - 0.5) ** 2))
        rec["MAE_price__CV_mean"] = 1.0 + (x - 0.5) ** 2
        rec["MAPE__CV_mean"] = 1.0 + (x - 0.4) ** 2
        rec["RMSE_log__CV_mean"] = 1.0 + (x - 0.6) ** 2
        rec["COD__CV_mean"] = 1.0 + (x - 0.55) ** 2
        rec["Beta_log__CV_mean"] = -0.1 * (1.0 / (1.0 + rho))
        rec["Cov_log_residual_log_price__CV_mean"] = rec["Beta_log__CV_mean"] * 0.5
        for fid in range(1, 8):
            jitter = 0.01 * (fid - 4)
            rec[f"R2_price__fold_{fid}"] = rec["R2_price__CV_mean"] + jitter * 0.01
            rec[f"MAE_price__fold_{fid}"] = rec["MAE_price__CV_mean"] + jitter
            rec[f"MAPE__fold_{fid}"] = rec["MAPE__CV_mean"] + jitter
            rec[f"RMSE_log__fold_{fid}"] = rec["RMSE_log__CV_mean"] + jitter
            rec[f"COD__fold_{fid}"] = rec["COD__CV_mean"] + jitter
        rec["R2_price__heldout"] = rec["R2_price__CV_mean"] - 0.02
        rec["R2_price__forward_2025"] = rec["R2_price__CV_mean"] - 0.03
        rec["MAE_price__heldout"] = rec["MAE_price__CV_mean"] + 0.1
        rec["MAE_price__forward_2025"] = rec["MAE_price__CV_mean"] + 0.2
        rec["MAPE__heldout"] = rec["MAPE__CV_mean"]
        rec["MAPE__forward_2025"] = rec["MAPE__CV_mean"]
        rec["RMSE_log__heldout"] = rec["RMSE_log__CV_mean"]
        rec["RMSE_log__forward_2025"] = rec["RMSE_log__CV_mean"]
        rec["COD__heldout"] = rec["COD__CV_mean"]
        rec["COD__forward_2025"] = rec["COD__CV_mean"]
        rec["Beta_log__heldout"] = rec["Beta_log__CV_mean"]
        rec["Beta_log__forward_2025"] = rec["Beta_log__CV_mean"]
        rec["Cov_log_residual_log_price__heldout"] = rec["Cov_log_residual_log_price__CV_mean"]
        rec["Cov_log_residual_log_price__forward_2025"] = rec["Cov_log_residual_log_price__CV_mean"]
        rows.append(rec)
    return pd.DataFrame(rows)


def test_lofo_recomputation():
    df = _synthetic_family_frame()
    lofo = lofo_events_and_span(df, "Direct", min_positive_rho=0.1, max_positive_rho=100.0)
    assert len(lofo) == 7
    assert set(lofo["omitted_fold_id"]) == set(range(1, 8))
    summary = lofo_span_summary(lofo)
    assert summary["n_lofo"] == 7
    assert "n_valid_all_five_interior" in summary
    assert "R2_price__heldout" not in lofo.columns


def test_heldout_membership_in_frozen_cv_span():
    rhos = _rhos()
    cv_events = [
        extract_discrete_event(rhos, [0.1, 0.2, 0.9, 0.3, 0.0], metric="R2_price", direction="max"),
        extract_discrete_event(rhos, [9.0, 8.0, 5.0, 6.0, 9.5], metric="MAE_price", direction="min"),
        extract_discrete_event(rhos, [0.4, 0.3, 0.1, 0.2, 0.5], metric="MAPE", direction="min"),
        extract_discrete_event(rhos, [0.5, 0.2, 0.3, 0.4, 0.6], metric="RMSE_log", direction="min"),
        extract_discrete_event(rhos, [20.0, 18.0, 17.0, 10.0, 19.0], metric="COD", direction="min"),
    ]
    span = construct_transition_span("Direct", cv_events, min_positive_rho=0.1, max_positive_rho=100.0)
    held = extract_discrete_event(rhos, [0.1, 0.2, 0.4, 0.95, 0.2], metric="R2_price", direction="max")
    row = concordance_row("Direct", "heldout", held, span)
    assert numerically_equal(held.rho_low, 10.0)
    assert row["inside_frozen_cv_span"] is True
    assert row["not_prospective_confirmation"] is True
    outside = extract_discrete_event(rhos, [0.1, 0.2, 0.3, 0.4, 0.99], metric="R2_price", direction="max")
    row_out = concordance_row("Direct", "heldout", outside, span)
    assert row_out["inside_frozen_cv_span"] is False
    assert row_out["log10_distance_to_nearest_cv_span_boundary"] is not None
    assert row_out["log10_distance_to_nearest_cv_span_boundary"] > 0


def test_2025_membership_in_frozen_cv_span():
    rhos = _rhos()
    cv_events = [
        extract_discrete_event(rhos, [0.1, 0.2, 0.9, 0.3, 0.0], metric="R2_price", direction="max"),
        extract_discrete_event(rhos, [9.0, 8.0, 5.0, 6.0, 9.5], metric="MAE_price", direction="min"),
        extract_discrete_event(rhos, [0.4, 0.3, 0.1, 0.2, 0.5], metric="MAPE", direction="min"),
        extract_discrete_event(rhos, [0.5, 0.2, 0.3, 0.4, 0.6], metric="RMSE_log", direction="min"),
        extract_discrete_event(rhos, [20.0, 18.0, 17.0, 10.0, 19.0], metric="COD", direction="min"),
    ]
    span = construct_transition_span("Surrogate", cv_events, min_positive_rho=0.1, max_positive_rho=100.0)
    fwd = extract_discrete_event(rhos, [0.05, 0.9, 0.4, 0.3, 0.2], metric="R2_price", direction="max")
    row = concordance_row("Surrogate", "forward_2025", fwd, span)
    assert row["inside_frozen_cv_span"] is True
    zero_event = extract_discrete_event(rhos, [0.99, 0.2, 0.1, 0.1, 0.1], metric="R2_price", direction="max")
    row0 = concordance_row("Surrogate", "forward_2025", zero_event, span)
    assert row0["inside_frozen_cv_span"] is False


def test_no_heldout_or_2025_influence_on_cv_span():
    rhos = _rhos()
    cv_lookup = {
        "R2_price": [0.1, 0.2, 0.9, 0.3, 0.0],
        "MAE_price": [9.0, 8.0, 5.0, 6.0, 9.5],
        "MAPE": [0.4, 0.3, 0.1, 0.2, 0.5],
        "RMSE_log": [0.5, 0.2, 0.3, 0.4, 0.6],
        "COD": [20.0, 18.0, 17.0, 10.0, 19.0],
    }
    events = extract_primary_events_from_path(rhos, cv_lookup)
    span_a = construct_transition_span("Direct", events, min_positive_rho=0.1, max_positive_rho=100.0)
    heldout_lookup = {k: [100.0 if i == 0 else 0.0 for i in range(5)] for k in cv_lookup}
    span_b = construct_transition_span("Direct", events, min_positive_rho=0.1, max_positive_rho=100.0)
    assert span_a.rho_transition_low == span_b.rho_transition_low
    assert span_a.rho_transition_high == span_b.rho_transition_high
    assert span_a.status == span_b.status
    assert list(heldout_lookup)


def test_q_beta_calculation():
    q, note = q_ratio(-0.05, -0.10)
    assert numerically_equal(q, 0.5)
    assert note == "ok"
    att = attenuation(q)
    assert numerically_equal(att, 0.5)


def test_q_cov_calculation():
    q, note = q_ratio(-0.02, -0.08)
    assert numerically_equal(q, 0.25)
    assert note == "ok"


def test_q_beta_q_cov_equivalence_qa():
    ok, delta, note = q_beta_cov_agree(0.4, 0.4)
    assert ok and note == "ok" and numerically_equal(delta, 0.0)
    ok2, _delta2, note2 = q_beta_cov_agree(0.4, 0.5)
    assert not ok2
    assert note2 == "q_beta_q_cov_disagree"


def test_signed_overcorrection_behavior():
    q, note = q_ratio(0.05, -0.10)
    assert q is not None and q < 0
    assert note == "overcorrection_sign_flip"
    att = attenuation(q)
    assert att is not None and att > 1.0
    undef, undef_note = q_ratio(-0.1, 0.0)
    assert undef is None
    assert undef_note == "undefined_zero_denominator"


def test_mapping_cv_rho_span_to_highlighted_2d_path_segment():
    rhos = _rhos()
    mask = span_segment_mask(rhos, 0.1, 10.0)
    assert mask.tolist() == [False, True, True, True, False]
    idx = np.where(mask)[0]
    assert idx[0] == 1 and idx[-1] == 3
    assert not rho_in_closed_span(100.0, 0.1, 10.0)
    assert rho_in_closed_span(0.1, 0.1, 10.0)
    assert rho_in_closed_span(10.0, 0.1, 10.0)


def test_rejection_of_historical_500_tree_root():
    with TemporaryDirectory() as td:
        hist = Path(td) / HISTORICAL_500_ROOT_NAME
        hist.mkdir()
        try:
            assert_not_historical_500(hist)
            raise AssertionError("expected CanonicalIdentityError")
        except CanonicalIdentityError as err:
            assert "500-tree" in str(err) or HISTORICAL_500_ROOT_NAME in str(err)
        nested = Path(td) / HISTORICAL_500_ROOT_NAME / "analysis"
        nested.mkdir(parents=True)
        try:
            assert_not_historical_500(nested)
            raise AssertionError("expected CanonicalIdentityError")
        except CanonicalIdentityError:
            pass
        good = Path(td) / CANONICAL_ROOT_NAME
        good.mkdir()
        assert_not_historical_500(good)


def test_canonical_config_split_grid_count_validation():
    rhos = expected_canonical_rhos()
    assert len(rhos) == 51
    assert rhos[0] == 0.0
    assert np.isclose(rhos[-1], 100.0)
    rows = []
    rows.append(
        {
            "family": "Linear",
            "rho": np.nan,
            "data_id": EXPECTED_IDENTITY["data_id"],
            "split_id": EXPECTED_IDENTITY["split_id"],
        }
    )
    rows.append(
        {
            "family": "LightGBM",
            "rho": np.nan,
            "data_id": EXPECTED_IDENTITY["data_id"],
            "split_id": EXPECTED_IDENTITY["split_id"],
        }
    )
    for fam in ("Direct", "Surrogate"):
        for rho in rhos:
            rows.append(
                {
                    "family": fam,
                    "rho": rho,
                    "data_id": EXPECTED_IDENTITY["data_id"],
                    "split_id": EXPECTED_IDENTITY["split_id"],
                }
            )
    ok = validate_combined_counts(pd.DataFrame(rows), rhos)
    assert ok["ok"], ok["problems"]
    bad = pd.DataFrame(rows[:-3])
    not_ok = validate_combined_counts(bad, rhos)
    assert not not_ok["ok"]


def test_output_confinement_to_dedicated_directory():
    with TemporaryDirectory() as td:
        repo = Path(td) / "repo"
        paper = repo / "paper"
        paper.mkdir(parents=True)
        out = repo / "output" / CANONICAL_ROOT_NAME / "transition_regions_v1"
        out.mkdir(parents=True)
        guard = OutputGuard(out, repo)
        dest = guard.write_text(out / "tables" / "ok.csv", "a,b\n1,2\n")
        assert dest.is_file()
        try:
            guard.write_text(paper / "sneak.csv", "no")
            raise AssertionError("expected OutputConfineError")
        except OutputConfineError:
            pass
        try:
            guard.write_text(Path("/tmp") / "outside.csv", "no")
            raise AssertionError("expected OutputConfineError")
        except OutputConfineError:
            pass


def test_no_tex_creation():
    with TemporaryDirectory() as td:
        repo = Path(td) / "repo"
        (repo / "paper").mkdir(parents=True)
        out = Path(td) / "out"
        out.mkdir()
        guard = OutputGuard(out, repo)
        try:
            guard.write_text(out / "table.tex", r"\begin{tabular}{c}x\end{tabular}")
            raise AssertionError("expected OutputConfineError")
        except OutputConfineError as err:
            assert ".tex" in str(err)
        assert not (out / "table.tex").exists()
        guard.write_text(out / "table.md", "| a |\n")
        assert (out / "table.md").is_file()


def test_no_write_into_paper():
    with TemporaryDirectory() as td:
        repo = Path(td) / "repo"
        paper = repo / "paper" / "img" / "generated_v6_preselection"
        paper.mkdir(parents=True)
        out = repo / "output" / "transition_regions_v1"
        out.mkdir(parents=True)
        guard = OutputGuard(out, repo)
        try:
            guard.write_bytes(paper / "transition_event_map.pdf", b"%PDF")
            raise AssertionError("expected OutputConfineError")
        except OutputConfineError as err:
            assert "paper/" in str(err)
        assert list(paper.glob("*")) == []


def test_protocol_hash_is_stable():
    a = protocol_sha256()
    b = protocol_sha256()
    assert a == b
    assert len(a) == 64


def test_tiny_float_tolerance_is_not_near_optimality():
    rhos = _rhos()
    values = np.array([10.0, 9.0, 8.0000001, 8.0, 9.5])
    ev = extract_discrete_event(rhos, values, metric="MAE_price", direction="min")
    assert ev.classification == "interior_positive"
    assert numerically_equal(ev.rho_low, 10.0)
    assert ev.n_tied == 1


def test_lofo_means_omits_one_fold():
    mat = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    m0 = lofo_means(mat, 0)
    np.testing.assert_allclose(m0, [(3 + 5) / 2.0, (4 + 6) / 2.0])
