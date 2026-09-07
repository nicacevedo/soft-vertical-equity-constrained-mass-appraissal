#!/usr/bin/env python3
"""Unit tests for the Duan smearing factor: SIGN and D3 row balancing.

A symmetric error distribution CANNOT discriminate the sign: for symmetric u,
E[exp(u)] == E[exp(-u)], so a Gaussian-only test would pass with the sign
reversed.  The suite therefore pairs the lognormal benchmark (which fixes the
magnitude) with an asymmetric two-point case that is exact and sign-discriminating.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code"))
import p1_common as c                                                   # noqa: E402
import p1_4_smearing_sensitivity as sm                                  # noqa: E402


def test_lognormal_benchmark_fixes_the_magnitude():
    """u ~ N(0, sigma^2)  =>  s = E[exp(u)] = exp(sigma^2 / 2).

    NOT sign-discriminating (the normal is symmetric): recorded explicitly so nobody
    mistakes this test for a check on the sign.  See the two-point test below.
    """
    sigma = 0.35
    rng = np.random.default_rng(20260907)
    u = rng.normal(0.0, sigma, size=2_000_000)
    w = np.ones_like(u)
    s = sm.smearing_factor(u, w)
    analytic = float(np.exp(sigma ** 2 / 2))
    assert abs(s - analytic) < 5e-4, f"s={s} vs analytic={analytic}"
    # and the symmetry that makes it useless for the sign:
    s_rev = sm.smearing_factor(-u, w)
    assert abs(s - s_rev) < 5e-4, "normal case should be sign-blind; it is the reason for test 2"


def test_asymmetric_two_point_is_sign_discriminating():
    """u in {a, -b} with P(u=a)=p, chosen so E[u]=0 but E[exp(u)] != E[exp(-u)].

    Exact: the expectations are finite sums, so there is no Monte-Carlo error.
    This test FAILS if the implementation uses exp(e) instead of exp(-e).
    """
    a, b, p = 1.5, 0.4, 0.25
    n = 2_000_000
    k = int(round(p * n))
    u = np.concatenate([np.full(k, a), np.full(n - k, -b)])
    w = np.ones_like(u)

    exact_correct = p * np.exp(a) + (1 - p) * np.exp(-b)      # E[exp(u)]  ~ 1.6231
    exact_reversed = p * np.exp(-a) + (1 - p) * np.exp(b)     # E[exp(-u)] ~ 1.1747
    sep = abs(exact_correct - exact_reversed)
    assert sep > 0.3, f"test case is not discriminating (separation {sep:.4f})"

    s = sm.smearing_factor(u, w)
    assert abs(s - exact_correct) < 1e-9, f"s={s} != E[exp(u)]={exact_correct}"
    assert abs(s - exact_reversed) > 0.3, "implementation matches the REVERSED sign"


def test_sign_convention_against_the_repo_residual_definition():
    """e = y_pred_log - y_true_log, so u = -e and s must use exp(-e).

    Built from an asymmetric error so the check cannot pass under a sign flip.
    """
    rng = np.random.default_rng(7)
    n = 400_000
    y_true_log = rng.normal(12.0, 0.8, size=n)
    a, b, p = 1.5, 0.4, 0.25
    u = np.where(rng.random(n) < p, a, -b)                    # u = y_true - y_pred
    y_pred_log = y_true_log - u
    e = y_pred_log - y_true_log                               # repo convention
    assert np.allclose(e, -u)

    w = np.ones(n)
    # exact targets for the REALIZED sample, so the tolerance can be tight
    target = float(np.mean(np.exp(u)))
    target_reversed = float(np.mean(np.exp(-u)))
    sep = abs(target - target_reversed)
    assert sep > 0.3, f"test case is not discriminating (separation {sep:.4f})"

    s_correct = sm.smearing_factor(y_true_log - y_pred_log, w)
    s_wrong = sm.smearing_factor(e, w)
    assert abs(s_correct - target) < 1e-9, f"{s_correct} vs {target}"
    assert abs(s_wrong - target) > 0.3, "exp(e) must NOT reproduce the correct factor"
    assert abs(s_wrong - target_reversed) < 1e-9, "the wrong-sign value should be E[exp(-u)]"


def test_d3_row_balancing_differs_from_naive_duplicate_weighting():
    """A constructed fold overlap: row-balanced s must equal the exact weighted
    expectation, and must differ from the duplicate-weighted (D2) factor."""
    # rows 0..9 appear once; rows 10..14 appear twice (multiplicity 2)
    singles = np.arange(0, 10)
    dups = np.arange(10, 15)
    row_ids = np.concatenate([singles, dups, dups])
    u = np.concatenate([np.full(10, 0.1), np.full(5, 1.2), np.full(5, 1.2)])

    w = c.d3_weights_for_pooled(row_ids)
    assert abs(w.sum() - 15.0) < 1e-12, "each unique row must carry total weight one"
    assert np.allclose(w[:10], 1.0) and np.allclose(w[10:], 0.5)

    s_bal = sm.smearing_factor(u, w)
    s_naive = sm.smearing_factor(u, np.ones_like(u))
    exact_bal = (10 * np.exp(0.1) + 5 * np.exp(1.2)) / 15.0
    exact_naive = (10 * np.exp(0.1) + 10 * np.exp(1.2)) / 20.0
    assert abs(s_bal - exact_bal) < 1e-12
    assert abs(s_naive - exact_naive) < 1e-12
    assert abs(s_bal - s_naive) > 0.05, "weighting must actually change the factor here"


def test_d3_multiplicity_identities_hold():
    c.assert_d3_multiplicity_identities()
    assert c.D3_DUPLICATED_UNIQUE_ROWS == 20_988
    assert c.D3_DUPLICATED_APPEARANCES == 41_976
    assert c.D3_UNIQUE_ROWS_MULT_1 + c.D3_DUPLICATED_UNIQUE_ROWS == 130_165
    assert c.D3_UNIQUE_ROWS_MULT_1 + 2 * c.D3_DUPLICATED_UNIQUE_ROWS == 151_153


if __name__ == "__main__":
    import traceback
    names = sorted(n for n in dir() if n.startswith("test_"))
    npass = nfail = 0
    for n in names:
        try:
            globals()[n](); print(f"  PASS  {n}"); npass += 1
        except Exception as ex:
            print(f"  FAIL  {n}: {type(ex).__name__}: {ex}"); traceback.print_exc(); nfail += 1
    print(f"\n{npass} passed, {nfail} failed")
    raise SystemExit(1 if nfail else 0)
