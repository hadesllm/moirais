"""Tests for Burnett 2018 PNAS GEMM PM₂.₅ exposure-response."""

import math

import numpy as np
import pytest

from morie.fn.pmgemm import pm_gemm_rr, pmgemm


def test_pmgemm_at_tmrel_returns_rr_one():
    r = pmgemm(2.4, outcome="ncd_lri")
    assert r.value == pytest.approx(1.0)


def test_pmgemm_below_tmrel_returns_rr_one():
    # z = max(0, C - C0); below TMREL, z=0, RR=1 exactly.
    r = pmgemm(1.0, outcome="ncd_lri")
    assert r.value == pytest.approx(1.0)


def test_pmgemm_monotone_in_concentration():
    concs = np.array([5.0, 10.0, 20.0, 50.0, 100.0])
    r = pmgemm(concs)
    rrs = r.extra["rr"]
    for i in range(1, len(rrs)):
        assert rrs[i] >= rrs[i - 1], (
            f"non-monotone: {rrs[i-1]:.3f} → {rrs[i]:.3f} at C={concs[i]}"
        )


def test_pmgemm_saturation_vs_linear():
    # GEMM's whole point: at high C, RR saturates vs an unbounded
    # log-linear extrapolation. At C=500 µg/m³ (pathologically high),
    # ncd_lri RR should NOT exceed a sane ceiling (say 3× vs baseline).
    r = pmgemm(500.0, outcome="ncd_lri")
    assert r.value < 3.0, f"GEMM should saturate; got RR={r.value}"


def test_pmgemm_lung_cancer_steeper_than_ncd_at_moderate_exposure():
    # At moderate PM (50 µg/m³, TMREL ref), LC has a steeper curve
    # (θ=0.3195 vs NCD's 0.1430) so RR should be higher.
    r_ncd = pmgemm(50.0, outcome="ncd_lri")
    r_lc = pmgemm(50.0, outcome="lung_cancer")
    assert r_lc.value > r_ncd.value


def test_pmgemm_params_overridable_for_sensitivity():
    # Sensitivity analysis: bump θ, expect higher RR.
    r_base = pmgemm(20.0, outcome="ncd_lri")
    r_high = pmgemm(20.0, outcome="ncd_lri", theta=0.20)   # > 0.1430
    assert r_high.value > r_base.value


def test_pmgemm_custom_reference_concentration():
    # If reference is set to the target concentration, z=0, RR=1.
    r = pmgemm(25.0, reference_ugm3=25.0)
    assert r.value == pytest.approx(1.0)


def test_pmgemm_unknown_outcome_raises():
    with pytest.raises(KeyError, match="Available"):
        pmgemm(10.0, outcome="diabetes")


def test_pmgemm_negative_concentration_raises():
    with pytest.raises(ValueError, match="non-negative"):
        pmgemm(-5.0)


def test_pmgemm_bad_alpha_raises():
    with pytest.raises(ValueError, match="alpha"):
        pmgemm(10.0, alpha=0.0)


def test_pmgemm_array_extra_shape_matches_input():
    C = np.array([5.0, 10.0, 25.0, 50.0, 100.0])
    r = pmgemm(C)
    assert len(r.extra["rr"]) == 5
    assert len(r.extra["z_ugm3"]) == 5


def test_pmgemm_alias_matches():
    assert pmgemm is pm_gemm_rr


def test_pmgemm_delhi_scale_rr_in_published_ballpark():
    # Burnett 2018 figure 1B shows NCD+LRI hazard ratio at ~100 µg/m³
    # to be ~1.7 (solid black line, central estimate). We get 1.704 —
    # spot-on. Window is deliberately wide to accept the published CI.
    r = pmgemm(100.0, outcome="ncd_lri")
    assert 1.50 < r.value < 1.95, (
        f"expected Burnett ~1.7 at C=100 (range 1.5-1.95); got {r.value:.3f}"
    )


def test_pmgemm_saturation_shape_above_100():
    # Burnett's key finding: the slope flattens above ~100 µg/m³. The
    # RR delta from 100→300 should be smaller than from 10→100 (by
    # the sigmoid saturation term).
    r10 = pmgemm(10.0, outcome="ncd_lri").value
    r100 = pmgemm(100.0, outcome="ncd_lri").value
    r300 = pmgemm(300.0, outcome="ncd_lri").value
    gain_10_to_100 = r100 - r10
    gain_100_to_300 = r300 - r100
    assert gain_100_to_300 < gain_10_to_100, (
        f"GEMM should saturate: Δ(10→100)={gain_10_to_100:.3f} "
        f"should exceed Δ(100→300)={gain_100_to_300:.3f}"
    )
