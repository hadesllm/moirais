"""Tests for temperature-mortality V-curve (Gasparrini 2015)."""

import math

import numpy as np
import pytest

from morie.fn.tmort import temperature_mortality_vcurve, tmort


def test_tmort_at_mmt_rr_is_one():
    r = tmort(22.0, mmt_C=22.0)
    assert r.value == pytest.approx(1.0)


def test_tmort_heat_arm_pooled_slope():
    # 10°C above MMT → exp(0.012 * 10) = 1.1275
    r = tmort(32.0, mmt_C=22.0)
    assert r.value == pytest.approx(math.exp(0.012 * 10), abs=1e-4)


def test_tmort_cold_arm_pooled_slope():
    # 37°C below MMT → exp(0.003 * 37)
    r = tmort(-15.0, mmt_C=22.0)
    assert r.value == pytest.approx(math.exp(0.003 * 37), abs=1e-4)


def test_tmort_heat_slope_steeper_than_cold_per_deg():
    # At equal deviations, heat has higher RR
    hot = tmort(32.0, mmt_C=22.0)  # 10 above
    cold = tmort(12.0, mmt_C=22.0)  # 10 below
    assert hot.value > cold.value


def test_tmort_attributable_fraction_makes_sense():
    r = tmort(32.0, mmt_C=22.0)
    # AF = (RR-1)/RR; with RR ≈ 1.1275, AF ≈ 0.113
    assert r.extra["mean_af"] == pytest.approx((1.1275 - 1) / 1.1275, abs=0.01)


def test_tmort_array_input():
    T = np.array([10, 15, 22, 27, 32])
    r = tmort(T, mmt_C=22.0)
    # RR at MMT (22) should be 1.0 exactly
    rrs = r.extra["rr"]
    assert rrs[2] == pytest.approx(1.0)
    # Monotone in each arm
    assert rrs[0] > rrs[1] > rrs[2]   # colder → higher RR on cold arm
    assert rrs[2] < rrs[3] < rrs[4]   # hotter → higher RR on heat arm


def test_tmort_custom_slopes():
    # With steeper cold slope, cold dominates
    r = tmort(-10.0, mmt_C=22.0, beta_hot=0.0, beta_cold=0.02)
    assert r.value == pytest.approx(math.exp(0.02 * 32), abs=1e-4)


def test_tmort_negative_beta_raises():
    with pytest.raises(ValueError, match="non-negative"):
        tmort(25, mmt_C=22, beta_hot=-0.01)


def test_tmort_alias_matches():
    assert tmort is temperature_mortality_vcurve
