"""Tests for PM relative-risk model (ACS / Harvard Six Cities)."""

import math

import numpy as np
import pytest

from moirais.fn.pmrr import pm_relative_risk, pmrr


def test_pmrr_reference_level_gives_rr_one():
    r = pmrr(5.0, reference_ugm3=5.0)
    assert r.value == pytest.approx(1.0)


def test_pmrr_10_ugm3_above_reference_matches_base():
    # PM2.5 all-cause RR per 10 µg/m³ = 1.070.
    # At C=10, ref=0: delta=1, RR = 1.07^1 = 1.07
    r = pmrr(10.0, reference_ugm3=0.0, outcome="all_cause")
    assert r.value == pytest.approx(1.070, abs=0.001)


def test_pmrr_25_ugm3_above_reference_scales_log_linearly():
    # delta=2.5, RR = 1.070^2.5
    r = pmrr(25.0, reference_ugm3=0.0, outcome="all_cause")
    assert r.value == pytest.approx(1.070 ** 2.5, abs=0.001)


def test_pmrr_stroke_vs_all_cause():
    # Stroke RR per 10 is higher (1.110) than all-cause (1.070)
    r_all = pmrr(30.0, reference_ugm3=5.0, outcome="all_cause")
    r_str = pmrr(30.0, reference_ugm3=5.0, outcome="stroke")
    assert r_str.value > r_all.value


def test_pmrr_ci_brackets_point_estimate():
    r = pmrr(50.0, reference_ugm3=5.0, outcome="all_cause")
    # At delta=4.5, using RR=1.07 (lo=1.04, hi=1.10) all raised to 4.5
    # lo ≤ point ≤ hi
    assert r.extra["rr_95lo"] < r.value < r.extra["rr_95hi"]


def test_pmrr_pm10_all_cause_is_weaker_than_pm25_all_cause():
    # PM10 RR per 10 (1.04) is smaller than PM2.5 (1.07)
    r_pm10 = pmrr(50.0, pollutant="pm10")
    r_pm25 = pmrr(50.0, pollutant="pm25")
    assert r_pm10.value < r_pm25.value


def test_pmrr_unknown_outcome_raises():
    with pytest.raises(KeyError, match="No published estimate"):
        pmrr(10.0, outcome="diabetes")


def test_pmrr_unknown_pollutant_raises():
    with pytest.raises(KeyError, match="No published estimate"):
        pmrr(10.0, pollutant="voc")


def test_pmrr_array_input():
    C = np.array([5.0, 10.0, 15.0, 20.0])
    r = pmrr(C, reference_ugm3=5.0)
    rrs = r.extra["rr"]
    assert rrs[0] == pytest.approx(1.0, abs=1e-9)
    # Monotone
    for i in range(1, len(rrs)):
        assert rrs[i] > rrs[i - 1]


def test_pmrr_alias_matches():
    assert pmrr is pm_relative_risk
