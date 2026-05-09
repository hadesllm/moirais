"""Tests for wildfire-smoke PM excess toxicity."""

import math

import numpy as np
import pytest

from moirais.fn.smokpm import smokpm, wildfire_smoke_rr


def test_smokpm_reference_level_rr_is_one():
    r = smokpm(smoke_pm25_ugm3=0, ambient_pm25_ugm3=0)
    assert r.value == pytest.approx(1.0)


def test_smokpm_respiratory_multiplier_is_10():
    r = smokpm(smoke_pm25_ugm3=10, outcome="respiratory")
    # rr_per_10_smoke = exp(10 * ln(1.010)) ≈ 1.1046
    expected = math.exp(10 * math.log(1.010))
    assert r.value == pytest.approx(expected, abs=1e-4)
    assert r.extra["smoke_multiplier"] == 10.0


def test_smokpm_combines_smoke_and_ambient():
    # 50 smoke + 10 ambient, respiratory
    # log_rr = 10*log(1.010)*5 + log(1.010)*1 = log(1.010)*(50+1)/1 ≈ 0.507
    r = smokpm(
        smoke_pm25_ugm3=50, ambient_pm25_ugm3=10,
        outcome="respiratory",
    )
    log_rr_amb = math.log(1.010)
    expected = math.exp(10 * log_rr_amb * 5 + log_rr_amb * 1)
    assert r.value == pytest.approx(expected, abs=1e-4)


def test_smokpm_outcome_ordering():
    # Respiratory (10×) > asthma_ed (8×) > copd (5×) > all_cause (3×)
    smoke = 30
    resp = smokpm(smoke, outcome="respiratory").value
    asth = smokpm(smoke, outcome="asthma_ed").value
    copd = smokpm(smoke, outcome="copd").value
    all_c = smokpm(smoke, outcome="all_cause").value
    assert resp > asth > copd > all_c


def test_smokpm_array_broadcast_ambient_scalar():
    S = np.array([10.0, 20.0, 30.0])
    r = smokpm(S, ambient_pm25_ugm3=5.0, outcome="respiratory")
    rrs = r.extra["rr"]
    assert len(rrs) == 3
    # Monotone in smoke concentration
    assert rrs[0] < rrs[1] < rrs[2]


def test_smokpm_array_size_1_broadcasts_to_array():
    # size-1 broadcasts (intentional pandas-style behavior)
    r = smokpm(np.array([10, 20]), ambient_pm25_ugm3=np.array([5]))
    assert len(r.extra["rr"]) == 2


def test_smokpm_array_shape_mismatch_raises():
    # Incompatible non-scalar shapes raise
    with pytest.raises(ValueError, match="match in shape"):
        smokpm(np.array([10, 20, 30]),
               ambient_pm25_ugm3=np.array([5, 6]))


def test_smokpm_negative_concentration_raises():
    with pytest.raises(ValueError, match="non-negative"):
        smokpm(-5.0)


def test_smokpm_unknown_outcome_raises():
    with pytest.raises(KeyError, match="Unknown outcome"):
        smokpm(10, outcome="diabetes")


def test_smokpm_alias_matches():
    assert smokpm is wildfire_smoke_rr


def test_smokpm_user_can_override_ambient_rr():
    r1 = smokpm(20, outcome="respiratory", ambient_rr_per_10=1.010)
    r2 = smokpm(20, outcome="respiratory", ambient_rr_per_10=1.020)
    # Doubling the ambient baseline should roughly double the log-RR
    assert r2.value > r1.value
