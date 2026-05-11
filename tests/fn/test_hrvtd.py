"""Tests for hrvtd — HRV time-domain metrics."""
import numpy as np
from morie.fn.hrvtd import hrv_time_domain
from morie.fn._containers import DescriptiveResult


def test_hrvtd_basic(rng):
    rr = 800 + rng.standard_normal(100) * 50
    result = hrv_time_domain(rr)
    assert isinstance(result, DescriptiveResult)
    assert "sdnn" in result.extra
    assert "rmssd" in result.extra
    assert "pnn50" in result.extra


def test_hrvtd_known_sdnn():
    rr = np.array([800.0, 850.0, 750.0, 800.0, 900.0])
    result = hrv_time_domain(rr)
    assert result.extra["sdnn"] > 0
    assert 0 <= result.extra["pnn50"] <= 100
