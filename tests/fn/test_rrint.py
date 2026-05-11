"""Tests for rrint — RR interval series."""
import numpy as np
from morie.fn.rrint import rr_intervals
from morie.fn._containers import DescriptiveResult


def test_rrint_basic():
    peaks = np.array([100, 900, 1700, 2500])
    result = rr_intervals(peaks, fs=1000.0)
    assert isinstance(result, DescriptiveResult)
    rr = result.extra["rr_ms"]
    assert len(rr) == 3
    np.testing.assert_allclose(rr, [800, 800, 800])


def test_rrint_single_peak():
    result = rr_intervals(np.array([100]), fs=1000.0)
    assert np.isnan(result.value)
