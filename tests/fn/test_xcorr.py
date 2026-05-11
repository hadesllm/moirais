"""Test cross_correlation (xcorr)."""
import numpy as np
import pytest

from morie.fn.xcorr import cross_correlation, xcorr
from morie.fn._containers import DescriptiveResult


class TestCrossCorrelation:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(128)
        y = np.random.default_rng(43).standard_normal(128)
        result = cross_correlation(x, y)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "cross_correlation"

    def test_autocorrelation_peak(self):
        x = np.random.default_rng(42).standard_normal(128)
        result = cross_correlation(x, x, max_lag=10)
        assert result.value == pytest.approx(1.0, abs=0.01)

    def test_correlation_array(self):
        x = np.random.default_rng(42).standard_normal(64)
        y = np.random.default_rng(43).standard_normal(64)
        result = cross_correlation(x, y, max_lag=5)
        assert "correlation" in result.extra
        assert len(result.extra["correlation"]) == 11

    def test_alias(self):
        assert xcorr is cross_correlation
