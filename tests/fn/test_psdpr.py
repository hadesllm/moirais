"""Test periodogram (psdpr)."""
import numpy as np
from morie.fn.psdpr import periodogram, psdpr
from morie.fn._containers import DescriptiveResult


class TestPsdpr:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = periodogram(x, fs=100.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "periodogram"

    def test_non_negative(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = periodogram(x)
        assert np.all(result.extra["psd"] >= 0)

    def test_alias(self):
        assert psdpr is periodogram
