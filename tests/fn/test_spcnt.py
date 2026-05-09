"""Test spectral_centroid (spcnt)."""
import numpy as np
from moirais.fn.spcnt import spectral_centroid, spcnt
from moirais.fn._containers import DescriptiveResult


class TestSpcnt:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = spectral_centroid(x, fs=100.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "spectral_centroid"

    def test_non_negative(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = spectral_centroid(x, fs=100.0)
        assert result.value >= 0

    def test_alias(self):
        assert spcnt is spectral_centroid
