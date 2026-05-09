"""Test spectral_slope (spslp)."""
import numpy as np
from moirais.fn.spslp import spectral_slope, spslp
from moirais.fn._containers import DescriptiveResult


class TestSpslp:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = spectral_slope(x, fs=100.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "spectral_slope"

    def test_finite(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = spectral_slope(x)
        assert np.isfinite(result.value)

    def test_alias(self):
        assert spslp is spectral_slope
