"""Test spectral_flatness (spflm)."""
import numpy as np
from morie.fn.spflm import spectral_flatness, spflm
from morie.fn._containers import DescriptiveResult


class TestSpflm:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = spectral_flatness(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "spectral_flatness"

    def test_range(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = spectral_flatness(x)
        assert 0 <= result.value <= 1.0 + 1e-10

    def test_alias(self):
        assert spflm is spectral_flatness
