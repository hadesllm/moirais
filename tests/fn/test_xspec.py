"""Test cross_spectral_density (xspec)."""
import numpy as np
import pytest

from morie.fn.xspec import cross_spectral_density, xspec
from morie.fn._containers import DescriptiveResult


class TestCrossSpectralDensity:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(512)
        y = rng.standard_normal(512)
        result = cross_spectral_density(x, y, fs=100.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "cross_spectral_density"

    def test_extra_keys(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(512)
        y = rng.standard_normal(512)
        result = cross_spectral_density(x, y, fs=100.0)
        assert "frequencies" in result.extra
        assert "csd" in result.extra

    def test_value_is_length(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(512)
        y = rng.standard_normal(512)
        result = cross_spectral_density(x, y, fs=100.0)
        assert result.value == len(result.extra["frequencies"])

    def test_alias(self):
        assert xspec is cross_spectral_density
