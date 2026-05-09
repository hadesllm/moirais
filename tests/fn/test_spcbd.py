"""Test spectral_bound (spcbd)."""
import numpy as np
import pytest

from moirais.fn.spcbd import spectral_bound, spcbd
from moirais.fn._containers import DescriptiveResult


class TestSpectralBound:
    def test_basic(self):
        rng = np.random.default_rng(42)
        x = rng.standard_normal(256)
        result = spectral_bound(x, fs=100.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "spectral_bound"
        assert result.value > 0

    def test_narrowband(self):
        t = np.arange(1024) / 1000.0
        x = np.sin(2 * np.pi * 50 * t)
        result = spectral_bound(x, fs=1000.0, fraction=0.99)
        assert result.value <= 500.0

    def test_zero_energy(self):
        with pytest.raises(ValueError):
            spectral_bound(np.zeros(64))

    def test_alias(self):
        assert spcbd is spectral_bound
