"""Test spectral_kurtosis."""
import numpy as np
from morie.fn.spkrt import spectral_kurtosis, spkrt
from morie.fn._containers import DescriptiveResult


class TestSpectralKurtosis:
    def test_basic(self):
        freqs = np.linspace(0, 50, 100)
        psd = np.random.default_rng(42).exponential(size=100)
        result = spectral_kurtosis(psd, freqs)
        assert isinstance(result, DescriptiveResult)

    def test_value_is_float(self):
        freqs = np.linspace(0, 50, 100)
        psd = np.random.default_rng(42).exponential(size=100)
        result = spectral_kurtosis(psd, freqs)
        assert isinstance(result.value, float)

    def test_zero_psd(self):
        freqs = np.linspace(0, 50, 100)
        psd = np.zeros(100)
        result = spectral_kurtosis(psd, freqs)
        assert result.value == 0.0

    def test_name(self):
        freqs = np.linspace(0, 50, 100)
        psd = np.ones(100)
        result = spectral_kurtosis(psd, freqs)
        assert result.name == "spectral_kurtosis"

    def test_alias(self):
        assert spkrt is spectral_kurtosis
