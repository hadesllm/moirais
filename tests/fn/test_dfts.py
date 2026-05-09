"""Tests for moirais.fn.dfts — power spectral density."""
import numpy as np
import pytest
from moirais.fn.dfts import dft_spectrum


class TestDFTSpectrum:
    def test_dominant_freq(self):
        t = np.arange(256)
        y = np.sin(2 * np.pi * 10 * t / 256)
        res = dft_spectrum(y, sampling_rate=256)
        assert abs(res.value - 10) < 2

    def test_too_short_raises(self):
        with pytest.raises(ValueError):
            dft_spectrum(np.array([1, 2]))
