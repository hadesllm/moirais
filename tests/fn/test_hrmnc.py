"""Test harmonic_ratio (hrmnc)."""
import numpy as np
from moirais.fn.hrmnc import harmonic_ratio, hrmnc
from moirais.fn._containers import DescriptiveResult


class TestHrmnc:
    def test_basic(self):
        fs = 8000.0
        t = np.arange(0, 0.1, 1.0 / fs)
        x = np.sin(2 * np.pi * 200.0 * t)
        result = harmonic_ratio(x, fs=fs)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "harmonic_ratio"

    def test_pure_tone_high_hnr(self):
        fs = 8000.0
        t = np.arange(0, 0.1, 1.0 / fs)
        x = np.sin(2 * np.pi * 200.0 * t)
        result = harmonic_ratio(x, fs=fs)
        assert result.value > 5.0

    def test_alias(self):
        assert hrmnc is harmonic_ratio
