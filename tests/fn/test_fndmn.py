"""Test fundamental_freq (fndmn)."""
import numpy as np
from morie.fn.fndmn import fundamental_freq, fndmn
from morie.fn._containers import DescriptiveResult


class TestFndmn:
    def test_basic(self):
        fs = 1000.0
        t = np.arange(0, 0.1, 1.0 / fs)
        x = np.sin(2 * np.pi * 100.0 * t)
        result = fundamental_freq(x, fs=fs)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "fundamental_freq"

    def test_sine_frequency(self):
        fs = 8000.0
        f0 = 200.0
        t = np.arange(0, 0.1, 1.0 / fs)
        x = np.sin(2 * np.pi * f0 * t)
        result = fundamental_freq(x, fs=fs)
        assert abs(result.value - f0) < 10.0

    def test_alias(self):
        assert fndmn is fundamental_freq
