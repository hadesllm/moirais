"""Test spectral_edge_freq (spedf)."""
import numpy as np
from moirais.fn.spedf import spectral_edge_freq, spedf
from moirais.fn._containers import DescriptiveResult


class TestSpedf:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = spectral_edge_freq(x, fs=100.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "spectral_edge_freq"

    def test_within_nyquist(self):
        fs = 100.0
        x = np.random.default_rng(42).standard_normal(64)
        result = spectral_edge_freq(x, fs=fs, pct=0.95)
        assert 0 <= result.value <= fs / 2

    def test_alias(self):
        assert spedf is spectral_edge_freq
