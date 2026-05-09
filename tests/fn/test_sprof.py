"""Test spectral_rolloff (sprof)."""
import numpy as np
from moirais.fn.sprof import spectral_rolloff, sprof
from moirais.fn._containers import DescriptiveResult


class TestSprof:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = spectral_rolloff(x, fs=100.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "spectral_rolloff"

    def test_within_nyquist(self):
        fs = 100.0
        x = np.random.default_rng(42).standard_normal(64)
        result = spectral_rolloff(x, fs=fs, pct=0.85)
        assert 0 <= result.value <= fs / 2

    def test_alias(self):
        assert sprof is spectral_rolloff
