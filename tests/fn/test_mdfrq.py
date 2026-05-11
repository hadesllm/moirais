"""Test median_frequency (mdfrq)."""
import numpy as np
from morie.fn.mdfrq import median_frequency, mdfrq
from morie.fn._containers import DescriptiveResult


class TestMdfrq:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = median_frequency(x, fs=100.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "median_frequency"

    def test_within_nyquist(self):
        fs = 100.0
        x = np.random.default_rng(42).standard_normal(64)
        result = median_frequency(x, fs=fs)
        assert 0 <= result.value <= fs / 2

    def test_deterministic(self):
        x = np.random.default_rng(42).standard_normal(256)
        r1 = median_frequency(x)
        r2 = median_frequency(x)
        assert r1.value == r2.value

    def test_alias(self):
        assert mdfrq is median_frequency
