"""Test mean_frequency (mnfrq)."""
import numpy as np
from moirais.fn.mnfrq import mean_frequency, mnfrq
from moirais.fn._containers import DescriptiveResult


class TestMnfrq:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = mean_frequency(x, fs=100.0)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "mean_frequency"

    def test_positive(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = mean_frequency(x, fs=100.0)
        assert result.value >= 0

    def test_in_range(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = mean_frequency(x, fs=100.0)
        assert 0 <= result.value <= 50.0

    def test_alias(self):
        assert mnfrq is mean_frequency
