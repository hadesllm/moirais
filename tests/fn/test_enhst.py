"""Test entropy_hist."""
import numpy as np
from morie.fn.enhst import entropy_hist, alias
from morie.fn._containers import DescriptiveResult


class TestEntropyHist:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = entropy_hist(x)
        assert isinstance(result, DescriptiveResult)

    def test_value_non_negative(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = entropy_hist(x)
        assert isinstance(result.value, float)
        assert result.value >= 0.0

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = entropy_hist(x)
        assert result.name == "entropy_histogram"

    def test_alias(self):
        assert alias is entropy_hist
