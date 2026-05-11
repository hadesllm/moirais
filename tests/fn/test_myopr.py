"""Test myopulse_rate_fn."""
import numpy as np
from morie.fn.myopr import myopulse_rate_fn, alias
from morie.fn._containers import DescriptiveResult


class TestMyopulseRateFn:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = myopulse_rate_fn(x)
        assert isinstance(result, DescriptiveResult)

    def test_value_range(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = myopulse_rate_fn(x)
        assert isinstance(result.value, float)
        assert 0.0 <= result.value <= 1.0

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = myopulse_rate_fn(x)
        assert result.name == "myopulse_rate"

    def test_alias(self):
        assert alias is myopulse_rate_fn
