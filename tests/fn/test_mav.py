"""Test mean_abs_value."""
import numpy as np
from morie.fn.mav import mean_abs_value, alias
from morie.fn._containers import DescriptiveResult


class TestMeanAbsValue:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = mean_abs_value(x)
        assert isinstance(result, DescriptiveResult)

    def test_value_non_negative(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = mean_abs_value(x)
        assert isinstance(result.value, float)
        assert result.value >= 0.0

    def test_known_value(self):
        x = np.ones(256)
        result = mean_abs_value(x)
        assert abs(result.value - 1.0) < 1e-9

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = mean_abs_value(x)
        assert result.name == "mean_absolute_value"

    def test_alias(self):
        assert alias is mean_abs_value
