"""Test shape_factor_fn."""
import numpy as np
from morie.fn.shpfc import shape_factor_fn, alias
from morie.fn._containers import DescriptiveResult


class TestShapeFactorFn:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = shape_factor_fn(x)
        assert isinstance(result, DescriptiveResult)

    def test_value_positive(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = shape_factor_fn(x)
        assert isinstance(result.value, float)
        assert result.value > 0.0

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = shape_factor_fn(x)
        assert result.name == "shape_factor"

    def test_alias(self):
        assert alias is shape_factor_fn
