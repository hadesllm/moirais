"""Test slope_sign_changes_fn."""
import numpy as np
from moirais.fn.sscfn import slope_sign_changes_fn, alias
from moirais.fn._containers import DescriptiveResult


class TestSlopeSignChangesFn:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = slope_sign_changes_fn(x)
        assert isinstance(result, DescriptiveResult)

    def test_value_non_negative(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = slope_sign_changes_fn(x)
        assert isinstance(result.value, int)
        assert result.value >= 0

    def test_name(self):
        x = np.random.default_rng(42).standard_normal(256)
        result = slope_sign_changes_fn(x)
        assert result.name == "slope_sign_changes"

    def test_monotone_zero_ssc(self):
        x = np.arange(256, dtype=float)
        result = slope_sign_changes_fn(x)
        assert result.value == 0

    def test_alias(self):
        assert alias is slope_sign_changes_fn
