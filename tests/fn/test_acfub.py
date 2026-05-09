"""Test acf_unbiased (acfub)."""
import numpy as np
from moirais.fn.acfub import acf_unbiased, acfub
from moirais.fn._containers import DescriptiveResult


class TestAcfub:
    def test_basic(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = acf_unbiased(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "acf_unbiased"

    def test_zero_lag_equals_variance(self):
        x = np.random.default_rng(42).standard_normal(64)
        result = acf_unbiased(x)
        x_centered = x - np.mean(x)
        var = np.var(x_centered, ddof=0)
        assert np.isclose(result.extra["acf"][0], var, rtol=0.01)

    def test_alias(self):
        assert acfub is acf_unbiased
