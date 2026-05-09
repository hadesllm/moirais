"""Test linearity_test (lnrty)."""
import numpy as np

from moirais.fn.lnrty import linearity_test, lnrty
from moirais.fn._containers import DescriptiveResult


class TestLinearityTest:
    def test_linear(self):
        x = np.arange(50, dtype=float)
        y = 3.0 * x + 2.0
        result = linearity_test(x, y)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "linearity_test"
        assert result.extra["linear"] is True
        assert np.isclose(result.value, 1.0, atol=1e-10)

    def test_nonlinear(self):
        rng = np.random.default_rng(42)
        x = np.arange(50, dtype=float)
        y = rng.standard_normal(50)
        result = linearity_test(x, y)
        assert result.extra["linear"] is False

    def test_alias(self):
        assert lnrty is linearity_test
