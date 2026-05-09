"""Test forward_difference (frdif)."""
import numpy as np

from moirais.fn.frdif import forward_difference, frdif
from moirais.fn._containers import DescriptiveResult


class TestForwardDifference:
    def test_basic(self):
        x = np.array([1.0, 3.0, 6.0, 10.0])
        result = forward_difference(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "forward_difference"

    def test_linear(self):
        x = np.arange(5, dtype=float) * 2.0
        result = forward_difference(x)
        assert np.isclose(result.value, 2.0)

    def test_second_order(self):
        x = np.array([1.0, 4.0, 9.0, 16.0, 25.0])
        result = forward_difference(x, order=2)
        assert np.isclose(result.value, 2.0)

    def test_alias(self):
        assert frdif is forward_difference
