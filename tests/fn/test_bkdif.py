"""Test backward_difference (bkdif)."""
import numpy as np

from morie.fn.bkdif import backward_difference, bkdif
from morie.fn._containers import DescriptiveResult


class TestBackwardDifference:
    def test_basic(self):
        x = np.array([1.0, 3.0, 6.0, 10.0])
        result = backward_difference(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "backward_difference"

    def test_linear(self):
        x = np.arange(5, dtype=float) * 3.0
        result = backward_difference(x)
        assert np.isclose(result.value, 3.0)

    def test_alias(self):
        assert bkdif is backward_difference
