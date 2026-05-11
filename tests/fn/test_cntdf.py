"""Test central_difference (cntdf)."""
import numpy as np

from morie.fn.cntdf import central_difference, cntdf
from morie.fn._containers import DescriptiveResult


class TestCentralDifference:
    def test_basic(self):
        x = np.array([1.0, 3.0, 6.0, 10.0, 15.0])
        result = central_difference(x)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "central_difference"

    def test_linear(self):
        x = np.arange(10, dtype=float) * 2.0
        result = central_difference(x)
        assert np.isclose(result.value, 2.0)

    def test_alias(self):
        assert cntdf is central_difference
