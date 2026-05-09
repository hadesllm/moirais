"""Test activity (sactv)."""
import numpy as np
from moirais.fn.sactv import activity, sactv
from moirais.fn._containers import DescriptiveResult


class TestActivity:
    def test_constant(self):
        x = np.array([3.0, 3.0, 3.0])
        result = activity(x)
        assert isinstance(result, DescriptiveResult)
        assert result.value == 0.0

    def test_positive(self):
        x = np.array([1.0, 2.0, 3.0])
        assert activity(x).value > 0

    def test_alias(self):
        assert sactv is activity
