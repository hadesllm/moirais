"""Test linear_convolution (lncon)."""
import numpy as np
from morie.fn.lncon import linear_convolution, lncon
from morie.fn._containers import DescriptiveResult


class TestLncon:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        h = np.array([1.0, 1.0])
        result = linear_convolution(x, h)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "linear_convolution"

    def test_output_length(self):
        x = np.array([1.0, 2.0, 3.0])
        h = np.array([1.0, 1.0])
        result = linear_convolution(x, h)
        assert len(result.extra["output"]) == 4

    def test_alias(self):
        assert lncon is linear_convolution
