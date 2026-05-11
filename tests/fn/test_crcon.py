"""Test circular_convolution (crcon)."""
import numpy as np
from morie.fn.crcon import circular_convolution, crcon
from morie.fn._containers import DescriptiveResult


class TestCrcon:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        h = np.array([1.0, 0.0, 0.0, 0.0])
        result = circular_convolution(x, h)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "circular_convolution"

    def test_identity(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        h = np.array([1.0, 0.0, 0.0, 0.0])
        result = circular_convolution(x, h)
        assert np.allclose(result.extra["output"], x)

    def test_alias(self):
        assert crcon is circular_convolution
