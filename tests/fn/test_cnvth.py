"""Test convolution_theorem_verify (cnvth)."""
import numpy as np
from moirais.fn.cnvth import convolution_theorem_verify, cnvth
from moirais.fn._containers import DescriptiveResult


class TestCnvth:
    def test_basic(self):
        x = np.array([1.0, 2.0, 3.0])
        h = np.array([1.0, 1.0])
        result = convolution_theorem_verify(x, h)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "convolution_theorem_verify"

    def test_small_error(self):
        x = np.random.default_rng(42).standard_normal(32)
        h = np.random.default_rng(43).standard_normal(8)
        result = convolution_theorem_verify(x, h)
        assert result.value < 1e-10

    def test_alias(self):
        assert cnvth is convolution_theorem_verify
