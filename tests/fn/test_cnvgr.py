"""Test convergence_rate (cnvgr)."""
import numpy as np
from morie.fn.cnvgr import convergence_rate, cnvgr
from morie.fn._containers import DescriptiveResult


class TestCnvgr:
    def test_basic(self):
        result = convergence_rate(0.01, [1.0, 2.0, 4.0])
        assert isinstance(result, DescriptiveResult)
        assert result.name == "convergence_rate"
        assert result.value > 0

    def test_known_value(self):
        result = convergence_rate(0.1, [1.0])
        assert abs(result.value - 5.0) < 1e-10

    def test_alias(self):
        assert cnvgr is convergence_rate
