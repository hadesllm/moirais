"""Test poles_zeros (plzro)."""
import numpy as np
from moirais.fn.plzro import poles_zeros, plzro
from moirais.fn._containers import DescriptiveResult


class TestPolesZeros:
    def test_basic(self):
        b = [1.0, -1.0]
        a = [1.0, -0.5]
        result = poles_zeros(b, a)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "poles_zeros"

    def test_known_pole(self):
        b = [1.0]
        a = [1.0, -0.5]
        result = poles_zeros(b, a)
        poles = result.extra["poles"]
        assert len(poles) == 1
        assert abs(poles[0] - 0.5) < 1e-10

    def test_alias(self):
        assert plzro is poles_zeros
