"""Test trapezoidal_integrate (trprl)."""
import numpy as np
import pytest

from moirais.fn.trprl import trapezoidal_integrate, trprl
from moirais.fn._containers import DescriptiveResult


class TestTrapezoidalIntegrate:
    def test_constant(self):
        x = np.ones(11)
        result = trapezoidal_integrate(x, dx=0.1)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "trapezoidal_integrate"
        assert np.isclose(result.value, 1.0)

    def test_linear(self):
        x = np.linspace(0, 1, 101)
        result = trapezoidal_integrate(x, dx=0.01)
        assert np.isclose(result.value, 0.5, atol=1e-4)

    def test_too_few(self):
        with pytest.raises(ValueError):
            trapezoidal_integrate([1.0])

    def test_alias(self):
        assert trprl is trapezoidal_integrate
