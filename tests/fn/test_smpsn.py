"""Test simpson_integrate (smpsn)."""
import numpy as np
import pytest

from morie.fn.smpsn import simpson_integrate, smpsn
from morie.fn._containers import DescriptiveResult


class TestSimpsonIntegrate:
    def test_constant(self):
        x = np.ones(11)
        result = simpson_integrate(x, dx=0.1)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "simpson_integrate"
        assert np.isclose(result.value, 1.0, atol=1e-10)

    def test_quadratic(self):
        t = np.linspace(0, 1, 101)
        x = t ** 2
        result = simpson_integrate(x, dx=0.01)
        assert np.isclose(result.value, 1.0 / 3.0, atol=1e-5)

    def test_too_few(self):
        with pytest.raises(ValueError):
            simpson_integrate([1.0, 2.0])

    def test_alias(self):
        assert smpsn is simpson_integrate
