"""Test gauss_quadrature (gqadr)."""
import numpy as np

from morie.fn.gqadr import gauss_quadrature, gqadr
from morie.fn._containers import DescriptiveResult


class TestGaussQuadrature:
    def test_polynomial(self):
        result = gauss_quadrature(lambda x: x ** 2, 0.0, 1.0, n=5)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "gauss_quadrature"
        assert np.isclose(result.value, 1.0 / 3.0, atol=1e-12)

    def test_sine(self):
        result = gauss_quadrature(np.sin, 0.0, np.pi, n=10)
        assert np.isclose(result.value, 2.0, atol=1e-10)

    def test_constant(self):
        result = gauss_quadrature(lambda x: np.ones_like(x) * 5.0, 0.0, 3.0)
        assert np.isclose(result.value, 15.0, atol=1e-10)

    def test_alias(self):
        assert gqadr is gauss_quadrature
