"""Test equalization_inverse (eqint)."""
import numpy as np

from morie.fn.eqint import equalization_inverse, eqint
from morie.fn._containers import DescriptiveResult


class TestEqualizationInverse:
    def test_basic(self):
        H = np.array([1.0 + 0j, 0.5 + 0.5j, 0.3 - 0.1j])
        result = equalization_inverse(H)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "equalization_inverse"
        assert result.value > 0

    def test_flat_channel(self):
        H = np.ones(10, dtype=complex)
        result = equalization_inverse(H, regularize=0.0)
        assert np.isclose(result.extra["mean_gain"], 1.0, atol=1e-10)

    def test_alias(self):
        assert eqint is equalization_inverse
