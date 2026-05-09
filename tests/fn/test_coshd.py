"""Test cosh_distance (coshd)."""
import numpy as np
from moirais.fn.coshd import cosh_distance, coshd
from moirais.fn._containers import DescriptiveResult


class TestCoshd:
    def test_identical(self):
        S = np.array([1.0, 2.0, 3.0])
        result = cosh_distance(S, S)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "cosh_distance"
        assert abs(result.value) < 1e-10

    def test_different(self):
        S1 = np.array([1.0, 2.0])
        S2 = np.array([2.0, 4.0])
        result = cosh_distance(S1, S2)
        assert result.value > 0

    def test_alias(self):
        assert coshd is cosh_distance
