"""Test wss_distance (wssds)."""
import numpy as np
from moirais.fn.wssds import wss_distance, wssds
from moirais.fn._containers import DescriptiveResult


class TestWssds:
    def test_identical(self):
        S = np.array([1.0, 2.0, 3.0, 4.0])
        result = wss_distance(S, S)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "wss_distance"
        assert abs(result.value) < 1e-10

    def test_different(self):
        S1 = np.array([1.0, 2.0, 4.0, 8.0])
        S2 = np.array([1.0, 1.0, 1.0, 1.0])
        result = wss_distance(S1, S2)
        assert result.value > 0

    def test_alias(self):
        assert wssds is wss_distance
