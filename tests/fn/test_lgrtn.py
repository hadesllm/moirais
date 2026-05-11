"""Test log_spectral_dist (lgrtn)."""
import numpy as np
from morie.fn.lgrtn import log_spectral_dist, lgrtn
from morie.fn._containers import DescriptiveResult


class TestLgrtn:
    def test_basic(self):
        S1 = np.array([1.0, 2.0, 3.0, 4.0])
        S2 = np.array([1.0, 2.0, 3.0, 4.0])
        result = log_spectral_dist(S1, S2)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "log_spectral_distance"
        assert abs(result.value) < 1e-10

    def test_different(self):
        S1 = np.array([1.0, 2.0, 3.0])
        S2 = np.array([2.0, 4.0, 6.0])
        result = log_spectral_dist(S1, S2)
        assert result.value > 0

    def test_alias(self):
        assert lgrtn is log_spectral_dist
