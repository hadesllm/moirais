"""Tests for nwstb -- network stability."""
import numpy as np
from morie.fn.nwstb import network_stability
from morie.fn._containers import DescriptiveResult


class TestNetworkStability:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))
        result = network_stability(X, n_boot=20)
        assert isinstance(result, DescriptiveResult)
        assert "CS_coefficient" in result.value

    def test_mean_cor_bounded(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((60, 4))
        result = network_stability(X, n_boot=10)
        assert -1 <= result.value["mean_cor"] <= 1
