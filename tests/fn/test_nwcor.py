"""Tests for nwcor -- partial correlation network."""
import numpy as np
from moirais.fn.nwcor import network_correlation
from moirais.fn._containers import DescriptiveResult


class TestNetworkCorrelation:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 5))
        R = np.corrcoef(X, rowvar=False)
        result = network_correlation(R)
        assert isinstance(result, DescriptiveResult)
        assert "edges" in result.value

    def test_threshold(self):
        R = np.eye(4)
        R[0, 1] = R[1, 0] = 0.8
        R[2, 3] = R[3, 2] = 0.3
        result = network_correlation(R, threshold=0.5)
        assert result.extra["n_nodes"] == 4
