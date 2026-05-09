"""Tests for nwcom -- network community detection."""
import numpy as np
from moirais.fn.nwcom import network_community
from moirais.fn._containers import DescriptiveResult


class TestNetworkCommunity:
    def test_basic(self):
        A = np.zeros((6, 6))
        A[0, 1] = A[1, 0] = 0.8
        A[0, 2] = A[2, 0] = 0.7
        A[1, 2] = A[2, 1] = 0.9
        A[3, 4] = A[4, 3] = 0.8
        A[3, 5] = A[5, 3] = 0.7
        A[4, 5] = A[5, 4] = 0.9
        result = network_community(A, n_communities=2)
        assert isinstance(result, DescriptiveResult)
        assert "labels" in result.value

    def test_modularity_nonneg(self):
        rng = np.random.default_rng(42)
        A = np.abs(rng.standard_normal((8, 8)))
        np.fill_diagonal(A, 0)
        A = (A + A.T) / 2
        result = network_community(A, n_communities=2)
        assert isinstance(result.value["modularity"], float)
