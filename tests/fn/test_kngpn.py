"""Tests for morie.fn.kngpn -- network centrality measures."""

import numpy as np
from morie.fn.kngpn import crime_network_centrality, kngpn
from morie.fn._containers import DescriptiveResult


class TestKngpn:
    def test_alias(self):
        assert kngpn is crime_network_centrality

    def test_star_graph(self):
        A = np.zeros((5, 5))
        for i in range(1, 5):
            A[0, i] = A[i, 0] = 1
        r = crime_network_centrality(A)
        assert isinstance(r, DescriptiveResult)
        assert r.value["most_central"] == 0

    def test_all_keys(self):
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        r = crime_network_centrality(A)
        for key in ["degree", "betweenness", "closeness", "eigenvector"]:
            assert key in r.value
