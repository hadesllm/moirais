"""Tests for moirais.fn.netcm — Community detection."""

import numpy as np
import pytest
from moirais.fn.netcm import network_communities


class TestNetworkCommunities:

    def test_returns_dict(self):
        A = np.array([[0, 1, 0, 0], [1, 0, 0, 0],
                       [0, 0, 0, 1], [0, 0, 1, 0]], dtype=float)
        result = network_communities(A, n_communities=2)
        assert "labels" in result
        assert "n_communities" in result

    def test_two_clear_communities(self):
        # Two disconnected cliques
        A = np.array([[0, 1, 1, 0, 0, 0],
                       [1, 0, 1, 0, 0, 0],
                       [1, 1, 0, 0, 0, 0],
                       [0, 0, 0, 0, 1, 1],
                       [0, 0, 0, 1, 0, 1],
                       [0, 0, 0, 1, 1, 0]], dtype=float)
        result = network_communities(A, n_communities=2)
        labels = list(result["labels"].values())
        # First 3 should be same community, last 3 same community
        assert labels[0] == labels[1] == labels[2]
        assert labels[3] == labels[4] == labels[5]
        assert labels[0] != labels[3]

    def test_modularity_method(self):
        A = np.array([[0, 1, 0, 0], [1, 0, 0, 0],
                       [0, 0, 0, 1], [0, 0, 1, 0]], dtype=float)
        result = network_communities(A, method="modularity")
        assert result["method"] == "modularity"

    def test_sizes_sum_to_n(self):
        A = np.ones((5, 5)) - np.eye(5)
        result = network_communities(A, n_communities=2)
        assert sum(result["sizes"].values()) == 5
