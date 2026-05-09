"""Tests for moirais.fn.netbr — Bridge centrality."""

import numpy as np
import pytest
from moirais.fn.netbr import network_bridge


class TestNetworkBridge:

    def test_returns_dict(self):
        A = np.array([[0, 0.5, 0.1, 0], [0.5, 0, 0, 0.2],
                       [0.1, 0, 0, 0.3], [0, 0.2, 0.3, 0]])
        communities = [0, 0, 1, 1]
        result = network_bridge(A, communities)
        assert "bridge_strength" in result

    def test_bridge_node_higher(self):
        # Node 1 connects community {0,1} to {2,3}
        A = np.array([[0, 1, 0, 0], [1, 0, 1, 0],
                       [0, 1, 0, 1], [0, 0, 1, 0]], dtype=float)
        communities = [0, 0, 1, 1]
        result = network_bridge(A, communities)
        assert result["bridge_strength"]["n1"] > result["bridge_strength"]["n0"]

    def test_dict_communities(self):
        A = np.array([[0, 0.5, 0.1], [0.5, 0, 0.2], [0.1, 0.2, 0]])
        communities = {"A": [0, 1], "B": [2]}
        result = network_bridge(A, communities)
        assert "bridge_expected_influence" in result

    def test_no_bridge_within_community(self):
        A = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=float)
        communities = [0, 0, 1]
        result = network_bridge(A, communities)
        # Nodes 0,1 have no edges to community 1
        assert result["bridge_strength"]["n0"] == 0.0
