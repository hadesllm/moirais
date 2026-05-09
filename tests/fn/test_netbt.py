"""Tests for moirais.fn.netbt — Node betweenness centrality."""

import numpy as np
import pytest
from moirais.fn.netbt import network_betweenness


class TestNetworkBetweenness:

    def test_returns_dict(self):
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        result = network_betweenness(A)
        assert "betweenness" in result

    def test_bridge_node_highest(self):
        # Star topology: center node should have highest betweenness
        A = np.array([
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ], dtype=float)
        result = network_betweenness(A)
        assert result["betweenness"]["n0"] >= result["betweenness"]["n1"]

    def test_nonnegative(self):
        A = np.array([[0, 0.5, 0.3], [0.5, 0, 0.2], [0.3, 0.2, 0]])
        result = network_betweenness(A)
        for v in result["betweenness"].values():
            assert v >= 0

    def test_isolated_node_zero(self):
        A = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
        result = network_betweenness(A)
        assert result["betweenness"]["n0"] == 0.0
