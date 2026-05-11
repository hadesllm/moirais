"""Tests for morie.fn.netcl — Node closeness centrality."""

import numpy as np
import pytest
from morie.fn.netcl import network_closeness


class TestNetworkCloseness:

    def test_returns_dict(self):
        A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=float)
        result = network_closeness(A)
        assert "closeness" in result

    def test_complete_graph_equal(self):
        A = np.ones((3, 3)) - np.eye(3)
        result = network_closeness(A)
        vals = list(result["closeness"].values())
        np.testing.assert_allclose(vals[0], vals[1], atol=1e-10)

    def test_nonnegative(self):
        A = np.array([[0, 0.5, 0], [0.5, 0, 0.3], [0, 0.3, 0]])
        result = network_closeness(A)
        for v in result["closeness"].values():
            assert v >= 0

    def test_isolated_node(self):
        A = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]], dtype=float)
        result = network_closeness(A)
        assert result["closeness"]["n0"] == 0.0
