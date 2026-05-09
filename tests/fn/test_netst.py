"""Tests for moirais.fn.netst — Node strength centrality."""

import numpy as np
import pytest
from moirais.fn.netst import network_strength


class TestNetworkStrength:

    def test_returns_dict(self):
        A = np.array([[0, 0.5, 0.3], [0.5, 0, 0.2], [0.3, 0.2, 0]])
        result = network_strength(A)
        assert "strength" in result and "mean" in result

    def test_correct_strength(self):
        A = np.array([[0, 1.0, 0.5], [1.0, 0, 0.0], [0.5, 0.0, 0]])
        result = network_strength(A)
        assert abs(result["strength"]["n0"] - 1.5) < 1e-10
        assert abs(result["strength"]["n1"] - 1.0) < 1e-10
        assert abs(result["strength"]["n2"] - 0.5) < 1e-10

    def test_custom_names(self):
        A = np.array([[0, 0.5], [0.5, 0]])
        result = network_strength(A, node_names=["a", "b"])
        assert "a" in result["strength"]

    def test_zero_matrix(self):
        A = np.zeros((3, 3))
        result = network_strength(A)
        assert result["mean"] == 0.0
