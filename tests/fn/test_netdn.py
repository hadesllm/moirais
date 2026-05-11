"""Tests for morie.fn.netdn — Network density."""

import numpy as np
import pytest
from morie.fn.netdn import network_density


class TestNetworkDensity:

    def test_complete_graph(self):
        A = np.ones((4, 4)) - np.eye(4)
        assert network_density(A) == 1.0

    def test_empty_graph(self):
        A = np.zeros((4, 4))
        assert network_density(A) == 0.0

    def test_partial(self):
        A = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
        # 2 edges out of 3 possible
        np.testing.assert_allclose(network_density(A), 2 / 3)

    def test_single_node(self):
        assert network_density(np.zeros((1, 1))) == 0.0

    def test_in_range(self, rng):
        A = rng.standard_normal((5, 5))
        A = (A + A.T) / 2
        np.fill_diagonal(A, 0)
        d = network_density(A)
        assert 0 <= d <= 1
