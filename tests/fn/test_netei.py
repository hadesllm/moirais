"""Tests for moirais.fn.netei — Expected influence."""

import numpy as np
import pytest
from moirais.fn.netei import network_expected_influence


class TestNetworkExpectedInfluence:

    def test_one_step(self):
        A = np.array([[0, 0.5, -0.3], [0.5, 0, 0.2], [-0.3, 0.2, 0]])
        result = network_expected_influence(A, step=1)
        assert "expected_influence" in result
        np.testing.assert_allclose(result["expected_influence"]["n0"], 0.2, atol=1e-10)

    def test_two_step(self):
        A = np.array([[0, 0.5, -0.3], [0.5, 0, 0.2], [-0.3, 0.2, 0]])
        result = network_expected_influence(A, step=2)
        assert result["step"] == 2

    def test_positive_network(self):
        A = np.array([[0, 0.5, 0.3], [0.5, 0, 0.2], [0.3, 0.2, 0]])
        result = network_expected_influence(A)
        for v in result["expected_influence"].values():
            assert v > 0

    def test_signed_edges(self):
        # All negative edges: expected influence should be negative
        A = np.array([[0, -1, -1], [-1, 0, -1], [-1, -1, 0]], dtype=float)
        result = network_expected_influence(A)
        for v in result["expected_influence"].values():
            assert v < 0
