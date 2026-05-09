"""Tests for network_degree."""
import numpy as np, pytest
from moirais.fn.netdg import network_degree

class TestNetDeg:
    def test_complete(self):
        A = np.ones((4, 4)) - np.eye(4)
        r = network_degree(A)
        assert r.extra["mean_degree"] == 3.0

    def test_empty(self):
        A = np.eye(3)
        r = network_degree(A)
        assert r.extra["mean_degree"] == 0.0
