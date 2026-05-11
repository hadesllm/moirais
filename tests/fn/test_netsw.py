"""Tests for network_small_world."""
import numpy as np, pytest
from morie.fn.netsw import network_small_world

class TestNetSW:
    def test_complete(self):
        A = np.ones((6, 6)) - np.eye(6)
        r = network_small_world(A, n_random=5, seed=0)
        assert r.measure == "small_world_sigma"
        assert np.isfinite(r.estimate)

    def test_ring(self):
        n = 20
        A = np.zeros((n, n))
        for i in range(n):
            for offset in [1, 2]:
                A[i, (i + offset) % n] = 1
                A[(i + offset) % n, i] = 1
        r = network_small_world(A, n_random=10, seed=0)
        assert r.estimate >= 0
        assert np.isfinite(r.estimate)
