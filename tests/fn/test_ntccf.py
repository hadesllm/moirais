"""Tests for network_clustering_coeff."""
import numpy as np, pytest
from moirais.fn.ntccf import network_clustering_coeff

class TestNetClust:
    def test_complete(self):
        A = np.ones((4, 4)) - np.eye(4)
        r = network_clustering_coeff(A)
        assert r.estimate == pytest.approx(1.0)

    def test_star(self):
        A = np.zeros((5, 5))
        A[0, 1:] = 1
        A[1:, 0] = 1
        r = network_clustering_coeff(A)
        assert r.estimate == pytest.approx(0.0)
