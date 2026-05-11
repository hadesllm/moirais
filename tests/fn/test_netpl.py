"""Tests for network_path_length."""
import numpy as np, pytest
from morie.fn.netpl import network_path_length

class TestNetPath:
    def test_complete(self):
        A = np.ones((4, 4)) - np.eye(4)
        r = network_path_length(A)
        assert r.estimate == pytest.approx(1.0)

    def test_chain(self):
        A = np.zeros((4, 4))
        A[0,1] = A[1,0] = 1
        A[1,2] = A[2,1] = 1
        A[2,3] = A[3,2] = 1
        r = network_path_length(A)
        assert r.estimate == pytest.approx(5/3, abs=0.01)
