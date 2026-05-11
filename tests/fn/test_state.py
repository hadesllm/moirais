"""Tests for morie.fn.state -- Kalman filter."""
import numpy as np
import pytest
from morie.fn.state import kalman_filter


class TestKalman:
    def test_basic(self):
        rng = np.random.default_rng(42)
        n = 50
        true_state = np.cumsum(rng.standard_normal(n)) * 0.1
        y = true_state + rng.normal(0, 1, n)
        F = np.array([[1.0]])
        H = np.array([[1.0]])
        Q = np.array([[0.01]])
        R = np.array([[1.0]])
        res = kalman_filter(y, F, H, Q, R)
        assert res.extra["filtered_states"].shape == (n, 1)

    def test_dimension_mismatch(self):
        with pytest.raises(ValueError):
            kalman_filter(np.ones(10), np.eye(2), np.ones((1, 3)), np.eye(2), np.eye(1))

    def test_cheatsheet(self):
        from morie.fn.state import cheatsheet
        assert isinstance(cheatsheet(), str)
