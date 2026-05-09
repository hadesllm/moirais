"""Tests for moirais.fn.sspace — state-space model via Kalman filter."""
import numpy as np
import pytest
from moirais.fn.sspace import state_space


class TestStateSpace:
    def test_random_walk(self):
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal(100))
        res = state_space(y, F=np.array([[1.0]]), H=np.array([[1.0]]))
        states = res.extra["filtered_states"]
        assert states.shape == (100, 1)
        assert np.isfinite(res.extra["log_likelihood"])

    def test_shape_multivariate(self):
        rng = np.random.default_rng(42)
        y = rng.standard_normal((80, 2))
        F = np.eye(2)
        H = np.eye(2)
        res = state_space(y, F=F, H=H)
        assert res.extra["T"] == 80
        assert res.extra["k"] == 2

    def test_defaults(self):
        rng = np.random.default_rng(42)
        y = np.cumsum(rng.standard_normal(50))
        res = state_space(y)
        assert res.extra["filtered_states"].shape[0] == 50
