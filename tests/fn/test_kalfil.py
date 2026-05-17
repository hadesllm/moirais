"""Tests for morie.fn.kalfil -- Kalman filter."""

import numpy as np
from morie.fn.kalfil import kalman_filter, kalfil
from morie.fn._containers import DescriptiveResult


class TestKalfil:
    def test_alias(self):
        assert kalfil is kalman_filter

    def test_tracks_constant(self):
        obs = np.ones(50) + np.random.default_rng(42).normal(0, 0.1, 50)
        r = kalman_filter(obs, R=0.1, Q=0.01)
        assert isinstance(r, DescriptiveResult)
        assert abs(r.value[-1] - 1.0) < 0.3

    def test_shape(self):
        obs = np.random.default_rng(42).normal(0, 1, 30)
        r = kalman_filter(obs)
        assert len(r.value) == 30
