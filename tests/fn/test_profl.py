"""Tests for moirais.fn.profl -- Procrustes rotation."""

import numpy as np
from moirais.fn.profl import procrustes, profl
from moirais.fn._containers import DescriptiveResult


class TestProcrustes:
    def test_alias(self):
        assert profl is procrustes

    def test_returns_result(self):
        rng = np.random.default_rng(42)
        S = rng.standard_normal((20, 3))
        T = S @ np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]).astype(float)
        res = procrustes(S, T)
        assert isinstance(res, DescriptiveResult)

    def test_disparity_small_for_rotated(self):
        rng = np.random.default_rng(42)
        S = rng.standard_normal((30, 2))
        theta = 0.5
        R = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta), np.cos(theta)]])
        T = S @ R
        res = procrustes(S, T)
        assert res.extra["disparity"] < 1e-6

    def test_rotation_matrix_orthogonal(self):
        rng = np.random.default_rng(42)
        S = rng.standard_normal((20, 3))
        T = rng.standard_normal((20, 3))
        res = procrustes(S, T)
        R = res.extra["rotation"]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-8)
