"""Tests for moirais.fn.vrimx -- Varimax rotation."""

import numpy as np
from moirais.fn.vrimx import varimax, vrimx
from moirais.fn._containers import DescriptiveResult


class TestVarimax:
    def test_alias(self):
        assert vrimx is varimax

    def test_returns_result(self):
        rng = np.random.default_rng(42)
        L = rng.standard_normal((6, 2))
        res = varimax(L)
        assert isinstance(res, DescriptiveResult)

    def test_rotation_orthogonal(self):
        rng = np.random.default_rng(42)
        L = rng.standard_normal((8, 3))
        res = varimax(L)
        R = res.extra["rotation_matrix"]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-6)

    def test_same_shape(self):
        rng = np.random.default_rng(42)
        L = rng.standard_normal((5, 2))
        res = varimax(L)
        assert res.value.shape == (5, 2)
