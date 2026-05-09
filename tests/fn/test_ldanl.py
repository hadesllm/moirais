"""Tests for moirais.fn.ldanl -- Linear discriminant analysis."""

import numpy as np
from moirais.fn.ldanl import lda, ldanl
from moirais.fn._containers import LdaRes


class TestLda:
    def test_alias(self):
        assert ldanl is lda

    def test_returns_lda_res(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 1, (30, 3)), rng.normal(3, 1, (30, 3))])
        y = np.array([0]*30 + [1]*30)
        res = lda(X, y, n_components=1)
        assert isinstance(res, LdaRes)

    def test_shapes(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 1, (20, 4)), rng.normal(2, 1, (20, 4)), rng.normal(4, 1, (20, 4))])
        y = np.array([0]*20 + [1]*20 + [2]*20)
        res = lda(X, y, n_components=2)
        assert res.components.shape == (4, 2)
        assert res.projected.shape == (60, 2)

    def test_variance_ratio_valid(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 1, (30, 3)), rng.normal(5, 1, (30, 3))])
        y = np.array([0]*30 + [1]*30)
        res = lda(X, y)
        assert np.all(res.explained_variance_ratio >= -1e-10)
