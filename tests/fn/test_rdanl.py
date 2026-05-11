"""Tests for morie.fn.rdanl -- Regularized discriminant analysis."""

import numpy as np
from morie.fn.rdanl import rda, rdanl
from morie.fn._containers import DescriptiveResult


class TestRda:
    def test_alias(self):
        assert rdanl is rda

    def test_returns_result(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 1, (30, 3)), rng.normal(5, 1, (30, 3))])
        y = np.array([0]*30 + [1]*30)
        res = rda(X, y, X[:5])
        assert isinstance(res, DescriptiveResult)

    def test_alpha_zero_like_lda(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 0.5, (40, 2)), rng.normal(5, 0.5, (40, 2))])
        y = np.array([0]*40 + [1]*40)
        res = rda(X, y, X, alpha=0.0)
        accuracy = np.mean(res.value == y)
        assert accuracy > 0.7

    def test_alpha_one_like_qda(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 0.5, (40, 2)), rng.normal(5, 0.5, (40, 2))])
        y = np.array([0]*40 + [1]*40)
        res = rda(X, y, X, alpha=1.0)
        accuracy = np.mean(res.value == y)
        assert accuracy > 0.7
