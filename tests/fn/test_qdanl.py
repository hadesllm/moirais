"""Tests for morie.fn.qdanl -- Quadratic discriminant analysis."""

import numpy as np
from morie.fn.qdanl import qda, qdanl
from morie.fn._containers import DescriptiveResult


class TestQda:
    def test_alias(self):
        assert qdanl is qda

    def test_returns_result(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 1, (30, 3)), rng.normal(5, 1, (30, 3))])
        y = np.array([0]*30 + [1]*30)
        res = qda(X, y, X[:5])
        assert isinstance(res, DescriptiveResult)

    def test_predictions_valid(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 0.5, (40, 2)), rng.normal(5, 0.5, (40, 2))])
        y = np.array([0]*40 + [1]*40)
        res = qda(X, y, X)
        preds = res.value
        assert len(preds) == 80
        accuracy = np.mean(preds == y)
        assert accuracy > 0.7

    def test_has_posteriors(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 1, (20, 2)), rng.normal(3, 1, (20, 2))])
        y = np.array([0]*20 + [1]*20)
        res = qda(X, y, X[:5])
        assert "log_posteriors" in res.extra
        assert res.extra["log_posteriors"].shape == (5, 2)
