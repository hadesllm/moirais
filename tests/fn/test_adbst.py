"""Test adaboost_bio (adbst)."""
import numpy as np
from moirais.fn.adbst import adaboost_bio, adbst
from moirais.fn._containers import DescriptiveResult


class TestAdbst:
    def test_basic(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.standard_normal((30, 2)) + 2, rng.standard_normal((30, 2)) - 2])
        y = np.array([1] * 30 + [0] * 30)
        result = adaboost_bio(X[:50], y[:50], X[50:], n_estimators=10)
        assert isinstance(result, DescriptiveResult)
        assert result.name == "adaboost_bio"
        assert len(result.extra["predictions"]) == 10

    def test_train_acc(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.standard_normal((30, 2)) + 3, rng.standard_normal((30, 2)) - 3])
        y = np.array([1] * 30 + [0] * 30)
        result = adaboost_bio(X, y, X, n_estimators=20)
        assert result.extra["train_accuracy"] >= 0.8

    def test_alias(self):
        assert adbst is adaboost_bio
