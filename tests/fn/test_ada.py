"""Tests for morie.fn.ada -- AdaBoost."""

import numpy as np
from morie.fn.ada import adaboost, ada
from morie.fn._containers import DescriptiveResult


class TestAda:
    def test_alias(self):
        assert ada is adaboost

    def test_better_than_random(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 1, (30, 2)), rng.normal(2, 1, (30, 2))])
        y = np.array([-1] * 30 + [1] * 30)
        result = adaboost(X, y, n_estimators=20)
        assert isinstance(result, DescriptiveResult)
        assert result.value > 0.5

    def test_auto_label_conversion(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.normal(0, 1, (20, 1)), rng.normal(3, 1, (20, 1))])
        y = np.array([0] * 20 + [1] * 20)
        result = adaboost(X, y, n_estimators=10)
        assert result.value > 0.5
