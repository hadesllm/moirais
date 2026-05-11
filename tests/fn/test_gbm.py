"""Tests for morie.fn.gbm -- Gradient boosting machine."""

import numpy as np
import pytest
from morie.fn.gbm import gradient_boosting


class TestGradientBoosting:
    def test_separable(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.standard_normal((30, 2)) + [3, 3],
                        rng.standard_normal((30, 2)) - [3, 3]])
        y = np.array([1.0] * 30 + [0.0] * 30)
        result = gradient_boosting(X, y, n_estimators=50, lr=0.1)
        assert "predictions" in result
        assert "feature_importance" in result
        acc = np.mean(result["predictions"] == y)
        assert acc >= 0.7

    def test_importance_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 3))
        y = (X[:, 0] > 0).astype(float)
        result = gradient_boosting(X, y, n_estimators=10)
        assert len(result["feature_importance"]) == 3

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            gradient_boosting(np.zeros((5, 2)), np.zeros(3))
