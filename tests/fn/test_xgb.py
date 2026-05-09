"""Tests for moirais.fn.xgb -- XGBoost/gradient boosting wrapper."""

import numpy as np
import pytest
from moirais.fn.xgb import xgb_classify


class TestXgbClassify:
    def test_linearly_separable(self):
        """Should perfectly classify linearly separable data."""
        rng = np.random.default_rng(42)
        X = np.vstack([rng.standard_normal((30, 2)) + [2, 2],
                        rng.standard_normal((30, 2)) - [2, 2]])
        y = np.array([1] * 30 + [0] * 30, dtype=float)
        result = xgb_classify(X, y, n_estimators=50)
        assert "predictions" in result
        assert "feature_importance" in result
        assert "accuracy" in result
        assert result["accuracy"] >= 0.8

    def test_returns_correct_shapes(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 3))
        y = (X[:, 0] > 0).astype(float)
        result = xgb_classify(X, y, n_estimators=10)
        assert len(result["predictions"]) == 20
        assert len(result["feature_importance"]) == 3

    def test_mismatched_shapes_raises(self):
        with pytest.raises(ValueError):
            xgb_classify(np.zeros((5, 2)), np.zeros(3))
