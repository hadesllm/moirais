"""Tests for moirais.fn.rforc -- Random forest classifier."""

import numpy as np
import pytest
from moirais.fn.rforc import random_forest


class TestRandomForest:
    def test_basic_classification(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.standard_normal((30, 2)) + [3, 3],
                        rng.standard_normal((30, 2)) - [3, 3]])
        y = np.array([1] * 30 + [0] * 30)
        result = random_forest(X, y, n_trees=5, max_depth=3)
        assert "predictions" in result
        assert "feature_importance" in result
        assert "oob_score" in result
        assert len(result["predictions"]) == 60

    def test_importance_shape(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((20, 4))
        y = (X[:, 0] > 0).astype(int)
        result = random_forest(X, y, n_trees=3, max_depth=2)
        assert len(result["feature_importance"]) == 4

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            random_forest(np.zeros((5, 2)), np.zeros(3))
