"""Tests for morie.fn.feat -- Permutation feature importance."""

import numpy as np
from morie.fn.feat import feature_importance


class TestFeatureImportance:
    def test_important_feature_ranked_first(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 3))
        y = (X[:, 0] > 0).astype(int)

        def model_fn(X):
            return (X[:, 0] > 0).astype(int)

        result = feature_importance(model_fn, X, y, n_repeats=10)
        assert len(result["importance_mean"]) == 3
        assert result["importance_mean"][0] > result["importance_mean"][1]

    def test_irrelevant_features_near_zero(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((50, 2))
        y = np.ones(50, dtype=int)

        def model_fn(X):
            return np.ones(X.shape[0], dtype=int)

        result = feature_importance(model_fn, X, y, n_repeats=5)
        assert abs(result["importance_mean"][0]) < 0.01
        assert abs(result["importance_mean"][1]) < 0.01

    def test_std_shape(self):
        X = np.ones((10, 4))
        y = np.ones(10)
        model_fn = lambda X: np.ones(X.shape[0])
        result = feature_importance(model_fn, X, y, n_repeats=3)
        assert len(result["importance_std"]) == 4
