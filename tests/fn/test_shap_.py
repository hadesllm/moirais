"""Tests for morie.fn.shap_ -- Permutation-based SHAP values."""

import numpy as np
from morie.fn.shap_ import shap_values


class TestShapValues:
    def test_shape(self):
        X = np.array([[1, 2], [3, 4], [5, 6]], dtype=float)
        model = lambda X: X[:, 0] * 2 + X[:, 1]
        sv = shap_values(model, X, n_repeats=5)
        assert sv.shape == (3, 2)

    def test_important_feature_has_higher_shap(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 3))
        # Feature 0 is the only one that matters
        model = lambda X: X[:, 0] * 10
        sv = shap_values(model, X, n_repeats=20)
        # Feature 0 should have largest absolute attribution
        assert np.abs(sv[:, 0]).mean() > np.abs(sv[:, 1]).mean()
        assert np.abs(sv[:, 0]).mean() > np.abs(sv[:, 2]).mean()

    def test_constant_model_zero_shap(self):
        X = np.ones((10, 2))
        model = lambda X: np.ones(X.shape[0]) * 5
        sv = shap_values(model, X, n_repeats=5)
        assert np.allclose(sv, 0.0, atol=1e-10)
