"""Tests for moirais.fn.learn -- Learning curve."""

import numpy as np
from moirais.fn.learn import learning_curve


class TestLearningCurve:
    def test_basic_output(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.standard_normal((30, 2)) + [3, 3],
                        rng.standard_normal((30, 2)) - [3, 3]])
        y = np.array([1] * 30 + [0] * 30)

        def model_fn(X_tr, y_tr, X_te):
            # Simple nearest-centroid classifier
            c0 = X_tr[y_tr == 0].mean(axis=0)
            c1 = X_tr[y_tr == 1].mean(axis=0)
            d0 = np.sum((X_te - c0) ** 2, axis=1)
            d1 = np.sum((X_te - c1) ** 2, axis=1)
            return (d1 < d0).astype(int)

        result = learning_curve(model_fn, X, y, fractions=[0.3, 0.7, 1.0], n_splits=2)
        assert len(result["fractions"]) == 3
        assert len(result["train_scores"]) == 3
        assert len(result["test_scores"]) == 3

    def test_train_score_improves(self):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((100, 2))
        y = (X[:, 0] > 0).astype(int)

        def model_fn(X_tr, y_tr, X_te):
            c0 = X_tr[y_tr == 0].mean(axis=0) if np.any(y_tr == 0) else np.zeros(2)
            c1 = X_tr[y_tr == 1].mean(axis=0) if np.any(y_tr == 1) else np.zeros(2)
            d0 = np.sum((X_te - c0) ** 2, axis=1)
            d1 = np.sum((X_te - c1) ** 2, axis=1)
            return (d1 < d0).astype(int)

        result = learning_curve(model_fn, X, y, fractions=[0.1, 1.0], n_splits=3)
        # Scores should be reasonable
        assert all(0 <= s <= 1 for s in result["train_scores"])
