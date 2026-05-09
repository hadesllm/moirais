"""Tests for moirais.fn.cvgen -- Cross-validation for genomic prediction."""

import numpy as np
import pytest
from moirais.fn.cvgen import cvgen


def _make_data(n=40, p=30, seed=42):
    rng = np.random.default_rng(seed)
    Z = rng.choice([0, 1, 2], size=(n, p)).astype(float)
    pf = np.clip(np.mean(Z, axis=0) / 2.0, 0.01, 0.99)
    M = Z - 2.0 * pf
    G = (M @ M.T) / max(2.0 * np.sum(pf * (1 - pf)), 1e-6)
    L = np.linalg.cholesky(G + np.eye(n) * 0.05)
    y = L @ rng.standard_normal(n) + rng.standard_normal(n) * 0.3
    return y, G


class TestCvgen:
    def test_fold_accuracies_length(self):
        y, G = _make_data()
        res = cvgen(y, G, n_folds=5)
        assert len(res.extra["fold_accuracies"]) == 5

    def test_predictions_length(self):
        y, G = _make_data()
        res = cvgen(y, G, n_folds=5)
        assert len(res.extra["predictions"]) == len(y)

    def test_se_positive(self):
        y, G = _make_data()
        res = cvgen(y, G, n_folds=5)
        assert res.extra["se_accuracy"] >= 0

    def test_dimension_mismatch(self):
        y, G = _make_data()
        with pytest.raises(ValueError):
            cvgen(y, np.eye(10))
