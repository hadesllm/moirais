"""Tests for morie.fn.svm_ -- SVM classifier wrapper."""

import numpy as np
import pytest
from morie.fn.svm_ import svm_classify


class TestSvmClassify:
    def test_linearly_separable(self):
        rng = np.random.default_rng(42)
        X = np.vstack([rng.standard_normal((30, 2)) + [3, 3],
                        rng.standard_normal((30, 2)) - [3, 3]])
        y = np.array([1] * 30 + [0] * 30, dtype=float)
        result = svm_classify(X, y, max_iter=500)
        assert "predictions" in result
        assert "support_vectors" in result
        assert "accuracy" in result
        assert result["accuracy"] >= 0.7

    def test_output_shapes(self):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((20, 2))
        y = (X[:, 0] > 0).astype(float)
        result = svm_classify(X, y, max_iter=100)
        assert len(result["predictions"]) == 20

    def test_mismatched_raises(self):
        with pytest.raises(ValueError):
            svm_classify(np.zeros((5, 2)), np.zeros(3))
