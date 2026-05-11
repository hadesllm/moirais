"""Tests for morie.fn.rocop -- ROC optimal cutoff."""

import numpy as np
import pytest
from morie.fn.rocop import roc_optimal_cutoff


class TestROCOptimal:
    def test_perfect_separation(self):
        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_score = np.array([0.1, 0.2, 0.3, 0.35, 0.4, 0.6, 0.7, 0.8, 0.85, 0.9])
        res = roc_optimal_cutoff(y_true, y_score)
        assert res["sensitivity"] == pytest.approx(1.0)
        assert res["specificity"] == pytest.approx(1.0)
        assert res["auc"] == pytest.approx(1.0)

    def test_youden_method(self):
        rng = np.random.default_rng(42)
        y_true = np.concatenate([np.zeros(100), np.ones(100)])
        y_score = np.concatenate([rng.normal(0, 1, 100), rng.normal(2, 1, 100)])
        res = roc_optimal_cutoff(y_true, y_score, method="youden")
        assert res["youden_index"] > 0.3
        assert 0 < res["cutoff"] < 4

    def test_closest_method(self):
        rng = np.random.default_rng(42)
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        y_score = np.concatenate([rng.normal(0, 1, 50), rng.normal(2, 1, 50)])
        res = roc_optimal_cutoff(y_true, y_score, method="closest")
        assert res["sensitivity"] > 0.5
        assert res["specificity"] > 0.5

    def test_auc_range(self):
        rng = np.random.default_rng(42)
        y_true = np.concatenate([np.zeros(50), np.ones(50)])
        y_score = np.concatenate([rng.normal(0, 1, 50), rng.normal(1, 1, 50)])
        res = roc_optimal_cutoff(y_true, y_score)
        assert 0.5 <= res["auc"] <= 1.0

    def test_shape_mismatch_raises(self):
        with pytest.raises(ValueError):
            roc_optimal_cutoff(np.array([0, 1]), np.array([0.5]))

    def test_invalid_method_raises(self):
        with pytest.raises(ValueError):
            roc_optimal_cutoff(np.array([0, 1]), np.array([0.3, 0.7]), method="bad")
