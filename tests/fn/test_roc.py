"""Tests for moirais.fn.roc — ROC curve and AUC."""
import numpy as np

from moirais.fn.roc import roc_auc, roc


def test_perfect_auc():
    """Perfect classifier: AUC = 1.0."""
    y_true = [0, 0, 0, 1, 1, 1]
    y_score = [0.1, 0.2, 0.3, 0.7, 0.8, 0.9]
    result = roc_auc(y_true, y_score)
    assert abs(result.estimate - 1.0) < 1e-10


def test_random_auc():
    """Random classifier: AUC near 0.5."""
    rng = np.random.default_rng(42)
    y_true = rng.integers(0, 2, size=500)
    y_score = rng.uniform(size=500)
    result = roc_auc(y_true, y_score)
    assert 0.35 < result.estimate < 0.65


def test_roc_has_curves():
    y_true = [0, 0, 1, 1]
    y_score = [0.1, 0.4, 0.6, 0.9]
    result = roc_auc(y_true, y_score)
    assert "fpr" in result.extra
    assert "tpr" in result.extra


def test_roc_alias():
    assert roc is roc_auc
