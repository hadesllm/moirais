"""Tests for morie.fn.bgpcl -- Bayesian GP classification."""

import numpy as np
from morie.fn.bgpcl import bayesian_gp_classification


def test_returns_dict():
    rng = np.random.default_rng(42)
    X_tr = rng.standard_normal((30, 1))
    y_tr = (X_tr.ravel() > 0).astype(float)
    X_te = np.array([[0.5], [-0.5]])
    result = bayesian_gp_classification(X_tr, y_tr, X_te)
    assert isinstance(result, dict)
    assert "pred_probs" in result


def test_probs_in_range():
    rng = np.random.default_rng(42)
    X_tr = rng.standard_normal((30, 1))
    y_tr = (X_tr.ravel() > 0).astype(float)
    X_te = rng.standard_normal((5, 1))
    result = bayesian_gp_classification(X_tr, y_tr, X_te)
    assert all(0 <= p <= 1 for p in result["pred_probs"])


def test_positive_input_higher_prob():
    rng = np.random.default_rng(42)
    X_tr = rng.standard_normal((50, 1))
    y_tr = (X_tr.ravel() > 0).astype(float)
    X_te = np.array([[3.0], [-3.0]])
    result = bayesian_gp_classification(X_tr, y_tr, X_te)
    assert result["pred_probs"][0] > result["pred_probs"][1]
