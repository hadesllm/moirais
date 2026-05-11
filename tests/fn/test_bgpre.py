"""Tests for morie.fn.bgpre -- Bayesian GP regression."""

import numpy as np
from morie.fn.bgpre import bayesian_gp_regression


def test_returns_dict():
    X_tr = np.linspace(0, 5, 20).reshape(-1, 1)
    y_tr = np.sin(X_tr.ravel())
    X_te = np.array([[1.0], [2.5]])
    result = bayesian_gp_regression(X_tr, y_tr, X_te)
    assert isinstance(result, dict)
    assert "pred_mean" in result


def test_pred_mean_length():
    X_tr = np.linspace(0, 5, 20).reshape(-1, 1)
    y_tr = np.sin(X_tr.ravel())
    X_te = np.linspace(0, 5, 10).reshape(-1, 1)
    result = bayesian_gp_regression(X_tr, y_tr, X_te)
    assert len(result["pred_mean"]) == 10


def test_pred_var_positive():
    X_tr = np.linspace(0, 5, 10).reshape(-1, 1)
    y_tr = np.sin(X_tr.ravel())
    X_te = np.array([[6.0]])
    result = bayesian_gp_regression(X_tr, y_tr, X_te)
    assert result["pred_var"][0] > 0
