"""Tests for moirais.fn.bxcsm — Semiparametric Box-Cox transformation."""

import numpy as np
import pytest
from moirais.fn.bxcsm import bxcsm


def test_returns_dict():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 2))
    y = np.exp(X @ np.array([0.5, 0.3]) + rng.normal(0, 0.2, n))
    result = bxcsm(y, X)
    assert isinstance(result, dict)
    for key in ("lambda_opt", "beta", "se", "t_stat", "pval", "log_likelihood", "n_obs"):
        assert key in result


def test_lambda_near_zero_for_log():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 1))
    y = np.exp(2 * X[:, 0] + rng.normal(0, 0.1, n))
    result = bxcsm(y, X)
    assert abs(result["lambda_opt"]) < 1.0


def test_negative_y_raises():
    with pytest.raises(ValueError, match="positive"):
        bxcsm(np.array([-1, 2, 3, 4, 5]), np.ones((5, 1)))


def test_se_positive():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 1))
    y = np.exp(X[:, 0] + rng.normal(0, 0.1, n))
    result = bxcsm(y, X)
    assert all(s > 0 for s in result["se"])
