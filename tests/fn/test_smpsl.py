"""Tests for morie.fn.smpsl — Semiparametric sample selection."""

import numpy as np
import pytest
from morie.fn.smpsl import smpsl


def test_returns_dict():
    rng = np.random.default_rng(42)
    n = 200
    Z = rng.standard_normal((n, 1))
    d = (Z[:, 0] + rng.normal(0, 0.5, n) > 0).astype(float)
    X = rng.standard_normal((n, 2))
    y = X @ np.array([1, -0.5]) + rng.normal(0, 0.3, n)
    result = smpsl(y, X, Z, d)
    assert isinstance(result, dict)
    for key in ("beta", "se", "t_stat", "pval", "lambda_coefs", "n_selected", "n_obs"):
        assert key in result


def test_n_selected_correct():
    rng = np.random.default_rng(42)
    n = 100
    d = np.zeros(n)
    d[:60] = 1
    Z = rng.standard_normal((n, 1))
    X = rng.standard_normal((n, 1))
    y = X[:, 0] + rng.normal(0, 0.1, n)
    result = smpsl(y, X, Z, d)
    assert result["n_selected"] == 60


def test_non_binary_d_raises():
    with pytest.raises(ValueError, match="binary"):
        smpsl(np.ones(10), np.ones((10, 1)), np.ones((10, 1)),
              np.array([0, 1, 2] * 3 + [0]))
