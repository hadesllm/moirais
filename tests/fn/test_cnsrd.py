"""Tests for moirais.fn.cnsrd — Semiparametric censored regression."""

import numpy as np
import pytest
from moirais.fn.cnsrd import cnsrd


def test_returns_dict():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 2))
    y_latent = X @ np.array([1, -0.5]) + rng.normal(0, 0.5, n)
    y = np.maximum(0, y_latent)
    result = cnsrd(y, X)
    assert isinstance(result, dict)
    for key in ("beta", "se", "t_stat", "pval", "n_censored", "n_obs"):
        assert key in result


def test_n_censored_correct():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 2))
    y = np.maximum(0, X[:, 0] + rng.normal(0, 1, n))
    result = cnsrd(y, X)
    assert result["n_censored"] == int((y == 0).sum())


def test_se_positive():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 2))
    y = np.maximum(0, X[:, 0] + 1 + rng.normal(0, 0.5, n))
    result = cnsrd(y, X)
    assert all(s > 0 for s in result["se"])
