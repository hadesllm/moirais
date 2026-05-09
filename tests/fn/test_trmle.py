"""Tests for moirais.fn.trmle — Transformation model MLE."""

import numpy as np
import pytest
from moirais.fn.trmle import trmle


def test_returns_dict():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 2))
    y = X @ np.array([1, -0.5]) + rng.normal(0, 0.5, n)
    result = trmle(y, X, n_basis=3)
    assert isinstance(result, dict)
    for key in ("beta", "se", "t_stat", "pval", "basis_coefs", "log_likelihood", "n_obs"):
        assert key in result


def test_basis_coefs_monotone():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 1))
    y = X[:, 0] + rng.normal(0, 0.2, n)
    result = trmle(y, X, n_basis=4)
    coefs = result["basis_coefs"]
    assert all(coefs[i] <= coefs[i + 1] + 1e-6 for i in range(len(coefs) - 1))


def test_se_finite():
    rng = np.random.default_rng(42)
    n = 80
    X = rng.standard_normal((n, 1))
    y = 2 * X[:, 0] + rng.normal(0, 0.3, n)
    result = trmle(y, X, n_basis=3)
    assert all(np.isfinite(s) for s in result["se"])
