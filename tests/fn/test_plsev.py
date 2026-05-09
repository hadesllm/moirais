"""Tests for moirais.fn.plsev — Partially linear sieve estimator."""

import numpy as np
import pytest
from moirais.fn.plsev import plsev


def test_returns_dict():
    rng = np.random.default_rng(42)
    n = 100
    Z = rng.uniform(0, 1, n)
    X = rng.standard_normal((n, 2))
    y = X[:, 0] + np.sin(Z) + rng.normal(0, 0.2, n)
    result = plsev(y, X, Z)
    assert isinstance(result, dict)
    for key in ("beta", "se", "t_stat", "pval", "gamma", "n_basis", "n_obs"):
        assert key in result


def test_cosine_basis():
    rng = np.random.default_rng(42)
    n = 100
    Z = rng.uniform(0, 1, n)
    X = rng.standard_normal((n, 1))
    y = X[:, 0] + np.cos(Z) + rng.normal(0, 0.1, n)
    result = plsev(y, X, Z, basis="cosine")
    assert len(result["gamma"]) > 0


def test_invalid_basis_raises():
    with pytest.raises(ValueError, match="basis"):
        plsev(np.ones(20), np.ones((20, 1)), np.ones(20), basis="wavelet")


def test_too_few_raises():
    with pytest.raises(ValueError, match="at least 10"):
        plsev(np.ones(5), np.ones((5, 1)), np.ones(5))
