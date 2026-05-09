"""Tests for moirais.fn.pldml — Partially linear DML."""

import numpy as np
import pytest
from moirais.fn.pldml import pldml


def test_returns_dict():
    rng = np.random.default_rng(42)
    n = 200
    Z = rng.uniform(0, 1, n)
    X = rng.standard_normal((n, 1))
    y = 2 * X[:, 0] + np.sin(2 * np.pi * Z) + rng.normal(0, 0.3, n)
    result = pldml(y, X, Z, seed=42)
    assert isinstance(result, dict)
    for key in ("beta", "se", "t_stat", "pval", "n_folds", "n_obs"):
        assert key in result


def test_beta_sign():
    rng = np.random.default_rng(42)
    n = 300
    Z = rng.uniform(0, 1, n)
    X = rng.standard_normal((n, 1))
    y = 3.0 * X[:, 0] + Z**2 + rng.normal(0, 0.2, n)
    result = pldml(y, X, Z, seed=42)
    assert result["beta"][0] > 0


def test_se_positive():
    rng = np.random.default_rng(42)
    n = 200
    Z = rng.uniform(0, 1, n)
    X = rng.standard_normal((n, 1))
    y = X[:, 0] + Z + rng.normal(0, 0.1, n)
    result = pldml(y, X, Z, seed=42)
    assert all(s > 0 for s in result["se"])


def test_too_few_raises():
    with pytest.raises(ValueError, match="at least 20"):
        pldml(np.ones(10), np.ones((10, 1)), np.ones(10))
