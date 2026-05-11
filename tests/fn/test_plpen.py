"""Tests for morie.fn.plpen — Partially linear penalized."""

import numpy as np
import pytest
from morie.fn.plpen import plpen


def test_returns_dict():
    rng = np.random.default_rng(42)
    n = 100
    Z = rng.uniform(0, 1, n)
    X = rng.standard_normal((n, 1))
    y = X[:, 0] + np.sin(Z) + rng.normal(0, 0.1, n)
    result = plpen(y, X, Z)
    assert isinstance(result, dict)
    for key in ("beta", "se", "t_stat", "pval", "gamma", "penalty", "n_obs"):
        assert key in result


def test_penalty_shrinks():
    rng = np.random.default_rng(42)
    n = 100
    Z = rng.uniform(0, 1, n)
    X = rng.standard_normal((n, 1))
    y = X[:, 0] + np.sin(Z) + rng.normal(0, 0.1, n)
    r1 = plpen(y, X, Z, penalty=0.01)
    r2 = plpen(y, X, Z, penalty=100.0)
    g1 = np.linalg.norm(r1["gamma"])
    g2 = np.linalg.norm(r2["gamma"])
    assert g2 <= g1 + 1e-6


def test_negative_penalty_raises():
    with pytest.raises(ValueError, match="penalty"):
        plpen(np.ones(20), np.ones((20, 1)), np.ones(20), penalty=-1)
