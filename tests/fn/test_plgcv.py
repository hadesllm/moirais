"""Tests for morie.fn.plgcv — PLM GCV bandwidth selection."""

import numpy as np
import pytest
from morie.fn.plgcv import plgcv


def test_returns_dict():
    rng = np.random.default_rng(42)
    n = 100
    Z = rng.uniform(0, 1, n)
    X = rng.standard_normal((n, 1))
    y = X[:, 0] + np.sin(Z) + rng.normal(0, 0.2, n)
    result = plgcv(y, X, Z, n_grid=10)
    assert isinstance(result, dict)
    for key in ("h_opt", "gcv_scores", "h_grid", "n_obs"):
        assert key in result


def test_h_opt_positive():
    rng = np.random.default_rng(42)
    n = 100
    Z = rng.uniform(0, 1, n)
    X = rng.standard_normal((n, 1))
    y = X[:, 0] + Z + rng.normal(0, 0.1, n)
    result = plgcv(y, X, Z, n_grid=5)
    assert result["h_opt"] > 0


def test_too_few_raises():
    with pytest.raises(ValueError, match="at least 10"):
        plgcv(np.ones(5), np.ones((5, 1)), np.ones(5))
