"""Tests for morie.fn.bnpkl — Binary NP kernel likelihood."""

import numpy as np
import pytest
from morie.fn.bnpkl import bnpkl


def test_returns_dict():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 2))
    prob = 1 / (1 + np.exp(-(X @ np.array([1, 0.5]))))
    y = (rng.uniform(size=n) < prob).astype(float)
    result = bnpkl(y, X)
    assert isinstance(result, dict)
    for key in ("beta", "prob_hat", "log_likelihood", "n_obs"):
        assert key in result


def test_beta_normalized():
    rng = np.random.default_rng(42)
    n = 80
    X = rng.standard_normal((n, 2))
    y = (X[:, 0] > 0).astype(float)
    result = bnpkl(y, X)
    assert abs(np.linalg.norm(result["beta"]) - 1.0) < 1e-4


def test_non_binary_raises():
    with pytest.raises(ValueError, match="binary"):
        bnpkl(np.array([0, 1, 2, 0, 1, 0, 1, 0, 1, 0]), np.ones((10, 2)))
