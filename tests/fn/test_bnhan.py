"""Tests for moirais.fn.bnhan — Han's maximum rank correlation."""

import numpy as np
import pytest
from moirais.fn.bnhan import bnhan


def test_returns_dict():
    rng = np.random.default_rng(42)
    n = 80
    X = rng.standard_normal((n, 2))
    y = (X[:, 0] > 0).astype(float)
    result = bnhan(y, X)
    assert isinstance(result, dict)
    for key in ("beta", "rank_correlation", "n_obs"):
        assert key in result


def test_beta_normalized():
    rng = np.random.default_rng(42)
    n = 60
    X = rng.standard_normal((n, 2))
    y = (X[:, 0] + X[:, 1] > 0).astype(float)
    result = bnhan(y, X)
    assert abs(np.linalg.norm(result["beta"]) - 1.0) < 1e-4


def test_rank_corr_bounded():
    rng = np.random.default_rng(42)
    n = 50
    X = rng.standard_normal((n, 2))
    y = (X[:, 0] > 0).astype(float)
    result = bnhan(y, X)
    assert 0 <= result["rank_correlation"] <= 1
