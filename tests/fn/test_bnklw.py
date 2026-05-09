"""Tests for moirais.fn.bnklw — Klein-Spady binary estimator."""

import numpy as np
import pytest
from moirais.fn.bnklw import bnklw


def test_returns_dict():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 2))
    prob = 1 / (1 + np.exp(-(X @ np.array([1, 0.5]))))
    y = (rng.uniform(size=n) < prob).astype(float)
    result = bnklw(y, X)
    assert isinstance(result, dict)
    for key in ("beta", "prob_hat", "log_likelihood", "n_trimmed", "n_obs"):
        assert key in result


def test_trimming_validation():
    with pytest.raises(ValueError, match="trimming"):
        bnklw(np.array([0, 1] * 10), np.ones((20, 2)), trimming=0.6)


def test_non_binary_raises():
    with pytest.raises(ValueError, match="binary"):
        bnklw(np.array([0, 1, 2] * 5), np.ones((15, 2)))
