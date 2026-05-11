"""Tests for morie.fn.krreg — Kernel ridge regression."""

import numpy as np
import pytest
from morie.fn.krreg import krreg


def test_returns_dict():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 50)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.1, 50)
    result = krreg(x, y)
    assert isinstance(result, dict)
    for key in ("x_eval", "y_hat", "alpha", "bandwidth", "penalty", "n_obs"):
        assert key in result


def test_interpolates_well():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 100)
    y = x**2 + rng.normal(0, 0.05, 100)
    result = krreg(x, y, penalty=0.01)
    y_hat = np.asarray(result["y_hat"])
    assert np.corrcoef(y, y_hat)[0, 1] > 0.9


def test_nonpositive_penalty_raises():
    with pytest.raises(ValueError, match="penalty"):
        krreg(np.ones(10), np.ones(10), penalty=0)


def test_alpha_length():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 30)
    y = x + rng.normal(0, 0.1, 30)
    result = krreg(x, y)
    assert len(result["alpha"]) == 30
