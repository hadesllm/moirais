"""Tests for morie.fn.llreg — Local linear regression."""

import numpy as np
import pytest
from morie.fn.llreg import llreg


def test_returns_dict():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 100)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.1, 100)
    result = llreg(x, y)
    assert isinstance(result, dict)
    for key in ("x_eval", "y_hat", "slope", "bandwidth", "n_obs"):
        assert key in result


def test_fits_linear():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 200)
    y = 3 * x + 1 + rng.normal(0, 0.05, 200)
    result = llreg(x, y)
    y_hat = np.asarray(result["y_hat"])
    corr = np.corrcoef(y, y_hat)[0, 1]
    assert corr > 0.9


def test_mismatch_raises():
    with pytest.raises(ValueError, match="must match"):
        llreg(np.ones(10), np.ones(5))


def test_too_few_raises():
    with pytest.raises(ValueError, match="at least 3"):
        llreg(np.ones(2), np.ones(2))
