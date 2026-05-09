"""Tests for moirais.fn.lpreg — Local polynomial regression."""

import numpy as np
import pytest
from moirais.fn.lpreg import lpreg


def test_returns_dict():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 100)
    y = x**2 + rng.normal(0, 0.1, 100)
    result = lpreg(x, y, degree=2)
    assert isinstance(result, dict)
    for key in ("x_eval", "y_hat", "coefficients", "degree", "bandwidth", "n_obs"):
        assert key in result


def test_degree_zero_is_nw():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 50)
    y = x + rng.normal(0, 0.1, 50)
    result = lpreg(x, y, degree=0)
    assert result["degree"] == 0


def test_negative_degree_raises():
    with pytest.raises(ValueError, match="degree"):
        lpreg(np.ones(10), np.ones(10), degree=-1)


def test_fits_quadratic():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 200)
    y = x**2 + rng.normal(0, 0.05, 200)
    result = lpreg(x, y, degree=2)
    y_hat = np.asarray(result["y_hat"])
    assert np.corrcoef(y, y_hat)[0, 1] > 0.85
