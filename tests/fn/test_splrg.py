"""Tests for moirais.fn.splrg — Spline regression."""

import numpy as np
import pytest
from moirais.fn.splrg import splrg


def test_returns_dict():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 100)
    y = np.sin(2 * np.pi * x) + rng.normal(0, 0.1, 100)
    result = splrg(x, y)
    assert isinstance(result, dict)
    for key in ("x_eval", "y_hat", "coefficients", "knots", "penalty", "n_obs"):
        assert key in result


def test_smoothing_spline():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 100)
    y = x**2 + rng.normal(0, 0.3, 100)
    result = splrg(x, y, penalty=10.0)
    assert result["penalty"] == 10.0


def test_custom_knots():
    rng = np.random.default_rng(42)
    x = rng.uniform(0, 1, 50)
    y = x + rng.normal(0, 0.1, 50)
    knots = np.array([0.25, 0.5, 0.75])
    result = splrg(x, y, knots=knots)
    assert len(result["knots"]) == 3


def test_too_few_raises():
    with pytest.raises(ValueError, match="at least 4"):
        splrg(np.ones(3), np.ones(3))
