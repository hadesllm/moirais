"""Tests for morie.fn.aftrg — accelerated failure time regression."""

import numpy as np
import pytest

from morie.fn.aftrg import aftrg


@pytest.fixture()
def aft_data():
    rng = np.random.default_rng(42)
    n = 200
    X = rng.standard_normal((n, 2))
    log_t = 1.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.5 * rng.standard_normal(n)
    time = np.exp(log_t)
    censor = rng.exponential(10.0, size=n)
    event = (time <= censor).astype(float)
    time = np.minimum(time, censor)
    return time, event, X


def test_basic_output(aft_data):
    time, event, X = aft_data
    result = aftrg(time, event, X)
    assert "beta" in result
    assert "sigma" in result
    assert result["sigma"] > 0


def test_intercept_direction(aft_data):
    time, event, X = aft_data
    result = aftrg(time, event, X)
    assert result["beta"][0] > 0


def test_nonpositive_time_raises():
    with pytest.raises(ValueError, match="positive"):
        aftrg(np.array([-1.0, 2.0]), np.array([1.0, 0.0]), np.array([[1.0], [2.0]]))


def test_loglogistic():
    rng = np.random.default_rng(42)
    n = 100
    X = rng.standard_normal((n, 1))
    time = np.exp(1.0 + X[:, 0] + rng.standard_normal(n))
    event = np.ones(n)
    result = aftrg(time, event, X, dist="loglogistic")
    assert result["dist"] == "loglogistic"
