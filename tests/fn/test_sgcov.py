"""Tests for spatial covariance function."""
import numpy as np
from morie.fn.sgcov import sgcov


def test_sgcov_smoke():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (50, 2))
    Z = rng.normal(0, 1, 50)
    r = sgcov(Z, coords)
    assert r.name == "covariance_function_estimate"
    assert "lag_distances" in r.extra
    assert "covariance_values" in r.extra


def test_sgcov_variance():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (50, 2))
    Z = rng.normal(0, 1, 50)
    r = sgcov(Z, coords)
    assert r.extra["variance_c0"] > 0
