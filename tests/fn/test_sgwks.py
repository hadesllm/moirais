"""Tests for weak stationarity test."""
import numpy as np
from moirais.fn.sgwks import sgwks


def test_sgwks_smoke():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (100, 2))
    Z = rng.normal(0, 1, 100)
    r = sgwks(Z, coords)
    assert r.name == "weak_stationarity_test"
    assert "relative_diff" in r.extra


def test_sgwks_stationary():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (200, 2))
    Z = rng.normal(0, 1, 200)
    r = sgwks(Z, coords)
    assert "stationary" in r.extra
