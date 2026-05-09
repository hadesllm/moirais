"""Tests for strict stationarity test."""
import numpy as np
from moirais.fn.sgstr import sgstr


def test_sgstr_smoke():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (100, 2))
    Z = rng.normal(0, 1, 100)
    r = sgstr(Z, coords)
    assert r.name == "strict_stationarity_test"
    assert "mean_ks" in r.extra
    assert "n_subregions" in r.extra


def test_sgstr_stationary():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (200, 2))
    Z = rng.normal(0, 1, 200)
    r = sgstr(Z, coords)
    assert isinstance(r.extra["stationary"], (bool, type(None)))
    assert "n_subregions" in r.extra
