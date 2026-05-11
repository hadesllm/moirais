"""Tests for cross-variogram."""
import numpy as np
from morie.fn.sgcrv import sgcrv


def test_sgcrv_smoke():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (40, 2))
    Z1 = rng.normal(0, 1, 40)
    Z2 = rng.normal(0, 1, 40)
    r = sgcrv(Z1, Z2, coords, n_lags=8)
    assert r.name == "cross_variogram"
    assert "gamma12_values" in r.extra
    assert len(r.extra["gamma12_values"]) == 8


def test_sgcrv_self():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (30, 2))
    Z = rng.normal(0, 1, 30)
    r = sgcrv(Z, Z, coords, n_lags=5)
    for g in r.extra["gamma12_values"]:
        assert g >= 0
