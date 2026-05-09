"""Tests for WLS variogram fitting."""
import numpy as np
from moirais.fn.sgwls import sgwls


def test_sgwls_smoke():
    lags = np.array([1, 2, 3, 4, 5], dtype=float)
    gamma = np.array([0.3, 0.6, 0.85, 0.95, 1.0], dtype=float)
    r = sgwls(gamma, lags, model="spherical")
    assert r.name == "wls_variogram_fit"
    assert "nugget" in r.extra
    assert "sill" in r.extra
    assert "range" in r.extra
    assert r.extra["rmse"] >= 0


def test_cheatsheet():
    from moirais.fn.sgwls import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
