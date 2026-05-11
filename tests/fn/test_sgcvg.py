"""Tests for cross-validation variogram."""
import numpy as np
from morie.fn.sgcvg import sgcvg


def test_sgcvg_smoke():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (30, 2))
    Z = rng.normal(0, 1, 30)
    r = sgcvg(Z, coords, model="spherical")
    assert r.name == "cross_validation_variogram"
    assert "rmse" in r.extra
    assert "mae" in r.extra
    assert r.extra["rmse"] >= 0


def test_cheatsheet():
    from morie.fn.sgcvg import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
