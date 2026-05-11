"""Tests for external drift kriging."""
import numpy as np
from morie.fn.sgdrft import sgdrft


def test_sgdrft_smoke():
    rng = np.random.default_rng(7)
    coords = rng.uniform(0, 5, (15, 2))
    elev = coords[:, 0] * 10
    Z = 0.5 * elev + rng.normal(0, 1, 15)
    r = sgdrft(Z, coords, np.array([2.5, 2.5]), elev, drift_target=25.0)
    assert r.name == "external_drift_kriging"
    assert "variance" in r.extra


def test_cheatsheet():
    from morie.fn.sgdrft import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
