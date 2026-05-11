"""Tests for universal kriging."""
import numpy as np
from morie.fn.sguk import sguk


def test_sguk_smoke():
    rng = np.random.default_rng(2)
    coords = rng.uniform(0, 10, (20, 2))
    Z = 2 * coords[:, 0] + 3 * coords[:, 1] + rng.normal(0, 0.5, 20)
    r = sguk(Z, coords, np.array([5.0, 5.0]), trend_order=1)
    assert r.name == "universal_kriging"
    assert "predictions" in r.extra
    assert "variances" in r.extra


def test_cheatsheet():
    from morie.fn.sguk import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
