"""Tests for variogram cloud."""
import numpy as np
from morie.fn.sgcld import sgcld


def test_sgcld_smoke():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (20, 2))
    Z = rng.normal(0, 1, 20)
    r = sgcld(Z, coords)
    assert r.name == "variogram_cloud"
    assert "distances" in r.extra
    assert "squared_differences" in r.extra
    assert r.extra["n_pairs"] == 20 * 19 // 2


def test_cheatsheet():
    from morie.fn.sgcld import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
