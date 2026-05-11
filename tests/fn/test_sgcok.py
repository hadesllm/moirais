"""Tests for cokriging."""
import numpy as np
from morie.fn.sgcok import sgcok


def test_sgcok_smoke():
    rng = np.random.default_rng(4)
    coords = rng.uniform(0, 5, (10, 2))
    Z1 = rng.normal(0, 1, 10)
    Z2 = 0.8 * Z1 + rng.normal(0, 0.3, 10)
    r = sgcok(Z1, Z2, coords, np.array([2.5, 2.5]))
    assert r.name == "cokriging"
    assert "variance" in r.extra
    assert "weights_primary" in r.extra


def test_cheatsheet():
    from morie.fn.sgcok import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
