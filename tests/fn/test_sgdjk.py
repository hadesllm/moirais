"""Tests for disjunctive kriging."""
import numpy as np
from moirais.fn.sgdjk import sgdjk


def test_sgdjk_smoke():
    rng = np.random.default_rng(3)
    coords = rng.uniform(0, 5, (20, 2))
    Z = rng.normal(5, 1, 20)
    r = sgdjk(Z, coords, np.array([2.5, 2.5]), n_hermite=5)
    assert r.name == "disjunctive_kriging"
    assert "n_hermite" in r.extra


def test_cheatsheet():
    from moirais.fn.sgdjk import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
