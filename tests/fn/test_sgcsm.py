"""Tests for conditional simulation."""
import numpy as np
from morie.fn.sgcsm import sgcsm


def test_sgcsm_smoke():
    rng = np.random.default_rng(23)
    coords = rng.uniform(0, 5, (10, 2))
    Z = rng.normal(0, 1, 10)
    tc = rng.uniform(0, 5, (5, 2))
    r = sgcsm(Z, coords, tc, n_sims=3)
    assert r.name == "conditional_simulation"
    assert r.extra["simulations"].shape == (3, 5)


def test_cheatsheet():
    from morie.fn.sgcsm import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
