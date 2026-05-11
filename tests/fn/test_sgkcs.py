"""Tests for kriging conditional simulation."""
import numpy as np
from morie.fn.sgkcs import sgkcs


def test_sgkcs_smoke():
    rng = np.random.default_rng(27)
    n = 10
    coords = rng.uniform(0, 5, (n, 2))
    Z = rng.normal(0, 1, n)
    usims = rng.normal(0, 1, (3, n))
    r = sgkcs(Z, coords, usims)
    assert r.name == "kriging_conditional_sim"
    assert "conditioned_sims" in r.extra
    assert r.extra["conditioned_sims"].shape == (3, n)


def test_cheatsheet():
    from morie.fn.sgkcs import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
