"""Tests for Monte Carlo spatial test."""
import numpy as np
from moirais.fn.sgmci import sgmci


def test_sgmci_smoke():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (30, 2))
    Z = rng.normal(0, 1, 30)
    r = sgmci(Z, coords, n_sim=49)
    assert r.name == "monte_carlo_spatial_test"
    assert "envelope_lo" in r.extra
    assert "envelope_hi" in r.extra


def test_cheatsheet():
    from moirais.fn.sgmci import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
