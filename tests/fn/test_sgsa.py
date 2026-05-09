"""Tests for simulated annealing spatial."""
import numpy as np
from moirais.fn.sgsa import sgsa


def test_sgsa_smoke():
    rng = np.random.default_rng(26)
    coords = rng.uniform(0, 5, (15, 2))
    Z = rng.normal(0, 1, 15)
    r = sgsa(Z, coords, n_iter=50)
    assert r.name == "simulated_annealing_spatial"
    assert "optimized_field" in r.extra
    assert "energy_history" in r.extra
    assert len(r.extra["energy_history"]) == 51


def test_cheatsheet():
    from moirais.fn.sgsa import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
