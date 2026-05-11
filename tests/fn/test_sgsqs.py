"""Tests for sequential Gaussian simulation."""
import numpy as np
from morie.fn.sgsqs import sgsqs


def test_sgsqs_smoke():
    rng = np.random.default_rng(24)
    coords = rng.uniform(0, 5, (8, 2))
    Z = rng.normal(0, 1, 8)
    grid = rng.uniform(0, 5, (10, 2))
    r = sgsqs(Z, coords, grid)
    assert r.name == "sequential_gaussian_sim"
    assert "simulated_values" in r.extra
    assert len(r.extra["simulated_values"]) == 10


def test_cheatsheet():
    from morie.fn.sgsqs import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
