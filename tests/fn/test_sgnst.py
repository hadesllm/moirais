"""Tests for nested variogram."""
import numpy as np
from morie.fn.sgnst import sgnst


def test_sgnst_smoke():
    h = np.array([0, 1, 2, 3, 5], dtype=float)
    models = [
        {"model": "spherical", "nugget": 0.0, "sill": 0.5, "range": 2.0},
        {"model": "exponential", "nugget": 0.0, "sill": 0.5, "range": 3.0},
    ]
    r = sgnst(h, models)
    assert r.name == "nested_variogram"
    assert r.extra["n_components"] == 2
    assert r.extra["gamma"][0] == 0.0


def test_cheatsheet():
    from morie.fn.sgnst import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
