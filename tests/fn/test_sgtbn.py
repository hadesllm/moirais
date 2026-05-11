"""Tests for turning bands simulation."""
import numpy as np
from morie.fn.sgtbn import sgtbn


def test_sgtbn_smoke():
    rng = np.random.default_rng(25)
    coords = rng.uniform(0, 10, (30, 2))
    r = sgtbn(coords, n_bands=50)
    assert r.name == "turning_bands_sim"
    assert "simulated_values" in r.extra
    assert len(r.extra["simulated_values"]) == 30
    assert r.extra["n_bands"] == 50


def test_cheatsheet():
    from morie.fn.sgtbn import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
