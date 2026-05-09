"""Tests for spectral GRF simulation."""
import numpy as np
from moirais.fn.sgsps import sgsps


def test_sgsps_smoke():
    x = np.linspace(0, 5, 10)
    y = np.linspace(0, 5, 10)
    xx, yy = np.meshgrid(x, y)
    coords = np.column_stack([xx.ravel(), yy.ravel()])
    r = sgsps(coords, n_sims=2)
    assert r.name == "spectral_grf_sim"
    assert r.extra["simulations"].shape[0] == 2


def test_cheatsheet():
    from moirais.fn.sgsps import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
