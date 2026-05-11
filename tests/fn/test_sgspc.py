"""Tests for spectral density estimation."""
import numpy as np
from morie.fn.sgspc import sgspc


def test_sgspc_smoke():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (100, 2))
    Z = rng.normal(0, 1, 100)
    r = sgspc(Z, coords, n_freq=20)
    assert r.name == "spectral_density"
    assert "frequencies" in r.extra
    assert "power" in r.extra


def test_cheatsheet():
    from morie.fn.sgspc import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
