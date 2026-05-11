"""Tests for morie.fn.dnneg — density negative weights."""
import numpy as np
from morie.fn.dnneg import dnneg


def test_dnneg_smoke():
    rng = np.random.default_rng(42)
    pos = rng.standard_normal(30)
    wts = np.array([1]*10 + [-1]*20, dtype=float)
    r = dnneg(pos, wts)
    assert r.name == "density_negative_weights"
    assert r.value == 20
    assert len(r.extra["density"]) == 100


def test_cheatsheet():
    from morie.fn.dnneg import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
