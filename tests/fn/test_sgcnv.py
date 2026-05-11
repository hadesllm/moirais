"""Tests for convolution representation."""
import numpy as np
from morie.fn.sgcnv import sgcnv


def test_sgcnv_smoke():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (20, 2))
    wn = rng.normal(0, 1, 30)
    kernel = lambda d: np.exp(-d ** 2)
    r = sgcnv(kernel, wn, coords)
    assert r.name == "convolution_representation"
    assert "field" in r.extra
    assert len(r.extra["field"]) == 20


def test_cheatsheet():
    from morie.fn.sgcnv import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
