"""Tests for morie.fn.bbdim — Blackbox dimensionality."""
import numpy as np
from morie.fn.bbdim import bbdim


def test_bbdim_smoke():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((30, 5))
    r = bbdim(X, max_dims=4)
    assert r.name == "bb_dimensionality_select"
    assert 1 <= r.value <= 4
    assert "eigenvalues" in r.extra


def test_cheatsheet():
    from morie.fn.bbdim import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
