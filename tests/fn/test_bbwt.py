"""Tests for morie.fn.bbwt — Blackbox weight matrix."""
import numpy as np
from morie.fn.bbwt import bbwt


def test_bbwt_smoke():
    rng = np.random.default_rng(42)
    Z = rng.standard_normal((20, 5))
    r = bbwt(Z, n_dims=2)
    assert r.name == "bb_weight_matrix"
    assert r.value == 2
    assert len(r.extra["weights"]) == 5


def test_cheatsheet():
    from morie.fn.bbwt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
