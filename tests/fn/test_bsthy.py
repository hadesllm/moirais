"""Tests for morie.fn.bsthy — basic space dimensionality test."""
import numpy as np
from morie.fn.bsthy import bsthy


def test_bsthy_smoke():
    rng = np.random.default_rng(42)
    X = rng.standard_normal((50, 4))
    r = bsthy(X, max_dims=3)
    assert r.name == "basic_space_dim_test"
    assert "eigenvalues" in r.extra
    assert len(r.extra["variance_ratios"]) == 3


def test_cheatsheet():
    from morie.fn.bsthy import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
