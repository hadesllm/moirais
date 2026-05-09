"""Tests for moirais.fn.amwt — A-M weight estimation."""
import numpy as np
from moirais.fn.amwt import amwt


def test_amwt_smoke():
    zhat = np.array([1.0, 3.0, 5.0])
    Z = np.array([[1.5, 3.5, 5.5], [5.0, 3.0, 1.0]])
    r = amwt(Z, zhat)
    assert r.name == "am_weight_estimate"
    assert len(r.extra["betas"]) == 2
    assert r.extra["betas"][1] < 0  # reversed


def test_cheatsheet():
    from moirais.fn.amwt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
