"""Tests for moirais.fn.bbr2 — Blackbox R-squared per issue."""
import numpy as np
from moirais.fn.bbr2 import bbr2


def test_bbr2_perfect():
    Z = np.array([[1, 2], [3, 4], [5, 6.0]])
    r = bbr2(Z, Z)
    assert r.name == "bb_r2_per_issue"
    assert abs(r.value - 1.0) < 1e-10


def test_cheatsheet():
    from moirais.fn.bbr2 import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
