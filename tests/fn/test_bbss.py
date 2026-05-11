"""Tests for morie.fn.bbss — Blackbox sum of squares."""
import numpy as np
from morie.fn.bbss import bbss


def test_bbss_smoke():
    Z = np.array([[1, 2], [3, 4.0]])
    r = bbss(Z)
    assert r.name == "bb_sum_squares"
    assert r.value > 0
    assert "grand_mean" in r.extra


def test_cheatsheet():
    from morie.fn.bbss import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
