"""Tests for ckmra (Cochran-Mantel-Haenszel test)."""
import numpy as np
from morie.fn.ckmra import cochran_mantel


def test_cmh_basic():
    s1 = np.array([[10, 5], [3, 12]])
    s2 = np.array([[8, 7], [4, 11]])
    r = cochran_mantel([s1, s2])
    assert r.extra["chi2"] > 0
    assert 0 <= r.extra["p_value"] <= 1


def test_cheatsheet():
    from morie.fn.ckmra import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
