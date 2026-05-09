"""Tests for moirais.fn.ameig — A-M eigensolve."""
import numpy as np
from moirais.fn.ameig import ameig


def test_ameig_smoke():
    M = np.array([[1, 2], [3, 4], [5, 6]])
    r = ameig(M)
    assert r.name == "am_eigensolve"
    assert r.value > 0
    assert len(r.extra["eigenvalues"]) == 2


def test_cheatsheet():
    from moirais.fn.ameig import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
