"""Tests for LU decomposition simulation."""
import numpy as np
from morie.fn.sglus import sglus


def test_sglus_smoke():
    C = np.array([[2, 1, 0], [1, 3, 1], [0, 1, 2]], dtype=float)
    r = sglus(C)
    assert r.name == "lu_decomposition_sim"
    assert "L" in r.extra
    assert "U" in r.extra
    assert "P" in r.extra
    recon = r.extra["P"] @ r.extra["L"] @ r.extra["U"]
    assert np.allclose(recon, C)


def test_cheatsheet():
    from morie.fn.sglus import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
