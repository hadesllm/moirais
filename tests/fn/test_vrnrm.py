"""Tests for morie.fn.vrnrm — normalized variance."""
import numpy as np
from morie.fn.vrnrm import vrnrm


def test_vrnrm_perfect():
    Z = np.array([[1, 2], [3, 4.0]])
    r = vrnrm(Z, Z)
    assert r.name == "normalized_variance"
    assert abs(r.value - 1.0) < 1e-10


def test_cheatsheet():
    from morie.fn.vrnrm import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
