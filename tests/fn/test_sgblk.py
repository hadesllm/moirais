"""Tests for block kriging."""
import numpy as np
from moirais.fn.sgblk import sgblk


def test_sgblk_smoke():
    coords = np.array([[0, 0], [2, 0], [0, 2], [2, 2]], dtype=float)
    Z = np.array([1.0, 2.0, 3.0, 4.0])
    block = np.array([[0.5, 0.5], [1.0, 1.0], [1.5, 1.5]])
    r = sgblk(Z, coords, block)
    assert r.name == "block_kriging"
    assert "variance" in r.extra
    assert r.extra["n_block_points"] == 3


def test_cheatsheet():
    from moirais.fn.sgblk import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
