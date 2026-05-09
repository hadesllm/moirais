"""Tests for moirais.fn.ammat — A-M matrix setup."""
import numpy as np
from moirais.fn.ammat import ammat


def test_ammat_smoke():
    Z = np.array([[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]])
    r = ammat(Z)
    assert r.name == "am_matrix_setup"
    assert "centered" in r.extra
    assert r.extra["shape"] == [2, 3]


def test_cheatsheet():
    from moirais.fn.ammat import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
