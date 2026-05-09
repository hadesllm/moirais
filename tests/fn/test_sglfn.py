"""Tests for L function."""
import numpy as np
from moirais.fn.sglfn import sglfn


def test_sglfn_smoke():
    r_vals = np.array([1, 2, 3, 4, 5], dtype=float)
    K_csr = np.pi * r_vals ** 2
    r = sglfn(K_csr, r_vals)
    assert r.name == "l_function"
    assert "L_values" in r.extra
    assert np.allclose(r.extra["L_values"], [0, 0, 0, 0, 0], atol=1e-10)


def test_cheatsheet():
    from moirais.fn.sglfn import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
