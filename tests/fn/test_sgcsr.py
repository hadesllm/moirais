"""Tests for CSR test."""
import numpy as np
from morie.fn.sgcsr import sgcsr


def test_sgcsr_smoke():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 10, (50, 2))
    r = sgcsr(pts, (0, 10, 0, 10), n_sim=49)
    assert r.name == "csr_test"
    assert "p_value" in r.extra
    assert "R_index" in r.extra


def test_cheatsheet():
    from morie.fn.sgcsr import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
