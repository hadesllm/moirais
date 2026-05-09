"""Tests for spatial permutation test."""
import numpy as np
from moirais.fn.sgpmt import sgpmt


def test_sgpmt_smoke():
    rng = np.random.default_rng(42)
    coords = rng.uniform(0, 10, (30, 2))
    Z = rng.normal(0, 1, 30)
    r = sgpmt(Z, coords, stat_fn=lambda z, c: float(np.var(z)), n_perm=99)
    assert r.name == "permutation_test_spatial"
    assert "p_value" in r.extra
    assert "observed" in r.extra


def test_cheatsheet():
    from moirais.fn.sgpmt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
