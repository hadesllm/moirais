"""Tests for pair correlation function."""
import numpy as np
from morie.fn.sgpcf import sgpcf


def test_sgpcf_smoke():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 10, (40, 2))
    r = sgpcf(pts, (0, 10, 0, 10), r_values=np.linspace(0.5, 3, 5))
    assert r.name == "pair_correlation_function"
    assert "g_values" in r.extra
    assert len(r.extra["g_values"]) == 5


def test_cheatsheet():
    from morie.fn.sgpcf import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
