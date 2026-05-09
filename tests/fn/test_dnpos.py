"""Tests for moirais.fn.dnpos — density positive weights."""
import numpy as np
from moirais.fn.dnpos import dnpos


def test_dnpos_smoke():
    rng = np.random.default_rng(42)
    pos = rng.standard_normal(30)
    wts = np.array([1]*20 + [-1]*10, dtype=float)
    r = dnpos(pos, wts)
    assert r.name == "density_positive_weights"
    assert r.value == 20
    assert len(r.extra["density"]) == 100


def test_cheatsheet():
    from moirais.fn.dnpos import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
