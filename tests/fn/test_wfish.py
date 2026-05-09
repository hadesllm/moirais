"""Tests for Wordfish scaling."""
import numpy as np
from moirais.fn.wfish import wfish


def test_wfish_smoke():
    rng = np.random.default_rng(42)
    dtm = rng.poisson(5, size=(8, 20)).astype(float)
    r = wfish(dtm, max_iter=20)
    assert r.name == "wordfish_scaling"
    assert "positions" in r.extra
    assert len(r.extra["positions"]) == 8


def test_cheatsheet():
    from moirais.fn.wfish import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
