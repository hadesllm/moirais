"""Tests for Ripley's K function."""
import numpy as np
from morie.fn.sgkfn import sgkfn


def test_sgkfn_smoke():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 10, (30, 2))
    r = sgkfn(pts, (0, 10, 0, 10), r_values=np.linspace(0.5, 3, 5))
    assert r.name == "ripley_k_function"
    assert "K_values" in r.extra
    assert len(r.extra["K_values"]) == 5


def test_cheatsheet():
    from morie.fn.sgkfn import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
