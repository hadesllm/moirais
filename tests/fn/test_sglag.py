"""Tests for Lagrange kriging system."""
import numpy as np
from morie.fn.sglag import sglag


def test_sglag_smoke():
    G = np.array([[0, 0.5, 0.8], [0.5, 0, 0.6], [0.8, 0.6, 0]])
    g0 = np.array([0.3, 0.4, 0.7])
    r = sglag(G, g0)
    assert r.name == "lagrange_kriging_system"
    assert "weights" in r.extra
    w = r.extra["weights"]
    assert abs(w.sum() - 1.0) < 1e-6


def test_cheatsheet():
    from morie.fn.sglag import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
