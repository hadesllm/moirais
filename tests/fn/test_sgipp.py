"""Tests for inhomogeneous Poisson process."""
import numpy as np
from morie.fn.sgipp import sgipp


def test_sgipp_smoke():
    fn = lambda x, y: 10.0
    r = sgipp(fn, (0, 10, 0, 10), max_intensity=10.0, seed=42)
    assert r.name == "inhomogeneous_poisson"
    assert "points" in r.extra
    assert r.extra["n_points"] > 0


def test_cheatsheet():
    from morie.fn.sgipp import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
