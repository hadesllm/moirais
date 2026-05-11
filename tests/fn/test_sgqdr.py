"""Tests for quadrat count test."""
import numpy as np
from morie.fn.sgqdr import sgqdr


def test_sgqdr_smoke():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 10, (100, 2))
    r = sgqdr(pts, (0, 10, 0, 10), nx=4, ny=4)
    assert r.name == "quadrat_count_test"
    assert "chi2" in r.extra
    assert "p_value" in r.extra
    assert r.extra["df"] == 15


def test_cheatsheet():
    from morie.fn.sgqdr import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
