"""Tests for marked point summary."""
import numpy as np
from morie.fn.sgmkd import sgmkd


def test_sgmkd_smoke():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0, 10, (30, 2))
    marks = rng.normal(5, 1, 30)
    r = sgmkd(pts, marks)
    assert r.name == "marked_point_summary"
    assert "mark_correlation" in r.extra
    assert "kmm_ratio" in r.extra


def test_cheatsheet():
    from morie.fn.sgmkd import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
