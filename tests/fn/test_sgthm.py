"""Tests for Thomas process."""
from morie.fn.sgthm import sgthm


def test_sgthm_smoke():
    r = sgthm(5.0, 10.0, 0.5, (0, 10, 0, 10), seed=42)
    assert r.name == "thomas_process"
    assert "points" in r.extra
    assert r.extra["n_points"] > 0


def test_cheatsheet():
    from morie.fn.sgthm import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
