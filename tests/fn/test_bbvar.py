"""Tests for moirais.fn.bbvar — Blackbox variance explained."""
from moirais.fn.bbvar import bbvar


def test_bbvar_smoke():
    r = bbvar([5.0, 3.0, 1.0, 0.5])
    assert r.name == "bb_variance_explained"
    assert abs(sum(r.extra["pct_variance"]) - 100.0) < 1e-6
    assert r.extra["cumulative"][-1] > 99.9


def test_cheatsheet():
    from moirais.fn.bbvar import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
