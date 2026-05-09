"""Tests for moirais.fn.dirvt — directional voting."""
from moirais.fn.dirvt import dirvt


def test_dirvt_smoke():
    r = dirvt([1.0], [[2.0], [-1.0]])
    assert r.name == "directional_vote"
    assert r.value == 0
    assert "scores" in r.extra


def test_cheatsheet():
    from moirais.fn.dirvt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
