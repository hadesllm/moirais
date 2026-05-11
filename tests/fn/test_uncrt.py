"""Tests for morie.fn.uncrt — utility with uncertainty."""
from morie.fn.uncrt import uncrt


def test_uncrt_smoke():
    r = uncrt(0.0, 1.0, sigma=0.5)
    assert r.name == "utility_uncertainty"
    assert r.value < 0
    assert "sigma" in r.extra


def test_cheatsheet():
    from morie.fn.uncrt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
