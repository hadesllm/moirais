"""Tests for morie.fn.alien — alienation index."""
from morie.fn.alien import alien


def test_alien_smoke():
    r = alien([0.0], [[5.0], [10.0]])
    assert r.name == "alienation_index"
    assert r.value == 5.0
    assert "mean_alienation" in r.extra


def test_cheatsheet():
    from morie.fn.alien import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
