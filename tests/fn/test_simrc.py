"""Tests for moirais.fn.simrc — simulate roll call."""
from moirais.fn.simrc import simrc


def test_simrc_smoke():
    r = simrc(n_leg=10, n_votes=20)
    assert r.name == "simulate_roll_call"
    assert r.extra["n_leg"] == 10
    assert len(r.extra["votes"]) == 10
    assert len(r.extra["votes"][0]) == 20


def test_cheatsheet():
    from moirais.fn.simrc import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
