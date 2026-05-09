"""Tests for moirais.fn.simip — simulate ideal points."""
from moirais.fn.simip import simip


def test_simip_smoke():
    r = simip(n=20, n_dims=2)
    assert r.name == "simulate_ideal_points"
    assert r.value == 20
    assert len(r.extra["points"]) == 20
    assert len(r.extra["points"][0]) == 2


def test_cheatsheet():
    from moirais.fn.simip import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
