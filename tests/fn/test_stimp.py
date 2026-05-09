"""Tests for moirais.fn.stimp — plot stimuli positions."""
from moirais.fn.stimp import stimp


def test_stimp_smoke():
    r = stimp([1.0, 3.0, 5.0], labels=["L", "M", "R"])
    assert r.name == "plot_stimuli_positions"
    assert r.value == 3
    assert r.extra["labels"] == ["L", "M", "R"]


def test_cheatsheet():
    from moirais.fn.stimp import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
