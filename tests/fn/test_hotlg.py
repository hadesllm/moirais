"""Tests for moirais.fn.hotlg — Hotelling model."""
from moirais.fn.hotlg import hotlg


def test_hotlg_smoke():
    r = hotlg(n_voters=50)
    assert r.name == "hotelling_model"
    assert 0 < r.value < 1
    assert len(r.extra["equilibrium_positions"]) == 2


def test_cheatsheet():
    from moirais.fn.hotlg import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
