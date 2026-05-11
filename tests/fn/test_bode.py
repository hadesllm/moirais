"""Tests for bode (Bode plot computation)."""
from morie.fn.bode import bode_plot


def test_bode_first_order():
    r = bode_plot(num=[1.0], den=[1.0, 1.0], n_points=100)
    assert "magnitude_db" in r.extra
    assert "phase_deg" in r.extra
    assert len(r.extra["magnitude_db"]) == 100


def test_cheatsheet():
    from morie.fn.bode import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
