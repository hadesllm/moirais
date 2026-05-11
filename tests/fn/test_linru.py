"""Tests for morie.fn.linru — linear utility."""
from morie.fn.linru import linru


def test_linru_smoke():
    r = linru(0.0, 3.0)
    assert r.name == "linear_utility"
    assert r.value == -3.0


def test_linru_zero():
    r = linru(5.0, 5.0)
    assert r.value == 0.0
