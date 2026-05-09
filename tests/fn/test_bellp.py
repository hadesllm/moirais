"""Tests for bellp (Bell numbers)."""
from moirais.fn.bellp import bell_polynomial


def test_bell_basic():
    assert bell_polynomial(0).value == 1
    assert bell_polynomial(1).value == 1
    assert bell_polynomial(2).value == 2
    assert bell_polynomial(3).value == 5
    assert bell_polynomial(4).value == 15


def test_cheatsheet():
    from moirais.fn.bellp import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
