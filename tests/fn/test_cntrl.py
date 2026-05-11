"""Test cntrl."""
import pytest
from morie.fn.cntrl import central_charge


def test_cntrl_bosonic():
    r = central_charge(d=26)
    assert r.value == 26.0
    assert r.extra["is_critical"]


def test_cntrl_super():
    r = central_charge(d=10, supersymmetric=True)
    assert r.value == 15.0
    assert r.extra["is_critical"]


def test_cntrl_invalid():
    with pytest.raises(ValueError):
        central_charge(d=0)
