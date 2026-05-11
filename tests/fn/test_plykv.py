"""Test plykv."""
from morie.fn.plykv import polyakov_action


def test_plykv_basic():
    r = polyakov_action()
    assert r.value is not None
    assert r.name == "polyakov_action"


def test_plykv_tension():
    r = polyakov_action(T=3.0)
    assert r.extra["tension"] == 3.0
