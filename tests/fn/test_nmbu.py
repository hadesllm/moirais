"""Test nmbu."""
from moirais.fn.nmbu import nambu_goto_action


def test_nmbu_basic():
    r = nambu_goto_action()
    assert r.value is not None
    assert r.name == "nambu_goto_action"


def test_nmbu_tension():
    r = nambu_goto_action(T=2.0)
    assert r.extra["tension"] == 2.0
