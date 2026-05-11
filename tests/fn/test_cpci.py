"""Tests for morie.fn.cpci."""
from morie.fn.cpci import cpci


def test_cpci_smoke():
    result = cpci(successes=30, trials=100)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.cpci import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
