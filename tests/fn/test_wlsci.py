"""Tests for moirais.fn.wlsci."""
from moirais.fn.wlsci import wlsci


def test_wlsci_smoke():
    result = wlsci(successes=30, trials=100)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.wlsci import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
