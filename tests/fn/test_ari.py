"""Tests for ari (Automated Readability Index)."""
from moirais.fn.ari import automated_readability


def test_ari_basic():
    text = "The cat sat on the mat. It was a fine day. The sun was shining brightly."
    r = automated_readability(text)
    assert isinstance(r.value, float)
    assert "characters" in r.extra


def test_cheatsheet():
    from moirais.fn.ari import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
