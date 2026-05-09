"""Tests for moirais.fn.tpsrt."""
from moirais.fn.tpsrt import tpsrt


def test_tpsrt_smoke():
    result = tpsrt(adj_list={0: [1, 2], 1: [3], 2: [3], 3: []})
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.tpsrt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
