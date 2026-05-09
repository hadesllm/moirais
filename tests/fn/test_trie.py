"""Tests for moirais.fn.trie."""
from moirais.fn.trie import trie_operations


def test_trie_smoke():
    result = trie_operations(words=["hello world", "foo bar baz", "hello foo"])
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.trie import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
