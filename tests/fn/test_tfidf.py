"""Tests for moirais.fn.tfidf."""
import numpy as np
from moirais.fn.tfidf import tfidf


def test_tfidf_smoke():
    rng = np.random.default_rng(42)
    result = tfidf(documents=["hello world", "foo bar baz", "hello foo"])
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.tfidf import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
