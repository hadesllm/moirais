"""Tests for morie.fn.sentn."""
import numpy as np
from morie.fn.sentn import sentiment_lexicon


def test_sentn_smoke():
    rng = np.random.default_rng(42)
    result = sentiment_lexicon(text="The quick brown fox jumps over the lazy dog")
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.sentn import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
