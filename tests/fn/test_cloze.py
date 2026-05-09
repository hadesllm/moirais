"""Tests for moirais.fn.cloze."""
import numpy as np
from moirais.fn.cloze import cloze


def test_cloze_smoke():
    rng = np.random.default_rng(42)
    result = cloze(text="The quick brown fox jumps over the lazy dog", n_deletions=3)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.cloze import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
