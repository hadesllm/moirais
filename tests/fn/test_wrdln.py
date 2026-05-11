"""Tests for morie.fn.wrdln."""
import numpy as np
from morie.fn.wrdln import wrdln


def test_wrdln_smoke():
    rng = np.random.default_rng(42)
    result = wrdln(text="The quick brown fox jumps over the lazy dog")
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.wrdln import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
