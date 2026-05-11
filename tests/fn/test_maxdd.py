"""Tests for morie.fn.maxdd."""
import numpy as np
from morie.fn.maxdd import maxdd


def test_maxdd_smoke():
    rng = np.random.default_rng(42)
    result = maxdd(prices=rng.uniform(10, 100, size=30))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.maxdd import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
