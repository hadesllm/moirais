"""Tests for moirais.fn.maxdd."""
import numpy as np
from moirais.fn.maxdd import maxdd


def test_maxdd_smoke():
    rng = np.random.default_rng(42)
    result = maxdd(prices=rng.uniform(10, 100, size=30))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.maxdd import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
