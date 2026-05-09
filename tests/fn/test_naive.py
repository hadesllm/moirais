"""Tests for moirais.fn.naive."""
import numpy as np
from moirais.fn.naive import naive


def test_naive_smoke():
    rng = np.random.default_rng(42)
    result = naive(y=rng.uniform(-80, -75, size=20))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.naive import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
