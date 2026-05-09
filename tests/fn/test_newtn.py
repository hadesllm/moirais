"""Tests for moirais.fn.newtn."""
import numpy as np
from moirais.fn.newtn import newtn


def test_newtn_smoke():
    rng = np.random.default_rng(42)
    result = newtn(f=lambda x: x**2 - 2, df=lambda x: 2*x, x0=1.0)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.newtn import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
