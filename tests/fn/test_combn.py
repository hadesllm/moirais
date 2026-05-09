"""Tests for moirais.fn.combn."""
import numpy as np
from moirais.fn.combn import combn


def test_combn_smoke():
    rng = np.random.default_rng(42)
    result = combn(n=5, k=3)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.combn import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
