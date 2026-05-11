"""Tests for morie.fn.cprem."""
import numpy as np
from morie.fn.cprem import cprem


def test_cprem_smoke():
    rng = np.random.default_rng(42)
    result = cprem(x_i=100.0, x_bar=80.0, n=5, k=1.0)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.cprem import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
