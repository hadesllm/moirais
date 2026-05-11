"""Tests for morie.fn.stcrs."""
import numpy as np
from morie.fn.stcrs import stcrs


def test_stcrs_smoke():
    rng = np.random.default_rng(42)
    n, T = 15, 8
    field_a = rng.standard_normal((n, T))
    field_b = rng.standard_normal((n, T))
    W = np.ones((n, n)) / (n - 1)
    np.fill_diagonal(W, 0)
    result = stcrs(field_a=field_a, field_b=field_b, W=W)
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.stcrs import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
