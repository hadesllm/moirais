"""Tests for moirais.fn.sthet."""
import numpy as np
from moirais.fn.sthet import sthet


def test_sthet_smoke():
    rng = np.random.default_rng(42)
    n, T = 15, 8
    values = rng.standard_normal((n, T))
    W = np.ones((n, n)) / (n - 1)
    np.fill_diagonal(W, 0)
    result = sthet(values=values, W=W)
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.sthet import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
