"""Tests for moirais.fn.lmrkt."""
import numpy as np
from moirais.fn.lmrkt import lmrkt


def test_lmrkt_smoke():
    rng = np.random.default_rng(42)
    n = 20
    values = rng.standard_normal((n, 5))
    W = np.ones((n, n)) / (n - 1)
    np.fill_diagonal(W, 0)
    result = lmrkt(values=values, W=W)
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.lmrkt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
