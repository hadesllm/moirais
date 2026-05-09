"""Tests for moirais.fn.stmrn."""
import numpy as np
from moirais.fn.stmrn import stmrn


def test_stmrn_smoke():
    rng = np.random.default_rng(42)
    n, T = 20, 5
    W = np.ones((n, n)) / (n - 1)
    np.fill_diagonal(W, 0)
    values = rng.standard_normal((n, T))
    result = stmrn(values=values, W=W)
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.stmrn import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
