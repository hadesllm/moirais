"""Tests for moirais.fn.sparma."""
import numpy as np
from moirais.fn.sparma import sparma


def test_sparma_smoke():
    rng = np.random.default_rng(42)
    n = 20
    y = rng.standard_normal(n)
    W = np.ones((n, n)) / (n - 1)
    np.fill_diagonal(W, 0)
    result = sparma(y=y, W=W)
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.sparma import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
