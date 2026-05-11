"""Tests for morie.fn.stgwr."""
import numpy as np
from morie.fn.stgwr import gtwr


def test_stgwr_smoke():
    rng = np.random.default_rng(42)
    n = 30
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = X @ [1, 0.5] + rng.normal(0, 1, n)
    coords = rng.uniform(size=(n, 2))
    times = np.arange(n, dtype=float)
    result = gtwr(X=X, y=y, coords=coords, times=times)
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.stgwr import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
