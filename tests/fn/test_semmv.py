"""Tests for morie.fn.semmv."""
import numpy as np
from morie.fn.semmv import spatial_error_ml


def test_semmv_smoke():
    rng = np.random.default_rng(42)
    n = 20
    X = np.column_stack([np.ones(n), rng.standard_normal(n)])
    y = X @ [1, 0.5] + rng.normal(0, 1, n)
    W = np.ones((n, n)) / (n - 1)
    np.fill_diagonal(W, 0)
    result = spatial_error_ml(X=X, y=y, W=W)
    assert result is not None
    assert hasattr(result, "name")
    assert result.statistic is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.semmv import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
