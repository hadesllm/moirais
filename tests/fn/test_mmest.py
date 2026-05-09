"""Tests for moirais.fn.mmest."""
import numpy as np
from moirais.fn.mmest import mmest


def test_mmest_smoke():
    rng = np.random.default_rng(42)
    n = 30
    X = rng.standard_normal((n, 3))
    y = X @ [1, 2, 3] + rng.normal(0, 1, n)
    result = mmest(X=X, y=y)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.mmest import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
