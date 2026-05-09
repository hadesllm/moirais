"""Tests for moirais.fn.cvxhl."""
import numpy as np
from moirais.fn.cvxhl import cvxhl


def test_cvxhl_smoke():
    rng = np.random.default_rng(42)
    result = cvxhl(points=rng.uniform(size=(20, 2)))
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.cvxhl import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
