"""Tests for moirais.fn.csomp."""
import numpy as np
from moirais.fn.csomp import csomp


def test_csomp_smoke():
    rng = np.random.default_rng(42)
    A = rng.standard_normal((20, 30))
    x_true = np.zeros(30); x_true[:3] = rng.standard_normal(3)
    y = A @ x_true
    result = csomp(A=A, y=y, sparsity=3)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.csomp import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
