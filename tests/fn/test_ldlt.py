"""Tests for moirais.fn.ldlt."""
import numpy as np
from moirais.fn.ldlt import ldlt_factorize


def test_ldlt_smoke():
    rng = np.random.default_rng(42)
    M = rng.standard_normal((4, 4))
    A = M @ M.T + np.eye(4)
    result = ldlt_factorize(A=A)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from moirais.fn.ldlt import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
