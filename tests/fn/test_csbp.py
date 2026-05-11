"""Tests for morie.fn.csbp."""
import numpy as np
from morie.fn.csbp import csbp


def test_csbp_smoke():
    rng = np.random.default_rng(42)
    A = rng.standard_normal((20, 30))
    x_true = np.zeros(30); x_true[:3] = rng.standard_normal(3)
    y = A @ x_true
    result = csbp(A=A, y=y)
    assert result is not None
    assert hasattr(result, "name")
    assert result.value is not None or result.extra is not None


def test_cheatsheet():
    from morie.fn.csbp import cheatsheet
    cs = cheatsheet()
    assert isinstance(cs, str)
    assert len(cs) > 0
